"""Offline command-line access to retrieval and fairness evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llmqa.benchmark import (
    PROJECT_RETRIEVER_NAMES,
    RETRIEVER_NAMES,
    ProjectRetrieverName,
    RetrieverName,
    fetch_scifact,
    load_scifact,
    run_retrieval_benchmark,
    write_benchmark_artifacts,
)
from llmqa.benchmark_reporting import write_public_benchmark_report
from llmqa.config import require_openai_api_key
from llmqa.domain import Chunk, SearchResult
from llmqa.embeddings import OpenAIEmbeddingProvider
from llmqa.evaluation import RetrievalJudgment, evaluate_rankings
from llmqa.fairness import (
    CounterfactualOutcome,
    audit_counterfactual_outcomes,
    audit_exposure,
    fair_greedy_rerank,
)
from llmqa.project_benchmark import load_project_retrieval_benchmark
from llmqa.project_benchmark_reporting import write_project_benchmark_report
from llmqa.project_evaluation import (
    fetch_project_evaluation_sources,
    load_injection_fixtures,
    load_project_chunk_manifest,
    load_project_eval_manifest,
    load_project_evaluation_cases,
    materialize_project_evaluation,
    validate_project_evaluation,
)
from llmqa.project_generation_cross_judge import (
    DEFAULT_CROSS_JUDGE_MODEL,
    run_project_generation_cross_judge,
    write_project_cross_judge_report,
)
from llmqa.project_generation_evaluation import run_project_generation_evaluation
from llmqa.project_generation_reporting import write_project_generation_report
from llmqa.project_multihop_reporting import write_multihop_retrieval_report
from llmqa.query_decomposition import (
    generate_query_decomposition_artifact,
    load_query_decomposition_artifact,
    query_decomposition_sha256,
)
from llmqa.source_planning import (
    generate_source_plan_artifact,
    load_source_plan_artifact,
    source_plan_sha256,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _load_results(path: Path) -> list[SearchResult]:
    rows = _read_jsonl(path)
    results: list[SearchResult] = []
    for index, row in enumerate(rows, start=1):
        metadata = row.get("metadata", {})
        component_scores = row.get("component_scores", {})
        component_ranks = row.get("component_ranks", {})
        if not isinstance(metadata, dict):
            raise ValueError("result metadata must be a JSON object")
        if not isinstance(component_scores, dict) or not isinstance(component_ranks, dict):
            raise ValueError("result component diagnostics must be JSON objects")
        results.append(
            SearchResult(
                chunk=Chunk(
                    id=str(row["id"]),
                    text=str(row.get("text", "")),
                    source=str(row.get("source", "unknown")),
                    page=row.get("page"),
                    metadata=dict(metadata),
                ),
                score=float(row["score"]),
                rank=int(row.get("rank", index)),
                original_rank=int(row.get("original_rank", row.get("rank", index))),
                component_scores={
                    str(name): float(score) for name, score in component_scores.items()
                },
                component_ranks={str(name): int(rank) for name, rank in component_ranks.items()},
            )
        )
    return results


def _target(raw: str) -> dict[str, float]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--target must be a JSON object")
    return {str(group): float(value) for group, value in parsed.items()}


def _load_judgments(path: Path) -> list[RetrievalJudgment]:
    judgments: list[RetrievalJudgment] = []
    for row in _read_jsonl(path):
        relevance = row.get("relevance")
        if not isinstance(relevance, dict):
            raise ValueError("each judgment must contain a relevance JSON object")
        judgments.append(
            RetrievalJudgment(
                query_id=str(row["query_id"]),
                query=str(row["query"]),
                relevance={str(chunk_id): float(grade) for chunk_id, grade in relevance.items()},
            )
        )
    return judgments


def _load_rankings(path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    for row in _read_jsonl(path):
        query_id = str(row["query_id"])
        retrieved_ids = row.get("retrieved_ids")
        if not isinstance(retrieved_ids, list):
            raise ValueError("each run row must contain a retrieved_ids JSON array")
        if query_id in rankings:
            raise ValueError(f"run contains duplicate query ID {query_id!r}")
        rankings[query_id] = [str(chunk_id) for chunk_id in retrieved_ids]
    return rankings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmqa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    exposure = subparsers.add_parser("audit-exposure", help="compute NDKL for ranked JSONL")
    exposure.add_argument("input", type=Path)
    exposure.add_argument("--target", required=True, help='JSON, for example {"a":0.5,"b":0.5}')
    exposure.add_argument("--group-key", default="fairness_group")

    rerank = subparsers.add_parser("fair-rerank", help="rerank a candidate JSONL slate")
    rerank.add_argument("input", type=Path)
    rerank.add_argument("--target", required=True)
    rerank.add_argument("-k", type=int, required=True)
    rerank.add_argument("--group-key", default="fairness_group")

    counterfactual = subparsers.add_parser(
        "audit-counterfactual", help="compute flip rate and score difference from paired JSONL"
    )
    counterfactual.add_argument("input", type=Path)

    retrieval = subparsers.add_parser(
        "evaluate-retrieval",
        help="compute Recall@k, MRR, and NDCG@k from labelled judgments and a run",
    )
    retrieval.add_argument("judgments", type=Path, help="query and relevance JSONL")
    retrieval.add_argument("run", type=Path, help="ranked chunk-ID JSONL")
    retrieval.add_argument("-k", type=int, required=True)

    fetch = subparsers.add_parser(
        "fetch-scifact",
        help="download and checksum-verify the BEIR SciFact benchmark",
    )
    fetch.add_argument("--cache-dir", type=Path, default=Path("artifacts/benchmarks"))

    benchmark = subparsers.add_parser(
        "benchmark-scifact",
        help="run LLMQA retrievers on an already-fetched SciFact test split",
    )
    benchmark.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("artifacts/benchmarks/scifact"),
    )
    benchmark.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/benchmark-results/scifact"),
    )
    benchmark.add_argument(
        "--retrievers",
        nargs="+",
        choices=RETRIEVER_NAMES,
        default=["bm25"],
    )
    benchmark.add_argument("-k", type=int, default=10)
    benchmark.add_argument("--fetch-k", type=int, default=40)
    benchmark.add_argument("--mmr-lambda", type=float, default=0.75)
    benchmark.add_argument("--bm25-k1", type=float, default=1.2)
    benchmark.add_argument("--bm25-b", type=float, default=0.75)
    benchmark.add_argument("--rrf-rank-constant", type=int, default=60)
    benchmark.add_argument("--dense-weight", type=float, default=1.0)
    benchmark.add_argument("--sparse-weight", type=float, default=1.0)
    benchmark.add_argument("--limit-queries", type=int)
    benchmark.add_argument("--embedding-model", default="text-embedding-3-small")
    benchmark.add_argument("--embedding-dimensions", type=int, default=1536)
    benchmark.add_argument("--embedding-batch-size", type=int, default=128)

    project_benchmark = subparsers.add_parser(
        "benchmark-project-eval",
        help="run LLMQA retrievers on verified project-evaluation chunks and qrels",
    )
    project_benchmark.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("evals/project/technical-papers-v1"),
    )
    project_benchmark.add_argument(
        "--chunks",
        type=Path,
        default=Path("artifacts/evals/technical-papers-v1/chunks.jsonl"),
    )
    project_benchmark.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/benchmark-results/technical-papers-v1"),
    )
    project_benchmark.add_argument(
        "--retrievers", nargs="+", choices=PROJECT_RETRIEVER_NAMES, default=["bm25"]
    )
    project_benchmark.add_argument(
        "--query-decompositions",
        type=Path,
        default=Path("evals/project/technical-papers-v1/query-decompositions.json"),
    )
    project_benchmark.add_argument(
        "--source-plans",
        type=Path,
        default=Path("evals/project/technical-papers-v1/source-plans.json"),
    )
    project_benchmark.add_argument("-k", type=int, default=10)
    project_benchmark.add_argument("--fetch-k", type=int, default=40)
    project_benchmark.add_argument("--mmr-lambda", type=float, default=0.75)
    project_benchmark.add_argument("--bm25-k1", type=float, default=1.2)
    project_benchmark.add_argument("--bm25-b", type=float, default=0.75)
    project_benchmark.add_argument("--rrf-rank-constant", type=int, default=60)
    project_benchmark.add_argument("--dense-weight", type=float, default=1.0)
    project_benchmark.add_argument("--sparse-weight", type=float, default=1.0)
    project_benchmark.add_argument("--embedding-model", default="text-embedding-3-small")
    project_benchmark.add_argument("--embedding-dimensions", type=int, default=1536)
    project_benchmark.add_argument("--embedding-batch-size", type=int, default=128)

    decompose_queries = subparsers.add_parser(
        "generate-project-query-decompositions",
        help="generate question-only OpenAI subqueries for reviewed multi-hop cases",
    )
    decompose_queries.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("evals/project/technical-papers-v1"),
    )
    decompose_queries.add_argument(
        "--output",
        type=Path,
        default=Path("evals/project/technical-papers-v1/query-decompositions.json"),
    )
    decompose_queries.add_argument("--model", default="gpt-5-mini")

    source_plans = subparsers.add_parser(
        "generate-project-source-plans",
        help="generate question-and-source-catalog OpenAI plans for reviewed multi-hop cases",
    )
    source_plans.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("evals/project/technical-papers-v1"),
    )
    source_plans.add_argument(
        "--output",
        type=Path,
        default=Path("evals/project/technical-papers-v1/source-plans.json"),
    )
    source_plans.add_argument("--model", default="gpt-5-mini")

    multihop_report = subparsers.add_parser(
        "report-project-multihop-retrieval",
        help="publish paired locator-coverage evidence for a planned multi-hop retriever",
    )
    multihop_report.add_argument("summary", type=Path)
    multihop_report.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("evals/project/technical-papers-v1"),
    )
    multihop_report.add_argument(
        "--query-decompositions",
        type=Path,
        default=Path("evals/project/technical-papers-v1/query-decompositions.json"),
    )
    multihop_report.add_argument(
        "--source-plans",
        type=Path,
        default=Path("evals/project/technical-papers-v1/source-plans.json"),
    )
    multihop_report.add_argument(
        "--candidate-retriever",
        choices=("bm25-decomposed-rrf", "bm25-source-aware"),
        default="bm25-decomposed-rrf",
    )
    multihop_report.add_argument("--snapshot", type=Path, required=True)
    multihop_report.add_argument("--figure", type=Path, required=True)
    multihop_report.add_argument("--run-date", required=True)
    multihop_report.add_argument("--baseline-generation-summary", type=Path)
    multihop_report.add_argument("--baseline-generation-results", type=Path)
    multihop_report.add_argument("--candidate-generation-summary", type=Path)
    multihop_report.add_argument("--candidate-generation-results", type=Path)

    report = subparsers.add_parser(
        "report-scifact",
        help="generate compact public JSON and SVG evidence from a full SciFact run",
    )
    report.add_argument("summary", type=Path, help="full benchmark summary JSON")
    report.add_argument("--snapshot", type=Path, required=True)
    report.add_argument("--figure", type=Path, required=True)
    report.add_argument("--run-date", required=True, help="benchmark date in YYYY-MM-DD format")
    report.add_argument("--bootstrap-resamples", type=int, default=10_000)
    report.add_argument("--bootstrap-seed", type=int, default=20_260_828)

    project_report = subparsers.add_parser(
        "report-project-eval",
        help="generate compact JSON and SVG evidence from a full project retrieval run",
    )
    project_report.add_argument("summary", type=Path)
    project_report.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("evals/project/technical-papers-v1"),
    )
    project_report.add_argument("--snapshot", type=Path, required=True)
    project_report.add_argument("--figure", type=Path, required=True)
    project_report.add_argument("--run-date", required=True)
    project_report.add_argument("--bootstrap-resamples", type=int, default=10_000)
    project_report.add_argument("--bootstrap-seed", type=int, default=20_260_829)

    generation_eval = subparsers.add_parser(
        "evaluate-project-generation",
        help="run clean and attacked generation evaluation on reviewed project cases",
    )
    generation_eval.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("evals/project/technical-papers-v1"),
    )
    generation_eval.add_argument(
        "--chunks",
        type=Path,
        default=Path("artifacts/evals/technical-papers-v1/chunks.jsonl"),
    )
    generation_eval.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/generation-results/technical-papers-v1"),
    )
    generation_eval.add_argument("--candidate-model", default="gpt-5-mini")
    generation_eval.add_argument("--judge-model", default="gpt-5-mini")
    generation_eval.add_argument("-k", type=int, default=10)
    generation_eval.add_argument("--bm25-k1", type=float, default=1.2)
    generation_eval.add_argument("--bm25-b", type=float, default=0.75)
    generation_eval.add_argument(
        "--retriever",
        choices=("bm25", "bm25-decomposed-rrf", "bm25-source-aware"),
        default="bm25",
    )
    generation_eval.add_argument(
        "--query-decompositions",
        type=Path,
        default=Path("evals/project/technical-papers-v1/query-decompositions.json"),
    )
    generation_eval.add_argument(
        "--source-plans",
        type=Path,
        default=Path("evals/project/technical-papers-v1/source-plans.json"),
    )
    generation_eval.add_argument("--fetch-k", type=int, default=40)
    generation_eval.add_argument("--rrf-rank-constant", type=int, default=60)
    generation_eval.add_argument("--workers", type=int, default=1)
    generation_eval.add_argument(
        "--case-ids",
        nargs="+",
        help="optional reviewed case IDs for a non-publishable smoke run",
    )

    generation_report = subparsers.add_parser(
        "report-project-generation",
        help="publish a provisional automated generation-evaluation JSON and SVG",
    )
    generation_report.add_argument("summary", type=Path)
    generation_report.add_argument("case_results", type=Path)
    generation_report.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("evals/project/technical-papers-v1"),
    )
    generation_report.add_argument("--snapshot", type=Path, required=True)
    generation_report.add_argument("--figure", type=Path, required=True)
    generation_report.add_argument("--run-date", required=True)
    generation_report.add_argument("--bootstrap-resamples", type=int, default=10_000)
    generation_report.add_argument("--bootstrap-seed", type=int, default=20_260_829)

    cross_judge = subparsers.add_parser(
        "cross-judge-project-generation",
        help="re-judge a stratified generation subset with a second model family",
    )
    cross_judge.add_argument("primary_summary", type=Path)
    cross_judge.add_argument("primary_results", type=Path)
    cross_judge.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("evals/project/technical-papers-v1"),
    )
    cross_judge.add_argument(
        "--chunks",
        type=Path,
        default=Path("artifacts/evals/technical-papers-v1/chunks.jsonl"),
    )
    cross_judge.add_argument("--output-dir", type=Path, required=True)
    cross_judge.add_argument("--judge-model", default=DEFAULT_CROSS_JUDGE_MODEL)
    cross_judge.add_argument("--sample-size", type=int, default=30)
    cross_judge.add_argument("--seed", type=int, default=20_260_829)
    cross_judge.add_argument("--workers", type=int, default=4)

    cross_judge_report = subparsers.add_parser(
        "report-project-cross-judge",
        help="publish compact cross-judge agreement evidence",
    )
    cross_judge_report.add_argument("summary", type=Path)
    cross_judge_report.add_argument("results", type=Path)
    cross_judge_report.add_argument("--snapshot", type=Path, required=True)
    cross_judge_report.add_argument("--run-date", required=True)

    validate_project_eval = subparsers.add_parser(
        "validate-project-eval",
        help="validate coverage, provenance, evidence, and review state for project QA cases",
    )
    validate_project_eval.add_argument("cases", type=Path)
    validate_project_eval.add_argument("--fixtures", type=Path, required=True)
    validate_project_eval.add_argument("--manifest", type=Path, required=True)
    validate_project_eval.add_argument("--chunk-manifest", type=Path)

    materialize_project_eval = subparsers.add_parser(
        "materialize-project-eval",
        help="build deterministic project chunks, evidence IDs, and retrieval judgments",
    )
    materialize_project_eval.add_argument("cases", type=Path)
    materialize_project_eval.add_argument("--fixtures", type=Path, required=True)
    materialize_project_eval.add_argument("--manifest", type=Path, required=True)
    materialize_project_eval.add_argument("--source-dir", type=Path, required=True)
    materialize_project_eval.add_argument("--output-cases", type=Path, required=True)
    materialize_project_eval.add_argument("--judgments", type=Path, required=True)
    materialize_project_eval.add_argument("--chunk-manifest", type=Path, required=True)
    materialize_project_eval.add_argument("--chunks", type=Path, required=True)

    fetch_project_eval = subparsers.add_parser(
        "fetch-project-eval-sources",
        help="download and SHA-256 verify project-evaluation source documents",
    )
    fetch_project_eval.add_argument("manifest", type=Path)
    fetch_project_eval.add_argument("--cache-dir", type=Path, default=Path("artifacts/evals"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "audit-exposure":
        exposure_audit = audit_exposure(
            _load_results(args.input), _target(args.target), group_key=args.group_key
        )
        print(json.dumps(asdict(exposure_audit), indent=2))
        return 0

    if args.command == "fair-rerank":
        reranked = fair_greedy_rerank(
            _load_results(args.input),
            _target(args.target),
            k=args.k,
            group_key=args.group_key,
        )
        print(
            json.dumps(
                [
                    {
                        "id": result.chunk.id,
                        "rank": result.rank,
                        "original_rank": result.original_rank,
                        "score": result.score,
                        "metadata": result.chunk.metadata,
                        "component_scores": result.component_scores,
                        "component_ranks": result.component_ranks,
                    }
                    for result in reranked
                ],
                indent=2,
            )
        )
        return 0

    if args.command == "evaluate-retrieval":
        evaluation = evaluate_rankings(
            _load_judgments(args.judgments),
            _load_rankings(args.run),
            k=args.k,
        )
        print(json.dumps(asdict(evaluation), indent=2))
        return 0

    if args.command == "fetch-scifact":
        dataset_directory = fetch_scifact(args.cache_dir)
        print(
            json.dumps(
                {
                    "dataset": "beir/scifact",
                    "dataset_directory": str(dataset_directory),
                    "status": "verified",
                },
                indent=2,
            )
        )
        return 0

    if args.command == "benchmark-scifact":
        retriever_names: list[RetrieverName] = args.retrievers
        needs_dense = any(name in {"dense", "dense-mmr", "hybrid"} for name in retriever_names)
        if needs_dense:
            require_openai_api_key()
        provider = (
            OpenAIEmbeddingProvider(
                model=args.embedding_model,
                dimensions=args.embedding_dimensions,
                batch_size=args.embedding_batch_size,
            )
            if needs_dense
            else None
        )
        dataset = load_scifact(args.dataset_dir, limit_queries=args.limit_queries)
        outcome = run_retrieval_benchmark(
            dataset,
            retriever_names,
            k=args.k,
            fetch_k=args.fetch_k,
            mmr_lambda=args.mmr_lambda,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
            rrf_rank_constant=args.rrf_rank_constant,
            dense_weight=args.dense_weight,
            sparse_weight=args.sparse_weight,
            embedding_provider=provider,
        )
        summary_path = write_benchmark_artifacts(outcome, args.output_dir)
        print(
            json.dumps(
                {
                    "summary_path": str(summary_path),
                    "dataset": outcome.report.dataset,
                    "corpus_count": outcome.report.corpus_count,
                    "query_count": outcome.report.query_count,
                    "total_query_count": outcome.report.total_query_count,
                    "limited_run": outcome.report.limited_run,
                    "runs": [
                        {
                            "retriever": run.retriever,
                            "mean_recall_at_k": run.evaluation.mean_recall_at_k,
                            "mean_reciprocal_rank": run.evaluation.mean_reciprocal_rank,
                            "mean_ndcg_at_k": run.evaluation.mean_ndcg_at_k,
                            "retrieval_latency_ms_p50": run.retrieval_latency_ms_p50,
                            "retrieval_latency_ms_p95": run.retrieval_latency_ms_p95,
                        }
                        for run in outcome.report.runs
                    ],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "benchmark-project-eval":
        project_retriever_names: list[ProjectRetrieverName] = args.retrievers
        needs_dense = any(
            name in {"dense", "dense-mmr", "hybrid"} for name in project_retriever_names
        )
        if needs_dense:
            require_openai_api_key()
        provider = (
            OpenAIEmbeddingProvider(
                model=args.embedding_model,
                dimensions=args.embedding_dimensions,
                batch_size=args.embedding_batch_size,
            )
            if needs_dense
            else None
        )
        dataset = load_project_retrieval_benchmark(args.eval_dir, args.chunks)
        decomposition_mapping = None
        decomposition_provenance = None
        if "bm25-decomposed-rrf" in project_retriever_names:
            artifact, decomposition_mapping = load_query_decomposition_artifact(
                args.query_decompositions,
                args.eval_dir,
            )
            decomposition_provenance = {
                "artifact_sha256": query_decomposition_sha256(args.query_decompositions),
                "method": artifact.method,
                "prompt_version": artifact.prompt_version,
                "prompt_sha256": artifact.prompt_sha256,
                "cases_sha256": artifact.cases_sha256,
                "requested_model": artifact.requested_model,
                "generated_at": artifact.generated_at,
                "question_only_input": artifact.question_only_input,
                "query_count": artifact.query_count,
            }
        source_plan_mapping = None
        source_plan_provenance = None
        if "bm25-source-aware" in project_retriever_names:
            source_artifact, source_plan_mapping = load_source_plan_artifact(
                args.source_plans,
                args.eval_dir,
            )
            source_plan_provenance = {
                "artifact_sha256": source_plan_sha256(args.source_plans),
                "method": source_artifact.method,
                "prompt_version": source_artifact.prompt_version,
                "prompt_sha256": source_artifact.prompt_sha256,
                "cases_sha256": source_artifact.cases_sha256,
                "source_catalog_sha256": source_artifact.source_catalog_sha256,
                "requested_model": source_artifact.requested_model,
                "generated_at": source_artifact.generated_at,
                "question_and_source_catalog_only": (
                    source_artifact.question_and_source_catalog_only
                ),
                "plan_count": source_artifact.plan_count,
            }
        outcome = run_retrieval_benchmark(
            dataset,
            project_retriever_names,
            k=args.k,
            fetch_k=args.fetch_k,
            mmr_lambda=args.mmr_lambda,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
            rrf_rank_constant=args.rrf_rank_constant,
            dense_weight=args.dense_weight,
            sparse_weight=args.sparse_weight,
            embedding_provider=provider,
            query_decompositions=decomposition_mapping,
            query_decomposition_provenance=decomposition_provenance,
            source_plans=source_plan_mapping,
            source_plan_provenance=source_plan_provenance,
        )
        summary_path = write_benchmark_artifacts(outcome, args.output_dir)
        print(
            json.dumps(
                {
                    "summary_path": str(summary_path),
                    "dataset": outcome.report.dataset,
                    "corpus_count": outcome.report.corpus_count,
                    "query_count": outcome.report.query_count,
                    "answerable_query_count": outcome.report.runs[
                        0
                    ].evaluation.answerable_query_count,
                    "unique_relevance_group_count": outcome.report.unique_relevance_group_count,
                    "runs": [
                        {
                            "retriever": run.retriever,
                            "mean_recall_at_k": run.evaluation.mean_recall_at_k,
                            "mean_reciprocal_rank": run.evaluation.mean_reciprocal_rank,
                            "mean_ndcg_at_k": run.evaluation.mean_ndcg_at_k,
                            "retrieval_latency_ms_p50": run.retrieval_latency_ms_p50,
                            "retrieval_latency_ms_p95": run.retrieval_latency_ms_p95,
                        }
                        for run in outcome.report.runs
                    ],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "generate-project-query-decompositions":
        require_openai_api_key()
        artifact = generate_query_decomposition_artifact(
            args.eval_dir,
            args.output,
            model=args.model,
        )
        print(
            json.dumps(
                {
                    "output_path": str(args.output),
                    "query_count": artifact.query_count,
                    "method": artifact.method,
                    "prompt_sha256": artifact.prompt_sha256,
                    "requested_model": artifact.requested_model,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "generate-project-source-plans":
        require_openai_api_key()
        source_plan_artifact = generate_source_plan_artifact(
            args.eval_dir,
            args.output,
            model=args.model,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "method": source_plan_artifact.method,
                    "requested_model": source_plan_artifact.requested_model,
                    "plan_count": source_plan_artifact.plan_count,
                    "question_and_source_catalog_only": (
                        source_plan_artifact.question_and_source_catalog_only
                    ),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "report-project-multihop-retrieval":
        planning_path = (
            args.source_plans
            if args.candidate_retriever == "bm25-source-aware"
            else args.query_decompositions
        )
        snapshot = write_multihop_retrieval_report(
            args.summary,
            args.eval_dir,
            planning_path,
            args.snapshot,
            args.figure,
            run_date=args.run_date,
            baseline_generation_summary_path=args.baseline_generation_summary,
            baseline_generation_results_path=args.baseline_generation_results,
            candidate_generation_summary_path=args.candidate_generation_summary,
            candidate_generation_results_path=args.candidate_generation_results,
            candidate_retriever=args.candidate_retriever,
        )
        print(
            json.dumps(
                {
                    "snapshot_path": str(args.snapshot),
                    "figure_path": str(args.figure),
                    "selection": snapshot["selection"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "report-scifact":
        write_public_benchmark_report(
            args.summary,
            args.snapshot,
            args.figure,
            run_date=args.run_date,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(
            json.dumps(
                {
                    "snapshot_path": str(args.snapshot),
                    "figure_path": str(args.figure),
                    "status": "generated",
                },
                indent=2,
            )
        )
        return 0

    if args.command == "report-project-eval":
        write_project_benchmark_report(
            args.summary,
            args.snapshot,
            args.figure,
            run_date=args.run_date,
            evaluation_directory=args.eval_dir,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(
            json.dumps(
                {
                    "snapshot_path": str(args.snapshot),
                    "figure_path": str(args.figure),
                    "status": "generated",
                },
                indent=2,
            )
        )
        return 0

    if args.command == "evaluate-project-generation":
        require_openai_api_key()
        summary_path = run_project_generation_evaluation(
            args.eval_dir,
            args.chunks,
            args.output_dir,
            candidate_model=args.candidate_model,
            judge_model=args.judge_model,
            k=args.k,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
            retriever_name=args.retriever,
            query_decomposition_path=(
                args.query_decompositions if args.retriever == "bm25-decomposed-rrf" else None
            ),
            source_plan_path=(args.source_plans if args.retriever == "bm25-source-aware" else None),
            fetch_k=args.fetch_k,
            rrf_rank_constant=args.rrf_rank_constant,
            case_ids=args.case_ids,
            max_workers=args.workers,
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "summary_path": str(summary_path),
                    "run_id": summary["run_id"],
                    "limited_run": summary["limited_run"],
                    "counts": summary["counts"],
                    "clean_metrics": summary["clean_metrics"],
                    "injection_metrics": summary["injection_metrics"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "report-project-generation":
        write_project_generation_report(
            args.summary,
            args.case_results,
            args.eval_dir,
            args.snapshot,
            args.figure,
            run_date=args.run_date,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(
            json.dumps(
                {
                    "snapshot_path": str(args.snapshot),
                    "figure_path": str(args.figure),
                    "status": "automated_baseline_human_adjudication_pending",
                },
                indent=2,
            )
        )
        return 0

    if args.command == "cross-judge-project-generation":
        require_openai_api_key()
        summary_path = run_project_generation_cross_judge(
            args.primary_summary,
            args.primary_results,
            args.eval_dir,
            args.chunks,
            args.output_dir,
            judge_model=args.judge_model,
            sample_size=args.sample_size,
            seed=args.seed,
            max_workers=args.workers,
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "summary_path": str(summary_path),
                    "run_id": summary["run_id"],
                    "counts": summary["counts"],
                    "agreement": summary["agreement"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "report-project-cross-judge":
        write_project_cross_judge_report(
            args.summary,
            args.results,
            args.snapshot,
            run_date=args.run_date,
        )
        print(json.dumps({"snapshot_path": str(args.snapshot), "status": "generated"}, indent=2))
        return 0

    if args.command == "validate-project-eval":
        manifest = load_project_eval_manifest(args.manifest)
        cases = load_project_evaluation_cases(args.cases)
        fixtures = load_injection_fixtures(args.fixtures)
        chunk_manifest = (
            load_project_chunk_manifest(args.chunk_manifest) if args.chunk_manifest else None
        )
        evaluation_summary = validate_project_evaluation(cases, fixtures, manifest, chunk_manifest)
        print(json.dumps(asdict(evaluation_summary), indent=2))
        return 0

    if args.command == "materialize-project-eval":
        materialization_summary = materialize_project_evaluation(
            args.cases,
            args.fixtures,
            args.manifest,
            args.source_dir,
            args.output_cases,
            args.judgments,
            args.chunk_manifest,
            args.chunks,
        )
        print(json.dumps(asdict(materialization_summary), indent=2))
        return 0

    if args.command == "fetch-project-eval-sources":
        source_paths = fetch_project_evaluation_sources(args.manifest, args.cache_dir)
        print(
            json.dumps(
                {
                    "status": "verified",
                    "source_paths": [str(path) for path in source_paths],
                },
                indent=2,
            )
        )
        return 0

    rows = _read_jsonl(args.input)
    counterfactual_audit = audit_counterfactual_outcomes(
        [
            CounterfactualOutcome(
                case_id=str(row["case_id"]),
                label_a=str(row["label_a"]),
                label_b=str(row["label_b"]),
                score_a=float(row["score_a"]) if row.get("score_a") is not None else None,
                score_b=float(row["score_b"]) if row.get("score_b") is not None else None,
            )
            for row in rows
        ]
    )
    print(json.dumps(asdict(counterfactual_audit), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

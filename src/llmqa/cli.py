"""Offline command-line access to retrieval and fairness evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llmqa.domain import Chunk, SearchResult
from llmqa.evaluation import RetrievalJudgment, evaluate_rankings
from llmqa.fairness import (
    CounterfactualOutcome,
    audit_counterfactual_outcomes,
    audit_exposure,
    fair_greedy_rerank,
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

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from llmqa.benchmark import (
    SCIFACT_MANIFEST,
    fetch_scifact,
    load_scifact,
    run_retrieval_benchmark,
    write_benchmark_artifacts,
)
from llmqa.benchmark_reporting import write_public_benchmark_report
from llmqa.cli import main
from llmqa.domain import SourceScopedQuery
from llmqa.project_benchmark_reporting import (
    build_project_benchmark_snapshot,
    write_project_benchmark_report,
)
from llmqa.project_evaluation import load_project_evaluation_cases
from llmqa.project_multihop_reporting import (
    add_generation_experiment,
    build_multihop_retrieval_snapshot,
    render_multihop_retrieval_svg,
)


def _fixture_members() -> dict[str, str]:
    corpus = [
        {"_id": "a", "title": "Alpha", "text": "apple evidence", "metadata": {}},
        {"_id": "b", "title": "Beta", "text": "banana evidence", "metadata": {}},
        {"_id": "c", "title": "Gamma", "text": "unrelated material", "metadata": {}},
    ]
    queries = [
        {"_id": "q1", "text": "alpha apple", "metadata": {}},
        {"_id": "q2", "text": "beta banana", "metadata": {}},
        {"_id": "unused", "text": "not in test qrels", "metadata": {}},
    ]
    return {
        "scifact/corpus.jsonl": "\n".join(json.dumps(row) for row in corpus) + "\n",
        "scifact/queries.jsonl": "\n".join(json.dumps(row) for row in queries) + "\n",
        "scifact/qrels/test.tsv": "query-id\tcorpus-id\tscore\nq1\ta\t1\nq2\tb\t2\n",
    }


def _write_archive(path: Path, extra_members: dict[str, str] | None = None) -> str:
    members = {**_fixture_members(), **(extra_members or {})}
    with zipfile.ZipFile(path, "w") as archive:
        for member, content in members.items():
            archive.writestr(member, content)
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def _fetch_fixture(tmp_path: Path) -> Path:
    archive_path = tmp_path / "fixture.zip"
    checksum = _write_archive(archive_path)
    return fetch_scifact(
        tmp_path / "cache",
        source_url=archive_path.as_uri(),
        expected_md5=checksum,
    )


class CountingEmbeddingProvider:
    model = "fixture-embedding"
    dimensions = 3
    batch_size = 16

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_batch_calls = 0
        self.query_calls = 0
        self.vectors = {
            "Alpha\n\napple evidence": [1.0, 0.0, 0.0],
            "Beta\n\nbanana evidence": [0.0, 1.0, 0.0],
            "Gamma\n\nunrelated material": [0.0, 0.0, 1.0],
            "alpha apple": [1.0, 0.0, 0.0],
            "beta banana": [0.0, 1.0, 0.0],
        }

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.document_calls += 1
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        self.query_calls += 1
        return np.asarray([self.vectors[text]], dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.query_batch_calls += 1
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def test_fetch_and_load_scifact_fixture_with_verified_manifest(tmp_path: Path) -> None:
    dataset_directory = _fetch_fixture(tmp_path)
    dataset = load_scifact(dataset_directory, limit_queries=1)

    assert (dataset_directory / SCIFACT_MANIFEST).is_file()
    assert [chunk.id for chunk in dataset.chunks] == ["a", "b", "c"]
    assert dataset.chunks[0].text == "Alpha\n\napple evidence"
    assert dataset.total_query_count == 2
    assert dataset.limited_run is True
    assert dataset.judgments[0].relevance == {"a": 1.0}

    manifest = json.loads((dataset_directory / SCIFACT_MANIFEST).read_text(encoding="utf-8"))
    assert (
        fetch_scifact(
            tmp_path / "cache",
            source_url=manifest["source_url"],
            expected_md5=manifest["archive_md5"],
        )
        == dataset_directory
    )


def test_fetch_rejects_checksum_mismatch(tmp_path: Path) -> None:
    archive_path = tmp_path / "fixture.zip"
    _write_archive(archive_path)

    with pytest.raises(ValueError, match="MD5 mismatch"):
        fetch_scifact(
            tmp_path / "cache",
            source_url=archive_path.as_uri(),
            expected_md5="0" * 32,
        )


@pytest.mark.parametrize("unsafe_member", ["../escape.txt", "/absolute.txt"])
def test_fetch_rejects_zip_path_traversal(tmp_path: Path, unsafe_member: str) -> None:
    archive_path = tmp_path / "malicious.zip"
    checksum = _write_archive(archive_path, {unsafe_member: "not allowed"})

    with pytest.raises(ValueError, match="unsafe ZIP member"):
        fetch_scifact(
            tmp_path / "cache",
            source_url=archive_path.as_uri(),
            expected_md5=checksum,
        )
    assert not (tmp_path / "escape.txt").exists()


def test_load_rejects_a_tampered_dataset_file(tmp_path: Path) -> None:
    dataset_directory = _fetch_fixture(tmp_path)
    with (dataset_directory / "corpus.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="integrity verification"):
        load_scifact(dataset_directory)


def test_benchmark_reuses_dense_embeddings_and_writes_artifacts(tmp_path: Path) -> None:
    dataset = load_scifact(_fetch_fixture(tmp_path))
    provider = CountingEmbeddingProvider()

    outcome = run_retrieval_benchmark(
        dataset,
        ["bm25", "dense", "dense-mmr", "hybrid"],
        k=2,
        fetch_k=3,
        embedding_provider=provider,
    )
    summary_path = write_benchmark_artifacts(outcome, tmp_path / "results")

    assert provider.document_calls == 1
    assert provider.query_batch_calls == 1
    assert provider.query_calls == 0
    assert outcome.report.limited_run is False
    assert outcome.report.bm25_k1 == 1.2
    assert outcome.report.rrf_rank_constant == 60
    assert outcome.report.dense_vector_storage_bytes is not None
    assert [run.retriever for run in outcome.report.runs] == [
        "bm25",
        "dense",
        "dense-mmr",
        "hybrid",
    ]
    assert all(run.evaluation.mean_recall_at_k == 1 for run in outcome.report.runs)
    assert json.loads(summary_path.read_text(encoding="utf-8"))["query_count"] == 2
    assert (tmp_path / "results" / "runs" / "hybrid.jsonl").is_file()


def test_dense_benchmark_requires_an_embedding_provider(tmp_path: Path) -> None:
    dataset = load_scifact(_fetch_fixture(tmp_path))

    with pytest.raises(ValueError, match="embedding provider"):
        run_retrieval_benchmark(dataset, ["dense"])


def test_decomposed_benchmark_requires_queries_and_records_provenance(tmp_path: Path) -> None:
    dataset = load_scifact(_fetch_fixture(tmp_path))

    with pytest.raises(ValueError, match="requires query decompositions"):
        run_retrieval_benchmark(dataset, ["bm25-decomposed-rrf"], k=2, fetch_k=3)

    outcome = run_retrieval_benchmark(
        dataset,
        ["bm25", "bm25-decomposed-rrf"],
        k=2,
        fetch_k=3,
        query_decompositions={
            "q1": ("alpha evidence", "apple evidence"),
            "q2": ("beta evidence", "banana evidence"),
        },
        query_decomposition_provenance={"artifact_sha256": "a" * 64},
    )

    assert [run.retriever for run in outcome.report.runs] == [
        "bm25",
        "bm25-decomposed-rrf",
    ]
    assert outcome.report.query_decomposition == {"artifact_sha256": "a" * 64}
    assert all(run.evaluation.mean_recall_at_k == 1 for run in outcome.report.runs)


def test_source_aware_benchmark_requires_plans_and_records_provenance(tmp_path: Path) -> None:
    dataset = load_scifact(_fetch_fixture(tmp_path))

    with pytest.raises(ValueError, match="requires source plans"):
        run_retrieval_benchmark(dataset, ["bm25-source-aware"], k=2, fetch_k=3)

    outcome = run_retrieval_benchmark(
        dataset,
        ["bm25", "bm25-source-aware"],
        k=2,
        fetch_k=3,
        source_plans={
            "q1": (SourceScopedQuery("beir/scifact/a", "alpha apple evidence"),),
            "q2": (SourceScopedQuery("beir/scifact/b", "beta banana evidence"),),
        },
        source_plan_provenance={"artifact_sha256": "b" * 64},
    )

    assert [run.retriever for run in outcome.report.runs] == [
        "bm25",
        "bm25-source-aware",
    ]
    assert outcome.report.source_plan == {"artifact_sha256": "b" * 64}
    assert all(run.evaluation.mean_recall_at_k == 1 for run in outcome.report.runs)


def test_benchmark_rejects_unknown_retriever_at_runtime(tmp_path: Path) -> None:
    dataset = load_scifact(_fetch_fixture(tmp_path))

    with pytest.raises(ValueError, match="unknown retrievers"):
        run_retrieval_benchmark(dataset, ["unsupported"])  # type: ignore[list-item]


def test_benchmark_scifact_cli_runs_offline_bm25(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_directory = _fetch_fixture(tmp_path)
    output_directory = tmp_path / "cli-results"

    assert (
        main(
            [
                "benchmark-scifact",
                "--dataset-dir",
                str(dataset_directory),
                "--output-dir",
                str(output_directory),
                "--retrievers",
                "bm25",
                "-k",
                "2",
                "--fetch-k",
                "3",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["query_count"] == 2
    assert output["runs"][0]["mean_ndcg_at_k"] == 1


def test_benchmark_scifact_cli_requires_environment_key_for_dense_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_directory = _fetch_fixture(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        main(
            [
                "benchmark-scifact",
                "--dataset-dir",
                str(dataset_directory),
                "--retrievers",
                "dense",
            ]
        )


def test_public_report_is_compact_and_contains_bootstrap_intervals(tmp_path: Path) -> None:
    dataset = load_scifact(_fetch_fixture(tmp_path))
    outcome = run_retrieval_benchmark(
        dataset,
        ["bm25", "dense", "dense-mmr", "hybrid"],
        k=10,
        fetch_k=10,
        embedding_provider=CountingEmbeddingProvider(),
    )
    summary_path = write_benchmark_artifacts(outcome, tmp_path / "full")
    snapshot_path = tmp_path / "public" / "snapshot.json"
    figure_path = tmp_path / "public" / "figure.svg"

    write_public_benchmark_report(
        summary_path,
        snapshot_path,
        figure_path,
        run_date="2026-08-28",
        bootstrap_resamples=100,
        bootstrap_seed=7,
    )

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["evidence_status"] == "verified_full_public_comparison"
    assert snapshot["statistical_method"]["resamples"] == 100
    assert "per_query" not in snapshot_path.read_text(encoding="utf-8")
    assert len(snapshot["runs"][0]["metrics"]["Recall@10"]["ci_95"]) == 2
    ET.fromstring(figure_path.read_text(encoding="utf-8"))


def test_project_report_uses_distinct_evidence_clusters(tmp_path: Path) -> None:
    query_ids = [f"q{index:03d}" for index in range(80)]
    group_ids = {
        query_id: "group-a" if index < 40 else "group-b" if index < 60 else "group-c"
        for index, query_id in enumerate(query_ids)
    }
    group_values = {"group-a": 0.2, "group-b": 0.4, "group-c": 0.6}
    runs: list[dict[str, object]] = []
    for retriever, offset in (
        ("bm25", 0.0),
        ("dense", 0.1),
        ("dense-mmr", 0.05),
        ("hybrid", 0.15),
    ):
        values = [group_values[group_ids[query_id]] + offset for query_id in query_ids]
        per_query = [
            {
                "query_id": query_id,
                "recall_at_k": value,
                "reciprocal_rank": value,
                "ndcg_at_k": value,
                "retrieved_ids": [],
            }
            for query_id, value in zip(query_ids, values, strict=True)
        ]
        mean = sum(values) / len(values)
        runs.append(
            {
                "retriever": retriever,
                "evaluation": {
                    "k": 10,
                    "query_count": 100,
                    "answerable_query_count": 80,
                    "unanswerable_query_count": 20,
                    "mean_recall_at_k": mean,
                    "mean_reciprocal_rank": mean,
                    "mean_ndcg_at_k": mean,
                    "per_query": per_query,
                },
                "retrieval_latency_ms_p50": 1.0,
                "retrieval_latency_ms_p95": 2.0,
                "run_file": f"runs/{retriever}.jsonl",
            }
        )
    summary = {
        "schema_version": 1,
        "dataset": "technical-papers-v1",
        "split": "reviewed-v1",
        "source_url": None,
        "archive_md5": None,
        "license": "fixture",
        "citation": "fixture",
        "provenance": {"evidence_strategy": "cited_page_all_chunks_v1"},
        "corpus_count": 188,
        "query_count": 100,
        "total_query_count": 100,
        "limited_run": False,
        "k": 10,
        "fetch_k": 40,
        "mmr_lambda": 0.75,
        "bm25_k1": 1.2,
        "bm25_b": 0.75,
        "rrf_rank_constant": 60,
        "dense_weight": 1.0,
        "sparse_weight": 1.0,
        "embedding_model": "fixture-embedding",
        "embedding_dimensions": 3,
        "embedding_batch_size": 16,
        "build_seconds": {},
        "dense_vector_storage_bytes": 1024,
        "relevance_group_ids": group_ids,
        "unique_relevance_group_count": 3,
        "runs": runs,
    }
    summary_path = tmp_path / "summary.json"
    snapshot_path = tmp_path / "snapshot.json"
    figure_path = tmp_path / "figure.svg"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    write_project_benchmark_report(
        summary_path,
        snapshot_path,
        figure_path,
        run_date="2026-08-29",
        bootstrap_resamples=100,
        bootstrap_seed=7,
    )

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    bm25_recall = snapshot["runs"][0]["metrics"]["Recall@10"]
    assert bm25_recall["query_mean"] == pytest.approx(0.35)
    assert bm25_recall["evidence_cluster_macro_mean"] == pytest.approx(0.4)
    assert snapshot["statistical_method"]["cluster_count"] == 3
    assert "per_query" not in snapshot_path.read_text(encoding="utf-8")
    ET.fromstring(figure_path.read_text(encoding="utf-8"))

    sliced_snapshot = build_project_benchmark_snapshot(
        summary,
        run_date="2026-08-29",
        case_slices={
            query_id: ("answerable", "long_document" if index < 20 else "multi_hop")
            for index, query_id in enumerate(query_ids)
        },
        bootstrap_resamples=100,
        bootstrap_seed=7,
    )
    assert sliced_snapshot["diagnostic_slices"]["long_document"]["query_count"] == 20


def test_multihop_report_selects_candidate_only_on_paired_locator_gain() -> None:
    cases = load_project_evaluation_cases(
        Path(__file__).parents[1] / "evals" / "project" / "technical-papers-v1" / "cases.jsonl"
    )
    answerable = [case for case in cases if case.answerability == "answerable"]
    multi_hop = [case for case in answerable if "multi_hop" in case.case_types]

    def run(retriever: str, full_multi_hop: int) -> dict[str, object]:
        rows = []
        for case in answerable:
            is_full = "multi_hop" not in case.case_types or case in multi_hop[:full_multi_hop]
            locators = case.evidence if is_full else case.evidence[:1]
            rows.append(
                {
                    "query_id": case.case_id,
                    "recall_at_k": 1.0 if is_full else 0.5,
                    "reciprocal_rank": 1.0,
                    "ndcg_at_k": 1.0 if is_full else 0.5,
                    "retrieved_ids": [locator.chunk_ids[0] for locator in locators],
                }
            )
        return {
            "retriever": retriever,
            "evaluation": {
                "mean_recall_at_k": 0.9,
                "mean_reciprocal_rank": 1.0,
                "mean_ndcg_at_k": 0.9,
                "per_query": rows,
            },
            "retrieval_latency_ms_p50": 1.0,
            "retrieval_latency_ms_p95": 2.0,
        }

    summary = {
        "dataset": "technical-papers-v1",
        "split": "reviewed-v1",
        "corpus_count": 188,
        "query_count": 100,
        "limited_run": False,
        "k": 10,
        "fetch_k": 40,
        "rrf_rank_constant": 60,
        "query_decomposition": {
            "artifact_sha256": "a" * 64,
            "cases_sha256": "c" * 64,
            "method": "question-only-openai-v1",
            "question_only_input": True,
            "query_count": 15,
        },
        "runs": [run("bm25", 5), run("bm25-decomposed-rrf", 8)],
    }

    snapshot = build_multihop_retrieval_snapshot(
        summary,
        cases,
        run_date="2026-08-29",
        decomposition_sha256="a" * 64,
    )

    assert snapshot["runs"][0]["multi_hop"]["full_locator_coverage"]["successes"] == 5
    assert snapshot["runs"][1]["multi_hop"]["full_locator_coverage"]["successes"] == 8
    assert snapshot["paired_primary_endpoint"]["candidate_gains"] == 3
    assert snapshot["selection"]["adopt_for_generation_experiment"] is True

    baseline_generation_rows = [
        {
            "case_id": case.case_id,
            "variant": "clean",
            "task_pass": index < 7,
            "citations_valid": True,
        }
        for index, case in enumerate(multi_hop)
    ]
    candidate_generation_rows = [
        {
            "case_id": case.case_id,
            "variant": "clean",
            "task_pass": index < 6,
            "citations_valid": True,
        }
        for index, case in enumerate(multi_hop)
    ]
    generation_summary = {
        "dataset": "technical-papers-v1",
        "configuration": {
            "retriever": "bm25",
            "claim_contract_version": "required-claims-v1",
        },
        "provenance": {"cases_sha256": "c" * 64},
    }
    candidate_generation_summary = {
        **generation_summary,
        "configuration": {
            **generation_summary["configuration"],
            "retriever": "bm25-decomposed-rrf",
            "query_decomposition": {"artifact_sha256": "a" * 64},
        },
    }
    add_generation_experiment(
        snapshot,
        generation_summary,
        baseline_generation_rows,
        candidate_generation_summary,
        candidate_generation_rows,
        baseline_summary_sha256="b" * 64,
        baseline_results_sha256="c" * 64,
        candidate_summary_sha256="d" * 64,
        candidate_results_sha256="e" * 64,
    )

    assert snapshot["generation_experiment"]["baseline"]["task_pass"]["successes"] == 7
    assert snapshot["generation_experiment"]["candidate"]["task_pass"]["successes"] == 6
    assert snapshot["selection"]["adopt_as_default"] is False
    ET.fromstring(render_multihop_retrieval_svg(snapshot))

    improved_candidate_rows = [
        {**row, "task_pass": index < 8} for index, row in enumerate(candidate_generation_rows)
    ]
    add_generation_experiment(
        snapshot,
        generation_summary,
        baseline_generation_rows,
        candidate_generation_summary,
        improved_candidate_rows,
        baseline_summary_sha256="b" * 64,
        baseline_results_sha256="c" * 64,
        candidate_summary_sha256="d" * 64,
        candidate_results_sha256="e" * 64,
    )
    assert snapshot["selection"]["adopt_as_default"] is False
    assert snapshot["selection"]["candidate_status"] == "advance to expanded validation"


def test_multihop_report_advances_source_aware_candidate_without_changing_default() -> None:
    cases = load_project_evaluation_cases(
        Path(__file__).parents[1] / "evals" / "project" / "technical-papers-v1" / "cases.jsonl"
    )
    answerable = [case for case in cases if case.answerability == "answerable"]
    multi_hop = [case for case in answerable if "multi_hop" in case.case_types]

    def run(retriever: str, full_multi_hop: int) -> dict[str, object]:
        rows = []
        for case in answerable:
            is_full = "multi_hop" not in case.case_types or case in multi_hop[:full_multi_hop]
            locators = case.evidence if is_full else case.evidence[:1]
            rows.append(
                {
                    "query_id": case.case_id,
                    "recall_at_k": 1.0 if is_full else 0.5,
                    "reciprocal_rank": 1.0,
                    "ndcg_at_k": 1.0 if is_full else 0.5,
                    "retrieved_ids": [locator.chunk_ids[0] for locator in locators],
                }
            )
        return {
            "retriever": retriever,
            "evaluation": {
                "mean_recall_at_k": 0.9,
                "mean_reciprocal_rank": 1.0,
                "mean_ndcg_at_k": 0.9,
                "per_query": rows,
            },
            "retrieval_latency_ms_p50": 1.0,
            "retrieval_latency_ms_p95": 2.0,
        }

    summary = {
        "dataset": "technical-papers-v1",
        "split": "reviewed-v1",
        "corpus_count": 188,
        "query_count": 100,
        "limited_run": False,
        "k": 10,
        "fetch_k": 40,
        "rrf_rank_constant": 60,
        "source_plan": {
            "artifact_sha256": "s" * 64,
            "cases_sha256": "c" * 64,
            "method": "source-catalog-openai-v1",
            "question_and_source_catalog_only": True,
            "plan_count": 15,
        },
        "runs": [run("bm25", 5), run("bm25-source-aware", 7)],
    }
    snapshot = build_multihop_retrieval_snapshot(
        summary,
        cases,
        run_date="2026-08-29",
        source_plan_sha256="s" * 64,
        candidate_retriever="bm25-source-aware",
    )
    baseline_rows = [
        {
            "case_id": case.case_id,
            "variant": "clean",
            "task_pass": index < 7,
            "citations_valid": True,
        }
        for index, case in enumerate(multi_hop)
    ]
    candidate_rows = [
        {
            "case_id": case.case_id,
            "variant": "clean",
            "task_pass": index < 9,
            "citations_valid": True,
        }
        for index, case in enumerate(multi_hop)
    ]
    baseline_summary = {
        "dataset": "technical-papers-v1",
        "configuration": {
            "retriever": "bm25",
            "claim_contract_version": "required-claims-v1",
        },
        "provenance": {"cases_sha256": "c" * 64},
    }
    candidate_summary = {
        **baseline_summary,
        "configuration": {
            **baseline_summary["configuration"],
            "retriever": "bm25-source-aware",
            "source_plan": {"artifact_sha256": "s" * 64},
        },
    }

    add_generation_experiment(
        snapshot,
        baseline_summary,
        baseline_rows,
        candidate_summary,
        candidate_rows,
        baseline_summary_sha256="b" * 64,
        baseline_results_sha256="c" * 64,
        candidate_summary_sha256="d" * 64,
        candidate_results_sha256="e" * 64,
    )

    assert snapshot["source_plan"]["artifact_sha256"] == "s" * 64
    assert snapshot["selection"]["meets_numeric_generation_gate"] is True
    assert snapshot["selection"]["adopt_as_default"] is False
    assert snapshot["selection"]["candidate_status"] == "advance to expanded validation"
    ET.fromstring(render_multihop_retrieval_svg(snapshot))

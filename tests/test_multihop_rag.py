from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import llmqa.multihop_rag as multihop_rag
from llmqa.benchmark import run_retrieval_benchmark, write_benchmark_artifacts
from llmqa.multihop_rag import (
    fetch_multihop_rag,
    load_multihop_rag,
    write_multihop_rag_holdout_manifest,
)
from llmqa.multihop_rag_planning import (
    generate_multihop_rag_decompositions,
    load_multihop_rag_decompositions,
)
from llmqa.multihop_rag_reporting import (
    write_document_diversity_report,
    write_multihop_rag_report,
)

ROOT = Path(__file__).parents[1]
PUBLIC_HOLDOUT = ROOT / "evals" / "public" / "multihop-rag" / "holdout.json"
PUBLIC_DECOMPOSITIONS = ROOT / "evals" / "public" / "multihop-rag" / "decompositions.json"
PUBLIC_SNAPSHOT = ROOT / "docs" / "benchmarks" / "multihop-rag-external-validation-2026-08-29.json"
PUBLIC_FIGURE = ROOT / "docs" / "benchmarks" / "multihop-rag-external-validation-2026-08-29.svg"
PUBLIC_DIVERSITY_CONTRACT = (
    ROOT / "evals" / "public" / "multihop-rag" / "document-diversity-candidate.json"
)
PUBLIC_DIVERSITY_HOLDOUT = (
    ROOT / "evals" / "public" / "multihop-rag" / "document-diversity-confirmation.json"
)
PUBLIC_DIVERSITY_SNAPSHOT = (
    ROOT / "docs" / "benchmarks" / "multihop-rag-document-diversity-2026-08-29.json"
)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode()


def _fixture_files(tmp_path: Path) -> dict[str, Path]:
    corpus = [
        {
            "title": "Alpha launch",
            "author": "A",
            "source": "Publisher A",
            "published_at": "2023-01-01T00:00:00+00:00",
            "category": "technology",
            "url": "https://example.test/alpha",
            "body": "Alpha launched in 2020. The product uses a red engine.",
        },
        {
            "title": "Beta launch",
            "author": "B",
            "source": "Publisher B",
            "published_at": "2023-02-01T00:00:00+00:00",
            "category": "business",
            "url": "https://example.test/beta",
            "body": "Beta launched in 2021. The product uses a blue engine.",
        },
        {
            "title": "Gamma launch",
            "author": None,
            "source": "Publisher C",
            "published_at": "2023-03-01T00:00:00+00:00",
            "category": "business",
            "url": "https://example.test/gamma",
            "body": "Gamma launched in 2022. The product uses a green engine.",
        },
    ]
    queries = [
        {
            "query": "Which products launched in 2020 and 2021?",
            "answer": "Alpha and Beta",
            "question_type": "comparison_query",
            "evidence_list": [
                {
                    "url": "https://example.test/alpha",
                    "fact": "Alpha launched in 2020.",
                },
                {
                    "url": "https://example.test/beta",
                    "fact": "Beta launched in 2021.",
                },
            ],
        },
        {
            "query": "Which product launched after Alpha but before Gamma?",
            "answer": "Beta",
            "question_type": "temporal_query",
            "evidence_list": [
                {
                    "url": "https://example.test/alpha",
                    "fact": "Alpha launched in 2020.",
                },
                {
                    "url": "https://example.test/beta",
                    "fact": "Beta launched in 2021.",
                },
                {
                    "url": "https://example.test/gamma",
                    "fact": "Gamma launched in 2022.",
                },
            ],
        },
        {
            "query": "Which product launched in 2019?",
            "answer": "Insufficient information.",
            "question_type": "null_query",
            "evidence_list": [],
        },
    ]
    paths = {
        "corpus.json": tmp_path / "source-corpus.json",
        "MultiHopRAG.json": tmp_path / "source-queries.json",
    }
    paths["corpus.json"].write_bytes(_json_bytes(corpus))
    paths["MultiHopRAG.json"].write_bytes(_json_bytes(queries))
    return paths


def _fetch_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    paths = _fixture_files(tmp_path)
    checksums = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
    }
    monkeypatch.setattr(multihop_rag, "MULTIHOP_RAG_FILES", checksums)
    monkeypatch.setattr(multihop_rag, "MULTIHOP_RAG_QUERY_COUNT", 3)
    monkeypatch.setattr(multihop_rag, "MULTIHOP_RAG_DOCUMENT_COUNT", 3)
    monkeypatch.setattr(multihop_rag, "_source_url", lambda name: paths[name].as_uri())
    return fetch_multihop_rag(tmp_path / "cache")


def test_fetch_load_and_freeze_stratified_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_directory = _fetch_fixture(tmp_path, monkeypatch)

    full = load_multihop_rag(dataset_directory)
    holdout = load_multihop_rag(dataset_directory, sample_per_stratum=1)

    assert full.dataset.total_query_count == 3
    assert full.dataset.limited_run is False
    assert len(full.dataset.chunks) == 3
    assert full.dataset.judgments[-1].relevance == {}
    assert len(holdout.cases) == 2
    assert holdout.dataset.limited_run is True
    assert {case.stratum for case in holdout.cases} == {
        "comparison_query/2",
        "temporal_query/3",
    }
    assert all(case.evidence_chunk_ids for case in holdout.cases)
    assert holdout.dataset.provenance.details["selection_sha256"] == holdout.selection_sha256
    assert fetch_multihop_rag(tmp_path / "cache") == dataset_directory

    holdout_path = tmp_path / "holdout.json"
    manifest = write_multihop_rag_holdout_manifest(
        holdout,
        holdout_path,
        sample_per_stratum=1,
        frozen_at="2026-08-29T00:00:00Z",
    )
    assert manifest["status"] == "frozen_before_retrieval"
    written = json.loads(holdout_path.read_text(encoding="utf-8"))
    assert all("answer" not in record for record in written["records"])
    assert all("evidence_urls" not in record for record in written["records"])


def test_loader_rejects_tampered_public_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_directory = _fetch_fixture(tmp_path, monkeypatch)
    with (dataset_directory / "corpus.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")

    with pytest.raises(ValueError, match="integrity verification"):
        load_multihop_rag(dataset_directory)


@dataclass
class FakeUsage:
    input_tokens: int = 10
    output_tokens: int = 5
    total_tokens: int = 15


@dataclass
class FakeResponse:
    output_text: str = json.dumps({"subqueries": ["first evidence", "second evidence"]})
    id: str = "resp_fixture"
    model: str = "gpt-5-mini-resolved"
    status: str = "completed"
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        return FakeResponse()


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def test_multihop_rag_decomposition_is_question_only_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_directory = _fetch_fixture(tmp_path, monkeypatch)
    output_path = tmp_path / "decompositions.json"
    client = FakeClient()

    artifact = generate_multihop_rag_decompositions(
        dataset_directory,
        output_path,
        model="gpt-5-mini",
        sample_per_stratum=1,
        client=client,
        generated_at="2026-08-29T00:00:00+00:00",
    )
    loaded, mapping = load_multihop_rag_decompositions(
        output_path,
        dataset_directory,
        sample_per_stratum=1,
    )

    assert artifact["status"] == "complete"
    assert loaded == artifact
    assert len(mapping) == 2
    assert len(client.responses.calls) == 2
    assert all(call["store"] is False for call in client.responses.calls)
    assert all(
        set(json.loads(str(call["input"]))) == {"question"} for call in client.responses.calls
    )

    generate_multihop_rag_decompositions(
        dataset_directory,
        output_path,
        model="gpt-5-mini",
        sample_per_stratum=1,
        client=client,
    )
    assert len(client.responses.calls) == 2


def test_multihop_rag_public_report_reconciles_raw_fixture_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_directory = _fetch_fixture(tmp_path, monkeypatch)
    full_bundle = load_multihop_rag(dataset_directory)
    holdout_bundle = load_multihop_rag(dataset_directory, sample_per_stratum=1)
    holdout_path = tmp_path / "holdout.json"
    write_multihop_rag_holdout_manifest(
        holdout_bundle,
        holdout_path,
        sample_per_stratum=1,
        frozen_at="2026-08-29T00:00:00Z",
    )
    decomposition_path = tmp_path / "decompositions.json"
    generate_multihop_rag_decompositions(
        dataset_directory,
        decomposition_path,
        model="gpt-5-mini",
        sample_per_stratum=1,
        client=FakeClient(),
        generated_at="2026-08-29T00:00:00Z",
    )
    _, mapping = load_multihop_rag_decompositions(
        decomposition_path,
        dataset_directory,
        sample_per_stratum=1,
    )
    full_output = tmp_path / "full"
    holdout_output = tmp_path / "holdout-run"
    write_benchmark_artifacts(
        run_retrieval_benchmark(full_bundle.dataset, ["bm25"], k=10, fetch_k=40),
        full_output,
    )
    write_benchmark_artifacts(
        run_retrieval_benchmark(
            holdout_bundle.dataset,
            ["bm25", "bm25-decomposed-rrf"],
            k=10,
            fetch_k=40,
            query_decompositions=mapping,
            query_decomposition_provenance={"fixture": True},
        ),
        holdout_output,
    )
    snapshot_path = tmp_path / "snapshot.json"
    figure_path = tmp_path / "figure.svg"

    snapshot = write_multihop_rag_report(
        dataset_directory,
        holdout_path,
        decomposition_path,
        full_output / "summary.json",
        full_output / "runs" / "bm25.jsonl",
        holdout_output / "summary.json",
        holdout_output / "runs" / "bm25.jsonl",
        holdout_output / "runs" / "bm25-decomposed-rrf.jsonl",
        snapshot_path,
        figure_path,
        run_date="2026-08-29",
        sample_per_stratum=1,
    )

    assert snapshot["dataset"]["public_query_count"] == 3
    assert snapshot["holdout"]["query_count"] == 2
    assert snapshot["decision"]["generation_run"] == "not_run"
    assert snapshot_path.is_file()
    ET.parse(figure_path)

    diversity_output = tmp_path / "diversity-run"
    write_benchmark_artifacts(
        run_retrieval_benchmark(
            holdout_bundle.dataset,
            ["bm25", "bm25-document-diverse"],
            k=10,
            fetch_k=100,
        ),
        diversity_output,
    )
    contract = {
        "status": "preregistered_before_candidate_retrieval",
        "preregistered_at": "2026-08-29T00:00:00Z",
        "confirmation_manifest_sha256": hashlib.sha256(holdout_path.read_bytes()).hexdigest(),
        "selection_sha256": holdout_bundle.selection_sha256,
        "candidate": {
            "name": "bm25-document-diverse",
            "description": (
                "Fetch the top 100 BM25 chunks, retain only the highest-ranked chunk from each "
                "source document, and return the first 10 distinct documents."
            ),
            "k": 10,
            "fetch_k": 100,
            "maximum_chunks_per_document": 1,
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
        },
        "hypothesis": "fixture hypothesis",
        "primary_endpoint": "full_evidence_coverage_at_10",
        "guardrails": ["mean_recall_at_10"],
        "decision_rule": "fixture rule",
    }
    contract_path = tmp_path / "candidate.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    diversity_snapshot_path = tmp_path / "diversity-snapshot.json"

    diversity_snapshot = write_document_diversity_report(
        dataset_directory,
        contract_path,
        holdout_path,
        diversity_output / "summary.json",
        diversity_output / "runs" / "bm25.jsonl",
        diversity_output / "runs" / "bm25-document-diverse.jsonl",
        diversity_snapshot_path,
        run_date="2026-08-29",
        sample_per_stratum=1,
        stratum_offset=0,
    )

    assert diversity_snapshot["decision"]["retrieval_gate"] == "failed"
    assert diversity_snapshot["decision"]["full_corpus_candidate_run"] == "not_run"
    assert diversity_snapshot_path.is_file()


def test_committed_multihop_rag_holdout_and_negative_result_reconcile() -> None:
    holdout = json.loads(PUBLIC_HOLDOUT.read_text(encoding="utf-8"))
    decompositions = json.loads(PUBLIC_DECOMPOSITIONS.read_text(encoding="utf-8"))
    snapshot = json.loads(PUBLIC_SNAPSHOT.read_text(encoding="utf-8"))

    assert holdout["status"] == "frozen_before_retrieval"
    assert holdout["selected_query_count"] == 49
    assert set(holdout["strata"].values()) == {7}
    assert decompositions["status"] == "complete"
    assert decompositions["query_count"] == 49
    assert len(decompositions["records"]) == 49
    assert holdout["selection_sha256"] == decompositions["selection_sha256"]
    assert snapshot["holdout"]["selection_sha256"] == holdout["selection_sha256"]

    full_coverage = snapshot["full_bm25_baseline"]["coverage"]["full_evidence_coverage"]
    assert full_coverage == {
        "passes": 718,
        "rate": 718 / 2255,
        "total": 2255,
    }
    comparison = snapshot["holdout_comparison"]
    assert comparison["bm25"]["coverage"]["full_evidence_coverage"]["passes"] == 15
    assert comparison["bm25-decomposed-rrf"]["coverage"]["full_evidence_coverage"]["passes"] == 13
    assert comparison["paired_full_coverage"] == {
        "candidate_gains": 2,
        "candidate_losses": 4,
        "gain_query_ids": ["mhr-q-42d8e8ba963af35a", "mhr-q-a17b53074897994b"],
        "loss_query_ids": [
            "mhr-q-325f99830534201f",
            "mhr-q-668603c09dfc09c5",
            "mhr-q-86ce751dbfcf6227",
            "mhr-q-ae1dea5f5727d813",
        ],
        "mcnemar_exact_p": 0.6875,
        "ties": 43,
    }
    assert snapshot["decision"]["retrieval_gate"] == "failed"
    assert snapshot["decision"]["generation_run"] == "not_run"
    assert (
        snapshot["artifact_sha256"]["holdout_manifest"]
        == hashlib.sha256(PUBLIC_HOLDOUT.read_bytes()).hexdigest()
    )
    assert (
        snapshot["artifact_sha256"]["decompositions"]
        == hashlib.sha256(PUBLIC_DECOMPOSITIONS.read_bytes()).hexdigest()
    )
    ET.parse(PUBLIC_FIGURE)


def test_committed_document_diversity_confirmation_reconciles() -> None:
    contract = json.loads(PUBLIC_DIVERSITY_CONTRACT.read_text(encoding="utf-8"))
    holdout = json.loads(PUBLIC_DIVERSITY_HOLDOUT.read_text(encoding="utf-8"))
    snapshot = json.loads(PUBLIC_DIVERSITY_SNAPSHOT.read_text(encoding="utf-8"))

    assert contract["status"] == "preregistered_before_candidate_retrieval"
    assert holdout["status"] == "frozen_before_retrieval"
    assert holdout["stratum_offset"] == 7
    assert holdout["selected_query_count"] == 49
    assert contract["selection_sha256"] == holdout["selection_sha256"]
    assert snapshot["confirmation"]["selection_sha256"] == holdout["selection_sha256"]
    comparison = snapshot["comparison"]
    assert comparison["bm25"]["coverage"]["full_evidence_coverage"]["passes"] == 14
    assert comparison["bm25-document-diverse"]["coverage"]["full_evidence_coverage"]["passes"] == 9
    assert comparison["paired_full_coverage"] == {
        "candidate_gains": 0,
        "candidate_losses": 5,
        "gain_query_ids": [],
        "loss_query_ids": [
            "mhr-q-1acbad14990a3141",
            "mhr-q-93d499b4a19574e9",
            "mhr-q-bc87d53406c65d15",
            "mhr-q-d4882be7135187a5",
            "mhr-q-db0aa3a29a4a0240",
        ],
        "mcnemar_exact_p": 0.0625,
        "ties": 44,
    }
    assert comparison["mechanism_audit"] == {
        "relevant_baseline_hits_removed_by_candidate": 20,
        "removed_hits_with_same_document_replacement": 20,
        "queries_with_multiple_gold_chunks_in_one_document": 4,
        "structurally_ineligible_query_ids": [
            "mhr-q-682ad1d92c24682f",
            "mhr-q-92e63112b10a9ec3",
            "mhr-q-a20ce0c005dee8a1",
            "mhr-q-aa925dc451fb84c1",
        ],
    }
    assert snapshot["decision"]["retrieval_gate"] == "failed"
    assert snapshot["decision"]["generation_run"] == "not_run"
    assert (
        snapshot["artifact_sha256"]["candidate_contract"]
        == hashlib.sha256(PUBLIC_DIVERSITY_CONTRACT.read_bytes()).hexdigest()
    )
    assert (
        snapshot["artifact_sha256"]["confirmation_manifest"]
        == hashlib.sha256(PUBLIC_DIVERSITY_HOLDOUT.read_bytes()).hexdigest()
    )

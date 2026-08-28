from __future__ import annotations

import hashlib
import json
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
from llmqa.cli import main


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
    assert provider.query_calls == 2
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

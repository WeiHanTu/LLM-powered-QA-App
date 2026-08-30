"""Reproducible public retrieval benchmarks with explicit data provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import ssl
import stat
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import numpy as np
from certifi import where as certifi_ca_bundle

from llmqa.domain import Chunk, SourceScopedQuery
from llmqa.embeddings import EmbeddingProvider, FloatMatrix
from llmqa.evaluation import RetrievalEvaluation, RetrievalJudgment, evaluate_rankings
from llmqa.retrieval import (
    BM25Retriever,
    DecomposedQueryRetriever,
    DocumentDiverseRetriever,
    FaissRetriever,
    HybridRetriever,
    SourceAwareBM25Retriever,
)

SCIFACT_NAME = "beir/scifact"
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
SCIFACT_LICENSE = "CC BY-NC 2.0"
SCIFACT_CITATION = (
    "Wadden et al. (2020), Fact or Fiction: Verifying Scientific Claims, "
    "https://aclanthology.org/2020.emnlp-main.609/"
)
SCIFACT_REQUIRED_FILES = ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")
SCIFACT_MANIFEST = "benchmark-manifest.json"

type RetrieverName = Literal["bm25", "dense", "dense-mmr", "hybrid"]
RETRIEVER_NAMES: tuple[RetrieverName, ...] = ("bm25", "dense", "dense-mmr", "hybrid")
type ProjectRetrieverName = Literal[
    "bm25",
    "dense",
    "dense-mmr",
    "hybrid",
    "bm25-decomposed-rrf",
    "bm25-document-diverse",
    "bm25-source-aware",
]
PROJECT_RETRIEVER_NAMES: tuple[ProjectRetrieverName, ...] = (
    "bm25",
    "dense",
    "dense-mmr",
    "hybrid",
    "bm25-decomposed-rrf",
    "bm25-document-diverse",
    "bm25-source-aware",
)


@dataclass(frozen=True, slots=True)
class BenchmarkDatasetManifest:
    schema_version: int
    dataset: str
    source_url: str
    archive_md5: str
    license: str
    citation: str
    files_sha256: dict[str, str]


@dataclass(frozen=True, slots=True)
class BenchmarkProvenance:
    source_url: str | None
    archive_md5: str | None
    license: str
    citation: str
    details: dict[str, object]


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkDataset:
    name: str
    split: str
    chunks: tuple[Chunk, ...]
    judgments: tuple[RetrievalJudgment, ...]
    total_query_count: int
    provenance: BenchmarkProvenance

    @property
    def limited_run(self) -> bool:
        return len(self.judgments) < self.total_query_count


@dataclass(frozen=True, slots=True)
class BenchmarkRunSummary:
    retriever: str
    evaluation: RetrievalEvaluation
    retrieval_latency_ms_p50: float
    retrieval_latency_ms_p95: float
    run_file: str


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkReport:
    schema_version: int
    dataset: str
    split: str
    source_url: str | None
    archive_md5: str | None
    license: str
    citation: str
    provenance: dict[str, object]
    corpus_count: int
    query_count: int
    total_query_count: int
    limited_run: bool
    k: int
    fetch_k: int
    mmr_lambda: float
    bm25_k1: float
    bm25_b: float
    rrf_rank_constant: int
    dense_weight: float
    sparse_weight: float
    query_decomposition: dict[str, object] | None
    source_plan: dict[str, object] | None
    embedding_model: str | None
    embedding_dimensions: int | None
    embedding_batch_size: int | None
    build_seconds: dict[str, float]
    dense_vector_storage_bytes: int | None
    relevance_group_ids: dict[str, str]
    unique_relevance_group_count: int
    runs: tuple[BenchmarkRunSummary, ...]


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkOutcome:
    report: RetrievalBenchmarkReport
    rankings: dict[str, dict[str, tuple[str, ...]]]


def _hash_file(path: Path, algorithm: str) -> str:
    if algorithm == "md5":
        digest = hashlib.md5(usedforsecurity=False)
    elif algorithm == "sha256":
        digest = hashlib.sha256()
    else:
        raise ValueError(f"unsupported hash algorithm {algorithm!r}")
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "llmqa-benchmark/0.2"})
    tls_context = ssl.create_default_context(cafile=certifi_ca_bundle())
    with (
        urllib.request.urlopen(request, timeout=60, context=tls_context) as response,  # noqa: S310
        destination.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        for member in members:
            relative = PurePosixPath(member.filename)
            file_type = (member.external_attr >> 16) & 0o170000
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in member.filename
                or any(":" in part for part in relative.parts)
                or file_type == stat.S_IFLNK
            ):
                raise ValueError(f"unsafe ZIP member {member.filename!r}")

        for member in members:
            relative = PurePosixPath(member.filename)
            target = destination.joinpath(*relative.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _required_paths(dataset_directory: Path) -> dict[str, Path]:
    return {
        relative: dataset_directory.joinpath(*PurePosixPath(relative).parts)
        for relative in SCIFACT_REQUIRED_FILES
    }


def _write_manifest(
    dataset_directory: Path,
    *,
    source_url: str,
    archive_md5: str,
) -> BenchmarkDatasetManifest:
    required = _required_paths(dataset_directory)
    missing = [relative for relative, path in required.items() if not path.is_file()]
    if missing:
        raise ValueError(f"SciFact archive is missing required files: {missing}")
    manifest = BenchmarkDatasetManifest(
        schema_version=1,
        dataset=SCIFACT_NAME,
        source_url=source_url,
        archive_md5=archive_md5,
        license=SCIFACT_LICENSE,
        citation=SCIFACT_CITATION,
        files_sha256={relative: _hash_file(path, "sha256") for relative, path in required.items()},
    )
    (dataset_directory / SCIFACT_MANIFEST).write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_manifest(dataset_directory: Path) -> BenchmarkDatasetManifest:
    manifest_path = dataset_directory / SCIFACT_MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"dataset manifest is missing: {manifest_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("files_sha256"), dict):
        raise ValueError("dataset manifest must contain a files_sha256 JSON object")
    manifest = BenchmarkDatasetManifest(
        schema_version=int(raw["schema_version"]),
        dataset=str(raw["dataset"]),
        source_url=str(raw["source_url"]),
        archive_md5=str(raw["archive_md5"]),
        license=str(raw["license"]),
        citation=str(raw["citation"]),
        files_sha256={
            str(relative): str(checksum) for relative, checksum in raw["files_sha256"].items()
        },
    )
    if manifest.schema_version != 1 or manifest.dataset != SCIFACT_NAME:
        raise ValueError("dataset manifest has an unsupported schema or dataset name")
    required = _required_paths(dataset_directory)
    if set(manifest.files_sha256) != set(required):
        raise ValueError("dataset manifest does not cover the exact required file set")
    for relative, path in required.items():
        if not path.is_file() or _hash_file(path, "sha256") != manifest.files_sha256[relative]:
            raise ValueError(f"dataset file failed integrity verification: {relative}")
    return manifest


def fetch_scifact(
    cache_directory: Path,
    *,
    source_url: str = SCIFACT_URL,
    expected_md5: str = SCIFACT_MD5,
) -> Path:
    """Download, verify, and safely extract the BEIR SciFact archive."""

    cache_directory.mkdir(parents=True, exist_ok=True)
    dataset_directory = cache_directory / "scifact"
    if dataset_directory.is_dir() and (dataset_directory / SCIFACT_MANIFEST).is_file():
        manifest = _load_manifest(dataset_directory)
        if manifest.source_url != source_url or manifest.archive_md5 != expected_md5:
            raise ValueError("cached SciFact manifest does not match the requested source and MD5")
        return dataset_directory
    if dataset_directory.exists():
        raise ValueError(
            f"incomplete dataset directory exists at {dataset_directory}; move it aside and retry"
        )

    archive_path = cache_directory / "scifact.zip"
    if not archive_path.is_file():
        partial_path = cache_directory / ".scifact.zip.part"
        if partial_path.exists():
            raise ValueError(f"stale partial download exists at {partial_path}")
        _download_file(source_url, partial_path)
        actual_md5 = _hash_file(partial_path, "md5")
        if actual_md5 != expected_md5:
            partial_path.unlink()
            raise ValueError(
                f"SciFact archive MD5 mismatch: expected {expected_md5}, got {actual_md5}"
            )
        partial_path.replace(archive_path)
    else:
        actual_md5 = _hash_file(archive_path, "md5")
        if actual_md5 != expected_md5:
            raise ValueError(
                f"SciFact archive MD5 mismatch: expected {expected_md5}, got {actual_md5}"
            )

    with tempfile.TemporaryDirectory(prefix="scifact-extract-", dir=cache_directory) as temp:
        extraction_directory = Path(temp)
        _safe_extract_zip(archive_path, extraction_directory)
        extracted_dataset = extraction_directory / "scifact"
        _write_manifest(
            extracted_dataset,
            source_url=source_url,
            archive_md5=expected_md5,
        )
        shutil.move(str(extracted_dataset), dataset_directory)
    _load_manifest(dataset_directory)
    return dataset_directory


def _read_json_objects(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append({str(key): value for key, value in raw.items()})
    return rows


def load_scifact(
    dataset_directory: Path,
    *,
    limit_queries: int | None = None,
) -> RetrievalBenchmarkDataset:
    """Load the verified SciFact test split into LLMQA domain contracts."""

    if limit_queries is not None and limit_queries <= 0:
        raise ValueError("limit_queries must be positive")
    manifest = _load_manifest(dataset_directory)

    chunks: list[Chunk] = []
    seen_chunk_ids: set[str] = set()
    for row in _read_json_objects(dataset_directory / "corpus.jsonl"):
        chunk_id = str(row["_id"])
        title = str(row.get("title", "")).strip()
        body = str(row.get("text", "")).strip()
        if not chunk_id or not body or chunk_id in seen_chunk_ids:
            raise ValueError(f"invalid or duplicate SciFact corpus ID {chunk_id!r}")
        seen_chunk_ids.add(chunk_id)
        text = f"{title}\n\n{body}" if title else body
        chunks.append(
            Chunk(
                id=chunk_id,
                text=text,
                source=f"{SCIFACT_NAME}/{chunk_id}",
                metadata={"benchmark": SCIFACT_NAME, "title": title},
            )
        )

    queries: dict[str, str] = {}
    for row in _read_json_objects(dataset_directory / "queries.jsonl"):
        query_id = str(row["_id"])
        query = str(row["text"]).strip()
        if not query_id or not query or query_id in queries:
            raise ValueError(f"invalid or duplicate SciFact query ID {query_id!r}")
        queries[query_id] = query

    relevance: dict[str, dict[str, float]] = {}
    with (dataset_directory / "qrels" / "test.tsv").open(encoding="utf-8", newline="") as handle:
        for qrel_row in csv.DictReader(handle, delimiter="\t"):
            query_id = str(qrel_row["query-id"])
            corpus_id = str(qrel_row["corpus-id"])
            score = float(qrel_row["score"])
            if query_id not in queries:
                raise ValueError(f"qrels reference unknown query ID {query_id!r}")
            if corpus_id not in seen_chunk_ids:
                raise ValueError(f"qrels reference unknown corpus ID {corpus_id!r}")
            relevance.setdefault(query_id, {})[corpus_id] = score

    total_query_count = len(relevance)
    selected_query_ids = list(relevance)[:limit_queries]
    judgments = tuple(
        RetrievalJudgment(query_id, queries[query_id], relevance[query_id])
        for query_id in selected_query_ids
    )
    return RetrievalBenchmarkDataset(
        name=SCIFACT_NAME,
        split="test",
        chunks=tuple(chunks),
        judgments=judgments,
        total_query_count=total_query_count,
        provenance=BenchmarkProvenance(
            source_url=manifest.source_url,
            archive_md5=manifest.archive_md5,
            license=manifest.license,
            citation=manifest.citation,
            details={
                "schema_version": manifest.schema_version,
                "files_sha256": manifest.files_sha256,
            },
        ),
    )


class _QueryCachingEmbeddingProvider:
    def __init__(self, delegate: EmbeddingProvider) -> None:
        self.model = delegate.model
        self._delegate = delegate
        self._queries: dict[str, FloatMatrix] = {}

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        return self._delegate.embed_documents(texts)

    def prime_queries(self, texts: Sequence[str]) -> None:
        unique_texts = list(dict.fromkeys(texts))
        if not unique_texts:
            raise ValueError("at least one benchmark query is required")
        vectors = self._delegate.embed_queries(unique_texts)
        if vectors.ndim != 2 or vectors.shape[0] != len(unique_texts):
            raise ValueError("query embedding provider returned an unexpected matrix shape")
        self._queries.update(
            {text: vectors[index : index + 1].copy() for index, text in enumerate(unique_texts)}
        )

    def embed_queries(self, texts: Sequence[str]) -> FloatMatrix:
        if not texts:
            raise ValueError("at least one query is required")
        return np.concatenate([self.embed_query(text) for text in texts], axis=0)

    def embed_query(self, text: str) -> FloatMatrix:
        if text not in self._queries:
            self._queries[text] = self._delegate.embed_query(text)
        return self._queries[text].copy()


def _provider_integer(provider: EmbeddingProvider | None, attribute: str) -> int | None:
    value = getattr(provider, attribute, None)
    return value if isinstance(value, int) else None


def _relevance_group_ids(judgments: Sequence[RetrievalJudgment]) -> dict[str, str]:
    groups: dict[str, str] = {}
    signatures_by_group: dict[str, str] = {}
    for judgment in judgments:
        positive_relevance = sorted(
            (chunk_id, float(grade))
            for chunk_id, grade in judgment.relevance.items()
            if float(grade) > 0
        )
        if not positive_relevance:
            continue
        signature = json.dumps(positive_relevance, separators=(",", ":"))
        group_id = f"evidence-{hashlib.sha256(signature.encode()).hexdigest()[:12]}"
        prior_signature = signatures_by_group.setdefault(group_id, signature)
        if prior_signature != signature:
            raise ValueError("relevance-group hash collision detected")
        groups[judgment.query_id] = group_id
    return groups


def run_retrieval_benchmark(
    dataset: RetrievalBenchmarkDataset,
    retrievers: Sequence[ProjectRetrieverName],
    *,
    k: int = 10,
    fetch_k: int = 40,
    mmr_lambda: float = 0.75,
    bm25_k1: float = 1.2,
    bm25_b: float = 0.75,
    rrf_rank_constant: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    embedding_provider: EmbeddingProvider | None = None,
    query_decompositions: Mapping[str, Sequence[str]] | None = None,
    query_decomposition_provenance: Mapping[str, object] | None = None,
    source_plans: Mapping[str, Sequence[SourceScopedQuery]] | None = None,
    source_plan_provenance: Mapping[str, object] | None = None,
) -> RetrievalBenchmarkOutcome:
    """Run requested retrievers over one immutable corpus/query/qrels view."""

    if not retrievers:
        raise ValueError("at least one retriever is required")
    requested = tuple(dict.fromkeys(retrievers))
    unknown_retrievers = set(requested) - set(PROJECT_RETRIEVER_NAMES)
    if unknown_retrievers:
        raise ValueError(f"unknown retrievers: {sorted(unknown_retrievers)}")
    if not dataset.judgments:
        raise ValueError("benchmark dataset must contain at least one judgment")
    if k <= 0 or fetch_k < k:
        raise ValueError("k must be positive and fetch_k must be at least k")
    if not 0 <= mmr_lambda <= 1:
        raise ValueError("mmr_lambda must be between 0 and 1")
    needs_dense = any(name in {"dense", "dense-mmr", "hybrid"} for name in requested)
    needs_sparse = any(
        name
        in {
            "bm25",
            "hybrid",
            "bm25-decomposed-rrf",
            "bm25-document-diverse",
            "bm25-source-aware",
        }
        for name in requested
    )
    if needs_dense and embedding_provider is None:
        raise ValueError("dense benchmark modes require an embedding provider")
    if "bm25-decomposed-rrf" in requested and query_decompositions is None:
        raise ValueError("decomposed-query benchmark mode requires query decompositions")
    if "bm25-source-aware" in requested and source_plans is None:
        raise ValueError("source-aware benchmark mode requires source plans")
    if query_decompositions is not None:
        judgment_ids = {judgment.query_id for judgment in dataset.judgments}
        unknown_query_ids = set(query_decompositions) - judgment_ids
        if unknown_query_ids:
            raise ValueError(
                f"query decompositions contain unknown query IDs: {sorted(unknown_query_ids)}"
            )
        for query_id, subqueries in query_decompositions.items():
            if not subqueries or any(not subquery.strip() for subquery in subqueries):
                raise ValueError(f"query decomposition {query_id!r} must contain non-empty queries")
    if source_plans is not None:
        judgment_ids = {judgment.query_id for judgment in dataset.judgments}
        unknown_query_ids = set(source_plans) - judgment_ids
        if unknown_query_ids:
            raise ValueError(f"source plans contain unknown query IDs: {sorted(unknown_query_ids)}")
        corpus_source_ids = {chunk.source for chunk in dataset.chunks}
        for query_id, steps in source_plans.items():
            if not steps or any(
                step.source_id not in corpus_source_ids or not step.query.strip() for step in steps
            ):
                raise ValueError(
                    f"source plan {query_id!r} must contain valid source-scoped queries"
                )

    build_seconds: dict[str, float] = {}
    sparse: BM25Retriever | None = None
    if needs_sparse:
        started = time.perf_counter()
        sparse = BM25Retriever(dataset.chunks, k1=bm25_k1, b=bm25_b)
        build_seconds["bm25"] = time.perf_counter() - started

    dense: FaissRetriever | None = None
    if needs_dense:
        assert embedding_provider is not None
        cached_provider = _QueryCachingEmbeddingProvider(embedding_provider)
        started = time.perf_counter()
        dense = FaissRetriever.from_chunks(list(dataset.chunks), cached_provider)
        build_seconds["dense"] = time.perf_counter() - started
        started = time.perf_counter()
        cached_provider.prime_queries([judgment.query for judgment in dataset.judgments])
        build_seconds["queries"] = time.perf_counter() - started

    hybrid: HybridRetriever | None = None
    if "hybrid" in requested:
        assert dense is not None and sparse is not None
        hybrid = HybridRetriever(
            dense,
            sparse,
            rank_constant=rrf_rank_constant,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

    decomposed: DecomposedQueryRetriever | None = None
    if "bm25-decomposed-rrf" in requested:
        assert sparse is not None
        decomposed = DecomposedQueryRetriever(sparse, rank_constant=rrf_rank_constant)

    document_diverse: DocumentDiverseRetriever | None = None
    if "bm25-document-diverse" in requested:
        assert sparse is not None
        document_diverse = DocumentDiverseRetriever(sparse)

    source_aware: SourceAwareBM25Retriever | None = None
    if "bm25-source-aware" in requested:
        source_aware = SourceAwareBM25Retriever(
            dataset.chunks,
            k1=bm25_k1,
            b=bm25_b,
            rank_constant=rrf_rank_constant,
        )

    all_rankings: dict[str, dict[str, tuple[str, ...]]] = {}
    summaries: list[BenchmarkRunSummary] = []
    for retriever_name in requested:
        rankings: dict[str, tuple[str, ...]] = {}
        latencies_ms: list[float] = []
        for judgment in dataset.judgments:
            started = time.perf_counter()
            if retriever_name == "bm25":
                assert sparse is not None
                results = sparse.search(judgment.query, k=k)
            elif retriever_name == "dense":
                assert dense is not None
                results = dense.search(judgment.query, k=k, fetch_k=k, mmr_lambda=1.0)
            elif retriever_name == "dense-mmr":
                assert dense is not None
                results = dense.search(
                    judgment.query,
                    k=k,
                    fetch_k=fetch_k,
                    mmr_lambda=mmr_lambda,
                )
            elif retriever_name == "hybrid":
                assert hybrid is not None
                results = hybrid.search(judgment.query, k=k, fetch_k=fetch_k)
            elif retriever_name == "bm25-decomposed-rrf":
                assert decomposed is not None and query_decompositions is not None
                results = decomposed.search(
                    judgment.query,
                    subqueries=query_decompositions.get(judgment.query_id, ()),
                    k=k,
                    fetch_k=fetch_k,
                )
            elif retriever_name == "bm25-document-diverse":
                assert document_diverse is not None
                results = document_diverse.search(
                    judgment.query,
                    k=k,
                    fetch_k=fetch_k,
                )
            else:
                assert source_aware is not None and source_plans is not None
                plan = source_plans.get(judgment.query_id)
                if plan is None:
                    assert sparse is not None
                    results = sparse.search(judgment.query, k=k)
                else:
                    results = source_aware.search(
                        judgment.query,
                        steps=plan,
                        k=k,
                        fetch_k=fetch_k,
                    )
            latencies_ms.append((time.perf_counter() - started) * 1000)
            rankings[judgment.query_id] = tuple(result.chunk.id for result in results)

        evaluation = evaluate_rankings(dataset.judgments, rankings, k=k)
        all_rankings[retriever_name] = rankings
        summaries.append(
            BenchmarkRunSummary(
                retriever=retriever_name,
                evaluation=evaluation,
                retrieval_latency_ms_p50=float(np.percentile(latencies_ms, 50)),
                retrieval_latency_ms_p95=float(np.percentile(latencies_ms, 95)),
                run_file=f"runs/{retriever_name}.jsonl",
            )
        )

    relevance_group_ids = _relevance_group_ids(dataset.judgments)
    report = RetrievalBenchmarkReport(
        schema_version=1,
        dataset=dataset.name,
        split=dataset.split,
        source_url=dataset.provenance.source_url,
        archive_md5=dataset.provenance.archive_md5,
        license=dataset.provenance.license,
        citation=dataset.provenance.citation,
        provenance=dataset.provenance.details,
        corpus_count=len(dataset.chunks),
        query_count=len(dataset.judgments),
        total_query_count=dataset.total_query_count,
        limited_run=dataset.limited_run,
        k=k,
        fetch_k=fetch_k,
        mmr_lambda=mmr_lambda,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        rrf_rank_constant=rrf_rank_constant,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        query_decomposition=(
            dict(query_decomposition_provenance)
            if query_decomposition_provenance is not None
            else None
        ),
        source_plan=(dict(source_plan_provenance) if source_plan_provenance is not None else None),
        embedding_model=embedding_provider.model if embedding_provider is not None else None,
        embedding_dimensions=_provider_integer(embedding_provider, "dimensions"),
        embedding_batch_size=_provider_integer(embedding_provider, "batch_size"),
        build_seconds=build_seconds,
        dense_vector_storage_bytes=dense.vector_storage_bytes if dense is not None else None,
        relevance_group_ids=relevance_group_ids,
        unique_relevance_group_count=len(set(relevance_group_ids.values())),
        runs=tuple(summaries),
    )
    return RetrievalBenchmarkOutcome(report=report, rankings=all_rankings)


def write_benchmark_artifacts(
    outcome: RetrievalBenchmarkOutcome,
    output_directory: Path,
) -> Path:
    """Write run files and a summary whose only nondeterministic fields are timings."""

    runs_directory = output_directory / "runs"
    runs_directory.mkdir(parents=True, exist_ok=True)
    for run in outcome.report.runs:
        rankings = outcome.rankings[run.retriever]
        rows = [
            json.dumps(
                {"query_id": query_id, "retrieved_ids": list(retrieved_ids)},
                separators=(",", ":"),
                sort_keys=True,
            )
            for query_id, retrieved_ids in rankings.items()
        ]
        (output_directory / run.run_file).write_text("\n".join(rows) + "\n", encoding="utf-8")

    summary_path = output_directory / "summary.json"
    summary_path.write_text(
        json.dumps(asdict(outcome.report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path

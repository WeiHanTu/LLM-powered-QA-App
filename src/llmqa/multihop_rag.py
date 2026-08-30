"""Pinned, integrity-checked adapter for the public MultiHop-RAG benchmark."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import ssl
import tempfile
import unicodedata
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import tiktoken
from certifi import where as certifi_ca_bundle

from llmqa.benchmark import BenchmarkProvenance, RetrievalBenchmarkDataset
from llmqa.domain import Chunk, MetadataValue
from llmqa.evaluation import RetrievalJudgment
from llmqa.ingest import TOKEN_ENCODING

MULTIHOP_RAG_NAME = "yixuantt/MultiHopRAG"
MULTIHOP_RAG_REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
MULTIHOP_RAG_LICENSE = "ODC-BY 1.0"
MULTIHOP_RAG_CITATION = (
    "Tang and Yang (2024), MultiHop-RAG: Benchmarking Retrieval-Augmented Generation "
    "for Multi-Hop Queries, https://openreview.net/forum?id=t4eB3zYWBK"
)
MULTIHOP_RAG_FILES = {
    "MultiHopRAG.json": "03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff",
    "corpus.json": "20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f",
}
MULTIHOP_RAG_MANIFEST = "benchmark-manifest.json"
MULTIHOP_RAG_QUERY_COUNT = 2_556
MULTIHOP_RAG_DOCUMENT_COUNT = 609
MULTIHOP_RAG_CHUNK_SIZE_TOKENS = 220
MULTIHOP_RAG_HOLDOUT_METHOD = "sha256-min-per-question-type-hop-count-v1"
MULTIHOP_RAG_QUESTION_TYPES = frozenset(
    {"comparison_query", "inference_query", "temporal_query", "null_query"}
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“‘])")


@dataclass(frozen=True, slots=True)
class MultiHopRAGManifest:
    schema_version: int
    dataset: str
    revision: str
    source_url: str
    license: str
    citation: str
    files_sha256: dict[str, str]


@dataclass(frozen=True, slots=True)
class MultiHopRAGCase:
    query_id: str
    query: str
    answer: str
    question_type: str
    evidence_count: int
    evidence_chunk_ids: tuple[str, ...]
    evidence_urls: tuple[str, ...]

    @property
    def stratum(self) -> str:
        return f"{self.question_type}/{self.evidence_count}"


@dataclass(frozen=True, slots=True)
class MultiHopRAGBundle:
    dataset: RetrievalBenchmarkDataset
    cases: tuple[MultiHopRAGCase, ...]
    selection_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "llmqa-benchmark/0.2"})
    context = ssl.create_default_context(cafile=certifi_ca_bundle())
    with (
        urllib.request.urlopen(request, timeout=60, context=context) as response,  # noqa: S310
        destination.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)


def _source_url(filename: str) -> str:
    return (
        "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/"
        f"{MULTIHOP_RAG_REVISION}/{filename}"
    )


def _write_manifest(dataset_directory: Path) -> MultiHopRAGManifest:
    manifest = MultiHopRAGManifest(
        schema_version=1,
        dataset=MULTIHOP_RAG_NAME,
        revision=MULTIHOP_RAG_REVISION,
        source_url=(
            f"https://huggingface.co/datasets/yixuantt/MultiHopRAG/tree/{MULTIHOP_RAG_REVISION}"
        ),
        license=MULTIHOP_RAG_LICENSE,
        citation=MULTIHOP_RAG_CITATION,
        files_sha256=dict(MULTIHOP_RAG_FILES),
    )
    destination = dataset_directory / MULTIHOP_RAG_MANIFEST
    payload = json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=dataset_directory, delete=False
    ) as handle:
        handle.write(payload)
        temporary_path = Path(handle.name)
    temporary_path.replace(destination)
    return manifest


def _load_manifest(dataset_directory: Path) -> MultiHopRAGManifest:
    path = dataset_directory / MULTIHOP_RAG_MANIFEST
    if not path.is_file():
        raise ValueError(f"dataset manifest is missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("files_sha256"), dict):
        raise ValueError("MultiHop-RAG manifest has an invalid schema")
    manifest = MultiHopRAGManifest(
        schema_version=int(raw.get("schema_version", 0)),
        dataset=str(raw.get("dataset", "")),
        revision=str(raw.get("revision", "")),
        source_url=str(raw.get("source_url", "")),
        license=str(raw.get("license", "")),
        citation=str(raw.get("citation", "")),
        files_sha256={str(key): str(value) for key, value in raw["files_sha256"].items()},
    )
    if (
        manifest.schema_version != 1
        or manifest.dataset != MULTIHOP_RAG_NAME
        or manifest.revision != MULTIHOP_RAG_REVISION
        or manifest.files_sha256 != MULTIHOP_RAG_FILES
    ):
        raise ValueError("MultiHop-RAG manifest does not match the pinned dataset contract")
    for filename, expected_sha256 in manifest.files_sha256.items():
        file_path = dataset_directory / filename
        if not file_path.is_file() or _sha256_file(file_path) != expected_sha256:
            raise ValueError(f"MultiHop-RAG file failed integrity verification: {filename}")
    return manifest


def fetch_multihop_rag(cache_directory: Path) -> Path:
    """Download the two pinned JSON files atomically and verify exact SHA-256 hashes."""

    cache_directory.mkdir(parents=True, exist_ok=True)
    dataset_directory = cache_directory / "multihop-rag"
    if dataset_directory.is_dir() and (dataset_directory / MULTIHOP_RAG_MANIFEST).is_file():
        _load_manifest(dataset_directory)
        return dataset_directory
    if dataset_directory.exists():
        raise ValueError(
            f"incomplete dataset directory exists at {dataset_directory}; move it aside and retry"
        )
    with tempfile.TemporaryDirectory(prefix="multihop-rag-", dir=cache_directory) as temporary:
        staging = Path(temporary)
        for filename, expected_sha256 in MULTIHOP_RAG_FILES.items():
            partial = staging / f".{filename}.part"
            _download(_source_url(filename), partial)
            actual_sha256 = _sha256_file(partial)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"MultiHop-RAG SHA-256 mismatch for {filename}: expected "
                    f"{expected_sha256}, got {actual_sha256}"
                )
            partial.replace(staging / filename)
        _write_manifest(staging)
        shutil.move(str(staging), dataset_directory)
    _load_manifest(dataset_directory)
    return dataset_directory


def _read_json_array(path: Path) -> list[Mapping[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        raise ValueError(f"{path} must contain an array of JSON objects")
    return cast(list[Mapping[str, Any]], raw)


def _required_text(row: Mapping[str, Any], key: str, *, label: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _body_units(body: str, *, encoding: Any) -> list[str]:
    units: list[str] = []
    for raw_paragraph in re.split(r"\n\s*\n+", body):
        paragraph = " ".join(raw_paragraph.split())
        if not paragraph:
            continue
        if len(encoding.encode_ordinary(paragraph)) <= MULTIHOP_RAG_CHUNK_SIZE_TOKENS:
            units.append(paragraph)
        else:
            units.extend(
                sentence.strip()
                for sentence in _SENTENCE_BOUNDARY.split(paragraph)
                if sentence.strip()
            )
    return units


def _body_chunks(body: str, *, encoding: Any) -> tuple[str, ...]:
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in _body_units(body, encoding=encoding):
        token_ids = encoding.encode_ordinary(unit)
        if len(token_ids) > MULTIHOP_RAG_CHUNK_SIZE_TOKENS:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            for start in range(0, len(token_ids), MULTIHOP_RAG_CHUNK_SIZE_TOKENS):
                text = encoding.decode(
                    token_ids[start : start + MULTIHOP_RAG_CHUNK_SIZE_TOKENS]
                ).strip()
                if text:
                    chunks.append(text)
            continue
        if current and current_tokens + len(token_ids) > MULTIHOP_RAG_CHUNK_SIZE_TOKENS:
            chunks.append(" ".join(current))
            current = [unit]
            current_tokens = len(token_ids)
        else:
            current.append(unit)
            current_tokens += len(token_ids)
    if current:
        chunks.append(" ".join(current))
    return tuple(chunks)


def _document_id(url: str) -> str:
    return f"mhr-doc-{hashlib.sha256(url.encode()).hexdigest()[:16]}"


def _query_id(query: str) -> str:
    return f"mhr-q-{hashlib.sha256(query.encode()).hexdigest()[:16]}"


def _materialize_chunks(
    corpus_rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Chunk, ...], dict[str, tuple[tuple[str, str], ...]]]:
    encoding = tiktoken.get_encoding(TOKEN_ENCODING)
    chunks: list[Chunk] = []
    text_by_url: dict[str, tuple[tuple[str, str], ...]] = {}
    for index, row in enumerate(corpus_rows):
        label = f"corpus[{index}]"
        url = _required_text(row, "url", label=label)
        title = _required_text(row, "title", label=label)
        source = _required_text(row, "source", label=label)
        published_at = _required_text(row, "published_at", label=label)
        category = _required_text(row, "category", label=label)
        body = _required_text(row, "body", label=label)
        if url in text_by_url:
            raise ValueError(f"duplicate MultiHop-RAG corpus URL {url!r}")
        document_id = _document_id(url)
        document_chunks: list[tuple[str, str]] = []
        prefix = (
            f"Title: {title}\nPublisher: {source}\nPublished: {published_at}\n"
            f"Category: {category}\n\n"
        )
        for chunk_index, body_chunk in enumerate(_body_chunks(body, encoding=encoding)):
            identity = "\0".join(
                (
                    MULTIHOP_RAG_REVISION,
                    url,
                    str(chunk_index),
                    str(MULTIHOP_RAG_CHUNK_SIZE_TOKENS),
                    body_chunk,
                )
            )
            chunk_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
            metadata: dict[str, MetadataValue] = {
                "benchmark": MULTIHOP_RAG_NAME,
                "document_id": document_id,
                "title": title,
                "publisher": source,
                "published_at": published_at,
                "category": category,
                "url": url,
                "chunk_index": chunk_index,
            }
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=f"{prefix}{body_chunk}",
                    source=document_id,
                    metadata=metadata,
                )
            )
            document_chunks.append((chunk_id, body_chunk))
        if not document_chunks:
            raise ValueError(f"MultiHop-RAG document {url!r} produced no chunks")
        text_by_url[url] = tuple(document_chunks)
    chunk_ids = [chunk.id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("MultiHop-RAG deterministic chunk ID collision detected")
    return tuple(chunks), text_by_url


def _parse_cases(
    query_rows: Sequence[Mapping[str, Any]],
    chunks_by_url: Mapping[str, Sequence[tuple[str, str]]],
) -> tuple[MultiHopRAGCase, ...]:
    cases: list[MultiHopRAGCase] = []
    for index, row in enumerate(query_rows):
        label = f"queries[{index}]"
        query = _required_text(row, "query", label=label)
        answer = _required_text(row, "answer", label=label)
        question_type = _required_text(row, "question_type", label=label)
        if question_type not in MULTIHOP_RAG_QUESTION_TYPES:
            raise ValueError(f"{label} has unsupported question_type {question_type!r}")
        evidence_raw = row.get("evidence_list")
        if not isinstance(evidence_raw, list) or any(
            not isinstance(item, dict) for item in evidence_raw
        ):
            raise ValueError(f"{label}.evidence_list must be an array of objects")
        if question_type == "null_query" and evidence_raw:
            raise ValueError(f"{label} null_query must not contain evidence")
        if question_type != "null_query" and not 2 <= len(evidence_raw) <= 4:
            raise ValueError(f"{label} answerable query must contain two to four evidence items")
        evidence_chunk_ids: list[str] = []
        evidence_urls: list[str] = []
        for evidence_index, evidence_object in enumerate(evidence_raw):
            evidence = cast(Mapping[str, Any], evidence_object)
            evidence_label = f"{label}.evidence_list[{evidence_index}]"
            url = _required_text(evidence, "url", label=evidence_label)
            fact = _required_text(evidence, "fact", label=evidence_label)
            document_chunks = chunks_by_url.get(url)
            if document_chunks is None:
                raise ValueError(f"{evidence_label} references an unknown corpus URL")
            normalized_fact = _normalize(fact)
            matching = [
                chunk_id
                for chunk_id, body_chunk in document_chunks
                if normalized_fact in _normalize(body_chunk)
            ]
            if not matching:
                raise ValueError(f"{evidence_label} fact is absent from its source document")
            evidence_chunk_ids.append(matching[0])
            evidence_urls.append(url)
        cases.append(
            MultiHopRAGCase(
                query_id=_query_id(query),
                query=query,
                answer=answer,
                question_type=question_type,
                evidence_count=len(evidence_raw),
                evidence_chunk_ids=tuple(evidence_chunk_ids),
                evidence_urls=tuple(evidence_urls),
            )
        )
    query_ids = [case.query_id for case in cases]
    queries = [case.query for case in cases]
    if len(query_ids) != len(set(query_ids)) or len(queries) != len(set(queries)):
        raise ValueError("MultiHop-RAG query IDs and query text must be unique")
    return tuple(cases)


def _holdout_split(sample_per_stratum: int, stratum_offset: int) -> str:
    split = f"heldout-{sample_per_stratum}-per-stratum"
    return split if stratum_offset == 0 else f"{split}-offset-{stratum_offset}"


def _holdout_cases(
    cases: Sequence[MultiHopRAGCase],
    sample_per_stratum: int,
    stratum_offset: int = 0,
) -> tuple[MultiHopRAGCase, ...]:
    if sample_per_stratum <= 0:
        raise ValueError("sample_per_stratum must be positive")
    if stratum_offset < 0:
        raise ValueError("stratum_offset must be non-negative")
    strata: dict[str, list[MultiHopRAGCase]] = {}
    for case in cases:
        if case.evidence_count:
            strata.setdefault(case.stratum, []).append(case)
    selected: list[MultiHopRAGCase] = []
    for stratum in sorted(strata):
        members = strata[stratum]
        required_members = stratum_offset + sample_per_stratum
        if len(members) < required_members:
            raise ValueError(
                f"MultiHop-RAG stratum {stratum!r} has only {len(members)} cases; "
                f"selection requires {required_members}"
            )
        members.sort(
            key=lambda case: hashlib.sha256(
                f"{MULTIHOP_RAG_REVISION}\0{case.query}".encode()
            ).hexdigest()
        )
        selected.extend(members[stratum_offset:required_members])
    selected.sort(key=lambda case: case.query_id)
    return tuple(selected)


def multihop_rag_question_contract_sha256(cases: Sequence[MultiHopRAGCase]) -> str:
    """Hash only selected IDs and questions, excluding answers and evidence."""

    return _stable_sha256([{"query_id": case.query_id, "query": case.query} for case in cases])


def load_multihop_rag(
    dataset_directory: Path,
    *,
    sample_per_stratum: int | None = None,
    stratum_offset: int = 0,
) -> MultiHopRAGBundle:
    """Load the verified benchmark, optionally using a frozen stratified answerable slice."""

    manifest = _load_manifest(dataset_directory)
    corpus_rows = _read_json_array(dataset_directory / "corpus.json")
    query_rows = _read_json_array(dataset_directory / "MultiHopRAG.json")
    if len(corpus_rows) != MULTIHOP_RAG_DOCUMENT_COUNT:
        raise ValueError(
            f"expected {MULTIHOP_RAG_DOCUMENT_COUNT} documents, found {len(corpus_rows)}"
        )
    if len(query_rows) != MULTIHOP_RAG_QUERY_COUNT:
        raise ValueError(f"expected {MULTIHOP_RAG_QUERY_COUNT} queries, found {len(query_rows)}")
    chunks, chunks_by_url = _materialize_chunks(corpus_rows)
    all_cases = _parse_cases(query_rows, chunks_by_url)
    if sample_per_stratum is None and stratum_offset:
        raise ValueError("stratum_offset requires sample_per_stratum")
    selected_cases = (
        tuple(all_cases)
        if sample_per_stratum is None
        else _holdout_cases(all_cases, sample_per_stratum, stratum_offset)
    )
    selection_contract = {
        "method": (
            "all-queries-in-source-order"
            if sample_per_stratum is None
            else MULTIHOP_RAG_HOLDOUT_METHOD
        ),
        "revision": MULTIHOP_RAG_REVISION,
        "sample_per_stratum": sample_per_stratum,
        "query_ids": [case.query_id for case in selected_cases],
        "strata": dict(sorted(Counter(case.stratum for case in selected_cases).items())),
    }
    if stratum_offset:
        selection_contract["stratum_offset"] = stratum_offset
    selection_sha256 = _stable_sha256(selection_contract)
    judgments = tuple(
        RetrievalJudgment(
            query_id=case.query_id,
            query=case.query,
            relevance=dict.fromkeys(case.evidence_chunk_ids, 1.0),
        )
        for case in selected_cases
    )
    dataset = RetrievalBenchmarkDataset(
        name=MULTIHOP_RAG_NAME,
        split=(
            "full"
            if sample_per_stratum is None
            else _holdout_split(sample_per_stratum, stratum_offset)
        ),
        chunks=chunks,
        judgments=judgments,
        total_query_count=len(all_cases),
        provenance=BenchmarkProvenance(
            source_url=manifest.source_url,
            archive_md5=None,
            license=manifest.license,
            citation=manifest.citation,
            details={
                "schema_version": manifest.schema_version,
                "revision": manifest.revision,
                "files_sha256": manifest.files_sha256,
                "document_count": len(corpus_rows),
                "chunk_count": len(chunks),
                "chunking": {
                    "method": "paragraph-and-sentence-pack-v1",
                    "encoding": TOKEN_ENCODING,
                    "body_chunk_size_tokens": MULTIHOP_RAG_CHUNK_SIZE_TOKENS,
                    "metadata_prefix_repeated": True,
                    "canonical_evidence_chunk": (
                        "first deterministic chunk containing each normalized evidence fact"
                    ),
                },
                "selection": selection_contract,
                "selection_sha256": selection_sha256,
                "question_contract_sha256": multihop_rag_question_contract_sha256(selected_cases),
            },
        ),
    )
    return MultiHopRAGBundle(dataset, selected_cases, selection_sha256)


def write_multihop_rag_holdout_manifest(
    bundle: MultiHopRAGBundle,
    output_path: Path,
    *,
    sample_per_stratum: int,
    stratum_offset: int = 0,
    frozen_at: str,
) -> Mapping[str, Any]:
    """Commit the pre-retrieval selection contract without gold answers or evidence text."""

    if sample_per_stratum <= 0:
        raise ValueError("sample_per_stratum must be positive")
    if stratum_offset < 0:
        raise ValueError("stratum_offset must be non-negative")
    expected_split = _holdout_split(sample_per_stratum, stratum_offset)
    if bundle.dataset.split != expected_split:
        raise ValueError(f"bundle split must be {expected_split!r}")
    if not frozen_at.strip():
        raise ValueError("frozen_at must be non-empty")
    strata = dict(sorted(Counter(case.stratum for case in bundle.cases).items()))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "frozen_before_retrieval",
        "dataset": MULTIHOP_RAG_NAME,
        "revision": MULTIHOP_RAG_REVISION,
        "split": bundle.dataset.split,
        "selection_method": MULTIHOP_RAG_HOLDOUT_METHOD,
        "sample_per_stratum": sample_per_stratum,
        "selected_query_count": len(bundle.cases),
        "strata": strata,
        "selection_sha256": bundle.selection_sha256,
        "question_contract_sha256": multihop_rag_question_contract_sha256(bundle.cases),
        "frozen_at": frozen_at.strip(),
        "records": [
            {
                "query_id": case.query_id,
                "question_sha256": hashlib.sha256(case.query.encode()).hexdigest(),
                "question_type": case.question_type,
                "evidence_count": case.evidence_count,
            }
            for case in bundle.cases
        ],
        "leakage_controls": {
            "selection_inputs": "question type, evidence count, and SHA-256 rank only",
            "planner_inputs": "query text only",
            "excluded_from_planner": [
                "answer",
                "evidence facts",
                "evidence URLs",
                "corpus passages",
                "retrieval judgments",
            ],
        },
        "limitations": [
            "The public questions were generated and quality-controlled by the dataset authors; "
            "this project did not independently human-review all 49 answers.",
            "The slice is a reproducible external holdout, not a random population sample.",
        ],
    }
    if stratum_offset:
        manifest["stratum_offset"] = stratum_offset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary_path = Path(handle.name)
    temporary_path.replace(output_path)
    return manifest

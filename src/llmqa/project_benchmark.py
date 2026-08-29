"""Verified adapter from project-evaluation artifacts to the retrieval benchmark core."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from llmqa.benchmark import (
    BenchmarkProvenance,
    RetrievalBenchmarkDataset,
)
from llmqa.domain import Chunk, MetadataValue
from llmqa.evaluation import RetrievalJudgment
from llmqa.project_evaluation import (
    ProjectChunkManifest,
    ProjectEvaluationCase,
    load_injection_fixtures,
    load_project_chunk_manifest,
    load_project_eval_manifest,
    load_project_evaluation_cases,
    validate_project_evaluation,
)

PROJECT_BENCHMARK_SPLIT = "reviewed-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(cast(Mapping[str, Any], raw))
    if not rows:
        raise ValueError(f"{path} must contain at least one JSON object")
    return rows


def _verified_chunks(path: Path, chunk_manifest: ProjectChunkManifest) -> tuple[Chunk, ...]:
    if _sha256(path) != chunk_manifest.raw_chunks_sha256:
        raise ValueError("raw chunk artifact failed SHA-256 verification")
    rows = _read_jsonl(path)
    if len(rows) != len(chunk_manifest.chunks):
        raise ValueError("raw chunk count does not match the chunk manifest")

    chunks: list[Chunk] = []
    for row, reference in zip(rows, chunk_manifest.chunks, strict=True):
        chunk_id = row.get("id")
        text = row.get("text")
        source = row.get("source")
        page = row.get("page")
        metadata = row.get("metadata")
        if (
            not isinstance(chunk_id, str)
            or not isinstance(text, str)
            or not text.strip()
            or not isinstance(source, str)
            or not isinstance(page, int)
            or isinstance(page, bool)
            or not isinstance(metadata, dict)
        ):
            raise ValueError(f"raw chunk {reference.chunk_id!r} has an invalid schema")
        if any(
            value is not None and not isinstance(value, str | int | float | bool)
            for value in metadata.values()
        ):
            raise ValueError(f"raw chunk {reference.chunk_id!r} has invalid metadata")
        if (
            chunk_id != reference.chunk_id
            or source != reference.source_id
            or page != reference.page
            or metadata.get("token_start") != reference.token_start
            or metadata.get("token_count") != reference.token_count
            or hashlib.sha256(text.encode("utf-8")).hexdigest() != reference.text_sha256
        ):
            raise ValueError(f"raw chunk {reference.chunk_id!r} does not match its manifest entry")
        chunks.append(
            Chunk(
                id=chunk_id,
                text=text,
                source=source,
                page=page,
                metadata=cast(dict[str, MetadataValue], dict(metadata)),
            )
        )
    return tuple(chunks)


def _expected_relevance(case: ProjectEvaluationCase, grade: int) -> dict[str, float]:
    return dict.fromkeys(
        (chunk_id for locator in case.evidence for chunk_id in locator.chunk_ids),
        float(grade),
    )


def _verified_judgments(
    path: Path,
    chunk_manifest: ProjectChunkManifest,
    cases: tuple[ProjectEvaluationCase, ...],
    relevance_grade: int,
) -> tuple[RetrievalJudgment, ...]:
    if _sha256(path) != chunk_manifest.retrieval_judgments_sha256:
        raise ValueError("retrieval judgments failed SHA-256 verification")
    rows = _read_jsonl(path)
    if len(rows) != len(cases):
        raise ValueError("retrieval judgment count does not match the reviewed cases")
    chunk_ids = {chunk.chunk_id for chunk in chunk_manifest.chunks}
    judgments: list[RetrievalJudgment] = []
    for row, case in zip(rows, cases, strict=True):
        relevance = row.get("relevance")
        if not isinstance(relevance, dict):
            raise ValueError(f"judgment {case.case_id!r} relevance must be a JSON object")
        parsed_relevance = {str(chunk_id): float(grade) for chunk_id, grade in relevance.items()}
        if (
            row.get("query_id") != case.case_id
            or row.get("query") != case.question
            or row.get("answerability") != case.answerability
            or row.get("case_types") != list(case.case_types)
            or parsed_relevance != _expected_relevance(case, relevance_grade)
        ):
            raise ValueError(f"judgment {case.case_id!r} does not match its reviewed case")
        unknown_chunk_ids = set(parsed_relevance) - chunk_ids
        if unknown_chunk_ids:
            raise ValueError(
                f"judgment {case.case_id!r} references unknown chunks: {sorted(unknown_chunk_ids)}"
            )
        judgments.append(
            RetrievalJudgment(
                query_id=case.case_id,
                query=case.question,
                relevance=parsed_relevance,
            )
        )
    return tuple(judgments)


def load_project_retrieval_benchmark(
    evaluation_directory: Path,
    raw_chunks_path: Path,
) -> RetrievalBenchmarkDataset:
    """Load only a fully reviewed, checksum-consistent project benchmark."""

    manifest_path = evaluation_directory / "manifest.json"
    cases_path = evaluation_directory / "cases.jsonl"
    fixtures_path = evaluation_directory / "injection-fixtures.jsonl"
    chunk_manifest_path = evaluation_directory / "chunk-manifest.json"
    judgments_path = evaluation_directory / "retrieval-judgments.jsonl"
    manifest = load_project_eval_manifest(manifest_path)
    cases = load_project_evaluation_cases(cases_path)
    fixtures = load_injection_fixtures(fixtures_path)
    chunk_manifest = load_project_chunk_manifest(chunk_manifest_path)
    readiness = validate_project_evaluation(cases, fixtures, manifest, chunk_manifest)
    if not readiness.ready_for_benchmark:
        raise ValueError("project evaluation is not ready for benchmark execution")
    if _sha256(cases_path) != chunk_manifest.cases_sha256:
        raise ValueError("reviewed cases failed SHA-256 verification")

    evidence_materialization = manifest["evidence_materialization"]
    assert isinstance(evidence_materialization, Mapping)
    relevance_grade = evidence_materialization["relevance_grade"]
    assert isinstance(relevance_grade, int)
    chunks = _verified_chunks(raw_chunks_path, chunk_manifest)
    judgments = _verified_judgments(
        judgments_path,
        chunk_manifest,
        cases,
        relevance_grade,
    )

    raw_sources = manifest["sources"]
    assert isinstance(raw_sources, list)
    sources = [cast(Mapping[str, Any], source) for source in raw_sources]
    source_details = [
        {
            key: source[key]
            for key in (
                "source_id",
                "title",
                "version",
                "landing_page",
                "download_url",
                "sha256",
                "pages",
                "license",
            )
        }
        for source in sources
    ]
    license_summary = "; ".join(f"{source['source_id']}: {source['license']}" for source in sources)
    citation = "; ".join(
        f"{source['title']} ({source['version']}), {source['landing_page']}" for source in sources
    )
    return RetrievalBenchmarkDataset(
        name=str(manifest["dataset"]),
        split=PROJECT_BENCHMARK_SPLIT,
        chunks=chunks,
        judgments=judgments,
        total_query_count=len(cases),
        provenance=BenchmarkProvenance(
            source_url=None,
            archive_md5=None,
            license=license_summary,
            citation=citation,
            details={
                "project_manifest_sha256": _sha256(manifest_path),
                "chunk_manifest_sha256": _sha256(chunk_manifest_path),
                "cases_sha256": chunk_manifest.cases_sha256,
                "retrieval_judgments_sha256": chunk_manifest.retrieval_judgments_sha256,
                "raw_chunks_sha256": chunk_manifest.raw_chunks_sha256,
                "evidence_strategy": chunk_manifest.evidence_strategy,
                "sources": source_details,
            },
        ),
    )

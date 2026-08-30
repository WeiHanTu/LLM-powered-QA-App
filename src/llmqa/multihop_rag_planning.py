"""Question-only OpenAI decomposition for the frozen MultiHop-RAG holdout."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from openai import OpenAI

from llmqa.generation import ResponsesClient
from llmqa.multihop_rag import (
    MULTIHOP_RAG_HOLDOUT_METHOD,
    MULTIHOP_RAG_NAME,
    MultiHopRAGCase,
    load_multihop_rag,
    multihop_rag_question_contract_sha256,
)
from llmqa.query_decomposition import (
    QUERY_DECOMPOSITION_PROMPT_VERSION,
    QUERY_DECOMPOSITION_VERSION,
    QueryDecomposition,
    decompose_query_text,
    query_decomposition_prompt_sha256,
)

MULTIHOP_RAG_DECOMPOSITION_VERSION = "multihop-rag-question-only-openai-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _record_from_raw(raw: Mapping[str, Any]) -> QueryDecomposition:
    subqueries = raw.get("subqueries")
    if not isinstance(subqueries, list):
        raise ValueError("MultiHop-RAG decomposition record requires a subqueries array")
    return QueryDecomposition(
        case_id=str(raw.get("case_id", "")),
        question_sha256=str(raw.get("question_sha256", "")),
        subqueries=tuple(str(value) for value in subqueries),
        response_id=str(raw["response_id"]) if isinstance(raw.get("response_id"), str) else None,
        response_model=(
            str(raw["response_model"]) if isinstance(raw.get("response_model"), str) else None
        ),
        input_tokens=(
            int(raw["input_tokens"]) if isinstance(raw.get("input_tokens"), int) else None
        ),
        output_tokens=(
            int(raw["output_tokens"]) if isinstance(raw.get("output_tokens"), int) else None
        ),
        total_tokens=(
            int(raw["total_tokens"]) if isinstance(raw.get("total_tokens"), int) else None
        ),
    )


def _validate_records(
    records: Sequence[QueryDecomposition],
    cases: Sequence[MultiHopRAGCase],
    *,
    complete: bool,
) -> dict[str, tuple[str, ...]]:
    case_by_id = {case.query_id: case for case in cases}
    by_id = {record.case_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("MultiHop-RAG decomposition records contain duplicate IDs")
    unknown = set(by_id) - set(case_by_id)
    if unknown:
        raise ValueError(f"decomposition records contain unknown query IDs: {sorted(unknown)}")
    if complete and set(by_id) != set(case_by_id):
        raise ValueError("decomposition records do not cover the complete frozen holdout")
    for query_id, record in by_id.items():
        case = case_by_id[query_id]
        expected_sha256 = hashlib.sha256(case.query.encode()).hexdigest()
        if record.question_sha256 != expected_sha256:
            raise ValueError(f"decomposition {query_id!r} does not match its question")
        if (
            not 2 <= len(record.subqueries) <= 4
            or any(not subquery.strip() for subquery in record.subqueries)
            or len(set(record.subqueries)) != len(record.subqueries)
            or case.query.strip() in record.subqueries
        ):
            raise ValueError(f"decomposition {query_id!r} has invalid subqueries")
    return {query_id: by_id[query_id].subqueries for query_id in sorted(by_id)}


def _contract(
    cases: Sequence[MultiHopRAGCase],
    *,
    selection_sha256: str,
    sample_per_stratum: int,
    requested_model: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_version": MULTIHOP_RAG_DECOMPOSITION_VERSION,
        "dataset": MULTIHOP_RAG_NAME,
        "split": f"heldout-{sample_per_stratum}-per-stratum",
        "selection_method": MULTIHOP_RAG_HOLDOUT_METHOD,
        "selection_sha256": selection_sha256,
        "question_contract_sha256": multihop_rag_question_contract_sha256(cases),
        "method": QUERY_DECOMPOSITION_VERSION,
        "prompt_version": QUERY_DECOMPOSITION_PROMPT_VERSION,
        "prompt_sha256": query_decomposition_prompt_sha256(),
        "requested_model": requested_model,
        "question_only_input": True,
        "query_count": len(cases),
    }


def generate_multihop_rag_decompositions(
    dataset_directory: Path,
    output_path: Path,
    *,
    model: str,
    sample_per_stratum: int = 7,
    client: ResponsesClient | None = None,
    generated_at: str | None = None,
) -> Mapping[str, Any]:
    """Generate or resume decompositions; persist every completed API result atomically."""

    bundle = load_multihop_rag(
        dataset_directory,
        sample_per_stratum=sample_per_stratum,
    )
    contract = _contract(
        bundle.cases,
        selection_sha256=bundle.selection_sha256,
        sample_per_stratum=sample_per_stratum,
        requested_model=model,
    )
    records: dict[str, QueryDecomposition] = {}
    started_at = generated_at or datetime.now(UTC).isoformat()
    if output_path.exists():
        raw = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("MultiHop-RAG decomposition artifact must be a JSON object")
        existing_contract = {key: raw.get(key) for key in contract}
        if existing_contract != contract:
            raise ValueError("existing decomposition artifact belongs to a different run")
        raw_records = raw.get("records")
        if not isinstance(raw_records, list) or any(
            not isinstance(record, dict) for record in raw_records
        ):
            raise ValueError("decomposition artifact records must be an array of objects")
        parsed = [_record_from_raw(cast(Mapping[str, Any], record)) for record in raw_records]
        _validate_records(parsed, bundle.cases, complete=raw.get("status") == "complete")
        records = {record.case_id: record for record in parsed}
        started_at = str(raw.get("started_at", started_at))
        if raw.get("status") == "complete":
            return raw

    openai_client = client or cast(ResponsesClient, OpenAI())

    def checkpoint(status: str) -> dict[str, object]:
        artifact: dict[str, object] = {
            **contract,
            "status": status,
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat() if status == "complete" else None,
            "records": [asdict(records[query_id]) for query_id in sorted(records)],
        }
        payload = _json_bytes(artifact)
        _atomic_write(output_path, payload)
        return cast(dict[str, object], json.loads(payload))

    checkpoint("in_progress")
    for case in bundle.cases:
        if case.query_id in records:
            continue
        records[case.query_id] = decompose_query_text(
            case.query_id,
            case.query,
            model=model,
            client=openai_client,
        )
        checkpoint("in_progress")
    _validate_records(tuple(records.values()), bundle.cases, complete=True)
    return checkpoint("complete")


def load_multihop_rag_decompositions(
    path: Path,
    dataset_directory: Path,
    *,
    sample_per_stratum: int = 7,
) -> tuple[Mapping[str, Any], dict[str, tuple[str, ...]]]:
    """Load only a complete artifact matching the current pinned data and selection."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("status") != "complete":
        raise ValueError("MultiHop-RAG decomposition artifact is not complete")
    bundle = load_multihop_rag(
        dataset_directory,
        sample_per_stratum=sample_per_stratum,
    )
    requested_model = raw.get("requested_model")
    if not isinstance(requested_model, str) or not requested_model:
        raise ValueError("decomposition artifact must declare its requested model")
    expected_contract = _contract(
        bundle.cases,
        selection_sha256=bundle.selection_sha256,
        sample_per_stratum=sample_per_stratum,
        requested_model=requested_model,
    )
    if {key: raw.get(key) for key in expected_contract} != expected_contract:
        raise ValueError("MultiHop-RAG decomposition provenance does not match the holdout")
    raw_records = raw.get("records")
    if not isinstance(raw_records, list) or any(
        not isinstance(record, dict) for record in raw_records
    ):
        raise ValueError("decomposition artifact records must be an array of objects")
    records = tuple(_record_from_raw(cast(Mapping[str, Any], record)) for record in raw_records)
    mapping = _validate_records(records, bundle.cases, complete=True)
    return raw, mapping


def multihop_rag_decomposition_sha256(path: Path) -> str:
    return _sha256_file(path)

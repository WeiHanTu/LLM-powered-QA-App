"""Measure the ceiling on parent-document expansion before building the candidate.

Parent-document expansion can only recover a missed evidence chunk when that chunk sits
near an already-retrieved chunk of the same document. This module computes that ceiling
directly from a baseline run, so a candidate is only built when it can possibly win.

The reported ceiling is optimistic by construction: it assumes expansion is free and that
every reachable gold chunk is retained. Expansion also enlarges the slate, so the ceiling
alone is not a decision. Each window therefore carries a budget-matched control: complete
coverage of the plain ranking truncated to the same slate size expansion would occupy. A
window whose ceiling loses to that control cannot win at equal cost, however it is built.

The control needs ranks below ``k``. Supply a run at least as deep as the widest slate;
``truncated_budget_queries`` reports how often the run was too short, which understates
the control and must be read as a limitation of the record.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llmqa.domain import Chunk
from llmqa.multihop_rag import MultiHopRAGCase

DIAGNOSTIC_VERSION = "parent-expansion-ceiling-v2"


@dataclass(frozen=True, slots=True)
class WindowCeiling:
    """The best possible outcome of expanding by ``window`` chunks either side."""

    window: int
    ceiling_full_coverage: int
    max_gain: int
    recoverable_evidence_facts: int
    mean_injected_chunks: float
    mean_slate_size: float
    budget_matched_full_coverage: int
    budget_matched_margin: int
    truncated_budget_queries: int
    dominated_by_budget_matched_baseline: bool
    recovered_query_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExpansionDiagnostic:
    """Whether parent-document expansion is worth preregistering at all."""

    diagnostic_version: str
    query_count: int
    excluded_zero_evidence_queries: int
    k: int
    baseline_full_coverage: int
    missed_evidence_facts: int
    total_evidence_facts: int
    queries_with_multiple_gold_chunks_in_one_document: int
    multi_gold_same_document_query_ids: tuple[str, ...]
    windows: tuple[WindowCeiling, ...]

    def as_record(self) -> dict[str, Any]:
        """Return a deterministic, JSON-serializable record."""

        return asdict(self)


def _chunk_positions(chunks: Sequence[Chunk]) -> dict[str, tuple[str, int]]:
    positions: dict[str, tuple[str, int]] = {}
    for chunk in chunks:
        document_id = chunk.metadata.get("document_id")
        chunk_index = chunk.metadata.get("chunk_index")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"chunk {chunk.id!r} has no document_id")
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool) or chunk_index < 0:
            raise ValueError(f"chunk {chunk.id!r} has no non-negative chunk_index")
        positions[chunk.id] = (document_id, chunk_index)
    if len(positions) != len(chunks):
        raise ValueError("chunk IDs must be unique")
    return positions


def _document_slots(positions: Mapping[str, tuple[str, int]]) -> dict[tuple[str, int], str]:
    slots: dict[tuple[str, int], str] = {}
    for chunk_id, key in positions.items():
        if key in slots:
            raise ValueError(f"duplicate (document, index) slot for chunk {chunk_id!r}")
        slots[key] = chunk_id
    return slots


def _neighbourhood(
    retrieved: Sequence[str],
    positions: Mapping[str, tuple[str, int]],
    slots: Mapping[tuple[str, int], str],
    window: int,
) -> set[str]:
    """Return every chunk ID reachable by expanding each retrieved chunk by ``window``."""

    reachable: set[str] = set()
    for chunk_id in retrieved:
        located = positions.get(chunk_id)
        if located is None:
            continue
        document_id, index = located
        for offset in range(-window, window + 1):
            neighbour = slots.get((document_id, index + offset))
            if neighbour is not None:
                reachable.add(neighbour)
    return reachable


def diagnose_parent_expansion(
    chunks: Sequence[Chunk],
    cases: Sequence[MultiHopRAGCase],
    run: Mapping[str, Sequence[str]],
    *,
    k: int,
    windows: Sequence[int],
) -> ExpansionDiagnostic:
    """Compute the parent-expansion ceiling for a baseline run over ``cases``."""

    if k <= 0:
        raise ValueError("k must be positive")
    if not windows or any(window <= 0 for window in windows):
        raise ValueError("windows must be a non-empty sequence of positive integers")
    if not cases:
        raise ValueError("cannot diagnose an empty case set")

    # Zero-evidence (unanswerable) queries have nothing to recover and would otherwise be
    # counted as trivially complete, inflating both the baseline and the ceiling.
    answerable = [case for case in cases if case.evidence_chunk_ids]
    if not answerable:
        raise ValueError("no answerable cases to diagnose")
    excluded = len(cases) - len(answerable)
    cases = answerable

    positions = _chunk_positions(chunks)
    slots = _document_slots(positions)

    baseline_full = 0
    missed_facts = 0
    total_facts = 0
    multi_gold_ids: list[str] = []
    per_window_recovered: dict[int, list[str]] = {window: [] for window in windows}
    per_window_ceiling: dict[int, int] = dict.fromkeys(windows, 0)
    per_window_recoverable_facts: dict[int, int] = dict.fromkeys(windows, 0)
    per_window_injected: dict[int, int] = dict.fromkeys(windows, 0)
    per_window_slate: dict[int, int] = dict.fromkeys(windows, 0)
    per_window_budget_matched: dict[int, int] = dict.fromkeys(windows, 0)
    per_window_truncated: dict[int, int] = dict.fromkeys(windows, 0)

    for case in cases:
        retrieved_all = run.get(case.query_id)
        if retrieved_all is None:
            raise ValueError(f"baseline run has no entry for query {case.query_id!r}")
        ranking = list(retrieved_all)
        retrieved = ranking[:k]
        retrieved_set = set(retrieved)
        gold = set(case.evidence_chunk_ids)
        total_facts += len(gold)
        missed = gold - retrieved_set
        missed_facts += len(missed)
        is_full = not missed
        if is_full:
            baseline_full += 1

        gold_documents = [positions[chunk_id][0] for chunk_id in gold if chunk_id in positions]
        if len(gold_documents) != len(set(gold_documents)):
            multi_gold_ids.append(case.query_id)

        for window in windows:
            reachable = _neighbourhood(retrieved, positions, slots, window)
            per_window_injected[window] += len(reachable - retrieved_set)
            recoverable = missed & reachable
            per_window_recoverable_facts[window] += len(recoverable)
            if is_full:
                per_window_ceiling[window] += 1
            elif recoverable == missed:
                per_window_ceiling[window] += 1
                per_window_recovered[window].append(case.query_id)

            # Budget-matched control: the plain ranking cut to the slate expansion occupies.
            slate_size = len(retrieved_set | reachable)
            per_window_slate[window] += slate_size
            if slate_size > len(ranking):
                per_window_truncated[window] += 1
            if gold <= set(ranking[:slate_size]):
                per_window_budget_matched[window] += 1

    window_reports = tuple(
        WindowCeiling(
            window=window,
            ceiling_full_coverage=per_window_ceiling[window],
            max_gain=per_window_ceiling[window] - baseline_full,
            recoverable_evidence_facts=per_window_recoverable_facts[window],
            mean_injected_chunks=per_window_injected[window] / len(cases),
            mean_slate_size=per_window_slate[window] / len(cases),
            budget_matched_full_coverage=per_window_budget_matched[window],
            budget_matched_margin=per_window_ceiling[window] - per_window_budget_matched[window],
            truncated_budget_queries=per_window_truncated[window],
            dominated_by_budget_matched_baseline=(
                per_window_ceiling[window] <= per_window_budget_matched[window]
            ),
            recovered_query_ids=tuple(sorted(per_window_recovered[window])),
        )
        for window in sorted(windows)
    )

    return ExpansionDiagnostic(
        diagnostic_version=DIAGNOSTIC_VERSION,
        query_count=len(cases),
        excluded_zero_evidence_queries=excluded,
        k=k,
        baseline_full_coverage=baseline_full,
        missed_evidence_facts=missed_facts,
        total_evidence_facts=total_facts,
        queries_with_multiple_gold_chunks_in_one_document=len(multi_gold_ids),
        multi_gold_same_document_query_ids=tuple(sorted(multi_gold_ids)),
        windows=window_reports,
    )


def load_retrieval_run(path: Path) -> dict[str, tuple[str, ...]]:
    """Load a ``{query_id, retrieved_ids}`` JSONL run file."""

    run: dict[str, tuple[str, ...]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} contains invalid JSON") from error
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        query_id = row.get("query_id")
        retrieved = row.get("retrieved_ids")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"{path}:{line_number} has no query_id")
        if not isinstance(retrieved, list) or any(
            not isinstance(item, str) or not item for item in retrieved
        ):
            raise ValueError(f"{path}:{line_number}.retrieved_ids must be non-empty strings")
        if query_id in run:
            raise ValueError(f"{path} repeats query {query_id!r}")
        run[query_id] = tuple(str(item) for item in retrieved)
    if not run:
        raise ValueError(f"{path} contains no retrieval rows")
    return run


def retrieval_run_sha256(path: Path) -> str:
    """Return the SHA-256 of a retrieval run file, for provenance in the diagnostic record."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(131072), b""):
            digest.update(block)
    return digest.hexdigest()

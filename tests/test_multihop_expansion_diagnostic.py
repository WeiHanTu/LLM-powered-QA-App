from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmqa.domain import Chunk
from llmqa.multihop_expansion_diagnostic import (
    diagnose_parent_expansion,
    load_retrieval_run,
)
from llmqa.multihop_rag import MultiHopRAGCase

ROOT = Path(__file__).parents[1]
PUBLIC_RECORD = ROOT / "docs" / "benchmarks" / "multihop-rag-parent-expansion-2026-08-29.json"


def _chunk(document: str, index: int) -> Chunk:
    return Chunk(
        id=f"{document}-c{index}",
        text=f"{document} chunk {index}",
        source=document,
        metadata={"document_id": document, "chunk_index": index},
    )


def _corpus() -> list[Chunk]:
    return [_chunk(document, index) for document in ("docA", "docB") for index in range(5)]


def _case(query_id: str, gold: tuple[str, ...]) -> MultiHopRAGCase:
    return MultiHopRAGCase(
        query_id=query_id,
        query=f"question {query_id}",
        answer="answer",
        question_type="inference_query",
        evidence_count=len(gold),
        evidence_chunk_ids=gold,
        evidence_urls=tuple(f"https://example.invalid/{item}" for item in gold),
    )


def test_a_query_is_recovered_only_when_every_missed_chunk_is_reachable() -> None:
    cases = [
        # both missed chunks sit one slot from the retrieved chunk -> recoverable
        _case("q-near", ("docA-c1", "docA-c3")),
        # one missed chunk is four slots away -> not recoverable at window 1
        _case("q-far", ("docA-c1", "docB-c4")),
    ]
    run = {"q-near": ["docA-c2"], "q-far": ["docA-c2"]}

    report = diagnose_parent_expansion(_corpus(), cases, run, k=10, windows=[1])

    assert report.baseline_full_coverage == 0
    window = report.windows[0]
    assert window.recovered_query_ids == ("q-near",)
    assert window.ceiling_full_coverage == 1
    assert window.max_gain == 1


def test_ceiling_is_monotone_in_window_and_reports_budget_cost() -> None:
    cases = [_case("q", ("docA-c0", "docA-c4"))]
    run = {"q": ["docA-c2"]}

    report = diagnose_parent_expansion(_corpus(), cases, run, k=10, windows=[3, 1, 2])

    assert [window.window for window in report.windows] == [1, 2, 3]
    ceilings = [window.ceiling_full_coverage for window in report.windows]
    assert ceilings == sorted(ceilings), "widening the window cannot lower the ceiling"
    assert ceilings[-1] == 1, "window 2 reaches both c0 and c4 from c2"
    # window 1 pulls in c1 and c3; window 2 additionally pulls in c0 and c4.
    assert report.windows[0].mean_injected_chunks == pytest.approx(2.0)
    assert report.windows[1].mean_injected_chunks == pytest.approx(4.0)


def test_already_complete_queries_count_toward_the_ceiling_but_not_the_gain() -> None:
    cases = [_case("q", ("docA-c1",))]
    run = {"q": ["docA-c1"]}

    report = diagnose_parent_expansion(_corpus(), cases, run, k=10, windows=[1])

    assert report.baseline_full_coverage == 1
    assert report.windows[0].ceiling_full_coverage == 1
    assert report.windows[0].max_gain == 0
    assert report.windows[0].recovered_query_ids == ()


def test_a_window_reports_a_tie_when_both_methods_cover_the_same_slate() -> None:
    # Expansion reaches c1 and c3 from c2, but so does reading three ranks of the run.
    cases = [_case("q", ("docA-c1", "docA-c3"))]
    run = {"q": ["docA-c2", "docA-c1", "docA-c3", "docB-c0"]}

    window = diagnose_parent_expansion(_corpus(), cases, run, k=1, windows=[1]).windows[0]

    assert window.ceiling_full_coverage == 1
    assert window.mean_slate_size == pytest.approx(3.0)
    assert window.budget_matched_full_coverage == 1
    assert window.budget_matched_margin == 0
    assert window.both_pass == 1
    assert window.expansion_only_wins == 0
    assert window.budget_only_wins == 0
    assert window.aggregate_coverage_not_higher_than_budget_matched_baseline


def test_a_window_is_credited_when_the_plain_ranking_cannot_match_it() -> None:
    cases = [_case("q", ("docA-c1", "docA-c3"))]
    run = {"q": ["docA-c2", "docB-c0", "docB-c1", "docB-c2"]}

    window = diagnose_parent_expansion(_corpus(), cases, run, k=1, windows=[1]).windows[0]

    assert window.ceiling_full_coverage == 1
    assert window.budget_matched_full_coverage == 0
    assert window.budget_matched_margin == 1
    assert window.expansion_only_wins == 1
    assert window.budget_only_wins == 0
    assert not window.aggregate_coverage_not_higher_than_budget_matched_baseline


def test_a_run_shallower_than_the_slate_is_reported_as_truncated() -> None:
    cases = [_case("q", ("docA-c1",))]
    run = {"q": ["docA-c2"]}

    window = diagnose_parent_expansion(_corpus(), cases, run, k=1, windows=[1]).windows[0]

    assert window.truncated_budget_queries == 1, "the control needs ranks the run does not have"


def test_multiple_gold_chunks_in_one_document_are_counted() -> None:
    cases = [_case("q-same", ("docA-c1", "docA-c2")), _case("q-split", ("docA-c1", "docB-c1"))]
    run = {"q-same": ["docA-c1"], "q-split": ["docA-c1"]}

    report = diagnose_parent_expansion(_corpus(), cases, run, k=10, windows=[1])

    assert report.queries_with_multiple_gold_chunks_in_one_document == 1
    assert report.multi_gold_same_document_query_ids == ("q-same",)


def test_k_truncates_the_baseline_run() -> None:
    cases = [_case("q", ("docB-c0",))]
    run = {"q": ["docA-c0", "docA-c1", "docB-c0"]}

    assert (
        diagnose_parent_expansion(_corpus(), cases, run, k=3, windows=[1]).baseline_full_coverage
        == 1
    )
    assert (
        diagnose_parent_expansion(_corpus(), cases, run, k=2, windows=[1]).baseline_full_coverage
        == 0
    )


def test_missing_query_in_the_run_fails_closed() -> None:
    cases = [_case("q", ("docA-c1",))]

    with pytest.raises(ValueError, match="no entry for query"):
        diagnose_parent_expansion(_corpus(), cases, {}, k=10, windows=[1])


def test_run_loader_rejects_repeated_queries(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    row = json.dumps({"query_id": "q", "retrieved_ids": ["a"]})
    path.write_text(f"{row}\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="repeats query"):
        load_retrieval_run(path)


def test_run_loader_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(
        json.dumps({"query_id": "q", "retrieved_ids": ["a", "b"]}) + "\n", encoding="utf-8"
    )

    assert load_retrieval_run(path) == {"q": ("a", "b")}


def test_zero_evidence_queries_are_excluded_not_counted_as_complete() -> None:
    cases = [_case("q-answerable", ("docA-c1",)), _case("q-null", ())]
    run = {"q-answerable": ["docA-c1"], "q-null": ["docA-c0"]}

    report = diagnose_parent_expansion(_corpus(), cases, run, k=10, windows=[1])

    assert report.query_count == 1
    assert report.excluded_zero_evidence_queries == 1
    assert report.baseline_full_coverage == 1


def test_committed_parent_expansion_record_reports_paired_non_dominance() -> None:
    record = json.loads(PUBLIC_RECORD.read_text(encoding="utf-8"))

    assert record["method"]["diagnostic_version"] == "parent-expansion-ceiling-v3"
    assert "fixed unconditional expansion" in record["decision"]["action"]
    assert "query-adaptive hybrid" in record["decision"]["reason"]
    for window in record["windows"]:
        assert window["aggregate_coverage_not_higher_than_budget_matched_baseline"] is True
        assert window["expansion_only_wins"] > 0
        assert (
            window["expansion_only_wins"]
            + window["budget_only_wins"]
            + window["both_pass"]
            + window["neither_pass"]
            == record["method"]["answerable_queries"]
        )

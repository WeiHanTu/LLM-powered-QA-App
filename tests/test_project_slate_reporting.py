from __future__ import annotations

import json
import xml.etree.ElementTree as element_tree
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from llmqa.project_evaluation import ProjectEvaluationCase, load_project_evaluation_cases
from llmqa.project_slate_reporting import (
    BASELINE_ARM,
    CANDIDATE_ARM,
    REVIEW_DESIGN,
    build_slate_size_snapshot,
    render_slate_size_svg,
    slate_review_selection_sha256,
)

ROOT = Path(__file__).parents[1]
EVAL_DIR = ROOT / "evals" / "project" / "technical-papers-v1"
PREREGISTRATION = EVAL_DIR / "slate-size-k10-k40-preregistration.json"


def _fixture() -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    tuple[ProjectEvaluationCase, ...],
    dict[str, Any],
    dict[str, str],
]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    all_cases = load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")
    case_by_id = {case.case_id: case for case in all_cases}
    selected_ids = tuple(preregistration["scope"]["case_ids"])
    cases = tuple(case_by_id[case_id] for case_id in selected_ids)
    fixed = preregistration["fixed_contract"]

    def arm(name: str, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        arm_contract = next(item for item in preregistration["arms"] if item["name"] == name)
        k = int(arm_contract["k"])
        token_multiplier = 1 if name == BASELINE_ARM else 2
        rows: list[dict[str, Any]] = []
        for index, case in enumerate(cases):
            retrieved = [locator.chunk_ids[0] for locator in case.evidence]
            retrieved.extend(
                f"padding-{name}-{index}-{offset}" for offset in range(k - len(retrieved))
            )
            automated_pass = index < (6 if name == BASELINE_ARM else 10)
            rows.append(
                {
                    "run_id": run_id,
                    "case_id": case.case_id,
                    "variant": "clean",
                    "answerability": "answerable",
                    "candidate_model": fixed["candidate_model"],
                    "candidate_input_tokens": 100 * token_multiplier,
                    "candidate_output_tokens": 10,
                    "candidate_latency_ms": 1000.0 + index,
                    "retrieved_chunk_ids": retrieved,
                    "citations_valid": True,
                    "task_pass": automated_pass,
                    "semantic_judgment": {
                        "judge_model": fixed["judge_model"],
                        "input_tokens": 50 * token_multiplier,
                        "output_tokens": 5,
                        "latency_ms": 500.0 + index,
                    },
                }
            )
        input_tokens = len(rows) * 150 * token_multiplier
        output_tokens = len(rows) * 15
        cost = (input_tokens * 0.25 + output_tokens * 2.0) / 1_000_000
        summary = {
            "run_id": run_id,
            "dataset": "technical-papers-v1",
            "complete": True,
            "limited_run": True,
            "configuration": {
                "candidate_model": fixed["candidate_model"],
                "judge_model": fixed["judge_model"],
                "candidate_max_output_tokens": fixed["candidate_max_output_tokens"],
                "judge_max_output_tokens": fixed["judge_max_output_tokens"],
                "candidate_prompt_sha256": fixed["candidate_prompt_sha256"],
                "judge_prompt_sha256": fixed["judge_prompt_sha256"],
                "judge_prompt_version": fixed["judge_prompt_version"],
                "claim_contract_version": fixed["claim_contract_version"],
                "retriever": fixed["retriever"],
                "bm25_k1": fixed["bm25_k1"],
                "bm25_b": fixed["bm25_b"],
                "k": k,
                "selected_case_ids": list(selected_ids),
                "budget_preflight": {
                    "preflight_id": arm_contract["preflight_id"],
                    "within_budget": True,
                    "max_cost_usd": arm_contract["max_cost_usd"],
                    "pricing_contract_sha256": fixed["pricing_contract_sha256"],
                    "input_per_million_usd": 0.25,
                    "output_per_million_usd": 2.0,
                },
            },
            "provenance": {
                "cases_sha256": fixed["cases_sha256"],
                "fixtures_sha256": fixed["fixtures_sha256"],
                "raw_chunks_sha256": fixed["raw_chunks_sha256"],
            },
            "usage": {
                "candidate_input_tokens": len(rows) * 100 * token_multiplier,
                "candidate_output_tokens": len(rows) * 10,
                "candidate_total_tokens": len(rows) * (100 * token_multiplier + 10),
                "judge_input_tokens": len(rows) * 50 * token_multiplier,
                "judge_output_tokens": len(rows) * 5,
                "judge_total_tokens": len(rows) * (50 * token_multiplier + 5),
                "candidate_latency_ms": {"p50": 1007.0, "p95": 1013.3},
                "judge_latency_ms": {"p50": 507.0, "p95": 513.3},
                "estimated_standard_token_cost_usd": cost,
            },
        }
        return summary, rows

    baseline_summary, baseline_rows = arm(BASELINE_ARM, "baseline-run")
    candidate_summary, candidate_rows = arm(CANDIDATE_ARM, "candidate-run")
    hashes = {
        "preregistration_sha256": "prereg-hash",
        "cases_sha256": "cases-file-hash",
        "baseline_summary_sha256": "baseline-summary-hash",
        "baseline_results_sha256": "baseline-results-hash",
        "candidate_summary_sha256": "candidate-summary-hash",
        "candidate_results_sha256": "candidate-results-hash",
    }
    decisions: list[dict[str, Any]] = []
    for arm_name, run_id, pass_count, rows in (
        (BASELINE_ARM, "baseline-run", 6, baseline_rows),
        (CANDIDATE_ARM, "candidate-run", 10, candidate_rows),
    ):
        for index, (case, _row) in enumerate(zip(cases, rows, strict=True)):
            passed = index < pass_count
            decisions.append(
                {
                    "arm": arm_name,
                    "run_id": run_id,
                    "case_id": case.case_id,
                    "decision": "pass" if passed else "fail",
                    "rationale": "Synthetic paired review finding.",
                    "review": {
                        "required_claims_satisfied": list(case.required_claims) if passed else [],
                        "required_claims_missing": [] if passed else list(case.required_claims),
                        "answer_correct": passed,
                        "fully_supported": True,
                        "contradiction_detected": False,
                        "citation_faithful": True,
                        "full_evidence_locator_coverage": True,
                    },
                }
            )
    review = {
        "schema_version": 1,
        "status": "complete_human_adjudication",
        "experiment_id": preregistration["experiment_id"],
        "reviewer_ids": ["reviewer-1"],
        "reviewed_at": "2026-08-30T12:00:00Z",
        "scope": {
            "review_design": REVIEW_DESIGN,
            "baseline_run_id": "baseline-run",
            "candidate_run_id": "candidate-run",
            "record_count": len(decisions),
            "selected_records_sha256": slate_review_selection_sha256(
                [(decision["arm"], decision["case_id"]) for decision in decisions]
            ),
        },
        "provenance": dict(hashes),
        "decisions": decisions,
    }
    return (
        preregistration,
        baseline_summary,
        baseline_rows,
        candidate_summary,
        candidate_rows,
        cases,
        review,
        hashes,
    )


def _build(
    fixture: tuple[
        dict[str, Any],
        dict[str, Any],
        list[dict[str, Any]],
        dict[str, Any],
        list[dict[str, Any]],
        tuple[ProjectEvaluationCase, ...],
        dict[str, Any],
        dict[str, str],
    ],
) -> dict[str, object]:
    (
        preregistration,
        baseline_summary,
        baseline_rows,
        candidate_summary,
        candidate_rows,
        cases,
        review,
        hashes,
    ) = fixture
    return build_slate_size_snapshot(
        preregistration,
        baseline_summary,
        baseline_rows,
        candidate_summary,
        candidate_rows,
        cases,
        review,
        hashes=hashes,
        run_date="2026-08-30",
    )


def test_reviewed_snapshot_reconciles_paired_results_and_cost() -> None:
    snapshot: Any = _build(_fixture())

    arms = snapshot["arms"]
    assert isinstance(arms, list)
    assert arms[0]["human_task_pass"]["successes"] == 6
    assert arms[1]["human_task_pass"]["successes"] == 10
    paired = snapshot["paired_analysis"]
    assert paired["candidate_only_pass"]["count"] == 4
    assert paired["baseline_only_pass"]["count"] == 0
    assert paired["mcnemar_exact_two_sided_p"] == pytest.approx(0.125)
    assert snapshot["advance_gate"]["passed"] is True
    assert snapshot["cost"]["cost_per_additional_human_pass_usd"] > 0


def test_report_rejects_ai_preaudit_without_human_approval() -> None:
    fixture = _fixture()
    review = deepcopy(fixture[-2])
    review["status"] = "awaiting_human_approval"
    review["reviewer_ids"] = []
    review["reviewed_at"] = None
    modified = (*fixture[:-2], review, fixture[-1])

    with pytest.raises(ValueError, match="complete human-approved review"):
        _build(modified)


def test_report_rejects_stale_run_binding() -> None:
    fixture = _fixture()
    review = deepcopy(fixture[-2])
    review["scope"]["candidate_run_id"] = "stale-run"
    modified = (*fixture[:-2], review, fixture[-1])

    with pytest.raises(ValueError, match="candidate run"):
        _build(modified)


def test_report_rejects_incorrect_reviewed_locator_coverage() -> None:
    fixture = _fixture()
    review = deepcopy(fixture[-2])
    review["decisions"][0]["review"]["full_evidence_locator_coverage"] = False
    modified = (*fixture[:-2], review, fixture[-1])

    with pytest.raises(ValueError, match="incorrect locator coverage"):
        _build(modified)


def test_report_rejects_correct_label_with_missing_required_claims() -> None:
    fixture = _fixture()
    review = deepcopy(fixture[-2])
    decision = review["decisions"][6]
    assert decision["decision"] == "fail"
    decision["review"]["answer_correct"] = True
    modified = (*fixture[:-2], review, fixture[-1])

    with pytest.raises(ValueError, match="answer_correct conflicts"):
        _build(modified)


def test_slate_figure_is_valid_accessible_svg() -> None:
    snapshot = _build(_fixture())
    svg = render_slate_size_svg(snapshot)
    root = element_tree.fromstring(svg)

    assert root.tag.endswith("svg")
    assert "McNemar exact p = 0.125" in svg
    assert "Human-reviewed task pass" in svg
    assert "aria-labelledby" in root.attrib

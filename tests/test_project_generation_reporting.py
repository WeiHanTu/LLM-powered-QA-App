from __future__ import annotations

import xml.etree.ElementTree as element_tree
from pathlib import Path

from llmqa.project_evaluation import load_project_evaluation_cases
from llmqa.project_generation_reporting import (
    build_project_generation_snapshot,
    render_project_generation_svg,
)

ROOT = Path(__file__).parents[1]
EVAL_DIR = ROOT / "evals" / "project" / "technical-papers-v1"


def test_generation_snapshot_is_full_and_explicitly_provisional() -> None:
    cases = load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")
    rows: list[dict[str, object]] = []
    for case in cases:
        clean: dict[str, object] = {
            "run_id": "fixture-run",
            "case_id": case.case_id,
            "variant": "clean",
            "answerability": case.answerability,
            "case_types": list(case.case_types),
            "task_pass": True,
            "citations_valid": True,
            "exact_abstention": case.answerability == "unanswerable",
            "semantic_judgment": (
                {
                    "answer_correct": True,
                    "fully_supported": True,
                    "contradiction_detected": False,
                }
                if case.answerability == "answerable"
                else None
            ),
        }
        rows.append(clean)
        if case.injection_fixture_id is not None:
            rows.append(
                {
                    **clean,
                    "variant": "injected",
                    "injection_score": {
                        "answer_or_abstention": True,
                        "no_fabricated_claims": True,
                        "no_forbidden_actions": True,
                        "no_injected_citation": True,
                        "no_canary_leakage": True,
                        "passed": True,
                    },
                }
            )
    summary = {
        "run_id": "fixture-run",
        "dataset": "technical-papers-v1",
        "limited_run": False,
        "complete": True,
        "ready_for_automated_report": True,
        "human_adjudication_status": "pending",
        "configuration": {
            "candidate_model": "candidate-test",
            "judge_model": "judge-test",
            "retriever": "bm25",
            "k": 10,
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
            "candidate_prompt_sha256": "candidate-hash",
            "judge_prompt_sha256": "judge-hash",
            "judge_schema_sha256": "schema-hash",
            "judge_prompt_version": "judge-v1",
            "injection_placement_version": "placement-v1",
        },
        "provenance": {},
        "usage": {},
        "limitations": ["fixture limitation"],
    }

    snapshot = build_project_generation_snapshot(
        summary,
        rows,
        cases,
        run_date="2026-08-29",
        bootstrap_resamples=100,
        bootstrap_seed=7,
    )
    svg = render_project_generation_svg(snapshot)

    assert snapshot["status"] == "automated_baseline_human_adjudication_pending"
    assert snapshot["metrics"]["clean_task"]["rate"] == 1.0  # type: ignore[index]
    assert snapshot["dataset"]["answerable_evidence_cluster_count"] == 45  # type: ignore[index]
    assert len(snapshot["per_case_outcomes"]) == 100
    element_tree.fromstring(svg)

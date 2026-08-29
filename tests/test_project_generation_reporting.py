from __future__ import annotations

import json
import xml.etree.ElementTree as element_tree
from pathlib import Path

from llmqa.project_evaluation import load_injection_fixtures, load_project_evaluation_cases
from llmqa.project_generation_reporting import (
    build_project_generation_snapshot,
    render_project_generation_svg,
)

ROOT = Path(__file__).parents[1]
EVAL_DIR = ROOT / "evals" / "project" / "technical-papers-v1"
PUBLIC_SNAPSHOT = (
    ROOT / "docs" / "benchmarks" / "technical-papers-v1-generation-automated-2026-08-29.json"
)


def test_generation_snapshot_is_full_and_explicitly_provisional() -> None:
    cases = load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")
    fixtures = load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl")
    rows: list[dict[str, object]] = []
    for case in cases:
        retrieved_chunk_ids = [
            chunk_id for locator in case.evidence for chunk_id in locator.chunk_ids
        ]
        if case.case_id == "tp-061":
            retrieved_chunk_ids = list(case.evidence[0].chunk_ids)
        clean: dict[str, object] = {
            "run_id": "fixture-run",
            "case_id": case.case_id,
            "variant": "clean",
            "answerability": case.answerability,
            "case_types": list(case.case_types),
            "task_pass": True,
            "citations_valid": True,
            "exact_abstention": case.answerability == "unanswerable",
            "retrieved_chunk_ids": retrieved_chunk_ids,
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
        fixtures,
        run_date="2026-08-29",
        bootstrap_resamples=100,
        bootstrap_seed=7,
    )
    svg = render_project_generation_svg(snapshot)

    assert snapshot["status"] == "automated_baseline_human_adjudication_pending"
    assert snapshot["metrics"]["clean_task"]["rate"] == 1.0  # type: ignore[index]
    assert snapshot["metrics"]["unanswerable_sentinel_compliance"]["rate"] == 1.0  # type: ignore[index]
    conditioned = snapshot["metrics"]["retrieval_conditioned"]  # type: ignore[index]
    assert conditioned["answerable_full_locator_coverage"]["successes"] == 79
    assert conditioned["answerable_task_pass_with_incomplete_locator_coverage"]["rate"] == 1.0
    assert snapshot["dataset"]["answerable_evidence_cluster_count"] == 45  # type: ignore[index]
    assert len(snapshot["per_case_outcomes"]) == 100
    element_tree.fromstring(svg)


def test_committed_generation_snapshot_discloses_retrieval_confound() -> None:
    snapshot = json.loads(PUBLIC_SNAPSHOT.read_text(encoding="utf-8"))

    assert snapshot["schema_version"] == 2
    assert "unanswerable_exact_abstention" not in snapshot["metrics"]
    assert snapshot["metrics"]["unanswerable_sentinel_compliance"]["successes"] == 19
    conditioned = snapshot["metrics"]["retrieval_conditioned"]
    assert conditioned["answerable_task_pass_given_full_locator_coverage"]["successes"] == 62
    assert conditioned["answerable_task_pass_given_full_locator_coverage"]["total"] == 70
    assert conditioned["multi_hop_task_pass_given_full_locator_coverage"]["successes"] == 4
    assert conditioned["multi_hop_task_pass_given_full_locator_coverage"]["total"] == 5
    assert conditioned["multi_hop_generation_status"].startswith("unmeasured_")
    retrieval = snapshot["audit_flags"]["retrieval_confound"]
    assert retrieval["answerable_case_count"] == 10
    assert retrieval["all_are_multi_hop"] is True
    assert "model_judge_spot_check" not in snapshot["audit_flags"]

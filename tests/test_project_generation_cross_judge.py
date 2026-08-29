from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llmqa.project_evaluation import load_project_evaluation_cases
from llmqa.project_generation_cross_judge import (
    binary_agreement,
    select_cross_judge_case_ids,
)

ROOT = Path(__file__).parents[1]
EVAL_DIR = ROOT / "evals" / "project" / "technical-papers-v1"
PUBLIC_SNAPSHOT = (
    ROOT / "docs" / "benchmarks" / "technical-papers-v1-generation-cross-judge-2026-08-29.json"
)
ADJUDICATION = (
    ROOT / "docs" / "benchmarks" / "technical-papers-v1-generation-adjudication-2026-08-29.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cross_judge_selection_covers_failures_attacks_and_seeded_passes() -> None:
    cases = load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")
    answerable_failure_ids = {case.case_id for case in cases[41:58]}
    rows = [
        {
            "case_id": case.case_id,
            "variant": "clean",
            "task_pass": case.case_id not in answerable_failure_ids,
        }
        for case in cases
    ]

    selected, reasons = select_cross_judge_case_ids(cases, rows, sample_size=30, seed=7)
    injection_ids = {case.case_id for case in cases if case.injection_fixture_id is not None}

    assert len(selected) == 30
    assert answerable_failure_ids <= set(selected)
    assert injection_ids <= set(selected)
    assert sum("seeded_clean_answerable_pass" in reasons[case_id] for case_id in selected) == (
        30 - len(answerable_failure_ids | injection_ids)
    )
    assert selected == select_cross_judge_case_ids(cases, rows, sample_size=30, seed=7)[0]


def test_binary_agreement_reports_directional_mcnemar_and_kappa() -> None:
    agreement = binary_agreement([(True, True), (True, False), (False, False), (False, False)])

    assert agreement["agreement_count"] == 3
    assert agreement["agreement_rate"] == 0.75
    assert agreement["cohen_kappa"] == 0.5
    assert agreement["directional_disagreement"] == {
        "primary_pass_cross_fail": 1,
        "primary_fail_cross_pass": 0,
        "mcnemar_exact_two_sided_p": 1.0,
    }
    assert agreement["confusion"] == {
        "primary_true_cross_true": 1,
        "primary_true_cross_false": 1,
        "primary_false_cross_true": 0,
        "primary_false_cross_false": 2,
    }


def test_committed_cross_judge_snapshot_reconciles() -> None:
    snapshot = json.loads(PUBLIC_SNAPSHOT.read_text(encoding="utf-8"))

    assert snapshot["status"] == "automated_cross_judge_complete_human_adjudication_pending"
    assert snapshot["counts"] == {
        "clean_answerable_variants": 25,
        "injected_variants": 10,
        "judged_variants": 35,
        "selected_cases": 30,
    }
    task = snapshot["agreement"]["task_pass_all_variants"]
    assert task["agreement_count"] == 24
    assert task["total"] == 35
    assert task["confusion"]["primary_true_cross_false"] == 0
    assert task["directional_disagreement"] == {
        "primary_pass_cross_fail": 0,
        "primary_fail_cross_pass": 11,
        "mcnemar_exact_two_sided_p": 0.0009765625,
    }
    clean = snapshot["agreement"]["task_pass_clean_answerable"]
    assert clean["directional_disagreement"] == {
        "primary_pass_cross_fail": 0,
        "primary_fail_cross_pass": 9,
        "mcnemar_exact_two_sided_p": 0.00390625,
    }
    injection = snapshot["agreement"]["injection_criteria"]["passed"]
    assert injection["agreement_count"] == 8
    assert injection["total"] == 10
    assert len(snapshot["disagreements"]) == 11
    sensitivity = snapshot["judge_sensitivity"]
    assert sensitivity["clean_answerable_failure_complete_sensitivity"] == {
        "audited_primary_passes": 8,
        "audited_primary_passes_retained": 8,
        "cross_judge_imputed": {"passes": 72, "rate": 0.9, "total": 80},
        "failure_complete": True,
        "interpretation": (
            "Failure-complete sensitivity scenario, not a full recomputation: every primary "
            "failure was re-judged, while unjudged primary passes are assumed to remain passes."
        ),
        "primary": {"passes": 63, "rate": 0.7875, "total": 80},
        "unjudged_primary_passes_assumed_retained": 55,
    }
    assert sensitivity["injection_joint_same_outputs"] == {
        "cross_judge_passes": 6,
        "primary_passes": 4,
        "status": "exact_rejudgment_of_all_ten_attacked_outputs",
        "total": 10,
    }
    assert snapshot["usage"]["actual_api_requests"] is None


def test_committed_human_adjudication_covers_every_disagreement() -> None:
    snapshot = json.loads(PUBLIC_SNAPSHOT.read_text(encoding="utf-8"))
    adjudication = json.loads(ADJUDICATION.read_text(encoding="utf-8"))
    decisions = adjudication["decisions"]

    disagreement_keys = {(row["case_id"], row["variant"]) for row in snapshot["disagreements"]}
    decision_keys = {(row["case_id"], row["variant"]) for row in decisions}
    assert decision_keys == disagreement_keys
    assert len(decision_keys) == len(decisions) == 11
    assert adjudication["summary"] == {
        "fail": 5,
        "pass": 6,
        "uphold_cross_judge": 6,
        "uphold_primary": 5,
    }
    provenance = adjudication["provenance"]
    assert provenance["cases_sha256"] == _sha256(EVAL_DIR / "cases.jsonl")
    assert provenance["cross_judge_snapshot_sha256"] == _sha256(PUBLIC_SNAPSHOT)
    assert provenance["primary_snapshot_sha256"] == _sha256(
        ROOT / "docs" / "benchmarks" / "technical-papers-v1-generation-automated-2026-08-29.json"
    )

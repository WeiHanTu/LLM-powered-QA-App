from __future__ import annotations

import json
import xml.etree.ElementTree as element_tree
from pathlib import Path

import pytest

from llmqa.project_evaluation import (
    InjectionFixture,
    ProjectEvaluationCase,
    load_injection_fixtures,
    load_project_evaluation_cases,
)
from llmqa.project_generation_reporting import (
    adjudication_selection_sha256,
    build_project_generation_snapshot,
    render_project_generation_svg,
)

ROOT = Path(__file__).parents[1]
EVAL_DIR = ROOT / "evals" / "project" / "technical-papers-v1"
PUBLIC_SNAPSHOT = (
    ROOT / "docs" / "benchmarks" / "technical-papers-v1-generation-automated-2026-08-29.json"
)
REQUIRED_CLAIMS_SNAPSHOT = (
    ROOT
    / "docs"
    / "benchmarks"
    / "technical-papers-v1-generation-required-claims-v1-2026-08-29.json"
)
HISTORICAL_ADJUDICATION = (
    ROOT / "docs" / "benchmarks" / "technical-papers-v1-generation-adjudication-2026-08-29.json"
)


def _fixture_run() -> tuple[
    dict[str, object],
    list[dict[str, object]],
    tuple[ProjectEvaluationCase, ...],
    tuple[InjectionFixture, ...],
]:
    """Build a synthetic all-passing run over the reviewed 100-case set."""

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
            "claim_contract_version": "required-claims-v1",
            "injection_placement_version": "placement-v1",
        },
        "provenance": {"cases_sha256": "fixture-cases-hash"},
        "usage": {},
        "limitations": ["fixture limitation"],
    }
    return summary, rows, cases, fixtures


def _review_artifact(
    summary: dict[str, object],
    decisions: list[dict[str, object]],
    *,
    status: str = "partial_human_adjudication",
) -> dict[str, object]:
    keys = [(str(row["case_id"]), str(row["variant"])) for row in decisions]
    configuration = summary["configuration"]
    provenance = summary["provenance"]
    assert isinstance(configuration, dict)
    assert isinstance(provenance, dict)
    return {
        "schema_version": 2,
        "status": status,
        "dataset": summary["dataset"],
        "reviewer_ids": ["reviewer-1"],
        "reviewed_at": "2026-08-30T00:00:00Z",
        "scope": {
            "primary_run_id": summary["run_id"],
            "adjudicated_field": "task_pass",
            "review_design": "direct_output_review",
            "variant_count": len(decisions),
            "selected_variants_sha256": adjudication_selection_sha256(keys),
        },
        "provenance": {
            "cases_sha256": provenance["cases_sha256"],
            "claim_contract_version": configuration["claim_contract_version"],
            "primary_summary_sha256": "summary-sha",
            "primary_results_sha256": "results-sha",
        },
        "decisions": decisions,
    }


def _decision(
    row: dict[str, object], verdict: str, rationale: str = "reviewed"
) -> dict[str, object]:
    return {
        "case_id": row["case_id"],
        "variant": row["variant"],
        "decision": verdict,
        "rationale": rationale,
    }


def test_generation_snapshot_is_full_and_explicitly_provisional() -> None:
    summary, rows, cases, fixtures = _fixture_run()

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
    assert "semantic_abstention_gap" not in snapshot["audit_flags"]
    element_tree.fromstring(svg)


def test_partial_human_review_is_bound_but_does_not_replace_headline_metrics() -> None:
    summary, rows, cases, fixtures = _fixture_run()
    artifact = _review_artifact(
        summary,
        [_decision(rows[0], "fail", "The reviewed output misses a required claim.")],
    )

    snapshot = build_project_generation_snapshot(
        summary,
        rows,
        cases,
        fixtures,
        run_date="2026-08-30",
        bootstrap_resamples=100,
        raw_summary_sha256="summary-sha",
        raw_cases_sha256="results-sha",
        adjudication=artifact,
        adjudication_sha256="adjudication-sha",
    )
    svg = render_project_generation_svg(snapshot)

    assert snapshot["status"] == "automated_baseline_partial_human_review"
    assert snapshot["metrics"]["clean_task"]["successes"] == 100  # type: ignore[index]
    reviewed = snapshot["metrics_with_reviewed_overrides"]  # type: ignore[assignment]
    assert reviewed["clean_task"]["successes"] == 99
    assert snapshot["per_case_outcomes"][0]["clean_task_pass"] is True  # type: ignore[index]
    human_review = snapshot["human_review"]  # type: ignore[assignment]
    assert human_review["source"] == {
        "artifact_sha256": "adjudication-sha",
        "provenance": artifact["provenance"],
    }
    assert human_review["overrides"][0]["rationale"] == (  # type: ignore[index]
        "The reviewed output misses a required claim."
    )
    assert "partial human review 1/110; headline remains automated" in svg
    assert "Human-reviewed RAG generation evaluation" not in svg


def test_complete_human_review_requires_and_labels_every_variant() -> None:
    summary, rows, cases, fixtures = _fixture_run()
    decisions = [_decision(row, "pass" if bool(row["task_pass"]) else "fail") for row in rows]
    artifact = _review_artifact(summary, decisions, status="complete_human_adjudication")

    snapshot = build_project_generation_snapshot(
        summary,
        rows,
        cases,
        fixtures,
        run_date="2026-08-30",
        bootstrap_resamples=100,
        raw_summary_sha256="summary-sha",
        raw_cases_sha256="results-sha",
        adjudication=artifact,
        adjudication_sha256="adjudication-sha",
    )

    assert snapshot["status"] == "human_adjudication_complete"
    assert snapshot["metrics_human_reviewed"] == snapshot["metrics"]
    assert "Human-reviewed RAG generation evaluation" in render_project_generation_svg(snapshot)


def test_adjudication_requires_all_three_artifact_hashes() -> None:
    summary, rows, cases, fixtures = _fixture_run()
    artifact = _review_artifact(summary, [_decision(rows[0], "fail")])

    with pytest.raises(ValueError, match="requires summary, results, and adjudication"):
        build_project_generation_snapshot(
            summary,
            rows,
            cases,
            fixtures,
            run_date="2026-08-30",
            bootstrap_resamples=10,
            adjudication=artifact,
        )


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("schema_version",), 1, "schema version 2"),
        (("dataset",), "other-dataset", "dataset does not match"),
        (("scope", "primary_run_id"), "historical-run", "scopes run"),
        (("scope", "adjudicated_field"), "citation", "only task_pass"),
        (("scope", "review_design"), "cross_judge", "only direct_output_review"),
        (("scope", "variant_count"), 2, "variant_count"),
        (("scope", "selected_variants_sha256"), "wrong", "selected-variant hash"),
        (("provenance", "cases_sha256"), "wrong", "cases hash"),
        (("provenance", "claim_contract_version"), "old", "claim contract"),
        (("provenance", "primary_summary_sha256"), "wrong", "summary hash"),
        (("provenance", "primary_results_sha256"), "wrong", "results hash"),
        (("reviewer_ids",), [], "reviewer_ids"),
        (("reviewed_at",), "", "reviewed_at"),
        (("decisions", 0, "rationale"), "", "requires a rationale"),
    ],
)
def test_adjudication_identity_and_schema_mismatches_fail_closed(
    path: tuple[str | int, ...], replacement: object, message: str
) -> None:
    summary, rows, cases, fixtures = _fixture_run()
    artifact = _review_artifact(summary, [_decision(rows[0], "fail")])
    target: object = artifact
    for part in path[:-1]:
        assert isinstance(target, dict | list)
        target = target[part]  # type: ignore[index]
    assert isinstance(target, dict | list)
    target[path[-1]] = replacement  # type: ignore[index]

    with pytest.raises(ValueError, match=message):
        build_project_generation_snapshot(
            summary,
            rows,
            cases,
            fixtures,
            run_date="2026-08-30",
            bootstrap_resamples=10,
            raw_summary_sha256="summary-sha",
            raw_cases_sha256="results-sha",
            adjudication=artifact,
            adjudication_sha256="adjudication-sha",
        )


def test_review_status_must_match_coverage() -> None:
    summary, rows, cases, fixtures = _fixture_run()
    partial_marked_complete = _review_artifact(
        summary,
        [_decision(rows[0], "fail")],
        status="complete_human_adjudication",
    )
    complete_marked_partial = _review_artifact(
        summary,
        [_decision(row, "pass") for row in rows],
        status="partial_human_adjudication",
    )

    for artifact, message in (
        (partial_marked_complete, "must review every run variant"),
        (complete_marked_partial, "must use complete_human_adjudication"),
    ):
        with pytest.raises(ValueError, match=message):
            build_project_generation_snapshot(
                summary,
                rows,
                cases,
                fixtures,
                run_date="2026-08-30",
                bootstrap_resamples=10,
                raw_summary_sha256="summary-sha",
                raw_cases_sha256="results-sha",
                adjudication=artifact,
                adjudication_sha256="adjudication-sha",
            )


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


def test_committed_required_claims_snapshot_reconciles() -> None:
    snapshot = json.loads(REQUIRED_CLAIMS_SNAPSHOT.read_text(encoding="utf-8"))

    assert snapshot["run_id"] == "68f31a98962453b5a9b6"
    assert snapshot["configuration"]["claim_contract_version"] == "required-claims-v1"
    assert snapshot["configuration"]["judge_prompt_version"] == (
        "generation-judge-required-claims-v2"
    )
    assert snapshot["metrics"]["clean_task"]["successes"] == 90
    assert snapshot["metrics"]["answerable_grounded_query"]["successes"] == 70
    assert snapshot["metrics"]["unanswerable_sentinel_compliance"]["successes"] == 20
    assert snapshot["metrics"]["injection_joint_pass"]["successes"] == 6
    conditioned = snapshot["metrics"]["retrieval_conditioned"]
    assert conditioned["answerable_task_pass_given_full_locator_coverage"]["successes"] == 68
    assert conditioned["answerable_task_pass_given_full_locator_coverage"]["total"] == 70
    assert conditioned["multi_hop_task_pass_given_full_locator_coverage"]["total"] == 5
    assert "semantic_abstention_gap" not in snapshot["audit_flags"]
    element_tree.parse(
        ROOT
        / "docs"
        / "benchmarks"
        / "technical-papers-v1-generation-required-claims-v1-2026-08-29.svg"
    )


def test_historical_adjudication_is_explicitly_nontransferable() -> None:
    historical = json.loads(HISTORICAL_ADJUDICATION.read_text(encoding="utf-8"))
    current = json.loads(REQUIRED_CLAIMS_SNAPSHOT.read_text(encoding="utf-8"))

    assert historical["schema_version"] == 1
    assert historical["scope"]["primary_run_id"] == "3992f7ad274839fdcb26"
    assert current["run_id"] == "68f31a98962453b5a9b6"
    assert historical["scope"]["primary_run_id"] != current["run_id"]
    assert current["status"] == "automated_baseline_human_adjudication_pending"
    assert any(
        "do not transfer automatically to a new generation run" in limitation
        for limitation in historical["limitations"]
    )

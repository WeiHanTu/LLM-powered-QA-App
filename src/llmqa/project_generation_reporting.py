"""Compact, explicitly provisional reporting for project generation evaluation."""

from __future__ import annotations

import hashlib
import html
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from llmqa.project_evaluation import (
    InjectionFixture,
    ProjectEvaluationCase,
    load_injection_fixtures,
    load_project_evaluation_cases,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(Mapping[str, Any], raw)


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(cast(Mapping[str, Any], raw))
    return rows


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("metric denominator must be positive")
    return numerator / denominator


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("Wilson interval requires 0 <= successes <= positive total")
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _evidence_signature(case: ProjectEvaluationCase) -> str:
    chunk_ids = sorted({chunk_id for locator in case.evidence for chunk_id in locator.chunk_ids})
    return hashlib.sha256("\n".join(chunk_ids).encode("utf-8")).hexdigest()[:12]


def _cluster_metric(
    cases: Sequence[ProjectEvaluationCase],
    clean_by_id: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        if case.answerability != "answerable":
            continue
        grouped[_evidence_signature(case)].append(
            float(bool(clean_by_id[case.case_id]["task_pass"]))
        )
    cluster_means = np.asarray(
        [sum(values) / len(values) for _, values in sorted(grouped.items())], dtype=np.float64
    )
    if len(cluster_means) != 45:
        raise ValueError(f"expected 45 answerable evidence clusters, found {len(cluster_means)}")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    generator = np.random.default_rng(bootstrap_seed)
    indices = generator.integers(
        0, len(cluster_means), size=(bootstrap_resamples, len(cluster_means))
    )
    means = cluster_means[indices].mean(axis=1)
    return {
        "cluster_count": len(cluster_means),
        "macro_mean": float(cluster_means.mean()),
        "bootstrap_95_ci": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
    }


def _metric(successes: int, total: int) -> dict[str, object]:
    return {
        "successes": successes,
        "total": total,
        "rate": _ratio(successes, total),
        "wilson_95_ci": _wilson_interval(successes, total),
    }


def _injection_score(row: Mapping[str, Any]) -> Mapping[str, Any]:
    score = row.get("injection_score")
    if not isinstance(score, dict):
        raise ValueError(f"injected row {row.get('case_id')!r} is missing injection_score")
    return cast(Mapping[str, Any], score)


def _evidence_locator_hits(case: ProjectEvaluationCase, row: Mapping[str, Any]) -> tuple[bool, ...]:
    raw_retrieved = row.get("retrieved_chunk_ids")
    if not isinstance(raw_retrieved, list) or not all(
        isinstance(chunk_id, str) for chunk_id in raw_retrieved
    ):
        raise ValueError(f"clean row {case.case_id!r} is missing retrieved_chunk_ids")
    retrieved = set(cast(list[str], raw_retrieved))
    return tuple(bool(retrieved.intersection(locator.chunk_ids)) for locator in case.evidence)


def _all_evidence_locators_retrieved(case: ProjectEvaluationCase, row: Mapping[str, Any]) -> bool:
    return all(_evidence_locator_hits(case, row))


ADJUDICATION_VARIANTS = ("clean", "injected")
ADJUDICATION_DECISIONS = ("pass", "fail")
ADJUDICATION_STATUSES = ("partial_human_adjudication", "complete_human_adjudication")
ADJUDICATION_REVIEW_DESIGN = "direct_output_review"


def adjudication_selection_sha256(keys: Sequence[tuple[str, str]]) -> str:
    """Hash the exact case/variant set selected for direct human review."""

    payload = [{"case_id": case_id, "variant": variant} for case_id, variant in sorted(set(keys))]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _validate_adjudication(
    adjudication: Mapping[str, Any],
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    raw_summary_sha256: str | None,
    raw_cases_sha256: str | None,
    adjudication_sha256: str | None,
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], dict[str, object]]:
    """Bind a human adjudication record to exactly one run, or refuse it.

    A stale record is the dangerous failure here: applying decisions made against a
    different run silently relabels outputs no reviewer ever saw. Every identity field
    is therefore checked, and any mismatch raises instead of degrading.
    """

    if raw_summary_sha256 is None or raw_cases_sha256 is None or adjudication_sha256 is None:
        raise ValueError("adjudication requires summary, results, and adjudication artifact hashes")
    review_status = adjudication.get("status")
    if adjudication.get("schema_version") != 2 or review_status not in ADJUDICATION_STATUSES:
        raise ValueError("adjudication requires schema version 2 and a supported review status")
    if adjudication.get("dataset") != summary.get("dataset"):
        raise ValueError("adjudication dataset does not match the run")

    scope = adjudication.get("scope")
    provenance = adjudication.get("provenance")
    if not isinstance(scope, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("adjudication requires object 'scope' and 'provenance' fields")
    if scope.get("primary_run_id") != summary["run_id"]:
        raise ValueError(
            f"adjudication scopes run {scope.get('primary_run_id')!r}, not {summary['run_id']!r}"
        )
    if scope.get("adjudicated_field") != "task_pass":
        raise ValueError("only task_pass adjudication is supported")
    if scope.get("review_design") != ADJUDICATION_REVIEW_DESIGN:
        raise ValueError("only direct_output_review adjudication is supported")
    summary_provenance = cast(Mapping[str, Any], summary["provenance"])
    if provenance.get("cases_sha256") != summary_provenance.get("cases_sha256"):
        raise ValueError("adjudication cases hash does not match the run")
    contract = cast(Mapping[str, Any], summary["configuration"]).get("claim_contract_version")
    if provenance.get("claim_contract_version") != contract:
        raise ValueError("adjudication claim contract does not match the run")
    if provenance.get("primary_summary_sha256") != raw_summary_sha256:
        raise ValueError("adjudication primary summary hash does not match the run")
    if provenance.get("primary_results_sha256") != raw_cases_sha256:
        raise ValueError("adjudication primary results hash does not match the run")
    reviewers = adjudication.get("reviewer_ids")
    if (
        not isinstance(reviewers, list)
        or not reviewers
        or any(not isinstance(item, str) or not item for item in reviewers)
    ):
        raise ValueError("adjudication requires a non-empty reviewer_ids list")
    if not isinstance(adjudication.get("reviewed_at"), str) or not adjudication["reviewed_at"]:
        raise ValueError("adjudication requires a reviewed_at timestamp")

    decisions = adjudication.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("adjudication requires a non-empty decisions list")
    row_index = {(str(row["case_id"]), str(row["variant"])): row for row in rows}
    validated: dict[tuple[str, str], Mapping[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError("each adjudication decision must be a JSON object")
        case_id = decision.get("case_id")
        variant = decision.get("variant")
        verdict = decision.get("decision")
        rationale = decision.get("rationale")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("adjudication decision requires a case_id")
        if variant not in ADJUDICATION_VARIANTS:
            raise ValueError(f"{case_id}: variant must be one of {ADJUDICATION_VARIANTS}")
        if verdict not in ADJUDICATION_DECISIONS:
            raise ValueError(f"{case_id}: decision must be one of {ADJUDICATION_DECISIONS}")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"{case_id}: adjudication decision requires a rationale")
        key = (case_id, str(variant))
        if key not in row_index:
            raise ValueError(f"adjudication decision {key} is absent from the run results")
        if key in validated:
            raise ValueError(f"adjudication repeats decision for {key}")
        validated[key] = decision

    selected_hash = adjudication_selection_sha256(list(validated))
    if scope.get("selected_variants_sha256") != selected_hash:
        raise ValueError("adjudication selected-variant hash does not match its decisions")
    variant_count = scope.get("variant_count")
    if isinstance(variant_count, bool) or variant_count != len(validated):
        raise ValueError("adjudication variant_count does not match its decisions")
    all_row_keys = set(row_index)
    reviewed_keys = set(validated)
    if review_status == "complete_human_adjudication" and reviewed_keys != all_row_keys:
        raise ValueError("complete adjudication must review every run variant")
    if review_status == "partial_human_adjudication" and reviewed_keys == all_row_keys:
        raise ValueError("a review of every run variant must use complete_human_adjudication")

    overrides = [
        {
            "case_id": case_id,
            "variant": variant,
            "automated_task_pass": bool(row_index[(case_id, variant)]["task_pass"]),
            "adjudicated_task_pass": validated[(case_id, variant)]["decision"] == "pass",
            "rationale": str(validated[(case_id, variant)]["rationale"]).strip(),
        }
        for case_id, variant in sorted(validated)
    ]
    changed = [
        item for item in overrides if item["automated_task_pass"] != item["adjudicated_task_pass"]
    ]
    record: dict[str, object] = {
        "status": review_status,
        "reviewer_ids": list(reviewers),
        "reviewed_at": adjudication["reviewed_at"],
        "scope": dict(scope),
        "source": {
            "artifact_sha256": adjudication_sha256,
            "provenance": dict(provenance),
        },
        "adjudicated_variant_count": len(validated),
        "total_variant_count": len(rows),
        "changed_verdict_count": len(changed),
        "coverage_statement": (
            "Human review covered the variants listed in 'overrides' only. Every other variant "
            "retains its automated judge verdict unless status is complete_human_adjudication."
        ),
        "overrides": overrides,
    }
    return validated, record


def _apply_adjudication(
    rows: Sequence[Mapping[str, Any]],
    decisions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return rows with adjudicated task_pass applied; other rows keep the automated verdict."""

    applied: list[Mapping[str, Any]] = []
    for row in rows:
        decision = decisions.get((str(row["case_id"]), str(row["variant"])))
        if decision is None:
            applied.append(row)
            continue
        updated = dict(row)
        updated["task_pass"] = decision["decision"] == "pass"
        applied.append(updated)
    return applied


def _metrics_delta(automated: Mapping[str, Any], adjudicated: Mapping[str, Any]) -> dict[str, Any]:
    """Report per-metric movement so the size of the human correction stays visible."""

    delta: dict[str, Any] = {}
    for key, value in automated.items():
        other = adjudicated.get(key)
        if not isinstance(value, Mapping) or not isinstance(other, Mapping):
            continue
        if "successes" in value and "successes" in other:
            delta[key] = {
                "automated_successes": value["successes"],
                "adjudicated_successes": other["successes"],
                "successes_delta": int(other["successes"]) - int(value["successes"]),
                "total": value.get("total"),
            }
    return delta


def build_project_generation_snapshot(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    cases: Sequence[ProjectEvaluationCase],
    fixtures: Sequence[InjectionFixture],
    *,
    run_date: str,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20_260_829,
    raw_summary_sha256: str | None = None,
    raw_cases_sha256: str | None = None,
    adjudication: Mapping[str, Any] | None = None,
    adjudication_sha256: str | None = None,
) -> dict[str, object]:
    """Build a public report, optionally applying a bound human adjudication record.

    Without ``adjudication`` the output is the automated baseline at schema 2, unchanged.
    With one, the automated metrics are retained verbatim and the adjudicated metrics are
    added beside them, so the size of the human correction stays auditable.
    """

    if bool(summary.get("limited_run")) or not bool(summary.get("complete")):
        raise ValueError("generation report requires one complete, non-limited run")
    if not bool(summary.get("ready_for_automated_report")):
        raise ValueError("generation summary is not ready for an automated report")
    if summary.get("human_adjudication_status") != "pending":
        raise ValueError("unsupported human adjudication status")
    run_id = str(summary["run_id"])
    adjudicated_decisions: dict[tuple[str, str], Mapping[str, Any]] | None = None
    adjudication_record: dict[str, object] | None = None
    if any(row.get("run_id") != run_id for row in rows):
        raise ValueError("case result run IDs do not match the summary")
    if len(cases) != 100:
        raise ValueError("project generation report requires the complete 100-case set")
    if len(fixtures) != 10:
        raise ValueError("project generation report requires the complete 10-fixture set")
    clean = [row for row in rows if row.get("variant") == "clean"]
    injected = [row for row in rows if row.get("variant") == "injected"]
    if len(clean) != 100 or len(injected) != 10:
        raise ValueError("project generation report requires 100 clean and 10 injected rows")
    clean_by_id = {str(row["case_id"]): row for row in clean}
    injected_by_id = {str(row["case_id"]): row for row in injected}
    case_by_id = {case.case_id: case for case in cases}
    fixture_by_id = {fixture.fixture_id: fixture for fixture in fixtures}
    if len(fixture_by_id) != len(fixtures):
        raise ValueError("injection fixture IDs must be unique")
    if set(clean_by_id) != set(case_by_id):
        raise ValueError("clean generation rows do not cover the reviewed cases exactly")
    expected_injected = {case.case_id for case in cases if case.injection_fixture_id is not None}
    if set(injected_by_id) != expected_injected:
        raise ValueError("injected generation rows do not cover the ten reviewed fixtures")
    expected_fixture_ids = {
        case.injection_fixture_id for case in cases if case.injection_fixture_id is not None
    }
    if set(fixture_by_id) != expected_fixture_ids:
        raise ValueError("reviewed fixtures do not match the fixture IDs referenced by cases")
    if adjudication is not None:
        adjudicated_decisions, adjudication_record = _validate_adjudication(
            adjudication,
            summary,
            rows,
            raw_summary_sha256=raw_summary_sha256,
            raw_cases_sha256=raw_cases_sha256,
            adjudication_sha256=adjudication_sha256,
        )

    answerable = [case for case in cases if case.answerability == "answerable"]
    unanswerable = [case for case in cases if case.answerability == "unanswerable"]
    answerable_passes = sum(bool(clean_by_id[case.case_id]["task_pass"]) for case in answerable)
    sentinel_passes = sum(
        bool(clean_by_id[case.case_id]["exact_abstention"]) for case in unanswerable
    )
    clean_passes = sum(bool(row["task_pass"]) for row in clean)
    citation_passes = sum(bool(row["citations_valid"]) for row in clean)
    joint_passes = sum(bool(_injection_score(row)["passed"]) for row in injected)
    attacked_task_passes = sum(bool(row["task_pass"]) for row in injected)
    clean_injection_passes = sum(
        bool(clean_by_id[case_id]["task_pass"]) for case_id in expected_injected
    )
    attack_induced_ids = sorted(
        case_id
        for case_id in expected_injected
        if bool(clean_by_id[case_id]["task_pass"])
        and not bool(injected_by_id[case_id]["task_pass"])
    )
    criteria = (
        "answer_or_abstention",
        "no_fabricated_claims",
        "no_forbidden_actions",
        "no_injected_citation",
        "no_canary_leakage",
    )
    criterion_metrics = {
        criterion: _metric(
            sum(bool(_injection_score(row)[criterion]) for row in injected), len(injected)
        )
        for criterion in criteria
    }
    fully_retrieved_answerable = [
        case
        for case in answerable
        if _all_evidence_locators_retrieved(case, clean_by_id[case.case_id])
    ]
    under_retrieved_answerable = [
        case
        for case in answerable
        if not _all_evidence_locators_retrieved(case, clean_by_id[case.case_id])
    ]
    multi_hop = [case for case in answerable if "multi_hop" in case.case_types]
    fully_retrieved_multi_hop = [
        case
        for case in multi_hop
        if _all_evidence_locators_retrieved(case, clean_by_id[case.case_id])
    ]
    answerable_only = [case for case in answerable if case.case_types == ("answerable",)]

    def task_passes(selected_cases: Sequence[ProjectEvaluationCase]) -> int:
        return sum(bool(clean_by_id[case.case_id]["task_pass"]) for case in selected_cases)

    under_retrieved_failure_ids = sorted(
        case.case_id
        for case in under_retrieved_answerable
        if not bool(clean_by_id[case.case_id]["task_pass"])
    )
    fully_retrieved_failure_ids = sorted(
        case.case_id
        for case in fully_retrieved_answerable
        if not bool(clean_by_id[case.case_id]["task_pass"])
    )
    incomplete_ids: list[str] = []
    answerable_abstention_ids: list[str] = []
    other_answerable_failure_ids: list[str] = []
    for case in answerable:
        row = clean_by_id[case.case_id]
        if bool(row["task_pass"]):
            continue
        judgment = row.get("semantic_judgment")
        if bool(row.get("exact_abstention")):
            answerable_abstention_ids.append(case.case_id)
        elif (
            isinstance(judgment, dict)
            and judgment.get("answer_correct") is False
            and judgment.get("fully_supported") is True
        ):
            incomplete_ids.append(case.case_id)
        else:
            other_answerable_failure_ids.append(case.case_id)

    public_outcomes = []
    for case in cases:
        clean_row = clean_by_id[case.case_id]
        outcome: dict[str, object] = {
            "case_id": case.case_id,
            "answerability": case.answerability,
            "case_types": list(case.case_types),
            "clean_task_pass": bool(clean_row["task_pass"]),
            "citations_valid": bool(clean_row["citations_valid"]),
            "sentinel_compliance": bool(clean_row["exact_abstention"]),
        }
        if case.answerability == "answerable":
            locator_hits = _evidence_locator_hits(case, clean_row)
            outcome["evidence_locator_count"] = len(locator_hits)
            outcome["retrieved_evidence_locator_count"] = sum(locator_hits)
            outcome["full_evidence_locator_coverage"] = all(locator_hits)
        if case.case_id in injected_by_id:
            injected_row = injected_by_id[case.case_id]
            score = _injection_score(injected_row)
            fixture = fixture_by_id[cast(str, case.injection_fixture_id)]
            outcome["injection_fixture_id"] = fixture.fixture_id
            outcome["injection_fixture_style"] = fixture.style
            outcome["injected_task_pass"] = bool(injected_row["task_pass"])
            outcome["injection_joint_pass"] = bool(score["passed"])
            outcome["injection_criteria"] = {
                criterion: bool(score[criterion]) for criterion in criteria
            }
        public_outcomes.append(outcome)

    raw_configuration = cast(Mapping[str, Any], summary["configuration"])
    configuration = {
        key: raw_configuration[key]
        for key in (
            "candidate_model",
            "judge_model",
            "retriever",
            "k",
            "bm25_k1",
            "bm25_b",
            "candidate_prompt_sha256",
            "judge_prompt_sha256",
            "judge_schema_sha256",
            "judge_prompt_version",
            "claim_contract_version",
            "injection_placement_version",
        )
    }
    noncompliant_unanswerable_ids = sorted(
        case.case_id
        for case in unanswerable
        if not bool(clean_by_id[case.case_id]["exact_abstention"])
    )
    audit_flags: dict[str, object] = {
        "retrieval_confound": {
            "answerable_case_count": len(under_retrieved_answerable),
            "case_ids": sorted(case.case_id for case in under_retrieved_answerable),
            "all_are_multi_hop": all(
                "multi_hop" in case.case_types for case in under_retrieved_answerable
            ),
        },
        "human_adjudication_status": (
            "pending" if adjudication_record is None else adjudication_record["status"]
        ),
    }
    if noncompliant_unanswerable_ids:
        audit_flags["semantic_abstention_gap"] = {
            "case_ids": noncompliant_unanswerable_ids,
            "issue": (
                "These responses failed exact sentinel compliance. Clean unanswerable outputs are "
                "not sent to the semantic judge, so this run does not establish whether any richer "
                "evidence-based refusal was acceptable."
            ),
        }
    snapshot: dict[str, object] = {
        "schema_version": 2 if adjudication_record is None else 3,
        "status": (
            "automated_baseline_human_adjudication_pending"
            if adjudication_record is None
            else (
                "human_adjudication_complete"
                if adjudication_record["status"] == "complete_human_adjudication"
                else "automated_baseline_partial_human_review"
            )
        ),
        "run_date": run_date,
        "run_id": run_id,
        "dataset": {
            "name": summary["dataset"],
            "case_count": len(cases),
            "answerable_count": len(answerable),
            "unanswerable_count": len(unanswerable),
            "injection_fixture_count": len(injected),
            "answerable_evidence_cluster_count": 45,
            "provenance": summary["provenance"],
        },
        "configuration": configuration,
        "metrics": {
            "clean_task": _metric(clean_passes, len(clean)),
            "answerable_grounded_query": _metric(answerable_passes, len(answerable)),
            "answerable_grounded_evidence_cluster": _cluster_metric(
                cases,
                clean_by_id,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "unanswerable_sentinel_compliance": _metric(sentinel_passes, len(unanswerable)),
            "citation_validity": _metric(citation_passes, len(clean)),
            "injection_joint_pass": _metric(joint_passes, len(injected)),
            "injected_task_pass": _metric(attacked_task_passes, len(injected)),
            "attack_induced_failure_among_clean_passes": _metric(
                len(attack_induced_ids), clean_injection_passes
            ),
            "injection_criteria": criterion_metrics,
            "retrieval_conditioned": {
                "definition": (
                    "A reviewed evidence locator is retrieved when at least one deterministic "
                    "chunk ID from its cited page appears in BM25 top-k context."
                ),
                "answerable_full_locator_coverage": _metric(
                    len(fully_retrieved_answerable), len(answerable)
                ),
                "answerable_task_pass_given_full_locator_coverage": _metric(
                    task_passes(fully_retrieved_answerable), len(fully_retrieved_answerable)
                ),
                "answerable_task_pass_with_incomplete_locator_coverage": _metric(
                    task_passes(under_retrieved_answerable), len(under_retrieved_answerable)
                ),
                "multi_hop_task_pass": _metric(task_passes(multi_hop), len(multi_hop)),
                "multi_hop_full_locator_coverage": _metric(
                    len(fully_retrieved_multi_hop), len(multi_hop)
                ),
                "multi_hop_task_pass_given_full_locator_coverage": _metric(
                    task_passes(fully_retrieved_multi_hop), len(fully_retrieved_multi_hop)
                ),
                "multi_hop_generation_status": ("unmeasured_insufficient_fully_retrieved_cases"),
                "answerable_only_task_pass": _metric(
                    task_passes(answerable_only), len(answerable_only)
                ),
                "answerable_only_full_locator_coverage": _metric(
                    sum(
                        _all_evidence_locators_retrieved(case, clean_by_id[case.case_id])
                        for case in answerable_only
                    ),
                    len(answerable_only),
                ),
            },
        },
        "failure_analysis": {
            "automated_judge_labels": {
                "answerable_incomplete": {
                    "count": len(incomplete_ids),
                    "case_ids": incomplete_ids,
                },
                "answerable_abstention": {
                    "count": len(answerable_abstention_ids),
                    "case_ids": answerable_abstention_ids,
                },
                "answerable_other": {
                    "count": len(other_answerable_failure_ids),
                    "case_ids": other_answerable_failure_ids,
                },
                "warning": (
                    "These labels describe output symptoms, not root causes; some rows are also "
                    "retrieval-constrained."
                ),
            },
            "answerable_by_retrieval_coverage": {
                "retrieval_constrained": {
                    "count": len(under_retrieved_failure_ids),
                    "case_ids": under_retrieved_failure_ids,
                },
                "fully_retrieved": {
                    "count": len(fully_retrieved_failure_ids),
                    "case_ids": fully_retrieved_failure_ids,
                },
            },
            "unanswerable_sentinel_noncompliance": {
                "count": len(unanswerable) - sentinel_passes,
                "case_ids": [
                    case.case_id
                    for case in unanswerable
                    if not bool(clean_by_id[case.case_id]["exact_abstention"])
                ],
            },
            "attack_induced": {
                "count": len(attack_induced_ids),
                "case_ids": attack_induced_ids,
            },
        },
        "audit_flags": audit_flags,
        "per_case_outcomes": public_outcomes,
        "usage": summary["usage"],
        "artifact_sha256": {
            "raw_summary": raw_summary_sha256,
            "raw_case_results": raw_cases_sha256,
        },
        "limitations": [
            *cast(Sequence[str], summary["limitations"]),
            (
                "The primary judge labeled fifteen answerable failures as supported but "
                "incomplete. That symptom label is not a causal diagnosis and includes "
                "retrieval-constrained rows."
            ),
            (
                "Ten answerable cases lacked at least one reviewed evidence locator in top-10 "
                "context; all ten are multi-hop, so their failures confound retrieval and "
                "generation. Only five multi-hop cases had full locator coverage."
            ),
            (
                "The unanswerable metric is exact sentinel compliance, not semantic abstention "
                "quality; unanswerable rows were not sent to the model judge."
            ),
            (
                "Wilson intervals describe binomial sampling uncertainty only; the answerable "
                "cluster bootstrap separately addresses repeated evidence sets."
            ),
        ],
    }
    if adjudicated_decisions is None or adjudication_record is None:
        return snapshot

    # Recompute the same metric block over adjudicated verdicts. Recursing with
    # adjudication=None reuses every definition above rather than duplicating it.
    reviewed_metrics = build_project_generation_snapshot(
        summary,
        _apply_adjudication(rows, adjudicated_decisions),
        cases,
        fixtures,
        run_date=run_date,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )["metrics"]
    review_status = str(adjudication_record["status"])
    reviewed_metrics_field = (
        "metrics_human_reviewed"
        if review_status == "complete_human_adjudication"
        else "metrics_with_reviewed_overrides"
    )
    snapshot[reviewed_metrics_field] = reviewed_metrics
    snapshot["human_review_metric_delta"] = _metrics_delta(
        cast(Mapping[str, Any], snapshot["metrics"]),
        cast(Mapping[str, Any], reviewed_metrics),
    )
    adjudication_record["automated_metrics_field"] = "metrics"
    adjudication_record["reviewed_metrics_field"] = reviewed_metrics_field
    snapshot["human_review"] = adjudication_record
    if review_status == "complete_human_adjudication":
        review_limitation = (
            "Every run variant received direct human review. The automated 'metrics' block is "
            "retained for comparison; metrics_human_reviewed contains the reviewed task verdicts."
        )
    else:
        review_limitation = (
            "Human adjudication covered "
            f"{adjudication_record['adjudicated_variant_count']} of "
            f"{adjudication_record['total_variant_count']} variants. Unreviewed variants keep "
            "their automated verdict in metrics_with_reviewed_overrides; the primary 'metrics' "
            "block and per-case outcomes remain the automated baseline."
        )
    snapshot["limitations"] = [
        *cast(Sequence[str], snapshot["limitations"]),
        review_limitation,
    ]
    return snapshot


def render_project_generation_svg(snapshot: Mapping[str, Any]) -> str:
    """Render automated metrics unless the entire run received direct human review."""

    human_metrics = snapshot.get("metrics_human_reviewed")
    metrics = cast(
        Mapping[str, Mapping[str, Any]],
        human_metrics if isinstance(human_metrics, Mapping) else snapshot["metrics"],
    )
    human_review = snapshot.get("human_review")
    if isinstance(human_metrics, Mapping):
        heading = "Human-reviewed RAG generation evaluation"
        review_note = "complete direct human review"
        review_desc = "Every variant received direct human review."
    elif isinstance(human_review, Mapping):
        heading = "Automated RAG generation evaluation"
        reviewed = int(human_review["adjudicated_variant_count"])
        total = int(human_review["total_variant_count"])
        review_note = f"partial human review {reviewed}/{total}; headline remains automated"
        review_desc = "Partial human review is recorded; displayed metrics remain automated."
    else:
        heading = "Automated RAG generation evaluation"
        review_note = "human adjudication pending"
        review_desc = "Human adjudication is pending."
    configuration = cast(Mapping[str, Any], snapshot["configuration"])
    cluster = metrics["answerable_grounded_evidence_cluster"]
    panels = [
        (
            "Answerable grounded pass",
            float(cluster["macro_mean"]),
            cast(Sequence[float], cluster["bootstrap_95_ci"]),
            "45 evidence clusters",
            False,
        ),
        (
            "Sentinel compliance",
            float(metrics["unanswerable_sentinel_compliance"]["rate"]),
            cast(Sequence[float], metrics["unanswerable_sentinel_compliance"]["wilson_95_ci"]),
            "19 / 20 unanswerable",
            False,
        ),
        (
            "Injection joint pass",
            float(metrics["injection_joint_pass"]["rate"]),
            cast(Sequence[float], metrics["injection_joint_pass"]["wilson_95_ci"]),
            "4 / 10 attacked",
            False,
        ),
        (
            "Attack-induced failure",
            float(metrics["attack_induced_failure_among_clean_passes"]["rate"]),
            cast(
                Sequence[float],
                metrics["attack_induced_failure_among_clean_passes"]["wilson_95_ci"],
            ),
            "5 / 9 clean passes",
            True,
        ),
    ]
    colors = ("#2563eb", "#0f9f72", "#d97706", "#dc2626")
    width = 1200
    card_width = 262
    card_gap = 18
    left = 45
    card_y = 132
    card_height = 360
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="650" '
        'viewBox="0 0 1200 650" role="img" '
        'aria-labelledby="title description">',
        f'<title id="title">{html.escape(heading)}</title>',
        (
            '<desc id="description">Four interval plots show answerable grounded pass, sentinel '
            "compliance, prompt-injection joint pass, and attack-induced failure. "
            f"{html.escape(review_desc)}</desc>"
        ),
        '<rect width="1200" height="650" rx="18" fill="#f8fafc"/>',
        '<text x="45" y="55" font-family="Arial,sans-serif" font-size="30" '
        f'font-weight="700" fill="#0f172a">{html.escape(heading)}</text>',
        '<text x="45" y="88" font-family="Arial,sans-serif" font-size="15" fill="#475569">'
        f"100 clean cases · 10 attacked variants · {html.escape(str(snapshot['run_date']))} · "
        f"{html.escape(review_note)}</text>",
    ]
    for index, (label, value, interval, denominator, lower_is_better) in enumerate(panels):
        x = left + index * (card_width + card_gap)
        bar_left = x + 28
        bar_right = x + card_width - 28
        bar_width = bar_right - bar_left
        y = card_y + 190
        low, high = float(interval[0]), float(interval[1])
        low_x = bar_left + low * bar_width
        high_x = bar_left + high * bar_width
        value_x = bar_left + value * bar_width
        direction = "lower is better" if lower_is_better else "higher is better"
        elements.extend(
            [
                f'<rect x="{x}" y="{card_y}" width="{card_width}" height="{card_height}" '
                'rx="14" fill="#ffffff" stroke="#dbe3ee"/>',
                f'<text x="{x + 22}" y="{card_y + 42}" font-family="Arial,sans-serif" '
                f'font-size="18" font-weight="700" fill="#0f172a">{html.escape(label)}</text>',
                f'<text x="{x + 22}" y="{card_y + 82}" font-family="Arial,sans-serif" '
                f'font-size="38" font-weight="700" fill="{colors[index]}">{value:.1%}</text>',
                f'<text x="{x + 22}" y="{card_y + 110}" font-family="Arial,sans-serif" '
                f'font-size="14" fill="#475569">{html.escape(denominator)}</text>',
                f'<line x1="{bar_left}" y1="{y}" x2="{bar_right}" y2="{y}" '
                'stroke="#cbd5e1" stroke-width="4" stroke-linecap="round"/>',
                f'<line x1="{low_x:.1f}" y1="{y}" x2="{high_x:.1f}" y2="{y}" '
                f'stroke="{colors[index]}" stroke-width="8" stroke-linecap="round"/>',
                f'<circle cx="{value_x:.1f}" cy="{y}" r="8" fill="{colors[index]}" '
                'stroke="#ffffff" stroke-width="3"/>',
                f'<text x="{bar_left}" y="{y + 35}" font-family="Arial,sans-serif" '
                'font-size="12" fill="#64748b">0%</text>',
                f'<text x="{bar_right - 28}" y="{y + 35}" font-family="Arial,sans-serif" '
                'font-size="12" fill="#64748b">100%</text>',
                f'<text x="{x + 22}" y="{card_y + 285}" font-family="Arial,sans-serif" '
                f'font-size="13" fill="#64748b">95% interval · {direction}</text>',
            ]
        )
    elements.extend(
        [
            '<text x="45" y="545" font-family="Arial,sans-serif" font-size="15" '
            'font-weight="700" fill="#0f172a">Provisional automated baseline</text>',
            '<text x="45" y="574" font-family="Arial,sans-serif" font-size="14" '
            'fill="#475569">Candidate: '
            f"{html.escape(str(configuration['candidate_model']))} · primary judge: "
            f"{html.escape(str(configuration['judge_model']))} · BM25 top-"
            f"{int(configuration['k'])} · storage disabled</text>",
            '<text x="45" y="603" font-family="Arial,sans-serif" font-size="14" '
            'fill="#475569">10 answerable cases missed at least one cited locator; all 10 are '
            "multi-hop. Only 5 / 15 multi-hop cases had full coverage.</text>",
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def write_project_generation_report(
    summary_path: Path,
    case_results_path: Path,
    evaluation_directory: Path,
    snapshot_path: Path,
    figure_path: Path,
    *,
    run_date: str,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20_260_829,
    adjudication_path: Path | None = None,
) -> dict[str, object]:
    summary = _read_json(summary_path)
    rows = _read_jsonl(case_results_path)
    cases_path = evaluation_directory / "cases.jsonl"
    cases = load_project_evaluation_cases(cases_path)
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("generation summary provenance must be a JSON object")
    if provenance.get("cases_sha256") != _sha256(cases_path):
        raise ValueError("generation summary cases hash does not match the reviewed cases")
    fixtures_path = evaluation_directory / "injection-fixtures.jsonl"
    if provenance.get("fixtures_sha256") != _sha256(fixtures_path):
        raise ValueError("generation summary fixture hash does not match reviewed fixtures")
    fixtures = load_injection_fixtures(fixtures_path)
    adjudication = _read_json(adjudication_path) if adjudication_path is not None else None
    snapshot = build_project_generation_snapshot(
        summary,
        rows,
        cases,
        fixtures,
        run_date=run_date,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        raw_summary_sha256=_sha256(summary_path),
        raw_cases_sha256=_sha256(case_results_path),
        adjudication=adjudication,
        adjudication_sha256=(_sha256(adjudication_path) if adjudication_path is not None else None),
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.write_text(render_project_generation_svg(snapshot), encoding="utf-8")
    return snapshot

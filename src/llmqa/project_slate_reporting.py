"""Run-bound paired reporting for the preregistered BM25 slate-size probe."""

from __future__ import annotations

import hashlib
import html
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from llmqa.project_evaluation import ProjectEvaluationCase, load_project_evaluation_cases

BASELINE_ARM = "bm25-k10"
CANDIDATE_ARM = "bm25-k40"
REVIEW_STATUS = "complete_human_adjudication"
REVIEW_DESIGN = "human_approved_ai_assisted_paired_review"


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


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _string_sequence(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a JSON string array")
    return tuple(cast(list[str], value))


def _metric(successes: int, total: int) -> dict[str, int | float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("metric requires 0 <= successes <= positive total")
    return {"successes": successes, "total": total, "rate": successes / total}


def _latency(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("latency requires at least one observation")
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def slate_review_selection_sha256(keys: Sequence[tuple[str, str]]) -> str:
    """Hash the exact arm/case records selected for paired human review."""

    payload = [{"arm": arm, "case_id": case_id} for arm, case_id in sorted(set(keys))]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _mcnemar_exact(gains: int, losses: int) -> float:
    """Two-sided exact McNemar p-value under a Binomial(n, 0.5) null."""

    if gains < 0 or losses < 0:
        raise ValueError("McNemar discordant counts cannot be negative")
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = min(gains, losses)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1))
    return float(min(1.0, 2.0 * probability / (2**discordant)))


def _locator_hits(case: ProjectEvaluationCase, row: Mapping[str, Any]) -> tuple[bool, ...]:
    retrieved = set(_string_sequence(row.get("retrieved_chunk_ids"), label="retrieved_chunk_ids"))
    return tuple(bool(retrieved.intersection(locator.chunk_ids)) for locator in case.evidence)


def _validate_arm(
    name: str,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
    cases: Sequence[ProjectEvaluationCase],
) -> tuple[str, dict[str, Mapping[str, Any]]]:
    scope = _mapping(preregistration.get("scope"), label="preregistration scope")
    expected_ids = _string_sequence(scope.get("case_ids"), label="preregistered case IDs")
    case_ids = tuple(case.case_id for case in cases)
    if case_ids != expected_ids:
        raise ValueError("selected cases do not match the frozen preregistration order")
    if summary.get("dataset") != "technical-papers-v1":
        raise ValueError(f"{name} summary has the wrong dataset")
    if summary.get("complete") is not True or summary.get("limited_run") is not True:
        raise ValueError(f"{name} must be a complete, explicitly limited run")
    run_id = summary.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{name} summary is missing run_id")

    arm_by_name = {
        arm.get("name"): arm
        for raw_arm in cast(list[object], preregistration.get("arms", []))
        if isinstance(raw_arm, dict)
        for arm in [cast(Mapping[str, Any], raw_arm)]
    }
    if name not in arm_by_name:
        raise ValueError(f"{name} is absent from the preregistration")
    arm = arm_by_name[name]
    configuration = _mapping(summary.get("configuration"), label=f"{name} configuration")
    fixed = _mapping(preregistration.get("fixed_contract"), label="preregistration fixed_contract")
    common_fields = (
        "candidate_model",
        "judge_model",
        "candidate_max_output_tokens",
        "judge_max_output_tokens",
        "candidate_prompt_sha256",
        "judge_prompt_sha256",
        "judge_prompt_version",
        "claim_contract_version",
        "retriever",
        "bm25_k1",
        "bm25_b",
    )
    for field in common_fields:
        if configuration.get(field) != fixed.get(field):
            raise ValueError(f"{name} configuration field {field!r} drifted from preregistration")
    if configuration.get("k") != arm.get("k"):
        raise ValueError(f"{name} k does not match the preregistered arm")
    if tuple(configuration.get("selected_case_ids", [])) != expected_ids:
        raise ValueError(f"{name} selected case IDs drifted from preregistration")
    preflight = _mapping(configuration.get("budget_preflight"), label=f"{name} budget preflight")
    if preflight.get("preflight_id") != arm.get("preflight_id"):
        raise ValueError(f"{name} preflight ID does not match preregistration")
    if preflight.get("within_budget") is not True:
        raise ValueError(f"{name} preflight was not within budget")
    if float(preflight.get("max_cost_usd", -1.0)) != float(arm.get("max_cost_usd", -2.0)):
        raise ValueError(f"{name} cost cap does not match preregistration")

    provenance = _mapping(summary.get("provenance"), label=f"{name} provenance")
    for field in ("cases_sha256", "fixtures_sha256", "raw_chunks_sha256"):
        if provenance.get(field) != fixed.get(field):
            raise ValueError(f"{name} provenance field {field!r} drifted from preregistration")
    if preflight.get("pricing_contract_sha256") != fixed.get("pricing_contract_sha256"):
        raise ValueError(f"{name} pricing contract drifted from preregistration")

    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in rows_by_id:
            raise ValueError(f"{name} has a missing or duplicate case ID")
        if row.get("run_id") != run_id or row.get("variant") != "clean":
            raise ValueError(f"{name}/{case_id} is not a clean record from run {run_id}")
        if row.get("answerability") != "answerable":
            raise ValueError(f"{name}/{case_id} must be answerable")
        if len(_string_sequence(row.get("retrieved_chunk_ids"), label="retrieved IDs")) != int(
            arm["k"]
        ):
            raise ValueError(f"{name}/{case_id} does not contain the preregistered slate size")
        if row.get("candidate_model") != fixed.get("candidate_model"):
            raise ValueError(f"{name}/{case_id} candidate model drifted")
        judgment = _mapping(row.get("semantic_judgment"), label=f"{name}/{case_id} judgment")
        if judgment.get("judge_model") != fixed.get("judge_model"):
            raise ValueError(f"{name}/{case_id} judge model drifted")
        rows_by_id[case_id] = row
    if tuple(rows_by_id) != expected_ids:
        raise ValueError(f"{name} result rows do not exactly cover the preregistered cases")

    usage = _mapping(summary.get("usage"), label=f"{name} usage")
    token_fields = {
        "candidate_input_tokens": sum(int(row["candidate_input_tokens"]) for row in rows),
        "candidate_output_tokens": sum(int(row["candidate_output_tokens"]) for row in rows),
        "judge_input_tokens": sum(
            int(_mapping(row["semantic_judgment"], label="judgment")["input_tokens"])
            for row in rows
        ),
        "judge_output_tokens": sum(
            int(_mapping(row["semantic_judgment"], label="judgment")["output_tokens"])
            for row in rows
        ),
    }
    for field, expected in token_fields.items():
        if usage.get(field) != expected:
            raise ValueError(f"{name} summary usage field {field!r} does not reconcile")
    input_tokens = token_fields["candidate_input_tokens"] + token_fields["judge_input_tokens"]
    output_tokens = token_fields["candidate_output_tokens"] + token_fields["judge_output_tokens"]
    expected_cost = (
        input_tokens * float(preflight["input_per_million_usd"])
        + output_tokens * float(preflight["output_per_million_usd"])
    ) / 1_000_000
    observed_cost = float(usage.get("estimated_standard_token_cost_usd", -1.0))
    if not math.isclose(observed_cost, expected_cost, abs_tol=1e-12):
        raise ValueError(f"{name} standard-token cost does not reconcile")
    if observed_cost > float(preflight["max_cost_usd"]):
        raise ValueError(f"{name} exceeded its cost cap")
    return run_id, rows_by_id


def _validate_review(
    review: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    cases: Sequence[ProjectEvaluationCase],
    run_ids: Mapping[str, str],
    rows_by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
    hashes: Mapping[str, str],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    if review.get("schema_version") != 1 or review.get("status") != REVIEW_STATUS:
        raise ValueError("paired report requires a complete human-approved review artifact")
    if review.get("experiment_id") != preregistration.get("experiment_id"):
        raise ValueError("review experiment ID does not match preregistration")
    reviewer_ids = _string_sequence(review.get("reviewer_ids"), label="reviewer_ids")
    if not reviewer_ids or not isinstance(review.get("reviewed_at"), str):
        raise ValueError("complete review requires reviewer identity and timestamp")
    scope = _mapping(review.get("scope"), label="review scope")
    if scope.get("review_design") != REVIEW_DESIGN:
        raise ValueError("review design is not the required paired human-approval design")
    if scope.get("baseline_run_id") != run_ids[BASELINE_ARM]:
        raise ValueError("review is not bound to the baseline run")
    if scope.get("candidate_run_id") != run_ids[CANDIDATE_ARM]:
        raise ValueError("review is not bound to the candidate run")
    expected_keys = [(arm, case.case_id) for arm in (BASELINE_ARM, CANDIDATE_ARM) for case in cases]
    if scope.get("record_count") != len(expected_keys):
        raise ValueError("review record count does not match both arms")
    if scope.get("selected_records_sha256") != slate_review_selection_sha256(expected_keys):
        raise ValueError("review selection hash does not match both arms")
    provenance = _mapping(review.get("provenance"), label="review provenance")
    for field, expected_hash in hashes.items():
        if provenance.get(field) != expected_hash:
            raise ValueError(f"review provenance field {field!r} does not match the raw artifact")

    raw_decisions = review.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError("review decisions must be a JSON array")
    decisions: dict[tuple[str, str], Mapping[str, Any]] = {}
    case_by_id = {case.case_id: case for case in cases}
    for raw in raw_decisions:
        decision = _mapping(raw, label="review decision")
        arm = decision.get("arm")
        case_id = decision.get("case_id")
        if not isinstance(arm, str) or not isinstance(case_id, str):
            raise ValueError("review decision is missing arm or case_id")
        key = (arm, case_id)
        if key in decisions or key not in set(expected_keys):
            raise ValueError(f"review contains a duplicate or unexpected decision {key!r}")
        if decision.get("run_id") != run_ids[arm]:
            raise ValueError(f"review decision {key!r} is bound to the wrong run")
        verdict = decision.get("decision")
        if verdict not in {"pass", "fail"}:
            raise ValueError(f"review decision {key!r} has an invalid verdict")
        if not isinstance(decision.get("rationale"), str) or not decision["rationale"]:
            raise ValueError(f"review decision {key!r} requires a rationale")
        finding = _mapping(decision.get("review"), label=f"review finding {key!r}")
        case = case_by_id[case_id]
        satisfied = _string_sequence(
            finding.get("required_claims_satisfied"), label=f"{key!r} satisfied claims"
        )
        missing = _string_sequence(
            finding.get("required_claims_missing"), label=f"{key!r} missing claims"
        )
        if len(set(satisfied)) != len(satisfied) or len(set(missing)) != len(missing):
            raise ValueError(f"review decision {key!r} repeats a required claim")
        if set(satisfied).intersection(missing) or set(satisfied).union(missing) != set(
            case.required_claims
        ):
            raise ValueError(f"review decision {key!r} does not partition required claims")
        boolean_fields = (
            "answer_correct",
            "fully_supported",
            "contradiction_detected",
            "citation_faithful",
            "full_evidence_locator_coverage",
        )
        if any(not isinstance(finding.get(field), bool) for field in boolean_fields):
            raise ValueError(f"review decision {key!r} is missing required booleans")
        if finding["answer_correct"] is not (not missing):
            raise ValueError(
                f"review decision {key!r} answer_correct conflicts with required claims"
            )
        row = rows_by_arm[arm][case_id]
        expected_coverage = all(_locator_hits(case, row))
        if finding["full_evidence_locator_coverage"] is not expected_coverage:
            raise ValueError(f"review decision {key!r} has incorrect locator coverage")
        expected_verdict = bool(
            finding["answer_correct"]
            and finding["fully_supported"]
            and not finding["contradiction_detected"]
            and finding["citation_faithful"]
            and row.get("citations_valid") is True
        )
        if (verdict == "pass") is not expected_verdict:
            raise ValueError(f"review decision {key!r} is inconsistent with its findings")
        decisions[key] = decision
    if set(decisions) != set(expected_keys):
        raise ValueError("review must cover all 30 preregistered arm-by-case records")
    return decisions


def _arm_metrics(
    arm: str,
    summary: Mapping[str, Any],
    cases: Sequence[ProjectEvaluationCase],
    rows: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, object]:
    human_passes = sum(decisions[(arm, case.case_id)]["decision"] == "pass" for case in cases)
    automated_passes = sum(bool(rows[case.case_id]["task_pass"]) for case in cases)
    citation_valid = sum(bool(rows[case.case_id]["citations_valid"]) for case in cases)
    full_coverage = sum(all(_locator_hits(case, rows[case.case_id])) for case in cases)
    supported = 0
    contradictions = 0
    faithful = 0
    for case in cases:
        finding = _mapping(decisions[(arm, case.case_id)]["review"], label="review")
        supported += bool(finding["fully_supported"])
        contradictions += bool(finding["contradiction_detected"])
        faithful += bool(finding["citation_faithful"])
    usage = _mapping(summary["usage"], label=f"{arm} usage")
    total_latency = [
        float(rows[case.case_id]["candidate_latency_ms"])
        + float(_mapping(rows[case.case_id]["semantic_judgment"], label="judgment")["latency_ms"])
        for case in cases
    ]
    return {
        "arm": arm,
        "k": int(_mapping(summary["configuration"], label="configuration")["k"]),
        "run_id": summary["run_id"],
        "human_task_pass": _metric(human_passes, len(cases)),
        "automated_task_pass": _metric(automated_passes, len(cases)),
        "citation_validity": _metric(citation_valid, len(cases)),
        "human_fully_supported": _metric(supported, len(cases)),
        "human_citation_faithful": _metric(faithful, len(cases)),
        "human_contradiction_count": contradictions,
        "full_evidence_locator_coverage": _metric(full_coverage, len(cases)),
        "usage": {
            "candidate_input_tokens": usage["candidate_input_tokens"],
            "candidate_output_tokens": usage["candidate_output_tokens"],
            "judge_input_tokens": usage["judge_input_tokens"],
            "judge_output_tokens": usage["judge_output_tokens"],
            "total_tokens": int(usage["candidate_total_tokens"]) + int(usage["judge_total_tokens"]),
            "candidate_latency_ms": usage["candidate_latency_ms"],
            "judge_latency_ms": usage["judge_latency_ms"],
            "end_to_end_latency_ms": _latency(total_latency),
            "estimated_standard_token_cost_usd": usage["estimated_standard_token_cost_usd"],
            "exact_billed_cost_usd": None,
        },
    }


def build_slate_size_snapshot(
    preregistration: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_summary: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    cases: Sequence[ProjectEvaluationCase],
    review: Mapping[str, Any],
    *,
    hashes: Mapping[str, str],
    run_date: str,
) -> dict[str, object]:
    """Build a reviewed paired snapshot, refusing stale or incomplete review state."""

    if preregistration.get("schema_version") != 1:
        raise ValueError("unsupported slate-size preregistration schema")
    if preregistration.get("status") != "frozen_before_provider_calls":
        raise ValueError("slate-size experiment was not frozen before provider calls")
    baseline_run_id, baseline_by_id = _validate_arm(
        BASELINE_ARM, baseline_summary, baseline_rows, preregistration, cases
    )
    candidate_run_id, candidate_by_id = _validate_arm(
        CANDIDATE_ARM, candidate_summary, candidate_rows, preregistration, cases
    )
    if baseline_run_id == candidate_run_id:
        raise ValueError("paired arms must have distinct run IDs")
    run_ids = {BASELINE_ARM: baseline_run_id, CANDIDATE_ARM: candidate_run_id}
    rows_by_arm = {BASELINE_ARM: baseline_by_id, CANDIDATE_ARM: candidate_by_id}
    decisions = _validate_review(
        review,
        preregistration=preregistration,
        cases=cases,
        run_ids=run_ids,
        rows_by_arm=rows_by_arm,
        hashes=hashes,
    )
    baseline_metrics = _arm_metrics(
        BASELINE_ARM, baseline_summary, cases, baseline_by_id, decisions
    )
    candidate_metrics = _arm_metrics(
        CANDIDATE_ARM, candidate_summary, cases, candidate_by_id, decisions
    )

    both_pass: list[str] = []
    baseline_only: list[str] = []
    candidate_only: list[str] = []
    both_fail: list[str] = []
    for case in cases:
        baseline_pass = decisions[(BASELINE_ARM, case.case_id)]["decision"] == "pass"
        candidate_pass = decisions[(CANDIDATE_ARM, case.case_id)]["decision"] == "pass"
        target = (
            both_pass
            if baseline_pass and candidate_pass
            else baseline_only
            if baseline_pass
            else candidate_only
            if candidate_pass
            else both_fail
        )
        target.append(case.case_id)
    gains = len(candidate_only)
    losses = len(baseline_only)
    baseline_rate = cast(Mapping[str, Any], baseline_metrics["human_task_pass"])["rate"]
    candidate_rate = cast(Mapping[str, Any], candidate_metrics["human_task_pass"])["rate"]

    baseline_cost = float(
        cast(Mapping[str, Any], baseline_metrics["usage"])["estimated_standard_token_cost_usd"]
    )
    candidate_cost = float(
        cast(Mapping[str, Any], candidate_metrics["usage"])["estimated_standard_token_cost_usd"]
    )
    incremental_cost = candidate_cost - baseline_cost
    exploratory = _mapping(
        _mapping(preregistration["endpoints"], label="endpoints").get("exploratory_slice"),
        label="exploratory slice",
    )
    coverage_gain_ids = _string_sequence(
        exploratory.get("coverage_gain_case_ids"), label="coverage gain case IDs"
    )
    coverage_gain_task_gains = [
        case_id for case_id in candidate_only if case_id in coverage_gain_ids
    ]

    baseline_supported = cast(Mapping[str, Any], baseline_metrics["human_fully_supported"])
    candidate_supported = cast(Mapping[str, Any], candidate_metrics["human_fully_supported"])
    baseline_citations = cast(Mapping[str, Any], baseline_metrics["citation_validity"])
    candidate_citations = cast(Mapping[str, Any], candidate_metrics["citation_validity"])
    prereg_ceiling = float(preregistration["combined_preflight_cost_upper_bound_usd"])
    combined_cap = float(preregistration["combined_max_cost_usd"])
    gates = {
        "more_human_approved_gains_than_losses": gains > losses,
        "citation_valid_count_did_not_decrease": candidate_citations["successes"]
        >= baseline_citations["successes"],
        "unsupported_or_contradictory_answers_did_not_increase": (
            cast(int, candidate_supported["successes"])
            >= cast(int, baseline_supported["successes"])
            and cast(int, candidate_metrics["human_contradiction_count"])
            <= cast(int, baseline_metrics["human_contradiction_count"])
        ),
        "cost_bounds_respected": (
            baseline_cost
            <= float(
                _mapping(
                    _mapping(baseline_summary["configuration"], label="configuration")[
                        "budget_preflight"
                    ],
                    label="preflight",
                )["max_cost_usd"]
            )
            and candidate_cost
            <= float(
                _mapping(
                    _mapping(candidate_summary["configuration"], label="configuration")[
                        "budget_preflight"
                    ],
                    label="preflight",
                )["max_cost_usd"]
            )
            and prereg_ceiling <= combined_cap
        ),
    }
    gate_passed = all(gates.values())

    return {
        "schema_version": 1,
        "status": "human_reviewed_screening_probe_complete",
        "run_date": run_date,
        "experiment_id": preregistration["experiment_id"],
        "dataset": {
            "name": "technical-papers-v1",
            "source_document_count": 2,
            "multi_hop_case_count": len(cases),
            "variants": ["clean"],
        },
        "review": {
            "status": review["status"],
            "review_design": REVIEW_DESIGN,
            "reviewer_ids": review["reviewer_ids"],
            "reviewed_at": review["reviewed_at"],
            "record_count": len(decisions),
        },
        "arms": [baseline_metrics, candidate_metrics],
        "paired_analysis": {
            "both_pass": {"count": len(both_pass), "case_ids": both_pass},
            "baseline_only_pass": {"count": losses, "case_ids": baseline_only},
            "candidate_only_pass": {"count": gains, "case_ids": candidate_only},
            "both_fail": {"count": len(both_fail), "case_ids": both_fail},
            "absolute_rate_difference": float(candidate_rate) - float(baseline_rate),
            "mcnemar_exact_two_sided_p": _mcnemar_exact(gains, losses),
            "interpretation": (
                "Screening evidence only: the paired sample has 15 cases and is not a "
                "definitive population comparison."
            ),
        },
        "retrieval_mechanism_slice": {
            "preregistered_coverage_gain_case_ids": list(coverage_gain_ids),
            "human_task_gain_case_ids": coverage_gain_task_gains,
            "human_task_gain_count": len(coverage_gain_task_gains),
            "interpretation": exploratory["interpretation"],
        },
        "cost": {
            "baseline_estimated_standard_token_cost_usd": baseline_cost,
            "candidate_estimated_standard_token_cost_usd": candidate_cost,
            "incremental_estimated_cost_usd": incremental_cost,
            "cost_per_additional_human_pass_usd": (
                incremental_cost / (gains - losses) if gains > losses else None
            ),
            "combined_estimated_standard_token_cost_usd": baseline_cost + candidate_cost,
            "combined_preflight_upper_bound_usd": prereg_ceiling,
            "combined_max_cost_usd": combined_cap,
            "exact_billed_cost_usd": None,
        },
        "advance_gate": {
            "passed": gate_passed,
            "criteria": gates,
            "next_step": (
                "Preregister a confirmation including single-hop and unanswerable cases."
                if gate_passed
                else "Keep k=10 as default and record the negative mechanism probe."
            ),
        },
        "provenance": dict(hashes),
        "limitations": [
            "Only 15 multi-hop questions over two technical papers were tested.",
            (
                "The candidate and automated judge used the same pinned model snapshot; "
                "human-approved review determines the reported task verdicts."
            ),
            "The experiment is a screening probe, not a powered confirmation study.",
            (
                "Provider billing data is unavailable; dollar values are conservative "
                "standard-token estimates that ignore cached-input discounts."
            ),
            (
                "Exact request interleaving was service-dependent despite contemporaneous "
                "one-worker arm execution."
            ),
        ],
    }


def render_slate_size_svg(snapshot: Mapping[str, Any]) -> str:
    """Render a compact README-ready figure from a reviewed slate-size snapshot."""

    arms = cast(list[Mapping[str, Any]], snapshot["arms"])
    if len(arms) != 2:
        raise ValueError("slate-size figure requires exactly two arms")
    baseline, candidate = arms
    paired = _mapping(snapshot["paired_analysis"], label="paired analysis")
    cost = _mapping(snapshot["cost"], label="cost")
    baseline_pass = _mapping(baseline["human_task_pass"], label="baseline pass")
    candidate_pass = _mapping(candidate["human_task_pass"], label="candidate pass")
    baseline_coverage = _mapping(
        baseline["full_evidence_locator_coverage"], label="baseline coverage"
    )
    candidate_coverage = _mapping(
        candidate["full_evidence_locator_coverage"], label="candidate coverage"
    )
    baseline_rate = float(baseline_pass["rate"])
    candidate_rate = float(candidate_pass["rate"])
    max_width = 390.0
    baseline_width = max_width * baseline_rate
    candidate_width = max_width * candidate_rate
    gains = int(_mapping(paired["candidate_only_pass"], label="gains")["count"])
    losses = int(_mapping(paired["baseline_only_pass"], label="losses")["count"])
    p_value = float(paired["mcnemar_exact_two_sided_p"])
    incremental_cost = float(cost["incremental_estimated_cost_usd"])
    title = html.escape("BM25 slate-size screening probe")
    return "\n".join(
        [
            (
                '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="430" '
                'viewBox="0 0 960 430" role="img" aria-labelledby="title desc">'
            ),
            f'<title id="title">{title}</title>',
            (
                '<desc id="desc">Human-reviewed multi-hop task pass and retrieval coverage '
                "for BM25 top 10 versus top 40 chunks.</desc>"
            ),
            '<rect width="960" height="430" rx="24" fill="#f8fafc"/>',
            (
                '<text x="52" y="58" fill="#0f172a" '
                'font-family="Inter,Arial,sans-serif" font-size="27" font-weight="700">'
                "BM25 slate-size screening probe</text>"
            ),
            (
                '<text x="52" y="88" fill="#475569" '
                'font-family="Inter,Arial,sans-serif" font-size="15">'
                "15 human-reviewed multi-hop questions · two-paper corpus · "
                "frozen gpt-5-mini snapshot</text>"
            ),
            (
                '<text x="52" y="132" fill="#334155" '
                'font-family="Inter,Arial,sans-serif" font-size="15" font-weight="650">'
                "Human-reviewed task pass</text>"
            ),
            (
                '<text x="52" y="175" fill="#0f172a" '
                'font-family="Inter,Arial,sans-serif" font-size="15">BM25 top 10</text>'
            ),
            '<rect x="178" y="155" width="390" height="27" rx="7" fill="#e2e8f0"/>',
            (
                f'<rect x="178" y="155" width="{baseline_width:.1f}" height="27" '
                'rx="7" fill="#64748b"/>'
            ),
            (
                '<text x="588" y="175" fill="#0f172a" '
                'font-family="Inter,Arial,sans-serif" font-size="16" font-weight="700">'
                f"{baseline_pass['successes']}/{baseline_pass['total']} "
                f"({baseline_rate:.1%})</text>"
            ),
            (
                '<text x="52" y="224" fill="#0f172a" '
                'font-family="Inter,Arial,sans-serif" font-size="15">BM25 top 40</text>'
            ),
            '<rect x="178" y="204" width="390" height="27" rx="7" fill="#e2e8f0"/>',
            (
                f'<rect x="178" y="204" width="{candidate_width:.1f}" height="27" '
                'rx="7" fill="#0f766e"/>'
            ),
            (
                '<text x="588" y="224" fill="#0f172a" '
                'font-family="Inter,Arial,sans-serif" font-size="16" font-weight="700">'
                f"{candidate_pass['successes']}/{candidate_pass['total']} "
                f"({candidate_rate:.1%})</text>"
            ),
            (
                '<rect x="52" y="265" width="856" height="92" rx="14" '
                'fill="#ecfeff" stroke="#99f6e4"/>'
            ),
            (
                '<text x="76" y="296" fill="#115e59" '
                'font-family="Inter,Arial,sans-serif" font-size="17" font-weight="700">'
                f"Paired gains {gains} · losses {losses} · McNemar exact p = "
                f"{p_value:.3f}</text>"
            ),
            (
                '<text x="76" y="325" fill="#334155" '
                'font-family="Inter,Arial,sans-serif" font-size="14">'
                f"Full locator coverage: {baseline_coverage['successes']}/"
                f"{baseline_coverage['total']} → {candidate_coverage['successes']}/"
                f"{candidate_coverage['total']} · incremental estimated API cost "
                f"${incremental_cost:.3f}</text>"
            ),
            (
                '<text x="52" y="393" fill="#64748b" '
                'font-family="Inter,Arial,sans-serif" font-size="13">'
                "Screening evidence only. Human approval is run-bound; exact provider billing "
                "is unavailable.</text>"
            ),
            "</svg>",
        ]
    )


def write_slate_size_report(
    preregistration_path: Path,
    baseline_summary_path: Path,
    baseline_results_path: Path,
    candidate_summary_path: Path,
    candidate_results_path: Path,
    eval_dir: Path,
    review_path: Path,
    snapshot_path: Path,
    figure_path: Path,
    *,
    run_date: str,
) -> dict[str, object]:
    """Write public paired evidence only after exact run-bound human approval."""

    preregistration = _read_json(preregistration_path)
    baseline_summary = _read_json(baseline_summary_path)
    candidate_summary = _read_json(candidate_summary_path)
    baseline_rows = _read_jsonl(baseline_results_path)
    candidate_rows = _read_jsonl(candidate_results_path)
    review = _read_json(review_path)
    all_cases = load_project_evaluation_cases(eval_dir / "cases.jsonl")
    case_by_id = {case.case_id: case for case in all_cases}
    scope = _mapping(preregistration["scope"], label="preregistration scope")
    case_ids = _string_sequence(scope["case_ids"], label="preregistered case IDs")
    cases = tuple(case_by_id[case_id] for case_id in case_ids)
    hashes = {
        "preregistration_sha256": _sha256(preregistration_path),
        "cases_sha256": _sha256(eval_dir / "cases.jsonl"),
        "baseline_summary_sha256": _sha256(baseline_summary_path),
        "baseline_results_sha256": _sha256(baseline_results_path),
        "candidate_summary_sha256": _sha256(candidate_summary_path),
        "candidate_results_sha256": _sha256(candidate_results_path),
        "review_sha256": _sha256(review_path),
    }
    review_hashes = {key: value for key, value in hashes.items() if key != "review_sha256"}
    snapshot = build_slate_size_snapshot(
        preregistration,
        baseline_summary,
        baseline_rows,
        candidate_summary,
        candidate_rows,
        cases,
        review,
        hashes=review_hashes,
        run_date=run_date,
    )
    cast(dict[str, object], snapshot["provenance"])["review_sha256"] = hashes["review_sha256"]
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.write_text(render_slate_size_svg(snapshot) + "\n", encoding="utf-8")
    return snapshot

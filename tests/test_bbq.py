from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import llmqa.bbq as bbq_module
from llmqa.bbq import (
    BBQCase,
    build_bbq_subset_manifest,
    fetch_bbq,
    load_bbq,
    load_frozen_bbq_subset,
    select_bbq_subset,
    write_bbq_subset_manifest,
)
from llmqa.bbq_evaluation import (
    BBQ_ARMS,
    build_bbq_preflight,
    render_bbq_input,
    run_bbq_evaluation,
    write_bbq_preflight,
)
from llmqa.bbq_reporting import build_bbq_diagnostic, write_bbq_diagnostic_report
from llmqa.bbq_review import (
    BBQ_REVIEW_APPROVED_STATUS,
    BBQ_REVIEW_AUDIT_RULE,
    BBQ_REVIEW_DESIGN,
    BBQ_REVIEW_DRAFT_STATUS,
    bbq_review_attestation,
    bbq_review_selection_sha256,
    build_bbq_public_snapshot,
    render_bbq_public_svg,
    validate_bbq_review,
    write_bbq_publication,
)
from llmqa.cli import main


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _row(
    example_id: int,
    *,
    condition: str,
    polarity: str,
    label: int,
    question_index: str,
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "question_index": question_index,
        "question_polarity": polarity,
        "context_condition": condition,
        "category": "Toy",
        "answer_info": {
            "ans0": ["group a", "target"],
            "ans1": ["group b", "non-target"],
            "ans2": ["Can't be determined", "unknown"],
        },
        "additional_metadata": {},
        "context": f"Context {example_id}",
        "question": f"Question {example_id}?",
        "ans0": "Group A",
        "ans1": "Group B",
        "ans2": "Can't be determined",
        "label": label,
    }


def _toy_rows() -> list[dict[str, Any]]:
    return [
        _row(0, condition="ambig", polarity="neg", label=2, question_index="a-neg"),
        _row(1, condition="ambig", polarity="nonneg", label=2, question_index="a-pos"),
        _row(2, condition="disambig", polarity="neg", label=0, question_index="d-na"),
        _row(3, condition="disambig", polarity="neg", label=1, question_index="d-nc"),
        _row(4, condition="disambig", polarity="nonneg", label=1, question_index="d-pa"),
        _row(5, condition="disambig", polarity="nonneg", label=0, question_index="d-pc"),
    ]


def _metadata_bytes(rows: list[dict[str, Any]]) -> bytes:
    import io

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["category", "question_index", "example_id", "target_loc", "label_type"],
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "category": "Toy",
                "question_index": row["question_index"],
                "example_id": row["example_id"],
                "target_loc": 0 if row["question_polarity"] == "neg" else 1,
                "label_type": "label",
            }
        )
    return output.getvalue().encode()


@pytest.fixture
def toy_bbq(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, tuple[BBQCase, ...]]:
    rows = _toy_rows()
    data = "".join(json.dumps(row) + "\n" for row in rows).encode()
    metadata = _metadata_bytes(rows)
    monkeypatch.setattr(bbq_module, "BBQ_DATA_FILES", {"Toy.jsonl": (6, _sha256(data))})
    monkeypatch.setattr(bbq_module, "BBQ_METADATA_SHA256", _sha256(metadata))
    monkeypatch.setattr(bbq_module, "BBQ_TOTAL_CASES", 6)
    monkeypatch.setattr(bbq_module, "BBQ_SCORABLE_CASES", 6)
    dataset = tmp_path / "bbq"
    dataset.mkdir()
    (dataset / "Toy.jsonl").write_bytes(data)
    (dataset / "additional_metadata.csv").write_bytes(metadata)
    bbq_module._write_manifest(dataset)
    return dataset, load_bbq(dataset)


def _write_pricing(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider": "openai",
                "service_tier": "standard",
                "currency": "USD",
                "verified_at": "2026-08-30",
                "source_url": "https://openai.com/api/pricing/",
                "models": {
                    "gpt-5-mini-2025-08-07": {
                        "input_per_million_usd": 0.25,
                        "cached_input_per_million_usd": 0.025,
                        "output_per_million_usd": 2.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


class _FakeResponses:
    def __init__(self, labels: dict[str, int]) -> None:
        self.labels = labels
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        payload = json.loads(str(kwargs["input"]))
        answer = self.labels[str(payload["question"])]
        return SimpleNamespace(
            id=f"response-{len(self.calls)}",
            model=str(kwargs["model"]),
            status="completed",
            output_text=json.dumps({"answer_index": answer}),
            usage=SimpleNamespace(input_tokens=20, output_tokens=3, total_tokens=23),
        )


class _FakeClient:
    def __init__(self, labels: dict[str, int]) -> None:
        self.responses = _FakeResponses(labels)


class _InvalidResponses:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs: object) -> object:
        self.calls += 1
        return SimpleNamespace(
            id="invalid-response",
            model=str(kwargs["model"]),
            status="completed",
            output_text="{}",
            usage=SimpleNamespace(input_tokens=20, output_tokens=1, total_tokens=21),
        )


class _InvalidClient:
    def __init__(self) -> None:
        self.responses = _InvalidResponses()


def test_load_select_freeze_and_bind_toy_dataset(
    toy_bbq: tuple[Path, tuple[BBQCase, ...]], tmp_path: Path
) -> None:
    dataset, cases = toy_bbq
    assert len(cases) == 6
    assert cases[0].unknown_index == 2
    assert cases[0].is_stereotype_choice(0)
    assert not cases[1].is_stereotype_choice(0)
    assert cases[1].is_stereotype_choice(1)
    assert not cases[1].is_stereotype_choice(2)
    with pytest.raises(ValueError, match="answer index"):
        cases[0].is_stereotype_choice(4)

    selection = select_bbq_subset(cases, sample_per_stratum=1, seed="fixed")
    assert len(selection.cases) == 6
    assert selection.template_count == 6
    assert set(selection.stratum_counts.values()) == {1}
    assert select_bbq_subset(cases, sample_per_stratum=1, seed="fixed") == selection
    with pytest.raises(ValueError, match="only 1"):
        select_bbq_subset(cases, sample_per_stratum=2)
    with pytest.raises(ValueError, match="positive"):
        select_bbq_subset(cases, sample_per_stratum=0)
    with pytest.raises(ValueError, match="non-empty"):
        select_bbq_subset(cases, seed="")

    subset_path = tmp_path / "subset.json"
    subset = write_bbq_subset_manifest(
        cases,
        subset_path,
        sample_per_stratum=1,
        seed="fixed",
        frozen_at="2026-08-31T00:00:00Z",
    )
    bound_cases, loaded = load_frozen_bbq_subset(dataset, subset_path)
    assert [case.case_id for case in bound_cases] == [row["case_id"] for row in subset["cases"]]
    assert loaded == subset
    assert "not a full BBQ score" in subset["diagnostic_scope"]
    with pytest.raises(ValueError, match="frozen_at"):
        build_bbq_subset_manifest(cases, frozen_at="")


def test_manifest_and_subset_tampering_fail_closed(
    toy_bbq: tuple[Path, tuple[BBQCase, ...]], tmp_path: Path
) -> None:
    dataset, cases = toy_bbq
    subset_path = tmp_path / "subset.json"
    write_bbq_subset_manifest(cases, subset_path, sample_per_stratum=1, frozen_at="2026-08-31")
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    subset["cases"][0]["stratum"] = "tampered"
    subset_path.write_text(json.dumps(subset), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata drifted"):
        load_frozen_bbq_subset(dataset, subset_path)

    manifest_path = dataset / "benchmark-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "moving-main"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned dataset contract"):
        load_bbq(dataset)


def test_fetch_is_atomic_reusable_and_integrity_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _toy_rows()
    data = "".join(json.dumps(row) + "\n" for row in rows).encode()
    metadata = _metadata_bytes(rows)
    monkeypatch.setattr(bbq_module, "BBQ_DATA_FILES", {"Toy.jsonl": (6, _sha256(data))})
    monkeypatch.setattr(bbq_module, "BBQ_METADATA_SHA256", _sha256(metadata))
    monkeypatch.setattr(bbq_module, "BBQ_TOTAL_CASES", 6)
    monkeypatch.setattr(bbq_module, "BBQ_SCORABLE_CASES", 6)

    def fake_download(url: str, destination: Path) -> None:
        destination.write_bytes(metadata if url.endswith("additional_metadata.csv") else data)

    monkeypatch.setattr(bbq_module, "_download", fake_download)
    dataset = fetch_bbq(tmp_path)
    assert fetch_bbq(tmp_path) == dataset
    assert len(load_bbq(dataset)) == 6
    (dataset / "Toy.jsonl").write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity verification"):
        fetch_bbq(tmp_path)


def test_preflight_runner_resume_and_report(
    toy_bbq: tuple[Path, tuple[BBQCase, ...]], tmp_path: Path
) -> None:
    dataset, cases = toy_bbq
    subset_path = tmp_path / "subset.json"
    write_bbq_subset_manifest(
        cases,
        subset_path,
        sample_per_stratum=1,
        frozen_at="2026-08-31T00:00:00Z",
    )
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    preflight_path = tmp_path / "preflight.json"
    preflight = write_bbq_preflight(
        preflight_path,
        dataset,
        subset_path,
        pricing_path,
        model="gpt-5-mini-2025-08-07",
        max_cost_usd=0.50,
        max_output_tokens=64,
    )
    assert preflight["status"] == "within_budget"
    assert preflight["execution"]["request_count"] == 12
    assert preflight["execution"]["sdk_max_retries"] == 0
    assert preflight["execution"]["reasoning_effort"] == "minimal"
    assert len(preflight["artifacts"]["evaluation_source_sha256"]) == 64
    assert len(preflight["artifacts"]["dataset_adapter_source_sha256"]) == 64
    assert preflight["paid_execution_authorized"] is False
    assert (
        build_bbq_preflight(
            dataset,
            subset_path,
            pricing_path,
            model="gpt-5-mini-2025-08-07",
            max_cost_usd=0.50,
            max_output_tokens=64,
        )
        == preflight
    )

    output = tmp_path / "run"
    with pytest.raises(ValueError, match="authorize_paid_run"):
        run_bbq_evaluation(
            dataset,
            subset_path,
            output,
            preflight_path,
            pricing_path,
            model="gpt-5-mini-2025-08-07",
            max_cost_usd=0.50,
            max_output_tokens=64,
        )
    labels = {case.question: case.label for case in cases}
    client = _FakeClient(labels)
    summary_path = run_bbq_evaluation(
        dataset,
        subset_path,
        output,
        preflight_path,
        pricing_path,
        model="gpt-5-mini-2025-08-07",
        max_cost_usd=0.50,
        max_output_tokens=64,
        client=client,
    )
    assert len(client.responses.calls) == 12
    assert all(call["store"] is False for call in client.responses.calls)
    assert all(call["reasoning"] == {"effort": "minimal"} for call in client.responses.calls)
    assert json.loads(summary_path.read_text(encoding="utf-8"))["status"] == "complete"
    run_bbq_evaluation(
        dataset,
        subset_path,
        output,
        preflight_path,
        pricing_path,
        model="gpt-5-mini-2025-08-07",
        max_cost_usd=0.50,
        max_output_tokens=64,
        client=client,
    )
    assert len(client.responses.calls) == 12

    report_path = tmp_path / "report.json"
    report = write_bbq_diagnostic_report(
        dataset,
        subset_path,
        output,
        report_path,
        bootstrap_resamples=100,
        bootstrap_seed=7,
    )
    assert report["arms"]["neutral"]["overall"]["accuracy"] == 1.0
    assert report["paired"]["template_clustered_ci"]["accuracy_delta"]["estimate"] == 0
    assert report["paired"]["exact_tests"]["overall_accuracy"]["discordant_count"] == 0
    assert report["run_observability"]["complete_request_count"] == 12
    assert report["run_observability"]["failed_request_count"] == 0
    assert report["provenance"]["run_id"]
    assert report_path.is_file()


def test_preflight_blocks_drift_and_over_budget(
    toy_bbq: tuple[Path, tuple[BBQCase, ...]], tmp_path: Path
) -> None:
    dataset, cases = toy_bbq
    subset_path = tmp_path / "subset.json"
    write_bbq_subset_manifest(cases, subset_path, sample_per_stratum=1, frozen_at="date")
    pricing = tmp_path / "pricing.json"
    _write_pricing(pricing)
    preflight = tmp_path / "preflight.json"
    blocked = write_bbq_preflight(
        preflight,
        dataset,
        subset_path,
        pricing,
        model="gpt-5-mini-2025-08-07",
        max_cost_usd=0.000001,
        max_output_tokens=64,
    )
    assert blocked["status"] == "blocked_over_budget"
    with pytest.raises(ValueError, match="blocked over budget"):
        run_bbq_evaluation(
            dataset,
            subset_path,
            tmp_path / "run",
            preflight,
            pricing,
            model="gpt-5-mini-2025-08-07",
            max_cost_usd=0.000001,
            max_output_tokens=64,
            client=_FakeClient({case.question: case.label for case in cases}),
        )
    approved = write_bbq_preflight(
        preflight,
        dataset,
        subset_path,
        pricing,
        model="gpt-5-mini-2025-08-07",
        max_cost_usd=0.50,
        max_output_tokens=64,
    )
    approved["preflight_id"] = "tampered"
    preflight.write_text(json.dumps(approved), encoding="utf-8")
    with pytest.raises(ValueError, match="exact execution contract"):
        run_bbq_evaluation(
            dataset,
            subset_path,
            tmp_path / "run",
            preflight,
            pricing,
            model="gpt-5-mini-2025-08-07",
            max_cost_usd=0.50,
            max_output_tokens=64,
            client=_FakeClient({case.question: case.label for case in cases}),
        )


def test_invalid_provider_attempt_is_checkpointed_and_never_repaid(
    toy_bbq: tuple[Path, tuple[BBQCase, ...]], tmp_path: Path
) -> None:
    dataset, cases = toy_bbq
    subset = tmp_path / "subset.json"
    write_bbq_subset_manifest(cases, subset, sample_per_stratum=1, frozen_at="date")
    pricing = tmp_path / "pricing.json"
    _write_pricing(pricing)
    preflight = tmp_path / "preflight.json"
    write_bbq_preflight(
        preflight,
        dataset,
        subset,
        pricing,
        model="gpt-5-mini-2025-08-07",
        max_cost_usd=0.50,
        max_output_tokens=64,
    )
    client = _InvalidClient()
    arguments = (
        dataset,
        subset,
        tmp_path / "run",
        preflight,
        pricing,
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="only answer_index"):
            run_bbq_evaluation(
                *arguments,
                model="gpt-5-mini-2025-08-07",
                max_cost_usd=0.50,
                max_output_tokens=64,
                client=client,
            )
    assert client.responses.calls == 1
    attempts = (tmp_path / "run" / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(attempts) == 1


def test_diagnostic_bias_direction_and_validation(
    toy_bbq: tuple[Path, tuple[BBQCase, ...]],
) -> None:
    _, cases = toy_bbq
    rows: list[dict[str, object]] = []
    for case in cases:
        neutral_answer = 0 if case.question_polarity == "neg" else 1
        rows.append({"case_id": case.case_id, "arm": "neutral", "answer_index": neutral_answer})
        rows.append({"case_id": case.case_id, "arm": "grounded", "answer_index": case.label})
    report = build_bbq_diagnostic(cases, rows, bootstrap_resamples=100, bootstrap_seed=2)
    neutral_ambig = report["arms"]["neutral"]["by_condition"]["ambig"]
    grounded_ambig = report["arms"]["grounded"]["by_condition"]["ambig"]
    assert neutral_ambig["reported_bias_score"] == 1.0
    assert grounded_ambig["reported_bias_score"] is None
    assert report["paired"]["template_clustered_ci"]["accuracy_delta"]["estimate"] > 0
    assert report["paired"]["exact_tests"]["overall_accuracy"]["grounded_only"] > 0
    assert report["arms"]["neutral"]["overall"]["reported_bias_score"] is None
    assert len(report["per_case_outcomes"]) == len(cases)
    assert all(
        set(outcome).issuperset(
            {
                "case_id",
                "official_label",
                "neutral_answer_index",
                "grounded_answer_index",
                "neutral_correct",
                "grounded_correct",
            }
        )
        for outcome in report["per_case_outcomes"]
    )
    assert set(report["paired"]["answer_flip_counts"]) <= {
        f"{left}->{right}" for left in range(3) for right in range(3)
    }
    with pytest.raises(ValueError, match="at least 100"):
        build_bbq_diagnostic(cases, rows, bootstrap_resamples=99)
    with pytest.raises(ValueError, match="incomplete"):
        build_bbq_diagnostic(cases, rows[:-1], bootstrap_resamples=100)


def test_model_visible_input_excludes_gold_and_scoring_fields(
    toy_bbq: tuple[Path, tuple[BBQCase, ...]],
) -> None:
    _, cases = toy_bbq
    payload = json.loads(render_bbq_input(cases[0]))
    assert set(payload) == {"context", "question", "answers"}
    assert "label" not in payload
    assert "target_index" not in payload
    assert tuple(BBQ_ARMS) == ("neutral", "grounded")


def test_committed_subset_is_id_only_and_frozen_before_provider_calls() -> None:
    path = Path("evals/bias/bbq-v1/subset.json")
    subset = json.loads(path.read_text(encoding="utf-8"))
    assert subset["status"] == "frozen_before_provider_calls"
    assert subset["selected_case_count"] == 180
    assert subset["selected_template_count"] == 130
    assert len(subset["stratum_counts"]) == 90
    assert set(subset["stratum_counts"].values()) == {2}
    assert subset["selection_sha256"] == (
        "18cb2cca43661fa8681c890d6c317ee4e9f1b9e8b67991f53b8dcb9228e7adff"
    )
    forbidden = {"context", "question", "answers", "label", "target_index", "unknown_index"}
    assert all(not forbidden.intersection(record) for record in subset["cases"])


def _review_report() -> dict[str, Any]:
    outcomes = [
        {
            "case_id": "ambig-gain",
            "context_condition": "ambig",
            "neutral_correct": False,
            "grounded_correct": True,
            "neutral_answer_index": 1,
            "grounded_answer_index": 2,
            "unknown_index": 2,
        },
        {
            "case_id": "disambig-gain",
            "context_condition": "disambig",
            "neutral_correct": False,
            "grounded_correct": True,
            "neutral_answer_index": 2,
            "grounded_answer_index": 0,
            "unknown_index": 2,
        },
        {
            "case_id": "over-abstain",
            "context_condition": "disambig",
            "neutral_correct": True,
            "grounded_correct": False,
            "neutral_answer_index": 1,
            "grounded_answer_index": 2,
            "unknown_index": 2,
        },
        {
            "case_id": "annotation-error",
            "context_condition": "disambig",
            "neutral_correct": False,
            "grounded_correct": False,
            "neutral_answer_index": 1,
            "grounded_answer_index": 1,
            "unknown_index": 2,
        },
        {
            "case_id": "unsupported-guess",
            "context_condition": "ambig",
            "neutral_correct": False,
            "grounded_correct": False,
            "neutral_answer_index": 1,
            "grounded_answer_index": 1,
            "unknown_index": 2,
        },
        {
            "case_id": "both-correct",
            "context_condition": "disambig",
            "neutral_correct": True,
            "grounded_correct": True,
            "neutral_answer_index": 0,
            "grounded_answer_index": 0,
            "unknown_index": 2,
        },
    ]
    interval = {"estimate": 0.1, "ci95_low": 0.01, "ci95_high": 0.2, "valid_resamples": 100}
    return {
        "scope": "BBQ-derived subset; not a full BBQ score or retrieval-fairness measure",
        "case_count": len(outcomes),
        "template_count": 6,
        "provenance": {"run_id": "run-1", "results_sha256": "results"},
        "per_case_outcomes": outcomes,
        "arms": {
            "neutral": {
                "by_condition": {
                    "ambig": {"unknown_selection_rate": 0.0, "reported_bias_score": 0.3},
                    "disambig": {"accuracy": 0.5, "reported_bias_score": 0.1},
                }
            },
            "grounded": {
                "by_condition": {
                    "ambig": {"unknown_selection_rate": 0.5, "reported_bias_score": 0.0},
                    "disambig": {"accuracy": 0.5, "reported_bias_score": 0.1},
                }
            },
        },
        "paired": {
            "exact_tests": {
                "overall_accuracy": {
                    "grounded_only": 2,
                    "neutral_only": 1,
                    "mcnemar_exact_two_sided_p": 1.0,
                },
                "ambiguous_unknown_selection": {},
                "disambiguated_accuracy": {},
            },
            "template_clustered_ci": {
                "accuracy_delta": interval,
                "ambiguous_unknown_selection_rate_delta": interval,
                "disambiguated_accuracy_delta": interval,
                "ambiguous_bias_magnitude_delta": interval,
                "disambiguated_bias_magnitude_delta": interval,
            },
        },
        "run_observability": {"estimated_standard_cost_usd": 0.01},
    }


def _review_record(report: dict[str, Any], *, approved: bool) -> dict[str, Any]:
    decisions = [
        {
            "case_id": "ambig-gain",
            "automated_transition": "grounded_only_correct",
            "decision": "uphold_grounded_gain_unsupported_ambiguous_guess",
            "rationale": "The neutral arm guessed in an ambiguous context.",
        },
        {
            "case_id": "disambig-gain",
            "automated_transition": "grounded_only_correct",
            "decision": "uphold_grounded_gain_disambiguated_evidence",
            "rationale": "The grounded arm followed disambiguating evidence.",
        },
        {
            "case_id": "over-abstain",
            "automated_transition": "neutral_only_correct",
            "decision": "uphold_official_label_prompt_over_abstention",
            "rationale": "The grounded contract was stricter than the benchmark.",
        },
        {
            "case_id": "annotation-error",
            "automated_transition": "both_incorrect",
            "decision": "probable_official_annotation_error_sensitivity_only",
            "rationale": "Both arms selected the context-supported non-unknown answer.",
        },
        {
            "case_id": "unsupported-guess",
            "automated_transition": "both_incorrect",
            "decision": "uphold_grounded_failure_unsupported_guess",
            "rationale": "The grounded arm guessed despite ambiguous evidence.",
        },
    ]
    failure_ids = [str(item["case_id"]) for item in decisions]
    status = BBQ_REVIEW_APPROVED_STATUS if approved else BBQ_REVIEW_DRAFT_STATUS
    return {
        "schema_version": 1,
        "status": status,
        "scope": {
            "audit_rule": BBQ_REVIEW_AUDIT_RULE,
            "case_count": len(decisions),
            "review_design": BBQ_REVIEW_DESIGN,
            "reviewed_case_ids_sha256": bbq_review_selection_sha256(failure_ids),
        },
        "provenance": {**report["provenance"], "automated_report_sha256": "report"},
        "review_method": {
            "decision_preparation": "ai_pre_audit",
            "human_action": "explicit_approval_of_all_decision_records",
            "independent_blinded_panel": False,
        },
        "reviewer_ids": ["wei-han"] if approved else [],
        "reviewed_at": "2026-08-31T12:00:00-07:00" if approved else None,
        "human_attestation": (
            bbq_review_attestation("run-1", len(decisions)) if approved else None
        ),
        "decisions": decisions,
    }


def test_bbq_review_is_run_bound_complete_and_fail_closed(tmp_path: Path) -> None:
    report = _review_report()
    draft = _review_record(report, approved=False)
    decisions = validate_bbq_review(draft, report, report_sha256="report")
    assert len(decisions) == 5
    with pytest.raises(ValueError, match="explicit human approval"):
        build_bbq_public_snapshot(report, draft, report_sha256="report")

    approved = _review_record(report, approved=True)
    snapshot = build_bbq_public_snapshot(report, approved, report_sha256="report")
    assert snapshot["status"] == "human_reviewed_bbq_derived_subset_diagnostic"
    assert snapshot["annotation_sensitivity"]["official_primary_scores_retained"] is True
    assert snapshot["review"]["decision_count"] == 5
    svg = render_bbq_public_svg(snapshot)
    assert "Human-approved AI-assisted failure audit" in svg
    assert "not full BBQ or retrieval fairness" in svg

    tampered = json.loads(json.dumps(approved))
    tampered["provenance"]["results_sha256"] = "different"
    with pytest.raises(ValueError, match="exact automated report"):
        validate_bbq_review(tampered, report, report_sha256="report")

    incomplete = json.loads(json.dumps(approved))
    incomplete["decisions"].pop()
    with pytest.raises(ValueError, match="one decision for every audited case"):
        validate_bbq_review(incomplete, report, report_sha256="report")

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    approved["provenance"]["automated_report_sha256"] = _sha256(report_path.read_bytes())
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(approved), encoding="utf-8")
    snapshot_path = tmp_path / "snapshot.json"
    figure_path = tmp_path / "figure.svg"
    written = write_bbq_publication(report_path, review_path, snapshot_path, figure_path)
    assert snapshot_path.is_file()
    assert figure_path.is_file()
    assert written["provenance"]["human_review_sha256"] == _sha256(review_path.read_bytes())
    assert written["provenance"]["publication_source_sha256"] == _sha256(
        Path("src/llmqa/bbq_review.py").read_bytes()
    )


def test_committed_bbq_review_remains_explicitly_unapproved() -> None:
    review = json.loads(Path("evals/bias/bbq-v1/human-review.json").read_text(encoding="utf-8"))
    assert review["status"] == BBQ_REVIEW_DRAFT_STATUS
    assert review["scope"]["case_count"] == 30
    assert len(review["decisions"]) == 30
    assert review["reviewer_ids"] == []
    assert review["reviewed_at"] is None
    assert review["human_attestation"] is None


def test_bbq_cli_freezes_and_plans_without_provider(
    toy_bbq: tuple[Path, tuple[BBQCase, ...]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset, _ = toy_bbq
    subset = tmp_path / "subset.json"
    assert (
        main(
            [
                "freeze-bbq-subset",
                "--dataset-dir",
                str(dataset),
                "--output",
                str(subset),
                "--sample-per-stratum",
                "1",
                "--frozen-at",
                "2026-08-31",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["selected_case_count"] == 6
    pricing = tmp_path / "pricing.json"
    _write_pricing(pricing)
    preflight = tmp_path / "preflight.json"
    assert (
        main(
            [
                "evaluate-bbq",
                "--dataset-dir",
                str(dataset),
                "--subset",
                str(subset),
                "--preflight",
                str(preflight),
                "--pricing-contract",
                str(pricing),
                "--max-output-tokens",
                "64",
                "--plan-only",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["paid_execution_authorized"] is False

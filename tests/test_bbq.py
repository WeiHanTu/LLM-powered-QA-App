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
    assert report["arms"]["neutral"]["overall"]["reported_bias_score"] is None
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

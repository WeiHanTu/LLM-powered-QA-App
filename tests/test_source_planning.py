from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from llmqa.project_evaluation import load_project_evaluation_cases
from llmqa.source_planning import (
    generate_source_plan_artifact,
    load_source_catalog,
    load_source_plan_artifact,
    plan_question_sources,
)

ROOT = Path(__file__).parents[1]
EVAL_DIR = ROOT / "evals" / "project" / "technical-papers-v1"


@dataclass
class FakeUsage:
    input_tokens: int = 20
    output_tokens: int = 10
    total_tokens: int = 30


@dataclass
class FakeResponse:
    output_text: str = json.dumps(
        {
            "steps": [
                {
                    "source_id": "attention-is-all-you-need",
                    "query": "Attention Is All You Need architecture evidence",
                },
                {"source_id": "kimi-k3", "query": "Kimi K3 architecture evidence"},
            ]
        }
    )
    id: str = "resp_source_plan"
    model: str = "gpt-5-mini-resolved"
    status: str = "completed"
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        return FakeResponse()


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def test_source_plan_sends_only_question_and_catalog_and_disables_storage() -> None:
    case = next(
        case
        for case in load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")
        if case.case_id == "tp-062"
    )
    catalog = load_source_catalog(EVAL_DIR)
    client = FakeClient()

    result = plan_question_sources(case, catalog, model="gpt-5-mini", client=client)

    call = client.responses.calls[0]
    payload = json.loads(str(call["input"]))
    assert call["store"] is False
    assert payload == {
        "question": case.question,
        "sources": [{"source_id": entry.source_id, "title": entry.title} for entry in catalog],
    }
    assert case.expected_answer not in str(call)
    assert all(locator.section not in str(call) for locator in case.evidence)
    assert result.case_id == case.case_id
    assert result.response_model == "gpt-5-mini-resolved"
    assert result.total_tokens == 30
    text = call["text"]
    assert isinstance(text, dict)
    assert text["format"]["strict"] is True


def test_source_plan_artifact_round_trip_requires_current_questions_and_sources(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    output_path = tmp_path / "source-plans.json"

    generated = generate_source_plan_artifact(
        EVAL_DIR,
        output_path,
        model="gpt-5-mini",
        client=client,
        generated_at="2026-08-29T00:00:00+00:00",
    )
    loaded, mapping = load_source_plan_artifact(output_path, EVAL_DIR)

    assert generated.plan_count == 15
    assert loaded == generated
    assert len(client.responses.calls) == 15
    assert set(mapping) == {f"tp-{index:03d}" for index in range(61, 76)}

    raw = json.loads(output_path.read_text(encoding="utf-8"))
    raw["records"][0]["steps"][0]["source_id"] = "unknown-paper"
    output_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid steps"):
        load_source_plan_artifact(output_path, EVAL_DIR)

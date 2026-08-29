from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from llmqa.project_evaluation import load_project_evaluation_cases
from llmqa.query_decomposition import (
    decompose_question,
    generate_query_decomposition_artifact,
    load_query_decomposition_artifact,
)

ROOT = Path(__file__).parents[1]
EVAL_DIR = ROOT / "evals" / "project" / "technical-papers-v1"


@dataclass
class FakeUsage:
    input_tokens: int = 12
    output_tokens: int = 8
    total_tokens: int = 20


@dataclass
class FakeResponse:
    output_text: str = json.dumps(
        {"subqueries": ["Transformer architecture evidence", "Kimi K3 architecture evidence"]}
    )
    id: str = "resp_decomposition"
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


def test_decomposition_sends_question_only_and_disables_storage() -> None:
    case = next(
        case
        for case in load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")
        if case.case_id == "tp-062"
    )
    client = FakeClient()

    result = decompose_question(case, model="gpt-5-mini", client=client)

    call = client.responses.calls[0]
    assert call["store"] is False
    assert json.loads(str(call["input"])) == {"question": case.question}
    assert case.expected_answer not in str(call)
    assert all(locator.section not in str(call) for locator in case.evidence)
    assert result.case_id == case.case_id
    assert result.response_model == "gpt-5-mini-resolved"
    assert result.total_tokens == 20
    text = call["text"]
    assert isinstance(text, dict)
    assert text["format"]["strict"] is True


def test_artifact_round_trip_requires_all_current_multi_hop_questions(tmp_path: Path) -> None:
    client = FakeClient()
    output_path = tmp_path / "query-decompositions.json"

    generated = generate_query_decomposition_artifact(
        EVAL_DIR,
        output_path,
        model="gpt-5-mini",
        client=client,
        generated_at="2026-08-29T00:00:00+00:00",
    )
    loaded, mapping = load_query_decomposition_artifact(output_path, EVAL_DIR)

    assert generated.query_count == 15
    assert loaded == generated
    assert len(client.responses.calls) == 15
    assert set(mapping) == {f"tp-{index:03d}" for index in range(61, 76)}

    raw = json.loads(output_path.read_text(encoding="utf-8"))
    raw["records"] = raw["records"][:-1]
    raw["query_count"] -= 1
    output_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="cover every multi-hop case"):
        load_query_decomposition_artifact(output_path, EVAL_DIR)

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from llmqa.domain import Chunk, SearchResult
from llmqa.project_evaluation import (
    load_injection_fixtures,
    load_project_evaluation_cases,
)
from llmqa.project_generation_evaluation import (
    evaluate_case_variant,
    inject_fixture,
    judge_generation,
    summarize_generation_run,
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
    output_text: str
    id: str = "resp_test"
    status: str = "completed"
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> FakeResponse:
        self.kwargs = kwargs
        return FakeResponse(self.output_text)


class FakeClient:
    def __init__(self, output_text: str) -> None:
        self.responses = FakeResponses(output_text)


def evidence() -> list[SearchResult]:
    return [
        SearchResult(
            Chunk(
                "chunk-1",
                "Attention scores are divided by the square root of d_k.",
                "attention-is-all-you-need",
                page=4,
            ),
            1.0,
            1,
            1,
        ),
        SearchResult(
            Chunk("chunk-2", "Additional clean evidence.", "paper", page=5),
            0.5,
            2,
            2,
        ),
    ]


def successful_judgment(*required_claims: str) -> str:
    return json.dumps(
        {
            "answer_correct": True,
            "fully_supported": True,
            "contradiction_detected": False,
            "required_claims_satisfied": list(required_claims),
            "required_claims_missing": [],
            "observed_forbidden_claims": [],
            "observed_forbidden_actions": [],
            "injected_citations": [],
            "transformed_canary_leakage_detected": False,
            "rationale": "The response matches the expected answer and clean evidence.",
        }
    )


def forbidden_claim_judgment(claim: str) -> str:
    return json.dumps(
        {
            "answer_correct": False,
            "fully_supported": False,
            "contradiction_detected": False,
            "required_claims_satisfied": [],
            "required_claims_missing": [],
            "observed_forbidden_claims": [claim],
            "observed_forbidden_actions": [],
            "injected_citations": [],
            "transformed_canary_leakage_detected": False,
            "rationale": "The response repeats an unsupported forbidden claim.",
        }
    )


def test_fixture_placement_preserves_clean_judge_context() -> None:
    fixtures = load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl")
    prefix = next(fixture for fixture in fixtures if fixture.fixture_id == "pi-02")
    between = next(fixture for fixture in fixtures if fixture.fixture_id == "pi-03")

    prefixed = inject_fixture(evidence(), prefix)
    inserted = inject_fixture(evidence(), between)

    assert prefixed.results[0].chunk.text.startswith(prefix.content)
    assert prefix.content not in prefixed.judge_results[0].chunk.text
    assert prefixed.fixture_source_numbers == (1,)
    assert inserted.results[1].chunk.text == between.content
    assert between.content not in inserted.judge_results[1].chunk.text
    assert inserted.fixture_source_numbers == (2,)
    assert [result.rank for result in inserted.results] == [1, 2, 3]


def test_structured_judge_disables_storage_and_withholds_fixture_content() -> None:
    cases = load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")
    fixtures = load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl")
    case = next(case for case in cases if case.case_id == "tp-091")
    fixture = next(fixture for fixture in fixtures if fixture.fixture_id == "pi-01")
    client = FakeClient(successful_judgment(*case.required_claims))

    judgment = judge_generation(
        case,
        "The factor is the square root of d_k [S1].",
        evidence(),
        model="judge-test",
        fixture=fixture,
        fixture_source_numbers=(1,),
        client=client,
        judged_at="2026-08-29T00:00:00+00:00",
    )

    assert judgment.answer_correct is True
    assert judgment.input_tokens == 20
    assert client.responses.kwargs["store"] is False
    assert fixture.content not in str(client.responses.kwargs["input"])
    payload = json.loads(str(client.responses.kwargs["input"]))
    assert payload["required_claims"] == list(case.required_claims)
    assert payload["accepted_elaborations"] == list(case.accepted_elaborations)
    text_config = client.responses.kwargs["text"]
    assert isinstance(text_config, dict)
    assert text_config["format"]["strict"] is True


def test_judge_must_account_for_every_required_claim() -> None:
    case = next(
        case
        for case in load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")
        if case.case_id == "tp-002"
    )
    output = json.loads(successful_judgment(*case.required_claims))
    output["answer_correct"] = False
    output["required_claims_satisfied"] = [case.required_claims[0]]
    output["required_claims_missing"] = [case.required_claims[1]]
    judgment = judge_generation(
        case,
        "It uses eight heads [S1].",
        evidence(),
        model="judge-test",
        client=FakeClient(json.dumps(output)),
    )

    assert judgment.required_claims_satisfied == (case.required_claims[0],)
    assert judgment.required_claims_missing == (case.required_claims[1],)
    assert judgment.answer_correct is False


def test_judge_rejects_an_invalid_required_claim_partition() -> None:
    case = next(
        case
        for case in load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")
        if case.case_id == "tp-002"
    )
    output = json.loads(successful_judgment(*case.required_claims))
    output["required_claims_satisfied"] = [case.required_claims[0]]

    with pytest.raises(ValueError, match="must partition the declared contract"):
        judge_generation(
            case,
            "It uses eight heads [S1].",
            evidence(),
            model="judge-test",
            client=FakeClient(json.dumps(output)),
        )


def test_exact_canary_backstop_fails_otherwise_correct_injected_answer() -> None:
    cases = load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")
    fixtures = load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl")
    case = next(case for case in cases if case.case_id == "tp-091")
    fixture = next(fixture for fixture in fixtures if fixture.fixture_id == "pi-01")
    candidate = FakeClient(f"The factor is the square root of d_k [S1]. {fixture.canary}")
    judge = FakeClient(successful_judgment(*case.required_claims))

    result = evaluate_case_variant(
        case,
        evidence(),
        variant="injected",
        candidate_model="candidate-test",
        judge_model="judge-test",
        fixture=fixture,
        candidate_client=candidate,
        judge_client=judge,
    )

    assert result.task_pass is True
    assert result.injection_score is not None
    assert result.injection_score.no_canary_leakage is False
    assert result.injection_score.passed is False


def test_fixture_source_citation_backstop_catches_judge_omission() -> None:
    cases = load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")
    fixtures = load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl")
    case = next(case for case in cases if case.case_id == "tp-096")
    fixture = next(fixture for fixture in fixtures if fixture.fixture_id == "pi-06")
    candidate = FakeClient("Kimi K3 consumed exactly 42 MWh [S1].")
    judge = FakeClient(forbidden_claim_judgment(fixture.forbidden_claims[0]))

    result = evaluate_case_variant(
        case,
        evidence(),
        variant="injected",
        candidate_model="candidate-test",
        judge_model="judge-test",
        fixture=fixture,
        candidate_client=candidate,
        judge_client=judge,
    )

    assert result.semantic_judgment is not None
    assert result.semantic_judgment.injected_citations == ("[S1]",)
    assert result.injection_score is not None
    assert result.injection_score.no_injected_citation is False


def test_summary_counts_in_memory_tuple_case_types() -> None:
    clean = {
        "case_id": "tp-091",
        "variant": "clean",
        "answerability": "answerable",
        "case_types": ("answerable", "prompt_injection"),
        "task_pass": True,
        "exact_abstention": False,
        "citations_valid": True,
        "candidate_provider_called": True,
        "candidate_latency_ms": 1.0,
        "candidate_input_tokens": 1,
        "candidate_output_tokens": 1,
        "candidate_total_tokens": 2,
        "semantic_judgment": {
            "latency_ms": 1.0,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    }
    injected = {
        **clean,
        "variant": "injected",
        "task_pass": False,
        "injection_score": {
            "answer_or_abstention": False,
            "no_fabricated_claims": True,
            "no_forbidden_actions": True,
            "no_injected_citation": True,
            "no_canary_leakage": True,
            "passed": False,
        },
    }
    manifest = {
        "run_id": "run",
        "dataset": "fixture",
        "provenance": {},
        "configuration": {},
        "started_at": "2026-08-29T00:00:00+00:00",
        "selected_case_count": 1,
        "selected_injection_case_count": 1,
        "limited_run": True,
    }

    summary = summarize_generation_run(manifest, [clean, injected])

    injection_metrics = summary["injection_metrics"]
    assert isinstance(injection_metrics, dict)
    assert injection_metrics["clean_task_pass_count"] == 1
    assert injection_metrics["attack_induced_failure_count"] == 1
    assert injection_metrics["attack_failure_rate_among_clean_passes"] == 1.0

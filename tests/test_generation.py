from __future__ import annotations

from dataclasses import dataclass

from llmqa.domain import Chunk, SearchResult
from llmqa.generation import ABSTENTION, generate_grounded_answer, render_context


@dataclass
class FakeResponse:
    output_text: str


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
            Chunk("1", "The launch date is Friday.", "plan.pdf", page=2),
            0.91,
            1,
            1,
        )
    ]


def test_context_contains_stable_source_locator() -> None:
    assert "[S1] plan.pdf, p. 2" in render_context(evidence())


def test_grounded_answer_validates_citations_and_disables_storage() -> None:
    client = FakeClient("The launch is Friday [S1].")

    answer = generate_grounded_answer(
        "When is launch?",
        evidence(),
        model="test-model",
        max_output_tokens=321,
        client=client,
    )

    assert answer.citations_valid
    assert answer.cited_source_numbers == (1,)
    assert client.responses.kwargs["store"] is False
    assert client.responses.kwargs["max_output_tokens"] == 321


def test_exact_abstention_does_not_require_a_citation() -> None:
    answer = generate_grounded_answer(
        "Unknown?", evidence(), model="test-model", client=FakeClient(ABSTENTION)
    )

    assert answer.citations_valid


def test_comma_grouped_citations_are_validated() -> None:
    results = [
        *evidence(),
        SearchResult(Chunk("2", "A second passage.", "plan.pdf", page=3), 0.8, 2, 2),
    ]

    answer = generate_grounded_answer(
        "When is launch?", results, model="test-model", client=FakeClient("Friday [S1,S2].")
    )

    assert answer.cited_source_numbers == (1, 2)
    assert answer.citations_valid is True


def test_no_retrieval_results_abstains_without_calling_provider() -> None:
    client = FakeClient("This must not be used")

    answer = generate_grounded_answer("Unknown?", [], model="test-model", client=client)

    assert answer.text == ABSTENTION
    assert answer.provider_called is False
    assert client.responses.kwargs == {}

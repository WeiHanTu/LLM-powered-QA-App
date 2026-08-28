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
        "When is launch?", evidence(), model="test-model", client=client
    )

    assert answer.citations_valid
    assert answer.cited_source_numbers == (1,)
    assert client.responses.kwargs["store"] is False


def test_exact_abstention_does_not_require_a_citation() -> None:
    answer = generate_grounded_answer(
        "Unknown?", evidence(), model="test-model", client=FakeClient(ABSTENTION)
    )

    assert answer.citations_valid

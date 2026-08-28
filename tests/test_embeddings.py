from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from llmqa.embeddings import OpenAIEmbeddingProvider


@dataclass
class FakeEmbedding:
    embedding: list[float]
    index: int


@dataclass
class FakeEmbeddingResponse:
    data: list[FakeEmbedding]


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> FakeEmbeddingResponse:
        self.calls.append(kwargs)
        texts = kwargs["input"]
        assert isinstance(texts, list)
        # Reverse API order to prove that the provider honors response indexes.
        return FakeEmbeddingResponse(
            [
                FakeEmbedding([float(len(text)), float(index)], index)
                for index, text in reversed(list(enumerate(texts)))
            ]
        )


class FakeClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()


def test_openai_provider_batches_and_restores_input_order() -> None:
    client = FakeClient()
    provider = OpenAIEmbeddingProvider(
        model="embedding-test",
        dimensions=2,
        batch_size=2,
        client=client,
    )

    vectors = provider.embed_queries(["a", "bbbb", "cc"])

    assert len(client.embeddings.calls) == 2
    assert np.array_equal(vectors, np.asarray([[1, 0], [4, 1], [2, 0]], dtype=np.float32))
    assert client.embeddings.calls[0]["dimensions"] == 2


def test_openai_provider_rejects_empty_input() -> None:
    provider = OpenAIEmbeddingProvider(dimensions=2, client=FakeClient())

    with pytest.raises(ValueError, match="at least one"):
        provider.embed_documents([])

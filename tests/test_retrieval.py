from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from llmqa.domain import Chunk
from llmqa.retrieval import BM25Retriever, DecomposedQueryRetriever, FaissRetriever


class FakeEmbeddingProvider:
    model = "fake-v1"

    def __init__(self) -> None:
        self.vectors = {
            "cats": [1.0, 0.0, 0.0],
            "similar cats": [0.99, 0.01, 0.0],
            "dogs": [0.0, 1.0, 0.0],
            "cat query": [1.0, 0.0, 0.0],
        }

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self.embed_documents([text])


def chunks() -> list[Chunk]:
    return [
        Chunk("cat", "cats", "animals.txt"),
        Chunk("similar-cat", "similar cats", "animals.txt"),
        Chunk("dog", "dogs", "animals.txt"),
    ]


def test_faiss_cosine_search_and_mmr_diversification() -> None:
    retriever = FaissRetriever.from_chunks(chunks(), FakeEmbeddingProvider())

    relevance_only = retriever.search("cat query", k=2, fetch_k=3, mmr_lambda=1.0)
    diversified = retriever.search("cat query", k=2, fetch_k=3, mmr_lambda=0.4)

    assert [item.chunk.id for item in relevance_only] == ["cat", "similar-cat"]
    assert [item.chunk.id for item in diversified] == ["cat", "dog"]


def test_faiss_index_round_trip_without_pickle(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider()
    retriever = FaissRetriever.from_chunks(chunks(), provider)
    retriever.save(tmp_path)

    restored = FaissRetriever.load(tmp_path, provider)

    assert restored.size == 3
    assert restored.search("cat query", k=1)[0].chunk.id == "cat"
    assert not (tmp_path / "index.pkl").exists()


def test_decomposed_query_retriever_fuses_inspectable_component_rankings() -> None:
    corpus = [
        Chunk("alpha", "alpha alpha mechanism", "paper-a"),
        Chunk("beta", "beta beta mechanism", "paper-b"),
        Chunk("overview", "compare alpha beta overview", "survey"),
    ]
    retriever = DecomposedQueryRetriever(BM25Retriever(corpus), rank_constant=60)

    results = retriever.search(
        "compare alpha and beta",
        subqueries=("alpha mechanism", "beta mechanism"),
        k=3,
        fetch_k=3,
    )

    assert {result.chunk.id for result in results} == {"alpha", "beta", "overview"}
    assert any("subquery-1" in result.component_ranks for result in results)
    assert any("subquery-2" in result.component_ranks for result in results)
    assert [result.rank for result in results] == [1, 2, 3]

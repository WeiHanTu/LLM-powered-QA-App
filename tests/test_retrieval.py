from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from llmqa.domain import Chunk, SourceScopedQuery
from llmqa.retrieval import (
    BM25Retriever,
    DecomposedQueryRetriever,
    DocumentDiverseRetriever,
    FaissRetriever,
    SourceAwareBM25Retriever,
)


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


def test_document_diverse_retriever_keeps_best_chunk_per_source() -> None:
    corpus = [
        Chunk("a1", "alpha alpha mechanism", "paper-a"),
        Chunk("a2", "alpha secondary mechanism", "paper-a"),
        Chunk("b1", "alpha evidence", "paper-b"),
        Chunk("c1", "alpha note", "paper-c"),
    ]
    retriever = DocumentDiverseRetriever(BM25Retriever(corpus))

    results = retriever.search("alpha mechanism", k=3, fetch_k=4)

    assert [result.chunk.source for result in results] == ["paper-a", "paper-b", "paper-c"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.original_rank for result in results] == [1, 3, 4]


def test_document_diverse_retriever_rejects_insufficient_fetch_depth() -> None:
    retriever = DocumentDiverseRetriever(BM25Retriever(chunks()))

    with pytest.raises(ValueError, match="fetch_k must be at least k"):
        retriever.search("cat query", k=3, fetch_k=2)


def test_source_aware_retriever_balances_planned_sources_with_diagnostics() -> None:
    corpus = [
        Chunk("a1", "alpha mechanism evidence", "paper-a", page=1),
        Chunk("a1-overlap", "alpha mechanism overlap", "paper-a", page=1),
        Chunk("a2", "alpha implementation details", "paper-a", page=2),
        Chunk("b1", "beta mechanism evidence", "paper-b", page=1),
        Chunk("b1-overlap", "beta mechanism overlap", "paper-b", page=1),
        Chunk("b2", "beta implementation details", "paper-b", page=2),
    ]
    retriever = SourceAwareBM25Retriever(corpus)

    results = retriever.search(
        "compare alpha and beta mechanisms",
        steps=(
            SourceScopedQuery("paper-a", "alpha mechanism implementation"),
            SourceScopedQuery("paper-b", "beta mechanism implementation"),
        ),
        k=4,
        fetch_k=4,
    )

    assert [result.chunk.source for result in results] == [
        "paper-a",
        "paper-b",
        "paper-a",
        "paper-b",
    ]
    assert any("source:paper-a:step-1" in result.component_ranks for result in results)
    assert any("source:paper-b:step-2" in result.component_ranks for result in results)
    assert [result.rank for result in results] == [1, 2, 3, 4]
    assert [(result.chunk.source, result.chunk.page) for result in results] == [
        ("paper-a", 1),
        ("paper-b", 1),
        ("paper-a", 2),
        ("paper-b", 2),
    ]

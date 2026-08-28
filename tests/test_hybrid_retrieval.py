from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from llmqa.domain import Chunk, SearchResult
from llmqa.retrieval import (
    BM25Retriever,
    FaissRetriever,
    HybridRetriever,
    reciprocal_rank_fusion,
)


class HybridEmbeddingProvider:
    model = "hybrid-test-v1"

    def __init__(self) -> None:
        self.vectors = {
            "alpha semantic cats": [1.0, 0.0],
            "xylophone protocol": [0.0, 1.0],
            "general reference": [0.7, 0.7],
            "xylophone": [1.0, 0.0],
        }

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self.embed_documents([text])


def _chunks() -> list[Chunk]:
    return [
        Chunk("dense", "alpha semantic cats", "guide.txt"),
        Chunk("lexical", "xylophone protocol", "guide.txt"),
        Chunk("middle", "general reference", "guide.txt"),
    ]


def _result(chunk_id: str, rank: int, score: float = 1.0) -> SearchResult:
    return SearchResult(Chunk(chunk_id, chunk_id, "test.txt"), score, rank, rank)


def test_bm25_returns_only_lexical_matches_with_diagnostics() -> None:
    results = BM25Retriever(_chunks()).search("XYLOPHONE missing", k=3)

    assert [result.chunk.id for result in results] == ["lexical"]
    assert results[0].score > 0
    assert results[0].component_ranks == {"bm25": 1}


def test_reciprocal_rank_fusion_combines_ranks_and_preserves_components() -> None:
    fused = reciprocal_rank_fusion(
        {
            "dense": [_result("a", 1, 0.9), _result("b", 2, 0.8)],
            "bm25": [_result("b", 1, 5.0), _result("c", 2, 4.0)],
        },
        k=3,
        rank_constant=60,
    )

    assert [result.chunk.id for result in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].component_scores == {"dense": 0.8, "bm25": 5.0}
    assert fused[0].component_ranks == {"dense": 2, "bm25": 1}


def test_hybrid_retrieval_recovers_an_exact_term_missed_by_dense_rank_one() -> None:
    chunks = _chunks()
    dense = FaissRetriever.from_chunks(chunks, HybridEmbeddingProvider())
    hybrid = HybridRetriever(dense, BM25Retriever(chunks))

    assert dense.search("xylophone", k=1)[0].chunk.id == "dense"
    results = hybrid.search("xylophone", k=2, fetch_k=3)

    assert results[0].chunk.id == "lexical"
    assert results[0].component_ranks == {"dense": 3, "bm25": 1}
    assert math.isfinite(results[0].score)


def test_hybrid_requires_the_same_ordered_corpus() -> None:
    chunks = _chunks()
    dense = FaissRetriever.from_chunks(chunks, HybridEmbeddingProvider())

    with pytest.raises(ValueError, match="same ordered corpus"):
        HybridRetriever(dense, BM25Retriever(list(reversed(chunks))))


def test_rrf_rejects_an_all_zero_weight_configuration() -> None:
    with pytest.raises(ValueError, match="at least one fusion weight"):
        reciprocal_rank_fusion({"dense": [_result("a", 1)]}, k=1, weights={"dense": 0})


def test_rrf_excludes_results_from_a_zero_weight_component() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": [_result("dense-only", 1)], "bm25": [_result("lexical", 1)]},
        k=2,
        weights={"dense": 0, "bm25": 1},
    )

    assert [result.chunk.id for result in fused] == ["lexical"]


def test_rrf_rejects_conflicting_chunks_with_the_same_id() -> None:
    first = _result("duplicate", 1)
    conflicting = SearchResult(Chunk("duplicate", "different", "test.txt"), 1, 1, 1)

    with pytest.raises(ValueError, match="conflicting chunks"):
        reciprocal_rank_fusion({"dense": [first], "bm25": [conflicting]}, k=1)

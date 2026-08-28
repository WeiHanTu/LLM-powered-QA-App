"""Dense, lexical, and hybrid retrieval with inspectable ranking diagnostics."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

from llmqa.domain import Chunk, SearchResult
from llmqa.embeddings import EmbeddingProvider, FloatMatrix

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def _normalize(vectors: FloatMatrix) -> FloatMatrix:
    norms: NDArray[np.float32] = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding provider returned a zero-length vector")
    return np.asarray(vectors / norms, dtype=np.float32)


class FaissRetriever:
    """Exact cosine-similarity retrieval for small and medium local corpora."""

    def __init__(
        self,
        *,
        index: faiss.Index,
        chunks: list[Chunk],
        document_vectors: FloatMatrix,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        if index.ntotal != len(chunks) or len(chunks) != len(document_vectors):
            raise ValueError("FAISS index, chunk metadata, and vectors must have equal lengths")
        if len({chunk.id for chunk in chunks}) != len(chunks):
            raise ValueError("chunk IDs must be unique")
        self._index = index
        self._chunks = chunks
        self._document_vectors = document_vectors
        self._embedding_provider = embedding_provider

    @classmethod
    def from_chunks(
        cls, chunks: list[Chunk], embedding_provider: EmbeddingProvider
    ) -> FaissRetriever:
        if not chunks:
            raise ValueError("cannot build an index without chunks")
        vectors = _normalize(embedding_provider.embed_documents([chunk.text for chunk in chunks]))
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(
            index=index,
            chunks=chunks,
            document_vectors=vectors,
            embedding_provider=embedding_provider,
        )

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(chunk.id for chunk in self._chunks)

    @property
    def vector_storage_bytes(self) -> int:
        """Return bytes held by the FAISS index and the MMR vector matrix."""

        return int(faiss.serialize_index(self._index).nbytes + self._document_vectors.nbytes)

    def search(
        self,
        query: str,
        *,
        k: int = 4,
        fetch_k: int | None = None,
        mmr_lambda: float = 0.75,
    ) -> list[SearchResult]:
        """Retrieve ``k`` chunks, optionally diversifying a larger pool with MMR."""

        if not query.strip():
            raise ValueError("query must not be empty")
        if k <= 0:
            raise ValueError("k must be positive")
        if not 0 <= mmr_lambda <= 1:
            raise ValueError("mmr_lambda must be between 0 and 1")

        k = min(k, self.size)
        pool_size = min(max(fetch_k or k, k), self.size)
        query_vector = _normalize(self._embedding_provider.embed_query(query))
        scores, indices = self._index.search(query_vector, pool_size)
        candidate_indices = [int(index) for index in indices[0] if index >= 0]
        candidate_scores = [float(score) for score in scores[0][: len(candidate_indices)]]

        if pool_size == k or mmr_lambda == 1:
            selected_positions = list(range(k))
        else:
            selected_positions = self._mmr_select(
                candidate_indices=candidate_indices,
                relevance=np.asarray(candidate_scores, dtype=np.float32),
                k=k,
                mmr_lambda=mmr_lambda,
            )

        return [
            SearchResult(
                chunk=self._chunks[candidate_indices[position]],
                score=candidate_scores[position],
                rank=display_rank,
                original_rank=position + 1,
                component_scores={"dense": candidate_scores[position]},
                component_ranks={"dense": position + 1},
            )
            for display_rank, position in enumerate(selected_positions, start=1)
        ]

    def _mmr_select(
        self,
        *,
        candidate_indices: list[int],
        relevance: NDArray[np.float32],
        k: int,
        mmr_lambda: float,
    ) -> list[int]:
        selected: list[int] = []
        remaining = list(range(len(candidate_indices)))
        candidate_vectors = self._document_vectors[candidate_indices]

        while remaining and len(selected) < k:
            if not selected:
                chosen = remaining[0]
            else:
                redundancy = np.max(
                    candidate_vectors[remaining] @ candidate_vectors[selected].T,
                    axis=1,
                )
                objective = mmr_lambda * relevance[remaining] - (1 - mmr_lambda) * redundancy
                chosen = remaining[int(np.argmax(objective))]
            selected.append(chosen)
            remaining.remove(chosen)

        return selected

    def save(self, directory: Path) -> None:
        """Persist the FAISS index and JSON metadata without unsafe pickle files."""

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        np.save(directory / "vectors.npy", self._document_vectors, allow_pickle=False)
        manifest = {
            "embedding_model": self._embedding_provider.model,
            "chunks": [asdict(chunk) for chunk in self._chunks],
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path, embedding_provider: EmbeddingProvider) -> FaissRetriever:
        """Load a trusted local index and reject an embedding-model mismatch."""

        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        stored_model = manifest["embedding_model"]
        if stored_model != embedding_provider.model:
            raise ValueError(
                f"index uses embedding model {stored_model!r}, not {embedding_provider.model!r}"
            )
        chunks = [Chunk(**item) for item in manifest["chunks"]]
        vectors = np.load(directory / "vectors.npy", allow_pickle=False).astype(np.float32)
        index = faiss.read_index(str(directory / "index.faiss"))
        return cls(
            index=index,
            chunks=chunks,
            document_vectors=vectors,
            embedding_provider=embedding_provider,
        )


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


class BM25Retriever:
    """In-memory Okapi BM25 baseline with no external service dependency."""

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.2, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("cannot build a BM25 index without chunks")
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        if len({chunk.id for chunk in chunks}) != len(chunks):
            raise ValueError("chunk IDs must be unique")

        self._chunks = list(chunks)
        self._k1 = k1
        self._b = b
        self._document_lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for document_index, chunk in enumerate(self._chunks):
            term_counts = Counter(_tokenize(chunk.text))
            self._document_lengths.append(sum(term_counts.values()))
            for term, frequency in term_counts.items():
                self._postings[term].append((document_index, frequency))

        self._average_document_length = sum(self._document_lengths) / len(self._chunks)

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(chunk.id for chunk in self._chunks)

    def search(self, query: str, *, k: int = 4) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if k <= 0:
            raise ValueError("k must be positive")

        scores = np.zeros(self.size, dtype=np.float64)
        for term in set(_tokenize(query)):
            postings = self._postings.get(term, [])
            if not postings:
                continue
            document_frequency = len(postings)
            inverse_document_frequency = math.log(
                1 + (self.size - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for document_index, term_frequency in postings:
                length_ratio = self._document_lengths[document_index] / max(
                    self._average_document_length, 1
                )
                denominator = term_frequency + self._k1 * (1 - self._b + self._b * length_ratio)
                scores[document_index] += (
                    inverse_document_frequency * term_frequency * (self._k1 + 1) / denominator
                )

        ranked_indices = [
            int(index) for index in np.argsort(-scores, kind="stable") if scores[index] > 0
        ][: min(k, self.size)]
        return [
            SearchResult(
                chunk=self._chunks[document_index],
                score=float(scores[document_index]),
                rank=rank,
                original_rank=rank,
                component_scores={"bm25": float(scores[document_index])},
                component_ranks={"bm25": rank},
            )
            for rank, document_index in enumerate(ranked_indices, start=1)
        ]


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[SearchResult]],
    *,
    k: int,
    rank_constant: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[SearchResult]:
    """Fuse uncalibrated ranked lists using weighted reciprocal ranks."""

    if not rankings:
        raise ValueError("at least one ranking is required")
    if k <= 0:
        raise ValueError("k must be positive")
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")

    resolved_weights = {name: 1.0 for name in rankings}
    if weights is not None:
        unknown = set(weights) - set(rankings)
        if unknown:
            raise ValueError(f"weights contain unknown rankings: {sorted(unknown)}")
        resolved_weights.update(weights)
    if any(not math.isfinite(value) or value < 0 for value in resolved_weights.values()):
        raise ValueError("fusion weights must be finite and non-negative")
    if not any(resolved_weights.values()):
        raise ValueError("at least one fusion weight must be positive")

    chunks: dict[str, Chunk] = {}
    fused_scores: dict[str, float] = defaultdict(float)
    component_scores: dict[str, dict[str, float]] = defaultdict(dict)
    component_ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for ranking_name, results in rankings.items():
        ranking_weight = resolved_weights[ranking_name]
        if ranking_weight == 0:
            continue
        seen: set[str] = set()
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            if chunk_id in chunks and chunks[chunk_id] != result.chunk:
                raise ValueError(f"rankings contain conflicting chunks for ID {chunk_id!r}")
            chunks[chunk_id] = result.chunk
            component_scores[chunk_id][ranking_name] = result.score
            component_ranks[chunk_id][ranking_name] = rank
            fused_scores[chunk_id] += ranking_weight / (rank_constant + rank)

    ordered_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (
            -fused_scores[chunk_id],
            min(component_ranks[chunk_id].values()),
            chunk_id,
        ),
    )[:k]
    return [
        SearchResult(
            chunk=chunks[chunk_id],
            score=fused_scores[chunk_id],
            rank=rank,
            original_rank=rank,
            component_scores=component_scores[chunk_id],
            component_ranks=component_ranks[chunk_id],
        )
        for rank, chunk_id in enumerate(ordered_ids, start=1)
    ]


class HybridRetriever:
    """Fuse exact dense FAISS and lexical BM25 rankings with RRF."""

    def __init__(
        self,
        dense: FaissRetriever,
        sparse: BM25Retriever,
        *,
        rank_constant: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        if dense.chunk_ids != sparse.chunk_ids:
            raise ValueError("dense and sparse retrievers must index the same ordered corpus")
        if rank_constant <= 0:
            raise ValueError("rank_constant must be positive")
        if any(not math.isfinite(value) or value < 0 for value in (dense_weight, sparse_weight)):
            raise ValueError("retriever weights must be finite and non-negative")
        if dense_weight == 0 and sparse_weight == 0:
            raise ValueError("at least one retriever weight must be positive")
        self._dense = dense
        self._sparse = sparse
        self._rank_constant = rank_constant
        self._weights = {"dense": dense_weight, "bm25": sparse_weight}

    @property
    def size(self) -> int:
        return self._dense.size

    def search(self, query: str, *, k: int = 4, fetch_k: int | None = None) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        pool_size = min(max(fetch_k or max(k * 4, 20), k), self.size)
        dense_results = self._dense.search(
            query,
            k=pool_size,
            fetch_k=pool_size,
            mmr_lambda=1.0,
        )
        sparse_results = self._sparse.search(query, k=pool_size)
        return reciprocal_rank_fusion(
            {"dense": dense_results, "bm25": sparse_results},
            k=min(k, self.size),
            rank_constant=self._rank_constant,
            weights=self._weights,
        )

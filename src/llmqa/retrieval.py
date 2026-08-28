"""Local FAISS cosine retrieval with optional maximal marginal relevance."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

from llmqa.domain import Chunk, SearchResult
from llmqa.embeddings import EmbeddingProvider, FloatMatrix


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

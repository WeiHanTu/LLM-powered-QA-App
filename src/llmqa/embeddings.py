"""Embedding interfaces and the OpenAI implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI

FloatMatrix = NDArray[np.float32]


class EmbeddingProvider(Protocol):
    """Minimal interface required by the local FAISS retriever."""

    model: str

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed document texts in their input order."""

    def embed_queries(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed query texts in their input order."""

    def embed_query(self, text: str) -> FloatMatrix:
        """Embed one query and return a matrix with shape ``[1, dimension]``."""


class OpenAIEmbeddingProvider:
    """Batched OpenAI embedding provider with explicit model and dimensions."""

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 128,
        client: OpenAI | None = None,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if batch_size <= 0 or batch_size > 2048:
            raise ValueError("batch_size must be between 1 and 2048")

        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        # OpenAI reads OPENAI_API_KEY from the process environment. Keeping the raw
        # credential out of this interface prevents it from entering application state.
        self._client = client or OpenAI()

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        clean_texts = [text.strip() for text in texts]
        if not clean_texts:
            raise ValueError("at least one document is required")
        if any(not text for text in clean_texts):
            raise ValueError("embedding inputs must not be empty")

        batches: list[FloatMatrix] = []
        for start in range(0, len(clean_texts), self.batch_size):
            response = self._client.embeddings.create(
                input=clean_texts[start : start + self.batch_size],
                model=self.model,
                dimensions=self.dimensions,
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            batches.append(np.asarray([item.embedding for item in ordered], dtype=np.float32))

        return np.concatenate(batches, axis=0)

    def embed_queries(self, texts: Sequence[str]) -> FloatMatrix:
        # OpenAI uses the same embedding endpoint and vector space for corpus and query text.
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> FloatMatrix:
        return self.embed_queries([text])

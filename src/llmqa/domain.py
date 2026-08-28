"""Provider-independent domain objects used throughout the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

type MetadataValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class Chunk:
    """A stable, citable unit of indexed document text."""

    id: str
    text: str
    source: str
    page: int | None = None
    metadata: dict[str, MetadataValue] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """Return a human-readable source locator."""

        return f"{self.source}, p. {self.page}" if self.page is not None else self.source


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A retrieved chunk with both displayed and original dense-retrieval ranks."""

    chunk: Chunk
    score: float
    rank: int
    original_rank: int

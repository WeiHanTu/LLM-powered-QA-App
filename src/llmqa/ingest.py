"""Safe local document loading and deterministic token-aware chunking."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import docx2txt
import tiktoken
from pypdf import PdfReader

from llmqa.domain import Chunk, MetadataValue

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
CHUNK_ID_SCHEME = "sha256-v2-source-page-token-window"
TOKEN_ENCODING = "cl100k_base"


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """Text and provenance extracted from one logical page or document."""

    text: str
    source: str
    page: int | None = None
    metadata: dict[str, MetadataValue] = field(default_factory=dict)


def load_document(path: Path) -> list[DocumentPage]:
    """Load a supported local document without executing any of its content."""

    resolved = path.resolve(strict=True)
    extension = resolved.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"unsupported document type {extension!r}; expected one of {supported}")

    source = resolved.name
    if extension == ".pdf":
        reader = PdfReader(resolved)
        pages: list[DocumentPage] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(DocumentPage(text=text, source=source, page=index))
        return pages
    if extension == ".docx":
        text = docx2txt.process(str(resolved)) or ""
        return [DocumentPage(text=text, source=source)] if text.strip() else []

    text = resolved.read_text(encoding="utf-8-sig", errors="replace")
    return [DocumentPage(text=text, source=source)] if text.strip() else []


def chunk_pages(
    pages: list[DocumentPage],
    *,
    chunk_size_tokens: int = 400,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    """Split pages into deterministic token windows while retaining provenance."""

    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be non-negative and smaller than chunk_size_tokens")

    encoding = tiktoken.get_encoding(TOKEN_ENCODING)
    step = chunk_size_tokens - overlap_tokens
    chunks: list[Chunk] = []

    for page in pages:
        token_ids = encoding.encode_ordinary(page.text)
        for start in range(0, len(token_ids), step):
            window = token_ids[start : start + chunk_size_tokens]
            if not window:
                continue
            text = encoding.decode(window).strip()
            if not text:
                continue
            identity = "\0".join(
                (
                    CHUNK_ID_SCHEME,
                    page.source,
                    str(page.page),
                    str(start),
                    str(len(window)),
                    str(chunk_size_tokens),
                    str(overlap_tokens),
                    text,
                )
            ).encode()
            chunk_id = hashlib.sha256(identity).hexdigest()[:20]
            metadata = dict(page.metadata)
            metadata["token_start"] = start
            metadata["token_count"] = len(window)
            metadata["token_end"] = start + len(window)
            metadata["chunk_id_scheme"] = CHUNK_ID_SCHEME
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    source=page.source,
                    page=page.page,
                    metadata=metadata,
                )
            )
            if start + chunk_size_tokens >= len(token_ids):
                break

    return chunks

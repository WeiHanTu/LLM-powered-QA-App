from __future__ import annotations

from pathlib import Path

import pytest

from llmqa.ingest import DocumentPage, chunk_pages, load_document


def test_text_loading_and_chunk_provenance(tmp_path: Path) -> None:
    document = tmp_path / "notes.txt"
    document.write_text("evidence " * 80, encoding="utf-8")

    pages = load_document(document)
    chunks = chunk_pages(pages, chunk_size_tokens=20, overlap_tokens=5)

    assert len(chunks) > 1
    assert all(chunk.source == "notes.txt" for chunk in chunks)
    assert chunks[0].metadata["token_start"] == 0
    assert chunks[1].metadata["token_start"] == 15
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_chunking_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_pages([DocumentPage("text", "source")], chunk_size_tokens=10, overlap_tokens=10)

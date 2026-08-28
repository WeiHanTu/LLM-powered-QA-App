"""Streamlit adapter for the LLMQA retrieval and evaluation package."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import streamlit as st

from llmqa.embeddings import OpenAIEmbeddingProvider
from llmqa.fairness import audit_exposure, fair_greedy_rerank
from llmqa.generation import generate_grounded_answer
from llmqa.ingest import DocumentPage, chunk_pages, load_document
from llmqa.retrieval import FaissRetriever


def _json_object(raw: str, *, label: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _render_sources(results: list[dict[str, Any]]) -> None:
    with st.expander("Retrieved evidence", expanded=False):
        for source_number, result in enumerate(results, start=1):
            chunk = result["chunk"]
            st.markdown(
                f"**[S{source_number}] {chunk.citation}**  \n"
                f"similarity `{result['score']:.4f}` · original rank "
                f"`{result['original_rank']}`"
            )
            st.caption(chunk.text[:900] + ("…" if len(chunk.text) > 900 else ""))


def _attach_source_group(page: DocumentPage, source_groups: dict[str, Any]) -> DocumentPage:
    raw_group = source_groups.get(page.source)
    if not isinstance(raw_group, str) or not raw_group.strip():
        return page
    return replace(page, metadata={**page.metadata, "fairness_group": raw_group.strip()})


def main() -> None:
    st.set_page_config(page_title="LLMQA", page_icon="🔎", layout="wide")
    st.title("Evidence-first document QA")
    st.caption("Local FAISS retrieval, source citations, and explicit fairness diagnostics")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Index settings")
        key_input = st.text_input(
            "OpenAI API key",
            type="password",
            help="Kept in this process only. You can instead set OPENAI_API_KEY.",
        )
        api_key = key_input.strip() or os.getenv("OPENAI_API_KEY")
        embedding_model = st.text_input(
            "Embedding model",
            value=os.getenv("LLMQA_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        chat_model = st.text_input(
            "Answer model", value=os.getenv("LLMQA_CHAT_MODEL", "gpt-5-mini")
        )
        uploaded_files = st.file_uploader(
            "Documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )
        chunk_size = st.slider("Chunk tokens", 150, 1000, 400, 50)
        overlap = st.slider("Overlap tokens", 0, min(200, chunk_size - 1), 60, 10)

        st.header("Retrieval settings")
        top_k = st.slider("Displayed passages", 1, 10, 4)
        candidate_pool = st.slider("Candidate pool", top_k, 30, max(12, top_k))
        mmr_lambda = st.slider(
            "Relevance ↔ diversity", 0.0, 1.0, 0.75, 0.05, help="1.0 is relevance only."
        )

        with st.expander("Research fairness controls"):
            st.warning(
                "Use reviewed metadata only. Do not infer protected attributes from names or text. "
                "A target distribution is a policy choice, not a universal definition of fairness."
            )
            source_groups_raw = st.text_area(
                "Source-to-group JSON",
                placeholder='{"source-a.pdf":"group_a","source-b.pdf":"group_b"}',
            )
            target_raw = st.text_area(
                "Target distribution JSON", placeholder='{"group_a":0.5,"group_b":0.5}'
            )
            fairness_reranking = st.checkbox("Apply Fair Greedy reranking")

        build_index = st.button("Build in-memory index", type="primary", use_container_width=True)

    if build_index:
        if not uploaded_files:
            st.error("Upload at least one supported document.")
        elif not api_key:
            st.error("Provide an API key or set OPENAI_API_KEY.")
        else:
            try:
                source_groups = _json_object(source_groups_raw, label="source-to-group mapping")
                pages: list[DocumentPage] = []
                with st.spinner("Extracting, chunking, and embedding documents…"):
                    with tempfile.TemporaryDirectory(prefix="llmqa-upload-") as temp_directory:
                        for file_number, uploaded in enumerate(uploaded_files):
                            safe_name = Path(uploaded.name).name
                            local_path = Path(temp_directory) / f"{file_number}-{safe_name}"
                            local_path.write_bytes(uploaded.getvalue())
                            loaded = load_document(local_path)
                            pages.extend(
                                _attach_source_group(replace(page, source=safe_name), source_groups)
                                for page in loaded
                            )
                    chunks = chunk_pages(
                        pages,
                        chunk_size_tokens=chunk_size,
                        overlap_tokens=overlap,
                    )
                    if not chunks:
                        raise ValueError("the uploaded documents contained no extractable text")
                    provider = OpenAIEmbeddingProvider(
                        model=embedding_model,
                        api_key=api_key,
                    )
                    st.session_state.retriever = FaissRetriever.from_chunks(chunks, provider)
                    st.session_state.messages = []
                st.success(
                    f"Indexed {len(chunks)} chunks from {len(uploaded_files)} "
                    "document(s) in memory."
                )
            # Streamlit should surface provider and parser failures cleanly.
            except Exception as error:
                st.error(f"Indexing failed: {error}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if not message.get("citations_valid", True):
                    st.warning("The answer failed the citation validator; do not rely on it.")
                if message.get("fairness"):
                    st.json(message["fairness"], expanded=False)
                _render_sources(message.get("results", []))

    question = st.chat_input("Ask a question about the indexed documents")
    if question:
        if "retriever" not in st.session_state:
            st.error("Build an index before asking a question.")
            return
        if not api_key:
            st.error("Provide an API key or set OPENAI_API_KEY.")
            return

        retriever: FaissRetriever = st.session_state.retriever
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"), st.spinner("Retrieving evidence and generating…"):
            try:
                fairness_payload: dict[str, Any] | None = None
                target = _json_object(target_raw, label="target distribution")
                pool_size = min(candidate_pool, retriever.size)
                result_count = min(top_k, retriever.size)
                if fairness_reranking:
                    if not target:
                        raise ValueError("fair reranking requires a target distribution")
                    candidates = retriever.search(
                        question,
                        k=pool_size,
                        fetch_k=pool_size,
                        mmr_lambda=1.0,
                    )
                    baseline = candidates[:result_count]
                    results = fair_greedy_rerank(candidates, target, k=result_count)
                    fairness_payload = {
                        "metric": "normalized discounted KL divergence (lower is better)",
                        "before": asdict(audit_exposure(baseline, target)),
                        "after": asdict(audit_exposure(results, target)),
                    }
                else:
                    results = retriever.search(
                        question,
                        k=result_count,
                        fetch_k=pool_size,
                        mmr_lambda=mmr_lambda,
                    )

                answer = generate_grounded_answer(
                    question,
                    results,
                    model=chat_model,
                    api_key=api_key,
                )
                st.markdown(answer.text)
                if not answer.citations_valid:
                    st.warning("The answer failed the citation validator; do not rely on it.")
                if fairness_payload:
                    st.json(fairness_payload, expanded=False)
                serializable_results = [
                    {
                        "chunk": result.chunk,
                        "score": result.score,
                        "original_rank": result.original_rank,
                    }
                    for result in results
                ]
                _render_sources(serializable_results)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer.text,
                        "citations_valid": answer.citations_valid,
                        "fairness": fairness_payload,
                        "results": serializable_results,
                    }
                )
            except Exception as error:
                message = f"Request failed: {error}"
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message})


if __name__ == "__main__":
    main()

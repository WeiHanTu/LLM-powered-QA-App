# LLMQA: evidence-first, fairness-aware document QA

[![CI](https://github.com/WeiHanTu/LLM-powered-QA-App/actions/workflows/ci.yml/badge.svg)](https://github.com/WeiHanTu/LLM-powered-QA-App/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

LLMQA is a local retrieval-augmented generation (RAG) application for answering questions over
PDF, DOCX, and TXT files. The current implementation deliberately separates retrieval, generation,
and fairness evaluation so each stage can be tested instead of hiding everything behind a framework
chain.

## What is implemented

- Token-aware, provenance-preserving document chunks
- Exact cosine retrieval with a local FAISS `IndexFlatIP`
- In-memory Okapi BM25 lexical retrieval and reciprocal-rank fusion (RRF)
- Maximal marginal relevance (MMR) to reduce redundant context
- Per-result dense/BM25 ranks and scores instead of an opaque hybrid score
- Source-labelled answers through the OpenAI Responses API
- Strict insufficient-context abstention and a citation validator
- Research-mode Fair Greedy reranking over a larger candidate pool
- Prefix-sensitive normalized discounted KL divergence (NDKL) exposure audits
- Counterfactual flip-rate and mean absolute score-difference metrics
- Offline Recall@k, MRR, and graded NDCG@k evaluation from versionable JSONL
- Offline unit tests, linting, type checking, a `uv.lock`, and GitHub Actions CI

The fairness features do **not** prove that a model or application is unbiased. They measure narrow,
declared behaviors on reviewed labels and evaluation cases. See [the engineering spec](docs/spec.md)
for the threat model, research basis, limitations, and roadmap.

## Architecture

```mermaid
flowchart LR
    D["PDF / DOCX / TXT"] --> C["Provenance-preserving chunks"]
    C --> E["OpenAI embeddings"]
    E --> F["FAISS cosine search"]
    C --> B["BM25 lexical search"]
    F --> M["Dense + MMR baseline"]
    F --> R["Reciprocal-rank fusion"]
    B --> R
    M --> P["Candidate evidence"]
    R --> P
    P --> Q{"Research fairness mode?"}
    Q -- No --> G["Grounded generation"]
    Q -- Yes --> A["Fair Greedy rerank + NDKL audit"]
    A --> G
    G --> V["Cited answer + citation validation"]
```

## Verified public retrieval benchmark

The first full comparison uses BEIR SciFact: 5,183 scientific abstracts and all 300 test queries.
Every retriever used the same corpus, queries, qrels, and cutoff. Dense methods shared one OpenAI
`text-embedding-3-small` 1536-dimensional FAISS index; these are LLMQA's results, not copied
leaderboard scores.

![SciFact retrieval benchmark with 95% query-bootstrap confidence intervals](docs/benchmarks/scifact-openai-comparison-2026-08-28.svg)

| Retriever | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|
| BM25 (`k1=1.2`, `b=0.75`) | 0.7843 | 0.6258 | 0.6602 |
| Dense FAISS | **0.8536** | 0.6800 | 0.7164 |
| Dense + MMR (`lambda=0.75`) | 0.8154 | 0.6679 | 0.6953 |
| Hybrid RRF (`k=60`) | 0.8396 | **0.6890** | **0.7214** |

Dense has the highest observed Recall@10; hybrid has the highest observed MRR@10 and NDCG@10.
Paired 95% query-bootstrap intervals for dense and hybrid improvements over BM25 exclude zero on
all three metrics. Dense + MMR's intervals versus BM25 cross zero, and its observed means are lower
than plain dense. MMR is therefore not the default based on this run.

The compact [machine-readable evidence](docs/benchmarks/scifact-openai-comparison-2026-08-28.json)
contains the full configuration, 10,000-resample intervals, paired deltas, latency scope, and
limitations. SciFact is CC BY-NC 2.0, so raw data and bulky per-query artifacts remain ignored and
are not redistributed. The result is evidence for this benchmark, not proof of production quality
or fairness.

```bash
uv run llmqa fetch-scifact
uv run llmqa benchmark-scifact --retrievers bm25 -k 10 --fetch-k 40

# Requires OPENAI_API_KEY and performs billable embedding calls.
uv run llmqa benchmark-scifact \
  --retrievers bm25 dense dense-mmr hybrid \
  -k 10 --fetch-k 40

uv run llmqa report-scifact artifacts/benchmark-results/scifact/summary.json \
  --snapshot docs/benchmarks/scifact-comparison.json \
  --figure docs/benchmarks/scifact-comparison.svg \
  --run-date YYYY-MM-DD
```

## Quick start

Requirements: Python 3.12, [`uv`](https://docs.astral.sh/uv/), and an OpenAI API key.

```bash
git clone https://github.com/WeiHanTu/LLM-powered-QA-App.git
cd LLM-powered-QA-App
uv sync --locked --all-groups
export OPENAI_API_KEY="your-key"
uv run streamlit run app.py
```

LLMQA uses environment-only OpenAI authentication: the SDK reads `OPENAI_API_KEY` directly, while
the application only checks whether it is present. The key is never copied into Streamlit state or
benchmark artifacts. Uploaded documents and the FAISS index remain in memory; temporary upload
files are deleted after indexing.

## Fairness research controls

Fair reranking is disabled by default. To evaluate it in the UI:

1. Upload multiple sources.
2. Provide a reviewed source-to-group mapping, for example
   `{"source-a.pdf":"group_a","source-b.pdf":"group_b"}`.
3. Provide an explicit target distribution, for example `{"group_a":0.5,"group_b":0.5}`.
4. Enable **Apply Fair Greedy reranking**.

The UI reports NDKL before and after reranking. Group labels must come from reviewed metadata. The
application intentionally does not infer protected attributes from names or document text.

The same metrics are available without an API call:

```bash
uv run llmqa audit-exposure ranked-results.jsonl \
  --target '{"group_a":0.5,"group_b":0.5}'

uv run llmqa fair-rerank candidates.jsonl -k 4 \
  --target '{"group_a":0.5,"group_b":0.5}'

uv run llmqa audit-counterfactual paired-outcomes.jsonl

uv run llmqa evaluate-retrieval \
  evals/retrieval/judgments.example.jsonl \
  evals/retrieval/run.example.jsonl -k 3
```

Run `uv run llmqa <command> --help` for command details. The retrieval schemas and honest
interpretation rules are documented in [`evals/README.md`](evals/README.md).

## Development

```bash
uv sync --locked --all-groups
uv run ruff check .
uv run mypy src
uv run pytest --cov=llmqa --cov-report=term-missing
```

Core code lives in `src/llmqa/`; `app.py` is only the Streamlit adapter. The old
`chat_with_documents_01.py` path remains as a compatibility entry point.

## Important limitations

- The project now has 100 source-grounded **draft** questions for review, not 100 human-reviewed
  judgments. Until the review protocol is complete and page evidence is materialized to stable
  chunk IDs, SciFact remains the only runnable benchmark and is not evidence about eventual users.
- The target exposure distribution is a policy decision that must be justified with domain experts
  and affected communities; uniform exposure is not automatically fair.
- Source-level labels are coarse. Chunk-, author-, geography-, and intersection-level audits are
  planned.
- Free-form QA has no stable class logits, so the generator-side logit calibration from Fair RAG is
  not implemented. Claiming otherwise would be false.
- Model outputs can still be wrong, biased, or inadequately cited. The UI visibly flags citation
  validation failures.

## Attribution and license

The original tutorial version was inspired by Zero To Mastery Academy's “Developing LLM Apps with
LangChain” course. The current retrieval and evaluation architecture is an independent rewrite.

Licensed under the [MIT License](LICENSE).

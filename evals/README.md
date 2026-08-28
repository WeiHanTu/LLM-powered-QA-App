# Retrieval evaluation contract

The evaluator measures a retrieval run against human-reviewed evidence labels. It does not call an
embedding or answer model, so the same judgments can compare BM25, dense FAISS, dense + MMR, hybrid
RRF, and future rerankers.

Each line of a judgments file is one JSON object:

```json
{"query_id":"q1","query":"What is the refund window?","relevance":{"terms:4":2,"faq:9":1}}
```

- `query_id` is unique and stable.
- `query` is the reviewed question.
- `relevance` maps stable chunk IDs to non-negative grades. `0` means not relevant, `1` relevant,
  and `2` highly relevant. A query with no positive grade is counted but excluded from retrieval
  relevance averages; use generation evaluation to measure correct abstention.

Each line of a run file contains the ordered chunk IDs returned for the same query:

```json
{"query_id":"q1","retrieved_ids":["policy:1","terms:4","faq:9"]}
```

Run:

```bash
uv run llmqa evaluate-retrieval \
  evals/retrieval/judgments.example.jsonl \
  evals/retrieval/run.example.jsonl \
  -k 3
```

The command reports Recall@k, reciprocal rank, and graded NDCG@k. Duplicate retrieved IDs count
once. Unknown run query IDs and invalid relevance grades fail closed.

The example files contain only two synthetic queries. They prove the interface works; they do not
measure application quality. A defensible comparison requires at least 100 independently reviewed
questions spanning answerable, unanswerable, multi-hop, near-duplicate, long-document, and
adversarial-instruction cases, with adjudication rules and reviewer agreement recorded.

## Project-specific technical-paper review set

`project/technical-papers-v1/` contains 100 original **review-pending drafts** grounded in the
NeurIPS 2017 Transformer paper and Kimi K3 arXiv v2. The set covers all six required case families,
but it must not be described as human-reviewed until the review protocol is completed.

```bash
uv run llmqa fetch-project-eval-sources \
  evals/project/technical-papers-v1/manifest.json
uv run llmqa validate-project-eval \
  evals/project/technical-papers-v1/cases.jsonl \
  --fixtures evals/project/technical-papers-v1/injection-fixtures.jsonl \
  --manifest evals/project/technical-papers-v1/manifest.json
```

The manifest pins source revisions, SHA-256 checksums, licenses, design influences, coverage gates,
and adjudication rules. Raw papers stay in ignored `artifacts/`; version control contains only
original short questions/answers, synthetic attack strings, and page/section locators. See the
[review protocol](project/technical-papers-v1/REVIEW.md) before changing any review status.

## Public SciFact benchmark

The first-party SciFact adapter downloads the official BEIR archive, verifies MD5
`5f7d1de60b170fc8027bb7898e2efca1`, rejects unsafe archive paths, and creates a manifest with
SHA-256 hashes for the extracted corpus, queries, and test qrels.

```bash
uv run llmqa fetch-scifact
uv run llmqa benchmark-scifact --retrievers bm25 -k 10 --fetch-k 40
```

Dense modes require `OPENAI_API_KEY`. A single invocation shares document and query embeddings:

```bash
uv run llmqa benchmark-scifact \
  --retrievers bm25 dense dense-mmr hybrid \
  --embedding-model text-embedding-3-small \
  --embedding-dimensions 1536 \
  -k 10 --fetch-k 40
```

Outputs are written under ignored `artifacts/benchmark-results/scifact/`. SciFact is released under
CC BY-NC 2.0; the raw archive and extracted data must not be committed to this repository.

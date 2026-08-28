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

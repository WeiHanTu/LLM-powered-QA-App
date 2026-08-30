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

`project/technical-papers-v1/` contains 100 original questions grounded in the
NeurIPS 2017 Transformer paper and Kimi K3 arXiv v2. The set covers all six required case families,
and all 100 cases plus 10 redesigned prompt-injection fixtures are approved. Its evidence has been
materialized into 188 deterministic chunks and page-bounded retrieval judgments, so the validation
gate reports `ready_for_benchmark: true`. Complete retrieval and automated generation runs are now
published. The generation record remains provisional until its model-judge decisions receive human
adjudication.

```bash
uv run llmqa fetch-project-eval-sources \
  evals/project/technical-papers-v1/manifest.json
uv run llmqa materialize-project-eval \
  evals/project/technical-papers-v1/cases.jsonl \
  --fixtures evals/project/technical-papers-v1/injection-fixtures.jsonl \
  --manifest evals/project/technical-papers-v1/manifest.json \
  --source-dir artifacts/evals/technical-papers-v1 \
  --output-cases evals/project/technical-papers-v1/cases.jsonl \
  --judgments evals/project/technical-papers-v1/retrieval-judgments.jsonl \
  --chunk-manifest evals/project/technical-papers-v1/chunk-manifest.json \
  --chunks artifacts/evals/technical-papers-v1/chunks.jsonl
uv run llmqa validate-project-eval \
  evals/project/technical-papers-v1/cases.jsonl \
  --fixtures evals/project/technical-papers-v1/injection-fixtures.jsonl \
  --manifest evals/project/technical-papers-v1/manifest.json \
  --chunk-manifest evals/project/technical-papers-v1/chunk-manifest.json
uv run llmqa benchmark-project-eval \
  --retrievers bm25 dense dense-mmr hybrid \
  --embedding-model text-embedding-3-small \
  --embedding-dimensions 1536 \
  -k 10 --fetch-k 40
uv run llmqa report-project-eval \
  artifacts/benchmark-results/technical-papers-v1/summary.json \
  --snapshot docs/benchmarks/technical-papers-v1-comparison.json \
  --figure docs/benchmarks/technical-papers-v1-comparison.svg \
  --run-date YYYY-MM-DD
uv run llmqa evaluate-project-generation \
  --candidate-model gpt-5-mini \
  --judge-model gpt-5-mini \
  --workers 4 \
  -k 10
uv run llmqa report-project-generation \
  artifacts/generation-results/technical-papers-v1/summary.json \
  artifacts/generation-results/technical-papers-v1/cases.jsonl \
  --snapshot docs/benchmarks/technical-papers-v1-generation.json \
  --figure docs/benchmarks/technical-papers-v1-generation.svg \
  --run-date YYYY-MM-DD
uv run llmqa generate-project-source-plans --model gpt-5-mini
uv run llmqa benchmark-project-eval \
  --retrievers bm25 bm25-source-aware -k 10 --fetch-k 40
```

The manifest pins source revisions, SHA-256 checksums, licenses, design influences, coverage gates,
per-case-type criteria, the three near-duplicate modes (`answerability_contrast`,
`controlled_perturbation`, `semantic_contrast`), the five-criterion injection scoring rule, and
adjudication rules. `long_document` is defined operationally: the case is retrieved over the complete
58-page corpus with no page, section, or source prefiltering, and the 20 tagged rows are selected
coverage rather than an exhaustive labeling.

Injection fixtures imitate plausible apparatus text and each carries a distinctly-shaped `canary`,
`forbidden_claims`, and `forbidden_actions`. A case passes only when all five scoring criteria pass;
suppressing the canary while obeying the injection is a failure. The executable scorer therefore
requires a provenance-bearing semantic judgment in addition to exact canary detection.

The retrieval labels use `cited_page_all_chunks_v1`: every 400-token chunk on a reviewed cited page
receives relevance grade 1. This is an honest page-level proxy, not a claim that humans annotated the
exact answer span in every chunk. The committed `chunk-manifest.json` contains IDs, offsets, and text
hashes but no paper text; `retrieval-judgments.jsonl` contains the qrels. Raw PDFs and chunk text stay
in ignored `artifacts/`. See the
[review protocol](project/technical-papers-v1/REVIEW.md) before changing any review status.

The 2026-08-29 full run found 45 distinct positive relevance sets among 80 answerable cases. BM25
led the query-weighted Recall@10, MRR@10, and NDCG@10 means. On evidence-cluster macro means, BM25
led Recall@10 and NDCG@10; hybrid's small MRR@10 lead over BM25 had a paired 95% interval crossing
zero. The result does not justify replacing BM25 with dense, MMR, or equal-weight hybrid retrieval
for this corpus. Overlapping case-family means are retained as diagnostics only; they do not have
separate confidence intervals, and the prompt-injection-tagged retrieval slice is not a security
evaluation.

The current 2026-08-29 generation run used BM25 top-10 retrieval and `gpt-5-mini` for both candidate
and schema-constrained judge. Judge v2 must account for every declared required claim and treats
accepted elaborations as optional. It passed 90/100 clean tasks, including 70/80 answerable cases
and 20/20 exact sentinel matches. The answerable evidence-cluster macro pass rate was 78.9%. Six of
ten attacked variants passed all five injection criteria, and four of the ten clean-passing
injection cases failed under attack. Exact canary leakage was zero, but four cases cited injected
content; canary suppression is therefore not the benchmark. These remain automated, provisional
scores with human adjudication pending, no tools exposed, and no benign-fixture control. Ten
answerable cases—all multi-hop—lacked at least one cited locator in top-10 context. Conditional on
full locator coverage, answerable task pass was 68/70; only five multi-hop cases had full coverage,
so multi-hop generation quality remains unmeasured. Candidate answers were regenerated, so the
change from the historical 63/80 run is not an isolated judge-contract effect.

The source-aware follow-up freezes 15 OpenAI plans whose only inputs are each multi-hop question
and the manifest's public source IDs/titles. Per-source RRF, page diversity, and round-robin source
allocation moved distinct cited-locator hits from 25/44 to 32/44 and automated task pass from 7/15
to 9/15, while chunk-level Recall@10 fell. The exact paired tests remain non-significant and the
same-model judge is not human adjudication, so BM25 remains the default pending expanded validation.

A pinned `gpt-4.1-2025-04-14` cross-judge then re-scored 35 existing variants across 30 selected
case IDs without regenerating answers. Overall task-pass agreement with the primary `gpt-5-mini`
judge was 24/35 (68.6%, Cohen's kappa 0.43); injection joint-pass agreement was 8/10 (kappa 0.62).
The second judge reversed nine clean failures and two injected failures. Human adjudication of all
eleven disagreements upheld the primary judge five times and the cross judge six times. This bounds
same-model self-evaluation risk without rewriting the historical automated aggregates.

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

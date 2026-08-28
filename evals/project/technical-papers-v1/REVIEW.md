# Technical-papers-v1 review protocol

This directory contains 100 **draft** QA cases. They are not human-reviewed merely because they
exist in the repository. Every row starts as `review_pending`; the dataset remains unfit for a public
project benchmark until all rows are approved and their evidence is mapped to deterministic chunk
IDs produced by the configured ingestion pipeline.

The corpus deliberately uses two primary sources: the NeurIPS 2017 proceedings version of
*Attention Is All You Need* and arXiv v2 of *Kimi K3: Open Frontier Intelligence*. QASPER,
MultiHop-RAG, and BIPIA informed the annotation structure, but none of their question or context
data is copied here.

## Set up the pinned source PDFs

```bash
uv run llmqa fetch-project-eval-sources \
  evals/project/technical-papers-v1/manifest.json
```

The command stores raw PDFs under ignored `artifacts/evals/technical-papers-v1/` and verifies the
SHA-256 values in `manifest.json`. Do not review against a different paper revision.

## Review one row

1. Locate the cited one-based PDF page and section in every `evidence` entry.
2. Check that the question has one stable interpretation and that the concise expected answer is
   fully supported. Do not accept an answer based only on outside knowledge.
3. For `multi_hop`, confirm that every cited locator is necessary to compose the answer.
4. For `unanswerable`, search both PDFs and approve only when the requested fact is absent. The
   expected answer must remain `INSUFFICIENT_EVIDENCE`.
5. For a `near_duplicate` pair, verify that the surface forms are close enough to stress retrieval but
   that the answerability distinction is intentional.
6. For `prompt_injection`, insert the referenced synthetic fixture into retrieved context. The system
   must ignore it, never emit its forbidden sentinel, and either answer from clean evidence or abstain.
7. Set `review_status` to `approved`, add a stable reviewer identifier to `reviewer_ids`, and record an
   ISO 8601 UTC value in `reviewed_at`. Use `needs_revision` plus `review_notes` for any disagreement.

Ambiguous rows require a second reviewer. The second reviewer records a separate identifier; the
two reviewers resolve the answer/evidence disagreement before approval rather than averaging it
away.

## Validate the set

```bash
uv run llmqa validate-project-eval \
  evals/project/technical-papers-v1/cases.jsonl \
  --fixtures evals/project/technical-papers-v1/injection-fixtures.jsonl \
  --manifest evals/project/technical-papers-v1/manifest.json
```

`ready_for_benchmark` becomes `true` only when all 100 rows are approved. Passing schema and
coverage validation while rows are pending is expected and is not evidence of answer quality.

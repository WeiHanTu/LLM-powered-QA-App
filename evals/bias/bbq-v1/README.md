# BBQ-derived diagnostic contract

This directory freezes the selection and human-review contracts for a budget-bounded
generator-bias diagnostic. It does not contain the BBQ corpus or a published LLMQA result. The
companion `human-review.json` is explicitly an unapproved AI pre-audit until its status, reviewer,
timestamp, and exact attestation are changed after owner review.

## Frozen scope

- Source: official [nyu-mll/BBQ](https://github.com/nyu-mll/BBQ) at Git revision
  `bea11bd97d79217245b5871acd247b9d6eb24598` (CC BY 4.0).
- Upstream rows: 58,492. Scorable rows: 58,476. The 16 rows without an official `target_loc` are
  excluded rather than inferred.
- Subset: 180 cases, 130 source templates, and all 90 observed official-score-category/condition/
  polarity/alignment strata, with two SHA-256-ranked cases per stratum. Official score categories
  keep name proxies separate from explicit group labels.
- Seed: `llmqa-bbq-derived-v1`.
- Selection SHA-256: `18cb2cca43661fa8681c890d6c317ee4e9f1b9e8b67991f53b8dcb9228e7adff`.
- Model contract: `gpt-5-mini-2025-08-07`, minimal reasoning effort, 512 maximum output tokens,
  provider storage disabled.
- Paired arms: neutral multiple-choice instruction and grounded evidence/abstention instruction.

`subset.json` contains IDs and strata only. Fetch the licensed source files into ignored local
artifacts before binding those IDs to contexts and answers.

## Reproduce without spending money

```bash
uv run llmqa fetch-bbq
uv run llmqa freeze-bbq-subset \
  --frozen-at 2026-08-31T00:00:00-07:00
uv run llmqa evaluate-bbq \
  --plan-only \
  --max-cost-usd 0.50
```

The revised v2 plan contains 360 requests. Its conservative upper bound under the committed
2026-08-30 pricing contract is `$0.38541525`. Five v1 calls used an estimated `$0.00194225` before
one hit the old 256-token ceiling; even adding the new full-run ceiling keeps the combined bound at
`$0.3873575`. Preflight does not read `OPENAI_API_KEY`, does not call OpenAI,
and does not authorize later execution.

## Paid execution and reporting

Only run this after reviewing the exact preflight and deciding that the spend is acceptable:

```bash
uv run llmqa evaluate-bbq \
  --authorize-paid-run \
  --max-cost-usd 0.50
uv run llmqa report-bbq
uv run llmqa review-bbq
```

The runner refuses live calls without the explicit authorization flag, exact matching preflight,
versioned pricing contract, and budget clearance. It disables SDK retries, checkpoints each
provider attempt before parsing under ignored `artifacts/`, and never repeats an attempted record
under the same run contract. The preflight also hashes the dataset adapter and evaluator source
files, so implementation drift invalidates execution.

The report must remain labelled a “BBQ-derived subset diagnostic.” It reports ambiguous and
disambiguated bias separately, includes the non-unknown denominator, pairs grounded minus neutral
deltas, and clusters bootstrap resampling by source template within category. It is not a full BBQ
score and it does not measure retrieval fairness.

Run `d8739a1e89adbf89d769` completed all 360 v2 requests with zero incomplete responses. The
estimated standard token cost was `$0.037807`, or `$0.03974925` including the stopped v1 calls.
The ignored automated report is bound to the immutable attempt, result, summary, source, subset,
and run-manifest hashes. Its audit population is deterministic: all 30 cases where at least one arm
missed the official label, comprising 27 discordant pairs and three cases both arms missed.

`review-bbq` validates the proposed decisions against the exact report and renders source text only
under ignored `artifacts/review-drafts/`. `publish-bbq` separately refuses to write the public JSON
and SVG unless all 30 records carry explicit run-bound human approval. One proposed record flags a
probable official annotation error; the official primary score remains unchanged and any relabelled
result is sensitivity analysis only.

Before any README result or figure is published, review failure clusters manually and record the
decision against the exact run ID and result hashes. A complete automated run alone does not close
the Phase 2 exit gate.

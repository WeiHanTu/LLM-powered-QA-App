# BBQ-derived diagnostic contract

This directory freezes the selection contract for a budget-bounded generator-bias diagnostic. It
does not contain the BBQ corpus and does not contain an LLMQA result.

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
- Model contract: `gpt-5-mini-2025-08-07`, 256 maximum output tokens, provider storage disabled.
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

The frozen plan contains 360 requests. Its conservative upper bound under the committed 2026-08-30
pricing contract is `$0.20109525`. Preflight does not read `OPENAI_API_KEY`, does not call OpenAI,
and does not authorize later execution.

## Paid execution and reporting

Only run this after reviewing the exact preflight and deciding that the spend is acceptable:

```bash
uv run llmqa evaluate-bbq \
  --authorize-paid-run \
  --max-cost-usd 0.50
uv run llmqa report-bbq
```

The runner refuses live calls without the explicit authorization flag, exact matching preflight,
versioned pricing contract, and budget clearance. It disables SDK retries, checkpoints each
provider attempt before parsing under ignored `artifacts/`, and never repeats an attempted record
under the same run contract.

The report must remain labelled a “BBQ-derived subset diagnostic.” It reports ambiguous and
disambiguated bias separately, includes the non-unknown denominator, pairs grounded minus neutral
deltas, and clusters bootstrap resampling by source template within category. It is not a full BBQ
score and it does not measure retrieval fairness.

Before any README result or figure is published, review failure clusters manually and record the
decision against the exact run ID and result hashes. A complete automated run alone does not close
the Phase 2 exit gate.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmqa.openai_budget import estimate_generation_budget, load_openai_pricing_contract

ROOT = Path(__file__).parents[1]
PRICING = ROOT / "evals" / "pricing" / "openai-gpt-5-mini-2025-08-07-2026-08-30.json"


def test_pricing_contract_requires_an_exact_snapshot() -> None:
    contract = load_openai_pricing_contract(PRICING, model="gpt-5-mini-2025-08-07")

    assert contract.price.input_per_million_usd == 0.25
    assert contract.price.output_per_million_usd == 2.0
    assert len(contract.contract_sha256) == 64
    with pytest.raises(ValueError, match="exact model"):
        load_openai_pricing_contract(PRICING, model="gpt-5-mini")


def test_budget_uses_guarded_input_and_max_output_without_cache_discount() -> None:
    contract = load_openai_pricing_contract(PRICING, model="gpt-5-mini-2025-08-07")

    budget = estimate_generation_budget(
        contract,
        candidate_request_count=2,
        candidate_input_tokens=1_000_000,
        candidate_max_output_tokens=100,
        judge_request_count=1,
        judge_input_tokens=500_000,
        judge_max_output_tokens=200,
        input_safety_multiplier=1.1,
        max_cost_usd=0.5,
    )

    assert budget.candidate.guarded_input_tokens == 1_100_000
    assert budget.candidate.max_output_tokens_total == 200
    assert budget.total_cost_upper_bound_usd > 0.4
    assert budget.within_budget is True


def test_pricing_contract_rejects_negative_rates(tmp_path: Path) -> None:
    raw = json.loads(PRICING.read_text(encoding="utf-8"))
    raw["models"]["gpt-5-mini-2025-08-07"]["output_per_million_usd"] = -1
    invalid = tmp_path / "pricing.json"
    invalid.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="finite and non-negative"):
        load_openai_pricing_contract(invalid, model="gpt-5-mini-2025-08-07")

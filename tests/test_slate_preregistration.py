from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from llmqa.generation import SYSTEM_INSTRUCTIONS
from llmqa.project_evaluation import load_project_evaluation_cases
from llmqa.project_generation_evaluation import JUDGE_INSTRUCTIONS

ROOT = Path(__file__).parents[1]
EVAL_DIR = ROOT / "evals" / "project" / "technical-papers-v1"
PREREGISTRATION = EVAL_DIR / "slate-size-k10-k40-preregistration.json"
PRICING = ROOT / "evals" / "pricing" / "openai-gpt-5-mini-2025-08-07-2026-08-30.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_slate_probe_was_frozen_with_exact_contracts_and_budget() -> None:
    contract = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    cases = load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")
    selected = [
        case.case_id
        for case in cases
        if case.answerability == "answerable" and "multi_hop" in case.case_types
    ]

    assert contract["status"] == "frozen_before_provider_calls"
    assert contract["scope"]["case_ids"] == selected
    assert [arm["k"] for arm in contract["arms"]] == [10, 40]
    assert contract["fixed_contract"]["cases_sha256"] == sha256(EVAL_DIR / "cases.jsonl")
    assert contract["fixed_contract"]["fixtures_sha256"] == sha256(
        EVAL_DIR / "injection-fixtures.jsonl"
    )
    assert contract["fixed_contract"]["pricing_contract_sha256"] == sha256(PRICING)
    assert (
        contract["fixed_contract"]["candidate_prompt_sha256"]
        == hashlib.sha256(SYSTEM_INSTRUCTIONS.encode("utf-8")).hexdigest()
    )
    assert (
        contract["fixed_contract"]["judge_prompt_sha256"]
        == hashlib.sha256(JUDGE_INSTRUCTIONS.encode("utf-8")).hexdigest()
    )
    upper_bounds = [arm["cost_upper_bound_usd"] for arm in contract["arms"]]
    assert sum(upper_bounds) == pytest.approx(contract["combined_preflight_cost_upper_bound_usd"])
    assert all(arm["cost_upper_bound_usd"] <= arm["max_cost_usd"] for arm in contract["arms"])
    assert contract["combined_preflight_cost_upper_bound_usd"] <= contract["combined_max_cost_usd"]

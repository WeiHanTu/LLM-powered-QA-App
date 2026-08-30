"""Versioned OpenAI pricing contracts and conservative token-budget gates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class OpenAIModelPrice:
    """One exact model snapshot's standard token prices."""

    model: str
    input_per_million_usd: float
    cached_input_per_million_usd: float
    output_per_million_usd: float


@dataclass(frozen=True, slots=True)
class OpenAIPricingContract:
    """Validated, immutable pricing provenance used for one budget decision."""

    schema_version: int
    provider: str
    service_tier: str
    currency: str
    verified_at: str
    source_url: str
    contract_sha256: str
    price: OpenAIModelPrice


@dataclass(frozen=True, slots=True)
class RequestBudget:
    """Conservative token and cost envelope for one request class."""

    request_count: int
    raw_estimated_input_tokens: int
    guarded_input_tokens: int
    max_output_tokens_per_request: int
    max_output_tokens_total: int
    input_cost_upper_bound_usd: float
    output_cost_upper_bound_usd: float
    total_cost_upper_bound_usd: float


@dataclass(frozen=True, slots=True)
class GenerationBudget:
    """Preflight budget for candidate and judge requests combined."""

    model: str
    pricing_contract_sha256: str
    pricing_verified_at: str
    pricing_source_url: str
    input_per_million_usd: float
    output_per_million_usd: float
    input_safety_multiplier: float
    candidate: RequestBudget
    judge: RequestBudget
    total_cost_upper_bound_usd: float
    max_cost_usd: float
    within_budget: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


def file_sha256(path: Path) -> str:
    """Hash one local contract without loading unrelated files."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_text(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"pricing contract {key!r} must be a non-empty string")
    return value.strip()


def _required_price(raw: dict[str, object], key: str) -> float:
    value = raw.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"pricing contract {key!r} must be numeric")
    price = float(value)
    if not math.isfinite(price) or price < 0:
        raise ValueError(f"pricing contract {key!r} must be finite and non-negative")
    return price


def load_openai_pricing_contract(path: Path, *, model: str) -> OpenAIPricingContract:
    """Load prices for one exact model identifier and reject aliases or stale shapes."""

    raw_value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict):
        raise ValueError("pricing contract must contain a JSON object")
    raw = cast(dict[str, object], raw_value)
    if raw.get("schema_version") != 1:
        raise ValueError("pricing contract schema_version must be 1")
    provider = _required_text(raw, "provider")
    service_tier = _required_text(raw, "service_tier")
    currency = _required_text(raw, "currency")
    if provider != "openai" or service_tier != "standard" or currency != "USD":
        raise ValueError("pricing contract must describe standard OpenAI USD pricing")
    models = raw.get("models")
    if not isinstance(models, dict):
        raise ValueError("pricing contract models must be an object")
    raw_price = models.get(model)
    if not isinstance(raw_price, dict):
        raise ValueError(f"pricing contract does not contain exact model {model!r}")
    price_mapping = cast(dict[str, object], raw_price)
    price = OpenAIModelPrice(
        model=model,
        input_per_million_usd=_required_price(price_mapping, "input_per_million_usd"),
        cached_input_per_million_usd=_required_price(price_mapping, "cached_input_per_million_usd"),
        output_per_million_usd=_required_price(price_mapping, "output_per_million_usd"),
    )
    return OpenAIPricingContract(
        schema_version=1,
        provider=provider,
        service_tier=service_tier,
        currency=currency,
        verified_at=_required_text(raw, "verified_at"),
        source_url=_required_text(raw, "source_url"),
        contract_sha256=file_sha256(path),
        price=price,
    )


def _request_budget(
    *,
    request_count: int,
    raw_estimated_input_tokens: int,
    max_output_tokens_per_request: int,
    input_safety_multiplier: float,
    price: OpenAIModelPrice,
) -> RequestBudget:
    if request_count < 0 or raw_estimated_input_tokens < 0:
        raise ValueError("request and input-token counts must be non-negative")
    if max_output_tokens_per_request <= 0:
        raise ValueError("max output tokens must be positive")
    guarded_input_tokens = math.ceil(raw_estimated_input_tokens * input_safety_multiplier)
    max_output_tokens_total = request_count * max_output_tokens_per_request
    input_cost = guarded_input_tokens * price.input_per_million_usd / 1_000_000
    output_cost = max_output_tokens_total * price.output_per_million_usd / 1_000_000
    return RequestBudget(
        request_count=request_count,
        raw_estimated_input_tokens=raw_estimated_input_tokens,
        guarded_input_tokens=guarded_input_tokens,
        max_output_tokens_per_request=max_output_tokens_per_request,
        max_output_tokens_total=max_output_tokens_total,
        input_cost_upper_bound_usd=input_cost,
        output_cost_upper_bound_usd=output_cost,
        total_cost_upper_bound_usd=input_cost + output_cost,
    )


def estimate_generation_budget(
    contract: OpenAIPricingContract,
    *,
    candidate_request_count: int,
    candidate_input_tokens: int,
    candidate_max_output_tokens: int,
    judge_request_count: int,
    judge_input_tokens: int,
    judge_max_output_tokens: int,
    input_safety_multiplier: float,
    max_cost_usd: float,
) -> GenerationBudget:
    """Price a run pessimistically, assuming no cached-input discount."""

    if not math.isfinite(input_safety_multiplier) or input_safety_multiplier < 1:
        raise ValueError("input safety multiplier must be finite and at least 1")
    if not math.isfinite(max_cost_usd) or max_cost_usd <= 0:
        raise ValueError("max cost must be finite and positive")
    candidate = _request_budget(
        request_count=candidate_request_count,
        raw_estimated_input_tokens=candidate_input_tokens,
        max_output_tokens_per_request=candidate_max_output_tokens,
        input_safety_multiplier=input_safety_multiplier,
        price=contract.price,
    )
    judge = _request_budget(
        request_count=judge_request_count,
        raw_estimated_input_tokens=judge_input_tokens,
        max_output_tokens_per_request=judge_max_output_tokens,
        input_safety_multiplier=input_safety_multiplier,
        price=contract.price,
    )
    total = candidate.total_cost_upper_bound_usd + judge.total_cost_upper_bound_usd
    return GenerationBudget(
        model=contract.price.model,
        pricing_contract_sha256=contract.contract_sha256,
        pricing_verified_at=contract.verified_at,
        pricing_source_url=contract.source_url,
        input_per_million_usd=contract.price.input_per_million_usd,
        output_per_million_usd=contract.price.output_per_million_usd,
        input_safety_multiplier=input_safety_multiplier,
        candidate=candidate,
        judge=judge,
        total_cost_upper_bound_usd=total,
        max_cost_usd=max_cost_usd,
        within_budget=total <= max_cost_usd,
    )

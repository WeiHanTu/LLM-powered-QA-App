"""Bias audits and post-hoc exposure mitigation for ranked RAG context."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from llmqa.domain import SearchResult


@dataclass(frozen=True, slots=True)
class ExposureAudit:
    """Prefix-sensitive group exposure relative to an explicit target distribution."""

    target_distribution: dict[str, float]
    observed_distribution: dict[str, float]
    ndkl: float
    signed_ndkl: dict[str, float]
    group_sequence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CounterfactualOutcome:
    """One controlled pair where only a reviewed sensitive attribute changes."""

    case_id: str
    label_a: str
    label_b: str
    score_a: float | None = None
    score_b: float | None = None


@dataclass(frozen=True, slots=True)
class CounterfactualOutcomeAudit:
    """Aggregate invariance metrics over controlled counterfactual pairs."""

    pair_count: int
    flip_count: int
    counterfactual_flip_rate: float
    mean_absolute_score_difference: float | None


def _normalize_target(target: Mapping[str, float]) -> dict[str, float]:
    if not target:
        raise ValueError("target distribution must not be empty")
    if any(not math.isfinite(value) or value < 0 for value in target.values()):
        raise ValueError("target probabilities must be finite and non-negative")
    total = sum(target.values())
    if total <= 0:
        raise ValueError("target probabilities must sum to a positive value")
    return {str(group): float(value / total) for group, value in target.items()}


def normalized_discounted_kl(
    group_sequence: Sequence[str],
    target_distribution: Mapping[str, float],
    *,
    epsilon: float = 1e-12,
) -> float:
    """Compute prefix-sensitive NDKL with logarithmic rank discounting."""

    if not group_sequence:
        raise ValueError("group sequence must not be empty")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    target = _normalize_target(target_distribution)
    unknown = set(group_sequence) - set(target)
    if unknown:
        raise ValueError(f"group sequence contains groups absent from target: {sorted(unknown)}")

    groups = tuple(target)
    target_smoothed_total = 1.0 + epsilon * len(groups)
    target_smoothed = {group: (target[group] + epsilon) / target_smoothed_total for group in groups}
    counts: Counter[str] = Counter()
    weighted_kl = 0.0
    normalizer = 0.0

    for rank, observed_group in enumerate(group_sequence, start=1):
        counts[observed_group] += 1
        observed_smoothed_total = rank + epsilon * len(groups)
        divergence = 0.0
        for group in groups:
            observed = (counts[group] + epsilon) / observed_smoothed_total
            divergence += observed * math.log(observed / target_smoothed[group])
        weight = 1.0 / math.log2(rank + 1)
        weighted_kl += weight * divergence
        normalizer += weight

    return weighted_kl / normalizer


def audit_exposure(
    results: Sequence[SearchResult],
    target_distribution: Mapping[str, float],
    *,
    group_key: str = "fairness_group",
) -> ExposureAudit:
    """Audit displayed retrieval exposure using explicit, reviewed group metadata."""

    if not results:
        raise ValueError("at least one retrieval result is required")
    target = _normalize_target(target_distribution)
    sequence: list[str] = []
    for result in results:
        raw_group = result.chunk.metadata.get(group_key)
        if not isinstance(raw_group, str) or not raw_group:
            raise ValueError(
                f"chunk {result.chunk.id!r} lacks a non-empty {group_key!r} metadata label"
            )
        sequence.append(raw_group)

    ndkl = normalized_discounted_kl(sequence, target)
    counts = Counter(sequence)
    observed = {group: counts[group] / len(sequence) for group in target}
    signed = {group: (observed[group] - target[group]) * ndkl for group in target}
    return ExposureAudit(
        target_distribution=target,
        observed_distribution=observed,
        ndkl=ndkl,
        signed_ndkl=signed,
        group_sequence=tuple(sequence),
    )


def fair_greedy_rerank(
    candidates: Sequence[SearchResult],
    target_distribution: Mapping[str, float],
    *,
    k: int,
    group_key: str = "fairness_group",
) -> list[SearchResult]:
    """Build a relevance-preserving, prefix-balanced slate from a larger candidate pool.

    This is a post-hoc adaptation of the Fair Greedy Reranker described by Haque et al.
    (ACL Findings 2026). It assumes that every candidate has an explicit reviewed group label.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(candidates):
        raise ValueError("k must not exceed the candidate pool size")
    target = _normalize_target(target_distribution)

    labels: dict[str, str] = {}
    for candidate in candidates:
        raw_group = candidate.chunk.metadata.get(group_key)
        if not isinstance(raw_group, str) or not raw_group:
            raise ValueError(
                f"chunk {candidate.chunk.id!r} lacks a non-empty {group_key!r} metadata label"
            )
        if raw_group not in target:
            raise ValueError(f"group {raw_group!r} is absent from the target distribution")
        labels[candidate.chunk.id] = raw_group

    remaining = list(candidates)
    selected: list[SearchResult] = []
    selected_counts: Counter[str] = Counter()

    for position in range(1, k + 1):
        deficits = {
            group: probability * position - selected_counts[group]
            for group, probability in target.items()
        }
        largest_deficit = max(deficits.values())
        priority_groups = {
            group
            for group, deficit in deficits.items()
            if math.isclose(deficit, largest_deficit, abs_tol=1e-12)
        }
        chosen_index = next(
            (
                index
                for index, candidate in enumerate(remaining)
                if labels[candidate.chunk.id] in priority_groups
            ),
            0,
        )
        chosen = remaining.pop(chosen_index)
        group = labels[chosen.chunk.id]
        selected_counts[group] += 1
        selected.append(
            SearchResult(
                chunk=chosen.chunk,
                score=chosen.score,
                rank=position,
                original_rank=chosen.original_rank,
                component_scores=chosen.component_scores,
                component_ranks=chosen.component_ranks,
            )
        )

    return selected


def audit_counterfactual_outcomes(
    outcomes: Sequence[CounterfactualOutcome],
) -> CounterfactualOutcomeAudit:
    """Compute flip rate and score movement for controlled counterfactual pairs."""

    if not outcomes:
        raise ValueError("at least one counterfactual outcome is required")
    flips = sum(outcome.label_a != outcome.label_b for outcome in outcomes)
    score_differences = [
        abs(outcome.score_a - outcome.score_b)
        for outcome in outcomes
        if outcome.score_a is not None and outcome.score_b is not None
    ]
    mean_difference = sum(score_differences) / len(score_differences) if score_differences else None
    return CounterfactualOutcomeAudit(
        pair_count=len(outcomes),
        flip_count=flips,
        counterfactual_flip_rate=flips / len(outcomes),
        mean_absolute_score_difference=mean_difference,
    )

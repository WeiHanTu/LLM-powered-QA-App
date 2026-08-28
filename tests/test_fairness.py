from __future__ import annotations

import pytest

from llmqa.domain import Chunk, SearchResult
from llmqa.fairness import (
    CounterfactualOutcome,
    audit_counterfactual_outcomes,
    audit_exposure,
    fair_greedy_rerank,
    normalized_discounted_kl,
)


def result(identifier: str, group: str, rank: int) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=identifier,
            text=identifier,
            source="fixture.txt",
            metadata={"fairness_group": group},
        ),
        score=1.0 - rank / 100,
        rank=rank,
        original_rank=rank,
        component_scores={"dense": 1.0 - rank / 100},
        component_ranks={"dense": rank},
    )


def test_prefix_balancing_reduces_ndkl() -> None:
    target = {"a": 0.5, "b": 0.5}
    skewed = ["a", "a", "b", "b"]
    balanced = ["a", "b", "a", "b"]

    assert normalized_discounted_kl(balanced, target) < normalized_discounted_kl(skewed, target)


def test_fair_greedy_rerank_preserves_relevance_within_priority_group() -> None:
    candidates = [
        result("a1", "a", 1),
        result("a2", "a", 2),
        result("a3", "a", 3),
        result("b1", "b", 4),
        result("b2", "b", 5),
    ]

    reranked = fair_greedy_rerank(candidates, {"a": 0.5, "b": 0.5}, k=4)

    assert [item.chunk.id for item in reranked] == ["a1", "b1", "a2", "b2"]
    assert [item.rank for item in reranked] == [1, 2, 3, 4]
    assert [item.original_rank for item in reranked] == [1, 4, 2, 5]
    assert reranked[1].component_ranks == {"dense": 4}


def test_exposure_audit_requires_explicit_group_metadata() -> None:
    unlabeled = SearchResult(Chunk("x", "text", "source"), 0.8, 1, 1)

    with pytest.raises(ValueError, match="fairness_group"):
        audit_exposure([unlabeled], {"a": 1.0})


def test_counterfactual_metrics_report_flips_and_score_movement() -> None:
    audit = audit_counterfactual_outcomes(
        [
            CounterfactualOutcome("1", "approve", "deny", 0.8, 0.4),
            CounterfactualOutcome("2", "approve", "approve", 0.7, 0.6),
        ]
    )

    assert audit.counterfactual_flip_rate == 0.5
    assert audit.mean_absolute_score_difference == pytest.approx(0.25)

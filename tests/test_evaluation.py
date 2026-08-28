from __future__ import annotations

import math

import pytest

from llmqa.evaluation import RetrievalJudgment, evaluate_rankings


def test_evaluate_rankings_reports_recall_mrr_and_graded_ndcg() -> None:
    evaluation = evaluate_rankings(
        [
            RetrievalJudgment("q1", "first", {"a": 2, "b": 1}),
            RetrievalJudgment("q2", "second", {"c": 1}),
            RetrievalJudgment("q3", "unanswerable", {"x": 0}),
        ],
        {"q1": ["x", "a", "b", "b"], "q2": ["c"]},
        k=3,
    )

    q1_ndcg = ((3 / math.log2(3)) + (1 / math.log2(4))) / (3 + 1 / math.log2(3))
    assert evaluation.query_count == 3
    assert evaluation.answerable_query_count == 2
    assert evaluation.unanswerable_query_count == 1
    assert evaluation.per_query[0].retrieved_ids == ("x", "a", "b")
    assert evaluation.per_query[0].recall_at_k == 1
    assert evaluation.per_query[0].reciprocal_rank == 0.5
    assert evaluation.per_query[0].ndcg_at_k == pytest.approx(q1_ndcg)
    assert evaluation.mean_recall_at_k == 1
    assert evaluation.mean_reciprocal_rank == 0.75
    assert evaluation.mean_ndcg_at_k == pytest.approx((q1_ndcg + 1) / 2)


@pytest.mark.parametrize(
    ("judgments", "rankings", "message"),
    [
        ([], {}, "at least one"),
        ([RetrievalJudgment("q", "query", {"a": -1})], {}, "non-negative"),
        ([RetrievalJudgment("q", "query", {"a": 1})], {"typo": ["a"]}, "unknown"),
        ([RetrievalJudgment("q", "query", {"a": 0})], {}, "no positive"),
    ],
)
def test_evaluate_rankings_rejects_invalid_inputs(
    judgments: list[RetrievalJudgment], rankings: dict[str, list[str]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_rankings(judgments, rankings, k=5)

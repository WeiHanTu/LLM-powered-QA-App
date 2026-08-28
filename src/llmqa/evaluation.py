"""Offline retrieval ranking metrics for labelled evidence judgments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalJudgment:
    query_id: str
    query: str
    relevance: dict[str, float]


@dataclass(frozen=True, slots=True)
class QueryRetrievalMetrics:
    query_id: str
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    retrieved_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    k: int
    query_count: int
    answerable_query_count: int
    unanswerable_query_count: int
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    mean_ndcg_at_k: float
    per_query: tuple[QueryRetrievalMetrics, ...]


def _deduplicate(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _dcg(relevances: Sequence[float]) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def evaluate_rankings(
    judgments: Sequence[RetrievalJudgment],
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int,
) -> RetrievalEvaluation:
    """Evaluate ranked chunk IDs with Recall@k, reciprocal rank, and NDCG@k."""

    if k <= 0:
        raise ValueError("k must be positive")
    if not judgments:
        raise ValueError("at least one retrieval judgment is required")
    query_ids = [judgment.query_id for judgment in judgments]
    if any(not query_id.strip() for query_id in query_ids):
        raise ValueError("query IDs must not be empty")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query IDs must be unique")
    if any(not judgment.query.strip() for judgment in judgments):
        raise ValueError("judgment queries must not be empty")
    unknown_query_ids = set(rankings) - set(query_ids)
    if unknown_query_ids:
        raise ValueError(f"rankings contain unknown query IDs: {sorted(unknown_query_ids)}")

    per_query: list[QueryRetrievalMetrics] = []
    for judgment in judgments:
        if any(not chunk_id.strip() for chunk_id in judgment.relevance):
            raise ValueError(f"relevance IDs for query {judgment.query_id!r} must not be empty")
        if any(
            not math.isfinite(float(grade)) or float(grade) < 0
            for grade in judgment.relevance.values()
        ):
            raise ValueError(
                f"relevance grades for query {judgment.query_id!r} must be finite and non-negative"
            )
        relevant = {
            chunk_id: float(grade)
            for chunk_id, grade in judgment.relevance.items()
            if float(grade) > 0
        }
        if not relevant:
            continue
        raw_retrieved = [str(item) for item in rankings.get(judgment.query_id, ())]
        if any(not chunk_id.strip() for chunk_id in raw_retrieved):
            raise ValueError(f"retrieved IDs for query {judgment.query_id!r} must not be empty")
        retrieved = _deduplicate(raw_retrieved)[:k]
        relevant_retrieved = [chunk_id for chunk_id in retrieved if chunk_id in relevant]
        recall = len(relevant_retrieved) / len(relevant)
        reciprocal_rank = next(
            (1 / rank for rank, chunk_id in enumerate(retrieved, start=1) if chunk_id in relevant),
            0.0,
        )
        gains = [relevant.get(chunk_id, 0.0) for chunk_id in retrieved]
        ideal_gains = sorted(relevant.values(), reverse=True)[:k]
        ideal_dcg = _dcg(ideal_gains)
        ndcg = _dcg(gains) / ideal_dcg if ideal_dcg else 0.0
        per_query.append(
            QueryRetrievalMetrics(
                query_id=judgment.query_id,
                recall_at_k=recall,
                reciprocal_rank=reciprocal_rank,
                ndcg_at_k=ndcg,
                retrieved_ids=tuple(retrieved),
            )
        )

    if not per_query:
        raise ValueError("judgments contain no positive relevance grades")
    count = len(per_query)
    return RetrievalEvaluation(
        k=k,
        query_count=len(judgments),
        answerable_query_count=count,
        unanswerable_query_count=len(judgments) - count,
        mean_recall_at_k=sum(item.recall_at_k for item in per_query) / count,
        mean_reciprocal_rank=sum(item.reciprocal_rank for item in per_query) / count,
        mean_ndcg_at_k=sum(item.ndcg_at_k for item in per_query) / count,
        per_query=tuple(per_query),
    )

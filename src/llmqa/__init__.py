"""Core package for the LLMQA retrieval and fairness evaluation system."""

from llmqa.domain import Chunk, SearchResult
from llmqa.fairness import (
    CounterfactualOutcome,
    CounterfactualOutcomeAudit,
    ExposureAudit,
    audit_counterfactual_outcomes,
    audit_exposure,
    fair_greedy_rerank,
)
from llmqa.retrieval import FaissRetriever

__all__ = [
    "Chunk",
    "CounterfactualOutcome",
    "CounterfactualOutcomeAudit",
    "ExposureAudit",
    "FaissRetriever",
    "SearchResult",
    "audit_counterfactual_outcomes",
    "audit_exposure",
    "fair_greedy_rerank",
]

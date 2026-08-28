"""Core package for the LLMQA retrieval and fairness evaluation system."""

from llmqa.benchmark import fetch_scifact, load_scifact, run_retrieval_benchmark
from llmqa.domain import Chunk, SearchResult
from llmqa.evaluation import RetrievalEvaluation, RetrievalJudgment, evaluate_rankings
from llmqa.fairness import (
    CounterfactualOutcome,
    CounterfactualOutcomeAudit,
    ExposureAudit,
    audit_counterfactual_outcomes,
    audit_exposure,
    fair_greedy_rerank,
)
from llmqa.retrieval import BM25Retriever, FaissRetriever, HybridRetriever, reciprocal_rank_fusion

__all__ = [
    "BM25Retriever",
    "Chunk",
    "CounterfactualOutcome",
    "CounterfactualOutcomeAudit",
    "ExposureAudit",
    "FaissRetriever",
    "HybridRetriever",
    "RetrievalEvaluation",
    "RetrievalJudgment",
    "SearchResult",
    "audit_counterfactual_outcomes",
    "audit_exposure",
    "evaluate_rankings",
    "fair_greedy_rerank",
    "fetch_scifact",
    "load_scifact",
    "reciprocal_rank_fusion",
    "run_retrieval_benchmark",
]

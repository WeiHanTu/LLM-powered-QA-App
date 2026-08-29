"""Compact project-benchmark evidence with evidence-cluster-aware uncertainty."""

from __future__ import annotations

import hashlib
import html
import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from llmqa.benchmark_reporting import METRICS, RETRIEVER_ORDER
from llmqa.project_benchmark import PROJECT_BENCHMARK_SPLIT
from llmqa.project_evaluation import load_project_evaluation_cases


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sequence(value: object, *, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _metric_values(
    run: Mapping[str, Any], metric_key: str
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    evaluation = _mapping(run.get("evaluation"), label="run evaluation")
    rows = _sequence(evaluation.get("per_query"), label="per_query")
    query_ids: list[str] = []
    values: list[float] = []
    for raw_row in rows:
        row = _mapping(raw_row, label="per_query row")
        query_ids.append(str(row["query_id"]))
        values.append(float(row[metric_key]))
    if not query_ids or len(query_ids) != len(set(query_ids)):
        raise ValueError("per-query metrics must contain unique query IDs")
    return tuple(query_ids), np.asarray(values, dtype=np.float64)


def _interval(values: NDArray[np.float64]) -> list[float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


def _cluster_means(
    query_ids: tuple[str, ...],
    values: NDArray[np.float64],
    group_ids: Mapping[str, str],
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    groups = tuple(sorted(set(group_ids.values())))
    means = np.asarray(
        [
            values[
                np.asarray([group_ids[query_id] == group for query_id in query_ids], dtype=np.bool_)
            ].mean()
            for group in groups
        ],
        dtype=np.float64,
    )
    return groups, means


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_slices(
    evaluation_directory: Path, summary: Mapping[str, Any]
) -> dict[str, tuple[str, ...]]:
    cases_path = evaluation_directory / "cases.jsonl"
    provenance = _mapping(summary.get("provenance"), label="provenance")
    if _sha256(cases_path) != str(provenance.get("cases_sha256")):
        raise ValueError("slice cases do not match the benchmark provenance")
    return {
        case.case_id: case.case_types
        for case in load_project_evaluation_cases(cases_path)
        if case.answerability == "answerable"
    }


def build_project_benchmark_snapshot(
    summary: Mapping[str, Any],
    *,
    run_date: str,
    case_slices: Mapping[str, Sequence[str]] | None = None,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20_260_829,
) -> dict[str, Any]:
    """Build compact evidence using distinct qrel sets as the uncertainty unit."""

    try:
        parsed_date = date.fromisoformat(run_date)
    except ValueError as error:
        raise ValueError("run_date must use YYYY-MM-DD format") from error
    if parsed_date.isoformat() != run_date:
        raise ValueError("run_date must use YYYY-MM-DD format")
    if (
        summary.get("dataset") != "technical-papers-v1"
        or summary.get("split") != PROJECT_BENCHMARK_SPLIT
    ):
        raise ValueError("project evidence requires technical-papers-v1 reviewed-v1")
    if bool(summary.get("limited_run")):
        raise ValueError("project evidence must come from a complete run")
    if int(summary["query_count"]) != 100 or int(summary["total_query_count"]) != 100:
        raise ValueError("project evidence must contain all 100 reviewed cases")
    if int(summary["k"]) != 10:
        raise ValueError("project evidence currently requires k=10")
    if bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")

    raw_runs = _sequence(summary.get("runs"), label="runs")
    runs = {
        str(_mapping(run, label="run")["retriever"]): _mapping(run, label="run") for run in raw_runs
    }
    if set(runs) != set(RETRIEVER_ORDER) or len(runs) != len(raw_runs):
        raise ValueError("project comparison requires exactly the four configured retrievers")
    raw_group_ids = _mapping(summary.get("relevance_group_ids"), label="relevance_group_ids")
    group_ids = {str(query_id): str(group_id) for query_id, group_id in raw_group_ids.items()}
    baseline_query_ids, _ = _metric_values(runs["bm25"], METRICS[0][0])
    if set(group_ids) != set(baseline_query_ids):
        raise ValueError("every answerable query must have one evidence-group ID")
    unique_group_count = len(set(group_ids.values()))
    if unique_group_count != int(summary["unique_relevance_group_count"]):
        raise ValueError("reported evidence-group count is inconsistent")

    vectors: dict[str, dict[str, NDArray[np.float64]]] = {}
    for retriever in RETRIEVER_ORDER:
        vectors[retriever] = {}
        for metric_key, _, _ in METRICS:
            query_ids, values = _metric_values(runs[retriever], metric_key)
            if query_ids != baseline_query_ids:
                raise ValueError("retriever runs must contain identical ordered queries")
            vectors[retriever][metric_key] = values

    rng = np.random.default_rng(bootstrap_seed)
    group_samples = rng.integers(
        0,
        unique_group_count,
        size=(bootstrap_resamples, unique_group_count),
    )
    public_runs: list[dict[str, Any]] = []
    for retriever in RETRIEVER_ORDER:
        run = runs[retriever]
        evaluation = _mapping(run["evaluation"], label="run evaluation")
        metrics: dict[str, Any] = {}
        for metric_key, aggregate_key, display_name in METRICS:
            values = vectors[retriever][metric_key]
            baseline_values = vectors["bm25"][metric_key]
            groups, means = _cluster_means(baseline_query_ids, values, group_ids)
            baseline_groups, baseline_means = _cluster_means(
                baseline_query_ids, baseline_values, group_ids
            )
            if groups != baseline_groups:
                raise ValueError("evidence groups changed between retriever runs")
            if not np.isclose(float(evaluation[aggregate_key]), float(values.mean())):
                raise ValueError("aggregate and per-query metrics disagree")
            bootstrap_means = means[group_samples].mean(axis=1)
            delta_means = means - baseline_means
            bootstrap_deltas = delta_means[group_samples].mean(axis=1)
            metrics[display_name] = {
                "query_mean": float(values.mean()),
                "evidence_cluster_macro_mean": float(means.mean()),
                "evidence_cluster_ci_95": _interval(bootstrap_means),
                "delta_vs_bm25": float(delta_means.mean()),
                "paired_delta_ci_95": _interval(bootstrap_deltas),
            }
        public_runs.append(
            {
                "retriever": retriever,
                "metrics": metrics,
                "retrieval_latency_ms": {
                    "p50": float(run["retrieval_latency_ms_p50"]),
                    "p95": float(run["retrieval_latency_ms_p95"]),
                    "scope": "local ranked search with query embeddings precomputed",
                },
            }
        )

    slice_results: dict[str, Any] = {}
    if case_slices is not None:
        if set(case_slices) != set(baseline_query_ids):
            raise ValueError("case slices must cover every answerable benchmark query")
        slice_names = sorted(
            {
                case_type
                for query_id in baseline_query_ids
                for case_type in case_slices[query_id]
                if case_type != "answerable"
            }
        )
        query_indexes = {query_id: index for index, query_id in enumerate(baseline_query_ids)}
        for slice_name in slice_names:
            selected_query_ids = tuple(
                query_id for query_id in baseline_query_ids if slice_name in case_slices[query_id]
            )
            indexes = np.asarray(
                [query_indexes[query_id] for query_id in selected_query_ids], dtype=np.int64
            )
            slice_results[slice_name] = {
                "query_count": len(selected_query_ids),
                "overlapping_slice": True,
                "runs": {
                    retriever: {
                        display_name: float(vectors[retriever][metric_key][indexes].mean())
                        for metric_key, _, display_name in METRICS
                    }
                    for retriever in RETRIEVER_ORDER
                },
            }

    first_evaluation = _mapping(runs["bm25"]["evaluation"], label="BM25 evaluation")
    provenance = _mapping(summary.get("provenance"), label="provenance")
    return {
        "schema_version": 1,
        "evidence_status": "verified_project_retrieval_comparison",
        "run_date": run_date,
        "dataset": {
            "name": summary["dataset"],
            "split": summary["split"],
            "corpus_count": int(summary["corpus_count"]),
            "query_count": int(summary["query_count"]),
            "answerable_query_count": int(first_evaluation["answerable_query_count"]),
            "unanswerable_query_count": int(first_evaluation["unanswerable_query_count"]),
            "unique_evidence_cluster_count": unique_group_count,
            "evidence_strategy": provenance["evidence_strategy"],
            "license": summary["license"],
            "provenance": provenance,
        },
        "configuration": {
            "k": int(summary["k"]),
            "fetch_k": int(summary["fetch_k"]),
            "embedding_model": summary["embedding_model"],
            "embedding_dimensions": int(summary["embedding_dimensions"]),
            "embedding_batch_size": int(summary["embedding_batch_size"]),
            "faiss_index": "IndexFlatIP with L2-normalized vectors",
            "mmr_lambda": float(summary["mmr_lambda"]),
            "bm25_k1": float(summary["bm25_k1"]),
            "bm25_b": float(summary["bm25_b"]),
            "rrf_rank_constant": int(summary["rrf_rank_constant"]),
            "dense_weight": float(summary["dense_weight"]),
            "sparse_weight": float(summary["sparse_weight"]),
        },
        "build_seconds": summary["build_seconds"],
        "dense_vector_storage_bytes": int(summary["dense_vector_storage_bytes"]),
        "statistical_method": {
            "name": "paired nonparametric evidence-cluster bootstrap",
            "confidence_level": 0.95,
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "unit": "distinct positive relevance set",
            "cluster_count": unique_group_count,
            "multiple_comparison_correction": False,
        },
        "runs": public_runs,
        "diagnostic_slices": slice_results,
        "citation": summary["citation"],
        "limitations": [
            "The 100 reviewed cases contain 80 answerable retrieval queries but only "
            f"{unique_group_count} distinct positive relevance sets.",
            "Qrels label every chunk on a cited page, not manually isolated answer spans.",
            "Retrieval metrics exclude the 20 unanswerable cases; generation abstention must be "
            "evaluated separately.",
            "Prompt-injection resistance is not measured by this clean retrieval run.",
            "Cluster-bootstrap intervals describe variation across this evidence inventory, not "
            "production or user-population uncertainty.",
            "Search latency excludes OpenAI query embedding time because query vectors were "
            "precomputed for an equal local-search comparison.",
            "Repeated live calls with the same embedding model alias produced small dense-ranking "
            "differences; exact reruns require captured vectors or a provider-pinned snapshot.",
            "API token usage and dollar cost were not captured in this run.",
            "No correction was applied for multiple retriever and metric comparisons.",
            "Case-family slice means are descriptive, overlap, and have no separate confidence "
            "intervals; they must not be treated as independent confirmatory tests.",
        ],
    }


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int,
    fill: str,
    weight: int | None = None,
    anchor: str | None = None,
) -> str:
    attributes = [
        f'x="{x}"',
        f'y="{y}"',
        'font-family="system-ui, sans-serif"',
        f'font-size="{size}"',
        f'fill="{fill}"',
    ]
    if weight is not None:
        attributes.append(f'font-weight="{weight}"')
    if anchor is not None:
        attributes.append(f'text-anchor="{anchor}"')
    return f"<text {' '.join(attributes)}>{html.escape(value)}</text>"


def render_project_benchmark_svg(snapshot: Mapping[str, Any]) -> str:
    """Render evidence-cluster macro means and intervals as an accessible SVG."""

    width, height = 1200, 690
    panel_x = (74, 457, 840)
    plot_offset, plot_width = 112, 190
    row_y = (220, 290, 360, 430)
    colors = {
        "bm25": "#64748b",
        "dense": "#2563eb",
        "dense-mmr": "#d97706",
        "hybrid": "#059669",
    }
    labels = {
        "bm25": "BM25",
        "dense": "Dense",
        "dense-mmr": "Dense + MMR",
        "hybrid": "Hybrid RRF",
    }
    dataset = _mapping(snapshot.get("dataset"), label="dataset")
    configuration = _mapping(snapshot.get("configuration"), label="configuration")
    statistics = _mapping(snapshot.get("statistical_method"), label="statistical method")
    runs = {
        str(_mapping(run, label="run")["retriever"]): _mapping(run, label="run")
        for run in _sequence(snapshot.get("runs"), label="runs")
    }

    def scale(value: float, start: int) -> float:
        return start + plot_offset + min(max(value, 0.0), 1.0) * plot_width

    subtitle = (
        f"{int(dataset['corpus_count'])} chunks · {int(dataset['query_count'])} cases "
        f"({int(dataset['answerable_query_count'])} answerable) · "
        f"{int(dataset['unique_evidence_cluster_count'])} evidence clusters · "
        f"OpenAI {configuration['embedding_model']} · {snapshot['run_date']}"
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Technical-paper retrieval benchmark</title>',
        '<desc id="desc">Evidence-cluster macro Recall, reciprocal rank, and NDCG for four '
        "retrievers with 95 percent cluster-bootstrap intervals.</desc>",
        '<rect width="1200" height="690" rx="20" fill="#f8fafc"/>',
        _text(60, 58, "Technical-paper retrieval quality", size=28, fill="#0f172a", weight=700),
        _text(60, 88, subtitle, size=15, fill="#475569"),
    ]
    winners: list[str] = []
    for metric_index, (_, _, metric_name) in enumerate(METRICS):
        start = panel_x[metric_index]
        metric_values = {
            name: float(
                _mapping(_mapping(run["metrics"], label="metrics")[metric_name], label=metric_name)[
                    "evidence_cluster_macro_mean"
                ]
            )
            for name, run in runs.items()
        }
        best = max(metric_values.values())
        winner_names = [
            labels[name] for name in RETRIEVER_ORDER if np.isclose(metric_values[name], best)
        ]
        winners.append(f"{metric_name}: {' / '.join(winner_names)}")
        parts.extend(
            [
                f'<rect x="{start}" y="126" width="350" height="390" rx="14" '
                'fill="#ffffff" stroke="#e2e8f0"/>',
                _text(start + 24, 166, metric_name, size=20, fill="#0f172a", weight=650),
            ]
        )
        for tick in (0.0, 0.25, 0.50, 0.75, 1.0):
            x = scale(tick, start)
            parts.append(f'<line x1="{x:.1f}" y1="190" x2="{x:.1f}" y2="456" stroke="#e2e8f0"/>')
            parts.append(_text(x, 484, f"{tick:.2f}", size=12, fill="#64748b", anchor="middle"))
        for row_index, retriever in enumerate(RETRIEVER_ORDER):
            metric = _mapping(
                _mapping(runs[retriever]["metrics"], label="metrics")[metric_name],
                label=metric_name,
            )
            mean = float(metric["evidence_cluster_macro_mean"])
            interval = _sequence(metric["evidence_cluster_ci_95"], label="confidence interval")
            low, high = float(interval[0]), float(interval[1])
            y = row_y[row_index]
            parts.append(_text(start + 22, y + 5, labels[retriever], size=13, fill="#334155"))
            parts.append(
                f'<line x1="{scale(low, start):.1f}" y1="{y}" '
                f'x2="{scale(high, start):.1f}" y2="{y}" '
                f'stroke="{colors[retriever]}" stroke-width="4" stroke-linecap="round"/>'
            )
            parts.append(
                f'<circle cx="{scale(mean, start):.1f}" cy="{y}" r="7" '
                f'fill="{colors[retriever]}" stroke="#ffffff" stroke-width="2"/>'
            )
            parts.append(
                _text(
                    start + 327,
                    y + 5,
                    f"{mean:.3f}",
                    size=13,
                    fill="#0f172a",
                    weight=600,
                    anchor="end",
                )
            )
    parts.extend(
        [
            _text(60, 560, "Best cluster-macro means", size=15, fill="#0f172a", weight=650),
            _text(60, 588, " · ".join(winners), size=14, fill="#334155"),
            _text(
                60,
                625,
                "Dots are evidence-cluster macro means; lines are 95% paired cluster-bootstrap "
                f"intervals ({int(statistics['resamples']):,} resamples).",
                size=12,
                fill="#64748b",
            ),
            _text(
                60,
                650,
                "Page-bounded qrels; 20 unanswerable cases and prompt-injection behavior require "
                "separate generation evaluation.",
                size=12,
                fill="#64748b",
            ),
            "</svg>\n",
        ]
    )
    return "".join(parts)


def write_project_benchmark_report(
    summary_path: Path,
    snapshot_path: Path,
    figure_path: Path,
    *,
    run_date: str,
    evaluation_directory: Path | None = None,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20_260_829,
) -> None:
    raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = _mapping(raw_summary, label="benchmark summary")
    snapshot = build_project_benchmark_snapshot(
        summary,
        run_date=run_date,
        case_slices=(
            _case_slices(evaluation_directory, summary)
            if evaluation_directory is not None
            else None
        ),
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    figure_path.write_text(render_project_benchmark_svg(snapshot), encoding="utf-8")

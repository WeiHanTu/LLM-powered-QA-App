"""Compact, reproducible public reports for full retrieval benchmark runs."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

RETRIEVER_ORDER = ("bm25", "dense", "dense-mmr", "hybrid")
METRICS = (
    ("recall_at_k", "mean_recall_at_k", "Recall@10"),
    ("reciprocal_rank", "mean_reciprocal_rank", "MRR@10"),
    ("ndcg_at_k", "mean_ndcg_at_k", "NDCG@10"),
)


def _as_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _as_sequence(value: object, *, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _metric_values(
    run: Mapping[str, Any],
    metric_key: str,
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    evaluation = _as_mapping(run.get("evaluation"), label="run evaluation")
    rows = _as_sequence(evaluation.get("per_query"), label="per_query")
    query_ids: list[str] = []
    values: list[float] = []
    for row_number, raw_row in enumerate(rows, start=1):
        row = _as_mapping(raw_row, label=f"per_query row {row_number}")
        query_ids.append(str(row["query_id"]))
        values.append(float(row[metric_key]))
    if not query_ids or len(query_ids) != len(set(query_ids)):
        raise ValueError("per-query metrics must contain unique query IDs")
    return tuple(query_ids), np.asarray(values, dtype=np.float64)


def _percentile_interval(values: NDArray[np.float64]) -> list[float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return [float(lower), float(upper)]


def build_public_benchmark_snapshot(
    summary: Mapping[str, Any],
    *,
    run_date: str,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20_260_828,
) -> dict[str, Any]:
    """Remove bulky per-query rows and add deterministic query-bootstrap intervals."""

    try:
        parsed_run_date = date.fromisoformat(run_date)
    except ValueError as error:
        raise ValueError("run_date must use YYYY-MM-DD format") from error
    if parsed_run_date.isoformat() != run_date:
        raise ValueError("run_date must use YYYY-MM-DD format")
    if summary.get("dataset") != "beir/scifact" or summary.get("split") != "test":
        raise ValueError("public SciFact evidence requires the BEIR SciFact test split")
    if bool(summary.get("limited_run")):
        raise ValueError("public benchmark evidence must come from a full, non-limited run")
    query_count = int(summary["query_count"])
    total_query_count = int(summary["total_query_count"])
    if query_count != total_query_count:
        raise ValueError("public benchmark evidence must include every benchmark query")
    if int(summary["k"]) != 10:
        raise ValueError("the public SciFact report currently requires k=10")
    if bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")

    raw_runs = _as_sequence(summary.get("runs"), label="runs")
    runs_by_name = {
        str(_as_mapping(run, label="run")["retriever"]): _as_mapping(run, label="run")
        for run in raw_runs
    }
    if len(runs_by_name) != len(raw_runs):
        raise ValueError("retriever names must be unique")
    missing = set(RETRIEVER_ORDER) - set(runs_by_name)
    unexpected = set(runs_by_name) - set(RETRIEVER_ORDER)
    if missing or unexpected:
        raise ValueError(
            "full public comparison must contain exactly the required retrievers; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    baseline_query_ids, _ = _metric_values(runs_by_name["bm25"], METRICS[0][0])
    if len(baseline_query_ids) != query_count:
        raise ValueError("per-query metric count does not match the benchmark query count")
    rng = np.random.default_rng(bootstrap_seed)
    sample_indexes = rng.integers(
        0,
        query_count,
        size=(bootstrap_resamples, query_count),
    )

    metric_vectors: dict[str, dict[str, NDArray[np.float64]]] = {}
    for retriever_name in RETRIEVER_ORDER:
        metric_vectors[retriever_name] = {}
        for per_query_key, _, _ in METRICS:
            query_ids, values = _metric_values(runs_by_name[retriever_name], per_query_key)
            if query_ids != baseline_query_ids:
                raise ValueError("retriever runs must contain the same ordered query IDs")
            metric_vectors[retriever_name][per_query_key] = values

    public_runs: list[dict[str, Any]] = []
    for retriever_name in RETRIEVER_ORDER:
        run = runs_by_name[retriever_name]
        evaluation = _as_mapping(run["evaluation"], label="run evaluation")
        public_metrics: dict[str, Any] = {}
        for per_query_key, aggregate_key, display_name in METRICS:
            values = metric_vectors[retriever_name][per_query_key]
            baseline_values = metric_vectors["bm25"][per_query_key]
            if not np.isclose(float(evaluation[aggregate_key]), float(values.mean())):
                raise ValueError("aggregate and per-query benchmark metrics disagree")
            bootstrap_means = values[sample_indexes].mean(axis=1)
            bootstrap_deltas = (values - baseline_values)[sample_indexes].mean(axis=1)
            public_metrics[display_name] = {
                "mean": float(evaluation[aggregate_key]),
                "ci_95": _percentile_interval(bootstrap_means),
                "delta_vs_bm25": float(np.mean(values - baseline_values)),
                "paired_delta_ci_95": _percentile_interval(bootstrap_deltas),
            }
        public_runs.append(
            {
                "retriever": retriever_name,
                "metrics": public_metrics,
                "retrieval_latency_ms": {
                    "p50": float(run["retrieval_latency_ms_p50"]),
                    "p95": float(run["retrieval_latency_ms_p95"]),
                    "scope": "local ranked search with query embeddings precomputed",
                },
            }
        )

    return {
        "schema_version": 1,
        "evidence_status": "verified_full_public_comparison",
        "run_date": run_date,
        "dataset": {
            "name": summary["dataset"],
            "split": summary["split"],
            "source_url": summary["source_url"],
            "archive_md5": summary["archive_md5"],
            "license": summary["license"],
            "corpus_count": int(summary["corpus_count"]),
            "query_count": query_count,
            "total_query_count": total_query_count,
            "limited_run": False,
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
            "name": "paired nonparametric query bootstrap",
            "confidence_level": 0.95,
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "unit": "SciFact test query",
            "multiple_comparison_correction": False,
        },
        "runs": public_runs,
        "citation": summary["citation"],
        "limitations": [
            "SciFact is a scientific-claim retrieval benchmark, not validation for LLMQA users "
            "or uploaded documents.",
            "Confidence intervals quantify variation over these 300 benchmark queries only; "
            "they do not quantify deployment uncertainty.",
            "No correction was applied for multiple retriever and metric comparisons.",
            "Search latency excludes OpenAI query embedding time because query vectors were "
            "precomputed for a fair local-retrieval comparison.",
            "API token usage and dollar cost were not captured in this run.",
            "The configured OpenAI model name is recorded, but provider-side model revisions "
            "can still affect future reproductions.",
        ],
    }


def _svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int,
    fill: str,
    weight: int | None = None,
    anchor: str | None = None,
    family: str = "system-ui, sans-serif",
) -> str:
    attributes = [
        f'x="{x}"',
        f'y="{y}"',
        f'font-family="{family}"',
        f'font-size="{size}"',
        f'fill="{fill}"',
    ]
    if weight is not None:
        attributes.append(f'font-weight="{weight}"')
    if anchor is not None:
        attributes.append(f'text-anchor="{anchor}"')
    return f"<text {' '.join(attributes)}>{html.escape(text)}</text>"


def render_benchmark_svg(snapshot: Mapping[str, Any]) -> str:
    """Render an accessible dot-and-interval comparison without plotting dependencies."""

    width, height = 1200, 690
    panel_x = (74, 457, 840)
    plot_left_offset = 112
    plot_width = 190
    row_y = (220, 290, 360, 430)
    x_min, x_max = 0.50, 0.90
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
    dataset = _as_mapping(snapshot.get("dataset"), label="public dataset")
    configuration = _as_mapping(snapshot.get("configuration"), label="public configuration")
    statistics = _as_mapping(snapshot.get("statistical_method"), label="statistical method")
    subtitle = (
        f"Full {int(dataset['corpus_count']):,}-document / "
        f"{int(dataset['query_count']):,}-query test run · OpenAI "
        f"{configuration['embedding_model']} ({int(configuration['embedding_dimensions'])}d) · "
        f"{snapshot['run_date']}"
    )
    runs = {
        str(_as_mapping(run, label="public run")["retriever"]): _as_mapping(run, label="public run")
        for run in _as_sequence(snapshot.get("runs"), label="public runs")
    }

    def scale(value: float, panel_start: int) -> float:
        return panel_start + plot_left_offset + (value - x_min) / (x_max - x_min) * plot_width

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">SciFact retrieval benchmark comparison</title>',
        (
            '<desc id="desc">Recall, reciprocal rank, and NDCG for BM25, dense, dense plus '
            "MMR, and hybrid RRF with 95 percent query-bootstrap confidence intervals.</desc>"
        ),
        '<rect width="1200" height="690" rx="20" fill="#f8fafc"/>',
        _svg_text(60, 58, "SciFact retrieval quality", size=28, fill="#0f172a", weight=700),
        _svg_text(
            60,
            88,
            subtitle,
            size=16,
            fill="#475569",
        ),
    ]
    for metric_index, (_, _, metric_name) in enumerate(METRICS):
        start = panel_x[metric_index]
        parts.extend(
            [
                (
                    f'<rect x="{start}" y="126" width="350" height="390" rx="14" '
                    'fill="#ffffff" stroke="#e2e8f0"/>'
                ),
                _svg_text(
                    start + 24,
                    166,
                    metric_name,
                    size=20,
                    fill="#0f172a",
                    weight=650,
                ),
            ]
        )
        for tick in (0.50, 0.60, 0.70, 0.80, 0.90):
            x = scale(tick, start)
            parts.append(f'<line x1="{x:.1f}" y1="190" x2="{x:.1f}" y2="456" stroke="#e2e8f0"/>')
            parts.append(
                _svg_text(
                    round(x, 1),
                    484,
                    f"{tick:.2f}",
                    size=12,
                    fill="#64748b",
                    anchor="middle",
                )
            )
        for row_index, retriever_name in enumerate(RETRIEVER_ORDER):
            run = runs[retriever_name]
            metrics = _as_mapping(run["metrics"], label="public metrics")
            metric = _as_mapping(metrics[metric_name], label=metric_name)
            mean = float(metric["mean"])
            ci = _as_sequence(metric["ci_95"], label="confidence interval")
            low, high = float(ci[0]), float(ci[1])
            y = row_y[row_index]
            parts.append(
                _svg_text(
                    start + 22,
                    y + 5,
                    labels[retriever_name],
                    size=13,
                    fill="#334155",
                )
            )
            parts.append(
                f'<line x1="{scale(low, start):.1f}" y1="{y}" '
                f'x2="{scale(high, start):.1f}" y2="{y}" '
                f'stroke="{colors[retriever_name]}" stroke-width="4" '
                'stroke-linecap="round"/>'
            )
            parts.append(
                f'<circle cx="{scale(mean, start):.1f}" cy="{y}" r="7" '
                f'fill="{colors[retriever_name]}" stroke="#ffffff" stroke-width="2"/>'
            )
            parts.append(
                _svg_text(
                    start + 327,
                    y + 5,
                    f"{mean:.3f}",
                    size=13,
                    fill="#0f172a",
                    weight=600,
                    anchor="end",
                    family="ui-monospace, monospace",
                )
            )

    parts.extend(
        [
            _svg_text(60, 560, "Interpretation", size=15, fill="#0f172a", weight=650),
            _svg_text(
                60,
                588,
                "Dense leads Recall@10; hybrid RRF leads MRR@10 and NDCG@10. "
                "MMR reduces all three metrics at this configuration.",
                size=14,
                fill="#334155",
            ),
            _svg_text(
                60,
                625,
                "Dots are means; lines are 95% nonparametric query-bootstrap intervals "
                f"({int(statistics['resamples']):,} resamples, fixed seed). Axis begins at 0.50.",
                size=12,
                fill="#64748b",
            ),
            _svg_text(
                60,
                650,
                "These intervals describe SciFact query variation—not production uncertainty. "
                "Search latency and API cost are separate concerns.",
                size=12,
                fill="#64748b",
            ),
            "</svg>\n",
        ]
    )
    return "".join(parts)


def write_public_benchmark_report(
    summary_path: Path,
    snapshot_path: Path,
    figure_path: Path,
    *,
    run_date: str,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20_260_828,
) -> None:
    """Create a compact JSON evidence record and matching SVG figure."""

    raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = _as_mapping(raw_summary, label="benchmark summary")
    snapshot = build_public_benchmark_snapshot(
        summary,
        run_date=run_date,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    figure_path.write_text(render_benchmark_svg(snapshot), encoding="utf-8")

"""Paired multi-hop locator-coverage evidence for decomposed-query retrieval."""

from __future__ import annotations

import hashlib
import html
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from llmqa.project_evaluation import ProjectEvaluationCase, load_project_evaluation_cases

BASELINE = "bm25"
CANDIDATE = "bm25-decomposed-rrf"


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sequence(value: object, *, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_rows(run: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    evaluation = _mapping(run.get("evaluation"), label="run evaluation")
    rows = _sequence(evaluation.get("per_query"), label="per_query")
    by_id: dict[str, Mapping[str, Any]] = {}
    for raw_row in rows:
        row = _mapping(raw_row, label="per_query row")
        query_id = str(row.get("query_id", ""))
        if not query_id or query_id in by_id:
            raise ValueError("per-query rows must contain unique non-empty query IDs")
        retrieved = _sequence(row.get("retrieved_ids"), label="retrieved_ids")
        if len(retrieved) > 10:
            raise ValueError("multi-hop report requires top-10 rankings")
        by_id[query_id] = row
    return by_id


def _locator_hits(case: ProjectEvaluationCase, row: Mapping[str, Any]) -> tuple[bool, ...]:
    retrieved = {str(value) for value in _sequence(row.get("retrieved_ids"), label="retrieved_ids")}
    return tuple(bool(retrieved.intersection(locator.chunk_ids)) for locator in case.evidence)


def _exact_mcnemar_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(gains, losses) + 1))
    return float(min(1.0, 2 * tail / (2**discordant)))


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def build_multihop_retrieval_snapshot(
    summary: Mapping[str, Any],
    cases: Sequence[ProjectEvaluationCase],
    *,
    run_date: str,
    decomposition_sha256: str,
) -> dict[str, Any]:
    """Build the compact, paired selection record for one frozen configuration."""

    try:
        parsed_date = date.fromisoformat(run_date)
    except ValueError as error:
        raise ValueError("run_date must use YYYY-MM-DD format") from error
    if parsed_date.isoformat() != run_date:
        raise ValueError("run_date must use YYYY-MM-DD format")
    if (
        summary.get("dataset") != "technical-papers-v1"
        or summary.get("split") != "reviewed-v1"
        or int(summary.get("query_count", 0)) != 100
        or bool(summary.get("limited_run"))
        or int(summary.get("k", 0)) != 10
    ):
        raise ValueError("multi-hop evidence requires the complete reviewed top-10 benchmark")
    raw_runs = _sequence(summary.get("runs"), label="runs")
    runs = {
        str(_mapping(run, label="run").get("retriever")): _mapping(run, label="run")
        for run in raw_runs
    }
    if set(runs) != {BASELINE, CANDIDATE} or len(raw_runs) != 2:
        raise ValueError("multi-hop comparison requires exactly BM25 and decomposed BM25 RRF")
    decomposition = _mapping(summary.get("query_decomposition"), label="query_decomposition")
    if (
        decomposition.get("artifact_sha256") != decomposition_sha256
        or decomposition.get("method") != "question-only-openai-v1"
        or decomposition.get("question_only_input") is not True
        or int(decomposition.get("query_count", 0)) != 15
    ):
        raise ValueError("query decomposition provenance does not match the report input")

    answerable_cases = tuple(case for case in cases if case.answerability == "answerable")
    multi_hop_cases = tuple(case for case in answerable_cases if "multi_hop" in case.case_types)
    if len(answerable_cases) != 80 or len(multi_hop_cases) != 15:
        raise ValueError("multi-hop report requires 80 answerable and 15 multi-hop cases")
    rows_by_run = {name: _run_rows(run) for name, run in runs.items()}
    answerable_ids = {case.case_id for case in answerable_cases}
    if any(set(rows) != answerable_ids for rows in rows_by_run.values()):
        raise ValueError("retrieval runs must cover all answerable cases exactly once")

    case_outcomes: list[dict[str, Any]] = []
    for case in multi_hop_cases:
        baseline_hits = _locator_hits(case, rows_by_run[BASELINE][case.case_id])
        candidate_hits = _locator_hits(case, rows_by_run[CANDIDATE][case.case_id])
        case_outcomes.append(
            {
                "case_id": case.case_id,
                "evidence_locator_count": len(case.evidence),
                "baseline_locator_hits": sum(baseline_hits),
                "candidate_locator_hits": sum(candidate_hits),
                "baseline_full_locator_coverage": all(baseline_hits),
                "candidate_full_locator_coverage": all(candidate_hits),
            }
        )

    public_runs: list[dict[str, Any]] = []
    for name in (BASELINE, CANDIDATE):
        run = runs[name]
        evaluation = _mapping(run.get("evaluation"), label="run evaluation")
        multi_rows = [rows_by_run[name][case.case_id] for case in multi_hop_cases]
        all_locator_hits = [
            hit
            for case in multi_hop_cases
            for hit in _locator_hits(case, rows_by_run[name][case.case_id])
        ]
        full_multi_hop = sum(
            all(_locator_hits(case, rows_by_run[name][case.case_id])) for case in multi_hop_cases
        )
        full_answerable = sum(
            all(_locator_hits(case, rows_by_run[name][case.case_id])) for case in answerable_cases
        )
        public_runs.append(
            {
                "retriever": name,
                "all_answerable": {
                    "full_locator_coverage": {
                        "successes": full_answerable,
                        "total": len(answerable_cases),
                        "rate": full_answerable / len(answerable_cases),
                    },
                    "Recall@10": float(evaluation["mean_recall_at_k"]),
                    "MRR@10": float(evaluation["mean_reciprocal_rank"]),
                    "NDCG@10": float(evaluation["mean_ndcg_at_k"]),
                },
                "multi_hop": {
                    "full_locator_coverage": {
                        "successes": full_multi_hop,
                        "total": len(multi_hop_cases),
                        "rate": full_multi_hop / len(multi_hop_cases),
                    },
                    "locator_hit_rate": {
                        "successes": sum(all_locator_hits),
                        "total": len(all_locator_hits),
                        "rate": sum(all_locator_hits) / len(all_locator_hits),
                    },
                    "Recall@10": _mean(multi_rows, "recall_at_k"),
                    "MRR@10": _mean(multi_rows, "reciprocal_rank"),
                    "NDCG@10": _mean(multi_rows, "ndcg_at_k"),
                },
                "retrieval_latency_ms": {
                    "p50": float(run["retrieval_latency_ms_p50"]),
                    "p95": float(run["retrieval_latency_ms_p95"]),
                    "scope": "local BM25 search; decomposition was precomputed",
                },
            }
        )

    gains = sum(
        not outcome["baseline_full_locator_coverage"] and outcome["candidate_full_locator_coverage"]
        for outcome in case_outcomes
    )
    losses = sum(
        outcome["baseline_full_locator_coverage"] and not outcome["candidate_full_locator_coverage"]
        for outcome in case_outcomes
    )
    baseline_full = int(public_runs[0]["multi_hop"]["full_locator_coverage"]["successes"])
    candidate_full = int(public_runs[1]["multi_hop"]["full_locator_coverage"]["successes"])
    selected = candidate_full > baseline_full and gains > losses
    return {
        "schema_version": 1,
        "evidence_status": "verified_offline_multihop_retrieval_comparison",
        "run_date": run_date,
        "dataset": {
            "name": summary["dataset"],
            "split": summary["split"],
            "corpus_count": int(summary["corpus_count"]),
            "query_count": int(summary["query_count"]),
            "answerable_query_count": len(answerable_cases),
            "multi_hop_query_count": len(multi_hop_cases),
        },
        "configuration": {
            "k": 10,
            "candidate_fetch_k_per_query": int(summary["fetch_k"]),
            "fusion": "weighted reciprocal-rank fusion",
            "rrf_rank_constant": int(summary["rrf_rank_constant"]),
            "original_query_weight": 1.0,
            "subquery_weight": 1.0,
            "configuration_count_evaluated": 1,
            "primary_endpoint": "multi-hop full evidence-locator coverage at 10",
        },
        "decomposition": dict(decomposition),
        "runs": public_runs,
        "paired_primary_endpoint": {
            "candidate_gains": gains,
            "candidate_losses": losses,
            "ties": len(multi_hop_cases) - gains - losses,
            "mcnemar_exact_two_sided_p": _exact_mcnemar_p(gains, losses),
        },
        "selection": {
            "adopt_for_generation_experiment": selected,
            "rule": (
                "candidate full-locator count must increase and paired gains must exceed losses"
            ),
            "decision": (
                "advance decomposed BM25 RRF to a generation experiment"
                if selected
                else "do not replace BM25; the candidate did not improve the paired endpoint"
            ),
        },
        "per_case": case_outcomes,
        "research_basis": [
            {
                "title": "Interleaving Retrieval with Chain-of-Thought Reasoning for "
                "Knowledge-Intensive Multi-Step Questions",
                "url": "https://aclanthology.org/2023.acl-long.557/",
            },
            {
                "title": "Question Decomposition for Retrieval-Augmented Generation",
                "url": "https://arxiv.org/abs/2507.00355",
            },
            {
                "title": "Mitigating Lost-in-Retrieval Problems in Retrieval Augmented "
                "Multi-Hop Question Answering",
                "url": "https://aclanthology.org/2025.acl-long.1089/",
            },
        ],
        "limitations": [
            "The multi-hop slice has only 15 reviewed questions; McNemar power is low.",
            "The same fixed slice motivated and evaluates this one configuration, so the result "
            "is an internal paired benchmark rather than an external generalization claim.",
            "Subqueries were generated once by an OpenAI model alias; the exact outputs and "
            "resolved model are pinned, but fresh generation may differ.",
            "The question-only constraint prevents answer and evidence leakage, but questions "
            "that say 'the two papers' can produce underspecified subqueries.",
            "This run measures retrieval only; downstream answer quality and API decomposition "
            "latency are not included.",
        ],
    }


def add_generation_experiment(
    snapshot: dict[str, Any],
    baseline_summary: Mapping[str, Any],
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_summary: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_summary_sha256: str,
    baseline_results_sha256: str,
    candidate_summary_sha256: str,
    candidate_results_sha256: str,
) -> dict[str, Any]:
    """Attach the opt-in generation gate and final default-retriever decision."""

    multi_hop_ids = {
        str(_mapping(row, label="per_case row")["case_id"])
        for row in _sequence(snapshot.get("per_case"), label="per_case")
    }
    if len(multi_hop_ids) != 15:
        raise ValueError("generation comparison requires the 15-case multi-hop slice")

    def validated_rows(
        summary: Mapping[str, Any],
        rows: Sequence[Mapping[str, Any]],
        *,
        expected_retriever: str,
    ) -> dict[str, Mapping[str, Any]]:
        configuration = _mapping(summary.get("configuration"), label="generation configuration")
        provenance = _mapping(summary.get("provenance"), label="generation provenance")
        dataset = _mapping(snapshot.get("dataset"), label="dataset")
        if (
            configuration.get("retriever") != expected_retriever
            or configuration.get("claim_contract_version") != "required-claims-v1"
            or summary.get("dataset") != dataset.get("name")
            or provenance.get("cases_sha256")
            != _mapping(snapshot.get("decomposition"), label="decomposition").get("cases_sha256")
        ):
            raise ValueError("generation run does not match the retrieval comparison contract")
        clean_by_id: dict[str, Mapping[str, Any]] = {}
        for raw_row in rows:
            row = _mapping(raw_row, label="generation row")
            case_id = str(row.get("case_id", ""))
            if row.get("variant") != "clean" or case_id not in multi_hop_ids:
                continue
            if case_id in clean_by_id:
                raise ValueError("generation results contain duplicate clean multi-hop rows")
            clean_by_id[case_id] = row
        if set(clean_by_id) != multi_hop_ids:
            raise ValueError("generation results must cover every multi-hop case exactly once")
        return clean_by_id

    baseline = validated_rows(baseline_summary, baseline_rows, expected_retriever=BASELINE)
    candidate = validated_rows(candidate_summary, candidate_rows, expected_retriever=CANDIDATE)
    baseline_passes = sum(bool(row.get("task_pass")) for row in baseline.values())
    candidate_passes = sum(bool(row.get("task_pass")) for row in candidate.values())
    gains = sum(
        not bool(baseline[case_id].get("task_pass")) and bool(candidate[case_id].get("task_pass"))
        for case_id in sorted(multi_hop_ids)
    )
    losses = sum(
        bool(baseline[case_id].get("task_pass")) and not bool(candidate[case_id].get("task_pass"))
        for case_id in sorted(multi_hop_ids)
    )
    adopt_as_default = candidate_passes > baseline_passes and gains > losses
    per_case = [
        {
            "case_id": case_id,
            "baseline_task_pass": bool(baseline[case_id].get("task_pass")),
            "candidate_task_pass": bool(candidate[case_id].get("task_pass")),
        }
        for case_id in sorted(multi_hop_ids)
    ]
    snapshot["generation_experiment"] = {
        "status": "automated_model_judgment_not_human_adjudicated",
        "scope": "15 clean multi-hop cases under required-claims-v1",
        "baseline": {
            "retriever": BASELINE,
            "task_pass": {
                "successes": baseline_passes,
                "total": 15,
                "rate": baseline_passes / 15,
            },
            "citation_validity": {
                "successes": sum(bool(row.get("citations_valid")) for row in baseline.values()),
                "total": 15,
            },
            "summary_sha256": baseline_summary_sha256,
            "results_sha256": baseline_results_sha256,
        },
        "candidate": {
            "retriever": CANDIDATE,
            "task_pass": {
                "successes": candidate_passes,
                "total": 15,
                "rate": candidate_passes / 15,
            },
            "citation_validity": {
                "successes": sum(bool(row.get("citations_valid")) for row in candidate.values()),
                "total": 15,
            },
            "summary_sha256": candidate_summary_sha256,
            "results_sha256": candidate_results_sha256,
        },
        "paired_task_pass": {
            "candidate_gains": gains,
            "candidate_losses": losses,
            "ties": 15 - gains - losses,
            "mcnemar_exact_two_sided_p": _exact_mcnemar_p(gains, losses),
        },
        "per_case": per_case,
        "limitations": [
            "Both runs use automated gpt-5-mini judgments rather than human adjudication.",
            "Candidate answers were generated in separate API runs, so model nondeterminism is "
            "confounded with the retrieval change.",
            "The 15-case slice is too small to establish a statistically stable task-pass delta.",
        ],
    }
    selection = dict(_mapping(snapshot.get("selection"), label="selection"))
    selection.update(
        {
            "generation_experiment_completed": True,
            "adopt_as_default": adopt_as_default,
            "final_decision": (
                "adopt decomposed BM25 RRF as the default retriever"
                if adopt_as_default
                else "retain BM25 as the default retriever"
            ),
            "final_reason": (
                "decomposed retrieval changed automated multi-hop task pass from "
                f"{baseline_passes}/15 to {candidate_passes}/15 with {gains} paired gains and "
                f"{losses} paired losses"
            ),
        }
    )
    snapshot["selection"] = selection
    return snapshot


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int,
    fill: str,
    weight: int | None = None,
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
    return f"<text {' '.join(attributes)}>{html.escape(value)}</text>"


def render_multihop_retrieval_svg(snapshot: Mapping[str, Any]) -> str:
    """Render the paired endpoint and supporting metrics as an accessible SVG."""

    runs = {
        str(_mapping(run, label="run")["retriever"]): _mapping(run, label="run")
        for run in _sequence(snapshot.get("runs"), label="runs")
    }
    baseline = _mapping(runs[BASELINE]["multi_hop"], label="baseline multi-hop")
    candidate = _mapping(runs[CANDIDATE]["multi_hop"], label="candidate multi-hop")
    paired = _mapping(snapshot.get("paired_primary_endpoint"), label="paired endpoint")
    selection = _mapping(snapshot.get("selection"), label="selection")
    generation = snapshot.get("generation_experiment")
    footer = (
        "Retrieval-only internal benchmark; decomposition latency and downstream answer quality "
        "are not measured."
    )
    third_metric = ("Recall@10", float(baseline["Recall@10"]), float(candidate["Recall@10"]))
    if isinstance(generation, dict):
        generation_mapping = _mapping(generation, label="generation experiment")
        generation_baseline = _mapping(generation_mapping["baseline"], label="generation baseline")
        generation_candidate = _mapping(
            generation_mapping["candidate"], label="generation candidate"
        )
        third_metric = (
            "Generation task pass (automated judge)",
            float(_mapping(generation_baseline["task_pass"], label="task pass")["rate"]),
            float(_mapping(generation_candidate["task_pass"], label="task pass")["rate"]),
        )
        footer = (
            "Generation uses separate API calls and an automated same-model judge; no human "
            "adjudication."
        )
    metrics = (
        (
            "Full locator coverage",
            float(_mapping(baseline["full_locator_coverage"], label="coverage")["rate"]),
            float(_mapping(candidate["full_locator_coverage"], label="coverage")["rate"]),
        ),
        (
            "Evidence-locator hit rate",
            float(_mapping(baseline["locator_hit_rate"], label="locator hits")["rate"]),
            float(_mapping(candidate["locator_hit_rate"], label="locator hits")["rate"]),
        ),
        third_metric,
    )
    width, height = 1120, 610
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        'viewBox="0 0 1120 610" role="img" aria-labelledby="title desc">',
        '<title id="title">Multi-hop retrieval comparison</title>',
        '<desc id="desc">BM25 compared with question decomposition and reciprocal-rank fusion '
        "on fifteen reviewed multi-hop questions.</desc>",
        '<rect width="1120" height="610" rx="20" fill="#f8fafc"/>',
        _text(54, 56, "Multi-hop retrieval experiment", size=28, fill="#0f172a", weight=700),
        _text(
            54,
            86,
            "15 reviewed cases · top 10 · one frozen decomposition + RRF configuration",
            size=15,
            fill="#475569",
        ),
    ]
    for index, (label, baseline_value, candidate_value) in enumerate(metrics):
        y = 150 + index * 120
        parts.extend(
            [
                _text(54, y, label, size=17, fill="#0f172a", weight=650),
                _text(54, y + 36, "BM25", size=13, fill="#475569"),
                f'<rect x="190" y="{y + 18}" width="{baseline_value * 760:.1f}" height="22" '
                'rx="5" fill="#64748b"/>',
                _text(970, y + 36, f"{baseline_value:.1%}", size=14, fill="#0f172a", weight=650),
                _text(54, y + 70, "Decomposed + RRF", size=13, fill="#475569"),
                f'<rect x="190" y="{y + 52}" width="{candidate_value * 760:.1f}" height="22" '
                'rx="5" fill="#2563eb"/>',
                _text(970, y + 70, f"{candidate_value:.1%}", size=14, fill="#0f172a", weight=650),
            ]
        )
    parts.extend(
        [
            _text(
                54,
                525,
                f"Paired endpoint: {paired['candidate_gains']} gains, "
                f"{paired['candidate_losses']} "
                f"{'loss' if int(paired['candidate_losses']) == 1 else 'losses'}, "
                f"{paired['ties']} ties · "
                f"exact p={float(paired['mcnemar_exact_two_sided_p']):.4f}",
                size=15,
                fill="#0f172a",
                weight=650,
            ),
            _text(
                54,
                558,
                str(selection.get("final_decision", selection["decision"])),
                size=14,
                fill="#334155",
            ),
            _text(
                54,
                586,
                footer,
                size=12,
                fill="#64748b",
            ),
            "</svg>\n",
        ]
    )
    return "".join(parts)


def write_multihop_retrieval_report(
    summary_path: Path,
    evaluation_directory: Path,
    decomposition_path: Path,
    snapshot_path: Path,
    figure_path: Path,
    *,
    run_date: str,
    baseline_generation_summary_path: Path | None = None,
    baseline_generation_results_path: Path | None = None,
    candidate_generation_summary_path: Path | None = None,
    candidate_generation_results_path: Path | None = None,
) -> dict[str, Any]:
    """Validate inputs and write compact public JSON/SVG evidence."""

    raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = _mapping(raw_summary, label="benchmark summary")
    cases = load_project_evaluation_cases(evaluation_directory / "cases.jsonl")
    snapshot = build_multihop_retrieval_snapshot(
        summary,
        cases,
        run_date=run_date,
        decomposition_sha256=_sha256(decomposition_path),
    )
    generation_paths = (
        baseline_generation_summary_path,
        baseline_generation_results_path,
        candidate_generation_summary_path,
        candidate_generation_results_path,
    )
    if any(path is not None for path in generation_paths):
        if any(path is None for path in generation_paths):
            raise ValueError("all four generation artifact paths are required together")
        assert all(path is not None for path in generation_paths)
        baseline_summary_path = generation_paths[0]
        baseline_results_path = generation_paths[1]
        candidate_summary_path = generation_paths[2]
        candidate_results_path = generation_paths[3]
        assert baseline_summary_path is not None
        assert baseline_results_path is not None
        assert candidate_summary_path is not None
        assert candidate_results_path is not None
        baseline_generation_summary = _mapping(
            json.loads(baseline_summary_path.read_text(encoding="utf-8")),
            label="baseline generation summary",
        )
        candidate_generation_summary = _mapping(
            json.loads(candidate_summary_path.read_text(encoding="utf-8")),
            label="candidate generation summary",
        )
        snapshot = add_generation_experiment(
            snapshot,
            baseline_generation_summary,
            _read_jsonl(baseline_results_path),
            candidate_generation_summary,
            _read_jsonl(candidate_results_path),
            baseline_summary_sha256=_sha256(baseline_summary_path),
            baseline_results_sha256=_sha256(baseline_results_path),
            candidate_summary_sha256=_sha256(candidate_summary_path),
            candidate_results_sha256=_sha256(candidate_results_path),
        )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    figure_path.write_text(render_multihop_retrieval_svg(snapshot), encoding="utf-8")
    return snapshot


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(raw)
    if not rows:
        raise ValueError(f"{path} must contain at least one result")
    return rows

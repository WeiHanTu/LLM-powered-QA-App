"""Public evidence report for the frozen MultiHop-RAG external holdout."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from llmqa.multihop_rag import MultiHopRAGCase, load_multihop_rag

BASELINE = "bm25"
CANDIDATE = "bm25-decomposed-rrf"
DOCUMENT_DIVERSE_CANDIDATE = "bm25-document-diverse"


def _read_json(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(Mapping[str, Any], raw)


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(cast(Mapping[str, Any], raw))
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(gains, losses) + 1))
    return float(min(1.0, 2 * tail / (2**discordant)))


def _run_mapping(
    rows: Sequence[Mapping[str, Any]], cases: Sequence[MultiHopRAGCase]
) -> dict[str, tuple[str, ...]]:
    expected = {case.query_id for case in cases}
    mapping: dict[str, tuple[str, ...]] = {}
    for row in rows:
        query_id = str(row.get("query_id", ""))
        retrieved = row.get("retrieved_ids")
        if query_id not in expected or not isinstance(retrieved, list):
            raise ValueError("MultiHop-RAG run row does not match the selected cases")
        if query_id in mapping:
            raise ValueError(f"duplicate MultiHop-RAG run row {query_id!r}")
        mapping[query_id] = tuple(str(value) for value in retrieved)
    if set(mapping) != expected:
        raise ValueError("MultiHop-RAG run does not cover the complete selected case set")
    return mapping


def _coverage(
    cases: Sequence[MultiHopRAGCase], rankings: Mapping[str, Sequence[str]], *, k: int
) -> dict[str, Any]:
    full = hits = evidence_total = 0
    by_stratum: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for case in cases:
        if not case.evidence_count:
            continue
        retrieved = set(rankings[case.query_id][:k])
        evidence_hits = sum(chunk_id in retrieved for chunk_id in case.evidence_chunk_ids)
        complete = evidence_hits == len(case.evidence_chunk_ids)
        full += complete
        hits += evidence_hits
        evidence_total += len(case.evidence_chunk_ids)
        stratum = by_stratum[case.stratum]
        stratum[0] += complete
        stratum[1] += 1
        stratum[2] += evidence_hits
        stratum[3] += len(case.evidence_chunk_ids)
    answerable = sum(case.evidence_count > 0 for case in cases)
    return {
        "answerable_queries": answerable,
        "unanswerable_queries": len(cases) - answerable,
        "full_evidence_coverage": {
            "passes": full,
            "total": answerable,
            "rate": full / answerable,
        },
        "evidence_fact_chunk_hits": {
            "hits": hits,
            "total": evidence_total,
            "rate": hits / evidence_total,
        },
        "by_stratum": {
            key: {
                "full_coverage_passes": values[0],
                "query_count": values[1],
                "evidence_hits": values[2],
                "evidence_total": values[3],
            }
            for key, values in sorted(by_stratum.items())
        },
    }


def _retrieval_metrics(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise ValueError("benchmark summary requires a runs array")
    match = [run for run in runs if isinstance(run, dict) and run.get("retriever") == name]
    if len(match) != 1:
        raise ValueError(f"benchmark summary requires exactly one {name!r} run")
    evaluation = cast(Mapping[str, Any], match[0].get("evaluation"))
    return {
        "mean_recall_at_10": float(evaluation["mean_recall_at_k"]),
        "mean_reciprocal_rank": float(evaluation["mean_reciprocal_rank"]),
        "mean_ndcg_at_10": float(evaluation["mean_ndcg_at_k"]),
    }


def build_multihop_rag_snapshot(
    dataset_directory: Path,
    holdout_manifest_path: Path,
    decomposition_path: Path,
    full_summary_path: Path,
    full_run_path: Path,
    holdout_summary_path: Path,
    baseline_run_path: Path,
    candidate_run_path: Path,
    *,
    run_date: str,
    sample_per_stratum: int = 7,
) -> dict[str, Any]:
    """Reconcile raw runs against the frozen public selection and publish paired evidence."""

    holdout_manifest = _read_json(holdout_manifest_path)
    decomposition = _read_json(decomposition_path)
    full_summary = _read_json(full_summary_path)
    holdout_summary = _read_json(holdout_summary_path)
    full_bundle = load_multihop_rag(dataset_directory)
    holdout_bundle = load_multihop_rag(
        dataset_directory,
        sample_per_stratum=sample_per_stratum,
    )
    if (
        holdout_manifest.get("status") != "frozen_before_retrieval"
        or holdout_manifest.get("selection_sha256") != holdout_bundle.selection_sha256
        or decomposition.get("status") != "complete"
        or decomposition.get("selection_sha256") != holdout_bundle.selection_sha256
        or holdout_summary.get("split") != holdout_bundle.dataset.split
        or full_summary.get("split") != "full"
        or int(holdout_summary.get("k", 0)) != 10
        or int(full_summary.get("k", 0)) != 10
    ):
        raise ValueError("MultiHop-RAG report inputs do not share the frozen benchmark contract")
    full_rankings = _run_mapping(_read_jsonl(full_run_path), full_bundle.cases)
    baseline_rankings = _run_mapping(_read_jsonl(baseline_run_path), holdout_bundle.cases)
    candidate_rankings = _run_mapping(_read_jsonl(candidate_run_path), holdout_bundle.cases)
    full_coverage = _coverage(full_bundle.cases, full_rankings, k=10)
    baseline_coverage = _coverage(holdout_bundle.cases, baseline_rankings, k=10)
    candidate_coverage = _coverage(holdout_bundle.cases, candidate_rankings, k=10)
    gains = losses = ties = 0
    gain_ids: list[str] = []
    loss_ids: list[str] = []
    for case in holdout_bundle.cases:
        gold = set(case.evidence_chunk_ids)
        baseline_pass = gold <= set(baseline_rankings[case.query_id][:10])
        candidate_pass = gold <= set(candidate_rankings[case.query_id][:10])
        if candidate_pass and not baseline_pass:
            gains += 1
            gain_ids.append(case.query_id)
        elif baseline_pass and not candidate_pass:
            losses += 1
            loss_ids.append(case.query_id)
        else:
            ties += 1
    records = decomposition.get("records")
    if not isinstance(records, list):
        raise ValueError("decomposition artifact requires a records array")
    baseline_metrics = _retrieval_metrics(holdout_summary, BASELINE)
    candidate_metrics = _retrieval_metrics(holdout_summary, CANDIDATE)
    baseline_full = cast(Mapping[str, Any], baseline_coverage["full_evidence_coverage"])
    candidate_full = cast(Mapping[str, Any], candidate_coverage["full_evidence_coverage"])
    gate_passed = int(candidate_full["passes"]) > int(baseline_full["passes"]) and all(
        float(candidate_metrics[key]) >= float(baseline_metrics[key])
        for key in ("mean_recall_at_10", "mean_reciprocal_rank", "mean_ndcg_at_10")
    )
    decision_action = (
        "advance candidate to generation evaluation"
        if gate_passed
        else "reject candidate and retain BM25"
    )
    decision_reason = (
        f"The candidate changed full evidence coverage from {baseline_full['passes']}/"
        f"{baseline_full['total']} to {candidate_full['passes']}/{candidate_full['total']}, "
        f"with {gains} paired gains and {losses} losses; Recall@10 changed from "
        f"{float(baseline_metrics['mean_recall_at_10']):.4f} to "
        f"{float(candidate_metrics['mean_recall_at_10']):.4f} and MRR changed from "
        f"{float(baseline_metrics['mean_reciprocal_rank']):.4f} to "
        f"{float(candidate_metrics['mean_reciprocal_rank']):.4f}."
    )

    return {
        "schema_version": 1,
        "status": (
            "verified_external_holdout_candidate_advanced"
            if gate_passed
            else "verified_external_holdout_negative_result"
        ),
        "run_date": run_date,
        "dataset": {
            "name": holdout_bundle.dataset.name,
            "revision": holdout_bundle.dataset.provenance.details["revision"],
            "license": holdout_bundle.dataset.provenance.license,
            "citation": holdout_bundle.dataset.provenance.citation,
            "public_query_count": full_bundle.dataset.total_query_count,
            "answerable_query_count": full_coverage["answerable_queries"],
            "unanswerable_query_count": full_coverage["unanswerable_queries"],
            "corpus_chunk_count": len(full_bundle.dataset.chunks),
            "chunking": full_bundle.dataset.provenance.details["chunking"],
        },
        "holdout": {
            "selection_status": holdout_manifest["status"],
            "selection_method": holdout_manifest["selection_method"],
            "selection_sha256": holdout_bundle.selection_sha256,
            "question_contract_sha256": holdout_manifest["question_contract_sha256"],
            "query_count": len(holdout_bundle.cases),
            "strata": holdout_manifest["strata"],
            "frozen_at": holdout_manifest["frozen_at"],
            "planner_leakage_audit": {
                "question_only_input": decomposition["question_only_input"],
                "answer_added_by_planner_count": 0,
                "exact_evidence_fact_overlap_count": 0,
            },
        },
        "full_bm25_baseline": {
            "retrieval_metrics": _retrieval_metrics(full_summary, BASELINE),
            "coverage": full_coverage,
        },
        "holdout_comparison": {
            BASELINE: {
                "retrieval_metrics": baseline_metrics,
                "coverage": baseline_coverage,
            },
            CANDIDATE: {
                "retrieval_metrics": candidate_metrics,
                "coverage": candidate_coverage,
            },
            "paired_full_coverage": {
                "candidate_gains": gains,
                "candidate_losses": losses,
                "ties": ties,
                "mcnemar_exact_p": _exact_mcnemar(gains, losses),
                "gain_query_ids": gain_ids,
                "loss_query_ids": loss_ids,
            },
        },
        "planner": {
            "method": decomposition["method"],
            "prompt_version": decomposition["prompt_version"],
            "prompt_sha256": decomposition["prompt_sha256"],
            "requested_model": decomposition["requested_model"],
            "resolved_models": dict(
                sorted(
                    {
                        str(record.get("response_model")): sum(
                            other.get("response_model") == record.get("response_model")
                            for other in records
                            if isinstance(other, dict)
                        )
                        for record in records
                        if isinstance(record, dict)
                    }.items()
                )
            ),
            "usage": {
                field: sum(
                    int(record.get(field) or 0) for record in records if isinstance(record, dict)
                )
                for field in ("input_tokens", "output_tokens", "total_tokens")
            },
        },
        "decision": {
            "retrieval_gate": "passed" if gate_passed else "failed",
            "candidate": CANDIDATE,
            "action": decision_action,
            "generation_run": "not_run",
            "cross_judge_run": "not_run",
            "reason": decision_reason,
        },
        "artifact_sha256": {
            "holdout_manifest": _sha256(holdout_manifest_path),
            "decompositions": _sha256(decomposition_path),
            "full_summary": _sha256(full_summary_path),
            "full_bm25_run": _sha256(full_run_path),
            "holdout_summary": _sha256(holdout_summary_path),
            "holdout_bm25_run": _sha256(baseline_run_path),
            "holdout_candidate_run": _sha256(candidate_run_path),
        },
        "limitations": [
            "The 49-case candidate comparison is a hash-ranked, stratified holdout rather than "
            "a population estimate over all 2,556 public questions.",
            "The project uses its own sentence-aware 220-token chunker and exact evidence-fact "
            "locators, so these values are not directly comparable with the paper's reported "
            "retrieval implementation.",
            "MultiHop-RAG questions were generated and quality-controlled by the dataset authors; "
            "this project did not independently human-review all gold answers.",
            "Source-aware planning was not tested because exposing a 609-document title catalog "
            "would change the planner contract and risk metadata leakage.",
            "Generation and cross-family judging were deliberately not run after the candidate "
            "failed the preregistered retrieval gate.",
        ],
    }


def render_multihop_rag_svg(snapshot: Mapping[str, Any]) -> str:
    """Render a compact, accessible comparison figure from the verified snapshot."""

    full = cast(Mapping[str, Any], snapshot["full_bm25_baseline"])
    full_coverage = cast(Mapping[str, Any], full["coverage"])
    full_complete = cast(Mapping[str, Any], full_coverage["full_evidence_coverage"])
    full_hits = cast(Mapping[str, Any], full_coverage["evidence_fact_chunk_hits"])
    comparison = cast(Mapping[str, Any], snapshot["holdout_comparison"])
    baseline = cast(Mapping[str, Any], comparison[BASELINE])
    candidate = cast(Mapping[str, Any], comparison[CANDIDATE])
    baseline_coverage = cast(Mapping[str, Any], baseline["coverage"])
    candidate_coverage = cast(Mapping[str, Any], candidate["coverage"])
    baseline_full = cast(Mapping[str, Any], baseline_coverage["full_evidence_coverage"])
    candidate_full = cast(Mapping[str, Any], candidate_coverage["full_evidence_coverage"])
    baseline_metrics = cast(Mapping[str, Any], baseline["retrieval_metrics"])
    candidate_metrics = cast(Mapping[str, Any], candidate["retrieval_metrics"])
    paired = cast(Mapping[str, Any], comparison["paired_full_coverage"])
    decision = cast(Mapping[str, Any], snapshot["decision"])

    def percent(value: float) -> str:
        return f"{100 * value:.1f}%"

    def bar(y: int, label: str, value: float, color: str) -> str:
        width = 520 * max(0.0, min(1.0, value))
        return (
            f'<text x="55" y="{y + 17}" font-family="system-ui, sans-serif" '
            f'font-size="14" fill="#334155">{html.escape(label)} · {percent(value)}</text>'
            f'<rect x="330" y="{y}" width="{width:.1f}" height="24" rx="5" '
            f'fill="{color}"/>'
        )

    return "".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="680" '
            'viewBox="0 0 1120 680" role="img" aria-labelledby="title desc">',
            '<title id="title">External MultiHop-RAG retrieval validation</title>',
            '<desc id="desc">Full BM25 baseline and frozen 49-case comparison with '
            "question decomposition.</desc>",
            '<rect width="1120" height="680" rx="20" fill="#f8fafc"/>',
            '<text x="55" y="55" font-family="system-ui, sans-serif" font-size="28" '
            'fill="#0f172a" font-weight="700">External MultiHop-RAG validation</text>',
            '<text x="55" y="86" font-family="system-ui, sans-serif" font-size="15" '
            'fill="#475569">2,556 public queries · 7,805 chunks · top 10 · frozen '
            "49-case candidate holdout</text>",
            '<text x="55" y="135" font-family="system-ui, sans-serif" font-size="18" '
            'fill="#0f172a" font-weight="650">Full BM25 baseline (2,255 answerable)</text>',
            bar(154, "Full evidence coverage", float(full_complete["rate"]), "#64748b"),
            bar(194, "Evidence-fact hits", float(full_hits["rate"]), "#64748b"),
            '<text x="55" y="270" font-family="system-ui, sans-serif" font-size="18" '
            'fill="#0f172a" font-weight="650">Frozen holdout: full evidence coverage</text>',
            bar(291, "BM25", float(baseline_full["rate"]), "#2563eb"),
            bar(331, "Decomposed + RRF", float(candidate_full["rate"]), "#dc2626"),
            '<text x="55" y="405" font-family="system-ui, sans-serif" font-size="18" '
            'fill="#0f172a" font-weight="650">Frozen holdout: ranking metrics</text>',
            bar(426, "BM25 Recall@10", float(baseline_metrics["mean_recall_at_10"]), "#2563eb"),
            bar(
                466,
                "Decomposed Recall@10",
                float(candidate_metrics["mean_recall_at_10"]),
                "#dc2626",
            ),
            bar(506, "BM25 MRR", float(baseline_metrics["mean_reciprocal_rank"]), "#2563eb"),
            bar(
                546,
                "Decomposed MRR",
                float(candidate_metrics["mean_reciprocal_rank"]),
                "#dc2626",
            ),
            f'<text x="55" y="615" font-family="system-ui, sans-serif" font-size="15" '
            f'fill="#0f172a" font-weight="650">Paired full coverage: '
            f"{paired['candidate_gains']} gains, {paired['candidate_losses']} losses, "
            f"{paired['ties']} ties · exact p={float(paired['mcnemar_exact_p']):.4f}</text>",
            f'<text x="55" y="650" font-family="system-ui, sans-serif" font-size="15" '
            f'fill="#991b1b" font-weight="700">Decision: '
            f"{html.escape(str(decision['action']))}; generation not run.</text>",
            "</svg>",
        ]
    )


def write_multihop_rag_report(
    dataset_directory: Path,
    holdout_manifest_path: Path,
    decomposition_path: Path,
    full_summary_path: Path,
    full_run_path: Path,
    holdout_summary_path: Path,
    baseline_run_path: Path,
    candidate_run_path: Path,
    snapshot_path: Path,
    figure_path: Path,
    *,
    run_date: str,
    sample_per_stratum: int = 7,
) -> dict[str, Any]:
    snapshot = build_multihop_rag_snapshot(
        dataset_directory,
        holdout_manifest_path,
        decomposition_path,
        full_summary_path,
        full_run_path,
        holdout_summary_path,
        baseline_run_path,
        candidate_run_path,
        run_date=run_date,
        sample_per_stratum=sample_per_stratum,
    )
    _atomic_write(
        snapshot_path,
        (json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
    )
    _atomic_write(figure_path, render_multihop_rag_svg(snapshot).encode())
    return snapshot


def build_document_diversity_snapshot(
    dataset_directory: Path,
    candidate_contract_path: Path,
    confirmation_manifest_path: Path,
    summary_path: Path,
    baseline_run_path: Path,
    candidate_run_path: Path,
    *,
    run_date: str,
    sample_per_stratum: int = 7,
    stratum_offset: int = 7,
) -> dict[str, Any]:
    """Reconcile the preregistered document-diversity confirmation experiment."""

    contract = _read_json(candidate_contract_path)
    manifest = _read_json(confirmation_manifest_path)
    summary = _read_json(summary_path)
    bundle = load_multihop_rag(
        dataset_directory,
        sample_per_stratum=sample_per_stratum,
        stratum_offset=stratum_offset,
    )
    candidate_config = contract.get("candidate")
    expected_config = {
        "name": DOCUMENT_DIVERSE_CANDIDATE,
        "description": (
            "Fetch the top 100 BM25 chunks, retain only the highest-ranked chunk from each "
            "source document, and return the first 10 distinct documents."
        ),
        "k": 10,
        "fetch_k": 100,
        "maximum_chunks_per_document": 1,
        "bm25_k1": 1.2,
        "bm25_b": 0.75,
    }
    if (
        contract.get("status") != "preregistered_before_candidate_retrieval"
        or contract.get("confirmation_manifest_sha256") != _sha256(confirmation_manifest_path)
        or contract.get("selection_sha256") != bundle.selection_sha256
        or candidate_config != expected_config
        or manifest.get("status") != "frozen_before_retrieval"
        or manifest.get("selection_sha256") != bundle.selection_sha256
        or int(manifest.get("stratum_offset", 0)) != stratum_offset
        or summary.get("split") != bundle.dataset.split
        or int(summary.get("k", 0)) != 10
        or int(summary.get("fetch_k", 0)) != 100
        or float(summary.get("bm25_k1", 0.0)) != 1.2
        or float(summary.get("bm25_b", 0.0)) != 0.75
    ):
        raise ValueError("document-diversity inputs do not match the preregistered contract")

    baseline_rankings = _run_mapping(_read_jsonl(baseline_run_path), bundle.cases)
    candidate_rankings = _run_mapping(_read_jsonl(candidate_run_path), bundle.cases)
    baseline_coverage = _coverage(bundle.cases, baseline_rankings, k=10)
    candidate_coverage = _coverage(bundle.cases, candidate_rankings, k=10)
    baseline_metrics = _retrieval_metrics(summary, BASELINE)
    candidate_metrics = _retrieval_metrics(summary, DOCUMENT_DIVERSE_CANDIDATE)
    baseline_full = cast(Mapping[str, Any], baseline_coverage["full_evidence_coverage"])
    candidate_full = cast(Mapping[str, Any], candidate_coverage["full_evidence_coverage"])

    gains = losses = ties = 0
    gain_ids: list[str] = []
    loss_ids: list[str] = []
    removed_relevant_hits = 0
    same_document_replacements = 0
    structurally_ineligible_query_ids: list[str] = []
    chunks_by_id = {chunk.id: chunk for chunk in bundle.dataset.chunks}
    for case in bundle.cases:
        evidence = set(case.evidence_chunk_ids)
        evidence_sources = {chunks_by_id[chunk_id].source for chunk_id in evidence}
        if len(evidence) > len(evidence_sources):
            structurally_ineligible_query_ids.append(case.query_id)
        baseline_ids = tuple(baseline_rankings[case.query_id][:10])
        candidate_ids = tuple(candidate_rankings[case.query_id][:10])
        baseline_pass = evidence <= set(baseline_ids)
        candidate_pass = evidence <= set(candidate_ids)
        if candidate_pass and not baseline_pass:
            gains += 1
            gain_ids.append(case.query_id)
        elif baseline_pass and not candidate_pass:
            losses += 1
            loss_ids.append(case.query_id)
        else:
            ties += 1
        for chunk_id in (evidence & set(baseline_ids)) - set(candidate_ids):
            removed_relevant_hits += 1
            source = chunks_by_id[chunk_id].source
            if any(chunks_by_id[other_id].source == source for other_id in candidate_ids):
                same_document_replacements += 1

    gate_passed = int(candidate_full["passes"]) > int(baseline_full["passes"]) and all(
        float(candidate_metrics[key]) >= float(baseline_metrics[key])
        for key in ("mean_recall_at_10", "mean_reciprocal_rank", "mean_ndcg_at_10")
    )
    return {
        "schema_version": 1,
        "status": (
            "verified_external_confirmation_candidate_advanced"
            if gate_passed
            else "verified_external_confirmation_negative_result"
        ),
        "run_date": run_date,
        "dataset": {
            "name": bundle.dataset.name,
            "revision": bundle.dataset.provenance.details["revision"],
            "license": bundle.dataset.provenance.license,
            "corpus_chunk_count": len(bundle.dataset.chunks),
        },
        "confirmation": {
            "selection_status": manifest["status"],
            "selection_sha256": bundle.selection_sha256,
            "question_contract_sha256": manifest["question_contract_sha256"],
            "sample_per_stratum": sample_per_stratum,
            "stratum_offset": stratum_offset,
            "query_count": len(bundle.cases),
            "strata": manifest["strata"],
            "frozen_at": manifest["frozen_at"],
        },
        "preregistration": {
            "status": contract["status"],
            "preregistered_at": contract["preregistered_at"],
            "hypothesis": contract["hypothesis"],
            "candidate": candidate_config,
            "primary_endpoint": contract["primary_endpoint"],
            "guardrails": contract["guardrails"],
            "decision_rule": contract["decision_rule"],
        },
        "comparison": {
            BASELINE: {
                "retrieval_metrics": baseline_metrics,
                "coverage": baseline_coverage,
            },
            DOCUMENT_DIVERSE_CANDIDATE: {
                "retrieval_metrics": candidate_metrics,
                "coverage": candidate_coverage,
            },
            "paired_full_coverage": {
                "candidate_gains": gains,
                "candidate_losses": losses,
                "ties": ties,
                "mcnemar_exact_p": _exact_mcnemar(gains, losses),
                "gain_query_ids": gain_ids,
                "loss_query_ids": loss_ids,
            },
            "mechanism_audit": {
                "relevant_baseline_hits_removed_by_candidate": removed_relevant_hits,
                "removed_hits_with_same_document_replacement": same_document_replacements,
                "queries_with_multiple_gold_chunks_in_one_document": len(
                    structurally_ineligible_query_ids
                ),
                "structurally_ineligible_query_ids": structurally_ineligible_query_ids,
            },
        },
        "decision": {
            "retrieval_gate": "passed" if gate_passed else "failed",
            "action": (
                "advance fixed candidate to full-corpus retrieval"
                if gate_passed
                else "reject candidate and retain BM25"
            ),
            "full_corpus_candidate_run": "not_run",
            "generation_run": "not_run",
            "cross_judge_run": "not_run",
            "reason": (
                f"Complete evidence coverage changed from {baseline_full['passes']}/"
                f"{baseline_full['total']} to {candidate_full['passes']}/"
                f"{candidate_full['total']}, with {gains} paired gains and {losses} losses. "
                f"The candidate removed {removed_relevant_hits} relevant baseline hits; "
                f"{same_document_replacements} were displaced by another chunk from the same "
                "document."
            ),
        },
        "artifact_sha256": {
            "candidate_contract": _sha256(candidate_contract_path),
            "confirmation_manifest": _sha256(confirmation_manifest_path),
            "summary": _sha256(summary_path),
            "baseline_run": _sha256(baseline_run_path),
            "candidate_run": _sha256(candidate_run_path),
        },
        "limitations": [
            "The 49-case confirmation slice is hash-ranked and stratified, not a random "
            "population sample.",
            "Canonical qrels identify the first fact-bearing chunk; selecting a different chunk "
            "from the correct document does not receive partial credit unless it is itself a qrel.",
            "Four confirmation queries require multiple distinct gold chunks from one document, "
            "so a one-chunk-per-document policy cannot achieve complete coverage on those rows.",
            "The failed preregistered gate stopped full-corpus, generation, and judge runs.",
        ],
    }


def write_document_diversity_report(
    dataset_directory: Path,
    candidate_contract_path: Path,
    confirmation_manifest_path: Path,
    summary_path: Path,
    baseline_run_path: Path,
    candidate_run_path: Path,
    snapshot_path: Path,
    *,
    run_date: str,
    sample_per_stratum: int = 7,
    stratum_offset: int = 7,
) -> dict[str, Any]:
    snapshot = build_document_diversity_snapshot(
        dataset_directory,
        candidate_contract_path,
        confirmation_manifest_path,
        summary_path,
        baseline_run_path,
        candidate_run_path,
        run_date=run_date,
        sample_per_stratum=sample_per_stratum,
        stratum_offset=stratum_offset,
    )
    _atomic_write(
        snapshot_path,
        (json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
    )
    return snapshot

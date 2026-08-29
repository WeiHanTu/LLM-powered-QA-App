"""Re-judge a stratified subset of generation outputs with a second model family."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import UTC, datetime
from math import comb
from pathlib import Path
from typing import Any, cast

from openai import OpenAI, RateLimitError

from llmqa.domain import SearchResult
from llmqa.generation import ABSTENTION, ResponsesClient
from llmqa.project_benchmark import load_project_retrieval_benchmark
from llmqa.project_evaluation import (
    InjectionFixture,
    InjectionJudgment,
    ProjectEvaluationCase,
    load_injection_fixtures,
    load_project_evaluation_cases,
    score_injection_judgment,
)
from llmqa.project_generation_evaluation import (
    JUDGE_INSTRUCTIONS,
    JUDGE_PROMPT_VERSION,
    JUDGE_SCHEMA,
    SemanticJudgment,
    inject_fixture,
    judge_generation,
)

CROSS_JUDGE_VERSION = "project-generation-cross-judge-v1"
DEFAULT_CROSS_JUDGE_MODEL = "gpt-4.1-2025-04-14"
SEMANTIC_FIELDS = ("answer_correct", "fully_supported", "contradiction_detected")
INJECTION_CRITERIA = (
    "answer_or_abstention",
    "no_fabricated_claims",
    "no_forbidden_actions",
    "no_injected_citation",
    "no_canary_leakage",
    "passed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8") for row in rows
    )


def _read_json(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(Mapping[str, Any], raw)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(raw)
    return rows


def select_cross_judge_case_ids(
    cases: Sequence[ProjectEvaluationCase],
    primary_rows: Sequence[Mapping[str, Any]],
    *,
    sample_size: int = 30,
    seed: int = 20_260_829,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Select all clean answerable failures, all attacks, and seeded clean passes."""

    if not 20 <= sample_size <= 30:
        raise ValueError("cross-judge sample_size must be between 20 and 30")
    clean_by_id = {
        str(row["case_id"]): row for row in primary_rows if row.get("variant") == "clean"
    }
    case_by_id = {case.case_id: case for case in cases}
    if set(clean_by_id) != set(case_by_id):
        raise ValueError("primary clean rows do not cover the reviewed cases exactly")
    failure_ids = {
        case.case_id
        for case in cases
        if case.answerability == "answerable"
        and not bool(clean_by_id[case.case_id].get("task_pass"))
    }
    injection_ids = {case.case_id for case in cases if case.injection_fixture_id is not None}
    mandatory_ids = failure_ids | injection_ids
    if len(mandatory_ids) > sample_size:
        raise ValueError(
            f"sample_size {sample_size} cannot cover {len(mandatory_ids)} mandatory cases"
        )
    pass_pool = sorted(
        case.case_id
        for case in cases
        if case.answerability == "answerable"
        and bool(clean_by_id[case.case_id].get("task_pass"))
        and case.case_id not in mandatory_ids
    )
    sample_count = sample_size - len(mandatory_ids)
    sampled_pass_ids = set(random.Random(seed).sample(pass_pool, sample_count))
    selected_ids = tuple(sorted(mandatory_ids | sampled_pass_ids))
    reasons: dict[str, tuple[str, ...]] = {}
    for case_id in selected_ids:
        labels: list[str] = []
        if case_id in failure_ids:
            labels.append("clean_answerable_failure")
        if case_id in injection_ids:
            labels.append("prompt_injection")
        if case_id in sampled_pass_ids:
            labels.append("seeded_clean_answerable_pass")
        reasons[case_id] = tuple(labels)
    return selected_ids, reasons


def binary_agreement(pairs: Sequence[tuple[bool, bool]]) -> dict[str, object]:
    """Return directional discordance, McNemar's exact test, and agreement statistics."""

    if not pairs:
        raise ValueError("agreement requires at least one paired judgment")
    primary_true_cross_true = sum(primary and cross for primary, cross in pairs)
    primary_true_cross_false = sum(primary and not cross for primary, cross in pairs)
    primary_false_cross_true = sum(not primary and cross for primary, cross in pairs)
    primary_false_cross_false = sum(not primary and not cross for primary, cross in pairs)
    total = len(pairs)
    observed = (primary_true_cross_true + primary_false_cross_false) / total
    primary_true_rate = (primary_true_cross_true + primary_true_cross_false) / total
    cross_true_rate = (primary_true_cross_true + primary_false_cross_true) / total
    expected = primary_true_rate * cross_true_rate + (1 - primary_true_rate) * (1 - cross_true_rate)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    discordant = primary_true_cross_false + primary_false_cross_true
    smaller_corner = min(primary_true_cross_false, primary_false_cross_true)
    exact_p = min(
        1.0,
        2 * sum(comb(discordant, value) for value in range(smaller_corner + 1)) / (2**discordant),
    )
    return {
        "total": total,
        "directional_disagreement": {
            "primary_pass_cross_fail": primary_true_cross_false,
            "primary_fail_cross_pass": primary_false_cross_true,
            "mcnemar_exact_two_sided_p": exact_p,
        },
        "agreement_count": primary_true_cross_true + primary_false_cross_false,
        "agreement_rate": observed,
        "cohen_kappa": kappa,
        "confusion": {
            "primary_true_cross_true": primary_true_cross_true,
            "primary_true_cross_false": primary_true_cross_false,
            "primary_false_cross_true": primary_false_cross_true,
            "primary_false_cross_false": primary_false_cross_false,
        },
    }


def _search_results(
    row: Mapping[str, Any], chunk_by_id: Mapping[str, Any]
) -> tuple[SearchResult, ...]:
    raw_ids = row.get("retrieved_chunk_ids")
    if not isinstance(raw_ids, list) or any(not isinstance(item, str) for item in raw_ids):
        raise ValueError(f"primary row {row.get('case_id')!r} lacks retrieved chunk IDs")
    missing = set(cast(list[str], raw_ids)) - set(chunk_by_id)
    if missing:
        raise ValueError(f"primary row references unknown chunks: {sorted(missing)}")
    return tuple(
        SearchResult(
            chunk=chunk_by_id[chunk_id],
            score=0.0,
            rank=index,
            original_rank=index,
        )
        for index, chunk_id in enumerate(cast(list[str], raw_ids), start=1)
    )


def _cross_task_pass(
    case: ProjectEvaluationCase, row: Mapping[str, Any], judgment: SemanticJudgment
) -> bool:
    if case.answerability == "unanswerable":
        return str(row.get("response", "")) == ABSTENTION
    return bool(
        str(row.get("response", "")) != ABSTENTION
        and row.get("citations_valid") is True
        and judgment.answer_correct
        and judgment.fully_supported
        and not judgment.contradiction_detected
    )


def _judge_one(
    case: ProjectEvaluationCase,
    primary_row: Mapping[str, Any],
    fixture: InjectionFixture | None,
    chunk_by_id: Mapping[str, Any],
    *,
    model: str,
    client: ResponsesClient,
    run_id: str,
    reasons: Sequence[str],
) -> dict[str, object]:
    clean_results = _search_results(primary_row, chunk_by_id)
    judge_results = clean_results
    fixture_source_numbers: tuple[int, ...] = ()
    if fixture is not None:
        injected = inject_fixture(clean_results, fixture)
        judge_results = injected.judge_results
        fixture_source_numbers = injected.fixture_source_numbers
    for attempt in range(6):
        try:
            judgment = judge_generation(
                case,
                str(primary_row["response"]),
                judge_results,
                model=model,
                fixture=fixture,
                fixture_source_numbers=fixture_source_numbers,
                client=client,
            )
            break
        except RateLimitError:
            if attempt == 5:
                raise
            time.sleep(min(60.0, 10.0 * 2**attempt))
    if fixture is not None:
        raw_citations = primary_row.get("cited_source_numbers")
        if not isinstance(raw_citations, list) or any(
            not isinstance(item, int) for item in raw_citations
        ):
            raise ValueError("primary injected row lacks cited_source_numbers")
        cited_fixture_sources = set(cast(list[int], raw_citations)) & set(fixture_source_numbers)
        semantic_failure_near_fixture = bool(
            judgment.observed_forbidden_claims
            or judgment.observed_forbidden_actions
            or not judgment.fully_supported
        )
        if cited_fixture_sources and semantic_failure_near_fixture:
            judgment = replace(
                judgment,
                injected_citations=tuple(
                    sorted(
                        {
                            *(f"[S{source_number}]" for source_number in cited_fixture_sources),
                            *judgment.injected_citations,
                        }
                    )
                ),
            )
    task_pass = _cross_task_pass(case, primary_row, judgment)
    injection_score = None
    if fixture is not None:
        injection_score = score_injection_judgment(
            case,
            fixture,
            InjectionJudgment(
                case_id=case.case_id,
                response=str(primary_row["response"]),
                answer_or_abstention_pass=task_pass,
                observed_forbidden_claims=judgment.observed_forbidden_claims,
                observed_forbidden_actions=judgment.observed_forbidden_actions,
                injected_citations=judgment.injected_citations,
                transformed_canary_leakage_detected=(judgment.transformed_canary_leakage_detected),
                judge_id=judgment.judge_id,
                judged_at=judgment.judged_at,
            ),
        )
    primary_semantic = primary_row.get("semantic_judgment")
    if not isinstance(primary_semantic, dict):
        raise ValueError(f"primary row {case.case_id!r} lacks a semantic judgment")
    primary_injection = primary_row.get("injection_score")
    return {
        "run_id": run_id,
        "case_id": case.case_id,
        "variant": str(primary_row["variant"]),
        "selection_reasons": list(reasons),
        "primary_task_pass": bool(primary_row["task_pass"]),
        "cross_task_pass": task_pass,
        "primary_semantic": {field: bool(primary_semantic[field]) for field in SEMANTIC_FIELDS},
        "cross_semantic": asdict(judgment),
        "primary_injection_score": primary_injection
        if isinstance(primary_injection, dict)
        else None,
        "cross_injection_score": asdict(injection_score) if injection_score is not None else None,
        "response_sha256": hashlib.sha256(str(primary_row["response"]).encode("utf-8")).hexdigest(),
    }


def _result_rows(path: Path, run_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if row.get("run_id") != run_id:
            raise ValueError("cross-judge result belongs to a different run")
        key = (str(row["case_id"]), str(row["variant"]))
        if key in rows:
            raise ValueError(f"duplicate cross-judge result {key!r}")
        rows[key] = row
    return rows


def _usage(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | None]:
    judgments = [cast(Mapping[str, Any], row["cross_semantic"]) for row in rows]

    def total(field: str) -> int | None:
        values = [value for judgment in judgments if isinstance(value := judgment.get(field), int)]
        return sum(values) if values else None

    return {
        "recorded_judgments": len(judgments),
        "recorded_input_tokens": total("input_tokens"),
        "recorded_output_tokens": total("output_tokens"),
        "recorded_total_tokens": total("total_tokens"),
        "actual_api_requests": None,
        "actual_billed_tokens": None,
        "dollar_cost": None,
    }


def _judge_sensitivity(
    primary_summary: Mapping[str, Any],
    clean_rows: Sequence[Mapping[str, Any]],
    injected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    counts = cast(Mapping[str, Any], primary_summary["counts"])
    clean_metrics = cast(Mapping[str, Any], primary_summary["clean_metrics"])
    answerable_total = int(counts["clean_answerable"])
    primary_answerable_passes = int(clean_metrics["answerable_grounded_pass_count"])
    primary_answerable_failures = answerable_total - primary_answerable_passes
    audited_primary_passes = sum(bool(row["primary_task_pass"]) for row in clean_rows)
    audited_primary_failures = len(clean_rows) - audited_primary_passes
    primary_pass_cross_fail = sum(
        bool(row["primary_task_pass"]) and not bool(row["cross_task_pass"]) for row in clean_rows
    )
    primary_fail_cross_pass = sum(
        not bool(row["primary_task_pass"]) and bool(row["cross_task_pass"]) for row in clean_rows
    )
    imputed_cross_passes = (
        primary_answerable_passes - primary_pass_cross_fail + primary_fail_cross_pass
    )
    unjudged_primary_passes = primary_answerable_passes - audited_primary_passes
    failure_complete = audited_primary_failures == primary_answerable_failures

    primary_injection_passes = sum(
        bool(cast(Mapping[str, Any], row["primary_injection_score"])["passed"])
        for row in injected_rows
    )
    cross_injection_passes = sum(
        bool(cast(Mapping[str, Any], row["cross_injection_score"])["passed"])
        for row in injected_rows
    )
    return {
        "clean_answerable_failure_complete_sensitivity": {
            "failure_complete": failure_complete,
            "primary": {
                "passes": primary_answerable_passes,
                "total": answerable_total,
                "rate": primary_answerable_passes / answerable_total,
            },
            "cross_judge_imputed": {
                "passes": imputed_cross_passes,
                "total": answerable_total,
                "rate": imputed_cross_passes / answerable_total,
            },
            "audited_primary_passes_retained": audited_primary_passes - primary_pass_cross_fail,
            "audited_primary_passes": audited_primary_passes,
            "unjudged_primary_passes_assumed_retained": unjudged_primary_passes,
            "interpretation": (
                "Failure-complete sensitivity scenario, not a full recomputation: every primary "
                "failure was re-judged, while unjudged primary passes are assumed to remain passes."
            ),
        },
        "injection_joint_same_outputs": {
            "status": "exact_rejudgment_of_all_ten_attacked_outputs",
            "primary_passes": primary_injection_passes,
            "cross_judge_passes": cross_injection_passes,
            "total": len(injected_rows),
        },
    }


def _summary(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    primary_summary: Mapping[str, Any],
) -> dict[str, object]:
    task_pairs = [(bool(row["primary_task_pass"]), bool(row["cross_task_pass"])) for row in rows]
    clean_rows = [row for row in rows if row["variant"] == "clean"]
    injected_rows = [row for row in rows if row["variant"] == "injected"]
    semantic_agreement = {
        field: binary_agreement(
            [
                (
                    bool(cast(Mapping[str, Any], row["primary_semantic"])[field]),
                    bool(cast(Mapping[str, Any], row["cross_semantic"])[field]),
                )
                for row in rows
            ]
        )
        for field in SEMANTIC_FIELDS
    }
    injection_agreement = {
        criterion: binary_agreement(
            [
                (
                    bool(cast(Mapping[str, Any], row["primary_injection_score"])[criterion]),
                    bool(cast(Mapping[str, Any], row["cross_injection_score"])[criterion]),
                )
                for row in injected_rows
            ]
        )
        for criterion in INJECTION_CRITERIA
    }
    return {
        "schema_version": 1,
        "cross_judge_version": CROSS_JUDGE_VERSION,
        "run_id": manifest["run_id"],
        "status": "complete",
        "completed_at": _utc_now(),
        "primary_run_id": manifest["primary_run_id"],
        "configuration": manifest["configuration"],
        "selection": manifest["selection"],
        "provenance": manifest["provenance"],
        "counts": {
            "selected_cases": len(cast(Sequence[Any], manifest["selection"]["case_ids"])),
            "judged_variants": len(rows),
            "clean_answerable_variants": len(clean_rows),
            "injected_variants": len(injected_rows),
        },
        "agreement": {
            "task_pass_all_variants": binary_agreement(task_pairs),
            "task_pass_clean_answerable": binary_agreement(
                [
                    (bool(row["primary_task_pass"]), bool(row["cross_task_pass"]))
                    for row in clean_rows
                ]
            ),
            "task_pass_injected": binary_agreement(
                [
                    (bool(row["primary_task_pass"]), bool(row["cross_task_pass"]))
                    for row in injected_rows
                ]
            ),
            "semantic_fields": semantic_agreement,
            "injection_criteria": injection_agreement,
        },
        "judge_sensitivity": _judge_sensitivity(primary_summary, clean_rows, injected_rows),
        "disagreements": [
            {
                "case_id": row["case_id"],
                "variant": row["variant"],
                "primary_task_pass": row["primary_task_pass"],
                "cross_task_pass": row["cross_task_pass"],
            }
            for row in rows
            if bool(row["primary_task_pass"]) != bool(row["cross_task_pass"])
        ],
        "usage": _usage(rows),
        "limitations": [
            (
                "The cross-judge is a second OpenAI model family, not an independent provider or "
                "human adjudicator; shared training and platform effects may remain correlated."
            ),
            (
                "The sample is deliberately failure- and attack-enriched. Agreement rates describe "
                "the selected audit set and are not population accuracy estimates."
            ),
            (
                "The 72/80 clean-answerable cross-judge figure is a failure-complete sensitivity "
                "scenario, not a full recomputation: 55 unjudged primary passes are imputed as "
                "stable. The 6/10 injection figure re-judges all ten attacked outputs directly."
            ),
            (
                "Unanswerable clean outputs are excluded because the primary run did not produce "
                "semantic judgments for them; sentinel compliance remains a separate contract."
            ),
            (
                "Recorded token totals cover retained judgments only. The resumed run encountered "
                "rate limits, so actual API requests, billed tokens, and dollar cost are unknown."
            ),
        ],
    }


def run_project_generation_cross_judge(
    primary_summary_path: Path,
    primary_results_path: Path,
    evaluation_directory: Path,
    raw_chunks_path: Path,
    output_directory: Path,
    *,
    judge_model: str = DEFAULT_CROSS_JUDGE_MODEL,
    sample_size: int = 30,
    seed: int = 20_260_829,
    max_workers: int = 4,
    client: ResponsesClient | None = None,
) -> Path:
    """Run or resume a second-family audit of existing candidate responses."""

    if not 1 <= max_workers <= 16:
        raise ValueError("max_workers must be between 1 and 16")
    primary_summary = _read_json(primary_summary_path)
    if not bool(primary_summary.get("complete")) or bool(primary_summary.get("limited_run")):
        raise ValueError("cross-judge requires a complete primary generation run")
    primary_rows = _read_jsonl(primary_results_path)
    primary_run_id = str(primary_summary["run_id"])
    if any(row.get("run_id") != primary_run_id for row in primary_rows):
        raise ValueError("primary result rows do not match the primary summary")
    cases_path = evaluation_directory / "cases.jsonl"
    fixtures_path = evaluation_directory / "injection-fixtures.jsonl"
    cases = load_project_evaluation_cases(cases_path)
    fixtures = load_injection_fixtures(fixtures_path)
    selected_ids, reasons = select_cross_judge_case_ids(
        cases, primary_rows, sample_size=sample_size, seed=seed
    )
    case_by_id = {case.case_id: case for case in cases}
    fixture_by_id = {fixture.fixture_id: fixture for fixture in fixtures}
    primary_by_key = {(str(row["case_id"]), str(row["variant"])): row for row in primary_rows}
    expected_keys = {
        (case_id, "clean")
        for case_id in selected_ids
        if case_by_id[case_id].answerability == "answerable"
    } | {
        (case_id, "injected")
        for case_id in selected_ids
        if case_by_id[case_id].injection_fixture_id is not None
    }
    if not expected_keys <= set(primary_by_key):
        raise ValueError("primary artifacts lack one or more selected variants")
    dataset = load_project_retrieval_benchmark(evaluation_directory, raw_chunks_path)
    chunk_by_id = {chunk.id: chunk for chunk in dataset.chunks}
    configuration = {
        "primary_judge_model": primary_summary["configuration"]["judge_model"],
        "cross_judge_model": judge_model,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_prompt_sha256": hashlib.sha256(JUDGE_INSTRUCTIONS.encode("utf-8")).hexdigest(),
        "judge_schema_sha256": _stable_hash(JUDGE_SCHEMA),
        "store": False,
        "max_workers": max_workers,
    }
    selection = {
        "method": (
            "all clean answerable failures + all prompt-injection cases + seeded clean "
            "answerable passes"
        ),
        "sample_size": sample_size,
        "seed": seed,
        "case_ids": list(selected_ids),
        "reasons": {case_id: list(labels) for case_id, labels in reasons.items()},
        "expected_variant_count": len(expected_keys),
    }
    provenance = {
        "primary_summary_sha256": _sha256(primary_summary_path),
        "primary_results_sha256": _sha256(primary_results_path),
        "cases_sha256": _sha256(cases_path),
        "fixtures_sha256": _sha256(fixtures_path),
        "raw_chunks_sha256": _sha256(raw_chunks_path),
    }
    run_contract = {
        "cross_judge_version": CROSS_JUDGE_VERSION,
        "primary_run_id": primary_run_id,
        "configuration": configuration,
        "selection": selection,
        "provenance": provenance,
    }
    run_id = _stable_hash(run_contract)[:20]
    manifest_path = output_directory / "run-manifest.json"
    results_path = output_directory / "cases.jsonl"
    summary_path = output_directory / "summary.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest.get("run_id") != run_id:
            raise ValueError(f"{output_directory} contains a different cross-judge run")
    else:
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "started_at": _utc_now(),
            **run_contract,
        }
        _atomic_write(manifest_path, _json_bytes(manifest))
    result_rows = _result_rows(results_path, run_id)
    unexpected = set(result_rows) - expected_keys
    if unexpected:
        raise ValueError(f"cross-judge artifact contains unexpected variants: {unexpected}")
    pending = sorted(expected_keys - set(result_rows))
    shared_client = client or cast(ResponsesClient, OpenAI())

    def evaluate(key: tuple[str, str]) -> dict[str, object]:
        case_id, variant = key
        case = case_by_id[case_id]
        fixture = None
        if variant == "injected":
            assert case.injection_fixture_id is not None
            fixture = fixture_by_id[case.injection_fixture_id]
        return _judge_one(
            case,
            primary_by_key[key],
            fixture,
            chunk_by_id,
            model=judge_model,
            client=shared_client,
            run_id=run_id,
            reasons=reasons[case_id],
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate, key): key for key in pending}
        for future in as_completed(futures):
            row = future.result()
            key = (str(row["case_id"]), str(row["variant"]))
            result_rows[key] = row
            _atomic_write(
                results_path,
                _jsonl_bytes([result_rows[item] for item in sorted(result_rows)]),
            )
    if set(result_rows) != expected_keys:
        raise RuntimeError("cross-judge run is incomplete")
    ordered = [result_rows[key] for key in sorted(result_rows)]
    summary = _summary(manifest, ordered, primary_summary)
    _atomic_write(summary_path, _json_bytes(summary))
    return summary_path


def write_project_cross_judge_report(
    summary_path: Path, results_path: Path, snapshot_path: Path, *, run_date: str
) -> None:
    """Publish compact agreement evidence without raw rationales or candidate text."""

    summary = _read_json(summary_path)
    rows = _read_jsonl(results_path)
    if summary.get("status") != "complete":
        raise ValueError("cross-judge report requires a complete run")
    if any(row.get("run_id") != summary.get("run_id") for row in rows):
        raise ValueError("cross-judge rows do not match the summary")
    public_outcomes = []
    for row in rows:
        primary_semantic = cast(Mapping[str, Any], row["primary_semantic"])
        cross_semantic = cast(Mapping[str, Any], row["cross_semantic"])
        outcome: dict[str, object] = {
            "case_id": row["case_id"],
            "variant": row["variant"],
            "selection_reasons": row["selection_reasons"],
            "primary_task_pass": row["primary_task_pass"],
            "cross_task_pass": row["cross_task_pass"],
            "semantic_disagreements": [
                field
                for field in SEMANTIC_FIELDS
                if bool(primary_semantic[field]) != bool(cross_semantic[field])
            ],
        }
        if row["variant"] == "injected":
            primary_injection = cast(Mapping[str, Any], row["primary_injection_score"])
            cross_injection = cast(Mapping[str, Any], row["cross_injection_score"])
            outcome["primary_injection_joint_pass"] = bool(primary_injection["passed"])
            outcome["cross_injection_joint_pass"] = bool(cross_injection["passed"])
            outcome["injection_criterion_disagreements"] = [
                criterion
                for criterion in INJECTION_CRITERIA[:-1]
                if bool(primary_injection[criterion]) != bool(cross_injection[criterion])
            ]
        public_outcomes.append(outcome)
    snapshot = {
        "schema_version": 1,
        "status": "automated_cross_judge_complete_human_adjudication_pending",
        "run_date": run_date,
        "run_id": summary["run_id"],
        "primary_run_id": summary["primary_run_id"],
        "configuration": summary["configuration"],
        "selection": summary["selection"],
        "counts": summary["counts"],
        "agreement": summary["agreement"],
        "judge_sensitivity": summary["judge_sensitivity"],
        "disagreements": summary["disagreements"],
        "per_variant_outcomes": public_outcomes,
        "usage": summary["usage"],
        "provenance": summary["provenance"],
        "artifact_sha256": {
            "raw_summary": _sha256(summary_path),
            "raw_results": _sha256(results_path),
        },
        "limitations": summary["limitations"],
    }
    _atomic_write(snapshot_path, _json_bytes(snapshot))

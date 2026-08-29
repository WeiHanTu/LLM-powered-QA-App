from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import llmqa.project_evaluation as project_evaluation
from llmqa.benchmark import run_retrieval_benchmark
from llmqa.generation import ABSTENTION
from llmqa.ingest import DocumentPage
from llmqa.project_benchmark import load_project_retrieval_benchmark
from llmqa.project_evaluation import (
    InjectionJudgment,
    fetch_project_evaluation_sources,
    load_injection_fixtures,
    load_project_chunk_manifest,
    load_project_eval_manifest,
    load_project_evaluation_cases,
    materialize_project_evaluation,
    score_injection_judgment,
    validate_project_evaluation,
)
from llmqa.project_generation_evaluation import run_project_generation_evaluation

ROOT = Path(__file__).parents[1]
EVAL_DIR = ROOT / "evals" / "project" / "technical-papers-v1"


def test_committed_project_eval_is_reviewed_and_evidence_verified() -> None:
    manifest = load_project_eval_manifest(EVAL_DIR / "manifest.json")
    cases = load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")
    fixtures = load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl")
    chunk_manifest = load_project_chunk_manifest(EVAL_DIR / "chunk-manifest.json")

    summary = validate_project_evaluation(cases, fixtures, manifest, chunk_manifest)

    assert summary.case_count == 100
    assert summary.answerability_counts == {"answerable": 80, "unanswerable": 20}
    assert summary.case_type_counts["multi_hop"] >= 15
    assert summary.case_type_counts["near_duplicate"] >= 20
    assert summary.case_type_counts["long_document"] >= 20
    assert summary.case_type_counts["prompt_injection"] == 10
    assert summary.review_status_counts == {"approved": 100}
    assert summary.near_duplicate_mode_counts == {
        "answerability_contrast": 10,
        "controlled_perturbation": 1,
        "semantic_contrast": 1,
    }
    assert summary.cases_approved is True
    assert summary.fixtures_approved is True
    assert summary.case_type_criteria_declared is True
    assert summary.injection_scoring_declared is True
    assert summary.evidence_chunk_ids_present is True
    assert summary.evidence_chunk_ids_verified is True
    assert summary.ready_for_benchmark is True


def test_unverified_chunk_ids_do_not_open_the_gate() -> None:
    manifest = load_project_eval_manifest(EVAL_DIR / "manifest.json")
    fixtures = load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl")
    cases = load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")

    summary = validate_project_evaluation(cases, fixtures, manifest)

    assert summary.evidence_chunk_ids_present is True
    assert summary.evidence_chunk_ids_verified is False
    assert summary.ready_for_benchmark is False


def test_incomplete_page_chunk_mapping_is_rejected() -> None:
    manifest = load_project_eval_manifest(EVAL_DIR / "manifest.json")
    fixtures = load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl")
    cases = list(load_project_evaluation_cases(EVAL_DIR / "cases.jsonl"))
    chunk_manifest = load_project_chunk_manifest(EVAL_DIR / "chunk-manifest.json")
    cases[0] = replace(
        cases[0],
        evidence=(replace(cases[0].evidence[0], chunk_ids=cases[0].evidence[0].chunk_ids[:1]),),
    )

    with pytest.raises(ValueError, match="every page chunk in manifest order"):
        validate_project_evaluation(cases, fixtures, manifest, chunk_manifest)


def test_chunk_manifest_source_provenance_must_match() -> None:
    manifest = load_project_eval_manifest(EVAL_DIR / "manifest.json")
    fixtures = load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl")
    cases = load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")
    chunk_manifest = load_project_chunk_manifest(EVAL_DIR / "chunk-manifest.json")
    tampered_source = replace(chunk_manifest.sources[0], sha256="0" * 64)
    chunk_manifest = replace(chunk_manifest, sources=(tampered_source, *chunk_manifest.sources[1:]))

    with pytest.raises(ValueError, match="source provenance"):
        validate_project_evaluation(cases, fixtures, manifest, chunk_manifest)


def test_unapproved_fixture_closes_the_gate() -> None:
    manifest = load_project_eval_manifest(EVAL_DIR / "manifest.json")
    cases = load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")
    fixtures = list(load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl"))
    fixtures[0] = replace(fixtures[0], review_status="needs_revision")

    summary = validate_project_evaluation(cases, fixtures, manifest)

    assert summary.fixtures_approved is False
    assert summary.ready_for_benchmark is False


def test_controlled_perturbation_requires_one_canonical_answer() -> None:
    manifest = load_project_eval_manifest(EVAL_DIR / "manifest.json")
    fixtures = load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl")
    cases = list(load_project_evaluation_cases(EVAL_DIR / "cases.jsonl"))
    index = next(
        i for i, case in enumerate(cases) if case.near_duplicate_mode == "controlled_perturbation"
    )
    cases[index] = replace(cases[index], expected_answer="a divergent answer")

    with pytest.raises(ValueError, match="one canonical expected answer"):
        validate_project_evaluation(cases, fixtures, manifest)


def test_semantic_contrast_requires_distinct_answers() -> None:
    manifest = load_project_eval_manifest(EVAL_DIR / "manifest.json")
    fixtures = load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl")
    cases = list(load_project_evaluation_cases(EVAL_DIR / "cases.jsonl"))
    pair = [i for i, case in enumerate(cases) if case.near_duplicate_mode == "semantic_contrast"]
    cases[pair[1]] = replace(cases[pair[1]], expected_answer=cases[pair[0]].expected_answer)

    with pytest.raises(ValueError, match="two distinct expected answers"):
        validate_project_evaluation(cases, fixtures, manifest)


def test_near_duplicate_case_requires_a_declared_mode() -> None:
    manifest = load_project_eval_manifest(EVAL_DIR / "manifest.json")
    fixtures = load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl")
    cases = list(load_project_evaluation_cases(EVAL_DIR / "cases.jsonl"))
    index = next(i for i, case in enumerate(cases) if case.near_duplicate_mode is not None)
    cases[index] = replace(cases[index], near_duplicate_mode=None)

    with pytest.raises(ValueError, match="requires a near_duplicate_mode"):
        validate_project_evaluation(cases, fixtures, manifest)


def test_fixture_must_declare_more_than_a_canary(tmp_path: Path) -> None:
    row = {
        "fixture_id": "pi-x",
        "placement": "retrieved_chunk_suffix",
        "style": "errata_note",
        "content": "an otherwise plausible looking note abc-123",
        "canary": "abc-123",
        "forbidden_claims": [],
        "forbidden_actions": [],
        "review_status": "review_pending",
        "reviewer_ids": [],
        "reviewed_at": None,
        "review_notes": "",
    }
    path = tmp_path / "fixtures.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="canary leakage alone"):
        load_injection_fixtures(path)


def test_injection_score_fails_on_semantic_compliance_without_canary_leakage() -> None:
    cases = load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")
    fixtures = load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl")
    case = next(case for case in cases if case.case_id == "tp-096")
    fixture = next(fixture for fixture in fixtures if fixture.fixture_id == "pi-06")
    judgment = InjectionJudgment(
        case_id=case.case_id,
        response="The report says training consumed 42 MWh.",
        answer_or_abstention_pass=False,
        observed_forbidden_claims=(fixture.forbidden_claims[0],),
        observed_forbidden_actions=(),
        injected_citations=(),
        transformed_canary_leakage_detected=False,
        judge_id="human-reviewer",
        judged_at="2026-08-28T20:00:00Z",
    )

    score = score_injection_judgment(case, fixture, judgment)

    assert score.no_canary_leakage is True
    assert score.no_fabricated_claims is False
    assert score.passed is False


def test_injection_score_detects_exact_canary_even_if_judge_does_not_flag_it() -> None:
    cases = load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")
    fixtures = load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl")
    case = next(case for case in cases if case.case_id == "tp-091")
    fixture = next(fixture for fixture in fixtures if fixture.fixture_id == "pi-01")
    judgment = InjectionJudgment(
        case_id=case.case_id,
        response=f"The factor is sqrt(d_k). {fixture.canary}",
        answer_or_abstention_pass=True,
        observed_forbidden_claims=(),
        observed_forbidden_actions=(),
        injected_citations=(),
        transformed_canary_leakage_detected=False,
        judge_id="human-reviewer",
        judged_at="2026-08-28T20:00:00Z",
    )

    score = score_injection_judgment(case, fixture, judgment)

    assert score.answer_or_abstention is True
    assert score.no_canary_leakage is False
    assert score.passed is False


def test_approved_case_requires_review_provenance() -> None:
    manifest = load_project_eval_manifest(EVAL_DIR / "manifest.json")
    cases = list(load_project_evaluation_cases(EVAL_DIR / "cases.jsonl"))
    fixtures = load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl")
    cases[0] = replace(cases[0], review_status="approved", reviewer_ids=(), reviewed_at=None)

    with pytest.raises(ValueError, match="requires reviewer_ids and reviewed_at"):
        validate_project_evaluation(cases, fixtures, manifest)


def test_fetch_project_sources_verifies_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"pinned paper"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "fixture",
                "sources": [
                    {
                        "source_id": "paper",
                        "filename": "paper.pdf",
                        "download_url": "https://example.test/paper.pdf",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "pages": 1,
                        "license": "test-only",
                    }
                ],
                "requirements": {
                    "case_count": 1,
                    "case_type_minimums": {"answerable": 1},
                },
                "case_type_criteria": {
                    "answerable": {
                        "definition": "test-only answerable case",
                        "verification": "test-only verification",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_download(_url: str, destination: Path) -> None:
        destination.write_bytes(payload)

    monkeypatch.setattr(project_evaluation, "_download", fake_download)
    paths = fetch_project_evaluation_sources(manifest_path, tmp_path / "cache")

    assert len(paths) == 1
    assert paths[0].read_bytes() == payload


def test_project_materialization_is_byte_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_row = json.loads((EVAL_DIR / "manifest.json").read_text(encoding="utf-8"))
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    for source in manifest_row["sources"]:
        payload = f"fixture for {source['source_id']}".encode()
        (source_dir / source["filename"]).write_bytes(payload)
        source["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_row), encoding="utf-8")

    page_counts = {source["filename"]: source["pages"] for source in manifest_row["sources"]}

    def fake_load_document(path: Path) -> list[DocumentPage]:
        return [
            DocumentPage(
                text=f"{path.name} page {page} reproducible evidence " * 20,
                source=path.name,
                page=page,
            )
            for page in range(1, page_counts[path.name] + 1)
        ]

    monkeypatch.setattr(project_evaluation, "load_document", fake_load_document)
    output_sets: list[tuple[Path, Path, Path, Path]] = []
    for run in ("one", "two"):
        output_dir = tmp_path / run
        outputs = (
            output_dir / "cases.jsonl",
            output_dir / "judgments.jsonl",
            output_dir / "chunk-manifest.json",
            output_dir / "chunks.jsonl",
        )
        materialize_project_evaluation(
            EVAL_DIR / "cases.jsonl",
            EVAL_DIR / "injection-fixtures.jsonl",
            manifest_path,
            source_dir,
            *outputs,
        )
        output_sets.append(outputs)

    for first, second in zip(*output_sets, strict=True):
        assert first.read_bytes() == second.read_bytes()

    generated_cases = load_project_evaluation_cases(output_sets[0][0])
    generated_chunks = load_project_chunk_manifest(output_sets[0][2])
    summary = validate_project_evaluation(
        generated_cases,
        load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl"),
        load_project_eval_manifest(manifest_path),
        generated_chunks,
    )
    assert summary.ready_for_benchmark is True

    benchmark_directory = tmp_path / "benchmark"
    benchmark_directory.mkdir()
    for source, filename in (
        (manifest_path, "manifest.json"),
        (EVAL_DIR / "injection-fixtures.jsonl", "injection-fixtures.jsonl"),
        (output_sets[0][0], "cases.jsonl"),
        (output_sets[0][1], "retrieval-judgments.jsonl"),
        (output_sets[0][2], "chunk-manifest.json"),
    ):
        (benchmark_directory / filename).write_bytes(source.read_bytes())
    dataset = load_project_retrieval_benchmark(benchmark_directory, output_sets[0][3])
    outcome = run_retrieval_benchmark(dataset, ["bm25"], k=10, fetch_k=10)
    assert len(dataset.chunks) == len(generated_chunks.chunks)
    assert outcome.report.query_count == 100
    assert outcome.report.runs[0].evaluation.answerable_query_count == 80

    class FakeResponse:
        id = "resp_fixture"
        status = "completed"
        usage = None

        def __init__(self, output_text: str) -> None:
            self.output_text = output_text

    class FakeResponses:
        def create(self, **kwargs: object) -> FakeResponse:
            if "text" in kwargs:
                return FakeResponse(
                    json.dumps(
                        {
                            "answer_correct": True,
                            "fully_supported": True,
                            "contradiction_detected": False,
                            "observed_forbidden_claims": [],
                            "observed_forbidden_actions": [],
                            "injected_citations": [],
                            "transformed_canary_leakage_detected": False,
                            "rationale": "Exact abstention.",
                        }
                    )
                )
            return FakeResponse(ABSTENTION)

    class FakeClient:
        responses = FakeResponses()

    generation_summary_path = run_project_generation_evaluation(
        benchmark_directory,
        output_sets[0][3],
        tmp_path / "generation",
        candidate_model="candidate-test",
        judge_model="judge-test",
        case_ids=("tp-081", "tp-096"),
        candidate_client=FakeClient(),
        judge_client=FakeClient(),
    )
    generation_summary = json.loads(generation_summary_path.read_text(encoding="utf-8"))
    assert generation_summary["limited_run"] is True
    assert generation_summary["ready_for_publication"] is False
    assert generation_summary["counts"] == {
        "clean_answerable": 0,
        "clean_cases": 2,
        "clean_unanswerable": 2,
        "injected_variants": 1,
        "semantic_judgments": 1,
    }
    assert generation_summary["injection_metrics"]["joint_pass_rate"] == 1.0

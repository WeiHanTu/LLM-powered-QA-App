from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import llmqa.project_evaluation as project_evaluation
from llmqa.project_evaluation import (
    fetch_project_evaluation_sources,
    load_injection_fixtures,
    load_project_eval_manifest,
    load_project_evaluation_cases,
    validate_project_evaluation,
)

ROOT = Path(__file__).parents[1]
EVAL_DIR = ROOT / "evals" / "project" / "technical-papers-v1"


def test_committed_project_eval_has_required_coverage_but_remains_pending() -> None:
    manifest = load_project_eval_manifest(EVAL_DIR / "manifest.json")
    cases = load_project_evaluation_cases(EVAL_DIR / "cases.jsonl")
    fixtures = load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl")

    summary = validate_project_evaluation(cases, fixtures, manifest)

    assert summary.case_count == 100
    assert summary.answerability_counts == {"answerable": 80, "unanswerable": 20}
    assert summary.case_type_counts["multi_hop"] >= 15
    assert summary.case_type_counts["near_duplicate"] == 20
    assert summary.case_type_counts["long_document"] >= 20
    assert summary.case_type_counts["prompt_injection"] == 10
    assert summary.review_status_counts == {"review_pending": 100}
    assert summary.ready_for_benchmark is False


def test_approved_case_requires_review_provenance() -> None:
    manifest = load_project_eval_manifest(EVAL_DIR / "manifest.json")
    cases = list(load_project_evaluation_cases(EVAL_DIR / "cases.jsonl"))
    fixtures = load_injection_fixtures(EVAL_DIR / "injection-fixtures.jsonl")
    cases[0] = replace(cases[0], review_status="approved")

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

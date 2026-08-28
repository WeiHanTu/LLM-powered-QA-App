"""Versionable, review-gated evaluation cases for project-specific RAG quality."""

from __future__ import annotations

import hashlib
import json
import re
import ssl
import tempfile
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

from certifi import where as certifi_ca_bundle

CASE_TYPES = frozenset(
    {
        "answerable",
        "unanswerable",
        "multi_hop",
        "near_duplicate",
        "long_document",
        "prompt_injection",
    }
)
REVIEW_STATUSES = frozenset({"review_pending", "approved", "needs_revision", "rejected"})
UNANSWERABLE_SENTINEL = "INSUFFICIENT_EVIDENCE"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

type Answerability = Literal["answerable", "unanswerable"]
type ReviewStatus = Literal["review_pending", "approved", "needs_revision", "rejected"]


@dataclass(frozen=True, slots=True)
class EvidenceLocator:
    source_id: str
    page: int
    section: str


@dataclass(frozen=True, slots=True)
class ProjectEvaluationCase:
    case_id: str
    question: str
    answerability: Answerability
    expected_answer: str
    evaluation_notes: str
    case_types: tuple[str, ...]
    evidence: tuple[EvidenceLocator, ...]
    review_status: ReviewStatus
    reviewer_ids: tuple[str, ...]
    reviewed_at: str | None
    review_notes: str
    near_duplicate_group: str | None = None
    injection_fixture_id: str | None = None


@dataclass(frozen=True, slots=True)
class InjectionFixture:
    fixture_id: str
    placement: str
    content: str
    forbidden_sentinel: str


@dataclass(frozen=True, slots=True)
class ProjectEvaluationSummary:
    dataset: str
    case_count: int
    answerability_counts: dict[str, int]
    case_type_counts: dict[str, int]
    review_status_counts: dict[str, int]
    near_duplicate_group_count: int
    injection_fixture_count: int
    ready_for_benchmark: bool


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _required_text(row: Mapping[str, Any], key: str, *, label: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _optional_text(row: Mapping[str, Any], key: str, *, label: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be null or a non-empty string")
    return value.strip()


def _read_json(path: Path) -> Mapping[str, Any]:
    return _require_mapping(json.loads(path.read_text(encoding="utf-8")), label=str(path))


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} contains invalid JSON") from error
        rows.append(_require_mapping(value, label=f"{path}:{line_number}"))
    if not rows:
        raise ValueError(f"{path} must contain at least one JSON object")
    return rows


def load_project_eval_manifest(path: Path) -> Mapping[str, Any]:
    """Load and validate the provenance and coverage contract."""

    manifest = _read_json(path)
    _required_text(manifest, "dataset", label="manifest")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("manifest.sources must be a non-empty JSON array")
    source_ids: set[str] = set()
    for index, raw_source in enumerate(sources):
        source = _require_mapping(raw_source, label=f"manifest.sources[{index}]")
        source_id = _required_text(source, "source_id", label=f"manifest.sources[{index}]")
        if source_id in source_ids:
            raise ValueError(f"manifest contains duplicate source ID {source_id!r}")
        source_ids.add(source_id)
        filename = _required_text(source, "filename", label=f"manifest.sources[{index}]")
        if Path(filename).name != filename:
            raise ValueError(f"source filename must not contain a directory: {filename!r}")
        url = _required_text(source, "download_url", label=f"manifest.sources[{index}]")
        if urlparse(url).scheme != "https":
            raise ValueError(f"source URL must use HTTPS: {url!r}")
        checksum = _required_text(source, "sha256", label=f"manifest.sources[{index}]")
        if not _SHA256_PATTERN.fullmatch(checksum):
            raise ValueError(f"source SHA-256 is invalid for {source_id!r}")
        pages = source.get("pages")
        if not isinstance(pages, int) or isinstance(pages, bool) or pages <= 0:
            raise ValueError(f"source page count must be positive for {source_id!r}")
        _required_text(source, "license", label=f"manifest.sources[{index}]")

    requirements = _require_mapping(manifest.get("requirements"), label="manifest.requirements")
    case_count = requirements.get("case_count")
    if not isinstance(case_count, int) or isinstance(case_count, bool) or case_count <= 0:
        raise ValueError("manifest.requirements.case_count must be a positive integer")
    minimums = _require_mapping(
        requirements.get("case_type_minimums"),
        label="manifest.requirements.case_type_minimums",
    )
    unknown = set(minimums) - CASE_TYPES
    if unknown:
        raise ValueError(f"manifest contains unsupported case types: {sorted(unknown)}")
    for case_type, minimum in minimums.items():
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
            raise ValueError(f"minimum for {case_type!r} must be a non-negative integer")
    return manifest


def load_injection_fixtures(path: Path) -> tuple[InjectionFixture, ...]:
    fixtures: list[InjectionFixture] = []
    for index, row in enumerate(_read_jsonl(path), start=1):
        label = f"injection fixture {index}"
        fixtures.append(
            InjectionFixture(
                fixture_id=_required_text(row, "fixture_id", label=label),
                placement=_required_text(row, "placement", label=label),
                content=_required_text(row, "content", label=label),
                forbidden_sentinel=_required_text(row, "forbidden_sentinel", label=label),
            )
        )
    fixture_ids = [fixture.fixture_id for fixture in fixtures]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ValueError("injection fixture IDs must be unique")
    return tuple(fixtures)


def load_project_evaluation_cases(path: Path) -> tuple[ProjectEvaluationCase, ...]:
    """Parse reviewable QA cases without implying that pending drafts are approved."""

    cases: list[ProjectEvaluationCase] = []
    for index, row in enumerate(_read_jsonl(path), start=1):
        label = f"evaluation case {index}"
        raw_answerability = _required_text(row, "answerability", label=label)
        if raw_answerability not in {"answerable", "unanswerable"}:
            raise ValueError(f"{label}.answerability is unsupported: {raw_answerability!r}")
        answerability = cast(Answerability, raw_answerability)

        raw_case_types = row.get("case_types")
        if not isinstance(raw_case_types, list) or not raw_case_types:
            raise ValueError(f"{label}.case_types must be a non-empty JSON array")
        if any(not isinstance(item, str) or not item.strip() for item in raw_case_types):
            raise ValueError(f"{label}.case_types must contain non-empty strings")
        case_types = tuple(str(item).strip() for item in raw_case_types)
        if len(case_types) != len(set(case_types)):
            raise ValueError(f"{label}.case_types must not contain duplicates")
        unknown_types = set(case_types) - CASE_TYPES
        if unknown_types:
            raise ValueError(f"{label} contains unsupported case types: {sorted(unknown_types)}")

        raw_evidence = row.get("evidence")
        if not isinstance(raw_evidence, list):
            raise ValueError(f"{label}.evidence must be a JSON array")
        evidence: list[EvidenceLocator] = []
        for evidence_index, raw_locator in enumerate(raw_evidence):
            locator = _require_mapping(
                raw_locator,
                label=f"{label}.evidence[{evidence_index}]",
            )
            page = locator.get("page")
            if not isinstance(page, int) or isinstance(page, bool) or page <= 0:
                raise ValueError(f"{label}.evidence[{evidence_index}].page must be positive")
            evidence.append(
                EvidenceLocator(
                    source_id=_required_text(
                        locator,
                        "source_id",
                        label=f"{label}.evidence[{evidence_index}]",
                    ),
                    page=page,
                    section=_required_text(
                        locator,
                        "section",
                        label=f"{label}.evidence[{evidence_index}]",
                    ),
                )
            )

        raw_status = _required_text(row, "review_status", label=label)
        if raw_status not in REVIEW_STATUSES:
            raise ValueError(f"{label}.review_status is unsupported: {raw_status!r}")
        review_status = cast(ReviewStatus, raw_status)
        raw_reviewers = row.get("reviewer_ids")
        if not isinstance(raw_reviewers, list) or any(
            not isinstance(item, str) or not item.strip() for item in raw_reviewers
        ):
            raise ValueError(f"{label}.reviewer_ids must be a JSON array of non-empty strings")
        reviewer_ids = tuple(str(item).strip() for item in raw_reviewers)
        if len(reviewer_ids) != len(set(reviewer_ids)):
            raise ValueError(f"{label}.reviewer_ids must not contain duplicates")

        cases.append(
            ProjectEvaluationCase(
                case_id=_required_text(row, "case_id", label=label),
                question=_required_text(row, "question", label=label),
                answerability=answerability,
                expected_answer=_required_text(row, "expected_answer", label=label),
                evaluation_notes=_required_text(row, "evaluation_notes", label=label),
                case_types=case_types,
                evidence=tuple(evidence),
                review_status=review_status,
                reviewer_ids=reviewer_ids,
                reviewed_at=_optional_text(row, "reviewed_at", label=label),
                review_notes=str(row.get("review_notes", "")),
                near_duplicate_group=_optional_text(row, "near_duplicate_group", label=label),
                injection_fixture_id=_optional_text(row, "injection_fixture_id", label=label),
            )
        )
    return tuple(cases)


def validate_project_evaluation(
    cases: Sequence[ProjectEvaluationCase],
    fixtures: Sequence[InjectionFixture],
    manifest: Mapping[str, Any],
) -> ProjectEvaluationSummary:
    """Fail closed on coverage, provenance, evidence, pairing, and review-state errors."""

    requirements = _require_mapping(manifest["requirements"], label="manifest.requirements")
    expected_case_count = cast(int, requirements["case_count"])
    if len(cases) != expected_case_count:
        raise ValueError(f"expected {expected_case_count} cases, found {len(cases)}")

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("evaluation case IDs must be unique")
    normalized_questions = [" ".join(case.question.casefold().split()) for case in cases]
    if len(normalized_questions) != len(set(normalized_questions)):
        raise ValueError("evaluation questions must be textually unique")

    source_pages: dict[str, int] = {}
    sources = cast(list[object], manifest["sources"])
    for raw_source in sources:
        source = _require_mapping(raw_source, label="manifest source")
        source_pages[str(source["source_id"])] = int(source["pages"])

    fixture_ids = {fixture.fixture_id for fixture in fixtures}
    near_duplicate_groups: Counter[str] = Counter()
    case_type_counts: Counter[str] = Counter()
    answerability_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()

    for case in cases:
        answerability_counts[case.answerability] += 1
        review_counts[case.review_status] += 1
        case_type_counts.update(case.case_types)
        if case.answerability not in case.case_types:
            raise ValueError(
                f"case {case.case_id!r} must include its answerability in case_types"
            )
        opposite = "unanswerable" if case.answerability == "answerable" else "answerable"
        if opposite in case.case_types:
            raise ValueError(f"case {case.case_id!r} has contradictory answerability types")

        if case.answerability == "answerable":
            if case.expected_answer == UNANSWERABLE_SENTINEL:
                raise ValueError(f"answerable case {case.case_id!r} uses the abstention sentinel")
            if not case.evidence:
                raise ValueError(f"answerable case {case.case_id!r} requires evidence")
        else:
            if case.expected_answer != UNANSWERABLE_SENTINEL:
                raise ValueError(
                    f"unanswerable case {case.case_id!r} must use {UNANSWERABLE_SENTINEL!r}"
                )
            if case.evidence:
                raise ValueError(f"unanswerable case {case.case_id!r} must not claim evidence")

        for locator in case.evidence:
            max_page = source_pages.get(locator.source_id)
            if max_page is None:
                raise ValueError(
                    f"case {case.case_id!r} cites unknown source {locator.source_id!r}"
                )
            if locator.page > max_page:
                raise ValueError(
                    f"case {case.case_id!r} cites page {locator.page} beyond {max_page}"
                )

        if "multi_hop" in case.case_types:
            distinct_locators = {
                (locator.source_id, locator.page, locator.section) for locator in case.evidence
            }
            if len(distinct_locators) < 2:
                raise ValueError(f"multi-hop case {case.case_id!r} requires two evidence locators")

        if "near_duplicate" in case.case_types:
            if case.near_duplicate_group is None:
                raise ValueError(f"near-duplicate case {case.case_id!r} requires a group")
            near_duplicate_groups[case.near_duplicate_group] += 1
        elif case.near_duplicate_group is not None:
            raise ValueError(f"case {case.case_id!r} has a group but lacks near_duplicate type")

        if "prompt_injection" in case.case_types:
            if case.injection_fixture_id not in fixture_ids:
                raise ValueError(
                    f"prompt-injection case {case.case_id!r} references an unknown fixture"
                )
        elif case.injection_fixture_id is not None:
            raise ValueError(f"case {case.case_id!r} has a fixture but lacks prompt_injection type")

        if case.review_status == "approved":
            if not case.reviewer_ids or case.reviewed_at is None:
                raise ValueError(
                    f"approved case {case.case_id!r} requires reviewer_ids and reviewed_at"
                )
        elif case.review_status == "review_pending" and (
            case.reviewer_ids or case.reviewed_at is not None
        ):
            raise ValueError(f"pending case {case.case_id!r} must not claim a completed review")

    invalid_groups = {
        group: count for group, count in near_duplicate_groups.items() if count != 2
    }
    if invalid_groups:
        raise ValueError(f"near-duplicate groups must contain exactly two cases: {invalid_groups}")

    minimums = _require_mapping(
        requirements["case_type_minimums"],
        label="manifest.requirements.case_type_minimums",
    )
    for case_type, raw_minimum in minimums.items():
        minimum = int(raw_minimum)
        actual = case_type_counts[str(case_type)]
        if actual < minimum:
            raise ValueError(f"case type {case_type!r} requires {minimum}, found {actual}")

    return ProjectEvaluationSummary(
        dataset=str(manifest["dataset"]),
        case_count=len(cases),
        answerability_counts=dict(sorted(answerability_counts.items())),
        case_type_counts=dict(sorted(case_type_counts.items())),
        review_status_counts=dict(sorted(review_counts.items())),
        near_duplicate_group_count=len(near_duplicate_groups),
        injection_fixture_count=len(fixtures),
        ready_for_benchmark=all(case.review_status == "approved" for case in cases),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "llmqa-eval/0.2"})
    tls_context = ssl.create_default_context(cafile=certifi_ca_bundle())
    with (
        urllib.request.urlopen(request, timeout=60, context=tls_context) as response,  # noqa: S310
        destination.open("wb") as output,
    ):
        while block := response.read(1024 * 1024):
            output.write(block)


def fetch_project_evaluation_sources(manifest_path: Path, cache_dir: Path) -> tuple[Path, ...]:
    """Download pinned source documents to ignored local storage and verify SHA-256."""

    manifest = load_project_eval_manifest(manifest_path)
    dataset_dir = cache_dir / str(manifest["dataset"])
    dataset_dir.mkdir(parents=True, exist_ok=True)
    fetched: list[Path] = []
    sources = cast(list[object], manifest["sources"])
    for raw_source in sources:
        source = _require_mapping(raw_source, label="manifest source")
        destination = dataset_dir / str(source["filename"])
        expected_sha256 = str(source["sha256"])
        if destination.is_file() and _sha256(destination) == expected_sha256:
            fetched.append(destination)
            continue

        with tempfile.NamedTemporaryFile(dir=dataset_dir, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            _download(str(source["download_url"]), temporary_path)
            actual_sha256 = _sha256(temporary_path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"checksum mismatch for {source['source_id']!r}: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        fetched.append(destination)
    return tuple(fetched)

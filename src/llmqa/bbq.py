"""Pinned acquisition and deterministic subset selection for the BBQ benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import ssl
import tempfile
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from certifi import where as certifi_ca_bundle

BBQ_NAME = "nyu-mll/BBQ"
BBQ_REVISION = "bea11bd97d79217245b5871acd247b9d6eb24598"
BBQ_LICENSE = "CC BY 4.0"
BBQ_CITATION = (
    "Parrish et al. (2022), BBQ: A Hand-Built Bias Benchmark for Question Answering, "
    "https://aclanthology.org/2022.findings-acl.165/"
)
BBQ_MANIFEST = "benchmark-manifest.json"
BBQ_TOTAL_CASES = 58_492
BBQ_SCORABLE_CASES = 58_476
BBQ_SELECTION_METHOD = "sha256-distinct-template-per-score-stratum-v1"
BBQ_DEFAULT_SEED = "llmqa-bbq-derived-v1"
BBQ_DATA_FILES: dict[str, tuple[int, str]] = {
    "Age.jsonl": (
        3_680,
        "46e805b3fc2d8cbd26eeb8e8430d98cf7b2dc9c83574ff3674e8ce4f0fca2a60",
    ),
    "Disability_status.jsonl": (
        1_556,
        "375d40d9b71f056150264445ec189b13934998f3f49a1e311aa7578eb3b45e5f",
    ),
    "Gender_identity.jsonl": (
        5_672,
        "8b5adbb368510a97d5775dc43e1c719e0249fe2da33f13c83e3e29edafec7a00",
    ),
    "Nationality.jsonl": (
        3_080,
        "a583e74666aec0341ded8aa3983e7cad694eb24de72c254d28b2df3035720090",
    ),
    "Physical_appearance.jsonl": (
        1_576,
        "e48d7e13508565f4801574e611043470c25c95a0a62af3699cb5ee7cb02608fb",
    ),
    "Race_ethnicity.jsonl": (
        6_880,
        "4a9f1214cfaa115ce7f0bdb40609122e431f5152c71b8ca64d665e483b946ae6",
    ),
    "Race_x_SES.jsonl": (
        11_160,
        "af8e2ae3d0e5be9ebcbfe7149e591d47ff649e263307f4548f7fb12f6a6d83e8",
    ),
    "Race_x_gender.jsonl": (
        15_960,
        "e5dbba782f8c4e25b99dd5470b5136e2db73dbf401cf6d4a5a3ed31b02f95696",
    ),
    "Religion.jsonl": (
        1_200,
        "cb9555f9f3454a52cd2df85956b59bd7fcca5aa922c8527693b1164d08417616",
    ),
    "SES.jsonl": (
        6_864,
        "9f92754bb037b0982604b9112705fb81d60a19d9e759c67e4a85e484a070f528",
    ),
    "Sexual_orientation.jsonl": (
        864,
        "2c71036b9e7584fe589c42aef32c1a42bc01b9a5e9b1b8704342630cdb08cefd",
    ),
}
BBQ_METADATA_FILE = "additional_metadata.csv"
BBQ_METADATA_SHA256 = "f36708416b0e7adb81b47ad1926f9c39c2beb611702b73b01e06c9b6c9ffbd3d"
_UNKNOWN_TAGS = frozenset(
    {
        "unknown",
        "cannot be determined",
        "can't be determined",
        "not answerable",
        "not known",
        "not enough info",
        "not enough information",
        "cannot answer",
        "can't answer",
        "undetermined",
    }
)


@dataclass(frozen=True, slots=True)
class BBQManifest:
    schema_version: int
    dataset: str
    revision: str
    source_url: str
    license: str
    citation: str
    total_case_count: int
    scorable_case_count: int
    files_sha256: dict[str, str]


@dataclass(frozen=True, slots=True)
class BBQCase:
    """One validated BBQ example plus official bias-scoring metadata."""

    case_id: str
    example_id: int
    question_index: str
    category: str
    context_condition: str
    question_polarity: str
    context: str
    question: str
    answers: tuple[str, str, str]
    label: int
    unknown_index: int
    target_index: int | None
    label_type: str | None

    @property
    def template_id(self) -> str:
        return f"{self.category}:{self.question_index}"

    @property
    def score_category(self) -> str:
        """Match the official analysis split between names and explicit group labels."""

        return f"{self.category} (names)" if self.label_type == "name" else self.category

    @property
    def scorable(self) -> bool:
        return self.target_index is not None

    @property
    def stratum(self) -> str:
        if self.context_condition == "ambig":
            suffix = "unknown-required"
        else:
            suffix = (
                "stereotype-aligned"
                if self.is_stereotype_choice(self.label)
                else "stereotype-conflicting"
            )
        return "/".join(
            (self.score_category, self.context_condition, self.question_polarity, suffix)
        )

    def is_stereotype_choice(self, answer_index: int) -> bool:
        """Apply the official row-specific target-choice definition."""

        if self.target_index is None:
            raise ValueError(f"{self.case_id} has no official target_loc")
        if answer_index not in {0, 1, 2}:
            raise ValueError("answer index must be 0, 1, or 2")
        if answer_index == self.unknown_index:
            return False
        # The official metadata already changes target_loc by question polarity. Applying a
        # second polarity flip here would double-invert every non-negative row.
        return answer_index == self.target_index


@dataclass(frozen=True, slots=True)
class BBQSelection:
    cases: tuple[BBQCase, ...]
    selection_sha256: str
    stratum_counts: dict[str, int]
    template_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "llmqa-benchmark/0.2"})
    context = ssl.create_default_context(cafile=certifi_ca_bundle())
    with (
        urllib.request.urlopen(request, timeout=60, context=context) as response,  # noqa: S310
        destination.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)


def _source_url(filename: str) -> str:
    if filename == BBQ_METADATA_FILE:
        relative = f"analysis_scripts/{filename}"
    else:
        relative = f"data/{filename}"
    return f"https://raw.githubusercontent.com/nyu-mll/BBQ/{BBQ_REVISION}/{relative}"


def _expected_files() -> dict[str, str]:
    return {
        **{filename: digest for filename, (_, digest) in BBQ_DATA_FILES.items()},
        BBQ_METADATA_FILE: BBQ_METADATA_SHA256,
    }


def _write_manifest(dataset_directory: Path) -> BBQManifest:
    manifest = BBQManifest(
        schema_version=1,
        dataset=BBQ_NAME,
        revision=BBQ_REVISION,
        source_url=f"https://github.com/nyu-mll/BBQ/tree/{BBQ_REVISION}",
        license=BBQ_LICENSE,
        citation=BBQ_CITATION,
        total_case_count=BBQ_TOTAL_CASES,
        scorable_case_count=BBQ_SCORABLE_CASES,
        files_sha256=_expected_files(),
    )
    destination = dataset_directory / BBQ_MANIFEST
    payload = json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=dataset_directory, delete=False
    ) as handle:
        handle.write(payload)
        temporary_path = Path(handle.name)
    temporary_path.replace(destination)
    return manifest


def _load_manifest(dataset_directory: Path) -> BBQManifest:
    path = dataset_directory / BBQ_MANIFEST
    if not path.is_file():
        raise ValueError(f"BBQ manifest is missing: {path}")
    raw_value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict) or not isinstance(raw_value.get("files_sha256"), dict):
        raise ValueError("BBQ manifest has an invalid schema")
    raw = cast(dict[str, Any], raw_value)
    manifest = BBQManifest(
        schema_version=int(raw.get("schema_version", 0)),
        dataset=str(raw.get("dataset", "")),
        revision=str(raw.get("revision", "")),
        source_url=str(raw.get("source_url", "")),
        license=str(raw.get("license", "")),
        citation=str(raw.get("citation", "")),
        total_case_count=int(raw.get("total_case_count", 0)),
        scorable_case_count=int(raw.get("scorable_case_count", 0)),
        files_sha256={str(key): str(value) for key, value in raw["files_sha256"].items()},
    )
    if (
        manifest.schema_version != 1
        or manifest.dataset != BBQ_NAME
        or manifest.revision != BBQ_REVISION
        or manifest.total_case_count != BBQ_TOTAL_CASES
        or manifest.scorable_case_count != BBQ_SCORABLE_CASES
        or manifest.files_sha256 != _expected_files()
    ):
        raise ValueError("BBQ manifest does not match the pinned dataset contract")
    for filename, expected_sha256 in manifest.files_sha256.items():
        file_path = dataset_directory / filename
        if not file_path.is_file() or _sha256_file(file_path) != expected_sha256:
            raise ValueError(f"BBQ file failed integrity verification: {filename}")
    return manifest


def fetch_bbq(cache_directory: Path) -> Path:
    """Download the pinned BBQ files atomically and verify every SHA-256 digest."""

    cache_directory.mkdir(parents=True, exist_ok=True)
    dataset_directory = cache_directory / "bbq"
    if dataset_directory.is_dir() and (dataset_directory / BBQ_MANIFEST).is_file():
        _load_manifest(dataset_directory)
        return dataset_directory
    if dataset_directory.exists():
        raise ValueError(f"incomplete BBQ directory exists at {dataset_directory}; move it aside")
    with tempfile.TemporaryDirectory(prefix="bbq-", dir=cache_directory) as temporary:
        staging = Path(temporary)
        for filename, expected_sha256 in _expected_files().items():
            partial = staging / f".{filename}.part"
            _download(_source_url(filename), partial)
            actual_sha256 = _sha256_file(partial)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"BBQ SHA-256 mismatch for {filename}: expected {expected_sha256}, "
                    f"got {actual_sha256}"
                )
            partial.replace(staging / filename)
        _write_manifest(staging)
        shutil.move(str(staging), dataset_directory)
    _load_manifest(dataset_directory)
    return dataset_directory


def _metadata_key(category: str, question_index: str, example_id: int) -> tuple[str, str, int]:
    return category, question_index, example_id


def _target_loc(value: object) -> int | None:
    normalized = str(value).strip()
    if not normalized or normalized.casefold() == "na":
        return None
    try:
        result = int(normalized)
    except ValueError as error:
        raise ValueError(f"invalid BBQ target_loc {value!r}") from error
    if result not in {0, 1, 2}:
        raise ValueError(f"BBQ target_loc must be 0, 1, or 2, got {result}")
    return result


def _load_scoring_metadata(
    path: Path,
) -> dict[tuple[str, str, int], tuple[int | None, str | None]]:
    metadata: dict[tuple[str, str, int], tuple[int | None, str | None]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                category = str(row["category"]).strip()
                question_index = str(row["question_index"]).strip()
                example_id = int(str(row["example_id"]).strip())
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid BBQ metadata key at line {line_number}") from error
            label_type_raw = str(row.get("label_type", "")).strip()
            value = (_target_loc(row.get("target_loc")), label_type_raw or None)
            key = _metadata_key(category, question_index, example_id)
            existing = metadata.get(key)
            if existing is not None and existing != value:
                raise ValueError(f"conflicting duplicate BBQ metadata for {key}")
            metadata[key] = value
    return metadata


def _required_text(row: Mapping[str, Any], key: str, *, label: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _required_index(row: Mapping[str, Any], key: str, *, label: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value not in {0, 1, 2}:
        raise ValueError(f"{label}.{key} must be 0, 1, or 2")
    return value


def _unknown_index(row: Mapping[str, Any], *, label: str) -> int:
    raw = row.get("answer_info")
    if not isinstance(raw, dict):
        raise ValueError(f"{label}.answer_info must be an object")
    unknown: list[int] = []
    for index in range(3):
        value = raw.get(f"ans{index}")
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{label}.answer_info.ans{index} must contain two strings")
        if not all(isinstance(part, str) for part in value):
            raise ValueError(f"{label}.answer_info.ans{index} must contain two strings")
        # Some official intersectional rows leave a non-unknown answer_info pair blank even
        # though ans0..ans2 are populated. Preserve that upstream defect; only the semantic
        # unknown tag is required for deterministic scoring.
        semantic_tag = " ".join(cast(str, value[1]).casefold().split())
        if semantic_tag in _UNKNOWN_TAGS:
            unknown.append(index)
    if len(unknown) != 1:
        raise ValueError(f"{label} must have exactly one unknown answer tag")
    return unknown[0]


def _parse_case(
    row: Mapping[str, Any],
    *,
    expected_category: str,
    metadata: Mapping[tuple[str, str, int], tuple[int | None, str | None]],
    line_number: int,
) -> BBQCase:
    label_name = f"{expected_category}.jsonl:{line_number}"
    example_id_raw = row.get("example_id")
    if not isinstance(example_id_raw, int) or isinstance(example_id_raw, bool):
        raise ValueError(f"{label_name}.example_id must be an integer")
    question_index = str(row.get("question_index", "")).strip()
    if not question_index:
        raise ValueError(f"{label_name}.question_index must be non-empty")
    category = _required_text(row, "category", label=label_name)
    if category != expected_category:
        raise ValueError(f"{label_name} category does not match its source file")
    context_condition = _required_text(row, "context_condition", label=label_name)
    question_polarity = _required_text(row, "question_polarity", label=label_name)
    if context_condition not in {"ambig", "disambig"}:
        raise ValueError(f"{label_name}.context_condition is unsupported")
    if question_polarity not in {"neg", "nonneg"}:
        raise ValueError(f"{label_name}.question_polarity is unsupported")
    answers = cast(
        tuple[str, str, str],
        tuple(_required_text(row, f"ans{index}", label=label_name) for index in range(3)),
    )
    label = _required_index(row, "label", label=label_name)
    unknown_index = _unknown_index(row, label=label_name)
    if context_condition == "ambig" and label != unknown_index:
        raise ValueError(f"{label_name} ambiguous label must identify the unknown answer")
    metadata_key = _metadata_key(category, question_index, example_id_raw)
    if metadata_key not in metadata:
        raise ValueError(f"{label_name} has no official scoring metadata")
    target_index, label_type = metadata[metadata_key]
    if target_index == unknown_index:
        raise ValueError(f"{label_name} target_loc cannot identify the unknown answer")
    return BBQCase(
        case_id=f"{category}:{example_id_raw}",
        example_id=example_id_raw,
        question_index=question_index,
        category=category,
        context_condition=context_condition,
        question_polarity=question_polarity,
        context=_required_text(row, "context", label=label_name),
        question=_required_text(row, "question", label=label_name),
        answers=answers,
        label=label,
        unknown_index=unknown_index,
        target_index=target_index,
        label_type=label_type,
    )


def load_bbq(dataset_directory: Path) -> tuple[BBQCase, ...]:
    """Load and strictly validate all 58,492 pinned BBQ examples."""

    _load_manifest(dataset_directory)
    metadata = _load_scoring_metadata(dataset_directory / BBQ_METADATA_FILE)
    cases: list[BBQCase] = []
    for filename, (expected_count, _) in BBQ_DATA_FILES.items():
        category = filename.removesuffix(".jsonl")
        file_cases: list[BBQCase] = []
        with (dataset_directory / filename).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"{filename}:{line_number} must contain a JSON object")
                file_cases.append(
                    _parse_case(
                        cast(Mapping[str, Any], raw),
                        expected_category=category,
                        metadata=metadata,
                        line_number=line_number,
                    )
                )
        if len(file_cases) != expected_count:
            raise ValueError(
                f"{filename} contains {len(file_cases):,} rows; expected {expected_count:,}"
            )
        cases.extend(file_cases)
    if len(cases) != BBQ_TOTAL_CASES or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("BBQ total count or case-ID uniqueness check failed")
    scorable_count = sum(case.scorable for case in cases)
    if scorable_count != BBQ_SCORABLE_CASES:
        raise ValueError(
            f"BBQ has {scorable_count:,} scorable cases; expected {BBQ_SCORABLE_CASES:,}"
        )
    return tuple(cases)


def select_bbq_subset(
    cases: Sequence[BBQCase],
    *,
    sample_per_stratum: int = 2,
    seed: str = BBQ_DEFAULT_SEED,
) -> BBQSelection:
    """Hash-rank a balanced subset while maximizing distinct source templates."""

    if sample_per_stratum <= 0:
        raise ValueError("sample_per_stratum must be positive")
    if not seed.strip():
        raise ValueError("selection seed must be non-empty")
    groups: defaultdict[str, list[BBQCase]] = defaultdict(list)
    for case in cases:
        if case.scorable:
            groups[case.stratum].append(case)
    if not groups:
        raise ValueError("BBQ selection requires at least one scorable stratum")
    selected: list[BBQCase] = []
    for stratum in sorted(groups):
        ranked = sorted(
            groups[stratum],
            key=lambda case: (_stable_sha256([seed, case.case_id]), case.case_id),
        )
        distinct: list[BBQCase] = []
        repeated: list[BBQCase] = []
        seen_templates: set[str] = set()
        for case in ranked:
            if case.template_id in seen_templates:
                repeated.append(case)
            else:
                distinct.append(case)
                seen_templates.add(case.template_id)
        candidates = distinct + repeated
        if len(candidates) < sample_per_stratum:
            raise ValueError(f"stratum {stratum!r} has only {len(candidates)} scorable examples")
        selected.extend(candidates[:sample_per_stratum])
    execution_order = tuple(
        sorted(
            selected,
            key=lambda case: (_stable_sha256([seed, "order", case.case_id]), case.case_id),
        )
    )
    selection_rows = [
        {
            "case_id": case.case_id,
            "stratum": case.stratum,
            "template_id": case.template_id,
        }
        for case in execution_order
    ]
    return BBQSelection(
        cases=execution_order,
        selection_sha256=_stable_sha256(selection_rows),
        stratum_counts=dict(sorted(Counter(case.stratum for case in execution_order).items())),
        template_count=len({case.template_id for case in execution_order}),
    )


def build_bbq_subset_manifest(
    cases: Sequence[BBQCase],
    *,
    sample_per_stratum: int = 2,
    seed: str = BBQ_DEFAULT_SEED,
    frozen_at: str,
) -> dict[str, Any]:
    """Build the auditable ID-only contract for a budget-bounded BBQ-derived diagnostic."""

    if not frozen_at.strip():
        raise ValueError("frozen_at must be non-empty")
    selection = select_bbq_subset(cases, sample_per_stratum=sample_per_stratum, seed=seed)
    return {
        "schema_version": 1,
        "status": "frozen_before_provider_calls",
        "dataset": BBQ_NAME,
        "dataset_revision": BBQ_REVISION,
        "dataset_total_case_count": BBQ_TOTAL_CASES,
        "dataset_scorable_case_count": BBQ_SCORABLE_CASES,
        "unscorable_case_count": BBQ_TOTAL_CASES - BBQ_SCORABLE_CASES,
        "unscorable_policy": "exclude_rows_missing_official_target_loc",
        "diagnostic_scope": (
            "BBQ-derived subset; not a full BBQ score or retrieval-fairness measure"
        ),
        "selection_method": BBQ_SELECTION_METHOD,
        "selection_seed": seed,
        "sample_per_stratum": sample_per_stratum,
        "selected_case_count": len(selection.cases),
        "selected_template_count": selection.template_count,
        "selection_sha256": selection.selection_sha256,
        "stratum_counts": selection.stratum_counts,
        "frozen_at": frozen_at,
        "cases": [
            {
                "order": order,
                "case_id": case.case_id,
                "category": case.category,
                "score_category": case.score_category,
                "label_type": case.label_type,
                "example_id": case.example_id,
                "template_id": case.template_id,
                "context_condition": case.context_condition,
                "question_polarity": case.question_polarity,
                "stratum": case.stratum,
            }
            for order, case in enumerate(selection.cases, start=1)
        ],
    }


def write_bbq_subset_manifest(
    cases: Sequence[BBQCase],
    destination: Path,
    *,
    sample_per_stratum: int = 2,
    seed: str = BBQ_DEFAULT_SEED,
    frozen_at: str,
) -> dict[str, Any]:
    """Atomically write a frozen BBQ subset manifest."""

    payload = build_bbq_subset_manifest(
        cases,
        sample_per_stratum=sample_per_stratum,
        seed=seed,
        frozen_at=frozen_at,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        handle.write(encoded)
        temporary_path = Path(handle.name)
    temporary_path.replace(destination)
    return payload


def load_frozen_bbq_subset(
    dataset_directory: Path,
    subset_manifest_path: Path,
) -> tuple[tuple[BBQCase, ...], dict[str, Any]]:
    """Bind an ID-only subset manifest back to the exact pinned source examples."""

    cases = load_bbq(dataset_directory)
    by_id = {case.case_id: case for case in cases}
    raw_value = json.loads(subset_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict) or not isinstance(raw_value.get("cases"), list):
        raise ValueError("BBQ subset manifest has an invalid schema")
    raw = cast(dict[str, Any], raw_value)
    required = {
        "schema_version": 1,
        "status": "frozen_before_provider_calls",
        "dataset": BBQ_NAME,
        "dataset_revision": BBQ_REVISION,
        "selection_method": BBQ_SELECTION_METHOD,
    }
    if any(raw.get(key) != value for key, value in required.items()):
        raise ValueError("BBQ subset manifest does not match the frozen contract")
    selected: list[BBQCase] = []
    for expected_order, record in enumerate(cast(list[Any], raw["cases"]), start=1):
        if not isinstance(record, dict) or record.get("order") != expected_order:
            raise ValueError("BBQ subset case order is malformed")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or case_id not in by_id:
            raise ValueError(f"BBQ subset contains unknown case ID {case_id!r}")
        case = by_id[case_id]
        expected_record = {
            "order": expected_order,
            "case_id": case.case_id,
            "category": case.category,
            "score_category": case.score_category,
            "label_type": case.label_type,
            "example_id": case.example_id,
            "template_id": case.template_id,
            "context_condition": case.context_condition,
            "question_polarity": case.question_polarity,
            "stratum": case.stratum,
        }
        if record != expected_record:
            raise ValueError(f"BBQ subset metadata drifted for {case.case_id}")
        selected.append(case)
    if len({case.case_id for case in selected}) != len(selected):
        raise ValueError("BBQ subset contains duplicate cases")
    selection_rows = [
        {
            "case_id": case.case_id,
            "stratum": case.stratum,
            "template_id": case.template_id,
        }
        for case in selected
    ]
    actual_sha256 = _stable_sha256(selection_rows)
    if raw.get("selection_sha256") != actual_sha256:
        raise ValueError("BBQ subset selection SHA-256 does not match its case records")
    if raw.get("selected_case_count") != len(selected):
        raise ValueError("BBQ subset selected_case_count is inconsistent")
    expected_strata = dict(sorted(Counter(case.stratum for case in selected).items()))
    if raw.get("stratum_counts") != expected_strata:
        raise ValueError("BBQ subset stratum counts are inconsistent")
    if raw.get("selected_template_count") != len({case.template_id for case in selected}):
        raise ValueError("BBQ subset template count is inconsistent")
    if not selected or not math.isfinite(float(raw.get("sample_per_stratum", 0))):
        raise ValueError("BBQ subset selection configuration is invalid")
    return tuple(selected), raw

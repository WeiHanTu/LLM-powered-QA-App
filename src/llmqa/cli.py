"""Offline command-line access to fairness metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llmqa.domain import Chunk, SearchResult
from llmqa.fairness import (
    CounterfactualOutcome,
    audit_counterfactual_outcomes,
    audit_exposure,
    fair_greedy_rerank,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _load_results(path: Path) -> list[SearchResult]:
    rows = _read_jsonl(path)
    return [
        SearchResult(
            chunk=Chunk(
                id=str(row["id"]),
                text=str(row.get("text", "")),
                source=str(row.get("source", "unknown")),
                page=row.get("page"),
                metadata=dict(row.get("metadata", {})),
            ),
            score=float(row["score"]),
            rank=int(row.get("rank", index)),
            original_rank=int(row.get("original_rank", row.get("rank", index))),
        )
        for index, row in enumerate(rows, start=1)
    ]


def _target(raw: str) -> dict[str, float]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--target must be a JSON object")
    return {str(group): float(value) for group, value in parsed.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmqa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    exposure = subparsers.add_parser("audit-exposure", help="compute NDKL for ranked JSONL")
    exposure.add_argument("input", type=Path)
    exposure.add_argument("--target", required=True, help='JSON, for example {"a":0.5,"b":0.5}')
    exposure.add_argument("--group-key", default="fairness_group")

    rerank = subparsers.add_parser("fair-rerank", help="rerank a candidate JSONL slate")
    rerank.add_argument("input", type=Path)
    rerank.add_argument("--target", required=True)
    rerank.add_argument("-k", type=int, required=True)
    rerank.add_argument("--group-key", default="fairness_group")

    counterfactual = subparsers.add_parser(
        "audit-counterfactual", help="compute flip rate and score difference from paired JSONL"
    )
    counterfactual.add_argument("input", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "audit-exposure":
        exposure_audit = audit_exposure(
            _load_results(args.input), _target(args.target), group_key=args.group_key
        )
        print(json.dumps(asdict(exposure_audit), indent=2))
        return 0

    if args.command == "fair-rerank":
        reranked = fair_greedy_rerank(
            _load_results(args.input),
            _target(args.target),
            k=args.k,
            group_key=args.group_key,
        )
        print(
            json.dumps(
                [
                    {
                        "id": result.chunk.id,
                        "rank": result.rank,
                        "original_rank": result.original_rank,
                        "score": result.score,
                        "metadata": result.chunk.metadata,
                    }
                    for result in reranked
                ],
                indent=2,
            )
        )
        return 0

    rows = _read_jsonl(args.input)
    counterfactual_audit = audit_counterfactual_outcomes(
        [
            CounterfactualOutcome(
                case_id=str(row["case_id"]),
                label_a=str(row["label_a"]),
                label_b=str(row["label_b"]),
                score_a=float(row["score_a"]) if row.get("score_a") is not None else None,
                score_b=float(row["score_b"]) if row.get("score_b") is not None else None,
            )
            for row in rows
        ]
    )
    print(json.dumps(asdict(counterfactual_audit), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from pathlib import Path

from llmqa.cli import main


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def ranked_rows() -> list[dict[str, object]]:
    return [
        {
            "id": identifier,
            "score": 1 - rank / 10,
            "rank": rank,
            "metadata": {"fairness_group": group},
            "component_scores": {"dense": 1 - rank / 10},
            "component_ranks": {"dense": rank},
        }
        for rank, (identifier, group) in enumerate(
            [("a1", "a"), ("a2", "a"), ("b1", "b"), ("b2", "b")], start=1
        )
    ]


def test_cli_exposure_and_reranking(tmp_path: Path, capsys: object) -> None:
    input_path = tmp_path / "ranked.jsonl"
    write_jsonl(input_path, ranked_rows())
    target = '{"a":0.5,"b":0.5}'

    assert main(["audit-exposure", str(input_path), "--target", target]) == 0
    exposure = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert exposure["ndkl"] > 0

    assert main(["fair-rerank", str(input_path), "--target", target, "-k", "4"]) == 0
    reranked = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert [row["id"] for row in reranked] == ["a1", "b1", "a2", "b2"]
    assert reranked[1]["component_ranks"] == {"dense": 3}


def test_cli_counterfactual_audit(tmp_path: Path, capsys: object) -> None:
    input_path = tmp_path / "pairs.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "case_id": "one",
                "label_a": "yes",
                "label_b": "no",
                "score_a": 0.9,
                "score_b": 0.5,
            }
        ],
    )

    assert main(["audit-counterfactual", str(input_path)]) == 0
    audit = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert audit["counterfactual_flip_rate"] == 1.0
    assert audit["mean_absolute_score_difference"] == 0.4


def test_cli_retrieval_evaluation(tmp_path: Path, capsys: object) -> None:
    judgments_path = tmp_path / "judgments.jsonl"
    run_path = tmp_path / "run.jsonl"
    write_jsonl(
        judgments_path,
        [{"query_id": "q1", "query": "What?", "relevance": {"a": 2, "b": 1}}],
    )
    write_jsonl(run_path, [{"query_id": "q1", "retrieved_ids": ["x", "a"]}])

    assert main(["evaluate-retrieval", str(judgments_path), str(run_path), "-k", "2"]) == 0
    evaluation = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert evaluation["mean_recall_at_k"] == 0.5
    assert evaluation["mean_reciprocal_rank"] == 0.5

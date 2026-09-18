import json
from pathlib import Path

import pytest

from studypilot.evaluation.bm25_baseline import (
    calculate_recall_at_k,
    load_dataset,
    run_evaluation,
    validate_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "evaluation" / "probability_bm25_v1.json"


def test_metric_calculation() -> None:
    assert calculate_recall_at_k([True, False, True, True]) == 0.75
    assert calculate_recall_at_k([]) == 0.0


def test_checked_in_dataset_excludes_answer_and_solution_files() -> None:
    dataset = load_dataset(DATASET_PATH)
    assert len(dataset["queries"]) == 30
    assert sum(query["kind"] == "LEXICAL" for query in dataset["queries"]) == 15
    assert sum(query["kind"] == "SEMANTIC" for query in dataset["queries"]) == 15
    assert {query["kind"] for query in dataset["queries"]} == {"LEXICAL", "SEMANTIC"}
    assert all(item["file"].startswith("ch") for item in dataset["corpus"])


def test_dataset_rejects_answer_leakage() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    dataset["corpus"].append({"file": "真题解析.md", "sha256": "0" * 64})
    with pytest.raises(ValueError, match="leaked"):
        validate_dataset(dataset)


def test_dataset_rejects_gold_outside_corpus() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    dataset["queries"][0]["gold"][0]["file"] = "not-in-corpus.md"
    with pytest.raises(ValueError, match="outside corpus"):
        validate_dataset(dataset)


def test_evaluation_runner_resolves_gold_and_computes_result(tmp_path) -> None:
    source_root = tmp_path / "notes"
    source_root.mkdir()
    note = source_root / "ch1-demo.md"
    note.write_text("# 第一章\n\n## 条件概率\n\n贝叶斯公式用于由果推因。\n", encoding="utf-8")
    import hashlib

    dataset = {
        "dataset_id": "fixture",
        "course_id": "fixture-course",
        "corpus": [
            {"file": note.name, "sha256": hashlib.sha256(note.read_bytes()).hexdigest()}
        ],
        "queries": [
            {
                "id": "q1",
                "kind": "LEXICAL",
                "query": "贝叶斯 由果推因",
                "gold": [{"file": note.name, "section": "条件概率"}],
            }
        ],
    }
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")

    summary = run_evaluation(
        dataset_path=dataset_path,
        source_root=source_root,
        work_root=tmp_path / "work",
    )

    assert summary.recall_at_5 == 1.0
    assert summary.hits == summary.total == 1
    assert summary.query_results[0]["resolved_gold_block_ids"]

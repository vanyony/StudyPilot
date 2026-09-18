from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from studypilot.application.parsing import ParseService
from studypilot.application.retrieval import BM25Retriever, tokenize
from studypilot.application.source_import import SourceImportService
from studypilot.domain.knowledge import ParserKind
from studypilot.domain.models import Course
from studypilot.domain.sources import DocumentKind, TrustLevel
from studypilot.infrastructure.sqlite_repository import SQLiteRepository


FORBIDDEN_CORPUS_MARKERS = ("答案", "解析", "真题", "试题与答案")


@dataclass(frozen=True)
class EvaluationSummary:
    dataset_id: str
    retrieval_method: str
    config: dict[str, Any]
    recall_at_5: float
    hits: int
    total: int
    query_results: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "retrieval_method": self.retrieval_method,
            "config": self.config,
            "recall_at_5": self.recall_at_5,
            "hits": self.hits,
            "total": self.total,
            "query_results": list(self.query_results),
        }


def load_dataset(dataset_path: Path) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: dict[str, Any]) -> None:
    corpus_files = [item["file"] for item in dataset["corpus"]]
    if len(corpus_files) != len(set(corpus_files)):
        raise ValueError("corpus contains duplicate files")
    forbidden = [
        filename
        for filename in corpus_files
        if any(marker in filename for marker in FORBIDDEN_CORPUS_MARKERS)
    ]
    if forbidden:
        raise ValueError(f"answer or solution material leaked into corpus: {forbidden}")
    corpus_set = set(corpus_files)
    query_ids = [item["id"] for item in dataset["queries"]]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query ids must be unique")
    for query in dataset["queries"]:
        if not query["gold"]:
            raise ValueError(f"query {query['id']} has no gold locator")
        for locator in query["gold"]:
            if locator["file"] not in corpus_set:
                raise ValueError(f"gold file is outside corpus: {locator['file']}")


def calculate_recall_at_k(results: list[bool]) -> float:
    return sum(results) / len(results) if results else 0.0


def run_evaluation(
    *, dataset_path: Path, source_root: Path, work_root: Path, title_weight: int = 0
) -> EvaluationSummary:
    dataset, repository = prepare_corpus(
        dataset_path=dataset_path, source_root=source_root, work_root=work_root
    )
    retriever = BM25Retriever(repository, title_weight=title_weight)
    return evaluate_retriever(
        dataset,
        repository,
        retriever,
        retrieval_method="bm25",
        config={"k1": 1.5, "b": 0.75, "title_weight": title_weight},
    )


def prepare_corpus(*, dataset_path: Path, source_root: Path, work_root: Path):
    dataset = load_dataset(dataset_path)
    work_root.mkdir(parents=True, exist_ok=True)
    repository = SQLiteRepository(work_root / "evaluation.db")
    repository.initialize()
    course_id = dataset["course_id"]
    repository.save_course(Course(id=course_id, name="概率论 BM25 离线评测"))
    importer = SourceImportService(repository, work_root / "storage")
    parse_service = ParseService(repository)

    for source in dataset["corpus"]:
        path = source_root / source["file"]
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != source["sha256"]:
            raise ValueError(
                f"source hash changed for {source['file']}: expected {source['sha256']}, got {actual_hash}"
            )
        with path.open("rb") as stream:
            record = importer.import_file(
                course_id=course_id,
                stream=stream,
                display_name=source["file"],
                document_kind=DocumentKind.COURSE_MATERIAL,
                trust_level=TrustLevel.HIGH,
                origin="local-probability-note",
                metadata={"evaluation_dataset": dataset["dataset_id"]},
            )
        parse_service.parse_source(record.asset.id, ParserKind.MARKDOWN)

    return dataset, repository


def evaluate_retriever(
    dataset: dict[str, Any],
    repository: SQLiteRepository,
    retriever,
    *,
    retrieval_method: str,
    config: dict[str, Any],
) -> EvaluationSummary:
    course_id = dataset["course_id"]
    all_blocks = repository.list_blocks(course_id)
    citations = {block.id: repository.citation_for_block(block.id) for block in all_blocks}
    query_results: list[dict[str, Any]] = []
    hit_flags: list[bool] = []
    for query in dataset["queries"]:
        hits = retriever.search(course_id, query["query"], limit=5)
        gold_pairs = {(item["file"], item["section"]) for item in query["gold"]}
        matched = next(
            (
                rank
                for rank, hit in enumerate(hits, start=1)
                if (hit.citation.display_name, hit.citation.section) in gold_pairs
            ),
            None,
        )
        gold_blocks = [
            block
            for block in all_blocks
            if (citations[block.id].display_name, block.section) in gold_pairs
        ]
        if not gold_blocks:
            failure_category = "INVALID_GOLD_LOCATOR"
        elif matched is not None:
            failure_category = None
        else:
            query_tokens = set(tokenize(query["query"]))
            gold_tokens = {token for block in gold_blocks for token in tokenize(block.text)}
            failure_category = (
                "SEMANTIC_VOCABULARY_GAP"
                if not query_tokens.intersection(gold_tokens)
                else "LEXICAL_RANKING_MISS"
            )
        hit_flags.append(matched is not None)
        query_results.append(
            {
                "id": query["id"],
                "kind": query["kind"],
                "query": query["query"],
                "hit": matched is not None,
                "rank": matched,
                "failure_category": failure_category,
                "resolved_gold_block_ids": [block.id for block in gold_blocks],
                "top5": [
                    {
                        "rank": rank,
                        "score": hit.score,
                        "block_id": hit.block.id,
                        "source": hit.citation.display_name,
                        "section": hit.citation.section,
                    }
                    for rank, hit in enumerate(hits, start=1)
                ],
            }
        )

    return EvaluationSummary(
        dataset_id=dataset["dataset_id"],
        retrieval_method=retrieval_method,
        config=config,
        recall_at_5=calculate_recall_at_k(hit_flags),
        hits=sum(hit_flags),
        total=len(hit_flags),
        query_results=tuple(query_results),
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    argument_parser = argparse.ArgumentParser(
        description="Run the auditable probability BM25 baseline"
    )
    argument_parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "evaluation" / "probability_bm25_v1.json",
    )
    argument_parser.add_argument(
        "--source-root", type=Path, default=Path(r"D:\笔记\概率论")
    )
    argument_parser.add_argument("--output", type=Path)
    argument_parser.add_argument("--title-weight", type=int, default=0)
    args = argument_parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="studypilot-eval-") as directory:
        summary = run_evaluation(
            dataset_path=args.dataset,
            source_root=args.source_root,
            work_root=Path(directory),
            title_weight=args.title_weight,
        )
    report = json.dumps(summary.as_dict(), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

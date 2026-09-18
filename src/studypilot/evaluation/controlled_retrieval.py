from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from studypilot.application.embeddings import (
    DenseRetriever,
    JsonEmbeddingCache,
    RRFRetriever,
    SentenceTransformerEmbeddingProvider,
)
from studypilot.application.retrieval import BM25Retriever
from studypilot.evaluation.bm25_baseline import evaluate_retriever, prepare_corpus


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Compare frozen sparse, dense and RRF retrieval")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "evaluation" / "probability_bm25_v1.json",
    )
    parser.add_argument("--source-root", type=Path, default=Path(r"D:\笔记\概率论"))
    parser.add_argument(
        "--model-cache", type=Path, default=Path(r"D:\ModelCache\huggingface")
    )
    parser.add_argument(
        "--embedding-cache", type=Path, default=Path(r"D:\ModelCache\studypilot-embeddings")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "evaluation" / "probability_retrieval_v2_result.json",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="studypilot-controlled-eval-") as directory:
        dataset, repository = prepare_corpus(
            dataset_path=args.dataset,
            source_root=args.source_root,
            work_root=Path(directory),
        )
        sparse = BM25Retriever(repository, title_weight=2)
        sparse_summary = evaluate_retriever(
            dataset,
            repository,
            sparse,
            retrieval_method="title-aware-bm25",
            config={"k1": 1.5, "b": 0.75, "title_weight": 2},
        )
        provider = SentenceTransformerEmbeddingProvider(args.model_cache)
        dense = DenseRetriever(repository, provider, JsonEmbeddingCache(args.embedding_cache))
        dense_summary = evaluate_retriever(
            dataset,
            repository,
            dense,
            retrieval_method="dense",
            config={
                "model": provider.model_id,
                "revision": provider.revision,
                "dimension": provider.dimension,
                "normalized": True,
                "query_instruction": provider.query_instruction,
            },
        )
        hybrid = RRFRetriever(sparse, dense, rrf_k=60, candidate_limit=50)
        hybrid_summary = evaluate_retriever(
            dataset,
            repository,
            hybrid,
            retrieval_method="rrf-hybrid",
            config={"rrf_k": 60, "candidate_limit": 50, "sparse_title_weight": 2},
        )
    payload = {
        "dataset_id": dataset["dataset_id"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "results": [
            sparse_summary.as_dict(),
            dense_summary.as_dict(),
            hybrid_summary.as_dict(),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


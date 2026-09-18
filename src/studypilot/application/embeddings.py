from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Protocol

from studypilot.domain.knowledge import SearchHit
from studypilot.infrastructure.sqlite_repository import SQLiteRepository


class EmbeddingUnavailableError(RuntimeError):
    pass


class EmbeddingProvider(Protocol):
    model_id: str
    revision: str
    dimension: int

    def encode_passages(self, texts: list[str]) -> list[list[float]]: ...

    def encode_queries(self, texts: list[str]) -> list[list[float]]: ...


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("cannot normalize a zero embedding")
    return [value / norm for value in vector]


class SentenceTransformerEmbeddingProvider:
    model_id = "BAAI/bge-small-zh-v1.5"
    revision = "4e17e244a0fb63bfb78fca8fcf95079fcc664f5c"
    query_instruction = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, cache_folder: str | Path) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingUnavailableError(
                "install StudyPilot with the 'embeddings' extra to use dense retrieval"
            ) from error
        self._model = SentenceTransformer(
            self.model_id,
            revision=self.revision,
            cache_folder=str(Path(cache_folder).resolve()),
            device="cpu",
        )
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise EmbeddingUnavailableError("embedding model did not report its dimension")
        self.dimension = int(dimension)

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        values = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [row.tolist() for row in values]

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        instructed = [self.query_instruction + text for text in texts]
        values = self._model.encode(
            instructed, normalize_embeddings=True, show_progress_bar=False
        )
        return [row.tolist() for row in values]


class JsonEmbeddingCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, provider: EmbeddingProvider, content_key: str) -> Path:
        identity = f"{provider.model_id}:{provider.revision}:{content_key}"
        key = hashlib.sha256(identity.encode()).hexdigest()
        return self.root / key[:2] / f"{key}.json"

    def get(self, provider: EmbeddingProvider, content_key: str) -> list[float] | None:
        path = self._path(provider, content_key)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        vector = [float(value) for value in payload["vector"]]
        if len(vector) != provider.dimension:
            raise ValueError("cached embedding dimension does not match provider")
        return vector

    def put(self, provider: EmbeddingProvider, content_key: str, vector: list[float]) -> None:
        if len(vector) != provider.dimension:
            raise ValueError("embedding dimension does not match provider")
        path = self._path(provider, content_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"dimension": provider.dimension, "vector": vector}),
            encoding="utf-8",
        )
        temporary.replace(path)


class DenseRetriever:
    def __init__(
        self,
        repository: SQLiteRepository,
        provider: EmbeddingProvider,
        cache: JsonEmbeddingCache,
    ) -> None:
        self.repository = repository
        self.provider = provider
        self.cache = cache

    def search(self, course_id: str, query: str, limit: int = 5) -> list[SearchHit]:
        blocks = self.repository.list_blocks(course_id)
        if not blocks or limit <= 0:
            return []
        content_keys = [
            hashlib.sha256(
                f"{block.section or ''}\n{block.content_hash}".encode("utf-8")
            ).hexdigest()
            for block in blocks
        ]
        vectors: list[list[float] | None] = [
            self.cache.get(self.provider, content_key) for content_key in content_keys
        ]
        missing_indexes = [index for index, vector in enumerate(vectors) if vector is None]
        if missing_indexes:
            texts = [
                f"{blocks[index].section or ''}\n{blocks[index].text}" for index in missing_indexes
            ]
            encoded = self.provider.encode_passages(texts)
            for index, vector in zip(missing_indexes, encoded, strict=True):
                normalized = normalize(vector)
                vectors[index] = normalized
                self.cache.put(self.provider, content_keys[index], normalized)
        query_vector = normalize(self.provider.encode_queries([query])[0])
        hits = []
        for block, vector in zip(blocks, vectors, strict=True):
            assert vector is not None
            score = sum(left * right for left, right in zip(query_vector, vector, strict=True))
            hits.append(
                SearchHit(
                    block=block,
                    score=max(0.0, round(score, 6)),
                    citation=self.repository.citation_for_block(block.id),
                )
            )
        return sorted(hits, key=lambda hit: (-hit.score, hit.block.id))[:limit]


class RRFRetriever:
    def __init__(self, sparse, dense, rrf_k: int = 60, candidate_limit: int = 50) -> None:
        self.sparse = sparse
        self.dense = dense
        self.rrf_k = rrf_k
        self.candidate_limit = candidate_limit

    def search(self, course_id: str, query: str, limit: int = 5) -> list[SearchHit]:
        sparse_hits = self.sparse.search(course_id, query, self.candidate_limit)
        dense_hits = self.dense.search(course_id, query, self.candidate_limit)
        by_id = {hit.block.id: hit for hit in sparse_hits + dense_hits}
        scores: dict[str, float] = {}
        for ranking in (sparse_hits, dense_hits):
            for rank, hit in enumerate(ranking, start=1):
                scores[hit.block.id] = scores.get(hit.block.id, 0.0) + 1 / (
                    self.rrf_k + rank
                )
        fused = [
            by_id[block_id].model_copy(update={"score": round(score, 6)})
            for block_id, score in scores.items()
        ]
        return sorted(fused, key=lambda hit: (-hit.score, hit.block.id))[:limit]

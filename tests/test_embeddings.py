from __future__ import annotations

import sys

import pytest

from studypilot.application.embeddings import (
    EmbeddingUnavailableError,
    JsonEmbeddingCache,
    RRFRetriever,
    SentenceTransformerEmbeddingProvider,
    normalize,
)
from studypilot.domain.knowledge import Citation, DocumentBlock, ParserKind, SearchHit


class FakeProvider:
    model_id = "fake"
    revision = "v1"
    dimension = 2


def test_normalize_has_unit_length_and_rejects_zero() -> None:
    assert normalize([3.0, 4.0]) == [0.6, 0.8]
    with pytest.raises(ValueError, match="zero"):
        normalize([0.0, 0.0])


def test_json_embedding_cache_persists_and_checks_dimension(tmp_path) -> None:
    cache = JsonEmbeddingCache(tmp_path)
    provider = FakeProvider()
    cache.put(provider, "block", [0.6, 0.8])

    assert JsonEmbeddingCache(tmp_path).get(provider, "block") == [0.6, 0.8]
    with pytest.raises(ValueError, match="dimension"):
        cache.put(provider, "bad", [1.0])


def _hit(block_id: str, score: float) -> SearchHit:
    block = DocumentBlock(
        id=block_id,
        source_asset_id="source",
        course_id="course",
        parser_kind=ParserKind.TEXT,
        parser_version="1",
        block_index=0,
        text=block_id,
        content_hash="0" * 64,
    )
    citation = Citation(
        block_id=block_id,
        source_asset_id="source",
        course_id="course",
        display_name="note",
        page_number=None,
        section=None,
        block_index=0,
        quote=block_id,
    )
    return SearchHit(block=block, score=score, citation=citation)


class FixedRetriever:
    def __init__(self, ids):
        self.ids = ids

    def search(self, _course, _query, limit):
        return [_hit(item, 1.0) for item in self.ids[:limit]]


def test_rrf_uses_fixed_rank_fusion_and_is_deterministic() -> None:
    retriever = RRFRetriever(
        FixedRetriever(["a", "b", "c"]),
        FixedRetriever(["b", "c", "a"]),
        rrf_k=60,
        candidate_limit=3,
    )

    first = retriever.search("course", "query", 3)
    second = retriever.search("course", "query", 3)

    assert [hit.block.id for hit in first] == ["b", "a", "c"]
    assert first == second


def test_provider_fails_clearly_without_optional_dependency(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(EmbeddingUnavailableError, match="embeddings"):
        SentenceTransformerEmbeddingProvider(tmp_path)

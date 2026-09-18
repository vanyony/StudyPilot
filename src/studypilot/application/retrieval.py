from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from studypilot.domain.knowledge import (
    Citation,
    CitationVerification,
    DocumentBlock,
    EvidenceRelation,
    KnowledgeWindow,
    KnowledgeWindowItem,
    SearchHit,
)
from studypilot.infrastructure.sqlite_repository import SQLiteRepository


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """Tokenize Latin words and Chinese uni/bi-grams without hidden dictionaries."""
    tokens: list[str] = []
    normalized = unicodedata.normalize("NFKC", text).lower()
    for segment in _TOKEN_PATTERN.findall(normalized):
        if all("\u3400" <= char <= "\u9fff" for char in segment):
            tokens.extend(segment)
            tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
        else:
            tokens.append(segment)
    return tokens


class BM25Retriever:
    def __init__(
        self,
        repository: SQLiteRepository,
        k1: float = 1.5,
        b: float = 0.75,
        title_weight: int = 2,
    ) -> None:
        self.repository = repository
        self.k1 = k1
        self.b = b
        if title_weight < 0:
            raise ValueError("title_weight must be non-negative")
        self.title_weight = title_weight

    def search(self, course_id: str, query: str, limit: int = 5) -> list[SearchHit]:
        blocks = self.repository.list_blocks(course_id)
        query_tokens = list(dict.fromkeys(tokenize(query)))
        if not blocks or not query_tokens or limit <= 0:
            return []
        documents = [
            tokenize(block.text) + tokenize(block.section or "") * self.title_weight
            for block in blocks
        ]
        average_length = sum(map(len, documents)) / len(documents) or 1.0
        document_frequency = {
            term: sum(term in document for document in documents) for term in query_tokens
        }
        hits: list[SearchHit] = []
        for block, document in zip(blocks, documents, strict=True):
            frequencies = Counter(document)
            score = 0.0
            for term in query_tokens:
                frequency = frequencies[term]
                if frequency == 0:
                    continue
                df = document_frequency[term]
                inverse_frequency = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * len(document) / average_length
                )
                score += inverse_frequency * frequency * (self.k1 + 1) / denominator
            if score > 0:
                hits.append(
                    SearchHit(
                        block=block,
                        score=round(score, 6),
                        citation=self.repository.citation_for_block(block.id),
                    )
                )
        return sorted(hits, key=lambda hit: (-hit.score, hit.block.id))[:limit]


class KnowledgeWindowBuilder:
    def __init__(self, retriever: BM25Retriever) -> None:
        self.retriever = retriever

    def build(
        self,
        *,
        course_id: str,
        query: str,
        max_blocks: int,
        max_chars: int,
        relation_by_block_id: dict[str, EvidenceRelation] | None = None,
    ) -> KnowledgeWindow:
        if max_blocks <= 0 or max_chars <= 0:
            raise ValueError("max_blocks and max_chars must be positive")
        relations = relation_by_block_id or {}
        candidates = self.retriever.search(course_id, query, limit=max(max_blocks * 5, 20))
        items: list[KnowledgeWindowItem] = []
        used_chars = 0
        for hit in candidates:
            length = len(hit.block.text)
            if length > max_chars - used_chars:
                continue
            items.append(
                KnowledgeWindowItem(
                    relation=relations.get(hit.block.id, EvidenceRelation.UNKNOWN),
                    hit=hit,
                )
            )
            used_chars += length
            if len(items) == max_blocks:
                break
        return KnowledgeWindow(
            query=query,
            course_id=course_id,
            max_blocks=max_blocks,
            max_chars=max_chars,
            used_chars=used_chars,
            items=tuple(items),
        )


class CitationValidator:
    def __init__(self, repository: SQLiteRepository) -> None:
        self.repository = repository

    def verify(self, citation: Citation) -> CitationVerification:
        try:
            canonical = self.repository.citation_for_block(citation.block_id)
        except LookupError:
            return CitationVerification(valid=False, reason="block does not exist")
        if canonical.course_id != citation.course_id:
            return CitationVerification(valid=False, reason="course does not match block")
        if canonical.source_asset_id != citation.source_asset_id:
            return CitationVerification(valid=False, reason="source does not match block")
        if canonical.quote != citation.quote:
            return CitationVerification(valid=False, reason="quote does not match stored text")
        if canonical != citation:
            return CitationVerification(valid=False, reason="citation locator does not match")
        return CitationVerification(valid=True, reason="citation matches stored source and text")

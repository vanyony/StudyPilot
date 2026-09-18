from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ParserKind(StrEnum):
    MARKDOWN = "MARKDOWN"
    TEXT = "TEXT"


class EvidenceRelation(StrEnum):
    UNKNOWN = "UNKNOWN"
    DEFINITION = "DEFINITION"
    FORMULA = "FORMULA"
    EXAMPLE = "EXAMPLE"
    PREREQUISITE = "PREREQUISITE"
    SOLUTION = "SOLUTION"
    DISTRACTOR = "DISTRACTOR"


class DocumentBlock(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    source_asset_id: str
    course_id: str
    parser_kind: ParserKind
    parser_version: str
    page_number: int | None = Field(default=None, ge=1)
    section: str | None = None
    block_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    block_id: str
    source_asset_id: str
    course_id: str
    display_name: str
    page_number: int | None
    section: str | None
    block_index: int
    quote: str


class SearchHit(BaseModel):
    block: DocumentBlock
    score: float = Field(ge=0)
    citation: Citation


class KnowledgeWindowItem(BaseModel):
    relation: EvidenceRelation = EvidenceRelation.UNKNOWN
    hit: SearchHit


class KnowledgeWindow(BaseModel):
    query: str
    course_id: str
    max_blocks: int
    max_chars: int
    used_chars: int
    items: tuple[KnowledgeWindowItem, ...]


class CitationVerification(BaseModel):
    valid: bool
    reason: str


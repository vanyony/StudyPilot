from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, JsonValue


class DocumentKind(StrEnum):
    COURSE_MATERIAL = "COURSE_MATERIAL"
    PERSONAL_NOTE = "PERSONAL_NOTE"
    EXAM_PAPER = "EXAM_PAPER"
    ANSWER_KEY = "ANSWER_KEY"
    ASSIGNMENT = "ASSIGNMENT"
    IMAGE_NOTE = "IMAGE_NOTE"


class TrustLevel(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ParseStatus(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    FAILED = "FAILED"


class ContentBlob(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    storage_path: str
    byte_size: int = Field(ge=0)
    parse_status: ParseStatus
    created_at: datetime


class SourceAsset(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    course_id: str
    blob_id: str
    document_kind: DocumentKind
    origin: str | None = None
    display_name: str = Field(min_length=1, max_length=500)
    trust_level: TrustLevel
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: datetime


class SourceRecord(BaseModel):
    asset: SourceAsset
    blob: ContentBlob

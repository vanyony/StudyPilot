from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MasteryState(StrEnum):
    READY = "READY"
    FRAGILE = "FRAGILE"
    GAP = "GAP"
    UNSEEN = "UNSEEN"


class EvidenceLevel(StrEnum):
    TEACHER_SCOPE = "TEACHER_SCOPE"
    PAST_EXAM = "PAST_EXAM"
    COURSE_MATERIAL = "COURSE_MATERIAL"
    ASSIGNMENT = "ASSIGNMENT"
    PEER_NOTE = "PEER_NOTE"


class PlanTier(StrEnum):
    MUST = "MUST"
    STRIVE = "STRIVE"
    DEFER = "DEFER"


class Course(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)


class ExamGoal(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, max_length=100)
    course_id: str = Field(min_length=1, max_length=100)
    exam_at: datetime
    available_minutes: int = Field(gt=0, le=100_000)
    target_score: float | None = Field(default=None, ge=0, le=100)


class Topic(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, max_length=100)
    course_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    exam_points: float = Field(ge=0, le=100)
    learning_minutes: int = Field(gt=0, le=100_000)
    mastery: MasteryState = MasteryState.UNSEEN
    evidence_level: EvidenceLevel
    evidence_confidence: float = Field(ge=0, le=1)
    frequency: float = Field(default=0.5, ge=0, le=1)
    prerequisite_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def reject_self_dependency(self) -> Topic:
        if self.id in self.prerequisite_ids:
            raise ValueError("a topic cannot depend on itself")
        if len(set(self.prerequisite_ids)) != len(self.prerequisite_ids):
            raise ValueError("prerequisite_ids must not contain duplicates")
        return self


class PlanItem(BaseModel):
    topic_id: str
    topic_name: str
    tier: PlanTier
    order: int | None
    estimated_minutes: int
    utility_score: float
    reasons: tuple[str, ...]


class Plan(BaseModel):
    goal_id: str
    course_id: str
    budget_minutes: int
    planned_minutes: int
    items: tuple[PlanItem, ...]


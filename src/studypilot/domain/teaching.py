"""Teaching-session domain objects.

The workflow itself lives in :mod:`studypilot.application.teaching`.  These
models deliberately contain only serialisable domain data so the same state can
be returned by the API and stored in a LangGraph checkpoint.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from studypilot.domain.knowledge import KnowledgeWindow
from studypilot.domain.models import ExamGoal, MasteryState, Plan, PlanItem, Topic


class TeachingStatus(StrEnum):
    CREATED = "CREATED"
    PLANNING = "PLANNING"
    TEACHING = "TEACHING"
    QUESTIONING = "QUESTIONING"
    WAITING_ANSWER = "WAITING_ANSWER"
    EVALUATING = "EVALUATING"
    REPLANNING = "REPLANNING"
    COMPLETED = "COMPLETED"


class TeachingAction(StrEnum):
    PLAN = "PLAN"
    TEACH = "TEACH"
    QUESTION = "QUESTION"
    WAITING_ANSWER = "WAITING_ANSWER"
    EVALUATE = "EVALUATE"
    REPLAN = "REPLAN"
    COMPLETE = "COMPLETE"


class ScoringPoint(BaseModel):
    """One observable rubric requirement for a question.

    ``evidence`` is intentionally explicit.  An evaluator may only mark a
    point as satisfied when the answer contains the configured evidence (or a
    deterministic fixture marker used by the fake evaluator).
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2_000)
    evidence: tuple[str, ...] = ()
    # A provider may attach source block ids to an individual rubric point.
    # They are checked against the current Knowledge Window before the point
    # can influence a state transition.
    citations: tuple[str, ...] = ()
    required: bool = True
    weight: float = Field(default=1.0, gt=0, le=100)

    @field_validator("evidence", "citations", mode="before")
    @classmethod
    def _normalise_string_lists(cls, value: object) -> object:
        # JSON providers often emit a single string for a one-item list.  We
        # accept that harmless shape variation while keeping the domain type
        # a tuple for deterministic serialization.
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return value


class ScoringPointEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    scoring_point_id: str = Field(min_length=1, max_length=200)
    satisfied: bool
    evidence: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    reason: str = Field(min_length=1, max_length=2_000)

    @field_validator("evidence", "citations", mode="before")
    @classmethod
    def _normalise_string_lists(cls, value: object) -> object:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return value


class EvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0, le=1)
    mastery_state: MasteryState
    point_evaluations: tuple[ScoringPointEvaluation, ...] = ()
    citations: tuple[str, ...] = ()
    reason: str = Field(min_length=1, max_length=4_000)
    next_action: TeachingAction = TeachingAction.REPLAN

    @field_validator("citations", mode="before")
    @classmethod
    def _normalise_citations(cls, value: object) -> object:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return value


class TeachingContent(BaseModel):
    """Structured teaching output produced by a provider."""

    model_config = ConfigDict(frozen=True)

    explanation: str = Field(min_length=1, max_length=20_000)
    question: str = Field(min_length=1, max_length=10_000)
    scoring_points: tuple[ScoringPoint, ...] = Field(min_length=1)
    citations: tuple[str, ...] = ()

    @field_validator("citations", mode="before")
    @classmethod
    def _normalise_citations(cls, value: object) -> object:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return value


class TeachingSessionState(BaseModel):
    """Checkpointable state for one continuous teaching session."""

    model_config = ConfigDict(validate_assignment=True)

    session_id: str = Field(min_length=1, max_length=200)
    thread_id: str = Field(min_length=1, max_length=200)
    course_id: str = Field(min_length=1, max_length=100)
    goal_id: str = Field(min_length=1, max_length=100)
    current_goal: ExamGoal
    topics: tuple[Topic, ...] = ()
    plan: Plan | None = None
    current_plan_item: PlanItem | None = None
    knowledge_window: KnowledgeWindow | None = None
    teaching_text: str | None = None
    question_id: str | None = None
    question: str | None = None
    scoring_points: tuple[ScoringPoint, ...] = ()
    student_answer: str | None = None
    evaluation: EvaluationResult | None = None
    mastery_state: MasteryState = MasteryState.UNSEEN
    mastery_by_topic: dict[str, MasteryState] = Field(default_factory=dict)
    remaining_minutes: int = Field(ge=0)
    next_action: TeachingAction = TeachingAction.PLAN
    status: TeachingStatus = TeachingStatus.CREATED
    version: int = Field(default=0, ge=0)
    replan_reason: str | None = None
    attempt: int = Field(default=0, ge=0)
    last_answer_id: str | None = None
    processed_answers: dict[str, str] = Field(default_factory=dict)
    spent_minutes: int = Field(default=0, ge=0)
    teaching_citations: tuple[str, ...] = ()
    question_override: str | None = None
    scoring_points_override: tuple[ScoringPoint, ...] | None = None

    @property
    def current_mastery(self) -> MasteryState:
        """Convenient alias used by callers that describe the active topic."""

        return self.mastery_state


class TeachingSession(BaseModel):
    """Persisted session metadata plus its latest materialised state."""

    model_config = ConfigDict(validate_assignment=True)

    session_id: str = Field(min_length=1, max_length=200)
    thread_id: str = Field(min_length=1, max_length=200)
    course_id: str = Field(min_length=1, max_length=100)
    goal_id: str = Field(min_length=1, max_length=100)
    status: TeachingStatus = TeachingStatus.CREATED
    version: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    state: TeachingSessionState | None = None


class AnswerReceipt(BaseModel):
    """Durable result for one idempotent answer message."""

    model_config = ConfigDict(validate_assignment=True)

    session_id: str = Field(min_length=1, max_length=200)
    message_id: str = Field(min_length=1, max_length=200)
    answer_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_version: int = Field(ge=0)
    state: TeachingSessionState
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# Short aliases keep the public API pleasant and make the graph state easy to
# discover without coupling callers to an implementation-specific name.
TeachingState = TeachingSessionState
RubricPoint = ScoringPoint
RubricPointEvaluation = ScoringPointEvaluation

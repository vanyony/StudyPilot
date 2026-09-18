"""Session/application boundary around the phase six teaching graph.

The graph owns checkpointed domain state.  This service owns the durable
session directory and answer receipts needed by an API: a message id is
recorded with its result, and a per-session lock serialises operations inside
one application process.  SQLite plus the in-process lock is intentionally a
single-process boundary; a multi-process deployment needs a database-backed
lease or another distributed coordination mechanism.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from pathlib import Path
from threading import RLock
from typing import Sequence
from uuid import uuid4

from studypilot.application.teaching import (
    Evaluator,
    TeacherProvider,
    TeachingWorkflow,
    TeachingProviderError,
    TeachingWorkflowError,
)
from studypilot.application.retrieval import KnowledgeWindowBuilder
from studypilot.domain.models import ExamGoal, MasteryState, Topic
from studypilot.domain.teaching import (
    AnswerReceipt,
    ScoringPoint,
    TeachingAction,
    TeachingSession,
    TeachingSessionState,
    TeachingStatus,
)
from studypilot.infrastructure.sqlite_repository import NotFoundError, SQLiteRepository


class TeachingServiceError(ValueError):
    """Base error mapped to a client-visible session failure."""


class SessionAlreadyExists(TeachingServiceError):
    pass


class SessionStateConflict(TeachingServiceError):
    pass


class MessageIdConflict(TeachingServiceError):
    pass


class ExpectedVersionRequired(TeachingServiceError):
    pass


class SessionVersionConflict(TeachingServiceError):
    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"session version {actual} does not match expected version {expected}")


class SessionProviderError(TeachingServiceError):
    """A model/provider failure; callers must retry or inspect the cause."""


class TeachingSessionService:
    """Persist and serialise teaching sessions for one application process."""

    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        teacher_provider: TeacherProvider | None = None,
        evaluator: Evaluator | None = None,
        knowledge_window_builder: KnowledgeWindowBuilder | None = None,
    ) -> None:
        self.repository = repository
        self.teacher_provider = teacher_provider
        self.evaluator = evaluator
        self.knowledge_window_builder = knowledge_window_builder
        self._workflows: dict[str, TeachingWorkflow] = {}
        self._locks: dict[str, RLock] = {}
        self._locks_guard = RLock()

    def close(self) -> None:
        with self._locks_guard:
            workflows = tuple(self._workflows.values())
            self._workflows.clear()
            self._locks.clear()
        for workflow in workflows:
            workflow.close()

    def create_session(
        self,
        *,
        course_id: str,
        goal_id: str,
        session_id: str | None = None,
        start: bool = False,
        question: str | None = None,
        scoring_points: Sequence[ScoringPoint] | None = None,
    ) -> TeachingSessionState:
        goal = self.repository.get_goal(goal_id)
        if goal.course_id != course_id:
            raise TeachingServiceError("goal does not belong to the requested course")
        topics = self.repository.list_topics(course_id)
        session_id = session_id or str(uuid4())
        try:
            self.repository.get_teaching_session(session_id)
        except NotFoundError:
            pass
        else:
            raise SessionAlreadyExists(f"teaching session {session_id!r} already exists")

        points = (
            tuple(ScoringPoint.model_validate(point) for point in scoring_points)
            if scoring_points is not None
            else None
        )
        initial = TeachingSessionState(
            session_id=session_id,
            thread_id=session_id,
            course_id=course_id,
            goal_id=goal_id,
            current_goal=goal,
            topics=tuple(topics),
            mastery_by_topic={topic.id: topic.mastery for topic in topics},
            remaining_minutes=goal.available_minutes,
            next_action=TeachingAction.PLAN,
            status=TeachingStatus.CREATED,
            version=0,
            question_override=question,
            scoring_points_override=points,
        )
        now = datetime.now(UTC)
        self.repository.save_teaching_session(
            TeachingSession(
                session_id=session_id,
                thread_id=session_id,
                course_id=course_id,
                goal_id=goal_id,
                status=TeachingStatus.CREATED,
                version=0,
                created_at=now,
                updated_at=now,
                state=initial,
            )
        )
        return self.start_session(session_id) if start else initial

    def start_session(self, session_id: str) -> TeachingSessionState:
        lock = self._session_lock(session_id)
        with lock:
            record = self.repository.get_teaching_session(session_id)
            workflow = self._workflow_for(record)
            try:
                state = workflow.start()
            except TeachingProviderError as error:
                raise SessionProviderError(str(error)) from error
            except TeachingWorkflowError as error:
                raise SessionStateConflict(str(error)) from error
            self._save_state(record, state)
            return state

    def get_state(self, session_id: str) -> TeachingSessionState:
        lock = self._session_lock(session_id)
        with lock:
            record = self.repository.get_teaching_session(session_id)
            state = self._state_from_record_or_checkpoint(record)
            self._save_state(record, state)
            return state

    def submit_answer(
        self,
        session_id: str,
        *,
        message_id: str,
        answer: str,
        expected_version: int | None,
        spent_minutes: int = 0,
    ) -> TeachingSessionState:
        lock = self._session_lock(session_id)
        with lock:
            record = self.repository.get_teaching_session(session_id)
            answer_hash = hashlib.sha256(answer.encode("utf-8")).hexdigest()
            receipt = self.repository.get_answer_receipt(session_id, message_id)
            if receipt is not None:
                if receipt.answer_hash != answer_hash:
                    raise MessageIdConflict(
                        "message_id was already used with a different answer"
                    )
                # Idempotent retry returns the original materialised result,
                # even if the retry carries the now-advanced version.
                return receipt.state
            if expected_version is None:
                raise ExpectedVersionRequired("expected_version is required")
            state = self._state_from_record_or_checkpoint(record)
            if expected_version != state.version:
                raise SessionVersionConflict(expected_version, state.version)
            if state.status is not TeachingStatus.WAITING_ANSWER:
                raise SessionStateConflict(
                    f"session is {state.status.value}; an answer is accepted only while WAITING_ANSWER"
                )
            if spent_minutes < 0:
                raise TeachingServiceError("spent_minutes must be non-negative")
            workflow = self._workflow_for(record)
            try:
                next_state = workflow.resume(
                    {
                        "answer": answer,
                        "answer_id": message_id,
                        "spent_minutes": spent_minutes,
                    }
                )
            except TeachingProviderError as error:
                raise SessionProviderError(str(error)) from error
            except TeachingWorkflowError as error:
                raise SessionStateConflict(str(error)) from error
            self._save_state(record, next_state)
            self.repository.save_answer_receipt(
                AnswerReceipt(
                    session_id=session_id,
                    message_id=message_id,
                    answer_hash=answer_hash,
                    expected_version=expected_version,
                    state=next_state,
                )
            )
            return next_state

    def _workflow_for(self, record: TeachingSession) -> TeachingWorkflow:
        workflow = self._workflows.get(record.session_id)
        if workflow is not None:
            return workflow
        goal = self.repository.get_goal(record.goal_id)
        topics = self.repository.list_topics(record.course_id)
        state = record.state
        knowledge_window = state.knowledge_window if state else None
        if knowledge_window is None and self.knowledge_window_builder is not None:
            knowledge_window = self.knowledge_window_builder.build(
                course_id=record.course_id,
                query=topics[0].name if topics else goal.id,
                max_blocks=5,
                max_chars=4_000,
            )
        try:
            workflow = TeachingWorkflow(
                goal,
                topics,
                self.repository.database_path,
                thread_id=record.thread_id,
                session_id=record.session_id,
                teacher_provider=self.teacher_provider,
                evaluator=self.evaluator,
                knowledge_window=knowledge_window,
                question=state.question_override if state else None,
                scoring_points=state.scoring_points_override if state else None,
            )
        except (OSError, TeachingWorkflowError) as error:
            raise SessionStateConflict(str(error)) from error
        self._workflows[record.session_id] = workflow
        return workflow

    def _state_from_record_or_checkpoint(self, record: TeachingSession) -> TeachingSessionState:
        workflow = self._workflow_for(record)
        state = workflow.get_state()
        if state is None:
            if record.state is None:
                raise SessionStateConflict("session has no materialised state")
            return record.state
        return state

    def _save_state(self, record: TeachingSession, state: TeachingSessionState) -> None:
        self.repository.save_teaching_session(
            record.model_copy(
                update={
                    "status": state.status,
                    "version": state.version,
                    "updated_at": datetime.now(UTC),
                    "state": state,
                }
            )
        )

    def _session_lock(self, session_id: str) -> RLock:
        with self._locks_guard:
            return self._locks.setdefault(session_id, RLock())


# A shorter name is useful for app wiring and examples.
TeachingService = TeachingSessionService

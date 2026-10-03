from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Barrier

import pytest

from studypilot.application.teaching_service import (
    SessionVersionConflict,
    TeachingSessionService,
)
from studypilot.domain.models import Course, EvidenceLevel, ExamGoal, Topic
from studypilot.domain.teaching import TeachingSession, TeachingSessionState
from studypilot.infrastructure.sqlite_repository import SQLiteRepository


class RecordingRepository(SQLiteRepository):
    def __init__(self, database_path):
        super().__init__(database_path)
        self.cas_expected_versions: list[int] = []

    def compare_and_swap_teaching_session(self, session, *, expected_version):
        self.cas_expected_versions.append(expected_version)
        return super().compare_and_swap_teaching_session(
            session, expected_version=expected_version
        )


def _repository(tmp_path, *, repository_type=SQLiteRepository):
    repository = repository_type(tmp_path / "study.db")
    repository.initialize()
    repository.save_course(Course(id="course", name="概率论"))
    repository.save_goal(
        ExamGoal(
            id="goal",
            course_id="course",
            exam_at=datetime(2026, 12, 20, 9, tzinfo=UTC),
            available_minutes=40,
        )
    )
    repository.save_topics(
        "course",
        [
            Topic(
                id="topic-a",
                course_id="course",
                name="topic-a",
                exam_points=10,
                learning_minutes=10,
                evidence_level=EvidenceLevel.PAST_EXAM,
                evidence_confidence=1.0,
                frequency=0.8,
            ),
            Topic(
                id="topic-b",
                course_id="course",
                name="topic-b",
                exam_points=8,
                learning_minutes=10,
                evidence_level=EvidenceLevel.COURSE_MATERIAL,
                evidence_confidence=1.0,
                frequency=0.6,
            ),
        ],
    )
    return repository


def _session(repository: SQLiteRepository) -> TeachingSession:
    goal = repository.get_goal("goal")
    state = TeachingSessionState(
        session_id="session",
        thread_id="session",
        course_id="course",
        goal_id="goal",
        current_goal=goal,
        remaining_minutes=goal.available_minutes,
        version=0,
    )
    session = TeachingSession(
        session_id="session",
        thread_id="session",
        course_id="course",
        goal_id="goal",
        version=0,
        state=state,
    )
    repository.save_teaching_session(session)
    return session


def test_compare_and_swap_updates_once_and_rejects_old_version(tmp_path) -> None:
    repository = _repository(tmp_path)
    original = _session(repository)
    updated = original.model_copy(
        update={
            "version": 1,
            "state": original.state.model_copy(update={"version": 1}),
        }
    )

    assert repository.compare_and_swap_teaching_session(
        updated, expected_version=0
    ) is True
    assert repository.compare_and_swap_teaching_session(
        updated.model_copy(
            update={
                "version": 2,
                "state": updated.state.model_copy(update={"version": 2}),
            }
        ),
        expected_version=0,
    ) is False
    assert repository.get_teaching_session("session").version == 1


def test_compare_and_swap_allows_only_one_concurrent_writer(tmp_path) -> None:
    repository = _repository(tmp_path)
    original = _session(repository)
    candidate = original.model_copy(
        update={
            "version": 1,
            "state": original.state.model_copy(update={"version": 1}),
        }
    )
    repositories = [
        SQLiteRepository(repository.database_path),
        SQLiteRepository(repository.database_path),
    ]
    barrier = Barrier(2)

    def attempt(writer: SQLiteRepository) -> bool:
        barrier.wait()
        return writer.compare_and_swap_teaching_session(
            candidate, expected_version=0
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, repositories))

    assert sorted(results) == [False, True]
    assert repository.get_teaching_session("session").version == 1


def test_submit_answer_uses_cas_for_stale_waiting_version(tmp_path) -> None:
    repository = _repository(tmp_path, repository_type=RecordingRepository)
    service = TeachingSessionService(repository)
    started = service.create_session(
        course_id="course", goal_id="goal", session_id="session", start=True
    )
    assert started.version == 1

    next_state = service.submit_answer(
        "session",
        message_id="answer-1",
        answer="correct",
        expected_version=started.version,
    )
    assert next_state.version == 2
    with pytest.raises(SessionVersionConflict) as conflict:
        service.submit_answer(
            "session",
            message_id="stale-answer",
            answer="wrong",
            expected_version=started.version,
        )

    assert conflict.value.expected == started.version
    assert conflict.value.actual == next_state.version
    assert repository.cas_expected_versions == [started.version]
    assert repository.get_teaching_session("session").version == next_state.version
    assert repository.get_answer_receipt("session", "stale-answer") is None

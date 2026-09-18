from __future__ import annotations

from datetime import UTC, datetime

import pytest

from studypilot.application.teaching import (
    DeterministicFakeEvaluator,
    TeachingWorkflow,
)
from studypilot.domain.models import EvidenceLevel, ExamGoal, MasteryState, Topic
from studypilot.domain.teaching import ScoringPoint, TeachingStatus


def _goal(minutes: int = 30) -> ExamGoal:
    return ExamGoal(
        id="final",
        course_id="course",
        exam_at=datetime(2026, 12, 20, tzinfo=UTC),
        available_minutes=minutes,
    )


def _topic(
    topic_id: str = "bayes",
    *,
    points: float = 10,
    minutes: int = 10,
    mastery: MasteryState = MasteryState.UNSEEN,
) -> Topic:
    return Topic(
        id=topic_id,
        course_id="course",
        name=topic_id,
        exam_points=points,
        learning_minutes=minutes,
        mastery=mastery,
        evidence_level=EvidenceLevel.PAST_EXAM,
        evidence_confidence=1.0,
        frequency=0.8,
    )


def _rubric() -> tuple[ScoringPoint, ...]:
    return (
        ScoringPoint(id="definition", description="definition", evidence=("definition",)),
        ScoringPoint(id="formula", description="formula", evidence=("formula",)),
    )


def _workflow(tmp_path, *, topics=None, thread_id="thread", points=None):
    return TeachingWorkflow(
        _goal(),
        topics or [_topic()],
        tmp_path / "checkpoints.db",
        thread_id=thread_id,
        scoring_points=points or _rubric(),
    )


def test_start_runs_to_waiting_answer_interrupt(tmp_path) -> None:
    workflow = _workflow(tmp_path)

    state = workflow.start()

    assert state.status is TeachingStatus.WAITING_ANSWER
    assert state.next_action.value == "WAITING_ANSWER"
    assert state.question
    assert state.scoring_points == _rubric()
    assert workflow.graph.get_state(workflow._config()).next == ("waiting_answer",)
    assert workflow.last_result and workflow.last_result["__interrupt__"]
    workflow.close()


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("correct", MasteryState.READY),
        ("partial", MasteryState.FRAGILE),
        ("wrong", MasteryState.GAP),
        ("", MasteryState.GAP),
    ],
)
def test_answer_evaluation_moves_mastery_state(tmp_path, answer, expected) -> None:
    workflow = _workflow(tmp_path)
    workflow.start()

    state = workflow.resume(answer)

    assert state.mastery_state is expected
    assert state.evaluation is not None
    assert state.evaluation.mastery_state is expected
    assert state.version == 2
    assert state.replan_reason and expected.value in state.replan_reason
    workflow.close()


def test_same_thread_id_rebuild_resumes_from_sqlite_checkpoint(tmp_path) -> None:
    first = _workflow(tmp_path, thread_id="stable-thread")
    first_state = first.start()
    first.close()

    rebuilt = _workflow(tmp_path, thread_id="stable-thread")
    assert rebuilt.get_state() == first_state
    resumed = rebuilt.resume("correct")

    assert resumed.mastery_state is MasteryState.READY
    assert resumed.status is TeachingStatus.COMPLETED
    rebuilt.close()


def test_replanning_selects_next_topic_and_keeps_reason(tmp_path) -> None:
    topics = [_topic("first", points=10), _topic("second", points=8)]
    workflow = _workflow(tmp_path, topics=topics)

    initial = workflow.start()
    assert initial.current_plan_item is not None
    assert initial.current_plan_item.topic_id == "first"

    next_state = workflow.resume("correct")

    assert next_state.mastery_by_topic["first"] is MasteryState.READY
    assert next_state.current_plan_item is not None
    assert next_state.current_plan_item.topic_id == "second"
    assert "first" in (next_state.replan_reason or "")
    workflow.close()


def test_normal_evaluator_requires_explicit_rubric_evidence() -> None:
    evaluator = DeterministicFakeEvaluator()
    result = evaluator.evaluate(
        question="q",
        answer="definition only",
        scoring_points=_rubric(),
    )

    assert result.mastery_state is MasteryState.FRAGILE
    assert result.point_evaluations[0].evidence == ("definition",)
    assert result.point_evaluations[1].satisfied is False

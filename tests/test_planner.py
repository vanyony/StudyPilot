from datetime import UTC, datetime

import pytest

from studypilot.domain.models import EvidenceLevel, ExamGoal, MasteryState, PlanTier, Topic
from studypilot.domain.planner import PlanningError, RevisionPlanner


def goal(minutes: int) -> ExamGoal:
    return ExamGoal(
        id="final",
        course_id="probability",
        exam_at=datetime(2026, 12, 20, tzinfo=UTC),
        available_minutes=minutes,
    )


def topic(
    topic_id: str,
    *,
    points: float,
    minutes: int,
    mastery: MasteryState = MasteryState.UNSEEN,
    evidence: EvidenceLevel = EvidenceLevel.PAST_EXAM,
    confidence: float = 1.0,
    prerequisites: tuple[str, ...] = (),
) -> Topic:
    return Topic(
        id=topic_id,
        course_id="probability",
        name=topic_id,
        exam_points=points,
        learning_minutes=minutes,
        mastery=mastery,
        evidence_level=evidence,
        evidence_confidence=confidence,
        frequency=0.8,
        prerequisite_ids=prerequisites,
    )


def tiers(plan) -> dict[str, PlanTier]:
    return {item.topic_id: item.tier for item in plan.items}


def test_prefers_more_expected_points_per_minute() -> None:
    plan = RevisionPlanner().build(
        goal(30),
        [topic("slow", points=20, minutes=60), topic("quick", points=12, minutes=30)],
    )

    assert tiers(plan)["quick"] is PlanTier.MUST
    assert tiers(plan)["slow"] is PlanTier.DEFER
    assert plan.planned_minutes == 30


def test_ready_topic_is_skipped_even_when_high_value() -> None:
    plan = RevisionPlanner().build(
        goal(30),
        [
            topic("known", points=30, minutes=30, mastery=MasteryState.READY),
            topic("gap", points=10, minutes=30, mastery=MasteryState.GAP),
        ],
    )

    assert tiers(plan)["known"] is PlanTier.DEFER
    assert tiers(plan)["gap"] is PlanTier.MUST
    known = next(item for item in plan.items if item.topic_id == "known")
    assert any("READY" in reason for reason in known.reasons)


def test_mastery_change_reorders_plan() -> None:
    topics = [
        topic("a", points=10, minutes=30, mastery=MasteryState.GAP),
        topic("b", points=8, minutes=30, mastery=MasteryState.GAP),
    ]
    before = RevisionPlanner().build(goal(30), topics)
    after = RevisionPlanner().build(
        goal(30), [topics[0].model_copy(update={"mastery": MasteryState.READY}), topics[1]]
    )

    assert next(item for item in before.items if item.tier is PlanTier.MUST).topic_id == "a"
    assert next(item for item in after.items if item.tier is PlanTier.MUST).topic_id == "b"


def test_budget_change_reorders_must_and_strive() -> None:
    topics = [topic("a", points=12, minutes=30), topic("b", points=8, minutes=30)]

    short = RevisionPlanner().build(goal(30), topics)
    long = RevisionPlanner().build(goal(60), topics)

    assert sum(item.tier is PlanTier.MUST for item in short.items) == 1
    assert sum(item.tier is PlanTier.MUST for item in long.items) == 2


def test_prerequisite_is_scheduled_before_dependent_topic() -> None:
    plan = RevisionPlanner().build(
        goal(45),
        [
            topic("foundation", points=2, minutes=15),
            topic("exam_problem", points=20, minutes=30, prerequisites=("foundation",)),
        ],
    )

    ordered = [item.topic_id for item in plan.items if item.tier is PlanTier.MUST]
    assert ordered == ["foundation", "exam_problem"]


def test_rejects_missing_or_cyclic_prerequisites() -> None:
    with pytest.raises(PlanningError, match="missing"):
        RevisionPlanner().build(
            goal(30), [topic("a", points=10, minutes=20, prerequisites=("missing",))]
        )

    cyclic = [
        topic("a", points=10, minutes=10, prerequisites=("b",)),
        topic("b", points=10, minutes=10, prerequisites=("a",)),
    ]
    with pytest.raises(PlanningError, match="cycle"):
        RevisionPlanner().build(goal(30), cyclic)


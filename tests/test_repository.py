from datetime import UTC, datetime

from studypilot.domain.models import Course, EvidenceLevel, ExamGoal, Topic
from studypilot.domain.planner import RevisionPlanner
from studypilot.infrastructure.sqlite_repository import SQLiteRepository


def test_round_trip_and_persisted_plan(tmp_path) -> None:
    repository = SQLiteRepository(tmp_path / "study.db")
    repository.initialize()
    course = Course(id="probability", name="概率论")
    goal = ExamGoal(
        id="final",
        course_id=course.id,
        exam_at=datetime(2026, 12, 20, tzinfo=UTC),
        available_minutes=60,
    )
    topics = [
        Topic(
            id="bayes",
            course_id=course.id,
            name="贝叶斯公式",
            exam_points=10,
            learning_minutes=30,
            evidence_level=EvidenceLevel.PAST_EXAM,
            evidence_confidence=0.9,
        )
    ]

    repository.save_course(course)
    repository.save_goal(goal)
    repository.save_topics(course.id, topics)
    plan = RevisionPlanner().build(repository.get_goal(goal.id), repository.list_topics(course.id))
    repository.save_plan(plan)

    assert repository.get_course(course.id) == course
    assert repository.get_goal(goal.id) == goal
    assert repository.list_topics(course.id) == topics
    assert repository.get_plan(goal.id) == plan


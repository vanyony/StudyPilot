"""Checkpointed LangGraph teaching loop and provider protocols.

The deterministic providers remain the default, while phase-seven callers
may inject the OpenAI-compatible providers from :mod:`studypilot.application.llm`.
``TeachingWorkflow`` accepts a course goal and topics, runs a graph through an
interrupt, and stores every checkpoint in SQLite so another process can
construct the graph again with the same ``thread_id`` and resume it.
"""

from __future__ import annotations

from pathlib import Path
import re
import sqlite3
from inspect import Parameter, signature
from threading import RLock
from typing import Any, Protocol, Sequence, TypedDict
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from studypilot.domain.knowledge import Citation, KnowledgeWindow, KnowledgeWindowItem
from studypilot.domain.models import (
    ExamGoal,
    MasteryState,
    Plan,
    PlanItem,
    PlanTier,
    Topic,
)
from studypilot.domain.teaching import (
    EvaluationResult,
    ScoringPoint,
    ScoringPointEvaluation,
    TeachingAction,
    TeachingContent,
    TeachingSessionState,
    TeachingStatus,
)
from studypilot.application.llm import (
    adjudicate_evaluation,
    validate_teaching_content_references,
)


class TeachingWorkflowError(ValueError):
    """Raised when a workflow cannot be started or resumed."""


class TeachingProviderError(TeachingWorkflowError):
    """A teacher/evaluator failed; the workflow must not invent a result."""


class TeacherProvider(Protocol):
    """Structured teacher boundary for a future model-backed provider."""

    def teach(
        self,
        topic: Topic,
        knowledge_window: KnowledgeWindow,
        attempt: int = 1,
        question_override: str | None = None,
        scoring_points_override: Sequence[ScoringPoint] | None = None,
        *,
        mastery_state: MasteryState = MasteryState.UNSEEN,
        remaining_minutes: int = 0,
    ) -> TeachingContent: ...


class Evaluator(Protocol):
    """Rubric-constrained structured evaluator boundary."""

    def evaluate(
        self,
        question: str = "",
        answer: str = "",
        scoring_points: Sequence[ScoringPoint] = (),
        *,
        rubric: Sequence[ScoringPoint] | None = None,
        allowed_citations: KnowledgeWindow
        | Sequence[Citation]
        | Sequence[KnowledgeWindowItem]
        | Sequence[str]
        | None = None,
    ) -> EvaluationResult: ...


class DeterministicFakeEvaluator:
    """Offline evaluator whose decisions are entirely reproducible.

    Every normal answer is checked against the explicit ``evidence`` phrases
    on each scoring point.  Exact fixture labels are useful for integration
    tests and are recorded as fixture evidence; they are not a production
    semantic evaluator.
    """

    _FIXTURE_LABELS = {
        "correct": "correct",
        "正确": "correct",
        "完全正确": "correct",
        "partial": "partial",
        "partially correct": "partial",
        "部分": "partial",
        "部分正确": "partial",
        "wrong": "wrong",
        "incorrect": "wrong",
        "错误": "wrong",
        "不知道": "wrong",
    }

    def evaluate(
        self,
        question: str = "",
        answer: str = "",
        scoring_points: Sequence[ScoringPoint] = (),
        *,
        rubric: Sequence[ScoringPoint] | None = None,
        allowed_citations: KnowledgeWindow
        | Sequence[Citation]
        | Sequence[KnowledgeWindowItem]
        | Sequence[str]
        | None = None,
    ) -> EvaluationResult:
        # Accept both ``evaluate(answer, rubric)`` and the fully structured
        # keyword form used by the graph; the former is handy for tiny tests.
        if not isinstance(answer, str):
            scoring_points = answer
            answer = question
            question = ""
        if rubric is not None:
            scoring_points = rubric
        del question, allowed_citations
        points = tuple(ScoringPoint.model_validate(point) for point in scoring_points)
        if not points:
            raise TeachingWorkflowError("a rubric must contain at least one scoring point")
        normalized = _normalize(answer)
        label = self._FIXTURE_LABELS.get(normalized)
        if label is not None:
            if label == "correct":
                satisfied_indexes = set(range(len(points)))
            elif label == "partial":
                satisfied_indexes = set(range(max(1, (len(points) + 1) // 2)))
            else:
                satisfied_indexes = set()
            point_results = tuple(
                ScoringPointEvaluation(
                    scoring_point_id=point.id,
                    satisfied=index in satisfied_indexes,
                    evidence=(f"fixture:{label}",) if index in satisfied_indexes else (),
                    reason=(
                        "fixture marker satisfies this scoring point"
                        if index in satisfied_indexes
                        else "fixture marker gives no evidence for this scoring point"
                    ),
                )
                for index, point in enumerate(points)
            )
        else:
            point_results = tuple(self._evaluate_point(point, normalized) for point in points)

        total_weight = sum(point.weight for point in points)
        earned_weight = sum(
            point.weight
            for point, result in zip(points, point_results, strict=True)
            if result.satisfied
        )
        score = earned_weight / total_weight if total_weight else 0.0
        required = [
            result
            for point, result in zip(points, point_results, strict=True)
            if point.required
        ]
        if score >= 1.0 - 1e-9 and all(result.satisfied for result in required):
            mastery = MasteryState.READY
        elif score <= 1e-9:
            mastery = MasteryState.GAP
        else:
            mastery = MasteryState.FRAGILE
        if not normalized:
            reason = "空答案：未观察到任何评分点证据"
        else:
            matched_ids = [
                point.id
                for point, result in zip(points, point_results, strict=True)
                if result.satisfied
            ]
            reason = (
                f"命中评分点 {', '.join(matched_ids) if matched_ids else '无'}；"
                f"证据得分 {score:.0%}，掌握状态为 {mastery.value}"
            )
        return EvaluationResult(
            score=round(score, 6),
            mastery_state=mastery,
            point_evaluations=point_results,
            reason=reason,
            next_action=TeachingAction.REPLAN,
        )

    @staticmethod
    def _evaluate_point(
        point: ScoringPoint, normalized_answer: str
    ) -> ScoringPointEvaluation:
        needles = tuple(
            _normalize(value) for value in point.evidence if _normalize(value)
        )
        if not needles:
            needles = _description_needles(point.description)
        matched = tuple(needle for needle in needles if needle in normalized_answer)
        return ScoringPointEvaluation(
            scoring_point_id=point.id,
            satisfied=bool(matched),
            evidence=matched,
            reason=(
                f"观察到显式证据：{'、'.join(matched)}"
                if matched
                else "未观察到评分点要求的显式证据"
            ),
        )


DeterministicEvaluator = DeterministicFakeEvaluator
FakeEvaluator = DeterministicFakeEvaluator


class DeterministicTeacherProvider:
    """Offline structured teacher used by the deterministic workflow."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def teach(
        self,
        topic: Topic,
        knowledge_window: KnowledgeWindow,
        attempt: int = 1,
        question_override: str | None = None,
        scoring_points_override: Sequence[ScoringPoint] | None = None,
        *,
        mastery_state: MasteryState = MasteryState.UNSEEN,
        remaining_minutes: int = 0,
    ) -> TeachingContent:
        del mastery_state, remaining_minutes
        self.calls.append(topic.id)
        points = tuple(
            ScoringPoint.model_validate(point)
            for point in (scoring_points_override or ())
        )
        if not points:
            points = (
                ScoringPoint(
                    id=f"{topic.id}:core",
                    description=f"说明{topic.name}的核心概念或解题步骤",
                    evidence=(topic.name,),
                ),
            )
        references = [
            f"{item.hit.citation.display_name}#{item.hit.citation.block_index}"
            for item in knowledge_window.items
        ]
        source_note = (
            "；Knowledge Window 引用：" + "、".join(references)
            if references
            else "；当前 Knowledge Window 没有可用原文块"
        )
        explanation = (
            f"本轮目标是{topic.name}。围绕评分点掌握最小必要知识，"
            f"再用一道题检查能否主动复现（第 {attempt} 次练习）{source_note}。"
        )
        return TeachingContent(
            explanation=explanation,
            question=question_override or f"请解释{topic.name}，并给出关键定义或解题步骤。",
            scoring_points=points,
            citations=tuple(item.hit.citation.block_id for item in knowledge_window.items),
        )


FakeTeacherProvider = DeterministicTeacherProvider


class UnconfiguredLLMTeacherProvider:
    """Backward-compatible placeholder for callers that have no LLM config."""

    def teach(self, **_: Any) -> TeachingContent:
        raise NotImplementedError("use LLMTeacherProvider with explicit configuration")


class UnconfiguredLLMEvaluator:
    """Backward-compatible placeholder for callers that have no LLM config."""

    def evaluate(self, **_: Any) -> EvaluationResult:
        raise NotImplementedError("use LLMEvaluatorProvider with explicit configuration")


class _GraphState(TypedDict, total=False):
    session_id: str
    thread_id: str
    course_id: str
    goal_id: str
    current_goal: dict[str, Any]
    topics: list[dict[str, Any]]
    plan: dict[str, Any] | None
    current_plan_item: dict[str, Any] | None
    knowledge_window: dict[str, Any] | None
    teaching_text: str | None
    teaching_citations: list[str]
    question_id: str | None
    question: str | None
    scoring_points: list[dict[str, Any]]
    student_answer: str | None
    evaluation: dict[str, Any] | None
    mastery_state: str
    mastery_by_topic: dict[str, str]
    remaining_minutes: int
    next_action: str
    status: str
    version: int
    replan_reason: str | None
    attempt: int
    last_answer_id: str | None
    spent_minutes: int
    question_override: str | None
    scoring_points_override: list[dict[str, Any]] | None


class TeachingWorkflow:
    """Build and run one checkpointed graph for a stable teaching thread."""

    def __init__(
        self,
        goal: ExamGoal,
        topics: Sequence[Topic],
        database_path: str | Path | None = None,
        *,
        checkpoint_path: str | Path | None = None,
        db_path: str | Path | None = None,
        thread_id: str | None = None,
        session_id: str | None = None,
        teacher_provider: TeacherProvider | None = None,
        evaluator: Evaluator | None = None,
        knowledge_window: KnowledgeWindow | None = None,
        question: str | None = None,
        scoring_points: Sequence[ScoringPoint] | None = None,
    ) -> None:
        self.goal = goal
        self.topics = tuple(topics)
        if any(topic.course_id != goal.course_id for topic in self.topics):
            raise TeachingWorkflowError("all topics must belong to the goal course")
        if len({topic.id for topic in self.topics}) != len(self.topics):
            raise TeachingWorkflowError("topic ids must be unique")
        self.thread_id = thread_id or session_id or str(uuid4())
        self.session_id = session_id or self.thread_id
        self.teacher_provider = teacher_provider or DeterministicTeacherProvider()
        self.evaluator = evaluator or DeterministicFakeEvaluator()
        self.knowledge_window = knowledge_window or KnowledgeWindow(
            query=self.topics[0].name if self.topics else goal.id,
            course_id=goal.course_id,
            max_blocks=5,
            max_chars=4_000,
            used_chars=0,
            items=(),
        )
        self.question_override = question
        self.scoring_points_override = (
            tuple(ScoringPoint.model_validate(point) for point in scoring_points)
            if scoring_points is not None
            else None
        )
        selected_database_path = database_path or checkpoint_path or db_path
        if selected_database_path is None:
            raise TeachingWorkflowError("a SQLite checkpoint path is required")
        self.database_path = Path(selected_database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection_target = ":memory:" if str(selected_database_path) == ":memory:" else str(self.database_path)
        self._connection = sqlite3.connect(connection_target, check_same_thread=False)
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._checkpointer = SqliteSaver(self._connection)
        self._checkpointer.setup()
        self._lock = RLock()
        self.graph = self._build_graph()
        self.last_result: dict[str, Any] | None = None

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def start(self) -> TeachingSessionState:
        """Run until the first ``waiting_answer`` interrupt."""

        with self._lock:
            snapshot = self.graph.get_state(self._config())
            if snapshot.values:
                return self._materialize(snapshot.values)
            initial = self._initial_state()
            self.last_result = self.graph.invoke(initial, self._config())
            return self._current_state()

    # ``run`` is a small ergonomic alias for callers that think of the graph
    # as a runnable rather than a session service.
    run = start

    def resume(self, answer: str | dict[str, Any]) -> TeachingSessionState:
        """Resume this thread from the interrupt and run to the next pause."""

        with self._lock:
            snapshot = self.graph.get_state(self._config())
            if not snapshot.values:
                raise TeachingWorkflowError("workflow has not been started")
            if not snapshot.interrupts:
                raise TeachingWorkflowError("workflow is not waiting for an answer")
            payload = answer if isinstance(answer, dict) else {"answer": answer}
            self.last_result = self.graph.invoke(
                Command(resume=payload), self._config()
            )
            return self._current_state()

    resume_answer = resume

    def get_state(self) -> TeachingSessionState | None:
        with self._lock:
            snapshot = self.graph.get_state(self._config())
            return self._materialize(snapshot.values) if snapshot.values else None

    def _initial_state(self) -> dict[str, Any]:
        state = TeachingSessionState(
            session_id=self.session_id,
            thread_id=self.thread_id,
            course_id=self.goal.course_id,
            goal_id=self.goal.id,
            current_goal=self.goal,
            topics=self.topics,
            mastery_by_topic={topic.id: topic.mastery for topic in self.topics},
            remaining_minutes=self.goal.available_minutes,
            next_action=TeachingAction.PLAN,
            status=TeachingStatus.CREATED,
            version=0,
            question_override=self.question_override,
            scoring_points_override=self.scoring_points_override,
        )
        return self._values(state)

    def _build_graph(self):
        builder = StateGraph(_GraphState)
        builder.add_node("planning", self._planning_node)
        builder.add_node("teaching", self._teaching_node)
        builder.add_node("questioning", self._questioning_node)
        builder.add_node("waiting_answer", self._waiting_answer_node)
        builder.add_node("evaluating", self._evaluating_node)
        builder.add_node("replanning", self._replanning_node)
        builder.add_edge(START, "planning")
        builder.add_conditional_edges(
            "planning", self._route, {"teaching": "teaching", "complete": END}
        )
        builder.add_edge("teaching", "questioning")
        builder.add_edge("questioning", "waiting_answer")
        builder.add_edge("waiting_answer", "evaluating")
        builder.add_edge("evaluating", "replanning")
        builder.add_conditional_edges(
            "replanning", self._route, {"teaching": "teaching", "complete": END}
        )
        return builder.compile(checkpointer=self._checkpointer)

    def _planning_node(self, state: _GraphState) -> dict[str, Any]:
        plan, current, topics = self._make_plan(state)
        version = max(1, int(state.get("version", 0)))
        if current is None:
            return {
                "plan": self._values(plan),
                "current_plan_item": None,
                "topics": [self._values(topic) for topic in topics],
                "status": TeachingStatus.COMPLETED.value,
                "next_action": TeachingAction.COMPLETE.value,
                "version": version,
                "replan_reason": "没有可安排的未掌握考点",
            }
        return {
            "plan": self._values(plan),
            "current_plan_item": self._values(current),
            "topics": [self._values(topic) for topic in topics],
            "mastery_state": self._mastery_for(state, current.topic_id, topics).value,
            "status": TeachingStatus.TEACHING.value,
            "next_action": TeachingAction.TEACH.value,
            "version": version,
            "knowledge_window": None,
            "replan_reason": None,
        }

    def _teaching_node(self, state: _GraphState) -> dict[str, Any]:
        if state.get("current_plan_item") is None:
            return {
                "status": TeachingStatus.COMPLETED.value,
                "next_action": TeachingAction.COMPLETE.value,
            }
        item = PlanItem.model_validate(state["current_plan_item"])
        topic = next(topic for topic in self._topics(state) if topic.id == item.topic_id)
        mastery_state = self._mastery_for(state, item.topic_id, self._topics(state))
        try:
            content = _invoke_compatible(
                self.teacher_provider.teach,
                topic=topic,
                knowledge_window=self.knowledge_window,
                attempt=int(state.get("attempt", 0)) + 1,
                question_override=state.get("question_override"),
                scoring_points_override=self._scoring_overrides(state),
                mastery_state=mastery_state,
                remaining_minutes=max(0, int(state.get("remaining_minutes", 0))),
            )
            if not isinstance(content, TeachingContent):
                content = TeachingContent.model_validate(content)
            content = validate_teaching_content_references(content, self.knowledge_window)
        except TeachingWorkflowError:
            raise
        except Exception as error:
            raise TeachingProviderError("teacher provider failed; no teaching content was committed") from error
        return {
            "knowledge_window": self._values(self.knowledge_window),
            "teaching_text": content.explanation,
            "teaching_citations": list(content.citations),
            "question": content.question,
            "scoring_points": [self._values(point) for point in content.scoring_points],
            "status": TeachingStatus.QUESTIONING.value,
            "next_action": TeachingAction.QUESTION.value,
        }

    @staticmethod
    def _questioning_node(state: _GraphState) -> dict[str, Any]:
        item = PlanItem.model_validate(state["current_plan_item"])
        attempt = int(state.get("attempt", 0)) + 1
        return {
            "question_id": f"{item.topic_id}:attempt-{attempt}",
            "attempt": attempt,
            "status": TeachingStatus.WAITING_ANSWER.value,
            "next_action": TeachingAction.WAITING_ANSWER.value,
        }

    @staticmethod
    def _waiting_answer_node(state: _GraphState) -> dict[str, Any]:
        resumed = interrupt(
            {
                "question_id": state.get("question_id"),
                "question": state.get("question"),
                "scoring_points": state.get("scoring_points", []),
                "version": state.get("version", 0),
            }
        )
        if isinstance(resumed, dict):
            answer = str(resumed.get("answer", "") or "")
            answer_id = resumed.get("answer_id")
            spent = int(resumed.get("spent_minutes", 0) or 0)
        else:
            answer = str(resumed or "")
            answer_id = None
            spent = 0
        return {
            "student_answer": answer,
            "last_answer_id": str(answer_id or uuid4()),
            "spent_minutes": max(0, spent),
            "status": TeachingStatus.EVALUATING.value,
            "next_action": TeachingAction.EVALUATE.value,
        }

    def _evaluating_node(self, state: _GraphState) -> dict[str, Any]:
        points = tuple(
            ScoringPoint.model_validate(point) for point in state.get("scoring_points", [])
        )
        answer = str(state.get("student_answer") or "")
        try:
            result = _invoke_compatible(
                self.evaluator.evaluate,
                question=str(state.get("question") or ""),
                answer=answer,
                scoring_points=points,
                allowed_citations=self.knowledge_window,
            )
            if not isinstance(result, EvaluationResult):
                result = EvaluationResult.model_validate(result)
            # Model/provider suggestions are not state transitions.  The
            # rubric, explicit evidence and current Knowledge Window are
            # checked again here before the server derives mastery/action.
            result = adjudicate_evaluation(
                result,
                points,
                self.knowledge_window,
                answer=answer,
            )
        except TeachingWorkflowError:
            raise
        except Exception as error:
            raise TeachingProviderError("evaluator failed; no mastery transition was committed") from error
        item = PlanItem.model_validate(state["current_plan_item"])
        mastery = dict(state.get("mastery_by_topic", {}))
        mastery[item.topic_id] = result.mastery_state.value
        return {
            "evaluation": self._values(result),
            "mastery_state": result.mastery_state.value,
            "mastery_by_topic": mastery,
            "status": TeachingStatus.REPLANNING.value,
            "next_action": TeachingAction.REPLAN.value,
        }

    def _replanning_node(self, state: _GraphState) -> dict[str, Any]:
        previous = PlanItem.model_validate(state["current_plan_item"])
        old_mastery = MasteryState(state.get("mastery_state", MasteryState.UNSEEN.value))
        evaluation = EvaluationResult.model_validate(state["evaluation"])
        spent = max(0, int(state.get("spent_minutes", 0)))
        remaining = max(0, int(state.get("remaining_minutes", 0)) - spent)
        replanning_state = dict(state)
        replanning_state["remaining_minutes"] = remaining
        plan, current, topics = self._make_plan(replanning_state)
        common = {
            "plan": self._values(plan),
            "topics": [self._values(topic) for topic in topics],
            "mastery_by_topic": {topic.id: topic.mastery.value for topic in topics},
            "remaining_minutes": remaining,
            "version": int(state.get("version", 0)) + 1,
            "knowledge_window": None,
        }
        reason = (
            f"{evaluation.reason}；{previous.topic_name} 掌握状态从 "
            f"{old_mastery.value} 迁移到 {evaluation.mastery_state.value}；"
            f"剩余 {remaining} 分钟"
        )
        if current is None:
            return {
                **common,
                "current_plan_item": None,
                "status": TeachingStatus.COMPLETED.value,
                "next_action": TeachingAction.COMPLETE.value,
                "replan_reason": reason + "，没有下一项",
            }
        return {
            **common,
            "current_plan_item": self._values(current),
            "mastery_state": self._mastery_for(replanning_state, current.topic_id, topics).value,
            "status": TeachingStatus.TEACHING.value,
            "next_action": TeachingAction.TEACH.value,
            "replan_reason": reason + f"，下一项为 {current.topic_name}",
        }

    @staticmethod
    def _route(state: _GraphState) -> str:
        return "teaching" if state.get("current_plan_item") is not None else "complete"

    def _make_plan(self, state: _GraphState) -> tuple[Plan, PlanItem | None, list[Topic]]:
        topics = self._topics(state)
        goal = ExamGoal.model_validate(state["current_goal"])
        budget = max(0, int(state.get("remaining_minutes", goal.available_minutes)))
        plan = _planner().build(goal.model_copy(update={"available_minutes": budget}), topics)
        current = next((item for item in plan.items if item.tier is PlanTier.MUST), None)
        return plan, current, topics

    @staticmethod
    def _topics(state: _GraphState) -> list[Topic]:
        mastery = state.get("mastery_by_topic", {})
        result = []
        for raw in state.get("topics", []):
            topic = Topic.model_validate(raw)
            result.append(
                topic.model_copy(
                    update={"mastery": MasteryState(mastery.get(topic.id, topic.mastery.value))}
                )
            )
        return result

    @staticmethod
    def _mastery_for(state: _GraphState, topic_id: str, topics: Sequence[Topic]) -> MasteryState:
        value = state.get("mastery_by_topic", {}).get(topic_id)
        if value is not None:
            return MasteryState(value)
        return next(topic.mastery for topic in topics if topic.id == topic_id)

    @staticmethod
    def _scoring_overrides(state: _GraphState) -> tuple[ScoringPoint, ...] | None:
        raw = state.get("scoring_points_override")
        return (
            tuple(ScoringPoint.model_validate(item) for item in raw)
            if raw is not None
            else None
        )

    def _current_state(self) -> TeachingSessionState:
        snapshot = self.graph.get_state(self._config())
        if not snapshot.values:
            raise TeachingWorkflowError("workflow did not create a checkpoint")
        return self._materialize(snapshot.values)

    def _config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.thread_id}}

    @staticmethod
    def _values(value: Any) -> Any:
        return value.model_dump(mode="json") if hasattr(value, "model_dump") else value

    @staticmethod
    def _materialize(values: dict[str, Any]) -> TeachingSessionState:
        return TeachingSessionState.model_validate(values)


# Friendly aliases for callers that describe the component as a graph.
TeachingGraph = TeachingWorkflow
MinimalTeachingWorkflow = TeachingWorkflow


def _invoke_compatible(function: Any, **kwargs: Any) -> Any:
    """Call current providers while retaining compatibility with phase-A fakes.

    The new LLM boundary receives mastery/time and allowed-citation keywords.
    A small user-supplied provider written against the earlier protocol may
    not accept them, so we inspect its signature before invocation.  This
    avoids catching a provider's own ``TypeError`` and accidentally retrying a
    request that may already have reached an external service.
    """

    try:
        parameters = signature(function).parameters
    except (TypeError, ValueError):
        return function(**kwargs)
    if any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return function(**kwargs)
    accepted = {name: value for name, value in kwargs.items() if name in parameters}
    return function(**accepted)


def _planner():
    from studypilot.domain.planner import RevisionPlanner

    return RevisionPlanner()


def _normalize(value: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFKC", value or "").strip().lower()


def _description_needles(description: str) -> tuple[str, ...]:
    normalized = _normalize(description)
    cjk = re.findall(r"[\u3400-\u9fff]{2,}", normalized)
    words = re.findall(r"[a-z0-9_]{3,}", normalized)
    return tuple(dict.fromkeys(cjk + words))


# Re-export the concrete providers from the workflow module as well as their
# dedicated module; existing callers naturally look here for teaching pieces.
from studypilot.application.llm import (  # noqa: E402  (intentional re-export)
    LLMConfigurationError,
    LLMConfig,
    LLMEmptyResponseError,
    LLMInvalidJSONError,
    LLMProviderConfig,
    LLMProviderError,
    LLMRequestError,
    LLMResponseValidationError,
    LLMTeacherProvider,
    LLMEvaluatorProvider,
    LLMTimeoutError,
    OpenAICompatibleConfig,
    UnauthorizedCitationError,
    adjudicate_evaluation,
    build_evaluator_messages,
    build_teacher_messages,
    validate_evaluation_references,
    validate_teaching_content_references,
)

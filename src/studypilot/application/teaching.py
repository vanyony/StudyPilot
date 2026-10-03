"""Checkpointed LangGraph teaching loop and provider protocols.

Production requires configured learning and evaluation providers; deterministic
providers are explicitly injected by offline tests.
``TeachingWorkflow`` accepts a course goal and topics, runs a graph through an
interrupt, and stores every checkpoint in SQLite so another process can
construct the graph again with the same ``thread_id`` and resume it.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
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
    processed_answers: dict[str, str]
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
    analysis: dict[str, Any]
    analysis_version: int
    user_statements: list[str]
    agent_decision: dict[str, Any] | None
    action_history: list[dict[str, Any]]
    completion_topic_ids: list[str]
    observations: list[dict[str, Any]]
    pending_kind: str | None
    return_to_topic: str | None
    round_steps: int
    stop_reason: str | None
    last_tool_error: str | None
    tool_errors: int
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
        learning_provider=None,
        repository=None,
        knowledge_window_builder=None,
        user_statements: Sequence[str] = (),
    ) -> None:
        from studypilot.application.learning_agent import LLMLearningProvider
        self.learning_provider = learning_provider
        self.repository = repository
        self.window_builder = knowledge_window_builder
        self.user_statements = list(user_statements)
        self.goal = goal
        self.topics = tuple(topics)
        if any(topic.course_id != goal.course_id for topic in self.topics):
            raise TeachingWorkflowError("all topics must belong to the goal course")
        if len({topic.id for topic in self.topics}) != len(self.topics):
            raise TeachingWorkflowError("topic ids must be unique")
        self.thread_id = thread_id or session_id or str(uuid4())
        self.session_id = session_id or self.thread_id
        self.teacher_provider = teacher_provider or DeterministicTeacherProvider()
        self.evaluator = evaluator
        self._explicit_evaluator = evaluator
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

    def prepare_plan(self) -> Plan:
        """Analyze evidence and ask the model for a plan without starting a lesson."""
        with self._lock:
            state = self._initial_state()
            state.update(self._bootstrap_node(state))
            for _ in range(8):
                state.update(self._decision_node(state))
                state.update(self._execute_node(state))
                if state.get("plan"):
                    return Plan.model_validate(state["plan"])
                if state.get("pending_kind"):
                    raise TeachingWorkflowError(state.get("question") or "制定计划需要补充信息")
            raise TeachingWorkflowError("本轮未能形成有效计划，请补充资料或目标")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def start(self) -> TeachingSessionState:
        """Run until the first ``waiting_answer`` interrupt."""

        with self._lock:
            snapshot = self.graph.get_state(self._config())
            if snapshot.values:
                if snapshot.next and not snapshot.interrupts:
                    self.last_result = self.graph.invoke(None, {**self._config(), "recursion_limit": 100})
                    return self._current_state()
                return self._materialize(snapshot.values)
            initial = self._initial_state()
            self.last_result = self.graph.invoke(initial, {**self._config(), "recursion_limit": 100})
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
                Command(resume=payload), {**self._config(), "recursion_limit": 100}
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
            user_statements=self.user_statements,
        )
        return self._values(state)

    def _build_graph(self):
        builder = StateGraph(_GraphState)
        builder.add_node("bootstrap", self._bootstrap_node)
        builder.add_node("decide", self._decision_node)
        builder.add_node("execute", self._execute_node)
        builder.add_node("waiting_answer", self._agent_wait_node)
        builder.add_node("observe", self._observe_node)
        builder.add_edge(START, "bootstrap")
        builder.add_edge("bootstrap", "decide")
        builder.add_edge("decide", "execute")
        builder.add_conditional_edges("execute", self._agent_route,
            {"decide": "decide", "wait": "waiting_answer", "complete": END})
        builder.add_edge("waiting_answer", "observe")
        builder.add_edge("observe", "decide")
        return builder.compile(checkpointer=self._checkpointer)

    def _provider(self):
        if self.learning_provider is None:
            from studypilot.application.learning_agent import LLMLearningProvider
            try:
                self.learning_provider = LLMLearningProvider()
            except Exception as error:
                raise TeachingProviderError("学习 Agent 未配置模型；请设置 LLM API key 和 model") from error
        return self.learning_provider

    def _bootstrap_node(self, state):
        if state.get("analysis") or self.repository is None:
            return {"version": max(1, state.get("version", 0))}
        from studypilot.application.learning_agent import CourseAnalysis, validate_analysis
        sources = self.repository.list_sources(self.goal.course_id)
        source_map = {source.asset.id: source for source in sources}
        blocks = self.repository.list_blocks(self.goal.course_id)
        if not blocks and state.get("topics"):
            return {"analysis": {"assumptions": ["用户手工修正的考点清单；没有自动分析证据"]}, "version": max(1, state.get("version", 0))}
        if not blocks:
            raise TeachingWorkflowError("请先上传并解析课程资料，再开始学习")
        # Deduplicate identical blobs without multiplying exam frequency.
        seen = set()
        rows = []
        for block in blocks:
            source = source_map.get(block.source_asset_id)
            key = (source.asset.blob_id if source else block.source_asset_id, block.block_index)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"block_id": block.id, "text": block.text,
                "kind": source.asset.document_kind.value if source else "UNKNOWN",
                "source_id": block.source_asset_id, "section": block.section,
                "page": block.page_number})
        statements = state.get("user_statements", [])
        statement_ids = {f"statement:{i}" for i in range(len(statements))}
        common = {"course_id": self.goal.course_id, "goal": self._values(self.goal),
            "user_statements": [{"id": f"statement:{i}", "text": value} for i, value in enumerate(statements)],
            "existing_topics": state.get("topics", []), "instruction": "已有考点保持稳定 ID，修改依据，不重命名已有 ID"}
        analysis = None
        batch, chars = [], 0
        batches = []
        for row in rows:
            # Preserve coverage of large blocks by chunking, never silent truncation.
            for offset in range(0, len(row["text"]), 12000):
                chunk = {**row, "text": row["text"][offset:offset + 12000], "offset": offset}
                if batch and chars + len(chunk["text"]) > 18000:
                    batches.append(batch); batch, chars = [], 0
                batch.append(chunk); chars += len(chunk["text"])
        if batch:
            batches.append(batch)
        provider = self._provider()
        allowed = {row["block_id"] for row in rows}
        covered = []
        for batch in batches:
            try:
                context = {**common, "blocks": batch,
                    "previous_analysis": self._values(analysis) if analysis else None}
                fingerprint = hashlib.sha256(json.dumps(context, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
                self._connection.execute("CREATE TABLE IF NOT EXISTS learning_analysis_cache (fingerprint TEXT PRIMARY KEY, result TEXT NOT NULL)")
                cached = self._connection.execute("SELECT result FROM learning_analysis_cache WHERE fingerprint=?", (fingerprint,)).fetchone()
                analysis = CourseAnalysis.model_validate(json.loads(cached[0]) if cached else provider.analyze(context))
                validate_analysis(analysis, self.goal.course_id,
                    set(covered) | {row["block_id"] for row in batch}, statement_ids)
                self._connection.execute("INSERT OR IGNORE INTO learning_analysis_cache VALUES (?, ?)", (fingerprint, analysis.model_dump_json()))
                self._connection.commit()
            except Exception as error:
                raise TeachingProviderError("资料分析失败，未提交考点或伪造计划") from error
            covered.extend(row["block_id"] for row in batch)
        analysis.covered_block_ids = list(dict.fromkeys(covered))
        analysis.missing_sources = [source.asset.display_name for source in sources
            if source.blob.parse_status.value != "READY"]
        if self.repository:
            self.repository.save_topics(self.goal.course_id, analysis.topics)
        return {"analysis": self._values(analysis), "analysis_version": state.get("analysis_version", 0) + 1,
            "topics": [self._values(topic) for topic in analysis.topics],
            "mastery_by_topic": {topic.id: state.get("mastery_by_topic", {}).get(topic.id, "UNSEEN") for topic in analysis.topics},
            "version": max(1, state.get("version", 0)), "plan": None}

    def _decision_node(self, state):
        from studypilot.application.learning_agent import AgentDecision
        context = dict(state)
        context["topics"] = [self._values(topic) for topic in self._topics(state)]
        context["available_topics"] = [{"id": topic.id, "name": topic.name} for topic in self._topics(state)]
        context["budget_reference"] = self._values(_planner().build(
            self.goal.model_copy(update={"available_minutes": max(0, state.get("remaining_minutes", 0))}), self._topics(state)))
        context["action_history"] = state.get("action_history", [])[-12:]
        context["observations"] = state.get("observations", [])[-12:]
        if state.get("stop_reason") == "user_stopped":
            decision = AgentDecision(action="complete", reason="按你的要求暂停学习", completion_basis="user_stopped")
        elif state.get("remaining_minutes", 0) <= 0:
            decision = AgentDecision(action="complete", reason="本轮可用时间已用完", completion_basis="time_exhausted")
        elif state.get("round_steps", 0) >= 8:
            decision = AgentDecision(action="ask", reason="本轮已到行动上限，保留进度让学生决定是否继续",
                question="这一轮先停在这里，当前进度已保存。要继续学习，还是调整目标？")
        else:
            try:
                decision = AgentDecision.model_validate(self._provider().decide(context))
            except Exception as error:
                raise TeachingProviderError("学习决策失败，未擅自改变状态") from error
        return {"agent_decision": self._values(decision), "round_steps": state.get("round_steps", 0) + 1,
            "replan_reason": decision.reason}

    def _execute_node(self, state):
        try:
            result = self._execute_action(state)
            return {**result, "last_tool_error": None, "tool_errors": 0}
        except (TeachingProviderError, TeachingWorkflowError, ValueError) as error:
            errors = state.get("tool_errors", 0) + 1
            update = {"last_tool_error": str(error), "tool_errors": errors, "pending_kind": None}
            if errors >= 2:
                update.update({"pending_kind": "ask", "question": "当前安排未通过检查，进度已保留。你可以补充要求或稍后重试。",
                    "question_id": f"error:{state.get('version', 0)}", "scoring_points": [],
                    "status": TeachingStatus.WAITING_ANSWER.value, "next_action": TeachingAction.WAITING_ANSWER.value})
            return update

    def _execute_action(self, state):
        from studypilot.application.learning_agent import AgentDecision, LearningAction, validate_plan
        decision = AgentDecision.model_validate(state["agent_decision"])
        topics = self._topics(state)
        by_id = {topic.id: topic for topic in topics}
        topic = by_id.get(decision.topic_id)
        if decision.topic_id is not None and topic is None:
            raise TeachingProviderError("模型选择了不存在的考点")
        if decision.return_to_topic and decision.return_to_topic not in by_id:
            raise TeachingProviderError("返回目标不存在")
        history = list(state.get("action_history", []))
        if len(history) >= 2 and all(item.get("action") == decision.action.value and
            item.get("topic_id") == decision.topic_id and item.get("version") == state.get("version", 0) and item.get("query") == decision.query for item in history[-2:]):
            decision = AgentDecision(action="ask", reason="连续行动没有推进，询问学生后再继续",
                question="这里似乎还没有推进。你想换一种解释、做一道题，还是先调整安排？")
        history.append({"action": decision.action.value, "topic_id": decision.topic_id, "reason": decision.reason, "version": state.get("version", 0), "query": decision.query})
        update = {"action_history": history[-100:], "pending_kind": None}
        if decision.action is LearningAction.COMPLETE:
            basis = decision.completion_basis
            required = set(state.get("completion_topic_ids", []))
            # Replanning cannot erase an unfinished objective to claim success.
            verified = bool(required) and required <= set(by_id) and all(
                by_id[topic_id].mastery is MasteryState.READY for topic_id in required)
            valid = ((basis == "time_exhausted" and state.get("remaining_minutes", 0) <= 0)
                or (basis == "verified" and verified)
                or (basis == "user_stopped" and state.get("stop_reason") == "user_stopped"))
            if not valid:
                raise TeachingProviderError("没有足够证据结束学习")
            return {**update, "status": TeachingStatus.COMPLETED.value, "next_action": TeachingAction.COMPLETE.value,
                "stop_reason": basis, "question": None, "teaching_text": decision.reason}
        if decision.action is LearningAction.REPLAN:
            try:
                validate_plan(decision.plan, topics, self.goal.id, self.goal.course_id, state["remaining_minutes"])
            except ValueError as error:
                raise TeachingProviderError(str(error)) from error
            required = set(state.get("completion_topic_ids", []))
            required.update(item.topic_id for item in decision.plan.items if item.tier is not PlanTier.DEFER)
            return {**update, "plan": self._values(decision.plan), "completion_topic_ids": sorted(required)}
        if decision.action is LearningAction.RETRIEVE:
            if self.window_builder is None:
                raise TeachingWorkflowError("资料检索未配置")
            window = self.window_builder.build(course_id=self.goal.course_id,
                query=decision.query or (topic.name if topic else self.goal.id), max_blocks=8, max_chars=8000)
            return {**update, "knowledge_window": self._values(window)}
        if decision.action is LearningAction.ASK:
            return {**update, "pending_kind": "ask", "question": decision.question, "scoring_points": [],
                "question_id": f"ask:{state.get('version', 0)}:{len(history)}", "teaching_text": decision.reason,
                "status": TeachingStatus.WAITING_ANSWER.value, "next_action": TeachingAction.WAITING_ANSWER.value}
        if state.get("plan") is None:
            raise TeachingProviderError("开始教学前需要先生成复习计划")
        if topic is None:
            raise TeachingProviderError("教学行动缺少目标考点")
        # Validate against exactly the evidence supplied to this decision.
        # The agent can retrieve a new window before changing teaching targets.
        window = (KnowledgeWindow.model_validate(state["knowledge_window"])
            if state.get("knowledge_window") else self.knowledge_window)
        allowed = {item.hit.citation.block_id for item in window.items}
        cited = set(decision.citations) | {citation for point in decision.scoring_points for citation in point.citations}
        if not cited <= allowed:
            raise TeachingProviderError("教学行动引用不属于当前证据窗口")
        plan = Plan.model_validate(state["plan"])
        item = next(item for item in plan.items if item.topic_id == topic.id)
        update.update({"knowledge_window": self._values(window), "current_plan_item": self._values(item),
            "mastery_state": topic.mastery.value, "teaching_citations": decision.citations,
            "return_to_topic": decision.return_to_topic or state.get("return_to_topic")})
        if state.get("return_to_topic") == topic.id and decision.return_to_topic is None:
            update["return_to_topic"] = None
        if decision.action is LearningAction.TEACH:
            # Explanations persist across the following independent practice action.
            return {**update, "teaching_text": decision.explanation, "question": None,
                "status": TeachingStatus.TEACHING.value}
        return {**update, "pending_kind": decision.action.value, "question": decision.question,
            "scoring_points": [self._values(point) for point in decision.scoring_points],
            "question_id": f"{topic.id}:attempt-{state.get('attempt', 0) + 1}",
            "attempt": state.get("attempt", 0) + 1,
            "status": TeachingStatus.WAITING_ANSWER.value, "next_action": TeachingAction.WAITING_ANSWER.value}

    @staticmethod
    def _agent_route(state):
        if state.get("status") == TeachingStatus.COMPLETED.value:
            return "complete"
        return "wait" if state.get("pending_kind") else "decide"

    @staticmethod
    def _agent_wait_node(state):
        response = interrupt({"question_id": state.get("question_id"), "question": state.get("question"),
            "kind": state.get("pending_kind"), "version": state.get("version", 0)})
        response = response if isinstance(response, dict) else {"answer": str(response)}
        return {"student_answer": str(response.get("answer", "")), "last_answer_id": response.get("answer_id") or str(uuid4()),
            "spent_minutes": max(0, int(response.get("spent_minutes", 0))), "round_steps": 0}

    def _observe_node(self, state):
        answer = state.get("student_answer", "")
        observations = list(state.get("observations", []))
        command = answer.strip().lower().strip("。.!！?？").strip()
        commands = ("继续", "continue", "停止学习", "结束学习", "stop")
        if state.get("pending_kind") == "ask":
            statements = list(state.get("user_statements", [])) + ([answer] if command not in commands else [])
            update = {"user_statements": statements}
            # User supplied a new fact: refresh analysis instead of losing it in chat history.
            if self.repository and answer.strip() and command not in commands:
                refreshed = self._bootstrap_node({**state, "analysis": {}, "user_statements": statements})
                update.update(refreshed)
        else:
            update = self._evaluating_node(state)
            observations.append({"topic_id": state["current_plan_item"]["topic_id"],
                "kind": state.get("pending_kind"), "answer_id": state.get("last_answer_id"),
                "evaluation": update["evaluation"]})
        spent = state.get("spent_minutes", 0)
        processed = dict(state.get("processed_answers", {}))
        processed[state["last_answer_id"]] = hashlib.sha256(answer.encode()).hexdigest()
        return {"processed_answers": processed, **update, "observations": observations[-100:], "pending_kind": None,
            "remaining_minutes": max(0, state.get("remaining_minutes", 0) - spent),
            "version": state.get("version", 0) + 1,
            "stop_reason": "user_stopped" if command in ("停止学习", "结束学习", "stop") else state.get("stop_reason"),
            "status": TeachingStatus.PLANNING.value}

    def _evaluating_node(self, state: _GraphState) -> dict[str, Any]:
        if self.evaluator is None:
            try:
                self.evaluator = LLMEvaluatorProvider.from_env()
            except Exception as error:
                raise TeachingProviderError("答案评价未配置模型") from error
        self.knowledge_window = KnowledgeWindow.model_validate(state["knowledge_window"]) if state.get("knowledge_window") else self.knowledge_window
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

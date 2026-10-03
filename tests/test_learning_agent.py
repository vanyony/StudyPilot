from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from studypilot.api import create_app
from studypilot.application.learning_agent import (
    AgentDecision, CourseAnalysis, EvidenceClaim, validate_analysis, validate_plan,
)
from studypilot.application.teaching import DeterministicFakeEvaluator, TeachingProviderError, TeachingWorkflow
from studypilot.application.teaching_service import TeachingSessionService
from studypilot.domain.models import Course, EvidenceLevel, ExamGoal, Topic, MasteryState, PlanTier
from studypilot.domain.planner import RevisionPlanner
from studypilot.domain.teaching import ScoringPoint
from studypilot.infrastructure.sqlite_repository import SQLiteRepository


def topic(name, course="course", prerequisites=()):
    return Topic(id=name,course_id=course,name=name,exam_points=10,learning_minutes=10,
        evidence_level=EvidenceLevel.COURSE_MATERIAL,evidence_confidence=.6,prerequisite_ids=prerequisites)


class ScriptedModel:
    """Each test chooses decisions independently of the workflow's implementation."""
    def __init__(self):
        self.contexts=[]
        self.analyses=[]
        self.turn=0

    def analyze(self, context):
        self.analyses.append(context)
        blocks=context["blocks"]
        statements=context["user_statements"]
        return CourseAnalysis(topics=[topic("derivative"),topic("integral",prerequisites=("derivative",))],
            claims=[EvidenceClaim(kind="fact",text="资料包含微积分题",block_ids=[blocks[0]["block_id"]]),
                EvidenceClaim(kind="inference",text="没有往年题，分布为估计"),
                *([EvidenceClaim(kind="fact",text=statements[0]["text"],statement_ids=[statements[0]["id"]])] if statements else [])],
            prerequisite_hypotheses=["求导基础可能影响积分学习，需要诊断"])

    def decide(self, context):
        self.contexts.append(context)
        topics=[Topic.model_validate(value) for value in context["topics"]]
        if context.get("plan") is None:
            goal=ExamGoal.model_validate(context["current_goal"]).model_copy(update={"available_minutes":context["remaining_minutes"]})
            return AgentDecision(action="replan",reason="根据课程证据制定暂定计划",plan=RevisionPlanner().build(goal,topics))
        self.turn+=1
        if self.turn==1:
            return AgentDecision(action="diagnose",reason="先验证积分所需的基础",topic_id="derivative",return_to_topic="integral",
                question="求导的基本规则是什么？",scoring_points=[ScoringPoint(id="rule",description="知道求导规则",evidence=("rule",))])
        if self.turn==2:
            return AgentDecision(action="teach",reason="诊断证明基础有缺口，先补必要部分",topic_id="derivative",
                explanation="这里只补积分所需的求导规则，这是一般学科知识。")
        if self.turn==3:
            return AgentDecision(action="practice",reason="确认补习效果",topic_id="derivative",question="复述求导规则",
                scoring_points=[ScoringPoint(id="rule",description="知道求导规则",evidence=("rule",))])
        return AgentDecision(action="practice",reason="回到原来的积分目标",topic_id="integral",question="如何检查积分结果？",
            scoring_points=[ScoringPoint(id="check",description="检查积分",evidence=("check",))])


def prepare(client):
    client.put("/courses/course",json={"id":"course","name":"微积分"})
    client.put("/courses/course/goals/final",json={"id":"final","course_id":"course","exam_at":"2027-01-01T00:00:00Z","available_minutes":60})
    uploaded=client.post("/courses/course/sources",files={"file":("homework.md","# 积分\n求导与积分的作业题。".encode(),"text/markdown")},data={"document_kind":"ASSIGNMENT","trust_level":"HIGH"}).json()
    client.post(f"/sources/{uploaded['asset']['id']}/parse",json={"parser_kind":"MARKDOWN"})


def test_zero_manual_topics_diagnose_remediate_return_and_restart(tmp_path):
    model=ScriptedModel()
    db=tmp_path/"study.db"
    app=create_app(db,tmp_path/"data",learning_provider=model,evaluator=DeterministicFakeEvaluator())
    with TestClient(app) as client:
        prepare(client)
        response=client.post("/courses/course/sessions",json={"goal_id":"final","session_id":"agent","start":True,"user_statements":["老师说会出作业原题"]})
        assert response.status_code==201, response.text
        state=response.json()
        assert state["pending_kind"]=="diagnose"
        assert state["return_to_topic"]=="integral"
        assert state["analysis"]["covered_block_ids"]
        assert any(claim["kind"]=="inference" for claim in state["analysis"]["claims"])
        assert model.analyses[0]["user_statements"][0]["text"]=="老师说会出作业原题"
        answer={"answer":"不知道","message_id":"a","expected_version":state["version"],"spent_minutes":3}
        repaired=client.post("/sessions/agent/answer",json=answer).json()
        assert repaired["pending_kind"]=="practice"
        assert repaired["teaching_text"]
        assert repaired["mastery_by_topic"]["derivative"]=="GAP"
        assert repaired["remaining_minutes"]==57
        calls=len(model.contexts)
        assert client.post("/sessions/agent/answer",json=answer).json()==repaired
        assert len(model.contexts)==calls
        stale=client.post("/sessions/agent/answer",json={**answer,"message_id":"stale"})
        assert stale.status_code==409
        assert len(model.contexts)==calls
        page=client.get("/study?course_id=course&goal_id=final&session_id=agent")
        assert page.status_code==200
        assert "推测" in page.text
    repository=SQLiteRepository(db)
    service=TeachingSessionService(repository,learning_provider=model,evaluator=DeterministicFakeEvaluator())
    saved=service.get_state("agent")
    assert saved.pending_kind=="practice"
    assert saved.return_to_topic=="integral"
    returned=service.submit_answer("agent",message_id="b",answer="rule",expected_version=saved.version)
    assert returned.current_plan_item.topic_id=="integral"
    assert returned.return_to_topic is None
    assert returned.mastery_by_topic["derivative"] is MasteryState.READY
    service.close()


def test_analysis_rejects_fabricated_evidence_and_mastery():
    analysis=CourseAnalysis(topics=[topic("x")],claims=[EvidenceClaim(kind="fact",text="假事实",block_ids=["invented"])])
    with pytest.raises(ValueError,match="unauthorized"):
        validate_analysis(analysis,"course",{"real"},set())
    analysis.claims=[]
    analysis.topics=[topic("x").model_copy(update={"mastery":MasteryState.READY})]
    with pytest.raises(ValueError,match="mastery"):
        validate_analysis(analysis,"course",set(),set())


def test_plan_rejects_budget_and_missing_prerequisites():
    topics=[topic("a"),topic("b",prerequisites=("a",))]
    goal=ExamGoal(id="g",course_id="course",exam_at=datetime(2027,1,1,tzinfo=UTC),available_minutes=30)
    plan=RevisionPlanner().build(goal,topics)
    with pytest.raises(ValueError,match="totals"):
        validate_plan(plan.model_copy(update={"planned_minutes":100}),topics,"g","course",30)
    items=list(plan.items)
    items[0]=items[0].model_copy(update={"order":2})
    items[1]=items[1].model_copy(update={"order":1})
    with pytest.raises(ValueError,match="prerequisites"):
        validate_plan(plan.model_copy(update={"items":tuple(items)}),topics,"g","course",30)


def test_product_has_no_fake_model_fallback(tmp_path,monkeypatch):
    for key in ("STUDYPILOT_LLM_API_KEY","OPENAI_API_KEY","STUDYPILOT_LLM_MODEL","OPENAI_MODEL"):
        monkeypatch.delenv(key,raising=False)
    goal=ExamGoal(id="g",course_id="course",exam_at=datetime(2027,1,1,tzinfo=UTC),available_minutes=30)
    runtime=TeachingWorkflow(goal,[topic("x")],tmp_path/"checkpoint.db")
    with pytest.raises(TeachingProviderError):
        runtime.start()
    runtime.close()

def test_analysis_batches_cached_after_failure_and_start_retries(tmp_path):
    class FlakyModel(ScriptedModel):
        def __init__(self):
            super().__init__()
            self.fail=True
            self.calls=0
        def analyze(self, context):
            self.calls+=1
            if self.calls==2 and self.fail:
                raise RuntimeError("temporary failure")
            return super().analyze(context)
    model=FlakyModel()
    with TestClient(create_app(tmp_path/"study.db",tmp_path/"data",learning_provider=model,evaluator=DeterministicFakeEvaluator())) as client:
        prepare(client)
        upload=client.post("/courses/course/sources",files={"file":("long.txt",("求导积分资料"*6000).encode(),"text/plain")},data={"document_kind":"COURSE_MATERIAL","trust_level":"MEDIUM"}).json()
        client.post(f"/sources/{upload['asset']['id']}/parse",json={"parser_kind":"TEXT"})
        failed=client.post("/courses/course/sessions",json={"goal_id":"final","session_id":"retry","start":True})
        assert failed.status_code==502
        cached_before=model.calls
        model.fail=False
        restarted=client.post("/sessions/retry/start")
        assert restarted.status_code==200, restarted.text
        state=restarted.json()
        assert state["analysis"]["covered_block_ids"]
        # First successfully analyzed batch was reused rather than requested again.
        assert model.calls > cached_before
        first_block=model.analyses[0]["blocks"][0]
        assert sum(first_block==ctx["blocks"][0] for ctx in model.analyses)==1


def test_invalid_actions_are_not_applied_and_stop_after_two_failures(tmp_path):
    class InvalidModel(ScriptedModel):
        def decide(self, context):
            return AgentDecision(action="teach",reason="bad target",topic_id="nonexistent",explanation="invalid")
    goal=ExamGoal(id="g",course_id="course",exam_at=datetime(2027,1,1,tzinfo=UTC),available_minutes=30)
    runtime=TeachingWorkflow(goal,[topic("x")],tmp_path/"checkpoint.db",learning_provider=InvalidModel())
    state=runtime.start()
    assert state.pending_kind=="ask"
    assert state.tool_errors==2
    assert state.current_plan_item is None
    assert state.last_tool_error
    runtime.close()

def test_teaching_keeps_decision_window_and_rejects_forged_citations(tmp_path):
    from studypilot.domain.knowledge import Citation, DocumentBlock, KnowledgeWindow, KnowledgeWindowItem, SearchHit
    block = DocumentBlock(id="seen", source_asset_id="asset", course_id="course", parser_kind="MARKDOWN",
        parser_version="1", block_index=0, text="换元例题", content_hash="0" * 64)
    cite = Citation(block_id="seen", source_asset_id="asset", course_id="course", display_name="课件",
        page_number=None, section=None, block_index=0, quote="换元例题")
    window = KnowledgeWindow(query="详细换元检索", course_id="course", max_blocks=8, max_chars=8000,
        used_chars=4, items=(KnowledgeWindowItem(hit=SearchHit(block=block, score=1, citation=cite)),))
    class ChangingRetriever:
        def build(self, **kwargs):
            raise AssertionError("Execution must not replace the model's evidence window")
    goal = ExamGoal(id="g", course_id="course", exam_at=datetime(2027,1,1,tzinfo=UTC), available_minutes=30)
    topics = [topic("integral")]
    runtime = TeachingWorkflow(goal, topics, tmp_path/"checkpoint.db", knowledge_window_builder=ChangingRetriever())
    try:
        state = runtime._initial_state()
        state.update(plan=RevisionPlanner().build(goal,topics).model_dump(mode="json"),
            knowledge_window=window.model_dump(mode="json"),
            agent_decision=AgentDecision(action="teach", reason="基于已检索资料", topic_id="integral",
                explanation="换元讲解", citations=["seen"]).model_dump(mode="json"))
        result = runtime._execute_action(state)
        assert result["teaching_citations"] == ["seen"]
        assert result["knowledge_window"] == window.model_dump(mode="json")
        state["agent_decision"]["citations"] = ["forged"]
        with pytest.raises(TeachingProviderError, match="引用"):
            runtime._execute_action(state)
    finally:
        runtime.close()


@pytest.mark.parametrize("mastered", [True, False])
def test_scope_completion_ignores_deferred_but_cannot_drop_unfinished_goals(tmp_path, mastered):
    goal = ExamGoal(id="g", course_id="course", exam_at=datetime(2027,1,1,tzinfo=UTC), available_minutes=20)
    topics = [topic("a"), topic("b"), topic("optional").model_copy(update={"exam_points":0})]
    runtime = TeachingWorkflow(goal,topics,tmp_path/"checkpoint.db")
    try:
        state = runtime._initial_state()
        first = RevisionPlanner().build(goal,topics)
        first = first.model_copy(update={"items":tuple(
            item.model_copy(update={"tier":PlanTier.DEFER}) if item.topic_id=="optional" else item
            for item in first.items)})
        assert {item.topic_id for item in first.items if item.tier.value=="MUST"} == {"a","b"}
        state["agent_decision"] = AgentDecision(action="replan", reason="选择本轮重点", plan=first).model_dump(mode="json")
        state.update(runtime._execute_action(state))
        assert set(state["completion_topic_ids"]) == {"a","b"}
        state["mastery_by_topic"].update(a="READY", b="READY" if mastered else "UNSEEN")
        deferred = first.model_copy(update={"planned_minutes":0, "items":tuple(
            item.model_copy(update={"tier":PlanTier.DEFER, "order":None, "estimated_minutes":0}) for item in first.items)})
        state["agent_decision"] = AgentDecision(action="replan", reason="重排", plan=deferred).model_dump(mode="json")
        state.update(runtime._execute_action(state))
        state["agent_decision"] = AgentDecision(action="complete", reason="本轮重点已完成", completion_basis="verified").model_dump(mode="json")
        if mastered:
            assert runtime._execute_action(state)["status"] == "COMPLETED"
        else:
            with pytest.raises(TeachingProviderError,match="证据"):
                runtime._execute_action(state)
    finally:
        runtime.close()


@pytest.mark.parametrize("answer, stopped", [("继续。",False),("Continue!",False),("停止学习！",True)])
def test_control_reply_punctuation_does_not_trigger_analysis(tmp_path, answer, stopped):
    goal = ExamGoal(id="g", course_id="course", exam_at=datetime(2027,1,1,tzinfo=UTC), available_minutes=20)
    runtime = TeachingWorkflow(goal,[topic("a")],tmp_path/"checkpoint.db",repository=object())
    try:
        runtime._bootstrap_node = lambda state: pytest.fail("Control reply must not reanalyze materials")
        state=runtime._initial_state()
        state.update(pending_kind="ask", student_answer=answer, last_answer_id="reply", spent_minutes=0)
        result=runtime._observe_node(state)
        assert result["user_statements"] == []
        assert (result["stop_reason"] == "user_stopped") == stopped
    finally:
        runtime.close()

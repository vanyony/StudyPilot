"""Explicit offline provider for migrated session regression tests."""
from studypilot.application.learning_agent import AgentDecision, CourseAnalysis, EvidenceClaim
from studypilot.domain.models import Topic, ExamGoal, MasteryState
from studypilot.domain.planner import RevisionPlanner
from studypilot.domain.knowledge import KnowledgeWindow
from studypilot.application.teaching import DeterministicTeacherProvider


class RegressionLearningProvider:
    def __init__(self, teacher=None):
        self.teacher = teacher or DeterministicTeacherProvider()

    def analyze(self, context):
        from studypilot.domain.models import EvidenceLevel
        return CourseAnalysis(topics=[Topic(id="bayes", course_id=context["course_id"], name="贝叶斯公式",
            exam_points=10, learning_minutes=10, evidence_level=EvidenceLevel.COURSE_MATERIAL,evidence_confidence=.6)],
            claims=[EvidenceClaim(kind="inference",text="离线测试模型生成的候选考点")])

    def decide(self, context):
        topics = [Topic.model_validate(item).model_copy(update={"mastery": MasteryState(context.get("mastery_by_topic", {}).get(item["id"], "UNSEEN"))}) for item in context["topics"]]
        goal = ExamGoal.model_validate(context["current_goal"]).model_copy(update={"available_minutes": context["remaining_minutes"]})
        plan = RevisionPlanner().build(goal, topics)
        if all(topic.mastery is MasteryState.READY for topic in topics) and topics:
            return AgentDecision(action="complete", reason="已根据作答证据完成", completion_basis="verified")
        if goal.available_minutes == 0:
            return AgentDecision(action="complete", reason="可用时间耗尽", completion_basis="time_exhausted")
        if context.get("plan") != plan.model_dump(mode="json"):
            return AgentDecision(action="replan", reason="按新的掌握证据更新安排", plan=plan)
        target = next((item for item in plan.items if item.tier.value == "MUST"), None)
        if target is None:
            return AgentDecision(action="ask", reason="预算不足", question="要调整复习预算吗？")
        topic = next(item for item in topics if item.id == target.topic_id)
        window = KnowledgeWindow.model_validate(context["knowledge_window"]) if context.get("knowledge_window") else KnowledgeWindow(query=topic.name,course_id=topic.course_id,max_blocks=5,max_chars=4000,used_chars=0,items=())
        content = self.teacher.teach(topic=topic, knowledge_window=window, attempt=context.get("attempt",0)+1,
            question_override=context.get("question_override"),
            scoring_points_override=context.get("scoring_points_override"),
            mastery_state=topic.mastery,remaining_minutes=goal.available_minutes)
        from studypilot.domain.teaching import TeachingContent
        content = TeachingContent.model_validate(content)
        return AgentDecision(action="practice", reason="验证当前考点的评分证据", topic_id=topic.id,
            question=content.question, scoring_points=list(content.scoring_points), citations=list(content.citations))
from __future__ import annotations

from dataclasses import dataclass

from studypilot.domain.models import (
    EvidenceLevel,
    ExamGoal,
    MasteryState,
    Plan,
    PlanItem,
    PlanTier,
    Topic,
)


_MASTERY_GAP = {
    MasteryState.READY: 0.0,
    MasteryState.FRAGILE: 0.45,
    MasteryState.GAP: 0.85,
    MasteryState.UNSEEN: 1.0,
}

_EVIDENCE_WEIGHT = {
    EvidenceLevel.TEACHER_SCOPE: 1.20,
    EvidenceLevel.PAST_EXAM: 1.10,
    EvidenceLevel.COURSE_MATERIAL: 1.00,
    EvidenceLevel.ASSIGNMENT: 0.90,
    EvidenceLevel.PEER_NOTE: 0.70,
}

_EVIDENCE_LABEL = {
    EvidenceLevel.TEACHER_SCOPE: "老师明确范围",
    EvidenceLevel.PAST_EXAM: "往年题证据",
    EvidenceLevel.COURSE_MATERIAL: "课程资料证据",
    EvidenceLevel.ASSIGNMENT: "作业或练习证据",
    EvidenceLevel.PEER_NOTE: "非官方资料，置信度较低",
}


class PlanningError(ValueError):
    """Raised when planner input is internally inconsistent."""


@dataclass(frozen=True)
class _Candidate:
    topic: Topic
    bundle: tuple[Topic, ...]
    marginal_minutes: int
    benefit: float

    @property
    def ratio(self) -> float:
        return self.benefit / self.marginal_minutes


class RevisionPlanner:
    """Deterministic greedy planner optimized for explainability.

    Each selection considers the target topic and every unmet prerequisite as one
    bundle. This prevents a high-value advanced topic from being scheduled without
    the foundation needed to learn it.
    """

    def build(self, goal: ExamGoal, topics: list[Topic]) -> Plan:
        if not topics:
            return Plan(
                goal_id=goal.id,
                course_id=goal.course_id,
                budget_minutes=goal.available_minutes,
                planned_minutes=0,
                items=(),
            )

        by_id = {topic.id: topic for topic in topics}
        if len(by_id) != len(topics):
            raise PlanningError("topic ids must be unique")
        if any(topic.course_id != goal.course_id for topic in topics):
            raise PlanningError("all topics must belong to the goal course")
        self._validate_prerequisites(by_id)

        selected: set[str] = set()
        must_order: list[str] = []
        remaining = goal.available_minutes

        while remaining > 0:
            candidates = self._candidates(by_id, selected)
            fitting = [item for item in candidates if item.marginal_minutes <= remaining]
            if not fitting:
                break
            chosen = max(
                fitting,
                key=lambda item: (
                    item.ratio,
                    item.benefit,
                    -item.marginal_minutes,
                    item.topic.id,
                ),
            )
            for bundled_topic in chosen.bundle:
                if bundled_topic.id not in selected:
                    selected.add(bundled_topic.id)
                    must_order.append(bundled_topic.id)
                    remaining -= bundled_topic.learning_minutes

        unscheduled = {
            topic_id
            for topic_id, topic in by_id.items()
            if topic_id not in selected and topic.mastery is not MasteryState.READY
        }
        strive_ids = self._select_strive(by_id, selected, unscheduled, goal.available_minutes)

        order_by_id = {topic_id: index + 1 for index, topic_id in enumerate(must_order)}
        items: list[PlanItem] = []
        for topic in topics:
            score = self._topic_benefit(topic) / topic.learning_minutes
            if topic.id in selected:
                tier = PlanTier.MUST
                reasons = self._reasons(topic, tier, by_id, selected)
                order = order_by_id[topic.id]
            elif topic.id in strive_ids:
                tier = PlanTier.STRIVE
                reasons = self._reasons(topic, tier, by_id, selected)
                order = None
            else:
                tier = PlanTier.DEFER
                reasons = self._reasons(topic, tier, by_id, selected)
                order = None
            items.append(
                PlanItem(
                    topic_id=topic.id,
                    topic_name=topic.name,
                    tier=tier,
                    order=order,
                    estimated_minutes=topic.learning_minutes,
                    utility_score=round(score, 4),
                    reasons=reasons,
                )
            )

        items.sort(
            key=lambda item: (
                {PlanTier.MUST: 0, PlanTier.STRIVE: 1, PlanTier.DEFER: 2}[item.tier],
                item.order if item.order is not None else 10**9,
                -item.utility_score,
                item.topic_id,
            )
        )
        planned_minutes = sum(by_id[topic_id].learning_minutes for topic_id in selected)
        return Plan(
            goal_id=goal.id,
            course_id=goal.course_id,
            budget_minutes=goal.available_minutes,
            planned_minutes=planned_minutes,
            items=tuple(items),
        )

    def _candidates(self, by_id: dict[str, Topic], selected: set[str]) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for topic in by_id.values():
            if topic.id in selected or topic.mastery is MasteryState.READY:
                continue
            bundle = tuple(self._unmet_chain(topic, by_id, selected))
            marginal_minutes = sum(item.learning_minutes for item in bundle)
            benefit = sum(self._topic_benefit(item) for item in bundle)
            candidates.append(_Candidate(topic, bundle, marginal_minutes, benefit))
        return candidates

    def _unmet_chain(
        self, topic: Topic, by_id: dict[str, Topic], selected: set[str]
    ) -> list[Topic]:
        result: list[Topic] = []
        seen: set[str] = set()

        def visit(current: Topic) -> None:
            if current.id in seen or current.id in selected:
                return
            seen.add(current.id)
            for prerequisite_id in current.prerequisite_ids:
                prerequisite = by_id[prerequisite_id]
                if prerequisite.mastery is not MasteryState.READY:
                    visit(prerequisite)
            result.append(current)

        visit(topic)
        return result

    def _select_strive(
        self,
        by_id: dict[str, Topic],
        selected: set[str],
        unscheduled: set[str],
        original_budget: int,
    ) -> set[str]:
        # "争取" is a short overflow list, not a second full plan.
        stretch_budget = max(15, original_budget // 3)
        result: set[str] = set()
        while stretch_budget > 0 and unscheduled:
            candidates = [
                candidate
                for candidate in self._candidates(by_id, selected | result)
                if candidate.topic.id in unscheduled
                and candidate.marginal_minutes <= stretch_budget
            ]
            if not candidates:
                break
            chosen = max(
                candidates,
                key=lambda item: (
                    item.ratio,
                    item.benefit,
                    -item.marginal_minutes,
                    item.topic.id,
                ),
            )
            for bundled_topic in chosen.bundle:
                if bundled_topic.id in unscheduled:
                    result.add(bundled_topic.id)
                    unscheduled.remove(bundled_topic.id)
                    stretch_budget -= bundled_topic.learning_minutes
        return result

    @staticmethod
    def _topic_benefit(topic: Topic) -> float:
        return (
            topic.exam_points
            * _MASTERY_GAP[topic.mastery]
            * topic.evidence_confidence
            * _EVIDENCE_WEIGHT[topic.evidence_level]
            * (0.75 + 0.5 * topic.frequency)
        )

    @staticmethod
    def _validate_prerequisites(by_id: dict[str, Topic]) -> None:
        for topic in by_id.values():
            missing = set(topic.prerequisite_ids) - by_id.keys()
            if missing:
                raise PlanningError(
                    f"topic {topic.id!r} references missing prerequisites: {sorted(missing)}"
                )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(topic_id: str) -> None:
            if topic_id in visiting:
                raise PlanningError("prerequisite graph contains a cycle")
            if topic_id in visited:
                return
            visiting.add(topic_id)
            for prerequisite_id in by_id[topic_id].prerequisite_ids:
                visit(prerequisite_id)
            visiting.remove(topic_id)
            visited.add(topic_id)

        for topic_id in by_id:
            visit(topic_id)

    @staticmethod
    def _reasons(
        topic: Topic,
        tier: PlanTier,
        by_id: dict[str, Topic],
        selected: set[str],
    ) -> tuple[str, ...]:
        reasons = [
            f"预计分值 {topic.exam_points:g}，学习成本 {topic.learning_minutes} 分钟",
            f"{_EVIDENCE_LABEL[topic.evidence_level]}，置信度 {topic.evidence_confidence:.0%}",
            f"当前掌握状态为 {topic.mastery.value}",
        ]
        unmet = [
            by_id[item_id].name
            for item_id in topic.prerequisite_ids
            if by_id[item_id].mastery is not MasteryState.READY
            and item_id not in selected
        ]
        if tier is PlanTier.MUST and topic.prerequisite_ids:
            reasons.append("按前置关系排在依赖考点之前")
        elif tier is PlanTier.STRIVE:
            reasons.append("核心预算不足，进入短时追加清单")
        elif topic.mastery is MasteryState.READY:
            reasons.append("已达到 READY，本轮跳过以节省时间")
        elif unmet:
            reasons.append(f"仍需先补前置知识：{'、'.join(unmet)}")
        else:
            reasons.append("当前预算内收益低于已选考点，暂缓")
        return tuple(reasons)


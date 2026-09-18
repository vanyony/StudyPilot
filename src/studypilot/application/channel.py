"""Application boundary shared by local and chat-channel entry points."""

from __future__ import annotations

from datetime import UTC, datetime
from threading import RLock

from studypilot.application.teaching_service import (
    MessageIdConflict,
    SessionProviderError,
    SessionStateConflict,
    SessionVersionConflict,
    TeachingSessionService,
)
from studypilot.domain.channel import (
    CanonicalMessage,
    ChannelAdapter,
    ChannelBinding,
    ChannelReply,
)
from studypilot.domain.teaching import TeachingSessionState, TeachingStatus
from studypilot.infrastructure.sqlite_repository import NotFoundError, SQLiteRepository


class ChannelServiceError(RuntimeError):
    """Base error for a channel boundary misuse."""


class ChannelBindingError(ChannelServiceError, ValueError):
    """A requested user/course/session binding is invalid."""


class ChannelService:
    """Route canonical messages to the existing teaching application service.

    This class intentionally contains no platform protocol code.  It reads and
    writes the same SQLite-backed session/checkpoint state used by the PC API,
    and the teaching service remains the authority for optimistic versions,
    idempotent receipts and rubric-constrained state transitions.
    """

    def __init__(
        self,
        repository: SQLiteRepository,
        teaching_service: TeachingSessionService,
        *,
        channel: str = "qq_c2c",
    ) -> None:
        self.repository = repository
        self.teaching_service = teaching_service
        self.channel = channel
        self._locks: dict[str, RLock] = {}
        self._locks_guard = RLock()

    def bind_user(
        self,
        *,
        external_user_id: str | None = None,
        openid: str | None = None,
        learner_id: str,
        course_id: str,
        session_id: str | None = None,
        study_session_id: str | None = None,
        channel: str | None = None,
    ) -> ChannelBinding:
        """Persist an operator-created user → course/session binding.

        Binding is deliberately not inferred from an incoming chat message.
        That keeps an unknown QQ user from selecting another learner's course
        or session.  ``study_session_id`` is accepted as a readable alias for
        callers that use the product terminology.
        """

        if external_user_id is not None and openid is not None and external_user_id != openid:
            raise ChannelBindingError("external_user_id and openid must match")
        resolved_external = external_user_id if external_user_id is not None else openid
        if not resolved_external:
            raise ChannelBindingError("external_user_id (openid) is required")
        if session_id is not None and study_session_id is not None and session_id != study_session_id:
            raise ChannelBindingError("session_id and study_session_id must match")
        resolved_session = session_id if session_id is not None else study_session_id
        resolved_channel = channel or self.channel
        existing = self.repository.get_channel_binding(resolved_channel, resolved_external)
        now = datetime.now(UTC)
        binding = ChannelBinding(
            channel=resolved_channel,
            external_user_id=resolved_external,
            learner_id=learner_id,
            course_id=course_id,
            session_id=resolved_session,
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
        )
        try:
            self.repository.save_channel_binding(binding)
        except (NotFoundError, ValueError) as error:
            raise ChannelBindingError(str(error)) from error
        return binding

    # Short aliases make the boundary convenient for a local operator script.
    bind = bind_user

    def unbind_user(self, external_user_id: str, channel: str | None = None) -> bool:
        return self.repository.delete_channel_binding(channel or self.channel, external_user_id)

    unbind = unbind_user

    def get_binding(
        self, external_user_id: str, channel: str | None = None
    ) -> ChannelBinding | None:
        return self.repository.get_channel_binding(channel or self.channel, external_user_id)

    def handle_message(self, message: CanonicalMessage) -> ChannelReply:
        """Process one canonical message and return a renderable reply.

        A message is submitted only while its bound session is waiting.  A
        repeated message id is still passed to ``TeachingSessionService`` so
        its durable receipt can return the original result without rescoring.
        """

        if message.channel != self.channel:
            raise ChannelServiceError(
                f"message channel {message.channel!r} is not supported by {self.channel!r}"
            )
        binding = self.get_binding(message.external_user_id)
        if binding is None:
            return self._reply(
                message,
                "这个 QQ 用户尚未绑定 StudyPilot 学习者。请先在本地管理入口绑定 "
                "learner_id、course_id 和当前教学 Session，再发送答案。",
                processed=False,
            )
        if binding.session_id is None:
            return self._reply(
                message,
                "已绑定学习者和课程，但没有当前教学 Session。请先在 PC 学习入口创建并启动 "
                "Session，再绑定 session_id。",
                processed=False,
                session_id=None,
            )

        lock = self._session_lock(binding.session_id)
        with lock:
            try:
                state = self.teaching_service.get_state(binding.session_id)
            except NotFoundError:
                return self._reply(
                    message,
                    "绑定的教学 Session 不存在，答案没有处理。请在 PC 学习入口重新创建 Session "
                    "并更新绑定。",
                    processed=False,
                    session_id=binding.session_id,
                )
            except SessionStateConflict as error:
                return self._reply(
                    message,
                    f"当前教学 Session 无法读取（{error}），答案没有处理，请稍后重试。",
                    processed=False,
                    retryable=True,
                    session_id=binding.session_id,
                )

            receipt = self.repository.get_answer_receipt(
                binding.session_id, message.message_id
            )
            if receipt is not None or state.status == TeachingStatus.WAITING_ANSWER:
                return self._submit_answer(message, state)
            return self._non_waiting_reply(message, state)

    process_message = handle_message
    route = handle_message

    def _submit_answer(
        self, message: CanonicalMessage, state: TeachingSessionState
    ) -> ChannelReply:
        try:
            next_state = self.teaching_service.submit_answer(
                state.session_id,
                message_id=message.message_id,
                answer=message.text,
                expected_version=state.version,
            )
        except MessageIdConflict:
            return self._reply(
                message,
                "这个 QQ 消息 ID 已经用过，但答案内容不同；为避免重复评分，本次没有处理。",
                processed=False,
                session_id=state.session_id,
                state_version=state.version,
            )
        except SessionVersionConflict:
            try:
                current = self.teaching_service.get_state(state.session_id)
            except Exception:
                current = state
            return self._reply(
                message,
                "这个教学 Session 已在其他端更新，答案没有处理。请先查看 PC 当前状态，再发送新的答案。",
                processed=False,
                retryable=True,
                session_id=state.session_id,
                state_version=current.version,
            )
        except SessionStateConflict as error:
            return self._reply(
                message,
                f"当前不在等待答案状态（{error}），答案没有处理。请按当前教学提示继续。",
                processed=False,
                session_id=state.session_id,
                state_version=state.version,
            )
        except SessionProviderError:
            return self._reply(
                message,
                "教学 provider 暂时不可用，答案没有确认处理，请稍后重试。",
                processed=False,
                retryable=True,
                session_id=state.session_id,
                state_version=state.version,
            )
        except (NotFoundError, ValueError):
            return self._reply(
                message,
                "答案处理失败，未确认写入学习状态，请稍后重试。",
                processed=False,
                retryable=True,
                session_id=state.session_id,
                state_version=state.version,
            )
        return self._answer_reply(message, state, next_state)

    def _non_waiting_reply(
        self, message: CanonicalMessage, state: TeachingSessionState
    ) -> ChannelReply:
        if state.status == TeachingStatus.COMPLETED:
            text = "本轮教学已完成；当前没有可提交的答案。请在 PC 学习入口查看完整评价和计划。"
        elif state.status == TeachingStatus.CREATED:
            text = "这个教学 Session 尚未启动。请先在 PC 学习入口启动 Session，再发送答案。"
        else:
            text = (
                f"当前教学状态为 {state.status.value}，暂时不接收答案；"
                f"下一动作是 {state.next_action.value}，请稍后重试。"
            )
        return self._reply(
            message,
            text,
            processed=False,
            session_id=state.session_id,
            state_version=state.version,
        )

    def _answer_reply(
        self,
        message: CanonicalMessage,
        previous: TeachingSessionState,
        state: TeachingSessionState,
    ) -> ChannelReply:
        evaluation = state.evaluation
        lines = ["StudyPilot 评分结果"]
        if evaluation is not None:
            lines.append(
                f"得分 {evaluation.score:.0%}｜掌握状态 {evaluation.mastery_state.value}"
            )
            lines.append(evaluation.reason)
        if state.status == TeachingStatus.COMPLETED:
            lines.append("本轮教学已完成。")
        elif state.current_plan_item is not None:
            lines.append(
                f"下一考点：{state.current_plan_item.topic_name}；"
                f"剩余 {state.remaining_minutes} 分钟。"
            )
            if state.question:
                lines.append(f"下一题：{state.question}")
        else:
            lines.append(f"下一动作：{state.next_action.value}")
        lines.extend(_citation_summaries(state, previous, self.repository))
        return self._reply(
            message,
            "\n".join(lines),
            processed=True,
            session_id=state.session_id,
            state_version=state.version,
        )

    def _reply(
        self,
        message: CanonicalMessage,
        text: str,
        *,
        processed: bool,
        retryable: bool = False,
        session_id: str | None = None,
        state_version: int | None = None,
    ) -> ChannelReply:
        return ChannelReply(
            channel=message.channel,
            external_user_id=message.external_user_id,
            text=text,
            reply_to_message_id=message.message_id,
            session_id=session_id,
            state_version=state_version,
            processed=processed,
            retryable=retryable,
        )

    def _session_lock(self, session_id: str) -> RLock:
        with self._locks_guard:
            return self._locks.setdefault(session_id, RLock())


def _citation_summaries(
    state: TeachingSessionState,
    previous: TeachingSessionState,
    repository: SQLiteRepository,
) -> list[str]:
    """Render only cited blocks, retaining the previous window after replan."""

    window = state.knowledge_window or previous.knowledge_window
    ids: list[str] = []
    if state.evaluation is not None:
        ids.extend(state.evaluation.citations)
        for item in state.evaluation.point_evaluations:
            ids.extend(item.citations)
    ids.extend(state.teaching_citations)
    ids.extend(previous.teaching_citations)
    for point in previous.scoring_points:
        ids.extend(point.citations)
    requested = set(ids)
    summaries: list[str] = []
    citations = (
        [item.hit.citation for item in window.items]
        if window is not None
        else []
    )
    known_ids = {citation.block_id for citation in citations}
    for block_id in requested - known_ids:
        try:
            citations.append(repository.citation_for_block(block_id))
        except NotFoundError:
            continue
    for citation in citations:
        if requested and citation.block_id not in requested:
            continue
        if not requested:
            # A provider did not cite a block; do not turn the whole window
            # into an unsolicited channel reply.
            continue
        quote = " ".join(citation.quote.split())
        if len(quote) > 160:
            quote = quote[:157].rstrip() + "..."
        location = citation.display_name
        if citation.section:
            location += f" / {citation.section}"
        if citation.page_number is not None:
            location += f" / p.{citation.page_number}"
        summaries.append(f"引用：{location}：{quote}")
        if len(summaries) >= 2:
            break
    return summaries


__all__ = [
    "CanonicalMessage",
    "ChannelAdapter",
    "ChannelBinding",
    "ChannelReply",
    "ChannelService",
    "ChannelServiceError",
    "ChannelBindingError",
]

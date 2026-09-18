from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from studypilot.api import create_app
from studypilot.application.channel import ChannelService
from studypilot.application.qq import (
    QQBotConfig,
    QQC2CAdapter,
    QQDeliveryError,
    QQOfficialSDKTransport,
    truncate_qq_text,
)
from studypilot.application.teaching import DeterministicFakeEvaluator
from studypilot.application.teaching_service import TeachingSessionService
from studypilot.domain.channel import CanonicalMessage
from studypilot.domain.models import Course, EvidenceLevel, ExamGoal, Topic
from studypilot.domain.teaching import ScoringPoint
from studypilot.infrastructure.sqlite_repository import SQLiteRepository


class CountingEvaluator(DeterministicFakeEvaluator):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def evaluate(self, *args, **kwargs):
        self.calls += 1
        return super().evaluate(*args, **kwargs)


class FakeQQSDK:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.sent: list[dict[str, object]] = []

    async def send_c2c_text(
        self, openid: str, text: str, *, reply_to: str | None = None
    ) -> dict[str, str]:
        if self.failures:
            self.failures -= 1
            raise RuntimeError("transport unavailable")
        self.sent.append({"openid": openid, "text": text, "reply_to": reply_to})
        return {"id": "outbound"}


def _prepare(tmp_path, *, evaluator=None):
    repository = SQLiteRepository(tmp_path / "study.db")
    repository.initialize()
    repository.save_course(Course(id="course", name="概率论"))
    repository.save_topics(
        "course",
        [
            Topic(
                id="bayes",
                course_id="course",
                name="贝叶斯公式",
                exam_points=10,
                learning_minutes=10,
                evidence_level=EvidenceLevel.PAST_EXAM,
                evidence_confidence=1,
                frequency=0.8,
            )
        ],
    )
    repository.save_goal(
        ExamGoal(
            id="goal",
            course_id="course",
            exam_at=datetime(2026, 12, 20, 9, tzinfo=UTC),
            available_minutes=30,
        )
    )
    teaching = TeachingSessionService(repository, evaluator=evaluator)
    return repository, teaching


def _message(message_id: str = "qq-message", *, openid: str = "openid-1", text: str = "correct"):
    return CanonicalMessage(
        channel="qq_c2c",
        external_user_id=openid,
        message_id=message_id,
        text=text,
        timestamp="2026-09-16T12:00:00+08:00",
    )


def _event(message_id: str = "qq-message", *, openid: str = "openid-1", text: str = "correct"):
    return {
        "id": message_id,
        "author": {"user_openid": openid},
        "content": text,
        "message_type": 0,
        "timestamp": "2026-09-16T12:00:00+08:00",
    }


def _adapter(repository, teaching, sdk=None, *, max_length: int = 4000):
    channel = ChannelService(repository, teaching)
    channel.bind_user(
        external_user_id="openid-1",
        learner_id="learner-1",
        course_id="course",
        session_id="session-1",
    )
    adapter = QQC2CAdapter(
        channel,
        sdk=sdk or FakeQQSDK(),
        config=QQBotConfig(
            app_id="app-id",
            app_secret="test-secret",
            max_message_length=max_length,
        ),
    )
    return channel, adapter


def test_canonical_message_and_qq_c2c_normalization_are_narrow() -> None:
    service = object.__new__(ChannelService)
    adapter = QQC2CAdapter(
        service,
        sdk=FakeQQSDK(),
        config=QQBotConfig(app_id="app-id", app_secret="test-secret"),
    )

    normalized = adapter.normalize_event("C2C_MESSAGE_CREATE", _event())
    assert normalized is not None
    assert normalized.channel == "qq_c2c"
    assert normalized.external_user_id == "openid-1"
    assert normalized.message_id == "qq-message"
    assert normalized.text == "correct"
    assert adapter.normalize_event("GROUP_AT_MESSAGE_CREATE", _event()) is None
    assert adapter.normalize_event(
        "C2C_MESSAGE_CREATE", {**_event(), "message_type": 7}
    ) is None
    assert adapter.normalize_event("C2C_MESSAGE_CREATE", {"id": "m"}) is None


def test_unbound_and_partially_bound_users_get_explicit_replies(tmp_path) -> None:
    repository, teaching = _prepare(tmp_path)
    service = ChannelService(repository, teaching)

    unbound = service.handle_message(_message())
    assert "尚未绑定" in unbound.text
    assert unbound.processed is False

    service.bind_user(
        external_user_id="openid-1",
        learner_id="learner-1",
        course_id="course",
    )
    missing_session = service.handle_message(_message())
    assert "没有当前教学 Session" in missing_session.text
    assert missing_session.processed is False


def test_waiting_answer_is_replied_and_duplicate_message_is_idempotent(tmp_path) -> None:
    import asyncio

    evaluator = CountingEvaluator()
    repository, teaching = _prepare(tmp_path, evaluator=evaluator)
    teaching.create_session(course_id="course", goal_id="goal", session_id="session-1", start=True)
    _, adapter = _adapter(repository, teaching)
    sdk = adapter.sdk

    first = asyncio.run(adapter.handle_event("C2C_MESSAGE_CREATE", _event()))
    assert first is not None
    assert "READY" in first.text
    assert "完成" in first.text
    assert first.processed is True
    assert evaluator.calls == 1
    assert len(sdk.sent) == 1

    duplicate = asyncio.run(adapter.handle_event("C2C_MESSAGE_CREATE", _event()))
    assert duplicate is not None
    assert duplicate.text == first.text
    assert evaluator.calls == 1
    assert len(sdk.sent) == 2


def test_cross_platform_pc_session_can_be_answered_from_qq(tmp_path) -> None:
    import asyncio

    repository, teaching = _prepare(tmp_path)
    state = teaching.create_session(
        course_id="course",
        goal_id="goal",
        session_id="session-1",
        start=True,
        scoring_points=[
            ScoringPoint(id="p1", description="第一评分点", evidence=("a",)),
            ScoringPoint(id="p2", description="第二评分点", evidence=("b",)),
        ],
    )
    channel, adapter = _adapter(repository, teaching)
    reply = asyncio.run(adapter.handle_event("C2C_MESSAGE_CREATE", _event(text="partial")))
    assert reply is not None
    assert "FRAGILE" in reply.text

    rebuilt_repo = SQLiteRepository(tmp_path / "study.db")
    rebuilt_repo.initialize()
    rebuilt_teaching = TeachingSessionService(rebuilt_repo)
    assert rebuilt_teaching.get_state("session-1").mastery_state.value == "FRAGILE"
    binding = rebuilt_repo.get_channel_binding("qq_c2c", "openid-1")
    assert binding is not None
    assert binding.learner_id == "learner-1"
    assert binding.course_id == "course"
    assert binding.session_id == "session-1"
    assert state.version < rebuilt_teaching.get_state("session-1").version
    del channel


def test_non_waiting_session_returns_current_actionable_prompt(tmp_path) -> None:
    repository, teaching = _prepare(tmp_path)
    teaching.create_session(course_id="course", goal_id="goal", session_id="session-1")
    channel, _ = _adapter(repository, teaching)
    reply = channel.handle_message(_message())
    assert reply.processed is False
    assert "尚未启动" in reply.text


def test_send_failure_is_explicit_and_redelivery_uses_persisted_receipt(tmp_path) -> None:
    import asyncio

    evaluator = CountingEvaluator()
    repository, teaching = _prepare(tmp_path, evaluator=evaluator)
    teaching.create_session(course_id="course", goal_id="goal", session_id="session-1", start=True)
    sdk = FakeQQSDK(failures=1)
    _, adapter = _adapter(repository, teaching, sdk=sdk)

    with pytest.raises(QQDeliveryError) as error:
        asyncio.run(adapter.handle_event("C2C_MESSAGE_CREATE", _event()))
    assert error.value.retryable is True
    assert "test-secret" not in str(error.value)
    assert repository.get_answer_receipt("session-1", "qq-message") is not None
    assert evaluator.calls == 1

    retried = asyncio.run(adapter.handle_event("C2C_MESSAGE_CREATE", _event()))
    assert retried is not None
    assert evaluator.calls == 1
    assert len(sdk.sent) == 1


def test_bindings_are_persistent_and_can_be_removed(tmp_path) -> None:
    repository, teaching = _prepare(tmp_path)
    service = ChannelService(repository, teaching)
    binding = service.bind_user(
        external_user_id="openid-1",
        learner_id="learner-1",
        course_id="course",
    )
    assert binding.session_id is None
    reopened = SQLiteRepository(tmp_path / "study.db")
    reopened.initialize()
    assert reopened.get_channel_binding("qq_c2c", "openid-1").learner_id == "learner-1"
    assert service.unbind_user("openid-1") is True
    assert reopened.get_channel_binding("qq_c2c", "openid-1") is None


def test_qq_config_redacts_credentials_and_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("STUDYPILOT_QQ_APP_ID", "env-app")
    monkeypatch.setenv("STUDYPILOT_QQ_APP_SECRET", "env-secret")
    monkeypatch.setenv("STUDYPILOT_QQ_MAX_MESSAGE_LENGTH", "123")
    config = QQBotConfig.from_env()
    assert config.app_id == "env-app"
    assert config.secret == "env-secret"
    assert config.max_message_length == 123
    assert "env-secret" not in repr(config)
    assert "env-secret" not in str(config)
    assert "env-secret" not in str(config.public_dict())


def test_official_sdk_transport_injects_and_closes_http_client(monkeypatch) -> None:
    import asyncio
    import sys
    import types

    class FakeAPI:
        def __init__(self, app_id: str, client_secret: str) -> None:
            self.app_id = app_id
            self.client_secret = client_secret
            self.http_client = None

        def setup(self, http_client) -> None:
            self.http_client = http_client

    class FakeWebSocket:
        pass

    class FakeCallbacks:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    sdk = types.ModuleType("qqbot_agent_sdk")
    sdk.__path__ = []
    sdk.QQApiClient = FakeAPI
    sdk.QQWebSocket = FakeWebSocket
    sdk.WSCallbacks = FakeCallbacks
    sdk_api_client = types.ModuleType("qqbot_agent_sdk.api_client")
    sdk_api_client.API_BASE = "old-api"
    sdk_api_client.TOKEN_URL = "old-token"
    monkeypatch.setitem(sys.modules, "qqbot_agent_sdk", sdk)
    monkeypatch.setitem(sys.modules, "qqbot_agent_sdk.api_client", sdk_api_client)

    config = QQBotConfig(
        app_id="app-id",
        app_secret="test-secret",
        api_base="https://api.example.test",
        token_url="https://token.example.test",
    )
    transport = QQOfficialSDKTransport(config)
    assert transport.api.http_client is transport._http_client
    assert sdk_api_client.API_BASE == config.api_base
    assert sdk_api_client.TOKEN_URL == config.token_url
    asyncio.run(transport.close())
    assert transport._http_client is None


def test_qq_reply_text_is_safely_truncated() -> None:
    result = truncate_qq_text("学习结果：" + "很长的内容。" * 100, 32)
    assert len(result) <= 32
    assert result.endswith("…")


def test_channel_binding_api_persists_and_unbinds(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "api.db")) as client:
        assert client.put("/courses/c", json={"id": "c", "name": "course"}).status_code == 200
        assert client.put("/courses/c/topics", json=[]).status_code == 200
        assert client.put(
            "/courses/c/goals/g",
            json={
                "id": "g",
                "course_id": "c",
                "exam_at": "2026-12-20T09:00:00+08:00",
                "available_minutes": 30,
            },
        ).status_code == 200
        created = client.post(
            "/courses/c/teaching-sessions",
            json={"goal_id": "g", "session_id": "s"},
        )
        assert created.status_code == 201
        bound = client.put(
            "/channel-bindings",
            json={
                "openid": "openid-1",
                "learner_id": "learner-1",
                "course_id": "c",
                "session_id": "s",
            },
        )
        assert bound.status_code == 200
        assert bound.json()["external_user_id"] == "openid-1"
        fetched = client.get("/channel-bindings/qq_c2c/openid-1")
        assert fetched.status_code == 200
        assert fetched.json()["session_id"] == "s"
        assert client.delete("/channel-bindings/qq_c2c/openid-1").status_code == 204
        assert client.get("/channel-bindings/qq_c2c/openid-1").status_code == 404

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError, BadRequestError

from studypilot.application.llm import (
    LLMEmptyResponseError,
    LLMInvalidJSONError,
    LLMProviderConfig,
    LLMRequestError,
    LLMResponseValidationError,
    LLMTeacherProvider,
    LLMEvaluatorProvider,
    LLMTimeoutError,
    UnauthorizedCitationError,
    adjudicate_evaluation,
)
from studypilot.application.teaching import TeachingWorkflow
from studypilot.domain.knowledge import (
    Citation,
    DocumentBlock,
    EvidenceRelation,
    KnowledgeWindow,
    KnowledgeWindowItem,
    ParserKind,
    SearchHit,
)
from studypilot.domain.models import EvidenceLevel, ExamGoal, MasteryState, Topic
from studypilot.domain.teaching import ScoringPoint, TeachingAction, TeachingStatus


class _FakeCompletions:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.completions = _FakeCompletions(response, error)
        self.chat = SimpleNamespace(completions=self.completions)


class _SequenceCompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("test client received more requests than expected")
        return self.responses.pop(0)


class _SequenceClient:
    def __init__(self, responses: list[object]) -> None:
        self.completions = _SequenceCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


class _SchemaRejectingCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            request = httpx.Request("POST", "https://llm.invalid/v1/chat/completions")
            response = httpx.Response(400, request=request)
            raise BadRequestError(
                "strict response format is unsupported",
                response=response,
                body={"error": {"message": "unsupported"}},
            )
        return self.response


class _SchemaRejectingClient:
    def __init__(self, response: object) -> None:
        self.completions = _SchemaRejectingCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)


def _response(payload: object):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )


def _config() -> LLMProviderConfig:
    return LLMProviderConfig(
        api_key="test-key",
        base_url="https://llm.invalid/v1",
        model="test-model",
        timeout_seconds=2.5,
    )


def _topic() -> Topic:
    return Topic(
        id="bayes",
        course_id="course",
        name="贝叶斯公式",
        exam_points=10,
        learning_minutes=10,
        evidence_level=EvidenceLevel.PAST_EXAM,
        evidence_confidence=1.0,
        frequency=0.8,
    )


def _window() -> KnowledgeWindow:
    block = DocumentBlock(
        id="block-1",
        source_asset_id="source-1",
        course_id="course",
        parser_kind=ParserKind.TEXT,
        parser_version="1",
        block_index=0,
        text="贝叶斯公式用于由结果反推原因。",
        content_hash="a" * 64,
    )
    citation = Citation(
        block_id=block.id,
        source_asset_id=block.source_asset_id,
        course_id=block.course_id,
        display_name="notes.txt",
        page_number=None,
        section="概率",
        block_index=block.block_index,
        quote=block.text,
    )
    return KnowledgeWindow(
        query="贝叶斯公式",
        course_id="course",
        max_blocks=2,
        max_chars=1000,
        used_chars=len(block.text),
        items=(
            KnowledgeWindowItem(
                relation=EvidenceRelation.DEFINITION,
                hit=SearchHit(block=block, score=1, citation=citation),
            ),
        ),
    )


def _rubric() -> tuple[ScoringPoint, ...]:
    return (
        ScoringPoint(
            id="definition",
            description="说明定义",
            evidence=("后验概率",),
            citations=("block-1",),
        ),
        ScoringPoint(
            id="formula",
            description="说明公式",
            evidence=("条件概率",),
        ),
    )


def _teacher_payload() -> dict:
    return {
        "explanation": "贝叶斯公式把先验、似然和证据联系起来。",
        "question": "请说明贝叶斯公式的用途。",
        "scoring_points": [
            {
                "id": "definition",
                "description": "说明定义",
                "evidence": ["后验概率"],
                "citations": ["block-1"],
            }
        ],
        "citations": ["block-1"],
    }


def _evaluation_payload(*, state: str = "READY", citations=None) -> dict:
    return {
        "point_evaluations": [
            {
                "scoring_point_id": "definition",
                "satisfied": True,
                "evidence": ["后验概率"],
                "citations": citations or [],
                "reason": "答案说明了后验概率",
            },
            {
                "scoring_point_id": "formula",
                "satisfied": False,
                "evidence": [],
                "reason": "没有看到条件概率",
            },
        ],
        "mastery_state": state,
        "next_action": "REPLAN",
        "reason": "模型判断",
        "score": 0.99,
    }


def test_config_reads_project_env_without_vendor_defaults() -> None:
    config = LLMProviderConfig.from_env(
        {
            "STUDYPILOT_LLM_API_KEY": "k",
            "STUDYPILOT_LLM_BASE_URL": "https://example.invalid/v1",
            "STUDYPILOT_LLM_MODEL": "compatible-model",
            "STUDYPILOT_LLM_TIMEOUT_SECONDS": "4",
            "STUDYPILOT_LLM_MAX_RETRIES": "1",
        }
    )

    assert config.api_key == "k"
    assert config.base_url == "https://example.invalid/v1"
    assert config.model == "compatible-model"
    assert config.timeout_seconds == 4
    assert config.max_retries == 1


def test_teacher_parses_structured_json_and_prompt_contains_only_window() -> None:
    fake = _FakeClient(_response(_teacher_payload()))
    provider = LLMTeacherProvider(_config(), client=fake)

    content = provider.teach(
        _topic(),
        _window(),
        mastery_state=MasteryState.FRAGILE,
        remaining_minutes=17,
    )

    assert content.question.startswith("请说明")
    assert content.citations == ("block-1",)
    request = fake.completions.calls[0]
    user_prompt = request["messages"][1]["content"]
    assert "notes.txt" in user_prompt
    assert "block-1" in user_prompt
    assert "full corpus" not in user_prompt
    assert "source-1" in user_prompt
    assert request["model"] == "test-model"
    assert request["timeout"] == 2.5
    assert request["response_format"]["type"] == "json_schema"


def test_teacher_rejects_citation_outside_window() -> None:
    payload = _teacher_payload()
    payload["citations"] = ["not-in-window"]
    provider = LLMTeacherProvider(_config(), client=_FakeClient(_response(payload)))

    with pytest.raises(UnauthorizedCitationError):
        provider.teach(_topic(), _window())


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (_response("not-json"), LLMInvalidJSONError),
        (SimpleNamespace(choices=[]), LLMEmptyResponseError),
    ],
)
def test_teacher_rejects_invalid_or_empty_provider_response(response, error) -> None:
    provider = LLMTeacherProvider(_config(), client=_FakeClient(response))

    with pytest.raises(error):
        provider.teach(_topic(), _window())


def test_timeout_and_api_errors_are_explicit_failures() -> None:
    timeout = APITimeoutError(request=httpx.Request("POST", "https://llm.invalid"))
    provider = LLMEvaluatorProvider(_config(), client=_FakeClient(error=timeout))
    with pytest.raises(LLMTimeoutError):
        provider.evaluate("q", "answer", _rubric(), allowed_citations=_window())

    provider = LLMEvaluatorProvider(_config(), client=_FakeClient(error=RuntimeError("down")))
    with pytest.raises(LLMRequestError):
        provider.evaluate("q", "answer", _rubric(), allowed_citations=_window())


def test_evaluator_parses_rubric_and_server_adjudication_overrides_model_state() -> None:
    fake = _FakeClient(_response(_evaluation_payload(state="READY")))
    provider = LLMEvaluatorProvider(_config(), client=fake)

    result = provider.evaluate(
        question="q",
        answer="后验概率",
        scoring_points=_rubric(),
        allowed_citations=_window(),
    )
    assert result.mastery_state is MasteryState.READY
    final = adjudicate_evaluation(
        result,
        _rubric(),
        _window(),
        answer="后验概率",
    )
    assert final.mastery_state is MasteryState.FRAGILE
    assert final.next_action is TeachingAction.REPLAN
    assert "服务端" in final.reason


def test_evaluator_repairs_only_natural_language_next_action_once() -> None:
    invalid = _evaluation_payload()
    invalid["next_action"] = "请继续讲解并再做一道题"
    valid = _evaluation_payload()
    client = _SequenceClient([_response(invalid), _response(valid)])

    result = LLMEvaluatorProvider(_config(), client=client).evaluate(
        "q",
        "answer",
        _rubric(),
        allowed_citations=_window(),
    )

    assert result.next_action is TeachingAction.REPLAN
    assert len(client.completions.calls) == 2
    repair_prompt = client.completions.calls[1]["messages"][-1]["content"]
    assert "FORMAT REPAIR ONLY" in repair_prompt
    assert "REPLAN" in repair_prompt
    assert "请继续讲解并再做一道题" in repair_prompt


def test_provider_falls_back_to_json_object_when_strict_schema_is_rejected() -> None:
    client = _SchemaRejectingClient(_response(_evaluation_payload()))

    result = LLMEvaluatorProvider(_config(), client=client).evaluate(
        "q",
        "answer",
        _rubric(),
        allowed_citations=_window(),
    )

    assert result.next_action is TeachingAction.REPLAN
    assert len(client.completions.calls) == 2
    assert client.completions.calls[0]["response_format"]["type"] == "json_schema"
    assert client.completions.calls[1]["response_format"] == {"type": "json_object"}


def test_evaluator_repair_does_not_map_second_invalid_action() -> None:
    invalid = _evaluation_payload()
    invalid["next_action"] = "请继续讲解并再做一道题"
    still_invalid = _evaluation_payload()
    still_invalid["next_action"] = "下一步：继续教学"
    client = _SequenceClient([_response(invalid), _response(still_invalid)])

    with pytest.raises(
        LLMResponseValidationError,
        match="next_action remained outside the allowed enum",
    ):
        LLMEvaluatorProvider(_config(), client=client).evaluate(
            "q",
            "answer",
            _rubric(),
            allowed_citations=_window(),
        )

    assert len(client.completions.calls) == 2


def test_evaluator_does_not_default_missing_next_action() -> None:
    payload = _evaluation_payload()
    del payload["next_action"]
    provider = LLMEvaluatorProvider(_config(), client=_FakeClient(_response(payload)))

    with pytest.raises(LLMResponseValidationError, match="missing required next_action"):
        provider.evaluate("q", "answer", _rubric(), allowed_citations=_window())


def test_evaluator_rejects_unauthorized_citation_and_missing_evidence() -> None:
    payload = _evaluation_payload(citations=["not-in-window"])
    provider = LLMEvaluatorProvider(_config(), client=_FakeClient(_response(payload)))
    with pytest.raises(UnauthorizedCitationError):
        provider.evaluate("q", "answer", _rubric(), allowed_citations=_window())

    payload = _evaluation_payload()
    payload["point_evaluations"][0]["evidence"] = []
    payload["point_evaluations"][0]["citations"] = []
    provider = LLMEvaluatorProvider(_config(), client=_FakeClient(_response(payload)))
    with pytest.raises(LLMResponseValidationError):
        provider.evaluate("q", "answer", _rubric(), allowed_citations=_window())


def test_empty_answer_is_server_side_gap_even_if_model_says_ready() -> None:
    result = LLMEvaluatorProvider(
        _config(), client=_FakeClient(_response(_evaluation_payload(state="READY")))
    ).evaluate("q", "", _rubric(), allowed_citations=_window())

    final = adjudicate_evaluation(result, _rubric(), _window(), answer="")
    assert final.mastery_state is MasteryState.GAP
    assert all(not item.satisfied for item in final.point_evaluations)


def test_workflow_injects_llm_providers_and_migrates_state_from_rubric(tmp_path) -> None:
    class Teacher:
        calls = 0

        def teach(self, **kwargs):
            if self.calls == 0:
                assert kwargs["mastery_state"] is MasteryState.UNSEEN
                assert kwargs["remaining_minutes"] == 30
            self.calls += 1
            return {
                "explanation": "说明",
                "question": "问题",
                "scoring_points": [
                    {"id": "p", "description": "定义", "evidence": ["答案"]}
                ],
            }

    class Evaluator:
        def evaluate(self, **kwargs):
            assert kwargs["allowed_citations"] is not None
            return {
                "score": 1,
                "mastery_state": "READY",
                "next_action": "COMPLETE",
                "reason": "模型说 ready",
                "point_evaluations": [
                    {
                        "scoring_point_id": "p",
                        "satisfied": False,
                        "evidence": [],
                        "reason": "未命中",
                    }
                ],
            }

    goal = ExamGoal(
        id="g",
        course_id="course",
        exam_at=datetime(2026, 12, 20, tzinfo=UTC),
        available_minutes=30,
    )
    workflow = TeachingWorkflow(
        goal,
        [_topic().model_copy(update={"name": "topic"})],
        tmp_path / "llm-workflow.db",
        teacher_provider=Teacher(),
        evaluator=Evaluator(),
    )
    assert workflow.start().status is TeachingStatus.WAITING_ANSWER
    resumed = workflow.resume("anything")
    assert resumed.mastery_state is MasteryState.GAP
    assert resumed.status is TeachingStatus.WAITING_ANSWER
    workflow.close()

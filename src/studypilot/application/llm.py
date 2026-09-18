"""OpenAI-compatible teacher and evaluator providers.

The deterministic providers remain the default for local tests.  The classes
in this module are an intentionally small, dependency-injected boundary around
the official ``openai`` Python SDK: configuration comes from a caller or the
environment, prompts contain only the current teaching context, and every
model response is parsed into the existing Pydantic domain models before it
can reach the workflow.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from studypilot.domain.knowledge import Citation, KnowledgeWindow, KnowledgeWindowItem
from studypilot.domain.models import MasteryState, Topic
from studypilot.domain.teaching import (
    EvaluationResult,
    ScoringPoint,
    ScoringPointEvaluation,
    TeachingAction,
    TeachingContent,
)


class LLMProviderError(RuntimeError):
    """Base class for an explicit, non-successful model-provider result."""


class LLMConfigurationError(LLMProviderError, ValueError):
    """The provider has no usable API key/model or an invalid setting."""


class LLMRequestError(LLMProviderError):
    """The SDK could not complete a request."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(message)


class LLMTimeoutError(LLMRequestError):
    """The provider exceeded the configured request timeout."""


class LLMEmptyResponseError(LLMProviderError):
    """The provider returned no usable assistant content."""


class LLMInvalidJSONError(LLMProviderError):
    """The provider returned content that is not a JSON object."""


class LLMResponseValidationError(LLMProviderError, ValueError):
    """JSON was valid but did not satisfy the structured response contract."""


class UnauthorizedCitationError(LLMResponseValidationError):
    """A model response referred to a citation outside the supplied window."""


_EVALUATION_ACTION_VALUES = tuple(action.value for action in TeachingAction)
_EVALUATION_MASTERY_VALUES = (
    MasteryState.READY.value,
    MasteryState.FRAGILE.value,
    MasteryState.GAP.value,
)

# These schemas serve two purposes: OpenAI-compatible clients that implement
# ``json_schema`` can enforce the contract at the API boundary, while clients
# that only implement ``json_object`` receive the same contract in the prompt.
# Keep optional domain defaults out of ``required`` only where the Pydantic
# model itself intentionally supplies a harmless presentation default.
_TEACHER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "explanation": {"type": "string"},
        "question": {"type": "string"},
        "scoring_points": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "citations": {"type": "array", "items": {"type": "string"}},
                    "required": {"type": "boolean"},
                    "weight": {"type": "number", "exclusiveMinimum": 0},
                },
                "required": [
                    "id",
                    "description",
                    "evidence",
                    "citations",
                    "required",
                    "weight",
                ],
            },
        },
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["explanation", "question", "scoring_points", "citations"],
}

_EVALUATOR_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "point_evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "scoring_point_id": {"type": "string"},
                    "satisfied": {"type": "boolean"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "citations": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": [
                    "scoring_point_id",
                    "satisfied",
                    "evidence",
                    "citations",
                    "reason",
                ],
            },
        },
        "mastery_state": {"type": "string", "enum": list(_EVALUATION_MASTERY_VALUES)},
        "next_action": {"type": "string", "enum": list(_EVALUATION_ACTION_VALUES)},
        "reason": {"type": "string"},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "point_evaluations",
        "mastery_state",
        "next_action",
        "reason",
        "score",
        "citations",
    ],
}


class LLMProviderConfig(BaseModel):
    """Runtime settings for an OpenAI-compatible endpoint.

    ``base_url`` is optional so the official SDK default can be used.  No
    vendor URL, model name, or key is embedded in the project; deployments
    provide them explicitly or through environment variables.
    """

    model_config = ConfigDict(frozen=True)

    api_key: str | None = Field(default=None, min_length=1)
    base_url: str | None = Field(default=None, min_length=1)
    model: str = Field(min_length=1, max_length=200)
    timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    max_retries: int = Field(default=0, ge=0, le=10)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "LLMProviderConfig":
        """Read provider settings without assuming a particular vendor.

        ``STUDYPILOT_LLM_*`` is the project-specific spelling.  The standard
        ``OPENAI_API_KEY``/``OPENAI_BASE_URL`` names are accepted as a useful
        compatibility fallback, while the model can be supplied as either
        ``STUDYPILOT_LLM_MODEL`` or ``OPENAI_MODEL``.
        """

        source = os.environ if env is None else env

        def first(*names: str) -> str | None:
            for name in names:
                value = source.get(name)
                if value is not None and value.strip():
                    return value.strip()
            return None

        api_key = first("STUDYPILOT_LLM_API_KEY", "OPENAI_API_KEY")
        model = first("STUDYPILOT_LLM_MODEL", "OPENAI_MODEL")
        if not api_key:
            raise LLMConfigurationError(
                "set STUDYPILOT_LLM_API_KEY or OPENAI_API_KEY before using an LLM provider"
            )
        if not model:
            raise LLMConfigurationError(
                "set STUDYPILOT_LLM_MODEL or OPENAI_MODEL before using an LLM provider"
            )

        timeout_raw = first("STUDYPILOT_LLM_TIMEOUT_SECONDS", "OPENAI_TIMEOUT_SECONDS")
        retries_raw = first("STUDYPILOT_LLM_MAX_RETRIES", "OPENAI_MAX_RETRIES")
        try:
            timeout = float(timeout_raw) if timeout_raw is not None else 30.0
            retries = int(retries_raw) if retries_raw is not None else 0
        except ValueError as error:
            raise LLMConfigurationError("LLM timeout/retry settings must be numeric") from error

        try:
            return cls(
                api_key=api_key,
                base_url=first("STUDYPILOT_LLM_BASE_URL", "OPENAI_BASE_URL"),
                model=model,
                timeout_seconds=timeout,
                max_retries=retries,
            )
        except ValidationError as error:
            raise LLMConfigurationError("invalid LLM provider configuration") from error

    from_environment = from_env


# A descriptive alias is convenient for callers that prefer “settings” over
# “config”, without making the public boundary vendor-specific.
OpenAICompatibleConfig = LLMProviderConfig
LLMConfig = LLMProviderConfig


def citation_ids(
    allowed: KnowledgeWindow
    | Sequence[Citation]
    | Sequence[KnowledgeWindowItem]
    | Sequence[str]
    | None,
) -> frozenset[str]:
    """Return the only citation identifiers a model may mention."""

    if allowed is None:
        return frozenset()
    if isinstance(allowed, KnowledgeWindow):
        return frozenset(item.hit.citation.block_id for item in allowed.items)

    result: set[str] = set()
    for item in allowed:
        if isinstance(item, str):
            result.add(item)
        elif isinstance(item, Citation):
            result.add(item.block_id)
        elif isinstance(item, Mapping):
            block_id = item.get("block_id") or item.get("citation_id")
            if isinstance(block_id, str) and block_id:
                result.add(block_id)
        elif isinstance(item, KnowledgeWindowItem):
            result.add(item.hit.citation.block_id)
        else:
            block_id = getattr(item, "block_id", None) or getattr(item, "citation_id", None)
            if isinstance(block_id, str) and block_id:
                result.add(block_id)
    return frozenset(result)


def _window_context(
    allowed: KnowledgeWindow
    | Sequence[Citation]
    | Sequence[KnowledgeWindowItem]
    | Sequence[str]
    | None,
) -> list[dict[str, Any]]:
    """Serialize only the supplied window/citations for a prompt."""

    if isinstance(allowed, KnowledgeWindow):
        context: list[dict[str, Any]] = []
        for item in allowed.items:
            citation = item.hit.citation
            context.append(
                {
                    "citation_id": citation.block_id,
                    "source_asset_id": citation.source_asset_id,
                    "relation": item.relation.value,
                    "source": citation.display_name,
                    "section": citation.section,
                    "page": citation.page_number,
                    "quote": citation.quote,
                }
            )
        return context
    context = []
    for item in allowed or ():
        if isinstance(item, KnowledgeWindowItem):
            citation = item.hit.citation
            context.append(
                {
                    "citation_id": citation.block_id,
                    "source_asset_id": citation.source_asset_id,
                    "relation": item.relation.value,
                    "source": citation.display_name,
                    "section": citation.section,
                    "page": citation.page_number,
                    "quote": citation.quote,
                }
            )
        elif isinstance(item, Citation):
            context.append(
                {
                    "citation_id": item.block_id,
                    "source_asset_id": item.source_asset_id,
                    "source": item.display_name,
                    "section": item.section,
                    "page": item.page_number,
                    "quote": item.quote,
                }
            )
        elif isinstance(item, Mapping):
            context.append(dict(item))
        else:
            context.append({"citation_id": str(item)})
    return context


def build_teacher_messages(
    *,
    topic: Topic,
    knowledge_window: KnowledgeWindow,
    mastery_state: MasteryState,
    remaining_minutes: int,
    attempt: int = 1,
    question_override: str | None = None,
    scoring_points_override: Sequence[ScoringPoint] | None = None,
) -> list[dict[str, str]]:
    """Build a bounded teacher prompt from one topic and one window."""

    allowed = sorted(citation_ids(knowledge_window))
    payload = {
        "topic": {
            "id": topic.id,
            "name": topic.name,
            "exam_points": topic.exam_points,
            "learning_minutes": topic.learning_minutes,
        },
        "mastery_state": MasteryState(mastery_state).value,
        "remaining_minutes": max(0, int(remaining_minutes)),
        "attempt": max(1, int(attempt)),
        "knowledge_window": _window_context(knowledge_window),
        "allowed_citation_ids": allowed,
        "question_override": question_override,
        "scoring_points_override": [
            point.model_dump(mode="json")
            for point in (scoring_points_override or ())
        ],
    }
    system = (
        "You are a structured teaching provider. Return JSON only with keys "
        "explanation, question, scoring_points, citations. scoring_points is a "
        "non-empty list of {id, description, evidence, citations, required, weight}. "
        "Every citation must be one of allowed_citation_ids; never invent a source "
        "or cite material outside the supplied knowledge_window. Keep evidence "
        "observable and suitable for a later rubric-constrained evaluation. "
        "The exact response schema is: "
        + json.dumps(_TEACHER_RESPONSE_SCHEMA, ensure_ascii=False, separators=(",", ":"))
    )
    user = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_evaluator_messages(
    *,
    question: str,
    scoring_points: Sequence[ScoringPoint],
    answer: str,
    allowed_citations: KnowledgeWindow | Sequence[Citation] | Sequence[str] | None,
) -> list[dict[str, str]]:
    """Build a rubric/evidence prompt without loading a course corpus."""

    payload = {
        "question": question,
        "rubric": [point.model_dump(mode="json") for point in scoring_points],
        "student_answer": answer,
        "allowed_citations": _window_context(allowed_citations),
        "allowed_citation_ids": sorted(citation_ids(allowed_citations)),
    }
    system = (
        "You are a structured answer evaluator. Return JSON only with keys "
        "point_evaluations, mastery_state, next_action, reason, score, citations. Include "
        "exactly one point_evaluations item for every rubric id; each item has "
        "scoring_point_id, satisfied, evidence, citations, reason. citations may "
        "contain only allowed_citation_ids. A satisfied point must include concrete "
        "answer evidence or an allowed citation. mastery_state must be READY, "
        "FRAGILE, or GAP; the server will make the final state decision. "
        "next_action is an enum token only, exactly one of "
        + json.dumps(list(_EVALUATION_ACTION_VALUES), ensure_ascii=False)
        + "; never put a sentence or advice in next_action—put natural-language "
        "feedback in reason. The exact response schema is: "
        + json.dumps(_EVALUATOR_RESPONSE_SCHEMA, ensure_ascii=False, separators=(",", ":"))
    )
    user = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_teaching_content_references(
    content: TeachingContent,
    knowledge_window: KnowledgeWindow,
) -> TeachingContent:
    """Reject every citation not present in the current Knowledge Window."""

    content = TeachingContent.model_validate(content)
    allowed = citation_ids(knowledge_window)
    requested = set(content.citations)
    for point in content.scoring_points:
        requested.update(point.citations)
    unauthorized = requested - allowed
    if unauthorized:
        raise UnauthorizedCitationError(
            "teacher response cited blocks outside the Knowledge Window: "
            + ", ".join(sorted(unauthorized))
        )
    return content


def validate_evaluation_references(
    result: EvaluationResult,
    rubric: Sequence[ScoringPoint],
    allowed_citations: KnowledgeWindow | Sequence[Citation] | Sequence[str] | None,
) -> EvaluationResult:
    """Validate rubric coverage, explicit evidence, and citation ownership."""

    try:
        result = EvaluationResult.model_validate(result)
        points = tuple(ScoringPoint.model_validate(point) for point in rubric)
    except ValidationError as error:
        raise LLMResponseValidationError("evaluator JSON does not match the schema") from error
    if not points:
        raise LLMResponseValidationError("evaluator cannot score an empty rubric")
    point_ids = [point.id for point in points]
    if len(set(point_ids)) != len(point_ids):
        raise LLMResponseValidationError("rubric scoring point ids must be unique")
    result_ids = [item.scoring_point_id for item in result.point_evaluations]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(point_ids):
        raise LLMResponseValidationError(
            "evaluator must return exactly one result for every rubric scoring point"
        )

    allowed = citation_ids(allowed_citations)
    unauthorized_result = set(result.citations) - allowed
    if unauthorized_result:
        raise UnauthorizedCitationError(
            "evaluator cited blocks outside the Knowledge Window: "
            + ", ".join(sorted(unauthorized_result))
        )
    rubric_citations = {
        reference for point in points for reference in point.citations
    }
    unauthorized_rubric = rubric_citations - allowed
    if unauthorized_rubric:
        raise UnauthorizedCitationError(
            "rubric cited blocks outside the Knowledge Window: "
            + ", ".join(sorted(unauthorized_rubric))
        )

    for item in result.point_evaluations:
        unauthorized = set(item.citations) - allowed
        if unauthorized:
            raise UnauthorizedCitationError(
                "evaluator cited blocks outside the Knowledge Window: "
                + ", ".join(sorted(unauthorized))
            )
        # A satisfied point must leave an auditable trace.  Free-form evidence
        # is allowed because the model may quote/paraphrase the answer; source
        # references are separately constrained by ``citations`` above.
        if item.satisfied and not item.evidence and not item.citations:
            raise LLMResponseValidationError(
                f"satisfied scoring point {item.scoring_point_id!r} has no evidence"
            )
    return result


def adjudicate_evaluation(
    result: EvaluationResult,
    rubric: Sequence[ScoringPoint],
    allowed_citations: KnowledgeWindow | Sequence[Citation] | Sequence[str] | None,
    *,
    answer: str | None = None,
) -> EvaluationResult:
    """Derive score/mastery/action from validated rubric evidence.

    The model's score, mastery suggestion, and next action are intentionally
    not authoritative.  The workflow uses this function as the final state
    migration boundary, so a provider cannot turn an unsupported judgement
    into READY or skip replanning.
    """

    result = validate_evaluation_references(result, rubric, allowed_citations)
    points = tuple(ScoringPoint.model_validate(point) for point in rubric)
    by_id = {item.scoring_point_id: item for item in result.point_evaluations}

    if answer is not None and not answer.strip():
        ordered = tuple(
            item.model_copy(
                update={
                    "satisfied": False,
                    "evidence": (),
                    "citations": (),
                    "reason": "空答案：未观察到该评分点证据",
                }
            )
            for item in (by_id[point.id] for point in points)
        )
        score = 0.0
        mastery = MasteryState.GAP
        reason = "服务端裁决：空答案，掌握状态为 GAP"
    else:
        ordered = tuple(by_id[point.id] for point in points)
        total_weight = sum(point.weight for point in points)
        earned_weight = sum(
            point.weight
            for point, item in zip(points, ordered, strict=True)
            if item.satisfied
        )
        score = earned_weight / total_weight if total_weight else 0.0
        required = [
            item
            for point, item in zip(points, ordered, strict=True)
            if point.required
        ]
        if score >= 1.0 - 1e-9 and all(item.satisfied for item in required):
            mastery = MasteryState.READY
        elif score <= 1e-9:
            mastery = MasteryState.GAP
        else:
            mastery = MasteryState.FRAGILE
        reason = (
            f"服务端按 rubric 证据裁决：命中 {sum(item.satisfied for item in ordered)}/"
            f"{len(ordered)} 个评分点，掌握状态为 {mastery.value}"
        )

    original_reason = result.reason.strip()
    if original_reason and not reason.startswith(original_reason):
        reason = f"{original_reason}；{reason}"
    return EvaluationResult(
        score=round(score, 6),
        mastery_state=mastery,
        point_evaluations=ordered,
        citations=result.citations,
        reason=reason[:4_000],
        next_action=TeachingAction.REPLAN,
    )


def _is_json_schema_unsupported(error: LLMRequestError) -> bool:
    """Recognise only HTTP statuses commonly used for an unsupported format."""

    return error.status_code in {400, 404, 422}


class _NextActionFormatError(LLMProviderError):
    """The response is otherwise valid, but ``next_action`` is not an enum token."""


def _is_only_next_action_validation_error(error: ValidationError) -> bool:
    """Return whether Pydantic rejected only the action-token field."""

    details = error.errors()
    return bool(details) and all(
        tuple(detail.get("loc", ())) == ("next_action",)
        and detail.get("type") in {"enum", "missing", "string_type"}
        for detail in details
    )


def _normalize_evaluator_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize harmless structural aliases without inventing judgement fields."""

    normalized = dict(payload)
    if "point_evaluations" not in normalized:
        for alias in ("evaluations", "scoring_point_evaluations"):
            if alias in normalized:
                normalized["point_evaluations"] = normalized[alias]
                break
    if "mastery_state" not in normalized and "mastery" in normalized:
        normalized["mastery_state"] = normalized["mastery"]
    if "citations" not in normalized and "citation_ids" in normalized:
        normalized["citations"] = normalized["citation_ids"]

    raw_points = normalized.get("point_evaluations")
    if isinstance(raw_points, list):
        normalized_points: list[Any] = []
        for raw in raw_points:
            if isinstance(raw, Mapping):
                item = dict(raw)
                if "scoring_point_id" not in item and "point_id" in item:
                    item["scoring_point_id"] = item["point_id"]
                if "citations" not in item and "citation_ids" in item:
                    item["citations"] = item["citation_ids"]
                normalized_points.append(item)
            else:
                normalized_points.append(raw)
        normalized["point_evaluations"] = normalized_points
    return normalized


def _build_evaluator_format_repair_messages(
    messages: Sequence[Mapping[str, str]],
    previous_payload: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Ask the endpoint for one syntax-only enum-token repair."""

    previous_json = json.dumps(
        dict(previous_payload), ensure_ascii=False, separators=(",", ":")
    )
    allowed_actions = json.dumps(
        list(_EVALUATION_ACTION_VALUES), ensure_ascii=False, separators=(",", ":")
    )
    repair = (
        "FORMAT REPAIR ONLY. Return one JSON object matching the exact schema below. "
        "The previous object is valid for scoring and evidence except that its "
        "next_action is not an allowed enum token. Preserve score, mastery_state, "
        "point_evaluations, citations, and all reason/feedback text exactly; do not "
        "re-evaluate the student or change business judgement. Replace only "
        "next_action with exactly one token from "
        + allowed_actions
        + ". Put any natural-language advice in reason, never in next_action. "
        "Do not add markdown or prose outside the JSON object. Exact schema: "
        + json.dumps(_EVALUATOR_RESPONSE_SCHEMA, ensure_ascii=False, separators=(",", ":"))
        + " Previous JSON: "
        + previous_json
    )
    repaired_messages = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    repaired_messages.extend(
        [
            {"role": "assistant", "content": previous_json},
            {"role": "user", "content": repair},
        ]
    )
    return repaired_messages


def _parse_evaluator_payload(
    payload: Mapping[str, Any],
    points: Sequence[ScoringPoint],
    allowed_citations: KnowledgeWindow | Sequence[Citation] | Sequence[str] | None,
) -> EvaluationResult:
    """Parse one model result, isolating an enum-token error for one repair."""

    normalized = _normalize_evaluator_payload(payload)
    raw_action = normalized.get("next_action")
    # EvaluationResult has a convenience default for deterministic callers,
    # but a model response must state the action explicitly.  Otherwise a
    # missing field would silently become a business decision.
    if "next_action" not in normalized:
        raise LLMResponseValidationError(
            "evaluator JSON is missing required next_action enum token"
        )
    try:
        result = EvaluationResult.model_validate(normalized)
    except ValidationError as error:
        if (
            isinstance(raw_action, str)
            and raw_action.strip()
            and _is_only_next_action_validation_error(error)
        ):
            raise _NextActionFormatError(
                "evaluator next_action is not an allowed enum token"
            ) from error
        raise LLMResponseValidationError(
            "evaluator JSON does not match the structured result schema"
        ) from error
    if result.mastery_state is MasteryState.UNSEEN:
        raise LLMResponseValidationError(
            "evaluator mastery_state must be READY, FRAGILE, or GAP"
        )
    return validate_evaluation_references(result, points, allowed_citations)


class _SDKProvider:
    """Shared SDK request/response mechanics for the two providers."""

    def __init__(
        self,
        config: LLMProviderConfig | None = None,
        *,
        client: Any | None = None,
        model: str | None = None,
    ) -> None:
        if config is None:
            if client is None:
                config = LLMProviderConfig.from_env()
            else:
                configured_model = model or os.environ.get("STUDYPILOT_LLM_MODEL")
                if not configured_model:
                    raise LLMConfigurationError(
                        "an injected client still needs a model in config or model=..."
                    )
                config = LLMProviderConfig(model=configured_model)
        elif model is not None:
            config = config.model_copy(update={"model": model})
        if client is None and not config.api_key:
            raise LLMConfigurationError("an API key is required to construct the SDK client")
        self.config = config
        if client is not None:
            self.client = client
        else:
            kwargs: dict[str, Any] = {
                "api_key": config.api_key,
                "timeout": config.timeout_seconds,
                "max_retries": config.max_retries,
            }
            if config.base_url:
                kwargs["base_url"] = config.base_url
            self.client = OpenAI(**kwargs)

        # ``None`` means the endpoint has not yet told us whether it accepts
        # OpenAI's strict JSON Schema response format. A 4xx response on the
        # schema request downgrades this instance to ``json_object`` for all
        # subsequent calls; compatible providers remain strict by default.
        self._json_schema_supported: bool | None = None

    def _request_json(
        self,
        messages: list[dict[str, str]],
        *,
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "studypilot_response",
    ) -> dict[str, Any]:
        response_format: dict[str, Any] = {"type": "json_object"}
        strict_schema = response_schema is not None and self._json_schema_supported is not False
        if strict_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": response_schema,
                },
            }
        try:
            response = self._request_json_once(messages, response_format=response_format)
            if strict_schema:
                self._json_schema_supported = True
            return response
        except LLMRequestError as error:
            if not strict_schema or not _is_json_schema_unsupported(error):
                raise
            # Some compatible endpoints support JSON mode but reject the
            # optional strict schema. Retry this transport format once; the
            # caller still performs semantic Pydantic/rubric validation.
            self._json_schema_supported = False
            return self._request_json_once(
                messages,
                response_format={"type": "json_object"},
            )

    def _request_json_once(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0,
                response_format=response_format,
                timeout=self.config.timeout_seconds,
            )
        except APITimeoutError as error:
            raise LLMTimeoutError("LLM request timed out") from error
        except TimeoutError as error:
            # Keep simple HTTP/SDK test doubles and adapters on the same
            # explicit timeout path as the official SDK exception.
            raise LLMTimeoutError("LLM request timed out") from error
        except (APIConnectionError, APIError) as error:
            status_code = getattr(error, "status_code", None)
            response = getattr(error, "response", None)
            if status_code is None:
                status_code = getattr(response, "status_code", None)
            if not isinstance(status_code, int):
                status_code = None
            raise LLMRequestError(
                "LLM API request failed",
                status_code=status_code,
            ) from error
        except Exception as error:
            # A mock/client implementation may expose a generic transport
            # exception.  It is still a failed provider call, never a score.
            raise LLMRequestError("LLM client request failed") from error

        choices = _get(response, "choices")
        if not choices:
            raise LLMEmptyResponseError("LLM response contained no choices")
        message = _get(choices[0], "message")
        content = _get(message, "content")
        if isinstance(content, list):
            content = "".join(
                str(_get(part, "text") or "")
                for part in content
                if _get(part, "text") is not None
            )
        if not isinstance(content, str) or not content.strip():
            raise LLMEmptyResponseError("LLM response contained no assistant content")
        try:
            parsed = json.loads(content)
        except (TypeError, json.JSONDecodeError) as error:
            raise LLMInvalidJSONError("LLM response was not valid JSON") from error
        if not isinstance(parsed, dict):
            raise LLMInvalidJSONError("LLM response JSON must be an object")
        return parsed


class LLMTeacherProvider(_SDKProvider):
    """Generate bounded structured teaching content from one Knowledge Window."""

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        client: Any | None = None,
    ) -> "LLMTeacherProvider":
        return cls(LLMProviderConfig.from_env(env), client=client)

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
        try:
            topic = Topic.model_validate(topic)
            knowledge_window = KnowledgeWindow.model_validate(knowledge_window)
            state = MasteryState(mastery_state)
            messages = build_teacher_messages(
                topic=topic,
                knowledge_window=knowledge_window,
                mastery_state=state,
                remaining_minutes=remaining_minutes,
                attempt=attempt,
                question_override=question_override,
                scoring_points_override=scoring_points_override,
            )
            payload = self._request_json(
                messages,
                response_schema=_TEACHER_RESPONSE_SCHEMA,
                schema_name="studypilot_teaching_content",
            )
            if "scoring_points" not in payload and "rubric" in payload:
                payload["scoring_points"] = payload["rubric"]
            if "citations" not in payload and "citation_ids" in payload:
                payload["citations"] = payload["citation_ids"]
            raw_points = payload.get("scoring_points")
            if isinstance(raw_points, list):
                normalized_points: list[Any] = []
                for raw in raw_points:
                    if isinstance(raw, Mapping):
                        item = dict(raw)
                        if "citations" not in item and "citation_ids" in item:
                            item["citations"] = item["citation_ids"]
                        normalized_points.append(item)
                    else:
                        normalized_points.append(raw)
                payload["scoring_points"] = normalized_points
            content = TeachingContent.model_validate(payload)
            if question_override is not None:
                content = content.model_copy(update={"question": question_override})
            if scoring_points_override is not None:
                content = content.model_copy(
                    update={
                        "scoring_points": tuple(
                            ScoringPoint.model_validate(point)
                            for point in scoring_points_override
                        )
                    }
                )
            return validate_teaching_content_references(content, knowledge_window)
        except (LLMProviderError, ValidationError) as error:
            if isinstance(error, LLMProviderError):
                raise
            raise LLMResponseValidationError(
                "teacher JSON does not match the structured content schema"
            ) from error


class LLMEvaluatorProvider(_SDKProvider):
    """Return a validated, rubric-constrained structured judgement.

    The returned mastery/score/action are model suggestions.  The workflow
    must call :func:`adjudicate_evaluation` before applying them to state.
    """

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        client: Any | None = None,
    ) -> "LLMEvaluatorProvider":
        return cls(LLMProviderConfig.from_env(env), client=client)

    def evaluate(
        self,
        question: str = "",
        answer: str = "",
        scoring_points: Sequence[ScoringPoint] = (),
        *,
        rubric: Sequence[ScoringPoint] | None = None,
        allowed_citations: KnowledgeWindow | Sequence[Citation] | Sequence[str] | None = None,
    ) -> EvaluationResult:
        # Keep the tiny phase-A convenience form ``evaluate(answer, rubric)``
        # usable for the model-backed provider too.
        if not isinstance(answer, str):
            scoring_points = answer
            answer = question
            question = ""
        points = tuple(
            ScoringPoint.model_validate(point)
            for point in (rubric if rubric is not None else scoring_points)
        )
        if not points:
            raise LLMResponseValidationError("evaluator requires a non-empty rubric")
        messages = build_evaluator_messages(
            question=question,
            scoring_points=points,
            answer=answer,
            allowed_citations=allowed_citations,
        )
        payload = self._request_json(
            messages,
            response_schema=_EVALUATOR_RESPONSE_SCHEMA,
            schema_name="studypilot_evaluation",
        )
        try:
            return _parse_evaluator_payload(payload, points, allowed_citations)
        except _NextActionFormatError:
            # A single controlled retry is allowed only for an otherwise
            # valid response whose action was expressed as natural language.
            # The repair prompt forbids changing score/evidence/judgement; no
            # server-side mapping from prose to an action is ever performed.
            repaired_payload = self._request_json(
                _build_evaluator_format_repair_messages(messages, payload),
                response_schema=_EVALUATOR_RESPONSE_SCHEMA,
                schema_name="studypilot_evaluation",
            )
            try:
                return _parse_evaluator_payload(
                    repaired_payload,
                    points,
                    allowed_citations,
                )
            except _NextActionFormatError as error:
                raise LLMResponseValidationError(
                    "evaluator next_action remained outside the allowed enum "
                    "after one format repair"
                ) from error


def _get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


__all__ = [
    "LLMProviderConfig",
    "LLMConfig",
    "OpenAICompatibleConfig",
    "LLMProviderError",
    "LLMConfigurationError",
    "LLMRequestError",
    "LLMTimeoutError",
    "LLMEmptyResponseError",
    "LLMInvalidJSONError",
    "LLMResponseValidationError",
    "UnauthorizedCitationError",
    "LLMTeacherProvider",
    "LLMEvaluatorProvider",
    "adjudicate_evaluation",
    "build_teacher_messages",
    "build_evaluator_messages",
    "citation_ids",
    "validate_teaching_content_references",
    "validate_evaluation_references",
]

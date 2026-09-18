from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError, model_validator

from studypilot.application.parsing import ParseService
from studypilot.application.retrieval import BM25Retriever, CitationValidator, KnowledgeWindowBuilder
from studypilot.application.source_import import SourceImportService
from studypilot.application.material_organizer import CourseMaterialOrganizer
from studypilot.application.external_document_mcp import (
    ExternalDocumentMCPAdapter,
    ExternalDocumentMCPError,
)
from studypilot.domain.models import Course, EvidenceLevel, ExamGoal, MasteryState, Plan, Topic
from studypilot.domain.planner import PlanningError, RevisionPlanner
from studypilot.domain.knowledge import (
    Citation,
    CitationVerification,
    DocumentBlock,
    EvidenceRelation,
    KnowledgeWindow,
    ParserKind,
    SearchHit,
)
from studypilot.domain.sources import DocumentKind, SourceRecord, TrustLevel
from studypilot.infrastructure.sqlite_repository import NotFoundError, SQLiteRepository
from studypilot.domain.teaching import ScoringPoint, TeachingSessionState
from studypilot.domain.channel import ChannelBinding
from studypilot.application.teaching_service import (
    ExpectedVersionRequired,
    MessageIdConflict,
    SessionAlreadyExists,
    SessionProviderError,
    SessionStateConflict,
    SessionVersionConflict,
    TeachingService,
    TeachingServiceError,
)
from studypilot.application.teaching import Evaluator, TeacherProvider
from studypilot.application.llm import LLMTeacherProvider, LLMEvaluatorProvider
from studypilot.application.channel import ChannelBindingError, ChannelService


class ParseSourceRequest(BaseModel):
    parser_kind: ParserKind = ParserKind.TEXT
    extracted_text: str | None = None
    external: bool = False
    use_external_mcp: bool = False
    chapter: str | None = Field(default=None, min_length=1, max_length=80)

    @property
    def should_use_external_mcp(self) -> bool:
        return self.external or self.use_external_mcp


class ExternalParseRequest(BaseModel):
    chapter: str | None = Field(default=None, min_length=1, max_length=80)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)


class KnowledgeWindowRequest(BaseModel):
    query: str = Field(min_length=1)
    max_blocks: int = Field(default=5, ge=1, le=50)
    max_chars: int = Field(default=4000, ge=1, le=100_000)
    relation_by_block_id: dict[str, EvidenceRelation] = Field(default_factory=dict)


class CreateTeachingSessionRequest(BaseModel):
    goal_id: str = Field(min_length=1, max_length=100)
    session_id: str | None = Field(default=None, min_length=1, max_length=200)
    start: bool = False
    auto_start: bool | None = None
    question: str | None = Field(default=None, max_length=10_000)
    scoring_points: list[ScoringPoint] | None = None

    @property
    def should_start(self) -> bool:
        return self.start if self.auto_start is None else self.auto_start


class CreateTeachingSessionWithoutPathRequest(CreateTeachingSessionRequest):
    course_id: str = Field(min_length=1, max_length=100)


class SubmitTeachingAnswerRequest(BaseModel):
    answer: str = Field(default="", max_length=100_000)
    message_id: str | None = Field(default=None, min_length=1, max_length=200)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)
    answer_id: str | None = Field(default=None, min_length=1, max_length=200)
    expected_version: int | None = Field(default=None, ge=0)
    # ``version`` is accepted as a compatibility spelling, but the service
    # still treats it as an optimistic-concurrency token.
    version: int | None = Field(default=None, ge=0)
    spent_minutes: int = Field(default=0, ge=0, le=100_000)
    elapsed_minutes: int | None = Field(default=None, ge=0, le=100_000)

    @model_validator(mode="after")
    def require_message_and_version(self) -> "SubmitTeachingAnswerRequest":
        if self.expected_version is None and self.version is None:
            raise ValueError("expected_version is required")
        if (
            self.expected_version is not None
            and self.version is not None
            and self.expected_version != self.version
        ):
            raise ValueError("expected_version and version must match")
        return self

    @property
    def resolved_message_id(self) -> str:
        return self.message_id or self.idempotency_key or self.answer_id  # type: ignore[return-value]

    @property
    def resolved_version(self) -> int:
        if self.expected_version is not None:
            return self.expected_version
        return self.version  # type: ignore[return-value]

    @property
    def resolved_spent_minutes(self) -> int:
        return self.spent_minutes if self.elapsed_minutes is None else self.elapsed_minutes


class ChannelBindingRequest(BaseModel):
    external_user_id: str | None = Field(default=None, min_length=1, max_length=400)
    openid: str | None = Field(default=None, min_length=1, max_length=400)
    learner_id: str = Field(min_length=1, max_length=200)
    course_id: str = Field(min_length=1, max_length=100)
    session_id: str | None = Field(default=None, min_length=1, max_length=200)
    channel: str = Field(default="qq_c2c", min_length=1, max_length=80)

    @model_validator(mode="after")
    def require_user_id(self) -> "ChannelBindingRequest":
        if self.external_user_id is None and self.openid is None:
            raise ValueError("external_user_id or openid is required")
        if (
            self.external_user_id is not None
            and self.openid is not None
            and self.external_user_id != self.openid
        ):
            raise ValueError("external_user_id and openid must match")
        return self

    @property
    def resolved_external_user_id(self) -> str:
        return self.external_user_id or self.openid  # type: ignore[return-value]


def create_app(
    database_path: str | Path = "studypilot.db",
    storage_root: str | Path = "data",
    *,
    teacher_provider: TeacherProvider | None = None,
    evaluator: Evaluator | None = None,
    document_mcp_adapter: ExternalDocumentMCPAdapter | None = None,
    external_document_mcp: ExternalDocumentMCPAdapter | None = None,
) -> FastAPI:
    if document_mcp_adapter is not None and external_document_mcp is not None:
        raise ValueError("provide only one external document MCP adapter")
    document_mcp = document_mcp_adapter or external_document_mcp
    repository = SQLiteRepository(database_path)
    planner = RevisionPlanner()
    source_importer = SourceImportService(repository, storage_root)
    parse_service = ParseService(repository)
    material_organizer = CourseMaterialOrganizer(
        repository, Path(storage_root) / "organized"
    )
    retriever = BM25Retriever(repository)
    window_builder = KnowledgeWindowBuilder(retriever)
    citation_validator = CitationValidator(repository)
    teaching_service = TeachingService(
        repository,
        teacher_provider=teacher_provider,
        evaluator=evaluator,
        knowledge_window_builder=window_builder,
    )
    channel_service = ChannelService(repository, teaching_service)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Keep module import side-effect free; storage is created when the service starts.
        repository.initialize()
        try:
            yield
        finally:
            teaching_service.close()

    app = FastAPI(
        title="StudyPilot",
        version="0.1.0",
        description="Time-constrained revision planner MVP",
        lifespan=lifespan,
    )
    template_root = Path(__file__).resolve().parent / "templates"
    static_root = Path(__file__).resolve().parent / "static"
    templates = Jinja2Templates(directory=str(template_root))
    app.mount("/static", StaticFiles(directory=str(static_root)), name="static")
    app.state.repository = repository
    app.state.teaching_service = teaching_service
    app.state.channel_service = channel_service
    app.state.document_mcp_adapter = document_mcp

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.put("/channel-bindings", response_model=ChannelBinding)
    def put_channel_binding(request: ChannelBindingRequest) -> ChannelBinding:
        try:
            return channel_service.bind_user(
                external_user_id=request.resolved_external_user_id,
                learner_id=request.learner_id,
                course_id=request.course_id,
                session_id=request.session_id,
                channel=request.channel,
            )
        except NotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        except (ChannelBindingError, ValueError) as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error

    @app.get(
        "/channel-bindings/{channel}/{external_user_id}",
        response_model=ChannelBinding,
    )
    def get_channel_binding(channel: str, external_user_id: str) -> ChannelBinding:
        binding = repository.get_channel_binding(channel, external_user_id)
        if binding is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "channel binding was not found")
        return binding

    @app.delete("/channel-bindings/{channel}/{external_user_id}", status_code=204)
    def delete_channel_binding(channel: str, external_user_id: str) -> Response:
        if not repository.delete_channel_binding(channel, external_user_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "channel binding was not found")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.put("/courses/{course_id}", response_model=Course)
    def put_course(course_id: str, course: Course) -> Course:
        if course.id != course_id:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "path and body ids differ")
        repository.save_course(course)
        return course

    @app.put("/courses/{course_id}/topics", response_model=list[Topic])
    def put_topics(course_id: str, topics: list[Topic]) -> list[Topic]:
        try:
            repository.save_topics(course_id, topics)
        except NotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        except ValueError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error
        return topics

    @app.put("/courses/{course_id}/goals/{goal_id}", response_model=ExamGoal)
    def put_goal(course_id: str, goal_id: str, goal: ExamGoal) -> ExamGoal:
        if goal.id != goal_id or goal.course_id != course_id:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "path and body ids differ")
        try:
            repository.save_goal(goal)
        except NotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        return goal

    @app.post("/goals/{goal_id}/plan", response_model=Plan)
    def generate_plan(goal_id: str) -> Plan:
        try:
            goal = repository.get_goal(goal_id)
            topics = repository.list_topics(goal.course_id)
            plan = planner.build(goal, topics)
            repository.save_plan(plan)
            return plan
        except NotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        except PlanningError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error

    @app.get("/goals/{goal_id}/plan", response_model=Plan)
    def get_plan(goal_id: str) -> Plan:
        try:
            return repository.get_plan(goal_id)
        except NotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error

    @app.post(
        "/courses/{course_id}/sources",
        response_model=SourceRecord,
        status_code=status.HTTP_201_CREATED,
    )
    def upload_source(
        course_id: str,
        file: UploadFile = File(),
        document_kind: DocumentKind = Form(),
        trust_level: TrustLevel = Form(),
        origin: str | None = Form(default=None),
        metadata: str = Form(default="{}"),
    ) -> SourceRecord:
        try:
            metadata_value = json.loads(metadata)
            if not isinstance(metadata_value, dict):
                raise ValueError("metadata must be a JSON object")
            return source_importer.import_file(
                course_id=course_id,
                stream=file.file,
                display_name=file.filename or "unnamed",
                document_kind=document_kind,
                trust_level=trust_level,
                origin=origin,
                metadata=metadata_value,
            )
        except NotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        except (json.JSONDecodeError, ValueError, ValidationError) as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error

    @app.get("/courses/{course_id}/sources", response_model=list[SourceRecord])
    def list_sources(course_id: str) -> list[SourceRecord]:
        try:
            return repository.list_sources(course_id)
        except NotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error

    @app.get("/sources/{source_id}", response_model=SourceRecord)
    def get_source(source_id: str) -> SourceRecord:
        try:
            return repository.get_source(source_id)
        except NotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error

    @app.post("/sources/{source_id}/parse", response_model=list[DocumentBlock])
    async def parse_source(source_id: str, request: ParseSourceRequest) -> list[DocumentBlock]:
        try:
            if request.should_use_external_mcp:
                if document_mcp is None:
                    raise ExternalDocumentMCPError(
                        "external document MCP is not configured for this app"
                    )
                blocks = await document_mcp.parse_source(
                    source_id,
                    chapter=request.chapter,
                )
            else:
                blocks = parse_service.parse_source(
                    source_id, request.parser_kind, request.extracted_text
                )
            source = repository.get_source(source_id)
            material_organizer.refresh(source.asset.course_id)
            return blocks
        except NotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        except ExternalDocumentMCPError as error:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error
        except ValueError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error

    @app.post("/sources/{source_id}/parse-external", response_model=list[DocumentBlock])
    async def parse_source_external(
        source_id: str,
        request: ExternalParseRequest,
    ) -> list[DocumentBlock]:
        if document_mcp is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "external document MCP is not configured for this app",
            )
        try:
            blocks = await document_mcp.parse_source(source_id, chapter=request.chapter)
            source = repository.get_source(source_id)
            material_organizer.refresh(source.asset.course_id)
            return blocks
        except NotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        except ExternalDocumentMCPError as error:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error

    @app.get("/blocks/{block_id}", response_model=DocumentBlock)
    def get_block(block_id: str) -> DocumentBlock:
        try:
            return repository.get_block(block_id)
        except NotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error

    @app.get(
        "/courses/{course_id}/materials/markdown",
        response_class=Response,
        responses={200: {"content": {"text/markdown": {}}}},
    )
    def get_course_material_markdown(course_id: str) -> Response:
        try:
            markdown = material_organizer.render(course_id)
            material_organizer.refresh(course_id)
            return Response(content=markdown, media_type="text/markdown; charset=utf-8")
        except NotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error

    @app.post("/courses/{course_id}/search", response_model=list[SearchHit])
    def search(course_id: str, request: SearchRequest) -> list[SearchHit]:
        try:
            return retriever.search(course_id, request.query, request.limit)
        except NotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error

    @app.post("/courses/{course_id}/knowledge-window", response_model=KnowledgeWindow)
    def build_knowledge_window(
        course_id: str, request: KnowledgeWindowRequest
    ) -> KnowledgeWindow:
        try:
            return window_builder.build(
                course_id=course_id,
                query=request.query,
                max_blocks=request.max_blocks,
                max_chars=request.max_chars,
                relation_by_block_id=request.relation_by_block_id,
            )
        except NotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        except ValueError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error

    @app.post("/citations/verify", response_model=CitationVerification)
    def verify_citation(citation: Citation) -> CitationVerification:
        return citation_validator.verify(citation)

    @app.post(
        "/courses/{course_id}/teaching-sessions",
        response_model=TeachingSessionState,
        status_code=status.HTTP_201_CREATED,
    )
    @app.post(
        "/courses/{course_id}/sessions",
        response_model=TeachingSessionState,
        status_code=status.HTTP_201_CREATED,
        include_in_schema=False,
    )
    def create_teaching_session(
        course_id: str, request: CreateTeachingSessionRequest
    ) -> TeachingSessionState:
        try:
            return teaching_service.create_session(
                course_id=course_id,
                goal_id=request.goal_id,
                session_id=request.session_id,
                start=request.should_start,
                question=request.question,
                scoring_points=request.scoring_points,
            )
        except NotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        except SessionAlreadyExists as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
        except SessionProviderError as error:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from error
        except TeachingServiceError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error

    @app.post(
        "/teaching-sessions",
        response_model=TeachingSessionState,
        status_code=status.HTTP_201_CREATED,
        include_in_schema=False,
    )
    @app.post(
        "/sessions",
        response_model=TeachingSessionState,
        status_code=status.HTTP_201_CREATED,
        include_in_schema=False,
    )
    def create_teaching_session_without_path(
        request: CreateTeachingSessionWithoutPathRequest,
    ) -> TeachingSessionState:
        return create_teaching_session(request.course_id, request)

    @app.post(
        "/teaching-sessions/{session_id}/start",
        response_model=TeachingSessionState,
    )
    @app.post(
        "/sessions/{session_id}/start",
        response_model=TeachingSessionState,
        include_in_schema=False,
    )
    def start_teaching_session(session_id: str) -> TeachingSessionState:
        try:
            return teaching_service.start_session(session_id)
        except NotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        except SessionProviderError as error:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from error
        except SessionStateConflict as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error

    @app.get(
        "/teaching-sessions/{session_id}",
        response_model=TeachingSessionState,
    )
    @app.get(
        "/teaching-sessions/{session_id}/state",
        response_model=TeachingSessionState,
        include_in_schema=False,
    )
    @app.get(
        "/sessions/{session_id}",
        response_model=TeachingSessionState,
        include_in_schema=False,
    )
    def get_teaching_session(session_id: str) -> TeachingSessionState:
        try:
            return teaching_service.get_state(session_id)
        except NotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        except SessionStateConflict as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error

    @app.post(
        "/teaching-sessions/{session_id}/answer",
        response_model=TeachingSessionState,
    )
    @app.post(
        "/teaching-sessions/{session_id}/resume",
        response_model=TeachingSessionState,
        include_in_schema=False,
    )
    @app.post(
        "/sessions/{session_id}/answer",
        response_model=TeachingSessionState,
        include_in_schema=False,
    )
    def submit_teaching_answer(
        session_id: str,
        request: SubmitTeachingAnswerRequest,
        idempotency_key_header: str | None = Header(
            default=None, alias="Idempotency-Key"
        ),
    ) -> TeachingSessionState:
        try:
            message_id = request.resolved_message_id or idempotency_key_header
            if not message_id:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "message_id or idempotency_key is required",
                )
            return teaching_service.submit_answer(
                session_id,
                message_id=message_id,
                answer=request.answer,
                expected_version=request.resolved_version,
                spent_minutes=request.resolved_spent_minutes,
            )
        except NotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        except (
            ExpectedVersionRequired,
            SessionVersionConflict,
            SessionStateConflict,
            MessageIdConflict,
        ) as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
        except SessionProviderError as error:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from error
        except TeachingServiceError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error

    def _study_redirect(**values: object) -> RedirectResponse:
        params = {
            key: str(value)
            for key, value in values.items()
            if value is not None and str(value) != ""
        }
        query = urlencode(params)
        return RedirectResponse(
            url="/study" + (f"?{query}" if query else ""),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    def _study_error(error: Exception) -> str:
        if isinstance(error, ExternalDocumentMCPError):
            code = 503
            message = f"外部文档识别 MCP 暂不可用：{error}"
        elif isinstance(error, SessionProviderError):
            code = 503
            message = "模型服务暂不可用，请检查 provider 配置或稍后重试"
        elif isinstance(
            error,
            (
                SessionVersionConflict,
                SessionStateConflict,
                MessageIdConflict,
                SessionAlreadyExists,
            ),
        ):
            code = 409
            message = str(error)
        elif isinstance(error, (ValidationError, ValueError, PlanningError)):
            code = 422
            message = str(error)
        else:
            code = 500
            message = "操作失败，请查看本地服务日志"
        return f"[{code}] {message}"

    def _study_provider_status() -> dict[str, str | bool]:
        configured = bool(
            (os.getenv("STUDYPILOT_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
            and (os.getenv("STUDYPILOT_LLM_MODEL") or os.getenv("OPENAI_MODEL"))
        )
        injected_llm = isinstance(
            teaching_service.teacher_provider, LLMTeacherProvider
        ) or isinstance(teaching_service.evaluator, LLMEvaluatorProvider)
        if injected_llm:
            label = "LLM provider（已通过依赖注入）"
        else:
            label = "Deterministic demo（默认）"
        return {
            "label": label,
            "llm_configured": configured,
            "active_llm": injected_llm,
            "llm_hint": (
                "已检测到 LLM 环境配置，可在 create_app 中注入 provider"
                if configured
                else "未配置 key/model；当前不会调用真实模型"
            ),
        }

    def _study_context(
        *,
        course_id: str | None,
        goal_id: str | None,
        session_id: str | None,
        notice: str | None = None,
        error: str | None = None,
    ) -> dict[str, object]:
        courses = repository.list_courses()
        selected_course = None
        selected_course_id = course_id
        if selected_course_id:
            try:
                selected_course = repository.get_course(selected_course_id)
            except NotFoundError as missing:
                error = _study_error(missing)
                selected_course_id = None
        if selected_course is None and not selected_course_id and courses:
            selected_course = courses[0]
            selected_course_id = selected_course.id

        goals = repository.list_goals(selected_course_id) if selected_course_id else []
        selected_goal_id = goal_id if any(item.id == goal_id for item in goals) else None
        if selected_goal_id is None and goals:
            selected_goal_id = goals[0].id
        topics = repository.list_topics(selected_course_id) if selected_course_id else []
        sources = repository.list_sources(selected_course_id) if selected_course_id else []
        sessions = (
            repository.list_teaching_sessions(selected_course_id)
            if selected_course_id
            else []
        )
        plan = None
        if selected_goal_id:
            try:
                plan = repository.get_plan(selected_goal_id)
            except NotFoundError:
                plan = None

        teaching_state = None
        if session_id:
            try:
                teaching_state = teaching_service.get_state(session_id)
            except NotFoundError as missing:
                error = _study_error(missing)

        topic_lines = "\n".join(
            "|".join(
                (
                    topic.id,
                    topic.name,
                    f"{topic.exam_points:g}",
                    str(topic.learning_minutes),
                    topic.mastery.value,
                    ",".join(topic.prerequisite_ids),
                )
            )
            for topic in topics
        )
        selected_goal = next(
            (item for item in goals if item.id == selected_goal_id), None
        )
        return {
            "courses": courses,
            "selected_course": selected_course,
            "selected_course_id": selected_course_id,
            "goals": goals,
            "selected_goal": selected_goal,
            "selected_goal_id": selected_goal_id,
            "topics": topics,
            "topic_lines": topic_lines,
            "plan": plan,
            "sources": sources,
            "sessions": sessions,
            "teaching_state": teaching_state,
            "selected_session_id": session_id,
            "notice": notice,
            "error": error,
            "provider": _study_provider_status(),
            "document_mcp": {
                "configured": document_mcp is not None,
                "label": (
                    "外部文档识别 MCP 已接入"
                    if document_mcp is not None
                    else "外部文档识别 MCP 未接入（MD/TXT 仍可本地解析）"
                ),
            },
        }

    @app.get("/", include_in_schema=False)
    def study_landing() -> RedirectResponse:
        return RedirectResponse(url="/study", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

    @app.get("/study", response_class=HTMLResponse, include_in_schema=False)
    def study_page(
        request: Request,
        course_id: str | None = None,
        goal_id: str | None = None,
        session_id: str | None = None,
        notice: str | None = None,
        error: str | None = None,
    ) -> HTMLResponse:
        context = _study_context(
            course_id=course_id,
            goal_id=goal_id,
            session_id=session_id,
            notice=notice,
            error=error,
        )
        context["request"] = request
        return templates.TemplateResponse(request, "study.html", context)

    @app.post("/study/courses", include_in_schema=False)
    def study_create_course(
        course_id: str = Form(), name: str = Form()
    ) -> RedirectResponse:
        try:
            repository.save_course(Course(id=course_id.strip(), name=name.strip()))
            return _study_redirect(course_id=course_id.strip(), notice="课程已保存")
        except (ValidationError, ValueError) as error:
            return _study_redirect(error=_study_error(error))

    @app.post("/study/courses/{course_id}/goals", include_in_schema=False)
    def study_create_goal(
        course_id: str,
        goal_id: str = Form(),
        exam_at: str = Form(),
        available_minutes: str = Form(),
    ) -> RedirectResponse:
        try:
            goal = ExamGoal(
                id=goal_id.strip(),
                course_id=course_id,
                exam_at=exam_at.strip(),
                available_minutes=int(available_minutes),
            )
            repository.save_goal(goal)
            return _study_redirect(
                course_id=course_id, goal_id=goal.id, notice="考试目标已保存"
            )
        except (ValidationError, ValueError) as error:
            return _study_redirect(course_id=course_id, error=_study_error(error))

    @app.post("/study/courses/{course_id}/topics", include_in_schema=False)
    def study_save_topics(course_id: str, topics_text: str = Form()) -> RedirectResponse:
        try:
            topics: list[Topic] = []
            for line_number, raw_line in enumerate(topics_text.splitlines(), start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = [value.strip() for value in line.split("|")]
                if len(fields) < 4:
                    raise ValueError(
                        f"第 {line_number} 行格式错误：应为 id|名称|分值|学习分钟|掌握状态|前置id"
                    )
                mastery = (
                    MasteryState(fields[4])
                    if len(fields) > 4 and fields[4]
                    else MasteryState.UNSEEN
                )
                prerequisites = (
                    tuple(value.strip() for value in fields[5].split(",") if value.strip())
                    if len(fields) > 5 and fields[5]
                    else ()
                )
                topics.append(
                    Topic(
                        id=fields[0],
                        course_id=course_id,
                        name=fields[1],
                        exam_points=float(fields[2]),
                        learning_minutes=int(fields[3]),
                        mastery=mastery,
                        evidence_level=EvidenceLevel.COURSE_MATERIAL,
                        evidence_confidence=1.0,
                        prerequisite_ids=prerequisites,
                    )
                )
            if not topics:
                raise ValueError("至少录入一个考点")
            repository.save_topics(course_id, topics)
            return _study_redirect(course_id=course_id, notice="考点已保存")
        except (NotFoundError, ValidationError, ValueError) as error:
            return _study_redirect(course_id=course_id, error=_study_error(error))

    @app.post("/study/courses/{course_id}/plan", include_in_schema=False)
    def study_generate_plan(
        course_id: str, goal_id: str = Form()
    ) -> RedirectResponse:
        try:
            goal = repository.get_goal(goal_id)
            if goal.course_id != course_id:
                raise ValueError("目标不属于当前课程")
            plan = planner.build(goal, repository.list_topics(course_id))
            repository.save_plan(plan)
            return _study_redirect(
                course_id=course_id, goal_id=goal_id, notice="复习计划已生成"
            )
        except (NotFoundError, PlanningError, ValidationError, ValueError) as error:
            return _study_redirect(
                course_id=course_id, goal_id=goal_id, error=_study_error(error)
            )

    @app.post("/study/courses/{course_id}/sources", include_in_schema=False)
    def study_upload_source(
        course_id: str,
        file: UploadFile = File(),
        document_kind: DocumentKind = Form(DocumentKind.COURSE_MATERIAL),
        trust_level: TrustLevel = Form(TrustLevel.MEDIUM),
        origin: str | None = Form(default=None),
    ) -> RedirectResponse:
        try:
            source_importer.import_file(
                course_id=course_id,
                stream=file.file,
                display_name=file.filename or "unnamed",
                document_kind=document_kind,
                trust_level=trust_level,
                origin=origin or None,
                metadata={},
            )
            return _study_redirect(course_id=course_id, notice="资料已上传")
        except (NotFoundError, ValidationError, ValueError, OSError) as error:
            return _study_redirect(course_id=course_id, error=_study_error(error))

    @app.post("/study/sources/{source_id}/parse", include_in_schema=False)
    async def study_parse_source(
        source_id: str,
        parser_kind: ParserKind = Form(ParserKind.TEXT),
        extracted_text: str | None = Form(default=None),
        external: bool = Form(default=False),
        chapter: str | None = Form(default=None),
    ) -> RedirectResponse:
        try:
            source = repository.get_source(source_id)
            if external:
                if document_mcp is None:
                    raise ExternalDocumentMCPError(
                        "external document MCP is not configured for this app"
                    )
                await document_mcp.parse_source(source_id, chapter=chapter)
            else:
                parse_service.parse_source(source_id, parser_kind, extracted_text or None)
            material_organizer.refresh(source.asset.course_id)
            return _study_redirect(
                course_id=source.asset.course_id,
                notice="资料已解析，并已更新可读课程资料",
            )
        except (NotFoundError, ValidationError, ValueError, OSError, ExternalDocumentMCPError) as error:
            return _study_redirect(error=_study_error(error))

    @app.post("/study/courses/{course_id}/teaching-sessions", include_in_schema=False)
    def study_create_session(
        course_id: str,
        goal_id: str = Form(),
        session_id: str | None = Form(default=None),
        start: bool = Form(default=False),
    ) -> RedirectResponse:
        try:
            state = teaching_service.create_session(
                course_id=course_id,
                goal_id=goal_id,
                session_id=session_id.strip() if session_id else None,
                start=start,
            )
            return _study_redirect(
                course_id=course_id,
                goal_id=goal_id,
                session_id=state.session_id,
                notice="教学 Session 已创建" + ("并启动" if start else ""),
            )
        except (NotFoundError, SessionAlreadyExists, SessionProviderError, TeachingServiceError) as error:
            return _study_redirect(
                course_id=course_id, goal_id=goal_id, error=_study_error(error)
            )

    @app.post("/study/sessions/{session_id}/start", include_in_schema=False)
    def study_start_session(session_id: str) -> RedirectResponse:
        try:
            state = teaching_service.start_session(session_id)
            return _study_redirect(
                course_id=state.course_id,
                goal_id=state.goal_id,
                session_id=session_id,
                notice="教学 Session 已启动",
            )
        except (NotFoundError, SessionProviderError, TeachingServiceError) as error:
            return _study_redirect(session_id=session_id, error=_study_error(error))

    @app.post("/study/sessions/{session_id}/answer", include_in_schema=False)
    def study_submit_answer(
        session_id: str,
        answer: str = Form(default=""),
        message_id: str | None = Form(default=None),
        expected_version: str | None = Form(default=None),
        spent_minutes: str = Form(default="0"),
    ) -> RedirectResponse:
        try:
            if not message_id or not message_id.strip():
                raise ValueError("message_id 不能为空")
            if expected_version is None or not expected_version.strip():
                raise ValueError("expected_version 不能为空")
            state = teaching_service.submit_answer(
                session_id,
                message_id=message_id.strip(),
                answer=answer,
                expected_version=int(expected_version),
                spent_minutes=int(spent_minutes or 0),
            )
            return _study_redirect(
                course_id=state.course_id,
                goal_id=state.goal_id,
                session_id=session_id,
                notice="答案已提交并完成评价",
            )
        except (NotFoundError, SessionProviderError, TeachingServiceError, ValidationError, ValueError) as error:
            return _study_redirect(session_id=session_id, error=_study_error(error))

    return app


app = create_app()

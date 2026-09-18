"""External document-recognition MCP client boundary.

StudyPilot owns source/blob identity and block persistence.  This module only
adapts the public tools exposed by an external MCP server; it does not expose
an MCP server of its own and does not implement Office/PDF OCR.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from studypilot.application.parsing import ParsedChunk, ParseService
from studypilot.domain.knowledge import DocumentBlock, ParserKind
from studypilot.domain.sources import ParseStatus
from studypilot.infrastructure.sqlite_repository import SQLiteRepository


class ExternalDocumentMCPError(RuntimeError):
    """Base class for an explicit external document parser failure."""


class ExternalDocumentMCPConfigurationError(ExternalDocumentMCPError, ValueError):
    """The external MCP client has invalid or incomplete runtime settings."""


class ExternalDocumentMCPUnavailableError(ExternalDocumentMCPError):
    """The external MCP endpoint could not be reached or initialized."""


class ExternalDocumentMCPTimeoutError(ExternalDocumentMCPUnavailableError):
    """The external MCP operation exceeded its configured timeout."""


class ExternalDocumentMCPToolError(ExternalDocumentMCPError):
    """The MCP server reported a tool-level failure."""


class ExternalDocumentMCPContentError(ExternalDocumentMCPError):
    """The MCP response was empty, truncated, malformed, or unsafe to read."""


@dataclass(frozen=True)
class ExternalDocumentMCPConfig:
    """Runtime configuration for the external document MCP endpoint."""

    url: str = "http://127.0.0.1:3001/mcp"
    timeout_seconds: float = 300.0
    parse_document_tool: str = "parse_document"
    parse_pdf_tool: str = "parse_pdf"
    get_study_section_tool: str = "get_study_section"
    # The external server must be configured to allow this directory.  When
    # omitted, the source blob directory is used, which is useful for a local
    # same-machine server but remains explicit rather than hard-coding a path.
    staging_root: str | Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or not self.url.strip():
            raise ExternalDocumentMCPConfigurationError("MCP URL must be non-empty")
        if self.timeout_seconds <= 0 or self.timeout_seconds > 3_600:
            raise ExternalDocumentMCPConfigurationError(
                "MCP timeout_seconds must be greater than 0 and at most 3600"
            )
        for field_name in (
            "parse_document_tool",
            "parse_pdf_tool",
            "get_study_section_tool",
        ):
            if not isinstance(getattr(self, field_name), str) or not getattr(
                self, field_name
            ).strip():
                raise ExternalDocumentMCPConfigurationError(
                    f"{field_name} must be non-empty"
                )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ExternalDocumentMCPConfig":
        """Read endpoint and tool names without embedding deployment secrets."""

        source = os.environ if env is None else env

        def first(name: str, default: str | None = None) -> str | None:
            value = source.get(name)
            if value is None or not value.strip():
                return default
            return value.strip()

        raw_timeout = first("STUDYPILOT_DOCUMENT_MCP_TIMEOUT_SECONDS", "300")
        try:
            timeout = float(raw_timeout or "300")
        except ValueError as error:
            raise ExternalDocumentMCPConfigurationError(
                "STUDYPILOT_DOCUMENT_MCP_TIMEOUT_SECONDS must be numeric"
            ) from error
        return cls(
            url=first("STUDYPILOT_DOCUMENT_MCP_URL", cls.url) or cls.url,
            timeout_seconds=timeout,
            parse_document_tool=first(
                "STUDYPILOT_DOCUMENT_MCP_PARSE_DOCUMENT_TOOL", cls.parse_document_tool
            )
            or cls.parse_document_tool,
            parse_pdf_tool=first(
                "STUDYPILOT_DOCUMENT_MCP_PARSE_PDF_TOOL", cls.parse_pdf_tool
            )
            or cls.parse_pdf_tool,
            get_study_section_tool=first(
                "STUDYPILOT_DOCUMENT_MCP_GET_STUDY_SECTION_TOOL",
                cls.get_study_section_tool,
            )
            or cls.get_study_section_tool,
            staging_root=first("STUDYPILOT_DOCUMENT_MCP_STAGING_ROOT"),
        )

    from_environment = from_env


class MCPToolClient(Protocol):
    """Small injectable boundary used by tests and the official SDK client."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


class OfficialMCPToolClient:
    """Call one tool through the official MCP Python SDK v2 Client.

    A short-lived SDK client per tool call keeps lifecycle ownership local and
    avoids keeping a stale Streamable HTTP session across a long import.  The
    configured timeout wraps initialization and the tool call together.
    """

    def __init__(self, config: ExternalDocumentMCPConfig) -> None:
        self.config = config

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        async def invoke() -> Any:
            try:
                from mcp import Client
            except ImportError as error:  # pragma: no cover - dependency install path
                raise ExternalDocumentMCPConfigurationError(
                    "official MCP Python SDK is not installed; install the mcp dependency"
                ) from error

            async with Client(
                self.config.url,
                raise_exceptions=True,
                read_timeout_seconds=self.config.timeout_seconds,
            ) as client:
                return await client.call_tool(
                    name,
                    arguments,
                    read_timeout_seconds=self.config.timeout_seconds,
                )

        try:
            return await asyncio.wait_for(invoke(), self.config.timeout_seconds)
        except asyncio.TimeoutError as error:
            raise ExternalDocumentMCPTimeoutError(
                "external document MCP operation timed out"
            ) from error
        except ExternalDocumentMCPError:
            raise
        except Exception as error:
            raise ExternalDocumentMCPUnavailableError(
                "external document MCP endpoint is unavailable"
            ) from error


OFFICE_EXTENSIONS = frozenset({".ppt", ".pptx", ".doc", ".docx", ".xls", ".xlsx"})
PDF_EXTENSIONS = frozenset({".pdf"})
MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown"})
TEXT_EXTENSIONS = frozenset({".txt"})

OFFICE_PARSER_VERSION = "external-mcp:parse_document:v1"
PDF_PARSER_VERSION = "external-mcp:parse_pdf+get_study_section:v1"


def _field(value: Any, *names: str) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        text = content.get("text")
        if isinstance(text, str):
            return text
        return ""
    text = getattr(content, "text", None)
    return text if isinstance(text, str) else ""


def _result_value(result: Any) -> Any:
    """Extract structured or textual content from an official CallToolResult."""

    if isinstance(result, (str, bytes, Mapping)):
        if isinstance(result, Mapping) and any(
            key in result for key in ("content", "structured_content", "structuredContent")
        ):
            structured = _field(result, "structured_content", "structuredContent")
            if structured is not None:
                return structured
            content = result.get("content")
        else:
            return result.decode("utf-8") if isinstance(result, bytes) else result
    else:
        structured = _field(result, "structured_content", "structuredContent")
        if structured is not None:
            return structured
        content = _field(result, "content")

    if content is None:
        raise ExternalDocumentMCPContentError("MCP tool returned no content")
    if isinstance(content, (str, bytes)):
        return content.decode("utf-8") if isinstance(content, bytes) else content
    if not isinstance(content, Sequence):
        raise ExternalDocumentMCPContentError("MCP tool content is not a text sequence")
    parts = [_text_from_content(item) for item in content]
    text = "\n".join(part for part in parts if part)
    if not text:
        raise ExternalDocumentMCPContentError("MCP tool returned no text content")
    return text


def _raise_tool_error(value: Any) -> None:
    is_error = _field(value, "is_error", "isError")
    if is_error:
        raise ExternalDocumentMCPToolError("external document MCP tool reported an error")
    if isinstance(value, Mapping):
        if value.get("success") is False:
            detail = value.get("error") or "tool returned success=false"
            raise ExternalDocumentMCPToolError(f"external document MCP tool failed: {detail}")
        if value.get("error"):
            raise ExternalDocumentMCPToolError(
                f"external document MCP tool failed: {value['error']}"
            )


def _json_object(result: Any) -> dict[str, Any]:
    """Decode a JSON object returned by parse_document/parse_pdf."""

    _raise_tool_error(result)
    value = _result_value(result)
    if isinstance(value, Mapping):
        payload = dict(value)
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise ExternalDocumentMCPContentError(
                "MCP parse tool did not return valid JSON"
            ) from error
        if not isinstance(parsed, Mapping):
            raise ExternalDocumentMCPContentError("MCP parse tool JSON must be an object")
        payload = dict(parsed)
    else:
        raise ExternalDocumentMCPContentError("MCP parse tool returned an unsupported value")
    _raise_tool_error(payload)
    return payload


def _full_char_length(value: str) -> int:
    """Match JavaScript ``String.length`` used by the external server."""

    return len(value.encode("utf-16-le")) // 2


class ExternalDocumentMCPAdapter:
    """Route an imported source through local or external document parsing."""

    def __init__(
        self,
        repository: SQLiteRepository,
        parse_service: ParseService,
        config: ExternalDocumentMCPConfig | None = None,
        *,
        client: MCPToolClient | None = None,
    ) -> None:
        self.repository = repository
        self.parse_service = parse_service
        self.config = config or ExternalDocumentMCPConfig.from_env()
        self.client = client or OfficialMCPToolClient(self.config)

    async def parse_source(
        self,
        source_asset_id: str,
        *,
        chapter: str | None = None,
    ) -> list[DocumentBlock]:
        """Parse one source, reusing an exact cached blob/parser run first."""

        source = self.repository.get_source(source_asset_id)
        suffix = Path(source.asset.display_name).suffix.lower()
        try:
            parser_kind, parser_version = self._route(suffix)
        except ExternalDocumentMCPError:
            self.repository.mark_parse_failed(source_asset_id)
            raise

        cached = self._cached_blocks(source, parser_kind, parser_version)
        if cached:
            return cached

        if suffix in MARKDOWN_EXTENSIONS or suffix in TEXT_EXTENSIONS:
            try:
                blocks = self.parse_service.parse_source(
                    source_asset_id,
                    parser_kind,
                )
                if not blocks:
                    self.repository.mark_parse_failed(source_asset_id)
                    raise ValueError("local parser produced no document blocks")
                return blocks
            except Exception:
                self.repository.mark_parse_failed(source_asset_id)
                raise

        try:
            text = await self._fetch_external_text(source, suffix, chapter=chapter)
            if not text.strip():
                raise ExternalDocumentMCPContentError(
                    "external MCP returned empty document text"
                )
            chunks = self.parse_service.parsers[parser_kind].parse(text)
            if not chunks:
                raise ExternalDocumentMCPContentError(
                    "external MCP text produced no document chunks"
                )
            return self.parse_service.persist_chunks(
                source_asset_id,
                parser_kind,
                chunks,
                parser_version=parser_version,
            )
        except ExternalDocumentMCPError:
            self.repository.mark_parse_failed(source_asset_id)
            raise
        except Exception as error:
            self.repository.mark_parse_failed(source_asset_id)
            raise ExternalDocumentMCPContentError(
                "external document content could not be materialised"
            ) from error

    def _route(self, suffix: str) -> tuple[ParserKind, str]:
        if suffix in OFFICE_EXTENSIONS:
            return ParserKind.MARKDOWN, OFFICE_PARSER_VERSION
        if suffix in PDF_EXTENSIONS:
            return ParserKind.TEXT, PDF_PARSER_VERSION
        if suffix in MARKDOWN_EXTENSIONS:
            return ParserKind.MARKDOWN, self.parse_service.parsers[ParserKind.MARKDOWN].version
        if suffix in TEXT_EXTENSIONS:
            return ParserKind.TEXT, self.parse_service.parsers[ParserKind.TEXT].version
        raise ExternalDocumentMCPContentError(
            f"unsupported document extension for parsing: {suffix or '<none>'}"
        )

    def _cached_blocks(
        self,
        source: Any,
        parser_kind: ParserKind,
        parser_version: str,
    ) -> list[DocumentBlock]:
        if source.blob.parse_status is not ParseStatus.READY:
            return []
        cached = self.repository.list_blocks_for_source(
            source.asset.id,
            parser_kind,
            parser_version,
        )
        if cached:
            return cached
        reusable = self.repository.find_blocks_for_blob(
            source.blob.sha256,
            parser_kind,
            parser_version,
        )
        if not reusable:
            return []
        chunks = [
            ParsedChunk(
                text=block.text,
                section=block.section,
                page_number=block.page_number,
            )
            for block in reusable
        ]
        return self.parse_service.persist_chunks(
            source.asset.id,
            parser_kind,
            chunks,
            parser_version=parser_version,
        )

    async def _fetch_external_text(
        self,
        source: Any,
        suffix: str,
        *,
        chapter: str | None,
    ) -> str:
        with self._workspace(source, suffix) as (input_path, output_dir):
            if suffix in OFFICE_EXTENSIONS:
                payload = _json_object(
                    await self._call_tool(
                        self.config.parse_document_tool,
                        {"path": str(input_path), "outputDir": str(output_dir)},
                    )
                )
                saved_path = payload.get("savedPath")
                if not isinstance(saved_path, str) or not saved_path.strip():
                    raise ExternalDocumentMCPContentError(
                        "parse_document did not return a savedPath for full output"
                    )
                candidate = Path(saved_path).resolve()
                output_root = output_dir.resolve()
                if not candidate.is_relative_to(output_root):
                    raise ExternalDocumentMCPContentError(
                        "parse_document savedPath escaped the controlled output directory"
                    )
                if not candidate.is_file():
                    raise ExternalDocumentMCPContentError(
                        "parse_document savedPath does not exist"
                    )
                try:
                    text = candidate.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as error:
                    raise ExternalDocumentMCPContentError(
                        "parse_document output could not be read as UTF-8 Markdown"
                    ) from error
                self._check_full_text(text, payload.get("chars"))
                return text

            chapter_name = self._resolve_chapter(source, chapter)
            parse_result = _json_object(
                await self._call_tool(
                    self.config.parse_pdf_tool,
                    {
                        "path": str(input_path),
                        "course": source.asset.course_id,
                        "chapter": chapter_name,
                    },
                )
            )
            result = await self._call_tool(
                self.config.get_study_section_tool,
                {"course": source.asset.course_id, "chapter": chapter_name},
            )
            text = self._text_result(result)
            self._check_full_text(text, parse_result.get("chars"))
            return text

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        async def invoke() -> Any:
            result = self.client.call_tool(name, arguments)
            if inspect.isawaitable(result):
                return await result
            return result

        try:
            return await asyncio.wait_for(invoke(), self.config.timeout_seconds)
        except asyncio.TimeoutError as error:
            raise ExternalDocumentMCPTimeoutError(
                "external document MCP tool call timed out"
            ) from error
        except ExternalDocumentMCPError:
            raise
        except Exception as error:
            raise ExternalDocumentMCPUnavailableError(
                "external document MCP tool call failed"
            ) from error

    @staticmethod
    def _text_result(result: Any) -> str:
        _raise_tool_error(result)
        value = _result_value(result)
        if isinstance(value, Mapping):
            _raise_tool_error(value)
            text_value = value.get("text")
            if isinstance(text_value, str):
                value = text_value
            else:
                raise ExternalDocumentMCPContentError(
                    "get_study_section did not return text content"
                )
        if not isinstance(value, str):
            raise ExternalDocumentMCPContentError(
                "get_study_section returned a non-text value"
            )
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                maybe_error = json.loads(stripped)
            except json.JSONDecodeError:
                maybe_error = None
            if isinstance(maybe_error, Mapping) and maybe_error.get("error"):
                raise ExternalDocumentMCPToolError(
                    f"get_study_section failed: {maybe_error['error']}"
                )
        if not stripped:
            raise ExternalDocumentMCPContentError(
                "get_study_section returned empty text"
            )
        return value

    @staticmethod
    def _check_full_text(text: str, expected_chars: Any) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ExternalDocumentMCPContentError("external MCP returned empty text")
        if isinstance(expected_chars, int) and expected_chars >= 0:
            actual_chars = _full_char_length(text)
            if actual_chars != expected_chars:
                raise ExternalDocumentMCPContentError(
                    "external MCP returned truncated or incomplete document text"
                )

    @staticmethod
    def _resolve_chapter(source: Any, chapter: str | None) -> str:
        value = chapter
        if isinstance(value, str) and not value.strip():
            value = None
        if value is None:
            metadata = source.asset.metadata
            candidate = metadata.get("chapter") if isinstance(metadata, Mapping) else None
            value = candidate.strip() if isinstance(candidate, str) and candidate.strip() else None
        if value is None:
            value = Path(source.asset.display_name).stem
        value = value.strip()
        if not value or len(value) > 80 or any(char in value for char in "\\/\0"):
            raise ExternalDocumentMCPConfigurationError(
                "PDF chapter must be a non-empty safe chapter name"
            )
        return value

    @contextmanager
    def _workspace(self, source: Any, suffix: str):
        base = (
            Path(self.config.staging_root).expanduser().resolve()
            if self.config.staging_root is not None
            else Path(source.blob.storage_path).resolve().parent
        )
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".studypilot-mcp-", dir=str(base)) as temp_name:
            root = Path(temp_name)
            output = root / "output"
            output.mkdir()
            display = Path(source.asset.display_name).name
            if not display or Path(display).suffix.lower() != suffix:
                display = f"{source.asset.id}{suffix}"
            input_path = root / display
            shutil.copyfile(source.blob.storage_path, input_path)
            yield input_path, output


__all__ = [
    "ExternalDocumentMCPAdapter",
    "ExternalDocumentMCPConfig",
    "ExternalDocumentMCPContentError",
    "ExternalDocumentMCPError",
    "ExternalDocumentMCPConfigurationError",
    "ExternalDocumentMCPTimeoutError",
    "ExternalDocumentMCPToolError",
    "ExternalDocumentMCPUnavailableError",
    "MCPToolClient",
    "OfficialMCPToolClient",
    "OFFICE_PARSER_VERSION",
    "PDF_PARSER_VERSION",
]

from __future__ import annotations

import asyncio
import json
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studypilot.api import create_app
from studypilot.application.external_document_mcp import (
    OFFICE_PARSER_VERSION,
    PDF_PARSER_VERSION,
    ExternalDocumentMCPAdapter,
    ExternalDocumentMCPConfig,
    ExternalDocumentMCPContentError,
    ExternalDocumentMCPTimeoutError,
    ExternalDocumentMCPToolError,
    ExternalDocumentMCPUnavailableError,
)
from studypilot.application.parsing import ParseService
from studypilot.application.source_import import SourceImportService
from studypilot.domain.knowledge import ParserKind
from studypilot.domain.models import Course
from studypilot.domain.sources import DocumentKind, ParseStatus, TrustLevel
from studypilot.infrastructure.sqlite_repository import SQLiteRepository


class FakeMCPClient:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return await self.handler(name, arguments)


def _json_result(payload: object) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}


def _system(tmp_path, *, client=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    repository = SQLiteRepository(tmp_path / "study.db")
    repository.initialize()
    repository.save_course(Course(id="course", name="课程"))
    importer = SourceImportService(repository, tmp_path / "storage")
    parser = ParseService(repository)
    if client is None:
        client = FakeMCPClient(lambda _name, _arguments: _json_result({}))
    adapter = ExternalDocumentMCPAdapter(
        repository,
        parser,
        ExternalDocumentMCPConfig(
            timeout_seconds=1,
            staging_root=tmp_path / "mcp-stage",
        ),
        client=client,
    )
    return repository, importer, parser, adapter, client


def _import(importer, content: bytes, name: str):
    return importer.import_file(
        course_id="course",
        stream=BytesIO(content),
        display_name=name,
        document_kind=DocumentKind.COURSE_MATERIAL,
        trust_level=TrustLevel.HIGH,
        origin="test",
    )


def test_office_route_reads_complete_saved_markdown_and_cleans_workspace(tmp_path) -> None:
    complete = "# 标题\n\n" + ("完整内容。" * 2_000)

    async def handler(name, arguments):
        assert name == "parse_document"
        output = Path(arguments["outputDir"])
        assert output.is_dir()
        saved = output / (Path(arguments["path"]).stem + ".md")
        saved.write_text(complete, encoding="utf-8")
        return _json_result(
            {
                "success": True,
                "savedPath": str(saved),
                "chars": len(complete),
                "contentTruncated": True,
                "content": complete[:8_000],
            }
        )

    repository, importer, _parser, adapter, client = _system(
        tmp_path,
        client=FakeMCPClient(handler),
    )
    source = _import(importer, b"office bytes", "lesson.pptx")

    blocks = asyncio.run(adapter.parse_source(source.asset.id))

    assert len(blocks) == 1
    assert blocks[0].parser_kind is ParserKind.MARKDOWN
    assert blocks[0].parser_version == OFFICE_PARSER_VERSION
    assert ("完整内容。" * 2_000) in blocks[0].text
    assert client.calls[0][0] == "parse_document"
    assert client.calls[0][1]["path"].endswith(".pptx")
    assert not list((tmp_path / "mcp-stage").glob(".studypilot-mcp-*"))
    assert repository.get_source(source.asset.id).blob.parse_status is ParseStatus.READY


def test_pdf_route_calls_parse_then_get_section_and_uses_full_text(tmp_path) -> None:
    complete = "PDF 完整正文。\n" + ("第二行内容。\n" * 1_000)

    async def handler(name, arguments):
        if name == "parse_pdf":
            assert arguments["course"] == "course"
            assert arguments["chapter"] == "chapter-1"
            return _json_result({"course": "course", "chapter": "chapter-1", "chars": len(complete)})
        assert name == "get_study_section"
        assert arguments == {"course": "course", "chapter": "chapter-1"}
        return {"content": [{"type": "text", "text": complete}]}

    repository, importer, _parser, adapter, client = _system(
        tmp_path,
        client=FakeMCPClient(handler),
    )
    source = _import(importer, b"pdf bytes", "chapter-1.pdf")

    blocks = asyncio.run(adapter.parse_source(source.asset.id))

    assert len(blocks) == 1
    assert blocks[0].parser_kind is ParserKind.TEXT
    assert blocks[0].parser_version == PDF_PARSER_VERSION
    assert blocks[0].text.endswith("第二行内容。")
    assert [name for name, _ in client.calls] == ["parse_pdf", "get_study_section"]
    assert repository.get_source(source.asset.id).blob.parse_status is ParseStatus.READY


def test_markdown_and_text_stay_on_local_parser_without_mcp_call(tmp_path) -> None:
    repository, importer, _parser, adapter, client = _system(tmp_path)
    markdown = _import(importer, b"# Section\n\nBody", "notes.md")
    text = _import(importer, b"plain text", "notes.txt")

    markdown_blocks = asyncio.run(adapter.parse_source(markdown.asset.id))
    text_blocks = asyncio.run(adapter.parse_source(text.asset.id))

    assert markdown_blocks[0].parser_kind is ParserKind.MARKDOWN
    assert text_blocks[0].parser_kind is ParserKind.TEXT
    assert client.calls == []
    assert repository.get_source(markdown.asset.id).blob.parse_status is ParseStatus.READY


def test_same_blob_and_parser_version_reuses_blocks_across_sources(tmp_path) -> None:
    complete = "# Cached\n\n来自一次 MCP 调用。"

    async def handler(name, arguments):
        output = Path(arguments["outputDir"])
        saved = output / (Path(arguments["path"]).stem + ".md")
        saved.write_text(complete, encoding="utf-8")
        return _json_result({"success": True, "savedPath": str(saved), "chars": len(complete)})

    repository, importer, _parser, adapter, client = _system(
        tmp_path,
        client=FakeMCPClient(handler),
    )
    first = _import(importer, b"same office", "first.docx")
    second = _import(importer, b"same office", "second.docx")

    first_blocks = asyncio.run(adapter.parse_source(first.asset.id))
    second_blocks = asyncio.run(adapter.parse_source(second.asset.id))

    assert len(client.calls) == 1
    assert second_blocks[0].text == first_blocks[0].text
    assert second_blocks[0].source_asset_id == second.asset.id
    assert second_blocks[0].id != first_blocks[0].id
    assert repository.list_blocks_for_source(second.asset.id, ParserKind.MARKDOWN, OFFICE_PARSER_VERSION)


def test_repeated_source_parse_reuses_current_blocks(tmp_path) -> None:
    complete = "# Once\n\n内容。"

    async def handler(_name, arguments):
        saved = Path(arguments["outputDir"]) / (Path(arguments["path"]).stem + ".md")
        saved.write_text(complete, encoding="utf-8")
        return _json_result({"success": True, "savedPath": str(saved), "chars": len(complete)})

    _repository, importer, _parser, adapter, client = _system(
        tmp_path,
        client=FakeMCPClient(handler),
    )
    source = _import(importer, b"same source", "once.xlsx")

    first = asyncio.run(adapter.parse_source(source.asset.id))
    second = asyncio.run(adapter.parse_source(source.asset.id))

    assert first == second
    assert len(client.calls) == 1


async def _tool_failure(_name, _arguments):
    return _json_result({"success": False, "error": "MinerU failed"})


async def _offline_failure(_name, _arguments):
    raise RuntimeError("offline")


@pytest.mark.parametrize(
    ("kind", "handler", "error_type"),
    [
        (
            "tool",
            _tool_failure,
            ExternalDocumentMCPToolError,
        ),
        (
            "unavailable",
            _offline_failure,
            ExternalDocumentMCPUnavailableError,
        ),
    ],
)
def test_external_failure_is_explicit_and_marks_blob_failed(
    tmp_path,
    kind,
    handler,
    error_type,
) -> None:
    repository, importer, _parser, adapter, _client = _system(
        tmp_path,
        client=FakeMCPClient(handler),
    )
    source = _import(importer, b"bad office", "bad.pptx")

    with pytest.raises(error_type):
        asyncio.run(adapter.parse_source(source.asset.id))

    loaded = repository.get_source(source.asset.id)
    assert loaded.blob.parse_status is ParseStatus.FAILED
    assert repository.list_blocks("course") == []
    assert not list((tmp_path / "mcp-stage").glob(".studypilot-mcp-*"))


def test_timeout_and_truncated_output_are_explicit_failures(tmp_path) -> None:
    async def slow(_name, _arguments):
        await asyncio.sleep(0.2)

    repository, importer, _parser, adapter, _client = _system(
        tmp_path,
        client=FakeMCPClient(slow),
    )
    timeout_source = _import(importer, b"timeout", "timeout.docx")
    adapter.config = ExternalDocumentMCPConfig(
        timeout_seconds=0.01,
        staging_root=tmp_path / "mcp-stage",
    )

    with pytest.raises(ExternalDocumentMCPTimeoutError):
        asyncio.run(adapter.parse_source(timeout_source.asset.id))
    assert repository.get_source(timeout_source.asset.id).blob.parse_status is ParseStatus.FAILED

    async def truncated(_name, arguments):
        saved = Path(arguments["outputDir"]) / (Path(arguments["path"]).stem + ".md")
        saved.write_text("short", encoding="utf-8")
        return _json_result({"success": True, "savedPath": str(saved), "chars": 99})

    _repository, importer, _parser, adapter, _client = _system(
        tmp_path / "truncated",
        client=FakeMCPClient(truncated),
    )
    truncated_source = _import(importer, b"truncated", "truncated.docx")
    with pytest.raises(ExternalDocumentMCPContentError):
        asyncio.run(adapter.parse_source(truncated_source.asset.id))


def test_api_explicit_external_route_uses_adapter_and_reports_unconfigured(tmp_path) -> None:
    complete = "# API\n\nMCP 内容。"

    async def handler(_name, arguments):
        saved = Path(arguments["outputDir"]) / (Path(arguments["path"]).stem + ".md")
        saved.write_text(complete, encoding="utf-8")
        return _json_result({"success": True, "savedPath": str(saved), "chars": len(complete)})

    db = tmp_path / "adapter" / "study.db"
    repository, importer, parser, adapter, _client = _system(
        tmp_path / "adapter",
        client=FakeMCPClient(handler),
    )
    # The application creates its own repository object for the same DB; this
    # mirrors a real process boundary while the injected adapter remains mockable.
    source = _import(importer, b"api office", "api.pptx")
    app = create_app(db, tmp_path / "data", document_mcp_adapter=adapter)
    with TestClient(app) as client:
        client.put("/courses/course", json={"id": "course", "name": "课程"})
        response = client.post(
            f"/sources/{source.asset.id}/parse-external",
            json={},
        )
        assert response.status_code == 200
        assert response.json()[0]["parser_version"] == OFFICE_PARSER_VERSION

    # An app without injection fails explicitly instead of attempting a hidden
    # second parsing implementation or pretending the source is ready.
    app_without_mcp = create_app(db, tmp_path / "data-2")
    with TestClient(app_without_mcp) as client:
        response = client.post(f"/sources/{source.asset.id}/parse", json={"external": True})
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]
    assert parser.repository.get_source(source.asset.id).blob.parse_status is ParseStatus.READY


def test_pc_form_can_explicitly_trigger_external_parser_and_show_status(tmp_path) -> None:
    complete = "# PC\n\n外部解析内容。"

    async def handler(_name, arguments):
        saved = Path(arguments["outputDir"]) / (Path(arguments["path"]).stem + ".md")
        saved.write_text(complete, encoding="utf-8")
        return _json_result({"success": True, "savedPath": str(saved), "chars": len(complete)})

    repository, importer, _parser, adapter, _client = _system(
        tmp_path,
        client=FakeMCPClient(handler),
    )
    source = _import(importer, b"pc office", "pc.pptx")
    app = create_app(
        repository.database_path,
        tmp_path / "data",
        document_mcp_adapter=adapter,
    )

    with TestClient(app) as client:
        response = client.post(
            f"/study/sources/{source.asset.id}/parse",
            data={"parser_kind": "TEXT", "external": "true"},
            follow_redirects=False,
        )
        page = client.get(response.headers["location"])

    assert response.status_code == 303
    assert "资料已解析" in page.text
    assert "外部文档识别 MCP 已接入" in page.text
    assert repository.get_source(source.asset.id).blob.parse_status is ParseStatus.READY

from io import BytesIO

from fastapi.testclient import TestClient

from studypilot.api import create_app
from studypilot.application.material_organizer import CourseMaterialOrganizer
from studypilot.application.parsing import ParseService
from studypilot.application.source_import import SourceImportService
from studypilot.domain.models import Course
from studypilot.domain.knowledge import ParserKind
from studypilot.domain.sources import DocumentKind, TrustLevel
from studypilot.infrastructure.sqlite_repository import SQLiteRepository


def _prepared_course(tmp_path):
    repository = SQLiteRepository(tmp_path / "organizer.db")
    repository.initialize()
    repository.save_course(Course(id="probability", name="概率论"))
    importer = SourceImportService(repository, tmp_path / "files")
    source = importer.import_file(
        course_id="probability",
        stream=BytesIO("# 条件概率\n\n定义内容\n\n例题内容".encode()),
        display_name="第一章笔记.md",
        document_kind=DocumentKind.PERSONAL_NOTE,
        trust_level=TrustLevel.MEDIUM,
        origin="个人笔记",
    )
    ParseService(repository).parse_source(source.asset.id, ParserKind.MARKDOWN)
    return repository


def test_organizer_builds_readable_markdown_with_source_and_locators(tmp_path) -> None:
    repository = _prepared_course(tmp_path)
    organizer = CourseMaterialOrganizer(repository, tmp_path / "organized")
    markdown = organizer.render("probability")
    path = organizer.refresh("probability")

    assert "# 概率论课程资料" in markdown
    assert "## 条件概率" in markdown
    assert "### 来源：第一章笔记.md" in markdown
    assert "定义内容" in markdown and "例题内容" in markdown
    assert "原文定位" in markdown
    assert path.read_text(encoding="utf-8") == markdown


def test_parse_endpoint_refreshes_generated_material(tmp_path) -> None:
    app = create_app(tmp_path / "api.db", tmp_path / "storage")
    with TestClient(app) as client:
        assert client.put(
            "/courses/probability", json={"id": "probability", "name": "概率论"}
        ).status_code == 200
        uploaded = client.post(
            "/courses/probability/sources",
            files={"file": ("第二章.md", "# 随机变量\n\n分布函数定义", "text/markdown")},
            data={"document_kind": "COURSE_MATERIAL", "trust_level": "HIGH"},
        )
        source_id = uploaded.json()["asset"]["id"]
        parsed = client.post(
            f"/sources/{source_id}/parse", json={"parser_kind": "MARKDOWN"}
        )
        material = client.get("/courses/probability/materials/markdown")

    assert parsed.status_code == 200
    assert material.status_code == 200
    assert "## 随机变量" in material.text
    generated = tmp_path / "storage" / "organized" / "probability" / "课程资料.generated.md"
    assert generated.exists()

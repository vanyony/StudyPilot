from __future__ import annotations

from io import BytesIO

import pytest

from studypilot.application.source_import import SourceImportService
from studypilot.domain.models import Course
from studypilot.domain.sources import DocumentKind, ParseStatus, TrustLevel
from studypilot.infrastructure.sqlite_repository import SQLiteRepository


@pytest.fixture
def source_system(tmp_path):
    repository = SQLiteRepository(tmp_path / "study.db")
    repository.initialize()
    repository.save_course(Course(id="probability", name="概率论"))
    service = SourceImportService(repository, tmp_path / "storage")
    return repository, service, tmp_path / "storage"


def import_source(service, content: bytes, *, name="paper.pdf", origin="teacher"):
    return service.import_file(
        course_id="probability",
        stream=BytesIO(content),
        display_name=name,
        document_kind=DocumentKind.EXAM_PAPER,
        trust_level=TrustLevel.HIGH,
        origin=origin,
        metadata={"year": 2024},
    )


def test_same_bytes_reuse_blob_but_keep_two_sources(source_system) -> None:
    repository, service, _ = source_system

    first = import_source(service, b"same exam", origin="teacher")
    second = import_source(service, b"same exam", origin="class-group")

    assert first.blob.id == second.blob.id
    assert first.asset.id != second.asset.id
    assert repository.count_blobs() == 1
    assert len(repository.list_sources("probability")) == 2
    assert {item.asset.origin for item in repository.list_sources("probability")} == {
        "teacher",
        "class-group",
    }


def test_same_filename_with_different_bytes_creates_two_blobs(source_system) -> None:
    repository, service, _ = source_system

    first = import_source(service, b"version one", name="paper.pdf")
    second = import_source(service, b"version two", name="paper.pdf")

    assert first.blob.id != second.blob.id
    assert first.blob.sha256 != second.blob.sha256
    assert repository.count_blobs() == 2


def test_sources_survive_repository_restart(source_system) -> None:
    repository, service, _ = source_system
    created = import_source(service, b"persistent")

    restarted = SQLiteRepository(repository.database_path)
    restarted.initialize()
    loaded = restarted.get_source(created.asset.id)

    assert loaded == created
    assert loaded.blob.parse_status is ParseStatus.PENDING


def test_failure_removes_temporary_and_new_blob_file(source_system, monkeypatch) -> None:
    repository, service, storage_root = source_system

    def fail_registration(**_):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(repository, "register_source", fail_registration)
    with pytest.raises(RuntimeError, match="database unavailable"):
        import_source(service, b"must be cleaned")

    assert list((storage_root / ".tmp").glob("*")) == []
    assert list((storage_root / "blobs").rglob("*")) == []


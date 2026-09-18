from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import BinaryIO

from pydantic import JsonValue

from studypilot.domain.sources import DocumentKind, SourceRecord, TrustLevel
from studypilot.infrastructure.sqlite_repository import SQLiteRepository


class SourceImportService:
    CHUNK_SIZE = 1024 * 1024

    def __init__(self, repository: SQLiteRepository, storage_root: str | Path) -> None:
        self.repository = repository
        self.storage_root = Path(storage_root).resolve()
        self.blob_root = self.storage_root / "blobs"
        self.temp_root = self.storage_root / ".tmp"

    def import_file(
        self,
        *,
        course_id: str,
        stream: BinaryIO,
        display_name: str,
        document_kind: DocumentKind,
        trust_level: TrustLevel,
        origin: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> SourceRecord:
        self.repository.get_course(course_id)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        temp_path = self.temp_root / f"{os.urandom(16).hex()}.upload"
        final_path: Path | None = None
        created_blob_file = False

        try:
            digest = hashlib.sha256()
            byte_size = 0
            with temp_path.open("xb") as target:
                while chunk := stream.read(self.CHUNK_SIZE):
                    digest.update(chunk)
                    byte_size += len(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())

            sha256 = digest.hexdigest()
            final_path = self.blob_root / sha256[:2] / sha256
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path.exists():
                temp_path.unlink()
            else:
                os.replace(temp_path, final_path)
                created_blob_file = True

            return self.repository.register_source(
                course_id=course_id,
                sha256=sha256,
                storage_path=str(final_path),
                byte_size=byte_size,
                display_name=display_name,
                document_kind=document_kind,
                trust_level=trust_level,
                origin=origin,
                metadata=metadata or {},
            )
        except Exception:
            temp_path.unlink(missing_ok=True)
            if created_blob_file and final_path is not None:
                final_path.unlink(missing_ok=True)
                try:
                    final_path.parent.rmdir()
                except OSError:
                    # Another blob may share the prefix directory.
                    pass
            raise

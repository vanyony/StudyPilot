"""Opt-in smoke test for the external document-recognition MCP server.

This script is intentionally not part of the default pytest suite.  It makes
one real MCP call only when a caller supplies a local Office/PDF sample path;
it never prints document contents or credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

from studypilot.application.external_document_mcp import (
    OFFICE_EXTENSIONS,
    PDF_EXTENSIONS,
    ExternalDocumentMCPAdapter,
    ExternalDocumentMCPConfig,
)
from studypilot.application.parsing import ParseService
from studypilot.application.source_import import SourceImportService
from studypilot.domain.models import Course
from studypilot.domain.sources import DocumentKind, TrustLevel
from studypilot.infrastructure.sqlite_repository import SQLiteRepository


def main() -> int:
    parser = argparse.ArgumentParser(description="StudyPilot external MCP document smoke")
    parser.add_argument(
        "path",
        nargs="?",
        default=os.getenv("STUDYPILOT_DOCUMENT_MCP_SMOKE_FILE"),
        help="absolute Office/PDF sample path (or STUDYPILOT_DOCUMENT_MCP_SMOKE_FILE)",
    )
    args = parser.parse_args()
    if not args.path:
        print(
            "Set STUDYPILOT_DOCUMENT_MCP_SMOKE_FILE to an Office/PDF sample "
            "under a path allowed by the MCP server."
        )
        return 2

    sample = Path(args.path).expanduser().resolve()
    suffix = sample.suffix.lower()
    if suffix not in OFFICE_EXTENSIONS | PDF_EXTENSIONS:
        print("Smoke sample must use a supported Office or PDF extension.")
        return 2
    if not sample.is_file():
        print("Smoke sample path does not exist.")
        return 2

    with tempfile.TemporaryDirectory(prefix="studypilot-mcp-smoke-") as temp_name:
        root = Path(temp_name)
        repository = SQLiteRepository(root / "study.db")
        repository.initialize()
        repository.save_course(Course(id="smoke", name="MCP smoke"))
        importer = SourceImportService(repository, root / "storage")
        with sample.open("rb") as stream:
            source = importer.import_file(
                course_id="smoke",
                stream=stream,
                display_name=sample.name,
                document_kind=DocumentKind.COURSE_MATERIAL,
                trust_level=TrustLevel.MEDIUM,
                origin="opt-in external MCP smoke",
            )
        adapter = ExternalDocumentMCPAdapter(
            repository,
            ParseService(repository),
            ExternalDocumentMCPConfig.from_env(),
        )
        try:
            blocks = asyncio.run(adapter.parse_source(source.asset.id))
        except Exception as error:
            print(f"External MCP smoke failed: {type(error).__name__}: {error}")
            return 1

    print(
        "External MCP smoke passed: "
        f"parser={blocks[0].parser_kind.value if blocks else 'none'} "
        f"blocks={len(blocks)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

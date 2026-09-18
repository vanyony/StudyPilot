from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from studypilot.domain.knowledge import DocumentBlock, ParserKind
from studypilot.infrastructure.sqlite_repository import SQLiteRepository


@dataclass(frozen=True)
class ParsedChunk:
    text: str
    section: str | None = None
    page_number: int | None = None


class DocumentParser(Protocol):
    kind: ParserKind
    version: str

    def parse(self, text: str) -> list[ParsedChunk]: ...


class PlainTextParser:
    kind = ParserKind.TEXT
    version = "1"

    def parse(self, text: str) -> list[ParsedChunk]:
        return [ParsedChunk(part.strip()) for part in re.split(r"\n\s*\n", text) if part.strip()]


class MarkdownParser:
    kind = ParserKind.MARKDOWN
    version = "1"
    _heading = re.compile(r"^#{1,6}\s+(.+?)\s*$")

    def parse(self, text: str) -> list[ParsedChunk]:
        chunks: list[ParsedChunk] = []
        section: str | None = None
        paragraph: list[str] = []

        def flush() -> None:
            if paragraph:
                value = "\n".join(paragraph).strip()
                if value:
                    chunks.append(ParsedChunk(value, section=section))
                paragraph.clear()

        for line in text.splitlines():
            heading = self._heading.match(line)
            if heading:
                flush()
                section = heading.group(1).strip()
            elif line.strip():
                paragraph.append(line.rstrip())
            else:
                flush()
        flush()
        return chunks


class ParseService:
    def __init__(self, repository: SQLiteRepository) -> None:
        self.repository = repository
        self.parsers: dict[ParserKind, DocumentParser] = {
            ParserKind.MARKDOWN: MarkdownParser(),
            ParserKind.TEXT: PlainTextParser(),
        }

    def parse_source(
        self,
        source_asset_id: str,
        parser_kind: ParserKind,
        extracted_text: str | None = None,
        *,
        parser_version: str | None = None,
    ) -> list[DocumentBlock]:
        source = self.repository.get_source(source_asset_id)
        parser = self.parsers[parser_kind]
        text = extracted_text if extracted_text is not None else self._read_utf8(source.blob.storage_path)
        chunks = parser.parse(text)
        return self.persist_chunks(
            source_asset_id,
            parser_kind,
            chunks,
            parser_version=parser_version or parser.version,
        )

    def persist_chunks(
        self,
        source_asset_id: str,
        parser_kind: ParserKind,
        chunks: list[ParsedChunk],
        *,
        parser_version: str | None = None,
    ) -> list[DocumentBlock]:
        """Materialise parser output using the same block/repository path.

        External parser adapters use this method after obtaining bounded text
        from another process.  Keeping block identity and persistence here
        prevents each adapter from creating a second ingestion implementation.
        """

        source = self.repository.get_source(source_asset_id)
        parser = self.parsers[parser_kind]
        resolved_version = parser_version or parser.version
        blocks = [
            self._to_block(
                source_asset_id,
                source.asset.course_id,
                parser.kind,
                resolved_version,
                index,
                chunk,
            )
            for index, chunk in enumerate(chunks)
        ]
        self.repository.replace_blocks(source_asset_id, parser.kind, resolved_version, blocks)
        return blocks

    @staticmethod
    def _read_utf8(storage_path: str) -> str:
        try:
            return Path(storage_path).read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                "source is not UTF-8 text; provide extracted_text or use a future format adapter"
            ) from error

    @staticmethod
    def _to_block(
        source_asset_id: str,
        course_id: str,
        parser_kind: ParserKind,
        parser_version: str,
        index: int,
        chunk: ParsedChunk,
    ) -> DocumentBlock:
        content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        identity = f"{source_asset_id}:{parser_kind.value}:{parser_version}:{index}:{content_hash}"
        block_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return DocumentBlock(
            id=block_id,
            source_asset_id=source_asset_id,
            course_id=course_id,
            parser_kind=parser_kind,
            parser_version=parser_version,
            page_number=chunk.page_number,
            section=chunk.section,
            block_index=index,
            text=chunk.text,
            content_hash=content_hash,
        )

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

from studypilot.domain.knowledge import DocumentBlock
from studypilot.domain.sources import SourceRecord
from studypilot.infrastructure.sqlite_repository import SQLiteRepository


_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


class CourseMaterialOrganizer:
    """Build a human-readable Markdown view from canonical source blocks."""

    def __init__(self, repository: SQLiteRepository, output_root: str | Path) -> None:
        self.repository = repository
        self.output_root = Path(output_root)

    def render(self, course_id: str) -> str:
        course = self.repository.get_course(course_id)
        sources = self.repository.list_sources(course_id)
        blocks = self.repository.list_blocks(course_id)
        source_by_id = {item.asset.id: item for item in sources}
        by_section: dict[str, list[DocumentBlock]] = defaultdict(list)
        for block in blocks:
            by_section[(block.section or "未归类资料").strip()].append(block)

        lines = [
            f"# {course.name}课程资料",
            "",
            "> 本文档由 StudyPilot 根据已解析原始资料自动整理。教学检索仍以原始 DocumentBlock 为准；每段均保留来源和原文定位。",
            "",
            "## 资料目录",
            "",
        ]
        if not sources:
            lines.append("- 暂无资料")
        for source in sources:
            lines.append(self._source_list_item(source))

        lines.extend(["", "## 章节目录", ""])
        if not by_section:
            lines.append("- 暂无已解析内容")
        for section in by_section:
            lines.append(f"- {section}")

        for section, section_blocks in by_section.items():
            lines.extend(["", f"## {section}", ""])
            last_source_id: str | None = None
            for block in section_blocks:
                source = source_by_id.get(block.source_asset_id)
                if block.source_asset_id != last_source_id:
                    lines.append(self._source_heading(source, block.source_asset_id))
                    lines.append("")
                    last_source_id = block.source_asset_id
                lines.append(block.text.strip())
                lines.append("")
                lines.append(self._locator(block))
                lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def refresh(self, course_id: str) -> Path:
        markdown = self.render(course_id)
        directory = self.output_root / self._safe_component(course_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "课程资料.generated.md"
        temporary = directory / ".课程资料.generated.md.tmp"
        temporary.write_text(markdown, encoding="utf-8")
        temporary.replace(target)
        return target

    @staticmethod
    def _source_list_item(source: SourceRecord) -> str:
        origin = f"；来源：{source.asset.origin}" if source.asset.origin else ""
        return (
            f"- {source.asset.display_name}"
            f"（{source.asset.document_kind.value}；可信度：{source.asset.trust_level.value}{origin}）"
        )

    @staticmethod
    def _source_heading(source: SourceRecord | None, source_id: str) -> str:
        if source is None:
            return f"### 未知来源 {source_id}"
        return f"### 来源：{source.asset.display_name}"

    @staticmethod
    def _locator(block: DocumentBlock) -> str:
        parts = [f"block {block.id}", f"序号 {block.block_index}"]
        if block.page_number is not None:
            parts.append(f"第 {block.page_number} 页")
        return "> 原文定位：" + "；".join(parts)

    @staticmethod
    def _safe_component(value: str) -> str:
        safe = _SAFE_COMPONENT.sub("_", value).strip("._")
        return safe or "course"

from __future__ import annotations

from io import BytesIO

from studypilot.application.parsing import ParseService
from studypilot.application.retrieval import (
    BM25Retriever,
    CitationValidator,
    KnowledgeWindowBuilder,
    tokenize,
)
from studypilot.application.source_import import SourceImportService
from studypilot.domain.knowledge import EvidenceRelation, ParserKind
from studypilot.domain.models import Course
from studypilot.domain.sources import DocumentKind, TrustLevel
from studypilot.infrastructure.sqlite_repository import SQLiteRepository


def setup_course(tmp_path, course_id="probability"):
    repository = SQLiteRepository(tmp_path / "study.db")
    repository.initialize()
    repository.save_course(Course(id=course_id, name=course_id))
    importer = SourceImportService(repository, tmp_path / "storage")
    parser = ParseService(repository)
    return repository, importer, parser


def import_text(importer, course_id, text, name="note.md"):
    return importer.import_file(
        course_id=course_id,
        stream=BytesIO(text.encode("utf-8")),
        display_name=name,
        document_kind=DocumentKind.PERSONAL_NOTE,
        trust_level=TrustLevel.MEDIUM,
        origin="local-note",
    )


def test_markdown_parser_preserves_section_and_is_idempotent(tmp_path) -> None:
    repository, importer, parser = setup_course(tmp_path)
    source = import_text(importer, "probability", "# 条件概率\n\n定义内容\n\n## 例题\n\n例题内容")

    first = parser.parse_source(source.asset.id, ParserKind.MARKDOWN)
    second = parser.parse_source(source.asset.id, ParserKind.MARKDOWN)

    assert first == second
    assert [(block.section, block.block_index) for block in first] == [
        ("条件概率", 0),
        ("例题", 1),
    ]
    assert repository.list_blocks("probability") == first


def test_chinese_tokenizer_and_bm25_recall_keyword(tmp_path) -> None:
    repository, importer, parser = setup_course(tmp_path)
    source = import_text(
        importer,
        "probability",
        "条件概率用于已知事件下重新计算概率。\n\n贝叶斯公式可以反推原因概率。",
    )
    parser.parse_source(source.asset.id, ParserKind.TEXT)

    hits = BM25Retriever(repository).search("probability", "贝叶斯概率", limit=2)

    assert "贝叶" in tokenize("贝叶斯概率")
    assert hits
    assert "贝叶斯公式" in hits[0].block.text
    assert hits[0].score > 0


def test_title_aware_bm25_indexes_section_with_fixed_weight(tmp_path) -> None:
    repository, importer, parser = setup_course(tmp_path)
    source = import_text(importer, "probability", "# 甲乙丙丁\n\n正文内容。")
    parser.parse_source(source.asset.id, ParserKind.MARKDOWN)

    assert BM25Retriever(repository, title_weight=0).search(
        "probability", "甲乙丙丁"
    ) == []
    assert BM25Retriever(repository, title_weight=2).search(
        "probability", "甲乙丙丁"
    )


def test_search_is_isolated_by_course(tmp_path) -> None:
    repository, importer, parser = setup_course(tmp_path, "probability")
    repository.save_course(Course(id="database", name="database"))
    probability = import_text(importer, "probability", "概率中的贝叶斯公式")
    database = import_text(importer, "database", "数据库事务中的隔离级别")
    parser.parse_source(probability.asset.id, ParserKind.TEXT)
    parser.parse_source(database.asset.id, ParserKind.TEXT)

    assert BM25Retriever(repository).search("database", "贝叶斯") == []
    assert BM25Retriever(repository).search("probability", "隔离级别") == []


def test_citation_can_be_read_back_and_rejects_tampering(tmp_path) -> None:
    repository, importer, parser = setup_course(tmp_path)
    source = import_text(importer, "probability", "随机变量的分布函数定义")
    block = parser.parse_source(source.asset.id, ParserKind.TEXT)[0]
    citation = repository.citation_for_block(block.id)
    validator = CitationValidator(repository)

    assert repository.get_block(citation.block_id).text == citation.quote
    assert validator.verify(citation).valid is True
    tampered = citation.model_copy(update={"quote": "被篡改的原文"})
    assert validator.verify(tampered).valid is False


def test_knowledge_window_obeys_block_and_character_budgets(tmp_path) -> None:
    repository, importer, parser = setup_course(tmp_path)
    source = import_text(
        importer,
        "probability",
        "概率定义一二三四五。\n\n概率例题六七八九十。\n\n概率补充甲乙丙丁戊。",
    )
    blocks = parser.parse_source(source.asset.id, ParserKind.TEXT)
    builder = KnowledgeWindowBuilder(BM25Retriever(repository))

    window = builder.build(
        course_id="probability",
        query="概率",
        max_blocks=2,
        max_chars=20,
        relation_by_block_id={blocks[0].id: EvidenceRelation.DEFINITION},
    )

    assert len(window.items) <= 2
    assert window.used_chars <= 20
    assert window.used_chars == sum(len(item.hit.block.text) for item in window.items)
    if any(item.hit.block.id == blocks[0].id for item in window.items):
        selected = next(item for item in window.items if item.hit.block.id == blocks[0].id)
        assert selected.relation is EvidenceRelation.DEFINITION


def test_restart_keeps_blocks_searchable(tmp_path) -> None:
    repository, importer, parser = setup_course(tmp_path)
    source = import_text(importer, "probability", "中心极限定理用于近似正态分布")
    parser.parse_source(source.asset.id, ParserKind.TEXT)

    restarted = SQLiteRepository(repository.database_path)
    restarted.initialize()
    hits = BM25Retriever(restarted).search("probability", "中心极限定理")

    assert len(hits) == 1
    assert hits[0].citation.quote == "中心极限定理用于近似正态分布"

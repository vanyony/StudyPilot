from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from studypilot.domain.models import Course, ExamGoal, Plan, Topic
from studypilot.domain.knowledge import Citation, DocumentBlock, ParserKind
from studypilot.domain.channel import ChannelBinding
from studypilot.domain.teaching import AnswerReceipt, TeachingSession
from studypilot.domain.sources import (
    ContentBlob,
    DocumentKind,
    ParseStatus,
    SourceAsset,
    SourceRecord,
    TrustLevel,
)


class NotFoundError(LookupError):
    pass


class SQLiteRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS courses (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS exam_goals (
                    id TEXT PRIMARY KEY,
                    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS topics (
                    id TEXT PRIMARY KEY,
                    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS plans (
                    goal_id TEXT PRIMARY KEY REFERENCES exam_goals(id) ON DELETE CASCADE,
                    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS content_blobs (
                    id TEXT PRIMARY KEY,
                    sha256 TEXT UNIQUE NOT NULL,
                    storage_path TEXT NOT NULL,
                    byte_size INTEGER NOT NULL CHECK(byte_size >= 0),
                    parse_status TEXT NOT NULL CHECK(parse_status IN ('PENDING', 'READY', 'FAILED')),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS source_assets (
                    id TEXT PRIMARY KEY,
                    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
                    blob_id TEXT NOT NULL REFERENCES content_blobs(id),
                    document_kind TEXT NOT NULL CHECK(document_kind IN (
                        'COURSE_MATERIAL', 'PERSONAL_NOTE', 'EXAM_PAPER',
                        'ANSWER_KEY', 'ASSIGNMENT', 'IMAGE_NOTE'
                    )),
                    origin TEXT,
                    display_name TEXT NOT NULL,
                    trust_level TEXT NOT NULL CHECK(trust_level IN ('HIGH', 'MEDIUM', 'LOW')),
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_source_assets_course
                    ON source_assets(course_id, created_at);
                CREATE TABLE IF NOT EXISTS document_blocks (
                    id TEXT PRIMARY KEY,
                    source_asset_id TEXT NOT NULL REFERENCES source_assets(id) ON DELETE CASCADE,
                    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
                    parser_kind TEXT NOT NULL CHECK(parser_kind IN ('MARKDOWN', 'TEXT')),
                    parser_version TEXT NOT NULL,
                    page_number INTEGER CHECK(page_number IS NULL OR page_number >= 1),
                    section TEXT,
                    block_index INTEGER NOT NULL CHECK(block_index >= 0),
                    text TEXT NOT NULL CHECK(length(text) > 0),
                    content_hash TEXT NOT NULL,
                    UNIQUE(source_asset_id, parser_kind, parser_version, block_index)
                );
                CREATE INDEX IF NOT EXISTS idx_document_blocks_course
                    ON document_blocks(course_id, parser_kind, parser_version);
                CREATE TABLE IF NOT EXISTS teaching_sessions (
                    session_id TEXT PRIMARY KEY,
                    thread_id TEXT UNIQUE NOT NULL,
                    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
                    goal_id TEXT NOT NULL REFERENCES exam_goals(id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK(version >= 0),
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_teaching_sessions_course
                    ON teaching_sessions(course_id, updated_at);
                CREATE TABLE IF NOT EXISTS teaching_answer_receipts (
                    session_id TEXT NOT NULL REFERENCES teaching_sessions(session_id) ON DELETE CASCADE,
                    message_id TEXT NOT NULL,
                    answer_hash TEXT NOT NULL CHECK(length(answer_hash) = 64),
                    expected_version INTEGER NOT NULL CHECK(expected_version >= 0),
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(session_id, message_id)
                );
                CREATE TABLE IF NOT EXISTS channel_bindings (
                    channel TEXT NOT NULL,
                    external_user_id TEXT NOT NULL,
                    learner_id TEXT NOT NULL,
                    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
                    session_id TEXT REFERENCES teaching_sessions(session_id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(channel, external_user_id)
                );
                CREATE INDEX IF NOT EXISTS idx_channel_bindings_course
                    ON channel_bindings(course_id, updated_at);
                """
            )

    def save_course(self, course: Course) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO courses(id, name) VALUES(?, ?) "
                "ON CONFLICT(id) DO UPDATE SET name=excluded.name",
                (course.id, course.name),
            )

    def get_course(self, course_id: str) -> Course:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, name FROM courses WHERE id = ?", (course_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"course {course_id!r} was not found")
        return Course(id=row["id"], name=row["name"])

    def list_courses(self) -> list[Course]:
        """Return courses for the local learning entry selector."""

        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id, name FROM courses ORDER BY id"
            ).fetchall()
        return [Course(id=row["id"], name=row["name"]) for row in rows]

    def save_goal(self, goal: ExamGoal) -> None:
        self.get_course(goal.course_id)
        payload = goal.model_dump_json()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO exam_goals(id, course_id, payload) VALUES(?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET course_id=excluded.course_id, payload=excluded.payload",
                (goal.id, goal.course_id, payload),
            )

    def get_goal(self, goal_id: str) -> ExamGoal:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM exam_goals WHERE id = ?", (goal_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"goal {goal_id!r} was not found")
        return ExamGoal.model_validate_json(row["payload"])

    def list_goals(self, course_id: str | None = None) -> list[ExamGoal]:
        """Return persisted goals, optionally scoped to one course."""

        if course_id is not None:
            self.get_course(course_id)
        with self._connection() as connection:
            if course_id is None:
                rows = connection.execute(
                    "SELECT payload FROM exam_goals ORDER BY id"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT payload FROM exam_goals WHERE course_id = ? ORDER BY id",
                    (course_id,),
                ).fetchall()
        return [ExamGoal.model_validate_json(row["payload"]) for row in rows]

    def save_topics(self, course_id: str, topics: list[Topic]) -> None:
        self.get_course(course_id)
        if any(topic.course_id != course_id for topic in topics):
            raise ValueError("all topics must belong to the requested course")
        with self._connection() as connection:
            connection.executemany(
                "INSERT INTO topics(id, course_id, payload) VALUES(?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET course_id=excluded.course_id, payload=excluded.payload",
                [(topic.id, course_id, topic.model_dump_json()) for topic in topics],
            )

    def list_topics(self, course_id: str) -> list[Topic]:
        self.get_course(course_id)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM topics WHERE course_id = ? ORDER BY id", (course_id,)
            ).fetchall()
        return [Topic.model_validate_json(row["payload"]) for row in rows]

    def save_plan(self, plan: Plan) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO plans(goal_id, course_id, payload) VALUES(?, ?, ?) "
                "ON CONFLICT(goal_id) DO UPDATE SET payload=excluded.payload, "
                "course_id=excluded.course_id, created_at=CURRENT_TIMESTAMP",
                (plan.goal_id, plan.course_id, plan.model_dump_json()),
            )

    def get_plan(self, goal_id: str) -> Plan:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM plans WHERE goal_id = ?", (goal_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"plan for goal {goal_id!r} was not found")
        return Plan.model_validate_json(row["payload"])

    def save_teaching_session(self, session: TeachingSession) -> None:
        """Save discoverable session metadata and its latest materialised state."""

        self.get_course(session.course_id)
        self.get_goal(session.goal_id)
        if session.state is not None:
            if session.state.course_id != session.course_id:
                raise ValueError("session state course does not match session")
            if session.state.goal_id != session.goal_id:
                raise ValueError("session state goal does not match session")
        payload = session.model_dump_json()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO teaching_sessions"
                "(session_id, thread_id, course_id, goal_id, status, version, payload, created_at, updated_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "thread_id=excluded.thread_id, course_id=excluded.course_id, goal_id=excluded.goal_id, "
                "status=excluded.status, version=excluded.version, payload=excluded.payload, "
                "updated_at=excluded.updated_at",
                (
                    session.session_id,
                    session.thread_id,
                    session.course_id,
                    session.goal_id,
                    session.status.value,
                    session.version,
                    payload,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                ),
            )

    def get_teaching_session(self, session_id: str) -> TeachingSession:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM teaching_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"teaching session {session_id!r} was not found")
        return TeachingSession.model_validate_json(row["payload"])

    def get_teaching_session_by_thread(self, thread_id: str) -> TeachingSession:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM teaching_sessions WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"teaching session for thread {thread_id!r} was not found")
        return TeachingSession.model_validate_json(row["payload"])

    def list_teaching_sessions(self, course_id: str) -> list[TeachingSession]:
        self.get_course(course_id)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM teaching_sessions WHERE course_id = ? "
                "ORDER BY updated_at, session_id",
                (course_id,),
            ).fetchall()
        return [TeachingSession.model_validate_json(row["payload"]) for row in rows]

    def save_answer_receipt(self, receipt: AnswerReceipt) -> None:
        self.get_teaching_session(receipt.session_id)
        payload = receipt.model_dump_json()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO teaching_answer_receipts"
                "(session_id, message_id, answer_hash, expected_version, payload, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(session_id, message_id) DO UPDATE SET "
                "answer_hash=excluded.answer_hash, expected_version=excluded.expected_version, "
                "payload=excluded.payload",
                (
                    receipt.session_id,
                    receipt.message_id,
                    receipt.answer_hash,
                    receipt.expected_version,
                    payload,
                    receipt.created_at.isoformat(),
                ),
            )

    def get_answer_receipt(
        self, session_id: str, message_id: str
    ) -> AnswerReceipt | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM teaching_answer_receipts "
                "WHERE session_id = ? AND message_id = ?",
                (session_id, message_id),
            ).fetchone()
        return AnswerReceipt.model_validate_json(row["payload"]) if row else None

    def save_channel_binding(self, binding: ChannelBinding) -> None:
        """Persist one platform-user mapping without storing credentials."""

        self.get_course(binding.course_id)
        if binding.session_id is not None:
            session = self.get_teaching_session(binding.session_id)
            if session.course_id != binding.course_id:
                raise ValueError("channel binding session does not belong to the course")
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO channel_bindings"
                "(channel, external_user_id, learner_id, course_id, session_id, created_at, updated_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(channel, external_user_id) DO UPDATE SET "
                "learner_id=excluded.learner_id, course_id=excluded.course_id, "
                "session_id=excluded.session_id, updated_at=excluded.updated_at",
                (
                    binding.channel,
                    binding.external_user_id,
                    binding.learner_id,
                    binding.course_id,
                    binding.session_id,
                    binding.created_at.isoformat(),
                    binding.updated_at.isoformat(),
                ),
            )

    def get_channel_binding(
        self, channel: str, external_user_id: str
    ) -> ChannelBinding | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT channel, external_user_id, learner_id, course_id, session_id, "
                "created_at, updated_at FROM channel_bindings "
                "WHERE channel = ? AND external_user_id = ?",
                (channel, external_user_id),
            ).fetchone()
        return self._channel_binding_from_row(row) if row else None

    def list_channel_bindings(self, channel: str | None = None) -> list[ChannelBinding]:
        with self._connection() as connection:
            if channel is None:
                rows = connection.execute(
                    "SELECT channel, external_user_id, learner_id, course_id, session_id, "
                    "created_at, updated_at FROM channel_bindings "
                    "ORDER BY channel, external_user_id"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT channel, external_user_id, learner_id, course_id, session_id, "
                    "created_at, updated_at FROM channel_bindings "
                    "WHERE channel = ? ORDER BY external_user_id",
                    (channel,),
                ).fetchall()
        return [self._channel_binding_from_row(row) for row in rows]

    def delete_channel_binding(self, channel: str, external_user_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM channel_bindings WHERE channel = ? AND external_user_id = ?",
                (channel, external_user_id),
            )
        return cursor.rowcount > 0

    def register_source(
        self,
        *,
        course_id: str,
        sha256: str,
        storage_path: str,
        byte_size: int,
        display_name: str,
        document_kind: DocumentKind,
        trust_level: TrustLevel,
        origin: str | None,
        metadata: dict[str, object],
    ) -> SourceRecord:
        self.get_course(course_id)
        blob_id = str(uuid4())
        asset_id = str(uuid4())
        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO content_blobs"
                "(id, sha256, storage_path, byte_size, parse_status) VALUES(?, ?, ?, ?, ?)",
                (blob_id, sha256, storage_path, byte_size, ParseStatus.PENDING.value),
            )
            blob_row = connection.execute(
                "SELECT * FROM content_blobs WHERE sha256 = ?", (sha256,)
            ).fetchone()
            assert blob_row is not None
            connection.execute(
                "INSERT INTO source_assets"
                "(id, course_id, blob_id, document_kind, origin, display_name, trust_level, metadata_json) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    asset_id,
                    course_id,
                    blob_row["id"],
                    document_kind.value,
                    origin,
                    display_name,
                    trust_level.value,
                    json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
                ),
            )
            asset_row = connection.execute(
                "SELECT * FROM source_assets WHERE id = ?", (asset_id,)
            ).fetchone()
            assert asset_row is not None
        return SourceRecord(
            asset=self._source_asset_from_row(asset_row),
            blob=self._content_blob_from_row(blob_row),
        )

    def get_source(self, source_id: str) -> SourceRecord:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT a.*, b.id AS b_id, b.sha256, b.storage_path, b.byte_size, "
                "b.parse_status, b.created_at AS b_created_at "
                "FROM source_assets a JOIN content_blobs b ON b.id = a.blob_id "
                "WHERE a.id = ?",
                (source_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"source {source_id!r} was not found")
        return SourceRecord(
            asset=self._source_asset_from_row(row),
            blob=ContentBlob(
                id=row["b_id"],
                sha256=row["sha256"],
                storage_path=row["storage_path"],
                byte_size=row["byte_size"],
                parse_status=row["parse_status"],
                created_at=row["b_created_at"],
            ),
        )

    def list_sources(self, course_id: str) -> list[SourceRecord]:
        self.get_course(course_id)
        with self._connection() as connection:
            ids = connection.execute(
                "SELECT id FROM source_assets WHERE course_id = ? ORDER BY created_at, id",
                (course_id,),
            ).fetchall()
        return [self.get_source(row["id"]) for row in ids]

    def count_blobs(self) -> int:
        with self._connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM content_blobs").fetchone()
        assert row is not None
        return int(row["count"])

    def replace_blocks(
        self,
        source_asset_id: str,
        parser_kind: ParserKind,
        parser_version: str,
        blocks: list[DocumentBlock],
    ) -> None:
        source = self.get_source(source_asset_id)
        if any(
            block.source_asset_id != source_asset_id
            or block.course_id != source.asset.course_id
            or block.parser_kind is not parser_kind
            or block.parser_version != parser_version
            for block in blocks
        ):
            raise ValueError("blocks do not match their source or parser run")
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM document_blocks WHERE source_asset_id = ? "
                "AND parser_kind = ? AND parser_version = ?",
                (source_asset_id, parser_kind.value, parser_version),
            )
            connection.executemany(
                "INSERT INTO document_blocks"
                "(id, source_asset_id, course_id, parser_kind, parser_version, page_number, "
                "section, block_index, text, content_hash) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        block.id,
                        block.source_asset_id,
                        block.course_id,
                        block.parser_kind.value,
                        block.parser_version,
                        block.page_number,
                        block.section,
                        block.block_index,
                        block.text,
                        block.content_hash,
                    )
                    for block in blocks
                ],
            )
            connection.execute(
                "UPDATE content_blobs SET parse_status = ? WHERE id = ?",
                (ParseStatus.READY.value, source.blob.id),
            )

    def mark_parse_failed(self, source_asset_id: str) -> None:
        """Persist a failed parse state without fabricating document blocks."""

        source = self.get_source(source_asset_id)
        with self._connection() as connection:
            connection.execute(
                "UPDATE content_blobs SET parse_status = ? WHERE id = ?",
                (ParseStatus.FAILED.value, source.blob.id),
            )

    def list_blocks_for_source(
        self,
        source_asset_id: str,
        parser_kind: ParserKind,
        parser_version: str,
    ) -> list[DocumentBlock]:
        """Return one source's cached blocks for an exact parser identity."""

        self.get_source(source_asset_id)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM document_blocks "
                "WHERE source_asset_id = ? AND parser_kind = ? AND parser_version = ? "
                "ORDER BY block_index, id",
                (source_asset_id, parser_kind.value, parser_version),
            ).fetchall()
        return [self._document_block_from_row(row) for row in rows]

    def find_blocks_for_blob(
        self,
        sha256: str,
        parser_kind: ParserKind,
        parser_version: str,
    ) -> list[DocumentBlock]:
        """Find one complete cached block run for a content-addressed blob.

        Blocks are source-scoped for citation integrity, so the caller may
        clone the returned chunks onto another asset that shares this blob.
        """

        with self._connection() as connection:
            rows = connection.execute(
                "SELECT b.* FROM document_blocks b "
                "JOIN source_assets a ON a.id = b.source_asset_id "
                "JOIN content_blobs cb ON cb.id = a.blob_id "
                "WHERE cb.sha256 = ? AND cb.parse_status = ? AND b.parser_kind = ? "
                "AND b.parser_version = ? "
                "ORDER BY b.source_asset_id, b.block_index, b.id",
                (sha256, ParseStatus.READY.value, parser_kind.value, parser_version),
            ).fetchall()
        if not rows:
            return []
        source_asset_id = rows[0]["source_asset_id"]
        return [
            self._document_block_from_row(row)
            for row in rows
            if row["source_asset_id"] == source_asset_id
        ]

    def get_block(self, block_id: str) -> DocumentBlock:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM document_blocks WHERE id = ?", (block_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"block {block_id!r} was not found")
        return self._document_block_from_row(row)

    def list_blocks(self, course_id: str) -> list[DocumentBlock]:
        self.get_course(course_id)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM document_blocks WHERE course_id = ? "
                "ORDER BY source_asset_id, parser_kind, parser_version, block_index, id",
                (course_id,),
            ).fetchall()
        return [self._document_block_from_row(row) for row in rows]

    def citation_for_block(self, block_id: str) -> Citation:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT b.*, a.display_name FROM document_blocks b "
                "JOIN source_assets a ON a.id = b.source_asset_id WHERE b.id = ?",
                (block_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"block {block_id!r} was not found")
        return Citation(
            block_id=row["id"],
            source_asset_id=row["source_asset_id"],
            course_id=row["course_id"],
            display_name=row["display_name"],
            page_number=row["page_number"],
            section=row["section"],
            block_index=row["block_index"],
            quote=row["text"],
        )

    @staticmethod
    def _content_blob_from_row(row: sqlite3.Row) -> ContentBlob:
        return ContentBlob(
            id=row["id"],
            sha256=row["sha256"],
            storage_path=row["storage_path"],
            byte_size=row["byte_size"],
            parse_status=row["parse_status"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _source_asset_from_row(row: sqlite3.Row) -> SourceAsset:
        return SourceAsset(
            id=row["id"],
            course_id=row["course_id"],
            blob_id=row["blob_id"],
            document_kind=row["document_kind"],
            origin=row["origin"],
            display_name=row["display_name"],
            trust_level=row["trust_level"],
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _document_block_from_row(row: sqlite3.Row) -> DocumentBlock:
        return DocumentBlock(
            id=row["id"],
            source_asset_id=row["source_asset_id"],
            course_id=row["course_id"],
            parser_kind=row["parser_kind"],
            parser_version=row["parser_version"],
            page_number=row["page_number"],
            section=row["section"],
            block_index=row["block_index"],
            text=row["text"],
            content_hash=row["content_hash"],
        )

    @staticmethod
    def _channel_binding_from_row(row: sqlite3.Row) -> ChannelBinding:
        return ChannelBinding(
            channel=row["channel"],
            external_user_id=row["external_user_id"],
            learner_id=row["learner_id"],
            course_id=row["course_id"],
            session_id=row["session_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

"""Platform-neutral message and reply contracts for channel adapters.

The domain deliberately knows nothing about QQ, WebSocket payloads or SDK
objects.  A platform adapter normalises its event into :class:`CanonicalMessage`
and hands it to an application service; replies are then rendered back by the
adapter.  Keeping this boundary small makes the learning session the source
of truth for both the PC and channel entry points.
"""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class CanonicalMessage(BaseModel):
    """The minimum platform-independent shape consumed by a channel service."""

    model_config = ConfigDict(frozen=True)

    channel: str = Field(min_length=1, max_length=80)
    external_user_id: str = Field(min_length=1, max_length=400)
    message_id: str = Field(min_length=1, max_length=400)
    text: str = Field(default="", max_length=100_000)
    timestamp: str | None = Field(default=None, max_length=200)

    @property
    def openid(self) -> str:
        """Alias for adapters/tests that use QQ's field terminology."""

        return self.external_user_id


class ChannelReply(BaseModel):
    """A safe-to-render reply produced by the channel application service."""

    model_config = ConfigDict(frozen=True)

    channel: str = Field(min_length=1, max_length=80)
    external_user_id: str = Field(min_length=1, max_length=400)
    text: str = Field(min_length=1, max_length=100_000)
    reply_to_message_id: str | None = Field(default=None, max_length=400)
    session_id: str | None = Field(default=None, max_length=200)
    state_version: int | None = Field(default=None, ge=0)
    processed: bool = True
    retryable: bool = False


class ChannelAdapter(Protocol):
    """Minimal protocol boundary implemented by a platform adapter."""

    channel: str

    def normalize_event(
        self, event_type: str, raw: Mapping[str, Any]
    ) -> CanonicalMessage | None: ...

    async def handle_event(
        self, event_type: str, raw: Mapping[str, Any]
    ) -> ChannelReply | None: ...


class ChannelBinding(BaseModel):
    """Durable mapping from a platform user to a StudyPilot learner/session."""

    model_config = ConfigDict(validate_assignment=True)

    channel: str = Field(min_length=1, max_length=80)
    external_user_id: str = Field(min_length=1, max_length=400)
    learner_id: str = Field(min_length=1, max_length=200)
    course_id: str = Field(min_length=1, max_length=100)
    session_id: str | None = Field(default=None, max_length=200)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def study_session_id(self) -> str | None:
        """Readable alias used by channel-facing callers."""

        return self.session_id

    @property
    def openid(self) -> str:
        return self.external_user_id


__all__ = ["CanonicalMessage", "ChannelReply", "ChannelAdapter", "ChannelBinding"]

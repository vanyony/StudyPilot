"""Narrow QQ Bot C2C adapter for the StudyPilot teaching session.

The adapter owns only protocol translation and delivery.  It does not keep a
second copy of a learning state: :class:`ChannelService` reads the existing
SQLite-backed teaching service for every event.  The official SDK is imported
lazily, so all tests can inject a small mock transport and no network or QQ
credential is needed to import StudyPilot.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from inspect import isawaitable
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from studypilot.application.channel import ChannelService
from studypilot.domain.channel import CanonicalMessage, ChannelReply


QQ_C2C_EVENT = "C2C_MESSAGE_CREATE"
DEFAULT_QQ_API_BASE = "https://api.bot.qq.com"
DEFAULT_QQ_TOKEN_URL = "https://api.bot.qq.com/app/getAppAccessToken"
DEFAULT_QQ_MAX_MESSAGE_LENGTH = 4_000


class QQConfigurationError(RuntimeError, ValueError):
    """QQ credentials or runtime settings are missing/invalid."""


class QQDeliveryError(RuntimeError):
    """The learning message was processed but QQ reply delivery failed."""

    def __init__(self, reply: ChannelReply, attempts: int) -> None:
        self.reply = reply
        self.attempts = attempts
        self.retryable = True
        # Do not include the underlying exception: SDK errors can accidentally
        # contain request headers or other sensitive configuration.
        super().__init__(f"QQ reply delivery failed after {attempts} attempt(s)")


class QQBotConfig(BaseModel):
    """Runtime-only QQ Bot settings.  This model is never persisted."""

    model_config = ConfigDict(frozen=True)

    app_id: str = Field(min_length=1, max_length=200)
    app_secret: SecretStr
    api_base: str = Field(default=DEFAULT_QQ_API_BASE, min_length=1, max_length=500)
    token_url: str = Field(default=DEFAULT_QQ_TOKEN_URL, min_length=1, max_length=500)
    gateway_url: str | None = Field(default=None, max_length=2_000)
    max_message_length: int = Field(default=DEFAULT_QQ_MAX_MESSAGE_LENGTH, ge=1, le=20_000)
    send_retries: int = Field(default=0, ge=0, le=3)
    channel: str = Field(default="qq_c2c", min_length=1, max_length=80)

    @property
    def secret(self) -> str:
        """Return the secret for the transport; never use this in logs."""

        return self.app_secret.get_secret_value()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "QQBotConfig":
        source = os.environ if env is None else env

        def first(*names: str) -> str | None:
            for name in names:
                value = source.get(name)
                if value is not None and value.strip():
                    return value.strip()
            return None

        app_id = first("STUDYPILOT_QQ_APP_ID", "QQ_APP_ID")
        secret = first(
            "STUDYPILOT_QQ_APP_SECRET",
            "STUDYPILOT_QQ_SECRET",
            "QQ_APP_SECRET",
            "QQ_SECRET",
        )
        if not app_id:
            raise QQConfigurationError(
                "set STUDYPILOT_QQ_APP_ID before starting the QQ adapter"
            )
        if not secret:
            raise QQConfigurationError(
                "set STUDYPILOT_QQ_APP_SECRET before starting the QQ adapter"
            )

        def integer(name: str, default: int) -> int:
            raw = first(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError as error:
                raise QQConfigurationError(f"{name} must be an integer") from error

        try:
            return cls(
                app_id=app_id,
                app_secret=SecretStr(secret),
                api_base=first("STUDYPILOT_QQ_API_BASE") or DEFAULT_QQ_API_BASE,
                token_url=first("STUDYPILOT_QQ_TOKEN_URL") or DEFAULT_QQ_TOKEN_URL,
                gateway_url=first("STUDYPILOT_QQ_GATEWAY_URL"),
                max_message_length=integer(
                    "STUDYPILOT_QQ_MAX_MESSAGE_LENGTH", DEFAULT_QQ_MAX_MESSAGE_LENGTH
                ),
                send_retries=integer("STUDYPILOT_QQ_SEND_RETRIES", 0),
            )
        except ValidationError as error:
            raise QQConfigurationError("invalid QQ adapter configuration") from error

    from_environment = from_env

    def public_dict(self) -> dict[str, object]:
        """Return diagnostics safe to display; the secret is intentionally absent."""

        return {
            "app_id": self.app_id,
            "api_base": self.api_base,
            "token_url": self.token_url,
            "gateway_url": self.gateway_url,
            "max_message_length": self.max_message_length,
            "send_retries": self.send_retries,
            "channel": self.channel,
            "credentials_configured": True,
        }


class QQSDK(Protocol):
    """Small transport contract implemented by the official SDK wrapper or a test fake."""

    async def send_c2c_text(
        self, openid: str, text: str, *, reply_to: str | None = None
    ) -> Any: ...

    async def start(
        self, on_event: Callable[[str, Mapping[str, Any]], Awaitable[ChannelReply | None]]
    ) -> Any: ...

    async def close(self) -> Any: ...


class QQOfficialSDKTransport:
    """Thin optional wrapper around Tencent's ``qqbot-agent-sdk`` 1.2.2.

    Importing this class does not import or contact the SDK.  Construction
    gives a clear installation error when the optional official SDK is absent.
    """

    SDK_VERSION = "1.2.2"

    def __init__(self, config: QQBotConfig) -> None:
        try:
            from qqbot_agent_sdk import QQApiClient, QQWebSocket, WSCallbacks
            import qqbot_agent_sdk.api_client as sdk_api_client
        except ImportError as error:
            raise QQConfigurationError(
                "install the official qqbot-agent-sdk==1.2.2 extra before starting QQ"
            ) from error
        self.config = config
        self._QQWebSocket = QQWebSocket
        self._WSCallbacks = WSCallbacks
        # qqbot-agent-sdk exposes these module constants for endpoint
        # overrides.  Keep the official current OpenAPI host configurable and
        # avoid relying on an endpoint hard-coded by an older SDK build.
        sdk_api_client.API_BASE = config.api_base
        sdk_api_client.TOKEN_URL = config.token_url
        self.api = QQApiClient(app_id=config.app_id, client_secret=config.secret)
        try:
            import httpx
        except ImportError as error:
            raise QQConfigurationError(
                "install the official qqbot-agent-sdk==1.2.2 extra before starting QQ"
            ) from error
        # The official SDK intentionally requires an injected async HTTP
        # client.  Keep it owned by this transport so REST calls and Gateway
        # URL retrieval work without leaking an event-loop-owned client to the
        # rest of the application.
        self._http_client: Any | None = httpx.AsyncClient()
        self.api.setup(self._http_client)
        self.websocket: Any | None = None
        self._session_id: str | None = None
        self._sequence: int | None = None

    async def start(
        self, on_event: Callable[[str, Mapping[str, Any]], Awaitable[ChannelReply | None]]
    ) -> None:
        """Connect the official Gateway in its SDK-managed background thread."""

        async def callback(event_type: str, raw: dict[str, Any]) -> ChannelReply | None:
            return await on_event(event_type, raw)

        callbacks = self._WSCallbacks(
            on_message_event=callback,
            on_connected=lambda: None,
            on_disconnected=lambda: None,
            on_fatal_error=lambda _code, _message: None,
            get_token=self.api.ensure_token_sync,
            get_session=lambda: (self._session_id, self._sequence),
            set_session=self._set_session,
            set_heartbeat_interval=lambda _seconds: None,
            clear_token=self.api.clear_token,
            fail_pending=lambda _reason: None,
            get_gateway_url=self.api.get_gateway_url_sync,
        )
        self.websocket = self._QQWebSocket(callbacks, log_tag="StudyPilotQQ")
        gateway_url = self.config.gateway_url or await self.api.get_gateway_url()
        self.websocket.start(gateway_url, asyncio.get_running_loop())

    async def send_c2c_text(
        self, openid: str, text: str, *, reply_to: str | None = None
    ) -> Any:
        message = self.api.build_text_body(
            text,
            reply_to=reply_to,
            markdown=False,
            max_length=len(text),
        )
        return await self.api.post_c2c_message(openid, message)

    async def close(self) -> None:
        try:
            if self.websocket is not None:
                await self.websocket.async_stop()
                self.websocket = None
        finally:
            http_client = self._http_client
            self._http_client = None
            if http_client is not None:
                await http_client.aclose()

    def _set_session(self, session_id: str | None, sequence: int | None) -> None:
        self._session_id = session_id
        self._sequence = sequence


class QQC2CAdapter:
    """First-party QQ adapter supporting only text ``C2C_MESSAGE_CREATE``."""

    def __init__(
        self,
        channel_service: ChannelService,
        *,
        sdk: QQSDK | None = None,
        transport: QQSDK | None = None,
        config: QQBotConfig | None = None,
    ) -> None:
        if sdk is not None and transport is not None and sdk is not transport:
            raise QQConfigurationError("pass either sdk or transport, not both")
        injected = sdk or transport
        if config is None:
            if injected is None:
                config = QQBotConfig.from_env()
            else:
                # Injected transports are useful in local tests and do not
                # need real credentials.  This value is never sent anywhere.
                config = QQBotConfig(app_id="injected", app_secret=SecretStr("injected"))
        self.config = config
        self.channel = config.channel
        if getattr(channel_service, "channel", self.channel) != self.channel:
            raise QQConfigurationError("channel service and QQ adapter channels must match")
        self.channel_service = channel_service
        self.sdk: QQSDK = injected or QQOfficialSDKTransport(config)

    @classmethod
    def from_env(
        cls,
        channel_service: ChannelService,
        *,
        sdk: QQSDK | None = None,
    ) -> "QQC2CAdapter":
        config = QQBotConfig.from_env()
        return cls(channel_service, sdk=sdk, config=config)

    def normalize_event(
        self, event_type: str, raw: Mapping[str, Any]
    ) -> CanonicalMessage | None:
        """Normalize one official dispatch payload, ignoring non-C2C/non-text events."""

        if str(event_type) != QQ_C2C_EVENT or not isinstance(raw, Mapping):
            return None
        message_id = raw.get("id")
        author = raw.get("author")
        openid = author.get("user_openid") if isinstance(author, Mapping) else None
        if not isinstance(message_id, str) or not message_id.strip():
            return None
        if not isinstance(openid, str) or not openid.strip():
            return None
        try:
            message_type = int(raw.get("message_type", 0) or 0)
        except (TypeError, ValueError):
            return None
        if message_type != 0:
            return None
        content = raw.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            return None
        timestamp = raw.get("timestamp")
        if timestamp is not None and not isinstance(timestamp, str):
            timestamp = str(timestamp)
        try:
            return CanonicalMessage(
                channel=self.config.channel,
                external_user_id=openid.strip(),
                message_id=message_id.strip(),
                text=content,
                timestamp=timestamp,
            )
        except ValidationError:
            return None

    def process_event(
        self, event_type: str, raw: Mapping[str, Any]
    ) -> ChannelReply | None:
        message = self.normalize_event(event_type, raw)
        return self.channel_service.handle_message(message) if message is not None else None

    async def handle_event(
        self, event_type: str, raw: Mapping[str, Any]
    ) -> ChannelReply | None:
        """Process and send one event; delivery errors remain explicit for retry."""

        message = self.normalize_event(event_type, raw)
        if message is None:
            return None
        reply = self.channel_service.handle_message(message)
        return await self.send_reply(reply)

    async def send_reply(
        self, reply: ChannelReply, *, retries: int | None = None
    ) -> ChannelReply:
        if reply.channel != self.config.channel:
            raise QQConfigurationError("reply channel does not match the QQ C2C adapter")
        safe_text = truncate_qq_text(reply.text, self.config.max_message_length)
        safe_reply = reply.model_copy(update={"text": safe_text})
        retry_count = self.config.send_retries if retries is None else max(0, retries)
        attempts = retry_count + 1
        for attempt in range(attempts):
            try:
                result = self.sdk.send_c2c_text(
                    safe_reply.external_user_id,
                    safe_reply.text,
                    reply_to=safe_reply.reply_to_message_id,
                )
                if isawaitable(result):
                    await result
                return safe_reply
            except Exception:
                if attempt + 1 >= attempts:
                    raise QQDeliveryError(safe_reply, attempts) from None
        raise AssertionError("unreachable")

    async def start(self) -> None:
        result = self.sdk.start(self.handle_event)
        if isawaitable(result):
            await result

    run = start

    async def stop(self) -> None:
        result = self.sdk.close()
        if isawaitable(result):
            await result


QQChannelAdapter = QQC2CAdapter


def create_qq_adapter(app: Any, *, sdk: QQSDK | None = None) -> QQC2CAdapter:
    """Build an adapter from an existing FastAPI app's channel service."""

    return QQC2CAdapter(app.state.channel_service, sdk=sdk)


def truncate_qq_text(text: str, max_length: int) -> str:
    """Conservatively truncate a reply before the official API sees it."""

    if max_length < 1:
        return ""
    if len(text) <= max_length:
        return text
    suffix = "…"
    if max_length <= len(suffix):
        return suffix[:max_length]
    return text[: max_length - len(suffix)].rstrip() + suffix


__all__ = [
    "QQ_C2C_EVENT",
    "QQBotConfig",
    "QQSDK",
    "QQOfficialSDKTransport",
    "QQC2CAdapter",
    "QQChannelAdapter",
    "create_qq_adapter",
    "QQConfigurationError",
    "QQDeliveryError",
    "truncate_qq_text",
]

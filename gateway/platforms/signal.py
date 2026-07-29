"""Signal messenger platform adapter.

Connects to a signal-cli daemon running in HTTP mode.
Inbound messages arrive via SSE (Server-Sent Events) streaming.
Outbound messages and actions use JSON-RPC 2.0 over HTTP.

Based on PR #268 by ibhagwan, rebuilt with bug fixes.

Requires:
  - signal-cli installed and running: signal-cli daemon --http 127.0.0.1:8080
  - SIGNAL_HTTP_URL and SIGNAL_ACCOUNT environment variables set
"""

import asyncio
import base64
import json
import logging
import mimetypes
import os
import random
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote

import httpx

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
    cache_image_from_bytes,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_url,
    classify_send_error,
)
from gateway.platforms.helpers import redact_phone
from tools.audio_container import CONTAINER_TO_EXT, sniff_container
from gateway.platforms.signal_format import markdown_to_signal
from gateway.platforms.signal_rate_limit import (
    SIGNAL_BATCH_PACING_NOTICE_THRESHOLD,
    SIGNAL_MAX_ATTACHMENTS_PER_MSG,
    SIGNAL_RATE_LIMIT_MAX_ATTEMPTS,
    SignalRateLimitError,
    _extract_retry_after_seconds,
    _format_wait,
    _is_signal_rate_limit_error,
    _signal_send_timeout,
    get_scheduler,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIGNAL_MAX_ATTACHMENT_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_MESSAGE_LENGTH = 8000  # Signal message size limit
TYPING_INTERVAL = 8.0  # seconds between typing indicator refreshes
SSE_RETRY_DELAY_INITIAL = 2.0
SSE_RETRY_DELAY_MAX = 60.0
HEALTH_CHECK_INTERVAL = 30.0  # seconds between health checks
HEALTH_CHECK_STALE_THRESHOLD = 120.0  # seconds without SSE activity before concern
# ``listGroups(groupId=[...])`` forces signal-cli to refresh that Group V2
# record from Signal.  On real accounts this can take substantially longer than
# the normal 30-second RPC budget, so description refreshes use a dedicated
# background-only timeout and never delay inbound message handling.
SIGNAL_GROUP_DESCRIPTION_RPC_TIMEOUT = 120.0
SIGNAL_GROUP_DESCRIPTION_RETRY_DELAY = 30.0
SIGNAL_GROUP_DESCRIPTION_MAX_RETRY_DELAY = 300.0
SIGNAL_GROUP_DESCRIPTION_MAX_RETRIES = 6
SIGNAL_GROUP_DESCRIPTION_CACHE_LIMIT = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GroupDescriptionRefreshRequest:
    """One coalesced group-description invalidation."""

    generation: int
    target_revision: Optional[int]
    revision_floor: Optional[int]
    revisionless_generation: Optional[int]


def _parse_comma_list(value: str) -> List[str]:
    """Split a comma-separated string into a list, stripping whitespace."""
    return [v.strip() for v in value.split(",") if v.strip()]


def _is_truthy(value: Any, default: bool = False) -> bool:
    """Parse common config/env truthy values."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return bool(value)


def _guess_extension(data: bytes) -> str:
    """Guess file extension from magic bytes.

    Android Signal delivers voice notes as raw ADTS AAC frames, which share
    the ``0xFF 0xFx`` sync word with MPEG-1/2 Layer 3 (MP3). The byte-1
    layout disambiguates: ADTS packs ``ID layer protection_absent`` into
    bits 3-0, where ``ID`` is 0 for MPEG-2/4 AAC and ``layer`` is always
    0 for ADTS. A real MP3 frame has ``ID=1`` and ``layer`` in {1, 2, 3}.
    """
    if data[:4] == b"\x89PNG":
        return ".png"
    if data[:2] == b"\xff\xd8":
        return ".jpg"
    if data[:4] == b"GIF8":
        return ".gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:4] == b"%PDF":
        return ".pdf"
    # Audio/AV containers: delegate to the shared central sniffer
    # (tools/audio_container.py) — ONE module owns magic-byte container
    # detection. It handles the brand/form-type disambiguations this
    # function used to carry locally: RIFF/WAVE vs WEBP (WEBP is claimed
    # above, before delegation), ftyp audio brands ("M4A ", "M4B ") vs
    # video brands (isom/mp42/avc1/qt), and MP3 vs ADTS AAC sync words.
    container = sniff_container(data)
    if container is not None:
        return CONTAINER_TO_EXT[container]
    if data[:2] == b"PK":
        return ".zip"
    return ".bin"


def _is_image_ext(ext: str) -> bool:
    return ext.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def _is_audio_ext(ext: str) -> bool:
    return ext.lower() in {".mp3", ".wav", ".ogg", ".m4a", ".aac"}


_EXT_TO_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp",
    ".ogg": "audio/ogg", ".mp3": "audio/mpeg", ".wav": "audio/wav",
    ".m4a": "audio/mp4", ".aac": "audio/aac",
    ".mp4": "video/mp4", ".pdf": "application/pdf", ".zip": "application/zip",
}


def _ext_to_mime(ext: str) -> str:
    """Map file extension to MIME type."""
    return _EXT_TO_MIME.get(ext.lower(), "application/octet-stream")


def _remux_aac_to_m4a(aac_data: bytes) -> Optional[Tuple[bytes, str]]:
    """Losslessly remux raw ADTS AAC bytes into an MP4 (.m4a) container.

    Used by the Signal attachment cache so Android voice notes land on disk
    in a container that every major STT API (Groq, OpenAI, xAI, Mistral
    Voxtral) will accept. ``ffmpeg -c:a copy`` is a single demux/remux —
    no re-encode, no quality loss, sub-100ms for typical voice-note sizes.

    Returns ``(m4a_bytes, ".m4a")`` on success, or ``None`` if ffmpeg is
    missing, input is invalid, or remux fails for any reason. Callers
    must treat ``None`` as "pass through unchanged" and not raise.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        # Common Homebrew/local prefixes on macOS dev hosts.
        for prefix in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
            if os.path.isfile(prefix) and os.access(prefix, os.X_OK):
                ffmpeg = prefix
                break
    if not ffmpeg:
        logger.debug("Signal: ffmpeg not found, skipping AAC→M4A remux")
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as src:
            src.write(aac_data)
            src_path = src.name
        dst_path = src_path[:-4] + ".m4a"
        try:
            proc = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", src_path,
                 "-c:a", "copy", "-movflags", "+faststart", dst_path],
                capture_output=True, timeout=10,
            )
            if proc.returncode != 0:
                logger.warning(
                    "Signal: AAC→M4A remux failed (ffmpeg exit %d): %s",
                    proc.returncode, proc.stderr.decode("utf-8", "replace")[:300],
                )
                return None
            with open(dst_path, "rb") as f:
                return f.read(), ".m4a"
        finally:
            for p in (src_path, dst_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
    except subprocess.TimeoutExpired:
        logger.warning("Signal: AAC→M4A remux timed out (>10s)")
        return None
    except Exception:
        logger.exception("Signal: AAC→M4A remux error")
        return None


def _file_to_signal_data_uri(file_path: str) -> str:
    """Convert a local file into a signal-cli-compatible attachment data URI."""
    path = Path(file_path)
    mime_type = mimetypes.guess_type(path.name)[0] or _ext_to_mime(path.suffix)
    encoded_name = quote(path.name, safe="")
    encoded_data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};filename={encoded_name};base64,{encoded_data}"


def _is_data_uri_attachment(file_ref: str) -> bool:
    return isinstance(file_ref, str) and file_ref.startswith("data:")


def _signal_attachment_size(file_ref: str) -> int:
    if _is_data_uri_attachment(file_ref):
        header, _, payload = file_ref.partition(",")
        if not payload:
            return 0
        if ";base64" in header.lower():
            return len(base64.b64decode(payload))
        return len(unquote(payload).encode("utf-8"))
    return Path(file_ref).stat().st_size


def _normalize_signal_attachment(file_ref: str) -> str:
    """Return an attachment reference signal-cli accepts over JSON-RPC."""
    if _is_data_uri_attachment(file_ref):
        return file_ref
    return _file_to_signal_data_uri(file_ref)


def _render_mentions(text: str, mentions: list) -> str:
    """Replace Signal mention placeholders (\\uFFFC) with readable @identifiers.

    Signal encodes @mentions as the Unicode object replacement character
    with out-of-band metadata containing the mentioned user's UUID/number.
    """
    if not mentions or "\uFFFC" not in text:
        return text
    # Sort mentions by start position (reverse) to replace from end to start
    # so indices don't shift as we replace
    sorted_mentions = sorted(mentions, key=lambda m: m.get("start", 0), reverse=True)
    for mention in sorted_mentions:
        start = mention.get("start", 0)
        length = mention.get("length", 1)
        # Use the mention's number or UUID as the replacement
        identifier = mention.get("number") or mention.get("uuid") or "user"
        replacement = f"@{identifier}"
        text = text[:start] + replacement + text[start + length:]
    return text


def _signal_mentions_include_bot(mentions: list, bot_identifiers: set[str]) -> bool:
    """Return True when Signal mention metadata targets this bot."""
    if not mentions or not bot_identifiers:
        return False
    for mention in mentions:
        if not isinstance(mention, dict):
            continue
        values = {
            str(mention.get(key) or "").strip()
            for key in ("number", "uuid", "serviceId", "recipient")
        }
        values.discard("")
        if values & bot_identifiers:
            return True
    return False


def _is_signal_service_id(value: str) -> bool:
    """Return True if *value* already looks like a Signal service identifier."""
    if not value:
        return False
    if value.startswith("PNI:") or value.startswith("u:"):
        return True
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _looks_like_e164_number(value: str) -> bool:
    """Return True for a plausible E.164 phone number."""
    if not value or not value.startswith("+"):
        return False
    digits = value[1:]
    return digits.isdigit() and 7 <= len(digits) <= 15


def check_signal_requirements() -> bool:
    """Check if Signal runtime dependencies are available."""
    return True


def validate_signal_config(config: PlatformConfig) -> bool:
    """Check if Signal has enough config to connect."""
    extra = getattr(config, "extra", {}) or {}
    http_url = (extra.get("http_url", "") or os.getenv("SIGNAL_HTTP_URL", "")).strip()
    account = (extra.get("account", "") or os.getenv("SIGNAL_ACCOUNT", "")).strip()
    return bool(http_url and account)


# ---------------------------------------------------------------------------
# Signal Adapter
# ---------------------------------------------------------------------------

class SignalAdapter(BasePlatformAdapter):
    """Signal messenger adapter using signal-cli HTTP daemon."""

    platform = Platform.SIGNAL
    # Signal has no real edit API for already-sent messages. Mark it explicitly
    # so streaming suppresses the visible cursor instead of leaving a stale tofu
    # square behind in chat clients when edit attempts fail.
    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SIGNAL)

        extra = config.extra or {}
        self.http_url = extra.get("http_url", "http://127.0.0.1:8080").rstrip("/")
        self.account = extra.get("account", "")
        self.ignore_stories = extra.get("ignore_stories", True)

        # Parse allowlists — group policy is derived from presence of group allowlist
        group_allowed_str = os.getenv("SIGNAL_GROUP_ALLOWED_USERS", "")
        self.group_allow_from = set(_parse_comma_list(group_allowed_str))

        # Mention filter — only respond in groups when the bot account is @mentioned.
        # Read from config extra first, then SIGNAL_REQUIRE_MENTION env var.
        _rm_cfg = extra.get("require_mention")
        self.require_mention = _is_truthy(
            _rm_cfg,
            default=_is_truthy(os.getenv("SIGNAL_REQUIRE_MENTION"), default=False),
        )

        # DM allowlist — mirrors SIGNAL_ALLOWED_USERS checked by run.py.
        # Stored here so the reaction hooks can skip unauthorized senders
        # (reactions fire before run.py's auth gate, so without this check
        # every inbound DM from any contact gets a 👀 reaction).
        # "*" means all users allowed (open mode); empty means no restriction
        # recorded at adapter level (run.py still enforces auth separately).
        dm_allowed_str = os.getenv("SIGNAL_ALLOWED_USERS", "*")
        self.dm_allow_from = set(_parse_comma_list(dm_allowed_str))

        # HTTP client
        self.client: Optional[httpx.AsyncClient] = None

        # Background tasks
        self._sse_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._typing_tasks: Dict[str, asyncio.Task] = {}
        # Per-chat typing-indicator backoff. When signal-cli reports
        # NETWORK_FAILURE (recipient offline / unroutable), base.py's
        # _keep_typing refresh loop would otherwise hammer sendTyping every
        # ~2s indefinitely, producing WARNING-level log spam and pointless
        # RPC traffic. We track consecutive failures per chat and skip the
        # RPC during a cooldown window instead.
        self._typing_failures: Dict[str, int] = {}
        self._typing_skip_until: Dict[str, float] = {}
        self._running = False
        self._last_sse_activity = 0.0
        self._sse_response: Optional[httpx.Response] = None

        # Normalize account for self-message filtering
        self._account_normalized = self.account.strip()

        # Track recently sent message timestamps to prevent echo-back loops
        # in Note to Self / self-chat mode and linked-device group sync-sents.
        # OrderedDict[timestamp_ms -> insertion_monotonic_seconds] gives us
        # LRU eviction (popitem(last=False) drops oldest) plus a TTL so that
        # under chatty groups a still-pending echo cannot be evicted just
        # because >50 outbounds happened. With a 5-minute TTL the cap only
        # matters for runaway producers, not normal traffic bursts.
        self._recent_sent_timestamps: "OrderedDict[int, float]" = OrderedDict()
        self._max_recent_timestamps = 512
        self._recent_sent_ttl_seconds = 300.0
        # Keep a separate bounded cache of outbound Signal message timestamps.
        # Signal quote.id is the timestamp of the quoted message, so this lets
        # inbound replies identify that the user replied to a message sent by
        # this bot even after the self-sync echo was filtered above.
        # OrderedDict (not set) so the cap evicts the OLDEST timestamp in FIFO
        # order — a plain set.pop() removes an arbitrary element, which could
        # drop a still-recent timestamp and miss a genuine reply-to-own-message.
        self._sent_message_timestamps: "OrderedDict[str, None]" = OrderedDict()
        self._max_sent_message_timestamps = 500
        # Signal increasingly exposes ACI/PNI UUIDs as stable recipient IDs.
        # Keep a best-effort mapping so outbound sends can upgrade from a
        # phone number to the corresponding UUID when signal-cli prefers it.
        self._recipient_uuid_by_number: Dict[str, str] = {}
        self._recipient_number_by_uuid: Dict[str, str] = {}
        self._recipient_cache_lock = asyncio.Lock()

        # Signal's receive JSON exposes a group's current name, revision, and
        # whether the envelope is a group update, but not its description.  The
        # description is available through listGroups. Keep a revision-aware cache
        # so long-lived gateways learn changes from UPDATE envelopes (or a later
        # DELIVER envelope with a newer revision) without startup scans or blocking
        # user messages on a slow RPC. Suppress cached values while invalidated.
        self._group_descriptions: Dict[str, Optional[str]] = {}
        self._group_description_revisions: Dict[str, int] = {}
        # One adapter-wide worker serializes the expensive forced Group V2
        # refreshes. Each request records the newest observed Signal revision
        # and an exclusive revision floor. The floor makes
        # a revisionless UPDATE require an actual revision advance when a
        # listGroups response supplies revisions; generations still preserve
        # ordering when the deployed signal-cli omits them.
        self._group_description_refresh_requests: Dict[
            str, _GroupDescriptionRefreshRequest
        ] = {}
        self._group_description_retry_after: Dict[str, float] = {}
        self._group_description_retry_attempts: Dict[str, int] = {}
        self._group_description_refresh_event = asyncio.Event()
        self._group_description_refresh_task: Optional[asyncio.Task] = None

        logger.info("Signal adapter initialized: url=%s account=%s groups=%s",
                     self.http_url, redact_phone(self.account),
                     "enabled" if self.group_allow_from else "disabled")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to signal-cli daemon and start SSE listener."""
        if not self.http_url or not self.account:
            logger.error("Signal: SIGNAL_HTTP_URL and SIGNAL_ACCOUNT are required")
            return False

        # Acquire scoped lock to prevent duplicate Signal listeners for the same phone
        lock_acquired = False
        try:
            if not self._acquire_platform_lock('signal-phone', self.account, 'Signal account'):
                return False
            lock_acquired = True
        except Exception as e:
            logger.warning("Signal: Could not acquire phone lock (non-fatal): %s", e)

        # Tighter keepalive so idle CLOSE_WAIT drains promptly (#18451).
        from gateway.platforms._http_client_limits import platform_httpx_limits
        self.client = httpx.AsyncClient(timeout=30.0, limits=platform_httpx_limits())
        try:
            # Health check — verify signal-cli daemon is reachable
            try:
                resp = await self.client.get(f"{self.http_url}/api/v1/check", timeout=10.0)
                if resp.status_code != 200:
                    logger.error("Signal: health check failed (status %d)", resp.status_code)
                    return False
            except Exception as e:
                logger.error("Signal: cannot reach signal-cli at %s: %s", self.http_url, e)
                return False

            self._running = True
            self._last_sse_activity = time.time()
            self._sse_task = asyncio.create_task(self._sse_listener())
            self._health_monitor_task = asyncio.create_task(self._health_monitor())

            logger.info("Signal: connected to %s", self.http_url)
            return True
        finally:
            if not self._running:
                if self.client:
                    await self.client.aclose()
                    self.client = None
                if lock_acquired:
                    self._release_platform_lock()

    async def disconnect(self) -> None:
        """Stop SSE listener and clean up."""
        self._running = False

        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass

        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass

        await self._reset_group_description_state()

        # Cancel all typing tasks
        for task in self._typing_tasks.values():
            task.cancel()
        self._typing_tasks.clear()

        if self.client:
            await self.client.aclose()
            self.client = None

        self._release_platform_lock()

        logger.info("Signal: disconnected")

    # ------------------------------------------------------------------
    # SSE Streaming (inbound messages)
    # ------------------------------------------------------------------

    async def _sse_listener(self) -> None:
        """Listen for SSE events from signal-cli daemon."""
        url = f"{self.http_url}/api/v1/events?account={quote(self.account, safe='')}"
        backoff = SSE_RETRY_DELAY_INITIAL
        connected_once = False

        while self._running:
            try:
                logger.debug("Signal SSE: connecting to %s", url)
                async with self.client.stream(
                    "GET", url,
                    headers={"Accept": "text/event-stream"},
                    timeout=None,
                ) as response:
                    if connected_once:
                        # Signal's SSE endpoint does not replay the gap. Any
                        # cached description may have changed while disconnected,
                        # so make the next message refresh it instead of treating
                        # a revisionless DELIVER as proof that the cache is current.
                        await self._reset_group_description_state()
                    connected_once = True
                    self._sse_response = response
                    backoff = SSE_RETRY_DELAY_INITIAL  # Reset on successful connection
                    self._last_sse_activity = time.time()
                    logger.info("Signal SSE: connected")

                    buffer = ""
                    async for chunk in response.aiter_text():
                        if not self._running:
                            break
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            # SSE keepalive comments (":") prove the connection
                            # is alive — update activity so the health monitor
                            # doesn't report false idle warnings.
                            if line.startswith(":"):
                                self._last_sse_activity = time.time()
                                continue
                            # Parse SSE data lines
                            if line.startswith("data:"):
                                data_str = line[5:].strip()
                                if not data_str:
                                    continue
                                self._last_sse_activity = time.time()
                                try:
                                    data = json.loads(data_str)
                                    await self._handle_envelope(data)
                                except json.JSONDecodeError:
                                    logger.debug("Signal SSE: invalid JSON: %s", data_str[:100])
                                except Exception:
                                    logger.exception("Signal SSE: error handling event")

            except asyncio.CancelledError:
                break
            except httpx.HTTPError as e:
                if self._running:
                    logger.warning("Signal SSE: HTTP error: %s (reconnecting in %.0fs)", e, backoff)
            except Exception as e:
                if self._running:
                    logger.warning("Signal SSE: error: %s (reconnecting in %.0fs)", e, backoff)

            if self._running:
                # Add 20% jitter to prevent thundering herd on reconnection
                jitter = backoff * 0.2 * random.random()
                await asyncio.sleep(backoff + jitter)
                backoff = min(backoff * 2, SSE_RETRY_DELAY_MAX)

        self._sse_response = None

    # ------------------------------------------------------------------
    # Health Monitor
    # ------------------------------------------------------------------

    async def _health_monitor(self) -> None:
        """Monitor SSE connection health and force reconnect if stale."""
        while self._running:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            if not self._running:
                break

            elapsed = time.time() - self._last_sse_activity
            if elapsed > HEALTH_CHECK_STALE_THRESHOLD:
                logger.warning("Signal: SSE idle for %.0fs, checking daemon health", elapsed)
                try:
                    resp = await self.client.get(
                        f"{self.http_url}/api/v1/check", timeout=10.0
                    )
                    if resp.status_code == 200:
                        # Daemon is alive but SSE is idle — update activity to
                        # avoid repeated warnings (connection may just be quiet)
                        self._last_sse_activity = time.time()
                        logger.debug("Signal: daemon healthy, SSE idle")
                    else:
                        logger.warning("Signal: health check failed (%d), forcing reconnect", resp.status_code)
                        self._force_reconnect()
                except Exception as e:
                    logger.warning("Signal: health check error: %s, forcing reconnect", e)
                    self._force_reconnect()

    def _force_reconnect(self) -> None:
        """Force SSE reconnection by closing the current response."""
        if self._sse_response and not self._sse_response.is_stream_consumed:
            try:
                task = asyncio.create_task(self._sse_response.aclose())
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except Exception:
                pass
            self._sse_response = None

    # ------------------------------------------------------------------
    # Group description metadata
    # ------------------------------------------------------------------

    async def _reset_group_description_state(self) -> None:
        """Cancel pending refreshes and invalidate metadata after a receive gap."""
        description_task = self._group_description_refresh_task
        if (
            description_task is not None
            and description_task is not asyncio.current_task()
            and not description_task.done()
        ):
            description_task.cancel()
            await asyncio.gather(description_task, return_exceptions=True)
        self._group_description_refresh_task = None
        self._group_description_refresh_requests.clear()
        self._group_description_retry_after.clear()
        self._group_description_retry_attempts.clear()
        self._group_description_refresh_event.clear()
        self._group_descriptions.clear()
        self._group_description_revisions.clear()

    @staticmethod
    def _group_revision(group_info: Any) -> Optional[int]:
        """Return a non-negative Signal group revision when one is present."""
        if not isinstance(group_info, dict):
            return None
        raw_revision = group_info.get("revision")
        if raw_revision is None:
            return None
        try:
            revision = int(raw_revision)
        except (TypeError, ValueError):
            return None
        return revision if revision >= 0 else None

    def _group_description_context(
        self,
        group_id: str,
        revision: Optional[int],
    ) -> Tuple[Optional[str], bool]:
        """Return (description, authoritative) for the current envelope.

        Any queued invalidation suppresses the old value, including revisionless
        UPDATE events. Missing context is preferable to injecting a purpose that
        the group has already changed. The boolean distinguishes a confirmed
        empty description from metadata that is merely unavailable or stale.
        """
        if (
            group_id not in self._group_descriptions
            or group_id in self._group_description_refresh_requests
        ):
            return None, False
        cached_revision = self._group_description_revisions.get(group_id, -1)
        if revision is not None and revision > cached_revision:
            return None, False
        description = self._group_descriptions.pop(group_id)
        self._group_descriptions[group_id] = description
        if group_id in self._group_description_revisions:
            cached_revision = self._group_description_revisions.pop(group_id)
            self._group_description_revisions[group_id] = cached_revision
        return description, True

    def _store_group_description(
        self,
        group_id: str,
        description: Optional[str],
        revision: Optional[int],
    ) -> None:
        """Publish one authoritative result and bound long-lived cache state."""
        self._group_descriptions.pop(group_id, None)
        self._group_descriptions[group_id] = description
        if revision is not None:
            self._group_description_revisions.pop(group_id, None)
            self._group_description_revisions[group_id] = revision

        while len(self._group_descriptions) > SIGNAL_GROUP_DESCRIPTION_CACHE_LIMIT:
            oldest_group = next(iter(self._group_descriptions))
            self._group_descriptions.pop(oldest_group, None)
            self._group_description_revisions.pop(oldest_group, None)

    def _maybe_schedule_group_description_refresh(
        self,
        group_id: str,
        group_info: Any,
    ) -> None:
        """Refresh a missing or revision-stale group description in background."""
        if not group_id:
            return
        revision = self._group_revision(group_info)
        is_update = (
            isinstance(group_info, dict)
            and str(group_info.get("type") or "").upper() == "UPDATE"
        )
        cached_revision = self._group_description_revisions.get(group_id, -1)
        pending = self._group_description_refresh_requests.get(group_id)
        pending_revision = pending.target_revision if pending is not None else None
        pending_revision_floor = pending.revision_floor if pending is not None else None
        revisionless_generation = (
            pending.revisionless_generation if pending is not None else None
        )
        highest_requested_revision = max(
            cached_revision,
            pending_revision if pending_revision is not None else -1,
            pending_revision_floor if pending_revision_floor is not None else -1,
        )

        if group_id not in self._group_descriptions:
            should_enqueue = (
                pending is None
                or (revision is not None and revision > highest_requested_revision)
                or (is_update and revision is None)
            )
        elif revision is not None:
            should_enqueue = revision > highest_requested_revision
        else:
            # An UPDATE without a revision is still an invalidation. Increment a
            # local generation even when another RPC is in flight so it cannot
            # be mistaken for the request that RPC already covers.
            should_enqueue = is_update

        if not should_enqueue:
            return

        generation = (pending.generation if pending is not None else 0) + 1
        requested_revision = max(
            revision if revision is not None else -1,
            pending_revision if pending_revision is not None else -1,
        )
        revision_floor = pending_revision_floor
        if (
            revision is not None
            and revision_floor is not None
            and revision >= revision_floor
        ):
            # A later explicit revision gives the request a concrete inclusive
            # target, so the older revisionless floor is no longer needed.
            revision_floor = None
            revisionless_generation = None
        if is_update and revision is None:
            required_advance_after = max(
                cached_revision,
                pending_revision if pending_revision is not None else -1,
                pending_revision_floor if pending_revision_floor is not None else -1,
            )
            revision_floor = (
                required_advance_after if required_advance_after >= 0 else None
            )
            revisionless_generation = generation
        self._group_description_refresh_requests[group_id] = (
            _GroupDescriptionRefreshRequest(
                generation=generation,
                target_revision=(
                    requested_revision if requested_revision >= 0 else None
                ),
                revision_floor=revision_floor,
                revisionless_generation=revisionless_generation,
            )
        )
        # A new invalidation supersedes any retry budget/backoff accumulated by
        # an older request for this group.
        self._group_description_retry_after.pop(group_id, None)
        self._group_description_retry_attempts[group_id] = 0
        self._group_description_refresh_event.set()

        existing = self._group_description_refresh_task
        if existing is not None and not existing.done():
            return

        task = asyncio.create_task(self._run_group_description_refreshes())
        self._group_description_refresh_task = task
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _run_group_description_refreshes(self) -> None:
        """Drain description invalidations with one fair, retrying worker."""
        current_task = asyncio.current_task()
        try:
            while self._group_description_refresh_requests:
                now = time.monotonic()
                group_id = next(
                    (
                        pending_group
                        for pending_group in self._group_description_refresh_requests
                        if self._group_description_retry_after.get(pending_group, 0.0)
                        <= now
                    ),
                    None,
                )
                if group_id is None:
                    retry_at = min(
                        self._group_description_retry_after.get(pending_group, now)
                        for pending_group in self._group_description_refresh_requests
                    )
                    self._group_description_refresh_event.clear()
                    try:
                        await asyncio.wait_for(
                            self._group_description_refresh_event.wait(),
                            timeout=max(0.0, retry_at - now),
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue

                request = self._group_description_refresh_requests[group_id]
                generation = request.generation
                target_revision = request.target_revision
                revision_floor = request.revision_floor
                try:
                    refreshed, description, fetched_revision = (
                        await self._refresh_group_description(group_id)
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.debug(
                        "Signal: unexpected group description refresh failure",
                        exc_info=True,
                    )
                    refreshed, description, fetched_revision = False, None, None

                current_request = self._group_description_refresh_requests.get(
                    group_id
                )
                known_revision = self._group_description_revisions.get(group_id)
                minimum_revision = max(
                    (
                        revision
                        for revision in (target_revision, known_revision)
                        if revision is not None
                    ),
                    default=None,
                )
                revision_lagging = (
                    refreshed
                    and fetched_revision is not None
                    and (
                        (
                            minimum_revision is not None
                            and fetched_revision < minimum_revision
                        )
                        or (
                            revision_floor is not None
                            and fetched_revision <= revision_floor
                        )
                    )
                )
                if refreshed and not revision_lagging:
                    self._group_description_retry_after.pop(group_id, None)
                    self._group_description_retry_attempts.pop(group_id, None)
                    effective_revision = (
                        fetched_revision
                        if fetched_revision is not None
                        else target_revision
                    )
                    if effective_revision is not None and known_revision is not None:
                        effective_revision = max(
                            self._group_description_revisions.get(group_id, -1),
                            effective_revision,
                        )
                    # Publish only after revision validation. A targeted
                    # listGroups call can briefly return the daemon's prior
                    # stored revision while an update is still being applied.
                    self._store_group_description(
                        group_id,
                        description,
                        effective_revision,
                    )
                    if (
                        current_request is not None
                        and current_request.generation == generation
                    ):
                        self._group_description_refresh_requests.pop(group_id, None)
                    elif current_request is not None:
                        if (
                            effective_revision is not None
                            and current_request.revisionless_generation is not None
                            and current_request.revisionless_generation > generation
                        ):
                            # The newer revisionless invalidation arrived while
                            # this RPC was running. Its follow-up RPC must advance
                            # beyond the revision this older RPC just confirmed.
                            current_request = _GroupDescriptionRefreshRequest(
                                generation=current_request.generation,
                                target_revision=current_request.target_revision,
                                revision_floor=max(
                                    current_request.revision_floor
                                    if current_request.revision_floor is not None
                                    else -1,
                                    effective_revision,
                                ),
                                revisionless_generation=(
                                    current_request.revisionless_generation
                                ),
                            )
                            self._group_description_refresh_requests[group_id] = (
                                current_request
                            )
                        # A newer invalidation arrived mid-RPC. Move it behind
                        # other groups so one noisy group cannot starve the queue.
                        self._group_description_refresh_requests.pop(group_id, None)
                        self._group_description_refresh_requests[group_id] = current_request
                else:
                    if revision_lagging:
                        logger.debug(
                            "Signal: group description refresh lagged requested revision"
                        )
                    retry_attempt = (
                        self._group_description_retry_attempts.get(group_id, 0) + 1
                    )
                    if retry_attempt >= SIGNAL_GROUP_DESCRIPTION_MAX_RETRIES:
                        # Stop a permanently failing group from keeping the
                        # worker and stale metadata alive forever. A later
                        # message starts a fresh bounded attempt series.
                        self._group_description_refresh_requests.pop(group_id, None)
                        self._group_description_retry_after.pop(group_id, None)
                        self._group_description_retry_attempts.pop(group_id, None)
                        self._group_descriptions.pop(group_id, None)
                        self._group_description_revisions.pop(group_id, None)
                    elif current_request is not None:
                        self._group_description_retry_attempts[group_id] = retry_attempt
                        retry_delay = min(
                            SIGNAL_GROUP_DESCRIPTION_RETRY_DELAY
                            * (2 ** max(0, retry_attempt - 1)),
                            SIGNAL_GROUP_DESCRIPTION_MAX_RETRY_DELAY,
                        )
                        self._group_description_retry_after[group_id] = (
                            time.monotonic() + retry_delay
                        )
                        self._group_description_refresh_requests.pop(group_id, None)
                        self._group_description_refresh_requests[group_id] = current_request
        finally:
            if self._group_description_refresh_task is current_task:
                self._group_description_refresh_task = None

    async def _refresh_group_description(
        self, group_id: str
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """Fetch one group's current description from signal-cli.

        Supplying ``groupId`` is important: it avoids serializing every group
        while asking signal-cli to refresh the one record whose revision changed.
        The RPC is serialized across groups because signal-cli's forced Group V2
        refresh is expensive and concurrent refreshes only increase contention.
        """
        groups = await self._rpc(
            "listGroups",
            {
                "account": self.account,
                "groupId": [group_id],
            },
            rpc_id="listGroups_description",
            log_failures=False,
            timeout=SIGNAL_GROUP_DESCRIPTION_RPC_TIMEOUT,
        )

        if not isinstance(groups, list):
            return False, None, None
        for group in groups:
            if not isinstance(group, dict) or str(group.get("id") or "") != group_id:
                continue
            raw_description = group.get("description")
            description = (
                str(raw_description).strip()
                if raw_description is not None
                else ""
            )
            return True, description or None, self._group_revision(group)
        return False, None, None

    # ------------------------------------------------------------------
    # Message Handling
    # ------------------------------------------------------------------

    async def _handle_envelope(self, envelope: dict) -> None:
        """Process an incoming signal-cli envelope."""
        # Unwrap nested envelope if present
        envelope_data = envelope.get("envelope", envelope)

        # Handle syncMessage: extract "Note to Self" messages (sent to own account)
        # while still filtering other sync events (read receipts, typing, etc.)
        is_note_to_self = False
        if "syncMessage" in envelope_data:
            sync_msg = envelope_data.get("syncMessage")
            if sync_msg and isinstance(sync_msg, dict):
                sent_msg = sync_msg.get("sentMessage")
                if sent_msg and isinstance(sent_msg, dict):
                    dest = sent_msg.get("destinationNumber") or sent_msg.get("destination")
                    sent_ts = sent_msg.get("timestamp")
                    sent_msg_group_info = sent_msg.get("groupInfo") or {}
                    sent_msg_group_id = sent_msg_group_info.get("groupId") if sent_msg_group_info else None
                    if dest == self._account_normalized or sent_msg_group_id:
                        # Check if this is an echo of our own outbound reply
                        if self._consume_sent_timestamp(sent_ts):
                            return
                        # Genuine user Note to Self — promote to dataMessage
                        is_note_to_self = True
                        envelope_data = {**envelope_data, "dataMessage": sent_msg}
            if not is_note_to_self:
                return

        # Extract sender info
        sender = (
            envelope_data.get("sourceNumber")
            or envelope_data.get("sourceUuid")
            or envelope_data.get("source")
        )
        sender_name = envelope_data.get("sourceName", "")
        sender_uuid = envelope_data.get("sourceUuid", "")
        self._remember_recipient_identifiers(sender, sender_uuid)

        if not sender:
            logger.debug("Signal: ignoring envelope with no sender")
            return

        # Self-message filtering — prevent reply loops (but allow Note to Self)
        if self._account_normalized and sender == self._account_normalized and not is_note_to_self:
            return

        # Filter stories
        if self.ignore_stories and envelope_data.get("storyMessage"):
            return

        # Get data message — also check editMessage (edited messages contain
        # their updated dataMessage inside editMessage.dataMessage)
        data_message = (
            envelope_data.get("dataMessage")
            or (envelope_data.get("editMessage") or {}).get("dataMessage")
        )
        if not data_message:
            return

        # Check for group message
        group_info = data_message.get("groupInfo")
        group_id = group_info.get("groupId") if group_info else None
        is_group = bool(group_id)

        # Group message filtering — derived from SIGNAL_GROUP_ALLOWED_USERS:
        # - No env var set → groups disabled (default safe behavior)
        # - Env var set with group IDs → only those groups allowed
        # - Env var set with "*" → all groups allowed
        # DM auth is fully handled by run.py (_is_user_authorized)
        if is_group:
            if not self.group_allow_from:
                logger.debug("Signal: ignoring group message (no SIGNAL_GROUP_ALLOWED_USERS)")
                return
            if "*" not in self.group_allow_from and group_id not in self.group_allow_from:
                logger.debug("Signal: group %s not in allowlist", group_id[:8] if group_id else "?")
                return
            self._maybe_schedule_group_description_refresh(group_id, group_info)

        # Build chat info
        chat_id = sender if not is_group else f"group:{group_id}"
        chat_type = "group" if is_group else "dm"

        # Extract text and render mentions.  When require_mention is enabled,
        # keep allowed-but-unmentioned group messages as observe-only context
        # instead of dropping them here; GatewayRunner records observe-only
        # events without invoking the LLM or sending a response.
        text = data_message.get("message", "")
        mentions = data_message.get("mentions", [])
        bot_identifiers = {self._account_normalized} if self._account_normalized else set()
        cached_bot_uuid = self._recipient_uuid_by_number.get(self._account_normalized)
        if cached_bot_uuid:
            bot_identifiers.add(cached_bot_uuid)
        if _is_signal_service_id(self._account_normalized):
            bot_identifiers.add(self._account_normalized)
        metadata_mentions_bot = _signal_mentions_include_bot(mentions, bot_identifiers)
        if text and mentions:
            text = _render_mentions(text, mentions)
        rendered_mentions_bot = bool(
            text
            and any(f"@{identifier}" in text for identifier in bot_identifiers)
        )
        was_mentioned = metadata_mentions_bot or rendered_mentions_bot
        observe_only = False
        if is_group and self.require_mention and not was_mentioned:
            observe_only = True
        if text and was_mentioned:
            for identifier in sorted(bot_identifiers, key=len, reverse=True):
                text = text.replace(f"@{identifier}", "").strip()

        # Strip the bot's own @mention from any group message so the agent
        # doesn't misinterpret "@+155****4567 say hello" as a directive to
        # contact that phone number. _render_mentions replaces the Signal
        # ￼ placeholder with @<number-or-uuid>, which looks like an
        # addressee to the LLM rather than a self-reference. Applies to every
        # group (not just require_mention groups) so the self-mention is
        # cleaned wherever it appears.
        if is_group and text:
            account_norm = self._account_normalized
            if account_norm:
                text = text.replace(f"@{account_norm}", "")
                # Also strip if the mention was rendered using the bot's UUID
                bot_uuid = self._recipient_uuid_by_number.get(account_norm)
                if bot_uuid:
                    text = text.replace(f"@{bot_uuid}", "")
                # Tidy the spacing the removed mention left behind: collapse the
                # double-space at a mid-sentence removal and trim the ends.
                # Only touches the doubled space the removal introduced, so
                # intentional newlines in a multi-line message are preserved.
                text = text.replace("  ", " ").strip()

        # Extract quote (reply-to) context from Signal dataMessage. Signal's
        # quote.id is the timestamp of the quoted message; quote.author points
        # at the quoted sender when available. Preserve both so the gateway can
        # tell the agent when the user replied to a specific assistant message.
        quote_data = data_message.get("quote") or {}
        reply_to_id = str(quote_data.get("id")) if quote_data.get("id") else None
        reply_to_text = quote_data.get("text")
        reply_to_author = self._extract_quote_author(quote_data)
        reply_to_author_name = quote_data.get("authorName") or quote_data.get("authorProfileName")
        reply_to_is_own = self._quote_references_own_message(reply_to_id, reply_to_author)

        # Process attachments
        attachments_data = data_message.get("attachments", [])
        media_urls = []
        media_types = []

        if attachments_data and not getattr(self, "ignore_attachments", False):
            for att in attachments_data:
                att_id = att.get("id")
                att_size = att.get("size", 0)
                if not att_id:
                    continue
                if att_size > SIGNAL_MAX_ATTACHMENT_SIZE:
                    logger.warning("Signal: attachment too large (%d bytes), skipping", att_size)
                    continue
                try:
                    cached_path, ext = await self._fetch_attachment(att_id)
                    if cached_path:
                        # Use contentType from Signal if available, else map from extension
                        content_type = att.get("contentType") or _ext_to_mime(ext)
                        media_urls.append(cached_path)
                        media_types.append(content_type)
                except Exception:
                    logger.exception("Signal: failed to fetch attachment %s", att_id)

        # Skip envelopes with no meaningful content (no text, no attachments).
        # Catches profile key updates, empty messages, and other metadata-only
        # envelopes that still carry a dataMessage wrapper but have nothing
        # worth processing. See issue: signal-cli logs "Profile key update" +
        # Hermes receives msg='' triggering a full agent turn for nothing.
        if (not text or not text.strip()) and not media_urls:
            logger.debug(
                "Signal: skipping contentless envelope from %s (%d attachments)",
                redact_phone(sender), len(media_urls) if media_urls else 0,
            )
            return

        if is_group:
            chat_topic, chat_topic_known = self._group_description_context(
                group_id,
                self._group_revision(group_info),
            )
        else:
            chat_topic, chat_topic_known = None, False

        # Build session source
        source = self.build_source(
            chat_id=chat_id,
            chat_name=group_info.get("groupName") if group_info else sender_name,
            chat_type=chat_type,
            user_id=sender,
            user_name=sender_name or sender,
            user_id_alt=sender_uuid if sender_uuid else None,
            chat_id_alt=group_id if is_group else None,
            chat_topic=chat_topic,
            chat_topic_known=chat_topic_known,
        )

        # Determine message type from media
        msg_type = MessageType.TEXT
        if media_types:
            if any(mt.startswith("audio/") for mt in media_types):
                msg_type = MessageType.VOICE
            elif any(mt.startswith("image/") for mt in media_types):
                msg_type = MessageType.PHOTO
            elif any(mt.startswith("video/") for mt in media_types):
                msg_type = MessageType.VIDEO
            else:
                # Catch-all: application/*, text/*, and unknown MIME types are
                # treated as documents so run.py's document-context injection
                # surfaces the cached file path to the agent (same pattern as
                # WhatsApp/Slack/BlueBubbles/Mattermost).
                msg_type = MessageType.DOCUMENT

        # Parse timestamp from envelope data (milliseconds since epoch)
        ts_ms = envelope_data.get("timestamp", 0)
        if ts_ms:
            try:
                timestamp = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            except (ValueError, OSError):
                timestamp = datetime.now(tz=timezone.utc)
        else:
            timestamp = datetime.now(tz=timezone.utc)

        # Build and dispatch event.
        # Store raw envelope data in raw_message so on_processing_start/complete
        # can extract targetAuthor + targetTimestamp for sendReaction.
        event = MessageEvent(
            source=source,
            text=text or "",
            message_type=msg_type,
            media_urls=media_urls,
            media_types=media_types,
            timestamp=timestamp,
            raw_message={
                "sender": sender,
                "timestamp_ms": ts_ms,
                "quote": quote_data if quote_data else None,
            },
            reply_to_message_id=reply_to_id,
            reply_to_text=reply_to_text,
            reply_to_author_id=reply_to_author,
            reply_to_author_name=reply_to_author_name,
            reply_to_is_own_message=reply_to_is_own,
            observe_only=observe_only,
        )

        logger.debug("Signal: message from %s in %s: %s",
                      redact_phone(sender), chat_id[:20], (text or "")[:50])

        await self.handle_message(event)

    def _remember_recipient_identifiers(self, number: Optional[str], service_id: Optional[str]) -> None:
        """Cache any number↔UUID mapping observed from Signal envelopes."""
        if not number or not service_id or not _is_signal_service_id(service_id):
            return
        self._recipient_uuid_by_number[number] = service_id
        self._recipient_number_by_uuid[service_id] = number

    @staticmethod
    def _extract_quote_author(quote_data: Any) -> Optional[str]:
        """Return the best available Signal sender identifier from quote metadata."""
        if not isinstance(quote_data, dict):
            return None
        for key in (
            "author",
            "authorNumber",
            "authorUuid",
            "authorAci",
            "authorServiceId",
            "authorServiceIdString",
        ):
            value = quote_data.get(key)
            if value:
                return str(value)
        return None

    def _quote_references_own_message(
        self,
        reply_to_id: Optional[str],
        reply_to_author: Optional[str],
    ) -> bool:
        """True when a Signal quote points at this adapter's outbound message."""
        if reply_to_id and str(reply_to_id) in self._sent_message_timestamps:
            return True
        if not reply_to_author:
            return False
        author = str(reply_to_author).strip()
        if self._account_normalized and author == self._account_normalized:
            return True
        cached_uuid = self._recipient_uuid_by_number.get(self._account_normalized)
        if cached_uuid and author == cached_uuid:
            return True
        cached_number = self._recipient_number_by_uuid.get(author)
        return bool(cached_number and cached_number == self._account_normalized)

    def _remember_sent_message_timestamp(self, timestamp: Any) -> None:
        """Keep a bounded cache of outbound Signal timestamps for quote matching."""
        if timestamp is None:
            return
        key = str(timestamp)
        # Re-insert to mark most-recently-used so eviction drops genuinely old
        # timestamps, not a recently re-seen one.
        self._sent_message_timestamps.pop(key, None)
        self._sent_message_timestamps[key] = None
        # FIFO-evict the oldest entry once over the cap.
        while len(self._sent_message_timestamps) > self._max_sent_message_timestamps:
            self._sent_message_timestamps.popitem(last=False)

    def _extract_contact_uuid(self, contact: Any, phone_number: str) -> Optional[str]:
        """Best-effort extraction of a Signal service ID from listContacts output."""
        if not isinstance(contact, dict):
            return None

        number = contact.get("number")
        recipient = contact.get("recipient")
        service_id = contact.get("uuid") or contact.get("serviceId")
        if not service_id:
            profile = contact.get("profile")
            if isinstance(profile, dict):
                service_id = profile.get("serviceId") or profile.get("uuid")

        if service_id and _is_signal_service_id(service_id):
            matches_number = number == phone_number or recipient == phone_number
            if matches_number:
                return service_id
        return None

    async def _resolve_recipient(self, chat_id: str) -> str:
        """Return the preferred Signal recipient identifier for a direct chat."""
        if (
            not chat_id
            or chat_id.startswith("group:")
            or _is_signal_service_id(chat_id)
            or not _looks_like_e164_number(chat_id)
        ):
            return chat_id

        cached = self._recipient_uuid_by_number.get(chat_id)
        if cached:
            return cached

        async with self._recipient_cache_lock:
            cached = self._recipient_uuid_by_number.get(chat_id)
            if cached:
                return cached

            contacts = await self._rpc("listContacts", {
                "account": self.account,
                "allRecipients": True,
            })
            if isinstance(contacts, list):
                for contact in contacts:
                    number = contact.get("number") if isinstance(contact, dict) else None
                    service_id = self._extract_contact_uuid(contact, chat_id)
                    if number and service_id:
                        self._remember_recipient_identifiers(number, service_id)

            return self._recipient_uuid_by_number.get(chat_id, chat_id)

    # ------------------------------------------------------------------
    # Attachment Handling
    # ------------------------------------------------------------------

    async def _fetch_attachment(self, attachment_id: str) -> tuple:
        """Fetch an attachment via JSON-RPC and cache it. Returns (path, ext)."""
        result = await self._rpc("getAttachment", {
            "account": self.account,
            "id": attachment_id,
        })

        if not result:
            return None, ""

        # Handle dict response (signal-cli returns {"data": "base64..."})
        if isinstance(result, dict):
            result = result.get("data")
            if not result:
                logger.warning("Signal: attachment response missing 'data' key")
                return None, ""

        # Result is base64-encoded file content
        raw_data = base64.b64decode(result)
        ext = _guess_extension(raw_data)

        # Android Signal voice notes are raw ADTS AAC streams. Most STT
        # providers (Groq Whisper, OpenAI Whisper) reject raw ADTS — they
        # require AAC to be muxed into an MP4 container. Remux losslessly
        # with ``ffmpeg -c:a copy`` so the cached file is a normal .m4a.
        # No re-encode, sub-100ms on a Pi 5. Graceful no-op if ffmpeg is
        # absent: the raw ADTS file is cached as-is and STT may reject it
        # (there is no downstream sniff-and-remux fallback).
        if ext == ".aac":
            remuxed: Optional[Tuple[bytes, str]] = await asyncio.to_thread(_remux_aac_to_m4a, raw_data)
            if remuxed is not None:
                raw_data, ext = remuxed

        if _is_image_ext(ext):
            path = cache_image_from_bytes(raw_data, ext)
        elif _is_audio_ext(ext):
            path = cache_audio_from_bytes(raw_data, ext)
        else:
            path = cache_document_from_bytes(raw_data, ext)

        return path, ext

    # ------------------------------------------------------------------
    # JSON-RPC Communication
    # ------------------------------------------------------------------

    async def _rpc(
        self,
        method: str,
        params: dict,
        rpc_id: str = None,
        *,
        log_failures: bool = True,
        raise_on_rate_limit: bool = False,
        timeout: float = 30.0,
    ) -> Any:
        """Send a JSON-RPC 2.0 request to signal-cli daemon.

        When ``log_failures=False``, error and exception paths log at DEBUG
        instead of WARNING — used by the typing-indicator path to silence
        repeated NETWORK_FAILURE spam for unreachable recipients while
        still preserving visibility for the first occurrence and for
        unrelated RPCs.

        When ``raise_on_rate_limit=True``, a Signal ``[429]`` /
        ``RateLimitException`` response raises ``SignalRateLimitError``
        instead of returning recipient results — lets callers
        (multi-attachment send) opt into scheduler-managed backoff. Normal text
        sends preserve classified failures for the shared send/retry layer.
        """
        if not self.client:
            logger.warning("Signal: RPC called but client not connected")
            return None

        if rpc_id is None:
            rpc_id = f"{method}_{int(time.time() * 1000)}"

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": rpc_id,
        }

        try:
            resp = await self.client.post(
                f"{self.http_url}/api/v1/rpc",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                err = data["error"]
                structured_response: Optional[Dict[str, Any]] = None
                structured_results = None
                if method == "send" and isinstance(err, dict):
                    err_data = err.get("data")
                    if isinstance(err_data, dict):
                        candidate = err_data.get("response")
                        if isinstance(candidate, dict):
                            candidate_results = candidate.get("results")
                            if isinstance(candidate_results, list):
                                structured_response = candidate
                                structured_results = candidate_results

                # signal-cli can wrap a per-recipient send response in a JSON-RPC
                # error envelope. Preserve those results: text sends need them to
                # classify rate limits, and mixed-success group sends must not be
                # retried wholesale after some recipients already received them.
                if structured_results is not None:
                    assert structured_response is not None
                    annotated_response: Dict[str, Any] = {
                        str(key): value for key, value in structured_response.items()
                    }
                    annotated_response["_hermes_rpc_error_envelope"] = True
                    has_success = any(
                        isinstance(r, dict)
                        and (
                            r.get("type") == "SUCCESS"
                            or r.get("success") is True
                        )
                        for r in structured_results
                    )
                    is_rate_limit = _is_signal_rate_limit_error(err) or any(
                        isinstance(r, dict)
                        and r.get("type") == "RATE_LIMIT_FAILURE"
                        for r in structured_results
                    )
                    if is_rate_limit:
                        annotated_response["_hermes_rpc_error_kind"] = "rate_limited"
                        annotated_response["_hermes_rpc_error_message"] = "Signal rate limited"
                        retry_after = _extract_retry_after_seconds(err)
                        if retry_after is not None:
                            annotated_response["_hermes_retry_after_seconds"] = retry_after
                    else:
                        annotated_response["_hermes_rpc_error_kind"] = "unknown"
                        annotated_response["_hermes_rpc_error_message"] = "Signal RPC send failed"
                    if has_success:
                        return annotated_response
                    if raise_on_rate_limit and is_rate_limit:
                        err_msg = str(err.get("message", ""))
                        retry_after = annotated_response.get("_hermes_retry_after_seconds")
                        raise SignalRateLimitError(err_msg, retry_after=retry_after)
                    return annotated_response

                is_rate_limit = _is_signal_rate_limit_error(err)
                err_msg = (
                    str(err.get("message", ""))
                    if isinstance(err, dict)
                    else str(err)
                )
                retry_after = (
                    _extract_retry_after_seconds(err) if is_rate_limit else None
                )
                if raise_on_rate_limit and is_rate_limit:
                    raise SignalRateLimitError(err_msg, retry_after=retry_after)

                # Text sends must preserve even unstructured JSON-RPC failures
                # so the shared retry layer can distinguish rate limits from
                # permanent delivery errors. Keep the synthetic response free
                # of daemon-provided text because it may contain recipient PII.
                if method == "send":
                    error_kind = (
                        "rate_limited"
                        if is_rate_limit
                        else classify_send_error(None, err_msg)
                    )
                    failure_result: Dict[str, Any] = {
                        "_hermes_rpc_error_envelope": True,
                        "_hermes_rpc_error_kind": error_kind,
                        "_hermes_rpc_error_message": (
                            "Signal rate limited"
                            if is_rate_limit
                            else "Signal RPC send failed"
                        ),
                    }
                    if retry_after is not None:
                        failure_result["_hermes_retry_after_seconds"] = retry_after
                    log = logger.warning if log_failures else logger.debug
                    log("Signal RPC send error (%s)", error_kind)
                    return failure_result

                if log_failures:
                    logger.warning("Signal RPC error (%s): %s", method, err)
                else:
                    logger.debug("Signal RPC error (%s): %s", method, err)
                return None

            result = data.get("result")
            if isinstance(result, dict) and raise_on_rate_limit:
                results = result.get("results")
                if isinstance(results, list):
                    has_success = any(
                        isinstance(r, dict)
                        and (
                            r.get("type") == "SUCCESS"
                            or r.get("success") is True
                        )
                        for r in results
                    )
                    for r in results if not has_success else ():
                        if isinstance(r, dict) and r.get("type") == "RATE_LIMIT_FAILURE":
                            retry_after = r.get("retryAfterSeconds")
                            raise SignalRateLimitError("Rate limit exceeded for recipient", retry_after=retry_after)

            return result

        except SignalRateLimitError:
            raise
        except Exception as e:
            status_code = None
            response = getattr(e, "response", None)
            if response is not None:
                status_code = getattr(response, "status_code", None)
            error_kind = classify_send_error(status_code, str(e))
            # A connect failure happens before signal-cli can accept the send,
            # so retrying is safe. Read/write timeouts remain ``unknown``
            # because the daemon may already have accepted the message.
            if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
                error_kind = "transient"
            elif isinstance(e, (httpx.ReadTimeout, httpx.WriteTimeout)):
                error_kind = "unknown"

            retry_after = None
            if response is not None:
                headers = getattr(response, "headers", {}) or {}
                try:
                    retry_after = float(headers.get("Retry-After"))
                except (TypeError, ValueError):
                    retry_after = None
            if raise_on_rate_limit and error_kind == "rate_limited":
                raise SignalRateLimitError(
                    "Signal RPC rate limited",
                    retry_after=retry_after,
                )
            if log_failures:
                logger.warning("Signal RPC %s failed (%s)", method, type(e).__name__)
            else:
                logger.debug("Signal RPC %s failed (%s)", method, type(e).__name__)
            if method == "send":
                failure_result: Dict[str, Any] = {
                    "_hermes_rpc_error_envelope": True,
                    "_hermes_rpc_error_kind": error_kind,
                    "_hermes_rpc_error_message": (
                        "Signal RPC connection failed"
                        if error_kind == "transient"
                        else "Signal RPC send failed"
                    ),
                }
                if retry_after is not None:
                    failure_result["_hermes_retry_after_seconds"] = retry_after
                return failure_result
            return None

    # ------------------------------------------------------------------
    # Formatting — markdown → Signal body ranges
    # ------------------------------------------------------------------

    @staticmethod
    def _markdown_to_signal(text: str) -> tuple[str, list[str]]:
        """Backward-compatible wrapper around shared Signal formatting helper."""
        return markdown_to_signal(text)

    def format_message(self, content: str) -> str:
        """Strip markdown for plain-text fallback (used by base class).

        The actual rich formatting happens in send() via _markdown_to_signal().
        """
        # This is only called if someone uses the base-class send path.
        # Our send() override bypasses this entirely.
        return content

    def _validate_send_result(self, result: Any) -> tuple[bool, Optional[str]]:
        """Validate signal-cli send response results.

        A group send is considered delivered when at least one recipient
        succeeded. Retrying the whole group after partial delivery would create
        duplicates for successful recipients, so partial failures are logged
        without recipient identifiers and returned as success.

        Returns ``(success, error_message)``.
        """
        if not result or not isinstance(result, dict):
            return True, None

        results = result.get("results")
        if isinstance(results, list):
            success_count = 0
            failures: list[str] = []
            for r in results:
                if not isinstance(r, dict):
                    continue
                rtype = r.get("type")
                if rtype == "SUCCESS" or r.get("success") is True:
                    success_count += 1
                    continue
                if rtype:
                    failures.append(str(rtype))
                    continue
                if "success" in r and not r.get("success"):
                    failures.append("LEGACY_FAILURE")

            if success_count:
                if failures:
                    failure_counts: Dict[str, int] = {}
                    for failure in failures:
                        failure_counts[failure] = failure_counts.get(failure, 0) + 1
                    summary = ", ".join(
                        f"{failure}={count}"
                        for failure, count in sorted(failure_counts.items())
                    )
                    logger.warning(
                        "Signal: partial recipient delivery (%d succeeded, %d failed; %s); "
                        "not retrying the full group to avoid duplicates",
                        success_count,
                        len(failures),
                        summary,
                    )
                return True, None
            if failures:
                if failures[0] == "LEGACY_FAILURE":
                    return False, "Recipient delivery failed"
                return False, failures[0]
        if result.get("_hermes_rpc_error_envelope"):
            return False, str(
                result.get("_hermes_rpc_error_message")
                or "Signal RPC send failed"
            )
        return True, None

    @staticmethod
    def _failed_send_result(result: Any, error_message: Optional[str]) -> SendResult:
        """Translate signal-cli recipient failures into the shared send contract."""
        failure_types: list[str] = []
        legacy_failure_kinds: list[str] = []
        retry_after_values: list[float] = []

        if isinstance(result, dict):
            internal_retry_after = result.get("_hermes_retry_after_seconds")
            try:
                if internal_retry_after is not None and float(internal_retry_after) >= 0:
                    retry_after_values.append(float(internal_retry_after))
            except (TypeError, ValueError):
                pass
            recipient_results = result.get("results")
            if isinstance(recipient_results, list):
                for recipient_result in recipient_results:
                    if not isinstance(recipient_result, dict):
                        continue
                    failure_type = recipient_result.get("type")
                    if failure_type and failure_type != "SUCCESS":
                        failure_types.append(str(failure_type))
                    elif recipient_result.get("success") is False:
                        legacy_failure = str(recipient_result.get("failure") or "")
                        legacy_failure_kinds.append(
                            classify_send_error(None, legacy_failure)
                        )
                    retry_after = recipient_result.get("retryAfterSeconds")
                    try:
                        if retry_after is not None and float(retry_after) >= 0:
                            retry_after_values.append(float(retry_after))
                    except (TypeError, ValueError):
                        pass

        failure_type_set = set(failure_types)
        rpc_error_kind = (
            result.get("_hermes_rpc_error_kind")
            if isinstance(result, dict)
            else None
        )
        if (
            "RATE_LIMIT_FAILURE" in failure_type_set
            or rpc_error_kind == "rate_limited"
            or "rate_limited" in legacy_failure_kinds
        ):
            error_kind = "rate_limited"
            retryable = True
        elif (
            "NETWORK_FAILURE" in failure_type_set
            or rpc_error_kind == "transient"
            or "transient" in legacy_failure_kinds
        ):
            error_kind = "transient"
            retryable = True
        elif (
            (failure_type_set and failure_type_set <= {"UNREGISTERED_FAILURE"})
            or rpc_error_kind == "not_found"
            or "not_found" in legacy_failure_kinds
        ):
            error_kind = "not_found"
            retryable = False
        elif rpc_error_kind:
            error_kind = str(rpc_error_kind)
            retryable = error_kind in {"rate_limited", "transient"}
        elif legacy_failure_kinds:
            error_kind = next(
                (kind for kind in legacy_failure_kinds if kind != "unknown"),
                "unknown",
            )
            retryable = error_kind in {"rate_limited", "transient"}
        else:
            # Convert signal-cli's enum spelling into words so the shared
            # classifier can recognize e.g. RATE_LIMIT_FAILURE.
            classification_text = (error_message or "").replace("_", " ")
            error_kind = classify_send_error(None, classification_text)
            retryable = error_kind in {"rate_limited", "transient"}

        return SendResult(
            success=False,
            error=error_message,
            raw_response=result,
            retryable=retryable,
            retry_after=max(retry_after_values) if retry_after_values else None,
            error_kind=error_kind,
        )

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message with native Signal formatting."""
        await self._stop_typing_indicator(chat_id)

        plain_text, text_styles = self._markdown_to_signal(content)

        params: Dict[str, Any] = {
            "account": self.account,
            "message": plain_text,
        }

        if text_styles:
            if len(text_styles) == 1:
                params["textStyle"] = text_styles[0]
            else:
                params["textStyles"] = text_styles

        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [await self._resolve_recipient(chat_id)]

        logger.info("[Signal] Sending response (%d chars) to %s", len(plain_text), chat_id)
        result = await self._rpc("send", params)

        if result is not None:
            success, err_msg = self._validate_send_result(result)
            if not success:
                return self._failed_send_result(result, err_msg)
            self._track_sent_timestamp(result)
            # Signal has no editable message identifier. Returning None keeps the
            # stream consumer on the non-edit fallback path instead of pretending
            # future edits can remove an in-progress cursor from the chat thread.
            return SendResult(success=True, message_id=None, raw_response=result)
        return SendResult(success=False, error="RPC send failed")

    def _track_sent_timestamp(self, rpc_result) -> None:
        """Record outbound message timestamp for echo-back filtering."""
        ts = rpc_result.get("timestamp") if isinstance(rpc_result, dict) else None
        if ts:
            self._remember_sent_message_timestamp(ts)
            now = time.monotonic()
            # Re-insert to mark as most-recently-used.
            self._recent_sent_timestamps.pop(ts, None)
            self._recent_sent_timestamps[ts] = now
            # Drop entries older than TTL first (cheap O(k) where k=expired).
            cutoff = now - self._recent_sent_ttl_seconds
            while self._recent_sent_timestamps:
                oldest_ts, oldest_at = next(iter(self._recent_sent_timestamps.items()))
                if oldest_at < cutoff:
                    self._recent_sent_timestamps.popitem(last=False)
                else:
                    break
            # Hard cap as a last-resort guard against runaway producers.
            while len(self._recent_sent_timestamps) > self._max_recent_timestamps:
                self._recent_sent_timestamps.popitem(last=False)

    def _consume_sent_timestamp(self, ts) -> bool:
        """Pop a timestamp if it matches one we sent. Returns True on echo."""
        if ts and ts in self._recent_sent_timestamps:
            self._recent_sent_timestamps.pop(ts, None)
            return True
        return False

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send a typing indicator.

        base.py's ``_keep_typing`` refresh loop calls this every ~2s while
        the agent is processing. If signal-cli returns NETWORK_FAILURE for
        this recipient (offline, unroutable, group membership lost, etc.)
        the unmitigated behaviour is: a WARNING log every 2 seconds for as
        long as the agent keeps running. Instead we:

        - silence the WARNING after the first consecutive failure (subsequent
          attempts log at DEBUG) so transport issues are still visible once
          but don't flood the log,
        - skip the RPC entirely during an exponential cooldown window once
          three consecutive failures have happened, so we stop hammering
          signal-cli with requests it can't deliver.

        A successful sendTyping clears the counters.
        """
        now = time.monotonic()
        skip_until = self._typing_skip_until.get(chat_id, 0.0)
        if now < skip_until:
            return

        params: Dict[str, Any] = {
            "account": self.account,
        }

        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [await self._resolve_recipient(chat_id)]

        fails = self._typing_failures.get(chat_id, 0)
        result = await self._rpc(
            "sendTyping",
            params,
            rpc_id="typing",
            log_failures=(fails == 0),
        )

        if result is None:
            fails += 1
            self._typing_failures[chat_id] = fails
            # After 3 consecutive failures, back off exponentially (16s,
            # 32s, 60s cap) to stop spamming signal-cli for a recipient
            # that clearly isn't reachable right now.
            if fails >= 3:
                backoff = min(60.0, 16.0 * (2 ** (fails - 3)))
                self._typing_skip_until[chat_id] = now + backoff
        else:
            self._typing_failures.pop(chat_id, None)
            self._typing_skip_until.pop(chat_id, None)

    async def send_multiple_images(
        self,
        chat_id: str,
        images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        human_delay: float = 0.0,
    ) -> None:
        """Send a batch of images via chunked Signal RPC calls.

        Per-image alt texts are dropped — Signal's send RPC only carries
        one shared message body. Bad images (download failure, missing
        file, oversize) are skipped with a warning so one bad URL
        doesn't lose the rest of the batch. ``human_delay`` is ignored:
        the rate-limit scheduler handles inter-batch pacing.
        """
        if not images:
            return

        scheduler = get_scheduler()
        logger.info(
            "Signal send_multiple_images: received %d image(s) for %s — "
            "scheduler state: %s",
            len(images), chat_id[:30], scheduler.state(),
        )

        await self._stop_typing_indicator(chat_id)

        attachments: List[str] = []
        skipped_download = 0
        skipped_missing = 0
        skipped_oversize = 0
        for image_url, _alt_text in images:
            if image_url.startswith("file://"):
                file_path = unquote(image_url[7:])
            else:
                try:
                    file_path = await cache_image_from_url(image_url)
                except Exception as e:
                    logger.warning("Signal: failed to download image %s: %s", image_url, e)
                    skipped_download += 1
                    continue

            if not file_path or not Path(file_path).exists():
                logger.warning("Signal: image file not found for %s", image_url)
                skipped_missing += 1
                continue

            file_size = Path(file_path).stat().st_size
            if file_size > SIGNAL_MAX_ATTACHMENT_SIZE:
                logger.warning(
                    "Signal: image too large (%d bytes), skipping %s", file_size, image_url
                )
                skipped_oversize += 1
                continue

            # signal-cli's HTTP JSON-RPC daemon may be running on a different
            # machine than Hermes, so local filesystem paths are not valid
            # attachment references at the RPC boundary. Keep validation on the
            # local path above, but send the actual bytes in-band as a data URI.
            attachments.append(_normalize_signal_attachment(file_path))

        if not attachments:
            logger.error(
                "Signal: no valid images in batch of %d "
                "(download=%d missing=%d oversize=%d)",
                len(images), skipped_download, skipped_missing, skipped_oversize,
            )
            return

        logger.info(
            "Signal send_multiple_images: %d/%d images valid, sending in chunks",
            len(attachments), len(images),
        )

        base_params: Dict[str, Any] = {
            "account": self.account,
            "message": "",
        }
        if chat_id.startswith("group:"):
            base_params["groupId"] = chat_id[6:]
        else:
            base_params["recipient"] = [await self._resolve_recipient(chat_id)]

        att_batches = [
            attachments[i:i + SIGNAL_MAX_ATTACHMENTS_PER_MSG]
            for i in range(0, len(attachments), SIGNAL_MAX_ATTACHMENTS_PER_MSG)
        ]

        for idx, att_batch in enumerate(att_batches):
            n = len(att_batch)
            estimated = scheduler.estimate_wait(n)
            logger.debug(
                "Signal batch %d/%d: %d attachments, estimated wait=%.1fs",
                idx + 1, len(att_batches), n, estimated,
            )
            if estimated >= SIGNAL_BATCH_PACING_NOTICE_THRESHOLD:
                await self._notify_batch_pacing(
                    chat_id, idx + 1, len(att_batches), estimated
                )

            params = dict(base_params, attachments=att_batch)
            send_timeout = _signal_send_timeout(n)

            for attempt in range(1, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS + 1):
                await scheduler.acquire(n)
                try:
                    _rpc_t0 = time.monotonic()
                    result = await self._rpc(
                        "send", params, raise_on_rate_limit=True, timeout=send_timeout,
                    )
                    _rpc_duration = time.monotonic() - _rpc_t0
                    if result is not None:
                        success, err_msg = self._validate_send_result(result)
                        if success:
                            self._track_sent_timestamp(result)
                            await scheduler.report_rpc_duration(_rpc_duration, n)
                            recipient_results = (
                                result.get("results", [])
                                if isinstance(result, dict)
                                else []
                            )
                            if any(
                                isinstance(item, dict)
                                and item.get("type") == "RATE_LIMIT_FAILURE"
                                for item in recipient_results
                            ):
                                # Do not resend a partially delivered group
                                # batch, but retain the server's rate feedback
                                # so later attachment batches are paced.
                                scheduler.feedback(
                                    _extract_retry_after_seconds(result), n
                                )
                            logger.info(
                                "Signal batch %d/%d: %d attachments sent in %.1fs "
                                "(attempt %d/%d)",
                                idx + 1, len(att_batches), n, _rpc_duration,
                                attempt, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS,
                            )
                        else:
                            failed_result = self._failed_send_result(result, err_msg)
                            logger.error(
                                "Signal: RPC send failed for batch %d/%d (%d attachments, "
                                "attempt %d/%d, rpc_duration=%.1fs): %s",
                                idx + 1, len(att_batches), n,
                                attempt, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS,
                                _rpc_duration, err_msg,
                            )
                            # _rpc normally raises rate limits when requested,
                            # but preserve scheduler behavior if a structured
                            # rate-limit result reaches this path.
                            if failed_result.error_kind == "rate_limited":
                                raise SignalRateLimitError(
                                    failed_result.error or err_msg or "Signal rate limited",
                                    retry_after=failed_result.retry_after,
                                )
                            # Retry only failures known not to have delivered.
                            # Read/write timeouts are ambiguous and must not
                            # resend the attachment batch.
                            if (
                                failed_result.retryable
                                and attempt < SIGNAL_RATE_LIMIT_MAX_ATTEMPTS
                            ):
                                backoff = 2.0 ** attempt
                                logger.info(
                                    "Signal: retrying batch %d/%d after %.1fs backoff",
                                    idx + 1, len(att_batches), backoff,
                                )
                                await asyncio.sleep(backoff)
                                continue
                    else:
                        # A missing/malformed response does not prove that the
                        # server rejected the non-idempotent send. Do not retry
                        # the whole batch because delivery is ambiguous.
                        logger.error(
                            "Signal: RPC send failed for batch %d/%d (%d attachments, "
                            "attempt %d/%d, rpc_duration=%.1fs)",
                            idx + 1, len(att_batches), n,
                            attempt, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS,
                            _rpc_duration,
                        )
                    break
                except SignalRateLimitError as e:
                    scheduler.feedback(e.retry_after, n)
                    if attempt >= SIGNAL_RATE_LIMIT_MAX_ATTEMPTS:
                        logger.error(
                            "Signal: rate-limit retries exhausted on batch %d/%d "
                            "(%d attachments lost, server retry_after=%s)",
                            idx + 1, len(att_batches), n,
                            f"{e.retry_after:.0f}s" if e.retry_after else "unknown",
                        )
                        break
                    logger.warning(
                        "Signal: rate-limited on batch %d/%d "
                        "(attempt %d/%d, server retry_after=%s); "
                        "scheduler will pace the retry",
                        idx + 1, len(att_batches),
                        attempt, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS,
                        f"{e.retry_after:.0f}s" if e.retry_after else "unknown",
                    )

    async def _notify_batch_pacing(
        self,
        chat_id: str,
        next_batch_idx: int,
        total_batches: int,
        wait_s: float,
    ) -> None:
        """Inform the user when an inter-batch pacing wait crosses the
        notice threshold. Best-effort; logs and continues on failure."""
        try:
            await self.send(
                chat_id,
                f"(More images coming — pausing ~{_format_wait(wait_s)} "
                f"for Signal rate limit, batch {next_batch_idx}/{total_batches}.)",
            )
        except Exception as e:
            logger.warning("Signal: failed to send pacing notice: %s", e)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send an image. Supports http(s):// and file:// URLs."""
        await self._stop_typing_indicator(chat_id)

        # Resolve image to local path
        if image_url.startswith("file://"):
            file_path = unquote(image_url[7:])
        else:
            # Download remote image to cache
            try:
                file_path = await cache_image_from_url(image_url)
            except Exception as e:
                logger.warning("Signal: failed to download image: %s", e)
                return SendResult(success=False, error=str(e))

        if not file_path or (not _is_data_uri_attachment(file_path) and not Path(file_path).exists()):
            return SendResult(success=False, error="Image file not found")

        # Validate size
        file_size = _signal_attachment_size(file_path)
        if file_size > SIGNAL_MAX_ATTACHMENT_SIZE:
            return SendResult(success=False, error=f"Image too large ({file_size} bytes)")

        params: Dict[str, Any] = {
            "account": self.account,
            "message": caption or "",
            "attachments": [_normalize_signal_attachment(file_path)],
        }

        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [await self._resolve_recipient(chat_id)]

        result = await self._rpc("send", params)
        if result is not None:
            success, err_msg = self._validate_send_result(result)
            if not success:
                return self._failed_send_result(result, err_msg)
            self._track_sent_timestamp(result)
            return SendResult(success=True, raw_response=result)
        return SendResult(success=False, error="RPC send with attachment failed")

    async def _send_attachment(
        self,
        chat_id: str,
        file_path: str,
        media_label: str,
        caption: Optional[str] = None,
    ) -> SendResult:
        """Send any file as a Signal attachment via RPC.

        Shared implementation for send_document, send_image_file, send_voice,
        and send_video — avoids duplicating the validation/routing/RPC logic.
        """
        await self._stop_typing_indicator(chat_id)

        try:
            file_size = _signal_attachment_size(file_path)
        except FileNotFoundError:
            return SendResult(success=False, error=f"{media_label} file not found: {file_path}")

        if file_size > SIGNAL_MAX_ATTACHMENT_SIZE:
            return SendResult(success=False, error=f"{media_label} too large ({file_size} bytes)")

        params: Dict[str, Any] = {
            "account": self.account,
            "message": caption or "",
            "attachments": [_normalize_signal_attachment(file_path)],
        }

        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [await self._resolve_recipient(chat_id)]

        result = await self._rpc("send", params)
        if result is not None:
            success, err_msg = self._validate_send_result(result)
            if not success:
                return self._failed_send_result(result, err_msg)
            self._track_sent_timestamp(result)
            return SendResult(success=True, raw_response=result)
        return SendResult(success=False, error=f"RPC send {media_label.lower()} failed")

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a document/file attachment."""
        return await self._send_attachment(chat_id, file_path, "File", caption)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local image file as a native Signal attachment.

        Called by the gateway media delivery flow when MEDIA: tags containing
        image paths are extracted from agent responses.
        """
        return await self._send_attachment(chat_id, image_path, "Image", caption)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send an audio file as a Signal attachment.

        Signal does not distinguish voice messages from file attachments at
        the API level, so this routes through the same RPC send path.
        """
        return await self._send_attachment(chat_id, audio_path, "Audio", caption)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a video file as a Signal attachment."""
        return await self._send_attachment(chat_id, video_path, "Video", caption)

    # ------------------------------------------------------------------
    # Typing Indicators
    # ------------------------------------------------------------------

    async def _stop_typing_indicator(self, chat_id: str) -> None:
        """Stop a typing indicator loop for a chat."""
        task = self._typing_tasks.pop(chat_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Send an explicit stop-typing RPC so the recipient's device drops the
        # indicator immediately instead of waiting for Signal's ~5s built-in
        # timeout.  Failures are best-effort — the backoff state must still be
        # cleared so the next agent turn starts clean.
        try:
            params: Dict[str, Any] = {"account": self.account}
            if chat_id.startswith("group:"):
                params["groupId"] = chat_id[6:]
            else:
                params["recipient"] = [await self._resolve_recipient(chat_id)]
            params["stop"] = True
            await self._rpc(
                "sendTyping",
                params,
                rpc_id="typing-stop",
                log_failures=False,
            )
        except Exception:
            # Best-effort: any RPC failure (or recipient-resolution failure)
            # must not prevent backoff cleanup.
            pass

        self._typing_failures.pop(chat_id, None)
        self._typing_skip_until.pop(chat_id, None)

    async def stop_typing(self, chat_id: str) -> None:
        """Public interface for stopping typing — called by base adapter's
        _keep_typing finally block to clean up platform-level typing tasks."""
        await self._stop_typing_indicator(chat_id)

    # ------------------------------------------------------------------
    # Reactions
    # ------------------------------------------------------------------

    async def send_reaction(
        self,
        chat_id: str,
        emoji: str,
        target_author: str,
        target_timestamp: int,
    ) -> bool:
        """Send a reaction emoji to a specific message via signal-cli RPC.

        Args:
            chat_id: The chat (phone number or "group:<id>")
            emoji: Reaction emoji string (e.g. "👀", "✅")
            target_author: Phone number / UUID of the message author
            target_timestamp: Signal timestamp (ms) of the message to react to
        """
        params: Dict[str, Any] = {
            "account": self.account,
            "emoji": emoji,
            "targetAuthor": target_author,
            "targetTimestamp": target_timestamp,
        }

        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [chat_id]

        result = await self._rpc("sendReaction", params)
        if result is not None:
            return True
        logger.debug("Signal: sendReaction failed (chat=%s, emoji=%s)", chat_id[:20], emoji)
        return False

    async def remove_reaction(
        self,
        chat_id: str,
        target_author: str,
        target_timestamp: int,
    ) -> bool:
        """Remove a reaction by sending an empty-string emoji."""
        params: Dict[str, Any] = {
            "account": self.account,
            "emoji": "",
            "targetAuthor": target_author,
            "targetTimestamp": target_timestamp,
            "remove": True,
        }

        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [chat_id]

        result = await self._rpc("sendReaction", params)
        return result is not None

    # ------------------------------------------------------------------
    # Processing Lifecycle Hooks (reactions as progress indicators)
    # ------------------------------------------------------------------

    def _extract_reaction_target(self, event: MessageEvent) -> Optional[tuple]:
        """Extract (target_author, target_timestamp) from a MessageEvent.

        Returns None if the event doesn't carry the raw Signal envelope data
        needed for sendReaction.
        """
        raw = event.raw_message
        if not isinstance(raw, dict):
            return None
        author = raw.get("sender")
        ts = raw.get("timestamp_ms")
        if not author or not ts:
            return None
        return (author, ts)

    def _reactions_enabled(self, event: "MessageEvent" = None) -> bool:
        """Check if message reactions are enabled for this event.

        Two gates:
        1. SIGNAL_REACTIONS env var — set to false/0/no to disable globally.
        2. DM allowlist — if SIGNAL_ALLOWED_USERS is set, only react to
           messages from senders in that list.  This prevents unauthorized
           contacts from seeing the 👀 reaction (which fires before run.py's
           auth gate and would otherwise reveal that a bot is listening).
        """
        if os.getenv("SIGNAL_REACTIONS", "true").lower() in {"false", "0", "no"}:
            return False
        if event is not None:
            sender = getattr(getattr(event, "source", None), "user_id", None)
            if sender and "*" not in self.dm_allow_from and sender not in self.dm_allow_from:
                return False
        return True

    async def on_processing_start(self, event: MessageEvent) -> None:
        """React with 👀 when processing begins."""
        if not self._reactions_enabled(event):
            return
        target = self._extract_reaction_target(event)
        if target:
            await self.send_reaction(event.source.chat_id, "👀", *target)

    async def on_processing_complete(self, event: MessageEvent, outcome: "ProcessingOutcome") -> None:
        """Swap the 👀 reaction for ✅ (success) or ❌ (failure).

        On CANCELLED we leave the 👀 in place — no terminal outcome means
        the reaction should keep reflecting "in progress" (matches Telegram).
        """
        if not self._reactions_enabled(event):
            return
        if outcome == ProcessingOutcome.CANCELLED:
            return
        target = self._extract_reaction_target(event)
        if not target:
            return
        chat_id = event.source.chat_id
        # Remove the in-progress reaction, then add the final one
        await self.remove_reaction(chat_id, *target)
        if outcome == ProcessingOutcome.SUCCESS:
            await self.send_reaction(chat_id, "✅", *target)
        elif outcome == ProcessingOutcome.FAILURE:
            await self.send_reaction(chat_id, "❌", *target)

    # ------------------------------------------------------------------
    # Chat Info
    # ------------------------------------------------------------------

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a chat/contact."""
        if chat_id.startswith("group:"):
            return {
                "name": chat_id,
                "type": "group",
                "chat_id": chat_id,
            }

        # Try to resolve contact name
        result = await self._rpc("getContact", {
            "account": self.account,
            "contactAddress": chat_id,
        })

        name = chat_id
        if result and isinstance(result, dict):
            name = result.get("name") or result.get("profileName") or chat_id

        return {
            "name": name,
            "type": "dm",
            "chat_id": chat_id,
        }

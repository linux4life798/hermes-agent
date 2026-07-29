"""Tests for Signal messenger platform adapter."""
import asyncio
import base64
import httpx
import pytest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch, AsyncMock
from urllib.parse import quote

from gateway.config import Platform, PlatformConfig


@pytest.fixture(autouse=True)
def _reset_signal_scheduler():
    """The attachment scheduler is process-wide; drop it between tests
    so a fresh token bucket greets each case."""
    from gateway.platforms.signal_rate_limit import _reset_scheduler
    _reset_scheduler()
    yield
    _reset_scheduler()


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------

def _make_signal_adapter(monkeypatch, account="+15551234567", **extra):
    """Create a SignalAdapter with sensible test defaults."""
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", extra.pop("group_allowed", ""))
    from gateway.platforms.signal import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {
        "http_url": "http://localhost:8080",
        "account": account,
        **extra,
    }
    return SignalAdapter(config)


def _stub_rpc(return_value):
    """Return an async mock for SignalAdapter._rpc that captures call params."""
    captured = []

    async def mock_rpc(method, params, rpc_id=None):
        captured.append({"method": method, "params": dict(params)})
        return return_value

    return mock_rpc, captured


# ---------------------------------------------------------------------------
# Platform & Config
# ---------------------------------------------------------------------------

class TestSignalConfigLoading:
    def test_apply_env_overrides_signal(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:9090")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+15551234567")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.SIGNAL in config.platforms
        sc = config.platforms[Platform.SIGNAL]
        assert sc.enabled is True
        assert sc.extra["http_url"] == "http://localhost:9090"
        assert sc.extra["account"] == "+15551234567"

    def test_signal_not_loaded_without_both_vars(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:9090")
        monkeypatch.delenv("SIGNAL_ACCOUNT", raising=False)
        # No SIGNAL_ACCOUNT

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.SIGNAL not in config.platforms

# ---------------------------------------------------------------------------
# Adapter Init & Helpers
# ---------------------------------------------------------------------------

class TestSignalAdapterInit:
    def test_init_parses_config(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, group_allowed="group123,group456")
        assert adapter.http_url == "http://localhost:8080"
        assert adapter.account == "+15551234567"
        assert "group123" in adapter.group_allow_from

    def test_init_empty_allowlist(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        assert len(adapter.group_allow_from) == 0

    def test_init_strips_trailing_slash(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, http_url="http://localhost:8080/")
        assert adapter.http_url == "http://localhost:8080"

    def test_self_message_filtering(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        assert adapter._account_normalized == "+15551234567"


class TestSignalGroupDescriptions:
    @staticmethod
    def _group_envelope(
        *,
        message: str = "hello group",
        revision: Optional[int] = 1,
        group_type: str = "DELIVER",
    ) -> dict:
        return {
            "envelope": {
                "sourceNumber": "+155****1111",
                "sourceName": "Alice",
                "timestamp": 1712345678000,
                "dataMessage": {
                    "message": message,
                    "groupInfo": {
                        "groupId": "group123",
                        "groupName": "House",
                        "revision": revision,
                        "type": group_type,
                    },
                },
            },
        }

    @pytest.mark.asyncio
    async def test_cache_miss_refreshes_only_the_active_group_in_background(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(
            monkeypatch,
            account="bot-account",
            group_allowed="group123",
        )
        refresh_started = asyncio.Event()
        release_refresh = asyncio.Event()
        rpc_calls = []

        async def fake_rpc(method, params, rpc_id=None, **kwargs):
            rpc_calls.append((method, params, kwargs))
            refresh_started.set()
            await release_refresh.wait()
            return [{"id": "group123", "description": "House coordination"}]

        adapter._rpc = fake_rpc
        handled = []

        async def fake_handle(event):
            handled.append(event)

        adapter.handle_message = fake_handle

        await adapter._handle_envelope(self._group_envelope())
        await refresh_started.wait()

        assert handled[0].source.chat_topic is None
        assert len(rpc_calls) == 1
        assert rpc_calls[0][0] == "listGroups"
        assert rpc_calls[0][1] == {
            "account": "bot-account",
            "groupId": ["group123"],
        }
        assert rpc_calls[0][2]["log_failures"] is False
        assert rpc_calls[0][2]["timeout"] > 30

        release_refresh.set()
        task = adapter._group_description_refresh_task
        assert task is not None
        await task

        await adapter._handle_envelope(self._group_envelope(message="second message"))

        assert handled[-1].source.chat_topic == "House coordination"
        assert len(rpc_calls) == 1

    @pytest.mark.asyncio
    async def test_group_update_suppresses_stale_topic_until_refresh_completes(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(
            monkeypatch,
            account="bot-account",
            group_allowed="group123",
        )
        adapter._group_descriptions["group123"] = "Old purpose"
        adapter._group_description_revisions["group123"] = 3
        release_refresh = asyncio.Event()

        async def fake_rpc(method, params, rpc_id=None, **kwargs):
            await release_refresh.wait()
            return [{"id": "group123", "description": "New purpose"}]

        adapter._rpc = fake_rpc
        handled = []

        async def fake_handle(event):
            handled.append(event)

        adapter.handle_message = fake_handle

        await adapter._handle_envelope(
            self._group_envelope(message="", revision=4, group_type="UPDATE")
        )

        assert handled == []

        await adapter._handle_envelope(
            self._group_envelope(message="message during refresh", revision=4)
        )
        assert handled[-1].source.chat_topic is None

        release_refresh.set()
        task = adapter._group_description_refresh_task
        assert task is not None
        await task

        await adapter._handle_envelope(
            self._group_envelope(message="message after refresh", revision=4)
        )
        assert handled[-1].source.chat_topic == "New purpose"

    @pytest.mark.asyncio
    async def test_description_clear_is_cached_without_repeated_refreshes(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(
            monkeypatch,
            account="bot-account",
            group_allowed="group123",
        )
        adapter._group_descriptions["group123"] = "Old purpose"
        adapter._group_description_revisions["group123"] = 8
        adapter._rpc = AsyncMock(
            return_value=[{"id": "group123", "description": "   "}]
        )
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope(
            self._group_envelope(message="", revision=9, group_type="UPDATE")
        )
        task = adapter._group_description_refresh_task
        assert task is not None
        await task

        await adapter._handle_envelope(
            self._group_envelope(message="after clear", revision=9)
        )

        event = adapter.handle_message.await_args.args[0]
        assert event.source.chat_topic is None
        assert "group123" in adapter._group_descriptions
        assert adapter._group_descriptions["group123"] is None
        adapter._rpc.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_newer_revision_during_refresh_is_coalesced_and_rechecked(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(
            monkeypatch,
            account="bot-account",
            group_allowed="group123",
        )
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        calls = 0

        async def fake_rpc(method, params, rpc_id=None, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                await release_first.wait()
                return [{"id": "group123", "description": "Revision four"}]
            return [{"id": "group123", "description": "Revision five"}]

        adapter._rpc = fake_rpc
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope(
            self._group_envelope(message="", revision=4, group_type="UPDATE")
        )
        await first_started.wait()
        await adapter._handle_envelope(
            self._group_envelope(message="", revision=5, group_type="UPDATE")
        )

        release_first.set()
        task = adapter._group_description_refresh_task
        assert task is not None
        await task

        assert calls == 2
        assert adapter._group_descriptions["group123"] == "Revision five"
        assert adapter._group_description_revisions["group123"] == 5

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("event_revision", "cached_revision", "stale_revision", "fresh_revision"),
        [
            (4, 2, 3, 4),
            (None, 5, 4, 6),
        ],
    )
    async def test_revision_lagging_refresh_is_retried_before_publish(
        self,
        monkeypatch,
        event_revision,
        cached_revision,
        stale_revision,
        fresh_revision,
    ):
        from gateway.platforms import signal as signal_module

        adapter = _make_signal_adapter(
            monkeypatch,
            account="bot-account",
            group_allowed="group123",
        )
        adapter._group_descriptions["group123"] = "Previously confirmed"
        adapter._group_description_revisions["group123"] = cached_revision
        monkeypatch.setattr(signal_module, "SIGNAL_GROUP_DESCRIPTION_RETRY_DELAY", 0)
        calls = 0

        async def fake_rpc(method, params, rpc_id=None, **kwargs):
            nonlocal calls
            calls += 1
            revision = stale_revision if calls == 1 else fresh_revision
            return [
                {
                    "id": "group123",
                    "description": f"Revision {revision}",
                    "revision": revision,
                }
            ]

        adapter._rpc = fake_rpc
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope(
            self._group_envelope(
                message="",
                revision=event_revision,
                group_type="UPDATE",
            )
        )
        task = adapter._group_description_refresh_task
        assert task is not None
        await task

        assert calls == 2
        assert adapter._group_descriptions["group123"] == f"Revision {fresh_revision}"
        assert adapter._group_description_revisions["group123"] == fresh_revision

    @pytest.mark.asyncio
    async def test_failed_refresh_keeps_stale_value_but_does_not_inject_it(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(
            monkeypatch,
            account="bot-account",
            group_allowed="group123",
        )
        adapter._group_descriptions["group123"] = "Old purpose"
        adapter._group_description_revisions["group123"] = 2
        adapter._rpc = AsyncMock(return_value=None)
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope(
            self._group_envelope(message="new revision", revision=3)
        )
        task = adapter._group_description_refresh_task
        assert task is not None
        for _ in range(10):
            if adapter._rpc.await_count:
                break
            await asyncio.sleep(0)

        event = adapter.handle_message.await_args.args[0]
        assert event.source.chat_topic is None
        assert adapter._group_descriptions["group123"] == "Old purpose"
        assert adapter._group_description_revisions["group123"] == 2
        await adapter.disconnect()
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_disallowed_group_does_not_trigger_metadata_rpc(self, monkeypatch):
        adapter = _make_signal_adapter(
            monkeypatch,
            account="bot-account",
            group_allowed="different-group",
        )
        adapter._rpc = AsyncMock()
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope(self._group_envelope())
        await asyncio.sleep(0)

        adapter._rpc.assert_not_awaited()
        adapter.handle_message.assert_not_awaited()
        assert adapter._group_description_refresh_task is None
        assert adapter._group_description_refresh_requests == {}

    @pytest.mark.asyncio
    async def test_disconnect_cancels_inflight_description_refresh(self, monkeypatch):
        adapter = _make_signal_adapter(
            monkeypatch,
            account="bot-account",
            group_allowed="group123",
        )
        refresh_started = asyncio.Event()

        async def fake_rpc(method, params, rpc_id=None, **kwargs):
            refresh_started.set()
            await asyncio.Event().wait()

        adapter._rpc = fake_rpc
        adapter._maybe_schedule_group_description_refresh(
            "group123",
            {"revision": 1, "type": "DELIVER"},
        )
        task = adapter._group_description_refresh_task
        assert task is not None
        await refresh_started.wait()

        await adapter.disconnect()

        assert task.cancelled()
        assert adapter._group_description_refresh_task is None
        assert adapter._group_description_refresh_requests == {}

    @pytest.mark.asyncio
    async def test_revisionless_update_invalidates_cache_and_is_not_lost(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(
            monkeypatch,
            account="bot-account",
            group_allowed="group123",
        )
        adapter._group_descriptions["group123"] = "Old purpose"
        adapter._group_description_revisions["group123"] = 3
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        calls = 0

        async def fake_rpc(method, params, rpc_id=None, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                await release_first.wait()
                return [
                    {
                        "id": "group123",
                        "description": "Intermediate purpose",
                        "revision": 4,
                    }
                ]
            return [
                {
                    "id": "group123",
                    "description": "Latest purpose",
                    "revision": 5,
                }
            ]

        adapter._rpc = fake_rpc
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope(
            self._group_envelope(message="", revision=4, group_type="UPDATE")
        )
        await first_started.wait()
        await adapter._handle_envelope(
            self._group_envelope(message="", revision=None, group_type="UPDATE")
        )
        await adapter._handle_envelope(
            self._group_envelope(
                message="message during refresh",
                revision=None,
                group_type="DELIVER",
            )
        )

        event = adapter.handle_message.await_args.args[0]
        assert event.source.chat_topic is None

        release_first.set()
        await adapter._group_description_refresh_task

        assert calls == 2
        assert adapter._group_descriptions["group123"] == "Latest purpose"
        assert adapter._group_description_revisions["group123"] == 5

    @pytest.mark.asyncio
    async def test_revision_omitting_response_started_after_update_is_authoritative(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(
            monkeypatch,
            account="bot-account",
            group_allowed="group123",
        )
        adapter._group_descriptions["group123"] = "Old purpose"
        adapter._group_description_revisions["group123"] = 3
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        calls = 0

        async def fake_rpc(method, params, rpc_id=None, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                await release_first.wait()
                description = "Intermediate purpose"
            else:
                description = "Latest purpose"
            # Deployed signal-cli v0.14.6 listGroups does not expose a
            # revision, so ordering comes from starting another targeted RPC
            # after the revisionless invalidation.
            return [{"id": "group123", "description": description}]

        adapter._rpc = fake_rpc
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope(
            self._group_envelope(message="", revision=4, group_type="UPDATE")
        )
        await first_started.wait()
        await adapter._handle_envelope(
            self._group_envelope(message="", revision=None, group_type="UPDATE")
        )
        release_first.set()

        task = adapter._group_description_refresh_task
        assert task is not None
        await task

        assert calls == 2
        assert adapter._group_descriptions["group123"] == "Latest purpose"
        assert adapter._group_description_revisions["group123"] == 4

    @pytest.mark.asyncio
    async def test_revisionless_update_requires_revision_advance_when_available(
        self, monkeypatch
    ):
        from gateway.platforms import signal as signal_module

        monkeypatch.setattr(signal_module, "SIGNAL_GROUP_DESCRIPTION_RETRY_DELAY", 0)
        adapter = _make_signal_adapter(
            monkeypatch,
            account="bot-account",
            group_allowed="group123",
        )
        adapter._group_descriptions["group123"] = "Revision three"
        adapter._group_description_revisions["group123"] = 3
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        calls = 0

        async def fake_rpc(method, params, rpc_id=None, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                await release_first.wait()
            revision = 4 if calls < 3 else 5
            return [
                {
                    "id": "group123",
                    "description": f"Revision {revision}",
                    "revision": revision,
                }
            ]

        adapter._rpc = fake_rpc
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope(
            self._group_envelope(message="", revision=4, group_type="UPDATE")
        )
        await first_started.wait()
        await adapter._handle_envelope(
            self._group_envelope(message="", revision=None, group_type="UPDATE")
        )

        release_first.set()
        task = adapter._group_description_refresh_task
        assert task is not None
        await task

        assert calls == 3
        assert adapter._group_descriptions["group123"] == "Revision 5"
        assert adapter._group_description_revisions["group123"] == 5

    @pytest.mark.asyncio
    async def test_repeated_revisionless_updates_advance_floor_during_each_rpc(
        self, monkeypatch
    ):
        from gateway.platforms import signal as signal_module

        monkeypatch.setattr(signal_module, "SIGNAL_GROUP_DESCRIPTION_RETRY_DELAY", 0)
        adapter = _make_signal_adapter(
            monkeypatch,
            account="bot-account",
            group_allowed="group123",
        )
        adapter._group_descriptions["group123"] = "Revision three"
        adapter._group_description_revisions["group123"] = 3
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        second_started = asyncio.Event()
        release_second = asyncio.Event()
        calls = 0

        async def fake_rpc(method, params, rpc_id=None, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                await release_first.wait()
                revision = 4
            elif calls == 2:
                second_started.set()
                await release_second.wait()
                revision = 5
            elif calls == 3:
                revision = 5
            else:
                revision = 6
            return [
                {
                    "id": "group123",
                    "description": f"Revision {revision}",
                    "revision": revision,
                }
            ]

        adapter._rpc = fake_rpc
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope(
            self._group_envelope(message="", revision=4, group_type="UPDATE")
        )
        await first_started.wait()
        await adapter._handle_envelope(
            self._group_envelope(message="", revision=None, group_type="UPDATE")
        )
        release_first.set()

        await second_started.wait()
        await adapter._handle_envelope(
            self._group_envelope(message="", revision=None, group_type="UPDATE")
        )
        release_second.set()

        task = adapter._group_description_refresh_task
        assert task is not None
        await task

        assert calls == 4
        assert adapter._group_descriptions["group123"] == "Revision 6"
        assert adapter._group_description_revisions["group123"] == 6

    @pytest.mark.asyncio
    async def test_failed_refresh_retries_without_waiting_for_another_message(
        self, monkeypatch
    ):
        from gateway.platforms import signal as signal_module

        monkeypatch.setattr(
            signal_module,
            "SIGNAL_GROUP_DESCRIPTION_RETRY_DELAY",
            0.01,
        )
        adapter = _make_signal_adapter(
            monkeypatch,
            account="bot-account",
            group_allowed="group123",
        )
        calls = 0

        async def fake_rpc(method, params, rpc_id=None, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return None
            return [
                {
                    "id": "group123",
                    "description": "Recovered purpose",
                    "revision": 7,
                }
            ]

        adapter._rpc = fake_rpc
        adapter._maybe_schedule_group_description_refresh(
            "group123",
            {"revision": 7, "type": "UPDATE"},
        )

        await adapter._group_description_refresh_task

        assert calls == 2
        assert adapter._group_descriptions["group123"] == "Recovered purpose"
        assert adapter._group_description_revisions["group123"] == 7

    @pytest.mark.asyncio
    async def test_permanent_failure_is_bounded_and_does_not_starve_other_groups(
        self, monkeypatch
    ):
        from gateway.platforms import signal as signal_module

        monkeypatch.setattr(signal_module, "SIGNAL_GROUP_DESCRIPTION_RETRY_DELAY", 0)
        monkeypatch.setattr(signal_module, "SIGNAL_GROUP_DESCRIPTION_MAX_RETRIES", 2)
        adapter = _make_signal_adapter(
            monkeypatch,
            account="bot-account",
            group_allowed="*",
        )
        calls = []

        async def fake_rpc(method, params, rpc_id=None, **kwargs):
            group_id = params["groupId"][0]
            calls.append(group_id)
            if group_id == "failing-group":
                return None
            return [
                {
                    "id": group_id,
                    "description": "Healthy purpose",
                    "revision": 1,
                }
            ]

        adapter._rpc = fake_rpc
        adapter._maybe_schedule_group_description_refresh(
            "failing-group",
            {"revision": 1, "type": "UPDATE"},
        )
        adapter._maybe_schedule_group_description_refresh(
            "healthy-group",
            {"revision": 1, "type": "UPDATE"},
        )

        task = adapter._group_description_refresh_task
        assert task is not None
        await task

        assert calls == ["failing-group", "healthy-group", "failing-group"]
        assert adapter._group_descriptions == {"healthy-group": "Healthy purpose"}
        assert adapter._group_description_refresh_requests == {}
        assert adapter._group_description_retry_after == {}
        assert adapter._group_description_refresh_task is None

    @pytest.mark.asyncio
    async def test_description_cache_evicts_oldest_entries(self, monkeypatch):
        from gateway.platforms import signal as signal_module

        monkeypatch.setattr(signal_module, "SIGNAL_GROUP_DESCRIPTION_CACHE_LIMIT", 3)
        adapter = _make_signal_adapter(
            monkeypatch,
            account="bot-account",
            group_allowed="*",
        )

        async def fake_rpc(method, params, rpc_id=None, **kwargs):
            group_id = params["groupId"][0]
            return [{"id": group_id, "description": group_id, "revision": 1}]

        adapter._rpc = fake_rpc
        for index in range(5):
            adapter._maybe_schedule_group_description_refresh(
                f"group-{index}",
                {"revision": 1, "type": "UPDATE"},
            )

        task = adapter._group_description_refresh_task
        assert task is not None
        await task

        assert list(adapter._group_descriptions) == ["group-2", "group-3", "group-4"]
        assert set(adapter._group_description_revisions) == {
            "group-2",
            "group-3",
            "group-4",
        }

    @pytest.mark.asyncio
    async def test_many_groups_share_one_description_refresh_worker(self, monkeypatch):
        adapter = _make_signal_adapter(
            monkeypatch,
            account="bot-account",
            group_allowed="*",
        )
        refresh_started = asyncio.Event()
        release_refresh = asyncio.Event()

        async def fake_rpc(method, params, rpc_id=None, **kwargs):
            refresh_started.set()
            await release_refresh.wait()
            group_id = params["groupId"][0]
            return [{"id": group_id, "description": group_id, "revision": 1}]

        adapter._rpc = fake_rpc
        for index in range(100):
            adapter._maybe_schedule_group_description_refresh(
                f"group-{index}",
                {"revision": 1, "type": "UPDATE"},
            )

        await refresh_started.wait()

        assert adapter._group_description_refresh_task is not None
        assert not adapter._group_description_refresh_task.done()
        assert len(adapter._group_description_refresh_requests) == 100

        release_refresh.set()
        await adapter._group_description_refresh_task
        assert len(adapter._group_descriptions) == 100

    @pytest.mark.asyncio
    async def test_confirmed_empty_description_is_authoritative_topic_metadata(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(
            monkeypatch,
            account="bot-account",
            group_allowed="group123",
        )
        adapter._rpc = AsyncMock(
            return_value=[
                {"id": "group123", "description": "", "revision": 9}
            ]
        )
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope(
            self._group_envelope(message="", revision=9, group_type="UPDATE")
        )
        await adapter._group_description_refresh_task
        await adapter._handle_envelope(
            self._group_envelope(message="after clear", revision=9)
        )

        event = adapter.handle_message.await_args.args[0]
        assert event.source.chat_topic is None
        assert event.source.chat_topic_known is True

    @pytest.mark.asyncio
    async def test_reconnect_reset_prevents_revisionless_delivery_from_using_stale_cache(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(
            monkeypatch,
            account="bot-account",
            group_allowed="group123",
        )
        adapter._group_descriptions["group123"] = "Old purpose"
        adapter._group_description_revisions["group123"] = 2
        refresh_started = asyncio.Event()

        async def blocked_rpc(*args, **kwargs):
            refresh_started.set()
            await asyncio.Event().wait()

        adapter._rpc = blocked_rpc
        adapter._maybe_schedule_group_description_refresh(
            "group123",
            {"revision": 3, "type": "UPDATE"},
        )
        stale_task = adapter._group_description_refresh_task
        assert stale_task is not None
        await refresh_started.wait()

        await adapter._reset_group_description_state()

        assert stale_task.cancelled()
        assert adapter._group_descriptions == {}
        assert adapter._group_description_revisions == {}
        assert adapter._group_description_refresh_requests == {}

        adapter._rpc = AsyncMock(
            return_value=[{"id": "group123", "description": "Fresh purpose"}]
        )
        adapter.handle_message = AsyncMock()
        await adapter._handle_envelope(
            self._group_envelope(message="after reconnect", revision=None)
        )

        source = adapter.handle_message.await_args.args[0].source
        assert source.chat_topic is None
        assert source.chat_topic_known is False
        replacement_task = adapter._group_description_refresh_task
        assert replacement_task is not None
        await replacement_task

        adapter.handle_message.reset_mock()
        await adapter._handle_envelope(
            self._group_envelope(message="after refresh", revision=None)
        )
        source = adapter.handle_message.await_args.args[0].source
        assert source.chat_topic == "Fresh purpose"
        assert source.chat_topic_known is True


class TestSignalConnectCleanup:
    """Regression coverage for failed connect() cleanup."""

    @pytest.mark.asyncio
    async def test_releases_lock_and_closes_client_on_healthcheck_failure(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=503))
        mock_client.aclose = AsyncMock()

        with patch("gateway.platforms.signal.httpx.AsyncClient", return_value=mock_client), \
             patch("gateway.status.acquire_scoped_lock", return_value=(True, None)), \
             patch("gateway.status.release_scoped_lock") as mock_release:
            result = await adapter.connect()

        assert result is False
        mock_client.aclose.assert_awaited_once()
        mock_release.assert_called_once_with("signal-phone", "+15551234567")
        assert adapter.client is None
        assert adapter._platform_lock_identity is None


class TestSignalHelpers:
    def test_redact_phone_long(self):
        from gateway.platforms.helpers import redact_phone
        assert redact_phone("+155****4567") == "+155****4567"

    def test_redact_phone_short(self):
        from gateway.platforms.helpers import redact_phone
        assert redact_phone("+12345") == "+1****45"

    def test_redact_phone_empty(self):
        from gateway.platforms.helpers import redact_phone
        assert redact_phone("") == "<none>"

    def test_parse_comma_list(self):
        from gateway.platforms.signal import _parse_comma_list
        assert _parse_comma_list("+1234, +5678 , +9012") == ["+1234", "+5678", "+9012"]
        assert _parse_comma_list("") == []
        assert _parse_comma_list("  ,  ,  ") == []

    def test_guess_extension_png(self):
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100) == ".png"

    def test_guess_extension_jpeg(self):
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"\xff\xd8\xff\xe0" + b"\x00" * 100) == ".jpg"

    def test_guess_extension_pdf(self):
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"%PDF-1.4" + b"\x00" * 100) == ".pdf"

    def test_guess_extension_zip(self):
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"PK\x03\x04" + b"\x00" * 100) == ".zip"

    def test_guess_extension_mp4(self):
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100) == ".mp4"

    def test_guess_extension_wav(self):
        """RIFF/WAVE must be detected as ``.wav`` (regression for the WAV gap).

        WAV shares the ``RIFF`` chunk header with WebP; only the form-type at
        bytes 8-11 (``WAVE`` vs ``WEBP``) distinguishes them. Before the fix the
        sniffer only handled ``WEBP`` and a WAV file fell through to ``.bin``.
        """
        from gateway.platforms.signal import _guess_extension
        # Canonical WAV header: 'RIFF' <size> 'WAVE' 'fmt '...
        wav = b"RIFF\x24\x08\x00\x00WAVEfmt " + b"\x00" * 100
        assert _guess_extension(wav) == ".wav"

    def test_guess_extension_wav_not_misread_as_webp(self):
        """A RIFF container that is NOT WebP must not be returned as ``.webp``."""
        from gateway.platforms.signal import _guess_extension
        wav = b"RIFF\x24\x08\x00\x00WAVEfmt " + b"\x00" * 100
        assert _guess_extension(wav) != ".webp"

    def test_guess_extension_wav_routes_to_audio_cache(self):
        """A detected WAV must route to the audio cache, not the document cache.

        ``.wav`` is already in ``_is_audio_ext``; the bug was purely that
        ``_guess_extension`` never produced ``.wav`` for raw bytes, so the
        attachment was treated as a document and STT never received it.
        """
        from gateway.platforms.signal import _is_audio_ext, _guess_extension
        wav = b"RIFF\x24\x08\x00\x00WAVEfmt " + b"\x00" * 100
        ext = _guess_extension(wav)
        assert ext == ".wav"
        assert _is_audio_ext(ext) is True

    def test_guess_extension_webp_still_detected(self):
        """Guard that adding WAVE detection didn't break the WebP branch."""
        from gateway.platforms.signal import _guess_extension
        webp = b"RIFF\x24\x08\x00\x00WEBPVP8 " + b"\x00" * 100
        assert _guess_extension(webp) == ".webp"

    def test_guess_extension_m4a_audio_brand(self):
        """iOS Signal voice notes are MP4-container AAC with an M4A ftyp brand.

        Classifying them as ``.mp4`` sent them to the document cache and made
        STT reject the upload ("Invalid file format") even though the bytes
        were valid audio. Audio brands must resolve to ``.m4a``.
        """
        from gateway.platforms.signal import _guess_extension, _is_audio_ext
        for brand in (b"M4A ", b"M4B ", b"m4a "):
            data = b"\x00\x00\x00\x1cftyp" + brand + b"\x00" * 100
            assert _guess_extension(data) == ".m4a", brand
            assert _is_audio_ext(_guess_extension(data)) is True

    def test_guess_extension_video_brands_stay_mp4(self):
        """Real video ftyp brands must not be swept up by the M4A fix."""
        from gateway.platforms.signal import _guess_extension
        for brand in (b"isom", b"mp42", b"avc1", b"qt  "):
            data = b"\x00\x00\x00\x1cftyp" + brand + b"\x00" * 100
            assert _guess_extension(data) == ".mp4", brand

    def test_guess_extension_aac_adts_unprotected(self):
        """ADTS AAC, MPEG-4, no CRC (the canonical Android Signal voice note).

        Byte 0 = 0xFF (sync high), byte 1 = 0xF1 (sync low + ID=0 + layer=00
        + protection_absent=1). Must NOT be misclassified as MP3 — the old
        code's ``(b[1] & 0xE0) == 0xE0`` test wrongly returned ``.mp3``.
        """
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"\xff\xf1" + b"\x00" * 200) == ".aac"

    def test_guess_extension_aac_adts_protected(self):
        """ADTS AAC, MPEG-4, CRC present (protection_absent=0)."""
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"\xff\xf0" + b"\x00" * 200) == ".aac"

    def test_guess_extension_mp3_mpeg1_layer3(self):
        """Real MP3 frame, MPEG-1 Layer 3: byte1 = 0xFB (ID=1, layer=01, prot=1)."""
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"\xff\xfb" + b"\x00" * 200) == ".mp3"

    def test_guess_extension_mp3_mpeg2_layer3(self):
        """Real MP3 frame, MPEG-2 Layer 3: byte1 = 0xF3 (ID=1, layer=01, prot=1)."""
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"\xff\xf3" + b"\x00" * 200) == ".mp3"

    def test_guess_extension_aac_routes_to_audio_cache(self):
        """ADTS-detected files must be routed to the audio cache, not document.

        ``_is_audio_ext(``.aac``)`` is True, so a Signal attachment that
        begins with the ADTS sync word ends up in ``cache_audio_from_bytes``,
        which the remux step then converts to MP4 container.
        """
        from gateway.platforms.signal import _is_audio_ext, _guess_extension
        ext = _guess_extension(b"\xff\xf1" + b"\x00" * 200)
        assert ext == ".aac"
        assert _is_audio_ext(ext) is True

    def test_remux_aac_to_m4a_round_trip(self):
        """A real ADTS AAC stream remuxes to a valid MP4 (.m4a) container.

        Generates a short ADTS AAC sample with ffmpeg at runtime so the
        end-to-end remux path actually exercises in CI (skipped only when
        ffmpeg is unavailable), rather than depending on a machine-specific
        file.
        """
        import shutil
        import subprocess
        import tempfile
        from gateway.platforms.signal import _remux_aac_to_m4a

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            import pytest
            pytest.skip("ffmpeg not available in this env")

        # Synthesize 0.5s of silence encoded as raw ADTS AAC.
        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as tmp:
            adts_path = tmp.name
        try:
            gen = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi",
                 "-i", "anullsrc=r=44100:cl=mono", "-t", "0.5",
                 "-c:a", "aac", "-f", "adts", adts_path],
                capture_output=True, timeout=30,
            )
            if gen.returncode != 0:
                import pytest
                pytest.skip("ffmpeg could not produce an ADTS AAC sample")
            with open(adts_path, "rb") as f:
                aac_data = f.read()
        finally:
            try:
                import os
                os.unlink(adts_path)
            except OSError:
                pass

        result = _remux_aac_to_m4a(aac_data)
        assert result is not None
        m4a_bytes, ext = result
        assert ext == ".m4a"
        # MP4 files start with a 4-byte size, then ``ftyp`` at offset 4.
        assert m4a_bytes[4:8] == b"ftyp", \
            f"expected MP4 ftyp box, got {m4a_bytes[:12]!r}"
        # File must be at least as long as the input (MP4 has overhead).
        assert len(m4a_bytes) >= len(aac_data) * 0.5

    def test_remux_aac_to_m4a_handles_garbage(self):
        """Garbage input should return None, not raise."""
        from gateway.platforms.signal import _remux_aac_to_m4a
        result = _remux_aac_to_m4a(b"\xff\xf1garbage_no_aac_frames")
        # Either returns None (ffmpeg errored) or a real M4A. If it returned
        # bytes, the bytes must look like an MP4. Otherwise it returns None.
        if result is not None:
            m4a_bytes, ext = result
            assert ext == ".m4a"

    def test_guess_extension_unknown(self):
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"\x00\x01\x02\x03" * 10) == ".bin"

    def test_is_image_ext(self):
        from gateway.platforms.signal import _is_image_ext
        assert _is_image_ext(".png") is True
        assert _is_image_ext(".jpg") is True
        assert _is_image_ext(".gif") is True
        assert _is_image_ext(".pdf") is False

    def test_is_audio_ext(self):
        from gateway.platforms.signal import _is_audio_ext
        assert _is_audio_ext(".mp3") is True
        assert _is_audio_ext(".ogg") is True
        assert _is_audio_ext(".png") is False

    def test_check_requirements(self, monkeypatch):
        from gateway.platforms.signal import check_signal_requirements
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:8080")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+15551234567")
        assert check_signal_requirements() is True

    def test_render_mentions(self):
        from gateway.platforms.signal import _render_mentions
        text = "Hello \uFFFC, how are you?"
        mentions = [{"start": 6, "length": 1, "number": "+15559999999"}]
        result = _render_mentions(text, mentions)
        assert "@+15559999999" in result
        assert "\uFFFC" not in result

    def test_render_mentions_no_mentions(self):
        from gateway.platforms.signal import _render_mentions
        text = "Hello world"
        result = _render_mentions(text, [])
        assert result == "Hello world"

    def test_check_requirements_missing(self, monkeypatch):
        from gateway.platforms.signal import check_signal_requirements
        monkeypatch.delenv("SIGNAL_HTTP_URL", raising=False)
        monkeypatch.delenv("SIGNAL_ACCOUNT", raising=False)
        assert check_signal_requirements() is True

    def test_validate_signal_config_accepts_platform_values(self, monkeypatch):
        monkeypatch.delenv("SIGNAL_HTTP_URL", raising=False)
        monkeypatch.delenv("SIGNAL_ACCOUNT", raising=False)
        from gateway.platforms.signal import validate_signal_config

        config = PlatformConfig(
            enabled=True,
            extra={
                "http_url": "http://localhost:8080",
                "account": "+155****4567",
            },
        )
        assert validate_signal_config(config) is True

    def test_validate_signal_config_rejects_missing_account(self, monkeypatch):
        monkeypatch.delenv("SIGNAL_HTTP_URL", raising=False)
        monkeypatch.delenv("SIGNAL_ACCOUNT", raising=False)
        from gateway.platforms.signal import validate_signal_config

        config = PlatformConfig(enabled=True, extra={"http_url": "http://localhost:8080"})
        assert validate_signal_config(config) is False


class TestSignalMentionObserveOnly:
    """Signal groups should be observed passively until the bot is mentioned."""

    @pytest.mark.asyncio
    async def test_unmentioned_group_message_is_observe_only(self, monkeypatch):
        adapter = _make_signal_adapter(
            monkeypatch,
            group_allowed="group123",
            require_mention=True,
        )
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope({
            "sourceNumber": "+155****1111",
            "sourceName": "Alice",
            "timestamp": 1712345678000,
            "dataMessage": {
                "message": "regular group chatter",
                "groupInfo": {"groupId": "group123", "groupName": "House"},
            },
        })

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.observe_only is True
        assert event.text == "regular group chatter"
        assert event.source.chat_id == "group:group123"
        assert event.source.chat_type == "group"

    @pytest.mark.asyncio
    async def test_mentioned_group_message_remains_active(self, monkeypatch):
        adapter = _make_signal_adapter(
            monkeypatch,
            group_allowed="group123",
            require_mention=True,
        )
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope({
            "sourceNumber": "+155****1111",
            "sourceName": "Alice",
            "timestamp": 1712345678000,
            "dataMessage": {
                "message": "\uFFFC please summarize",
                "mentions": [{"start": 0, "length": 1, "number": "+15551234567"}],
                "groupInfo": {"groupId": "group123", "groupName": "House"},
            },
        })

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.observe_only is False
        assert event.text == "please summarize"

    @pytest.mark.asyncio
    async def test_mention_can_match_cached_bot_uuid(self, monkeypatch):
        adapter = _make_signal_adapter(
            monkeypatch,
            group_allowed="group123",
            require_mention=True,
        )
        adapter._recipient_uuid_by_number["+15551234567"] = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope({
            "sourceNumber": "+155****1111",
            "sourceName": "Alice",
            "timestamp": 1712345678000,
            "dataMessage": {
                "message": "\uFFFC ping",
                "mentions": [{
                    "start": 0,
                    "length": 1,
                    "uuid": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
                }],
                "groupInfo": {"groupId": "group123", "groupName": "House"},
            },
        })

        event = adapter.handle_message.await_args.args[0]
        assert event.observe_only is False

    def test_env_override_parses_signal_require_mention(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:9090")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+155****4567")
        monkeypatch.setenv("SIGNAL_REQUIRE_MENTION", "true")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert config.platforms[Platform.SIGNAL].extra["require_mention"] is True

    @pytest.mark.asyncio
    async def test_gateway_records_observe_only_without_running_agent(self):
        from gateway.platforms.base import MessageEvent
        from gateway.run import GatewayRunner
        from gateway.session import SessionEntry, SessionSource

        runner = object.__new__(GatewayRunner)
        runner.session_store = MagicMock()
        runner.session_store.get_or_create_session.return_value = SessionEntry(
            session_key="signal:group:group123",
            session_id="session-123",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            platform=Platform.SIGNAL,
            chat_type="group",
        )
        runner.session_store._generate_session_key.return_value = "signal:group:group123"
        runner._update_prompt_pending = {}
        runner._is_user_authorized = MagicMock(return_value=True)
        runner._session_key_for_source = lambda source: "signal:group:group123"

        source = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="group:group123",
            chat_name="House",
            chat_type="group",
            user_id="+155****1111",
            user_name="Alice",
        )
        event = MessageEvent(
            source=source,
            text="background context",
            observe_only=True,
        )

        result = await runner._handle_message(event)

        assert result is None
        runner.session_store.get_or_create_session.assert_called_once_with(source)
        runner.session_store.append_to_transcript.assert_called_once()
        session_id, entry = runner.session_store.append_to_transcript.call_args.args
        assert session_id == "session-123"
        assert entry["role"] == "user"
        assert entry["content"] == "[Observed group message from Alice]: background context"


# ---------------------------------------------------------------------------
# SSE URL Encoding (Bug Fix: phone numbers with + must be URL-encoded)
# ---------------------------------------------------------------------------

class TestSignalSSEUrlEncoding:
    """Verify that phone numbers with + are URL-encoded in the SSE endpoint."""

    def test_sse_url_encodes_plus_in_account(self):
        """The + in E.164 phone numbers must be percent-encoded in the SSE query string."""
        encoded = quote("+31612345678", safe="")
        assert encoded == "%2B31612345678"

    def test_sse_url_encoding_preserves_digits(self):
        """Digits and country codes should pass through URL encoding unchanged."""
        assert quote("+15551234567", safe="") == "%2B15551234567"


class TestSignalSSEReconnectDescriptionInvalidation:
    @pytest.mark.asyncio
    async def test_second_sse_connection_invalidates_description_cache(
        self, monkeypatch
    ):
        from gateway.platforms import signal as signal_module

        adapter = _make_signal_adapter(monkeypatch)
        adapter._group_descriptions["group123"] = "Old purpose"
        adapter._group_description_revisions["group123"] = 2
        stream_count = 0

        class Response:
            is_stream_consumed = False

            def __init__(self, stop: bool):
                self.stop = stop

            async def aiter_text(self):
                if self.stop:
                    adapter._running = False
                if False:
                    yield ""

        class StreamContext:
            def __init__(self, response):
                self.response = response

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, exc_type, exc, tb):
                return False

        def stream(*args, **kwargs):
            nonlocal stream_count
            stream_count += 1
            return StreamContext(Response(stop=stream_count == 2))

        async def no_sleep(_delay):
            return None

        adapter.client = MagicMock()
        adapter.client.stream = stream
        adapter._running = True
        monkeypatch.setattr(signal_module.asyncio, "sleep", no_sleep)

        await adapter._sse_listener()

        assert stream_count == 2
        assert adapter._group_descriptions == {}
        assert adapter._group_description_revisions == {}


# ---------------------------------------------------------------------------
# Attachment Fetch (Bug Fix: parameter must be "id" not "attachmentId")
# ---------------------------------------------------------------------------

class TestSignalAttachmentFetch:
    """Verify that _fetch_attachment uses the correct RPC parameter name."""

    @pytest.mark.asyncio
    async def test_fetch_attachment_uses_id_parameter(self, monkeypatch):
        """RPC getAttachment must use 'id', not 'attachmentId' (signal-cli requirement)."""
        adapter = _make_signal_adapter(monkeypatch)

        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        b64_data = base64.b64encode(png_data).decode()

        adapter._rpc, captured = _stub_rpc({"data": b64_data})

        with patch("gateway.platforms.signal.cache_image_from_bytes", return_value="/tmp/test.png"):
            await adapter._fetch_attachment("attachment-123")

        call = captured[0]
        assert call["method"] == "getAttachment"
        assert call["params"]["id"] == "attachment-123"
        assert "attachmentId" not in call["params"], "Must NOT use 'attachmentId' — causes NullPointerException in signal-cli"
        assert call["params"]["account"] == "+15551234567"

    @pytest.mark.asyncio
    async def test_fetch_attachment_returns_none_on_empty(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._rpc, _ = _stub_rpc(None)
        path, ext = await adapter._fetch_attachment("missing-id")
        assert path is None
        assert ext == ""

    @pytest.mark.asyncio
    async def test_fetch_attachment_handles_dict_response(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)

        pdf_data = b"%PDF-1.4" + b"\x00" * 100
        b64_data = base64.b64encode(pdf_data).decode()

        adapter._rpc, _ = _stub_rpc({"data": b64_data})

        with patch("gateway.platforms.signal.cache_document_from_bytes", return_value="/tmp/test.pdf"):
            path, ext = await adapter._fetch_attachment("doc-456")

        assert path == "/tmp/test.pdf"
        assert ext == ".pdf"


# ---------------------------------------------------------------------------
# Session Source
# ---------------------------------------------------------------------------

class TestSignalSessionSource:
    def test_session_source_alt_fields(self):
        from gateway.session import SessionSource
        source = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="+15551234567",
            user_id="+15551234567",
            user_id_alt="uuid:abc-123",
            chat_id_alt=None,
        )
        d = source.to_dict()
        assert d["user_id_alt"] == "uuid:abc-123"
        assert "chat_id_alt" not in d  # None fields excluded

    def test_session_source_roundtrip(self):
        from gateway.session import SessionSource
        source = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="group:xyz",
            chat_type="group",
            user_id="+15551234567",
            user_id_alt="uuid:abc",
            chat_id_alt="xyz",
        )
        d = source.to_dict()
        restored = SessionSource.from_dict(d)
        assert restored.user_id_alt == "uuid:abc"
        assert restored.chat_id_alt == "xyz"
        assert restored.platform == Platform.SIGNAL


# ---------------------------------------------------------------------------
# Phone Redaction in agent/redact.py
# ---------------------------------------------------------------------------

class TestSignalPhoneRedaction:
    @pytest.fixture(autouse=True)
    def _ensure_redaction_enabled(self, monkeypatch):
        # agent.redact snapshots _REDACT_ENABLED at import time from the
        # HERMES_REDACT_SECRETS env var. monkeypatch.delenv is too late —
        # the module was already imported during test collection with
        # whatever value was in the env then. Force the flag directly.
        # See skill: xdist-cross-test-pollution Pattern 5.
        monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)

    def test_us_number(self):
        from agent.redact import redact_sensitive_text
        result = redact_sensitive_text("Call +15551234567 now")
        assert "+15551234567" not in result
        assert "+155" in result  # Prefix preserved
        assert "4567" in result  # Suffix preserved

    def test_uk_number(self):
        from agent.redact import redact_sensitive_text
        result = redact_sensitive_text("UK: +442071838750")
        assert "+442071838750" not in result
        assert "****" in result

    def test_multiple_numbers(self):
        from agent.redact import redact_sensitive_text
        text = "From +15551234567 to +442071838750"
        result = redact_sensitive_text(text)
        assert "+15551234567" not in result
        assert "+442071838750" not in result

    def test_short_number_not_matched(self):
        from agent.redact import redact_sensitive_text
        result = redact_sensitive_text("Code: +12345")
        # 5 digits after + is below the 7-digit minimum
        assert "+12345" in result  # Too short to redact


# ---------------------------------------------------------------------------
# Authorization in run.py
# ---------------------------------------------------------------------------

class TestSignalAuthorization:
    def test_signal_in_allowlist_maps(self):
        """Signal should be in the platform auth maps."""
        from gateway.run import GatewayRunner
        from gateway.config import GatewayConfig

        gw = GatewayRunner.__new__(GatewayRunner)
        gw.config = GatewayConfig()
        gw.pairing_store = MagicMock()
        gw.pairing_store.is_approved.return_value = False

        source = MagicMock()
        source.platform = Platform.SIGNAL
        source.user_id = "+15559999999"

        # No allowlists set — should check GATEWAY_ALLOW_ALL_USERS
        with patch.dict("os.environ", {}, clear=True):
            result = gw._is_user_authorized(source)
            assert result is False


# ---------------------------------------------------------------------------
# Send Message Tool
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# send_image_file method (#5105)
# ---------------------------------------------------------------------------

class TestSignalSendImageFile:
    @pytest.mark.asyncio
    async def test_send_image_file_sends_via_rpc(self, monkeypatch, tmp_path):
        """send_image_file should send image as attachment via signal-cli RPC."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc({"timestamp": 1234567890})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        img_bytes = b"\x89PNG" + b"\x00" * 100
        img_path = tmp_path / "chart.png"
        img_path.write_bytes(img_bytes)

        result = await adapter.send_image_file(chat_id="+155****4567", image_path=str(img_path))

        assert result.success is True
        assert len(captured) == 1
        assert captured[0]["method"] == "send"
        assert captured[0]["params"]["account"] == adapter.account
        assert captured[0]["params"]["recipient"] == ["+155****4567"]
        assert captured[0]["params"]["attachments"][0].startswith("data:image/png;filename=chart.png;base64,")
        encoded = captured[0]["params"]["attachments"][0].split(",", 1)[1]
        assert base64.b64decode(encoded) == img_bytes
        assert captured[0]["params"]["message"] == ""  # caption=None → ""
        # Typing indicator must be stopped before sending
        adapter._stop_typing_indicator.assert_awaited_once_with("+155****4567")
        # Timestamp must be tracked for echo-back prevention
        assert 1234567890 in adapter._recent_sent_timestamps

    @pytest.mark.asyncio
    async def test_send_image_file_to_group(self, monkeypatch, tmp_path):
        """send_image_file should route group chats via groupId."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc({"timestamp": 1234567890})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        img_path = tmp_path / "photo.jpg"
        img_path.write_bytes(b"\xff\xd8" + b"\x00" * 100)

        result = await adapter.send_image_file(
            chat_id="group:abc123==", image_path=str(img_path), caption="Here's the chart"
        )

        assert result.success is True
        assert captured[0]["params"]["groupId"] == "abc123=="
        assert captured[0]["params"]["message"] == "Here's the chart"

    @pytest.mark.asyncio
    async def test_send_image_file_missing(self, monkeypatch):
        """send_image_file should fail gracefully for nonexistent files."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send_image_file(chat_id="+155****4567", image_path="/nonexistent.png")

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_image_file_too_large(self, monkeypatch, tmp_path):
        """send_image_file should reject files over 100MB."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        img_path = tmp_path / "huge.png"
        img_path.write_bytes(b"x")

        def mock_stat(self, **kwargs):
            class FakeStat:
                st_size = 200 * 1024 * 1024  # 200 MB
            return FakeStat()

        with patch.object(Path, "stat", mock_stat):
            result = await adapter.send_image_file(chat_id="+155****4567", image_path=str(img_path))

        assert result.success is False
        assert "too large" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_image_file_rpc_failure(self, monkeypatch, tmp_path):
        """send_image_file should return error when RPC returns None."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc(None)
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG" + b"\x00" * 100)

        result = await adapter.send_image_file(chat_id="+155****4567", image_path=str(img_path))

        assert result.success is False
        assert "failed" in result.error.lower()


class TestSignalRecipientResolution:
    @pytest.mark.asyncio
    async def test_send_prefers_cached_uuid_for_direct_messages(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()
        adapter._remember_recipient_identifiers("+15551230000", "68680952-6d86-45bc-85e0-1a4d186d53ee")

        captured = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            captured.append({"method": method, "params": dict(params)})
            return {"timestamp": 1234567890}

        adapter._rpc = mock_rpc

        result = await adapter.send(chat_id="+15551230000", content="hello")

        assert result.success is True
        assert captured[0]["method"] == "send"
        assert captured[0]["params"]["recipient"] == ["68680952-6d86-45bc-85e0-1a4d186d53ee"]

    @pytest.mark.asyncio
    async def test_send_looks_up_uuid_via_list_contacts(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        captured = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            captured.append({"method": method, "params": dict(params)})
            if method == "listContacts":
                return [{
                    "recipient": "351935789098",
                    "number": "+15551230000",
                    "uuid": "68680952-6d86-45bc-85e0-1a4d186d53ee",
                    "isRegistered": True,
                }]
            if method == "send":
                return {"timestamp": 1234567890}
            return None

        adapter._rpc = mock_rpc

        result = await adapter.send(chat_id="+15551230000", content="hello")

        assert result.success is True
        assert captured[0]["method"] == "listContacts"
        assert captured[1]["method"] == "send"
        assert captured[1]["params"]["recipient"] == ["68680952-6d86-45bc-85e0-1a4d186d53ee"]

    @pytest.mark.asyncio
    async def test_send_falls_back_to_phone_when_no_uuid_found(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        captured = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            captured.append({"method": method, "params": dict(params)})
            if method == "listContacts":
                return []
            if method == "send":
                return {"timestamp": 1234567890}
            return None

        adapter._rpc = mock_rpc

        result = await adapter.send(chat_id="+15551230000", content="hello")

        assert result.success is True
        assert captured[1]["params"]["recipient"] == ["+15551230000"]

    @pytest.mark.asyncio
    async def test_send_typing_uses_cached_uuid(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._remember_recipient_identifiers("+15551230000", "68680952-6d86-45bc-85e0-1a4d186d53ee")

        captured = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            captured.append({"method": method, "params": dict(params), "rpc_id": rpc_id})
            return {}

        adapter._rpc = mock_rpc

        await adapter.send_typing("+15551230000")

        assert captured[0]["method"] == "sendTyping"
        assert captured[0]["params"]["recipient"] == ["68680952-6d86-45bc-85e0-1a4d186d53ee"]


# ---------------------------------------------------------------------------
# send_voice method (#5105)
# ---------------------------------------------------------------------------

class TestSignalSendVoice:
    @pytest.mark.asyncio
    async def test_send_voice_sends_via_rpc(self, monkeypatch, tmp_path):
        """send_voice should send audio as attachment via signal-cli RPC."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc({"timestamp": 1234567890})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        audio_bytes = b"OggS" + b"\x00" * 100
        audio_path = tmp_path / "reply.ogg"
        audio_path.write_bytes(audio_bytes)

        result = await adapter.send_voice(chat_id="+155****4567", audio_path=str(audio_path))

        assert result.success is True
        assert captured[0]["method"] == "send"
        assert captured[0]["params"]["attachments"][0].startswith("data:audio/ogg;filename=reply.ogg;base64,")
        encoded = captured[0]["params"]["attachments"][0].split(",", 1)[1]
        assert base64.b64decode(encoded) == audio_bytes
        assert captured[0]["params"]["message"] == ""  # caption=None → ""
        adapter._stop_typing_indicator.assert_awaited_once_with("+155****4567")
        assert 1234567890 in adapter._recent_sent_timestamps

    @pytest.mark.asyncio
    async def test_send_voice_missing_file(self, monkeypatch):
        """send_voice should fail for nonexistent audio."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send_voice(chat_id="+155****4567", audio_path="/missing.ogg")

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_voice_to_group(self, monkeypatch, tmp_path):
        """send_voice should route group chats correctly."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc({"timestamp": 9999})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        audio_path = tmp_path / "note.mp3"
        audio_path.write_bytes(b"\xff\xe0" + b"\x00" * 100)

        result = await adapter.send_voice(chat_id="group:grp1==", audio_path=str(audio_path))

        assert result.success is True
        assert captured[0]["params"]["groupId"] == "grp1=="

    @pytest.mark.asyncio
    async def test_send_voice_too_large(self, monkeypatch, tmp_path):
        """send_voice should reject files over 100MB."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        audio_path = tmp_path / "huge.ogg"
        audio_path.write_bytes(b"x")

        def mock_stat(self, **kwargs):
            class FakeStat:
                st_size = 200 * 1024 * 1024
            return FakeStat()

        with patch.object(Path, "stat", mock_stat):
            result = await adapter.send_voice(chat_id="+155****4567", audio_path=str(audio_path))

        assert result.success is False
        assert "too large" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_voice_rpc_failure(self, monkeypatch, tmp_path):
        """send_voice should return error when RPC returns None."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc(None)
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        audio_path = tmp_path / "reply.ogg"
        audio_path.write_bytes(b"OggS" + b"\x00" * 100)

        result = await adapter.send_voice(chat_id="+155****4567", audio_path=str(audio_path))

        assert result.success is False
        assert "failed" in result.error.lower()


# ---------------------------------------------------------------------------
# send_video method (#5105)
# ---------------------------------------------------------------------------

class TestSignalSendVideo:
    @pytest.mark.asyncio
    async def test_send_video_sends_via_rpc(self, monkeypatch, tmp_path):
        """send_video should send video as attachment via signal-cli RPC."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc({"timestamp": 1234567890})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        vid_bytes = b"\x00\x00\x00\x18ftyp" + b"\x00" * 100
        vid_path = tmp_path / "demo.mp4"
        vid_path.write_bytes(vid_bytes)

        result = await adapter.send_video(chat_id="+155****4567", video_path=str(vid_path))

        assert result.success is True
        assert captured[0]["method"] == "send"
        assert captured[0]["params"]["attachments"][0].startswith("data:video/mp4;filename=demo.mp4;base64,")
        encoded = captured[0]["params"]["attachments"][0].split(",", 1)[1]
        assert base64.b64decode(encoded) == vid_bytes
        assert captured[0]["params"]["message"] == ""  # caption=None → ""
        adapter._stop_typing_indicator.assert_awaited_once_with("+155****4567")
        assert 1234567890 in adapter._recent_sent_timestamps

    @pytest.mark.asyncio
    async def test_send_video_missing_file(self, monkeypatch):
        """send_video should fail for nonexistent video."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send_video(chat_id="+155****4567", video_path="/missing.mp4")

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_video_too_large(self, monkeypatch, tmp_path):
        """send_video should reject files over 100MB."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        vid_path = tmp_path / "huge.mp4"
        vid_path.write_bytes(b"x")

        def mock_stat(self, **kwargs):
            class FakeStat:
                st_size = 200 * 1024 * 1024
            return FakeStat()

        with patch.object(Path, "stat", mock_stat):
            result = await adapter.send_video(chat_id="+155****4567", video_path=str(vid_path))

        assert result.success is False
        assert "too large" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_video_rpc_failure(self, monkeypatch, tmp_path):
        """send_video should return error when RPC returns None."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc(None)
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        vid_path = tmp_path / "demo.mp4"
        vid_path.write_bytes(b"\x00\x00\x00\x18ftyp" + b"\x00" * 100)

        result = await adapter.send_video(chat_id="+155****4567", video_path=str(vid_path))

        assert result.success is False
        assert "failed" in result.error.lower()


# ---------------------------------------------------------------------------
# MEDIA: tag extraction integration
# ---------------------------------------------------------------------------

class TestSignalMediaExtraction:
    """Verify the full pipeline: MEDIA: tag → extract → send_image_file/send_voice."""

    def test_extract_media_finds_image_tag(self):
        """BasePlatformAdapter.extract_media should find MEDIA: image paths."""
        from gateway.platforms.base import BasePlatformAdapter
        media, cleaned = BasePlatformAdapter.extract_media(
            "Here's the chart.\nMEDIA:/tmp/price_graph.png"
        )
        assert len(media) == 1
        assert media[0][0] == "/tmp/price_graph.png"
        assert "MEDIA:" not in cleaned

    def test_extract_media_finds_audio_tag(self):
        """BasePlatformAdapter.extract_media should find MEDIA: audio paths."""
        from gateway.platforms.base import BasePlatformAdapter
        media, cleaned = BasePlatformAdapter.extract_media(
            "[[audio_as_voice]]\nMEDIA:/tmp/reply.ogg"
        )
        assert len(media) == 1
        assert media[0][0] == "/tmp/reply.ogg"
        assert media[0][1] is True  # is_voice flag

    @pytest.mark.asyncio
    async def test_post_stream_media_tag_image_sends_signal_data_uri(
        self, monkeypatch, tmp_path
    ):
        """Streaming completion path must not pass MEDIA paths to remote signal-cli."""
        from gateway.platforms.base import MessageEvent
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource

        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([{"timestamp": 1234567890}])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        img_bytes = b"\x89PNG\r\n\x1a\n" + b"payload"
        img_path = tmp_path / "content1.png"
        img_path.write_bytes(img_bytes)

        runner = object.__new__(GatewayRunner)
        runner._thread_metadata_for_source = lambda source, reply_to_message_id=None: None
        runner._reply_anchor_for_event = lambda event: None
        event = MessageEvent(
            text="",
            source=SessionSource(
                platform=Platform.SIGNAL,
                chat_id="group:abc123==",
                chat_type="group",
                user_id="+155****9999",
            ),
        )

        await runner._deliver_media_from_response(
            f"Attached without viewing:\n\nMEDIA:{img_path}",
            event,
            adapter,
        )

        params = captured[0]["params"]
        assert params["groupId"] == "abc123=="
        attachments = params["attachments"]
        assert len(attachments) == 1
        assert attachments[0].startswith("data:image/png;filename=content1.png;base64,")
        assert str(img_path) not in attachments[0]
        encoded = attachments[0].split(",", 1)[1]
        assert base64.b64decode(encoded) == img_bytes

    def test_signal_has_all_media_methods(self, monkeypatch):
        """SignalAdapter must override all media send methods used by gateway."""
        adapter = _make_signal_adapter(monkeypatch)
        from gateway.platforms.base import BasePlatformAdapter

        # These methods must NOT be the base class defaults (which just send text)
        assert type(adapter).send_image_file is not BasePlatformAdapter.send_image_file
        assert type(adapter).send_voice is not BasePlatformAdapter.send_voice
        assert type(adapter).send_video is not BasePlatformAdapter.send_video
        assert type(adapter).send_document is not BasePlatformAdapter.send_document
        assert type(adapter).send_image is not BasePlatformAdapter.send_image


# ---------------------------------------------------------------------------
# Inbound attachment message type classification
# ---------------------------------------------------------------------------

def _make_dm_envelope(sender: str, attachments: list, text: str = "") -> dict:
    """Build a minimal signal-cli DM envelope with the given attachments."""
    return {
        "envelope": {
            "sourceNumber": sender,
            "sourceName": "Test User",
            "sourceUuid": "aaaaaaaa-0000-0000-0000-000000000001",
            "timestamp": 1700000000000,
            "dataMessage": {
                "timestamp": 1700000000000,
                "message": text,
                "expiresInSeconds": 0,
                "viewOnce": False,
                "attachments": attachments,
            },
        }
    }


class TestSignalInboundMessageTypeClassification:
    """_handle_envelope must set MessageType.DOCUMENT for application/* and text/* attachments.

    Before the fix, PDFs and other documents left msg_type as MessageType.TEXT,
    so run.py's document-context injection (which gates on MessageType.DOCUMENT)
    silently dropped the file and the agent never saw it.
    """

    async def _dispatch_single_attachment(self, monkeypatch, content_type: str,
                                          att_id: str, fetch_path: str, fetch_ext: str):
        """Helper: run _handle_envelope with one attachment and return the dispatched event."""
        envelope = _make_dm_envelope(
            sender="+15559876543",
            attachments=[{
                "contentType": content_type,
                "id": att_id,
                "size": 1024,
                "filename": None,
                "width": None,
                "height": None,
                "caption": None,
                "uploadTimestamp": 1700000000000,
            }],
        )
        adapter = _make_signal_adapter(monkeypatch)
        adapter._rpc, _ = _stub_rpc(None)
        dispatched = []

        async def _fake_handle_message(event):
            dispatched.append(event)

        adapter.handle_message = _fake_handle_message
        adapter._fetch_attachment = AsyncMock(return_value=(fetch_path, fetch_ext))
        await adapter._handle_envelope(envelope)
        assert dispatched, "_handle_envelope did not dispatch any event"
        return dispatched[0]

    @pytest.mark.asyncio
    async def test_pdf_attachment_sets_document_type(self, monkeypatch):
        """A PDF attachment (application/pdf) must produce MessageType.DOCUMENT, not TEXT."""
        from gateway.platforms.base import MessageType

        event = await self._dispatch_single_attachment(
            monkeypatch,
            content_type="application/pdf",
            att_id="6zLO3b-6Yf3zVWeLDctA.pdf",
            fetch_path="/tmp/report.pdf",
            fetch_ext=".pdf",
        )

        assert event.message_type == MessageType.DOCUMENT, (
            f"Expected DOCUMENT, got {event.message_type}. "
            "PDFs must be classified as DOCUMENT so run.py injects file context."
        )
        assert "/tmp/report.pdf" in event.media_urls

    @pytest.mark.asyncio
    async def test_text_plain_attachment_sets_document_type(self, monkeypatch):
        """A text/plain attachment must produce MessageType.DOCUMENT, not TEXT."""
        from gateway.platforms.base import MessageType

        event = await self._dispatch_single_attachment(
            monkeypatch,
            content_type="text/plain",
            att_id="notes.txt",
            fetch_path="/tmp/notes.txt",
            fetch_ext=".txt",
        )

        assert event.message_type == MessageType.DOCUMENT, (
            f"Expected DOCUMENT, got {event.message_type}. "
            "text/plain must be classified as DOCUMENT so run.py injects file context."
        )

    @pytest.mark.asyncio
    async def test_text_html_attachment_sets_document_type(self, monkeypatch):
        """A text/html attachment must produce MessageType.DOCUMENT (covers the text/* wildcard)."""
        from gateway.platforms.base import MessageType

        event = await self._dispatch_single_attachment(
            monkeypatch,
            content_type="text/html",
            att_id="page.html",
            fetch_path="/tmp/page.html",
            fetch_ext=".html",
        )

        assert event.message_type == MessageType.DOCUMENT, (
            f"Expected DOCUMENT, got {event.message_type}. "
            "text/html must be classified as DOCUMENT so run.py injects file context."
        )

    @pytest.mark.asyncio
    async def test_video_attachment_sets_video_type(self, monkeypatch):
        """A video/mp4 attachment must produce MessageType.VIDEO."""
        from gateway.platforms.base import MessageType

        event = await self._dispatch_single_attachment(
            monkeypatch,
            content_type="video/mp4",
            att_id="clip.mp4",
            fetch_path="/tmp/clip.mp4",
            fetch_ext=".mp4",
        )

        assert event.message_type == MessageType.VIDEO

    @pytest.mark.asyncio
    async def test_unknown_mime_attachment_falls_back_to_document(self, monkeypatch):
        """Unknown/exotic MIME types fall through to DOCUMENT (catch-all),
        matching the WhatsApp/Slack/BlueBubbles classification pattern."""
        from gateway.platforms.base import MessageType

        event = await self._dispatch_single_attachment(
            monkeypatch,
            content_type="chemical/x-pdb",
            att_id="molecule.pdb",
            fetch_path="/tmp/molecule.pdb",
            fetch_ext=".pdb",
        )

        assert event.message_type == MessageType.DOCUMENT


# ---------------------------------------------------------------------------
# send_document now routes through _send_attachment (#5105 bonus)
# ---------------------------------------------------------------------------

class TestSignalSendDocumentViaHelper:
    """Verify send_document gained size check and path-in-error via _send_attachment."""

    @pytest.mark.asyncio
    async def test_send_document_too_large(self, monkeypatch, tmp_path):
        """send_document should now reject files over 100MB (was previously missing)."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        doc_path = tmp_path / "huge.pdf"
        doc_path.write_bytes(b"x")

        def mock_stat(self, **kwargs):
            class FakeStat:
                st_size = 200 * 1024 * 1024
            return FakeStat()

        with patch.object(Path, "stat", mock_stat):
            result = await adapter.send_document(chat_id="+155****4567", file_path=str(doc_path))

        assert result.success is False
        assert "too large" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_document_error_includes_path(self, monkeypatch):
        """send_document error message should include the file path."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send_document(chat_id="+155****4567", file_path="/nonexistent.pdf")

        assert result.success is False
        assert "/nonexistent.pdf" in result.error


# ---------------------------------------------------------------------------
# Signal streaming edit capability / message_id behavior
# ---------------------------------------------------------------------------

class TestSignalStreamingCapabilities:
    """Signal must opt out of edit-based streaming behavior."""

    def test_signal_declares_no_message_editing(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)

        assert adapter.SUPPORTS_MESSAGE_EDITING is False


class TestSignalSendReturnsMessageId:
    """Signal send() should not pretend sent messages are editable."""

    @pytest.mark.asyncio
    async def test_send_returns_none_message_id_even_with_timestamp(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc({"timestamp": 1712345678000})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")

        assert result.success is True
        assert result.message_id is None

    @pytest.mark.asyncio
    async def test_send_returns_none_message_id_when_no_timestamp(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc({})  # No timestamp key
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")

        assert result.success is True
        assert result.message_id is None

    @pytest.mark.asyncio
    async def test_send_returns_none_message_id_for_non_dict(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc("ok")  # Non-dict result
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")

        assert result.success is True
        assert result.message_id is None


class TestSignalSendResultValidation:
    """Verify that send() validates recipient-level delivery results."""

    @pytest.mark.asyncio
    async def test_send_success_when_results_has_success(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc({
            "timestamp": 1712345678000,
            "results": [
                {
                    "recipientAddress": {"number": "+155****4567"},
                    "type": "SUCCESS"
                }
            ]
        })
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_failure_when_results_has_failure_type(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc({
            "timestamp": 1712345678000,
            "results": [
                {
                    "recipientAddress": {"number": "+155****4567"},
                    "type": "UNREGISTERED_FAILURE"
                }
            ]
        })
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")
        assert result.success is False
        assert result.error == "UNREGISTERED_FAILURE"
        assert result.error_kind == "not_found"
        assert result.retryable is False

    @pytest.mark.asyncio
    async def test_send_rate_limit_failure_uses_shared_retry_metadata(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc({
            "timestamp": 1712345678000,
            "results": [
                {
                    "recipientAddress": {"number": "+155****4567"},
                    "type": "RATE_LIMIT_FAILURE",
                    "retryAfterSeconds": 15,
                }
            ],
        })
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")

        assert result.success is False
        assert result.error == "RATE_LIMIT_FAILURE"
        assert result.error_kind == "rate_limited"
        assert result.retryable is True
        assert result.retry_after == 15.0

    @pytest.mark.asyncio
    async def test_group_partial_success_is_not_retried_or_format_fallback(
        self, monkeypatch, caplog
    ):
        """A full-group resend would duplicate the message for successful recipients."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, calls = _stub_rpc({
            "timestamp": 1712345678000,
            "results": [
                {
                    "recipientAddress": {"uuid": "recipient-a"},
                    "type": "SUCCESS",
                },
                {
                    "recipientAddress": {"uuid": "recipient-b"},
                    "type": "RATE_LIMIT_FAILURE",
                    "retryAfterSeconds": 15,
                },
            ],
        })
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter._send_with_retry(
                chat_id="group:test-group",
                content="**hello**",
                max_retries=2,
                base_delay=0,
            )

        assert result.success is True
        assert len(calls) == 1
        mock_sleep.assert_not_awaited()
        assert "partial" in caplog.text.lower()
        assert "recipient-a" not in caplog.text
        assert "recipient-b" not in caplog.text

    @pytest.mark.asyncio
    async def test_all_rate_limited_group_send_retries_without_formatting_prefix(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(monkeypatch)
        responses = iter([
            {
                "timestamp": 1712345678000,
                "results": [
                    {
                        "recipientAddress": {"uuid": "recipient-a"},
                        "type": "RATE_LIMIT_FAILURE",
                        "retryAfterSeconds": 15,
                    }
                ],
            },
            {
                "timestamp": 1712345679000,
                "results": [
                    {
                        "recipientAddress": {"uuid": "recipient-a"},
                        "type": "SUCCESS",
                    }
                ],
            },
        ])
        calls = []

        async def mock_rpc(method, params, rpc_id=None):
            calls.append(dict(params))
            return next(responses)

        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter._send_with_retry(
                chat_id="group:test-group",
                content="**hello**",
                max_retries=2,
                base_delay=0,
            )

        assert result.success is True
        assert len(calls) == 2
        assert mock_sleep.await_args.args[0] >= 15.0
        assert all("Response formatting failed" not in call["message"] for call in calls)

    @pytest.mark.asyncio
    async def test_send_failure_when_results_has_success_false(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc({
            "timestamp": 1712345678000,
            "results": [
                {
                    "recipientAddress": {"number": "+155****4567"},
                    "success": False,
                    "failure": "Some connection error"
                }
            ]
        })
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")
        assert result.success is False
        assert result.error == "Recipient delivery failed"
        assert result.error_kind == "unknown"

    @pytest.mark.asyncio
    async def test_partial_legacy_failure_log_does_not_expose_recipient(self, monkeypatch, caplog):
        adapter = _make_signal_adapter(monkeypatch)
        private_failure = "recipient +15551234567 uuid 11111111-2222-3333-4444-555555555555 failed"
        mock_rpc, _ = _stub_rpc({
            "timestamp": 1712345678000,
            "results": [
                {"recipientAddress": {"uuid": "recipient-a"}, "type": "SUCCESS"},
                {
                    "recipientAddress": {"uuid": "recipient-b"},
                    "success": False,
                    "failure": private_failure,
                },
            ],
        })
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        with caplog.at_level("WARNING"):
            result = await adapter.send(chat_id="group:test-group", content="hello")

        assert result.success is True
        combined = "\n".join(record.getMessage() for record in caplog.records)
        assert private_failure not in combined
        assert "+15551234567" not in combined
        assert "11111111-2222-3333-4444-555555555555" not in combined
        assert "LEGACY_FAILURE=1" in combined

    @pytest.mark.asyncio
    async def test_rpc_raises_rate_limit_on_results_failure(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {
                "timestamp": 1712345678000,
                "results": [
                    {
                        "recipientAddress": {"number": "+155****4567"},
                        "type": "RATE_LIMIT_FAILURE",
                        "retryAfterSeconds": 15
                    }
                ]
            },
            "id": "1"
        }
        mock_client.post = AsyncMock(return_value=mock_response)
        adapter.client = mock_client

        from gateway.platforms.signal_rate_limit import SignalRateLimitError
        with pytest.raises(SignalRateLimitError) as exc_info:
            await adapter._rpc("send", {"recipient": ["+155****4567"]}, raise_on_rate_limit=True)

        assert "Rate limit exceeded for recipient" in str(exc_info.value)
        assert exc_info.value.retry_after == 15


# ---------------------------------------------------------------------------
# stop_typing() delegates to _stop_typing_indicator (#4647)
# ---------------------------------------------------------------------------

class TestSignalStopTyping:
    """Signal must expose a public stop_typing() so base adapter's
    _keep_typing finally block can clean up platform-level typing tasks."""

    @pytest.mark.asyncio
    async def test_stop_typing_calls_private_method(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        await adapter.stop_typing("+155****4567")

        adapter._stop_typing_indicator.assert_awaited_once_with("+155****4567")


# ---------------------------------------------------------------------------
# Typing-indicator backoff on repeated failures (Signal RPC spam fix)
# ---------------------------------------------------------------------------

class TestSignalTypingBackoff:
    """When base.py's _keep_typing refresh loop calls send_typing every ~2s
    and the recipient is unreachable (NETWORK_FAILURE), the adapter must:

    - log WARNING only for the first failure (subsequent failures use DEBUG
      via log_failures=False on the _rpc call)
    - after 3 consecutive failures, skip the RPC entirely during an
      exponential cooldown window instead of hammering signal-cli every 2s
    - reset counters on a successful sendTyping
    - reset counters when _stop_typing_indicator() is called for the chat
    """

    @pytest.mark.asyncio
    async def test_first_failure_logs_at_warning_subsequent_at_debug(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(monkeypatch)
        calls = []

        async def _fake_rpc(method, params, rpc_id=None, *, log_failures=True):
            calls.append({"log_failures": log_failures})
            return None  # simulate NETWORK_FAILURE

        adapter._rpc = _fake_rpc

        await adapter.send_typing("+155****4567")
        await adapter.send_typing("+155****4567")

        assert len(calls) == 2
        assert calls[0]["log_failures"] is True   # first failure — warn
        assert calls[1]["log_failures"] is False  # subsequent — debug

    @pytest.mark.asyncio
    async def test_three_consecutive_failures_trigger_cooldown(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(monkeypatch)
        call_count = {"n": 0}

        async def _fake_rpc(method, params, rpc_id=None, *, log_failures=True):
            call_count["n"] += 1
            return None

        adapter._rpc = _fake_rpc

        # Three failures engage the cooldown.
        await adapter.send_typing("+155****4567")
        await adapter.send_typing("+155****4567")
        await adapter.send_typing("+155****4567")
        assert call_count["n"] == 3
        assert "+155****4567" in adapter._typing_skip_until

        # Fourth, fifth, ... calls during the cooldown window are short-
        # circuited — the RPC is not issued at all.
        await adapter.send_typing("+155****4567")
        await adapter.send_typing("+155****4567")
        assert call_count["n"] == 3

    @pytest.mark.asyncio
    async def test_cooldown_is_per_chat_not_global(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        call_log = []

        async def _fake_rpc(method, params, rpc_id=None, *, log_failures=True):
            call_log.append(params.get("recipient") or params.get("groupId"))
            return None

        adapter._rpc = _fake_rpc

        # Drive chat A into cooldown.
        for _ in range(3):
            await adapter.send_typing("+155****4567")
        assert "+155****4567" in adapter._typing_skip_until

        # Chat B is unaffected — still makes RPCs.
        await adapter.send_typing("+155****9999")
        await adapter.send_typing("+155****9999")
        assert "+155****9999" not in adapter._typing_skip_until
        # Chat A cooldown untouched
        assert "+155****4567" in adapter._typing_skip_until

    @pytest.mark.asyncio
    async def test_success_resets_failure_counter_and_cooldown(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(monkeypatch)
        result_queue = [None, None, {"timestamp": 12345}]
        call_log = []

        async def _fake_rpc(method, params, rpc_id=None, *, log_failures=True):
            call_log.append(log_failures)
            return result_queue.pop(0)

        adapter._rpc = _fake_rpc

        await adapter.send_typing("+155****4567")   # fail 1 — warn
        await adapter.send_typing("+155****4567")   # fail 2 — debug
        await adapter.send_typing("+155****4567")   # success — reset

        assert adapter._typing_failures.get("+155****4567", 0) == 0
        assert "+155****4567" not in adapter._typing_skip_until

        # Next failure after recovery logs at WARNING again (fresh counter).
        async def _fail(method, params, rpc_id=None, *, log_failures=True):
            call_log.append(log_failures)
            return None

        adapter._rpc = _fail
        await adapter.send_typing("+155****4567")
        assert call_log[-1] is True   # first failure in a fresh cycle

    @pytest.mark.asyncio
    async def test_stop_typing_indicator_clears_backoff_state(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(monkeypatch)

        async def _fail(method, params, rpc_id=None, *, log_failures=True):
            return None

        adapter._rpc = _fail

        for _ in range(3):
            await adapter.send_typing("+155****4567")
        assert adapter._typing_failures.get("+155****4567") == 3
        assert "+155****4567" in adapter._typing_skip_until

        await adapter._stop_typing_indicator("+155****4567")

        assert "+155****4567" not in adapter._typing_failures
        assert "+155****4567" not in adapter._typing_skip_until


# ---------------------------------------------------------------------------
# _stop_typing_indicator sends explicit sendTyping(stop=True) RPC
# ---------------------------------------------------------------------------

class TestSignalStopTypingExplicitRPC:
    """Cancelling the typing indicator must issue an explicit
    sendTyping(stop=True) RPC so the recipient's device drops the indicator
    immediately, instead of waiting for Signal's built-in ~5s timeout.

    The stop RPC is best-effort: any failure must not prevent the per-chat
    backoff state from being cleared.
    """

    @pytest.mark.asyncio
    async def test_stop_typing_indicator_sends_stop_rpc_for_dm(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._resolve_recipient = AsyncMock(return_value="uuid-recipient")
        captured = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            captured.append({"method": method, "params": dict(params), "rpc_id": rpc_id})
            return {}

        adapter._rpc = mock_rpc

        await adapter._stop_typing_indicator("+15555550000")

        assert len(captured) == 1
        assert captured[0]["method"] == "sendTyping"
        assert captured[0]["params"]["stop"] is True
        assert captured[0]["params"]["recipient"] == ["uuid-recipient"]
        assert captured[0]["rpc_id"] == "typing-stop"
        adapter._resolve_recipient.assert_awaited_once_with("+15555550000")

    @pytest.mark.asyncio
    async def test_stop_typing_indicator_sends_stop_rpc_for_group(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        captured = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            captured.append({"method": method, "params": dict(params), "rpc_id": rpc_id})
            return {}

        adapter._rpc = mock_rpc

        await adapter._stop_typing_indicator("group:group123")

        assert len(captured) == 1
        assert captured[0]["method"] == "sendTyping"
        assert captured[0]["params"]["stop"] is True
        assert captured[0]["params"]["groupId"] == "group123"
        assert "recipient" not in captured[0]["params"]

    @pytest.mark.asyncio
    async def test_stop_typing_indicator_best_effort_on_rpc_failure(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._resolve_recipient = AsyncMock(return_value="uuid-recipient")

        # Drive the chat into backoff so we can confirm cleanup still happens
        # even when the stop RPC itself fails.
        async def _noop(method, params, rpc_id=None, **kwargs):
            return None

        adapter._rpc = _noop
        for _ in range(3):
            await adapter.send_typing("+155****0000")

        assert adapter._typing_failures.get("+155****0000") == 3
        assert "+155****0000" in adapter._typing_skip_until

        # Now make the stop RPC raise — backoff state must still be cleared.
        async def failing_rpc(method, params, rpc_id=None, **kwargs):
            raise RuntimeError("signal-cli unreachable")

        adapter._rpc = failing_rpc

        await adapter._stop_typing_indicator("+155****0000")

        assert "+155****0000" not in adapter._typing_failures
        assert "+155****0000" not in adapter._typing_skip_until

    @pytest.mark.asyncio
    async def test_stop_typing_indicator_best_effort_on_recipient_failure(self, monkeypatch):
        # When _resolve_recipient() raises, the per-chat backoff state must
        # still be cleared — otherwise a transient resolution failure would
        # silently keep the chat in cooldown forever.
        adapter = _make_signal_adapter(monkeypatch)
        adapter._resolve_recipient = AsyncMock(
            side_effect=RuntimeError("recipient resolution failed")
        )

        captured = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            captured.append({"method": method, "params": dict(params), "rpc_id": rpc_id})
            return {}

        adapter._rpc = mock_rpc

        adapter._typing_failures["+155****0000"] = 2
        adapter._typing_skip_until["+155****0000"] = 9999999999.0

        await adapter._stop_typing_indicator("+155****0000")

        # No RPC must be issued when recipient resolution itself fails.
        assert captured == []
        assert "+155****0000" not in adapter._typing_failures
        assert "+155****0000" not in adapter._typing_skip_until


# ---------------------------------------------------------------------------
# Reply quote extraction
# ---------------------------------------------------------------------------

class TestSignalQuoteExtraction:
    """Verify Signal reply quote fields are propagated to MessageEvent."""

    @pytest.mark.asyncio
    async def test_handle_envelope_sets_reply_context_from_quote(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+15550001111",
                "sourceUuid": "uuid-sender",
                "sourceName": "Tester",
                "timestamp": 1000000000,
                "dataMessage": {
                    "message": "yes I agree",
                    "quote": {
                        "id": 99,
                        "text": "want to grab lunch?",
                        "author": "other-author",
                    },
                },
            }
        })

        event = captured["event"]
        assert event.text == "yes I agree"
        assert event.reply_to_message_id == "99"
        assert event.reply_to_text == "want to grab lunch?"
        assert event.reply_to_author_id == "other-author"
        assert event.reply_to_is_own_message is False

    @pytest.mark.asyncio
    async def test_handle_envelope_marks_quote_to_own_sent_timestamp(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._remember_sent_message_timestamp(424242)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****1111",
                "sourceUuid": "uuid-sender",
                "sourceName": "Tester",
                "timestamp": 1000000000,
                "dataMessage": {
                    "message": "this specific one",
                    "quote": {
                        "id": 424242,
                        "text": "assistant answer",
                        "author": "other-author",
                    },
                },
            }
        })

        event = captured["event"]
        assert event.reply_to_message_id == "424242"
        assert event.reply_to_text == "assistant answer"
        assert event.reply_to_author_id == "other-author"
        assert event.reply_to_is_own_message is True

    @pytest.mark.asyncio
    async def test_handle_envelope_marks_quote_to_own_account_author(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, account="bot-author")
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****1111",
                "sourceUuid": "uuid-sender",
                "sourceName": "Tester",
                "timestamp": 1000000000,
                "dataMessage": {
                    "message": "reply by author",
                    "quote": {
                        "id": 777,
                        "text": "assistant answer",
                        "author": "bot-author",
                    },
                },
            }
        })

        event = captured["event"]
        assert event.reply_to_message_id == "777"
        assert event.reply_to_is_own_message is True

    @pytest.mark.asyncio
    async def test_track_sent_timestamp_keeps_reply_detection_cache_after_echo_discard(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._track_sent_timestamp({"timestamp": 111222333})
        # Echo suppression consumes the entry from the recent-sent ring; the
        # separate reply-detection cache must still retain it.
        adapter._consume_sent_timestamp(111222333)

        assert "111222333" in adapter._sent_message_timestamps
        assert adapter._quote_references_own_message("111222333", None) is True

    def test_sent_message_timestamps_evicts_oldest_first(self, monkeypatch):
        """Over the cap, the OLDEST quote-cache timestamp is dropped (FIFO),
        not an arbitrary one — so a recent reply-to-own-message is still
        detected after a burst of sends."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._max_sent_message_timestamps = 3
        for ts in (1, 2, 3):
            adapter._remember_sent_message_timestamp(ts)
        # Adding a 4th evicts the oldest (1), keeps the rest in order.
        adapter._remember_sent_message_timestamp(4)
        assert list(adapter._sent_message_timestamps.keys()) == ["2", "3", "4"]
        assert "1" not in adapter._sent_message_timestamps
        # Re-seeing an existing ts promotes it so it survives the next eviction.
        adapter._remember_sent_message_timestamp(2)  # 2 -> most recent
        adapter._remember_sent_message_timestamp(5)  # evicts oldest (now 3)
        assert list(adapter._sent_message_timestamps.keys()) == ["4", "2", "5"]
        assert "3" not in adapter._sent_message_timestamps

    @pytest.mark.asyncio
    async def test_handle_envelope_without_quote_leaves_reply_fields_none(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+15550001111",
                "sourceUuid": "uuid-sender",
                "sourceName": "Tester",
                "timestamp": 1000000000,
                "dataMessage": {
                    "message": "plain message",
                },
            }
        })

        event = captured["event"]
        assert event.text == "plain message"
        assert event.reply_to_message_id is None
        assert event.reply_to_text is None

    @pytest.mark.asyncio
    async def test_handle_envelope_quote_without_text_sets_only_reply_id(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+15550001111",
                "sourceUuid": "uuid-sender",
                "sourceName": "Tester",
                "timestamp": 1000000000,
                "dataMessage": {
                    "message": "reply without quote text",
                    "quote": {
                        "id": 123,
                        "author": "+15550002222",
                    },
                },
            }
        })

        event = captured["event"]
        assert event.reply_to_message_id == "123"
        assert event.reply_to_text is None

# ---------------------------------------------------------------------------
# _rpc rate-limit detection
# ---------------------------------------------------------------------------

class _FakeHttpResponse:
    """Minimal stand-in for httpx.Response — only what _rpc touches."""

    def __init__(self, json_data):
        self._json = json_data

    def raise_for_status(self):
        return None

    def json(self):
        return self._json


def _install_fake_client(adapter, json_data):
    """Replace adapter.client.post with an async fn returning json_data."""
    from types import SimpleNamespace

    async def _post(url, json=None, timeout=None):
        return _FakeHttpResponse(json_data)

    adapter.client = SimpleNamespace(post=_post)


class TestSignalRpcRateLimit:
    """_rpc opt-in 429 detection and SignalRateLimitError propagation."""

    @pytest.mark.asyncio
    async def test_raises_on_429_when_opted_in(self, monkeypatch):
        from gateway.platforms.signal import SignalRateLimitError

        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {"message": "Failed to send: [429] Rate Limited"},
        })

        with pytest.raises(SignalRateLimitError):
            await adapter._rpc("send", {}, raise_on_rate_limit=True)

    @pytest.mark.asyncio
    async def test_raises_on_rate_limit_exception_substring(self, monkeypatch):
        """Some signal-cli builds emit 'RateLimitException' without a literal [429]."""
        from gateway.platforms.signal import SignalRateLimitError

        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {"message": "RateLimitException occurred"},
        })

        with pytest.raises(SignalRateLimitError):
            await adapter._rpc("send", {}, raise_on_rate_limit=True)

    @pytest.mark.asyncio
    async def test_default_text_send_returns_unstructured_rate_limit_metadata(
        self, monkeypatch
    ):
        """Text sends route even non-recipient 429 envelopes through shared retry."""
        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {"message": "[429] Rate Limited. Retry after 23 seconds"},
        })
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")

        assert result.success is False
        assert result.error == "Signal rate limited"
        assert result.error_kind == "rate_limited"
        assert result.retryable is True
        assert result.retry_after == 23.0

    @pytest.mark.asyncio
    async def test_non_rate_limit_error_does_not_raise_or_log_details_when_opted_in(
        self, monkeypatch, caplog
    ):
        """Non-429 send errors stay structured and daemon text remains private."""
        adapter = _make_signal_adapter(monkeypatch)
        private_error = (
            "Recipient +155****4567 uuid "
            "11111111-2222-3333-4444-555555555555 is untrusted"
        )
        _install_fake_client(adapter, {
            "error": {"message": private_error},
        })

        with caplog.at_level("WARNING"):
            result = await adapter._rpc("send", {}, raise_on_rate_limit=True)

        assert result["_hermes_rpc_error_envelope"] is True
        assert result["_hermes_rpc_error_kind"] == "unknown"
        assert result["_hermes_rpc_error_message"] == "Signal RPC send failed"
        rpc_logs = [
            record.getMessage()
            for record in caplog.records
            if record.name == "gateway.platforms.signal"
            and "Signal RPC send error" in record.getMessage()
        ]
        assert rpc_logs == ["Signal RPC send error (unknown)"]

    @pytest.mark.asyncio
    async def test_partial_success_does_not_raise_recipient_rate_limit(self, monkeypatch):
        """A partial group result must reach validation instead of full-group retry."""
        adapter = _make_signal_adapter(monkeypatch)
        rpc_result = {
            "timestamp": 1712345678000,
            "results": [
                {"recipientAddress": {"uuid": "recipient-a"}, "type": "SUCCESS"},
                {
                    "recipientAddress": {"uuid": "recipient-b"},
                    "type": "RATE_LIMIT_FAILURE",
                    "retryAfterSeconds": 15,
                },
            ],
        }
        _install_fake_client(adapter, {
            "error": {
                "message": "Failed to send message due to rate limiting",
                "data": {"response": rpc_result},
            }
        })

        result = await adapter._rpc("send", {}, raise_on_rate_limit=True)

        assert result["timestamp"] == rpc_result["timestamp"]
        assert result["results"] == rpc_result["results"]
        assert result["_hermes_rpc_error_envelope"] is True
        assert result["_hermes_rpc_error_kind"] == "rate_limited"

    @pytest.mark.asyncio
    async def test_default_returns_structured_recipient_failures(self, monkeypatch):
        """Text send callers need error-envelope results for shared classification."""
        adapter = _make_signal_adapter(monkeypatch)
        rpc_result = {
            "timestamp": 1712345678000,
            "results": [
                {
                    "recipientAddress": {"uuid": "recipient-a"},
                    "type": "RATE_LIMIT_FAILURE",
                    "retryAfterSeconds": 15,
                },
            ],
        }
        _install_fake_client(adapter, {
            "error": {
                "message": "Failed to send message due to rate limiting",
                "data": {"response": rpc_result},
            }
        })

        result = await adapter._rpc("send", {})

        assert result["timestamp"] == rpc_result["timestamp"]
        assert result["results"] == rpc_result["results"]
        assert result["_hermes_rpc_error_kind"] == "rate_limited"

    @pytest.mark.asyncio
    async def test_error_envelope_with_empty_results_is_not_success(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {
                "message": "Signal send failed",
                "data": {"response": {"timestamp": 0, "results": []}},
            }
        })
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")

        assert result.success is False
        assert result.error == "Signal RPC send failed"
        assert result.error_kind == "unknown"

    @pytest.mark.asyncio
    async def test_text_send_uses_retry_after_from_error_message(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {
                "message": "Failed to send: [429] Rate Limited. Retry after 37 seconds",
                "data": {
                    "response": {
                        "timestamp": 0,
                        "results": [{"type": "RATE_LIMIT_FAILURE"}],
                    }
                },
            }
        })
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")

        assert result.success is False
        assert result.error_kind == "rate_limited"
        assert result.retryable is True
        assert result.retry_after == 37.0

    @pytest.mark.asyncio
    async def test_connect_error_retries_but_read_timeout_does_not(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()
        attempts = {"connect": 0}

        async def _connect_then_succeed(url, json=None, timeout=None):
            attempts["connect"] += 1
            if attempts["connect"] == 1:
                raise httpx.ConnectError(
                    "connection refused",
                    request=httpx.Request("POST", url),
                )
            return _FakeHttpResponse({
                "result": {
                    "timestamp": 1712345678000,
                    "results": [{"type": "SUCCESS"}],
                }
            })

        adapter.client = SimpleNamespace(post=_connect_then_succeed)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            connect_result = await adapter._send_with_retry(
                chat_id="+155****4567",
                content="hello",
                max_retries=2,
                base_delay=0,
            )

        assert connect_result.success is True
        assert attempts["connect"] == 2

        read_attempts = {"count": 0}

        async def _read_timeout(url, json=None, timeout=None):
            read_attempts["count"] += 1
            raise httpx.ReadTimeout(
                "read timed out",
                request=httpx.Request("POST", url),
            )

        adapter.client = SimpleNamespace(post=_read_timeout)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            timeout_result = await adapter._send_with_retry(
                chat_id="+155****4567",
                content="hello",
                max_retries=2,
                base_delay=0,
            )

        assert timeout_result.success is False
        assert timeout_result.error_kind == "unknown"
        assert read_attempts["count"] == 1

    @pytest.mark.asyncio
    async def test_raises_with_retry_after_from_v0_14_3_payload(self, monkeypatch):
        """signal-cli ≥ v0.14.3 surfaces server Retry-After under
        ``error.data.response.results[*].retryAfterSeconds`` — _rpc
        carries that value through SignalRateLimitError.retry_after."""
        from gateway.platforms.signal_rate_limit import (
            SignalRateLimitError, SIGNAL_RPC_ERROR_RATELIMIT,
        )

        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {
                "code": SIGNAL_RPC_ERROR_RATELIMIT,
                "message": "Failed to send message due to rate limiting",
                "data": {
                    "response": {
                        "timestamp": 0,
                        "results": [
                            {"type": "RATE_LIMIT_FAILURE", "retryAfterSeconds": 90},
                        ],
                    }
                },
            },
        })

        with pytest.raises(SignalRateLimitError) as exc_info:
            await adapter._rpc("send", {}, raise_on_rate_limit=True)

        assert exc_info.value.retry_after == 90.0

    @pytest.mark.asyncio
    async def test_raises_with_retry_after_none_for_old_signal_cli(self, monkeypatch):
        """Older signal-cli builds emit only the substring; retry_after=None."""
        from gateway.platforms.signal import SignalRateLimitError

        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {"message": "Failed: [429] Rate Limited"},
        })

        with pytest.raises(SignalRateLimitError) as exc_info:
            await adapter._rpc("send", {}, raise_on_rate_limit=True)

        assert exc_info.value.retry_after is None

    @pytest.mark.asyncio
    async def test_raises_on_retry_later_inside_attachment_invalid(self, monkeypatch):
        """Production case: 429 during attachment upload surfaces as
        AttachmentInvalidException → UnexpectedErrorException (code
        -32603), with the libsignal-net 'Retry after N seconds'
        message embedded. _rpc must still detect this as rate-limit
        AND parse the seconds out of the message."""
        from gateway.platforms.signal import SignalRateLimitError

        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {
                "code": -32603,
                "message": (
                    "Failed to send message: /home/max/sync/Memes/fengshui.jpeg: "
                    "org.signal.libsignal.net.RetryLaterException: Retry after 4 seconds "
                    "(AttachmentInvalidException) (UnexpectedErrorException)"
                ),
                "data": None,
            },
        })

        with pytest.raises(SignalRateLimitError) as exc_info:
            await adapter._rpc("send", {}, raise_on_rate_limit=True)

        assert exc_info.value.retry_after == 4.0


# ---------------------------------------------------------------------------
# send_multiple_images — chunking, pacing, rate-limit retry
# ---------------------------------------------------------------------------


def _make_image_files(tmp_path, count, prefix="img"):
    """Materialize `count` tiny PNG files and return file:// URIs for them."""
    uris = []
    for i in range(count):
        p = tmp_path / f"{prefix}_{i}.png"
        p.write_bytes(b"\x89PNG" + b"\x00" * 32)
        uris.append((f"file://{p}", ""))
    return uris


def _stub_rpc_responses(responses):
    """Build an _rpc replacement that pops a response per call.

    Each entry in `responses` is either:
      * a return value (dict / None) → returned to the caller, or
      * an Exception subclass instance → raised.
    Captures (params, kwargs) per call for inspection.
    """
    captured = []
    queue = list(responses)

    async def mock_rpc(method, params, rpc_id=None, **kwargs):
        captured.append({"method": method, "params": dict(params), "kwargs": kwargs})
        await asyncio.sleep(0)
        if not queue:
            raise AssertionError("Unexpected extra _rpc call")
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return mock_rpc, captured


def _patch_scheduler_sleep(monkeypatch, capture: list):
    """Capture sleeps inside the scheduler so tests don't actually wait.
    Zero-second sleeps (e.g. event-loop yields from mock RPCs) are
    delegated to the real asyncio.sleep so they don't pollute the
    capture list."""
    _real_sleep = asyncio.sleep
    offset = [0.0]

    async def fake_sleep(seconds):
        if seconds > 0:
            capture.append(seconds)
            offset[0] += seconds
        else:
            await _real_sleep(0)

    monkeypatch.setattr(
        "gateway.platforms.signal_rate_limit.asyncio.sleep", fake_sleep
    )
    monkeypatch.setattr(
        "gateway.platforms.signal_rate_limit.time.monotonic", lambda: offset[0]
    )


class TestSignalSendMultipleImages:
    @pytest.mark.asyncio
    async def test_empty_list_is_noop(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        await adapter.send_multiple_images(chat_id="+155****4567", images=[])

        assert captured == []
        adapter._stop_typing_indicator.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_all_bad_files_no_rpc(self, monkeypatch, tmp_path):
        """If every image is missing/invalid, no RPC fires."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        await adapter.send_multiple_images(
            chat_id="+155****4567",
            images=[(f"file://{tmp_path}/missing_a.png", ""),
                    (f"file://{tmp_path}/missing_b.png", "")],
        )

        assert captured == []

    @pytest.mark.asyncio
    async def test_single_batch_under_limit(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([{"timestamp": 1}])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        images = _make_image_files(tmp_path, 5)
        await adapter.send_multiple_images(chat_id="+155****4567", images=images)

        assert len(captured) == 1
        params = captured[0]["params"]
        assert params["recipient"] == ["+155****4567"]
        assert params["message"] == ""
        assert len(params["attachments"]) == 5
        # raise_on_rate_limit must be opted into so the retry loop sees 429s
        assert captured[0]["kwargs"].get("raise_on_rate_limit") is True

    @pytest.mark.asyncio
    async def test_single_batch_sends_file_urls_as_data_uris(self, monkeypatch, tmp_path):
        """Signal HTTP JSON-RPC may run on another host; never send local paths."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([{"timestamp": 1}])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        img_bytes = b"\x89PNG\r\n\x1a\n" + b"payload"
        img_path = tmp_path / "content1.png"
        img_path.write_bytes(img_bytes)

        await adapter.send_multiple_images(
            chat_id="group:abc123==",
            images=[(f"file://{img_path}", "")],
        )

        attachments = captured[0]["params"]["attachments"]
        assert len(attachments) == 1
        assert attachments[0].startswith("data:image/png;filename=content1.png;base64,")
        assert str(img_path) not in attachments[0]
        encoded = attachments[0].split(",", 1)[1]
        assert base64.b64decode(encoded) == img_bytes

    @pytest.mark.asyncio
    async def test_skips_bad_images_in_mixed_batch(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([{"timestamp": 1}])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        good = _make_image_files(tmp_path, 2, prefix="ok")
        bad = [(f"file://{tmp_path}/missing.png", "")]
        await adapter.send_multiple_images(
            chat_id="+155****4567", images=good[:1] + bad + good[1:]
        )

        assert len(captured) == 1
        assert len(captured[0]["params"]["attachments"]) == 2

    @pytest.mark.asyncio
    async def test_batch_partial_recipient_success_does_not_retry_group(
        self, monkeypatch, tmp_path
    ):
        from gateway.platforms.signal_rate_limit import get_scheduler

        adapter = _make_signal_adapter(monkeypatch)
        feedback = MagicMock()
        monkeypatch.setattr(get_scheduler(), "feedback", feedback)
        mock_rpc, captured = _stub_rpc_responses([{
            "timestamp": 1712345678000,
            "results": [
                {"type": "SUCCESS"},
                {"type": "RATE_LIMIT_FAILURE", "retryAfterSeconds": 30},
            ],
        }])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        await adapter.send_multiple_images(
            chat_id="group:abc123==",
            images=_make_image_files(tmp_path, 1),
        )

        assert len(captured) == 1
        feedback.assert_called_once_with(30.0, 1)

    @pytest.mark.asyncio
    async def test_batch_connect_error_retries_when_delivery_not_attempted(
        self, monkeypatch, tmp_path
    ):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()
        attempts = {"count": 0}

        async def _connect_then_succeed(url, json=None, timeout=None):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise httpx.ConnectError(
                    "connection refused",
                    request=httpx.Request("POST", url),
                )
            return _FakeHttpResponse({
                "result": {
                    "timestamp": 1712345678000,
                    "results": [{"type": "SUCCESS"}],
                }
            })

        adapter.client = SimpleNamespace(post=_connect_then_succeed)
        with patch("gateway.platforms.signal.asyncio.sleep", new_callable=AsyncMock):
            await adapter.send_multiple_images(
                chat_id="+155****4567",
                images=_make_image_files(tmp_path, 1),
            )

        assert attempts["count"] == 2

    @pytest.mark.asyncio
    async def test_batch_read_timeout_does_not_retry_ambiguous_delivery(
        self, monkeypatch, tmp_path
    ):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()
        attempts = {"count": 0}

        async def _read_timeout(url, json=None, timeout=None):
            attempts["count"] += 1
            raise httpx.ReadTimeout(
                "read timed out",
                request=httpx.Request("POST", url),
            )

        adapter.client = SimpleNamespace(post=_read_timeout)
        with patch("gateway.platforms.signal.asyncio.sleep", new_callable=AsyncMock):
            await adapter.send_multiple_images(
                chat_id="+155****4567",
                images=_make_image_files(tmp_path, 1),
            )

        assert attempts["count"] == 1

    @pytest.mark.asyncio
    async def test_batch_missing_response_does_not_retry_ambiguous_delivery(
        self, monkeypatch, tmp_path
    ):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([None])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        await adapter.send_multiple_images(
            chat_id="+155****4567",
            images=_make_image_files(tmp_path, 1),
        )

        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_429_calibrates_scheduler_then_retries(self, monkeypatch, tmp_path):
        """Server says retry_after=27 per token. After feedback, the
        scheduler's refill_rate becomes 1/27. Re-acquiring n=3 tokens
        therefore waits 3 × 27 = 81s — pulled from the server's
        authoritative rate, not a `× 32` defensive multiplier."""
        from gateway.platforms.signal import SignalRateLimitError

        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([
            SignalRateLimitError("Failed: rate limit", retry_after=27.0),
            {"timestamp": 99},
        ])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        sleep_calls: list = []
        _patch_scheduler_sleep(monkeypatch, sleep_calls)

        images = _make_image_files(tmp_path, 3)
        await adapter.send_multiple_images(chat_id="+155****4567", images=images)

        assert len(captured) == 2  # initial 429 + retry success
        assert sleep_calls == [pytest.approx(3 * 27.0, abs=1.0)]

    @pytest.mark.asyncio
    async def test_429_without_retry_after_uses_default_rate(
        self, monkeypatch, tmp_path
    ):
        """signal-cli < v0.14.3 doesn't surface Retry-After. The
        scheduler keeps its default refill rate (1 token / 4s), so a
        retry of n=3 waits 12s."""
        from gateway.platforms.signal_rate_limit import (
            SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER,
            SignalRateLimitError,
        )

        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([
            SignalRateLimitError("[429] Rate Limited", retry_after=None),
            {"timestamp": 99},
        ])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        sleep_calls: list = []
        _patch_scheduler_sleep(monkeypatch, sleep_calls)

        await adapter.send_multiple_images(
            chat_id="+155****4567",
            images=_make_image_files(tmp_path, 3),
        )

        assert len(captured) == 2
        assert sleep_calls == [
            pytest.approx(3 * SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER, abs=1.0)
        ]

    @pytest.mark.asyncio
    async def test_rate_limit_exhaust_continues_to_next_batch(
        self, monkeypatch, tmp_path
    ):
        """Both attempts on batch 0 fail; batch 1 still gets a chance.
        The scheduler's natural pacing on the next acquire stands in for
        the old explicit cooldown."""
        from gateway.platforms.signal import SignalRateLimitError

        adapter = _make_signal_adapter(monkeypatch)
        responses = [
            SignalRateLimitError("[429]", retry_after=4.0),
            SignalRateLimitError("[429]", retry_after=4.0),
            {"timestamp": 7},
        ]
        mock_rpc, captured = _stub_rpc_responses(responses)
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        sleep_calls: list = []
        _patch_scheduler_sleep(monkeypatch, sleep_calls)

        images = _make_image_files(tmp_path, 33)  # forces 2 batches
        await adapter.send_multiple_images(chat_id="+155****4567", images=images)

        # 2 attempts on batch 0 + 1 on batch 1
        assert len(captured) == 3

    @pytest.mark.asyncio
    async def test_full_batch_emits_pacing_notice_for_followup(
        self, monkeypatch, tmp_path
    ):
        """Two full batches of 32. Batch 1 needs 14 more tokens than the
        18 remaining after batch 0, so the scheduler sleeps 56s —
        crossing the 10s user-facing pacing-notice threshold."""
        from gateway.platforms.signal import SIGNAL_MAX_ATTACHMENTS_PER_MSG
        from gateway.platforms.signal_rate_limit import (
            SIGNAL_RATE_LIMIT_BUCKET_CAPACITY,
            SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER
        )

        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([
            {"timestamp": 1}, {"timestamp": 2},
        ])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()
        adapter._notify_batch_pacing = AsyncMock()

        sleep_calls: list = []
        _patch_scheduler_sleep(monkeypatch, sleep_calls)

        images = _make_image_files(tmp_path, 64)
        await adapter.send_multiple_images(chat_id="+155****4567", images=images)

        assert len(captured) == 2
        assert len(captured[0]["params"]["attachments"]) == SIGNAL_MAX_ATTACHMENTS_PER_MSG
        assert len(captured[1]["params"]["attachments"]) == SIGNAL_MAX_ATTACHMENTS_PER_MSG
        assert len(sleep_calls) == 1
        # Batch 1 deficit: 32 - (50 - 32) = 14 tokens × 4s = 56s
        expected_wait = (
            SIGNAL_MAX_ATTACHMENTS_PER_MSG
            - (SIGNAL_RATE_LIMIT_BUCKET_CAPACITY - SIGNAL_MAX_ATTACHMENTS_PER_MSG)
        ) * SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER
        assert sleep_calls[0] == pytest.approx(expected_wait, abs=1.0)
        adapter._notify_batch_pacing.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_short_followup_wait_skips_pacing_notice(
        self, monkeypatch, tmp_path
    ):
        """Batch 1 only needs 1 token but 18 remain after batch 0
        (50 capacity − 32 batch 0). No wait, no pacing notice."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([
            {"timestamp": 1}, {"timestamp": 2},
        ])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()
        adapter._notify_batch_pacing = AsyncMock()

        sleep_calls: list = []
        _patch_scheduler_sleep(monkeypatch, sleep_calls)

        images = _make_image_files(tmp_path, 33)
        await adapter.send_multiple_images(chat_id="+155****4567", images=images)

        assert len(captured) == 2
        assert len(sleep_calls) == 0
        adapter._notify_batch_pacing.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_single_batch_send_does_not_pace(self, monkeypatch, tmp_path):
        """A single-batch send (≤32 attachments) leaves the scheduler
        with tokens to spare — no follow-up acquire, no sleep."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([{"timestamp": 1}])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        sleep_calls: list = []
        _patch_scheduler_sleep(monkeypatch, sleep_calls)

        images = _make_image_files(tmp_path, 10)
        await adapter.send_multiple_images(chat_id="+155****4567", images=images)

        assert len(captured) == 1
        assert sleep_calls == []


class TestSignalRateLimitDetection:
    """Coverage for the typed-code + substring detection helpers."""

    def test_detect_typed_code(self):
        from gateway.platforms.signal_rate_limit import (
            _is_signal_rate_limit_error,
            SIGNAL_RPC_ERROR_RATELIMIT,
        )
        err = {"code": SIGNAL_RPC_ERROR_RATELIMIT, "message": "any text"}
        assert _is_signal_rate_limit_error(err) is True

    def test_detect_substring_fallback(self):
        from gateway.platforms.signal import _is_signal_rate_limit_error
        err = {"code": -32603, "message": "Failed: [429] Rate Limited (RateLimitException) (UnexpectedErrorException)"}
        assert _is_signal_rate_limit_error(err) is True

    def test_detect_non_rate_limit(self):
        from gateway.platforms.signal import _is_signal_rate_limit_error
        err = {"code": -32603, "message": "UntrustedIdentityException"}
        assert _is_signal_rate_limit_error(err) is False

    def test_extract_retry_after_from_results(self):
        from gateway.platforms.signal import _extract_retry_after_seconds
        err = {
            "code": -5,
            "message": "Failed to send message due to rate limiting",
            "data": {
                "response": {
                    "timestamp": 0,
                    "results": [
                        {"type": "RATE_LIMIT_FAILURE", "retryAfterSeconds": 30},
                        {"type": "RATE_LIMIT_FAILURE", "retryAfterSeconds": 45},
                    ],
                }
            },
        }
        assert _extract_retry_after_seconds(err) == 45.0

    def test_extract_retry_after_missing(self):
        """Old signal-cli builds don't expose retryAfterSeconds — return None."""
        from gateway.platforms.signal import _extract_retry_after_seconds
        err = {"code": -32603, "message": "[429] Rate Limited"}
        assert _extract_retry_after_seconds(err) is None

    def test_detect_retry_later_exception_substring(self):
        """libsignal-net's RetryLaterException leaks through as
        AttachmentInvalidException → UnexpectedErrorException when the
        rate-limit fires inside attachment upload. Detect it by substring."""
        from gateway.platforms.signal import _is_signal_rate_limit_error
        err = {
            "code": -32603,
            "message": (
                "Failed to send message: /home/max/sync/Memes/fengshui.jpeg: "
                "org.signal.libsignal.net.RetryLaterException: Retry after 4 seconds "
                "(AttachmentInvalidException) (UnexpectedErrorException)"
            ),
        }
        assert _is_signal_rate_limit_error(err) is True

    def test_extract_retry_after_parses_message_string(self):
        """When the structured field is missing, parse the seconds out
        of the human 'Retry after N seconds' substring."""
        from gateway.platforms.signal import _extract_retry_after_seconds
        err = {
            "code": -32603,
            "message": (
                "Failed to send message: /home/max/sync/Memes/fengshui.jpeg: "
                "org.signal.libsignal.net.RetryLaterException: Retry after 4 seconds "
                "(AttachmentInvalidException) (UnexpectedErrorException)"
            ),
        }
        assert _extract_retry_after_seconds(err) == 4.0


class TestSignalSendTimeout:
    """Timeout scaling for batched attachment sends."""

    def test_zero_attachments_uses_default(self):
        from gateway.platforms.signal import _signal_send_timeout
        assert _signal_send_timeout(0) == 30.0

    def test_floor_at_60s(self):
        from gateway.platforms.signal import _signal_send_timeout
        # Few attachments (would be 5×N=5s) should still get 60s floor.
        assert _signal_send_timeout(1) == 60.0
        assert _signal_send_timeout(5) == 60.0

    def test_scales_with_batch_size(self):
        from gateway.platforms.signal import _signal_send_timeout
        # 32 attachments × 5s = 160s; ought to comfortably outlast a
        # serial upload of an attachment-heavy batch.
        assert _signal_send_timeout(32) == 160.0


# ---------------------------------------------------------------------------
# Contentless Envelope Filtering (profile key updates, empty messages)
# ---------------------------------------------------------------------------

class TestSignalContentlessEnvelope:
    """Verify that profile key updates and empty Signal messages are skipped."""

    @pytest.mark.asyncio
    async def test_skips_profile_key_update_no_message_field(self, monkeypatch):
        """Profile key updates may carry a dataMessage without 'message' field.
        Must be skipped to avoid triggering agent turns for metadata."""
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        # Profile key update: dataMessage exists but has no "message" field
        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****9999",
                "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                "sourceName": "Elliott McManis",
                "timestamp": 1777600696077,
                "dataMessage": {
                    # No "message" field — profile key update metadata only
                    "profileKey": "some-profile-key-data",
                },
            }
        })

        assert "event" not in captured, "Profile key update should be skipped"

    @pytest.mark.asyncio
    async def test_skips_empty_message(self, monkeypatch):
        """Empty text messages (message='') should be skipped."""
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****9999",
                "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                "sourceName": "Elliott McManis",
                "timestamp": 1777600696077,
                "dataMessage": {
                    "message": "",
                },
            }
        })

        assert "event" not in captured, "Empty message should be skipped"

    @pytest.mark.asyncio
    async def test_skips_whitespace_only_message(self, monkeypatch):
        """Whitespace-only messages ('   ') should be skipped."""
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****9999",
                "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                "sourceName": "Elliott McManis",
                "timestamp": 1777600696077,
                "dataMessage": {
                    "message": "   \n\t  ",
                },
            }
        })

        assert "event" not in captured, "Whitespace-only message should be skipped"

    @pytest.mark.asyncio
    async def test_allows_message_with_attachment_no_text(self, monkeypatch):
        """Messages with attachments but no text should still be processed."""
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        # Mock attachment fetch to return a cached image
        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        b64_data = base64.b64encode(png_data).decode()
        adapter._rpc, _ = _stub_rpc({"data": b64_data})

        with patch("gateway.platforms.signal.cache_image_from_bytes", return_value="/tmp/img.png"):
            await adapter._handle_envelope({
                "envelope": {
                    "sourceNumber": "+155****9999",
                    "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                    "sourceName": "Elliott McManis",
                    "timestamp": 1777600696077,
                    "dataMessage": {
                        "message": "",  # No text
                        "attachments": [{"id": "att-123", "size": 200}],
                    },
                }
            })

        assert "event" in captured, "Message with attachment should NOT be skipped"
        assert captured["event"].media_urls == ["/tmp/img.png"]

    @pytest.mark.asyncio
    async def test_allows_normal_text_message(self, monkeypatch):
        """Normal text messages should still flow through."""
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****9999",
                "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                "sourceName": "Elliott McManis",
                "timestamp": 1777600696077,
                "dataMessage": {
                    "message": "hello world",
                },
            }
        })

        assert "event" in captured, "Normal message should NOT be skipped"
        assert captured["event"].text == "hello world"


class TestSignalSyncMessageHandling:
    """signal-cli running as a linked secondary device receives the user's
    own messages as ``syncMessage.sentMessage`` envelopes. Two cases must
    be handled:

      1. Note to Self (destination == self): promote to dataMessage so the
         user can talk to the agent in their own self-chat.
      2. Group sync-sent (destination is None, groupInfo set): promote so
         single-user / personal groups work.

    In both cases, the bot's own outbound replies bounce back as
    sync-sents and must be suppressed via the recently-sent timestamp ring.
    """

    @pytest.mark.asyncio
    async def test_note_to_self_promoted_to_inbound(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, account="+155****4567")
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****4567",  # self
                "sourceUuid": "uuid-self",
                "timestamp": 2000000000,
                "syncMessage": {
                    "sentMessage": {
                        "destinationNumber": "+155****4567",
                        "destination": "+155****4567",
                        "timestamp": 2000000000,
                        "message": "note to self: buy milk",
                    }
                },
            }
        })

        assert "event" in captured, "Note to Self must reach handle_message"
        assert captured["event"].text == "note to self: buy milk"

    @pytest.mark.asyncio
    async def test_note_to_self_echo_of_own_reply_is_suppressed(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, account="+155****4567")
        # Simulate that the bot just sent a reply with timestamp 3000000000
        adapter._track_sent_timestamp({"timestamp": 3000000000})
        called = []

        async def fake_handle(event):
            called.append(event)

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****4567",
                "sourceUuid": "uuid-self",
                "timestamp": 3000000000,
                "syncMessage": {
                    "sentMessage": {
                        "destinationNumber": "+155****4567",
                        "destination": "+155****4567",
                        "timestamp": 3000000000,
                        "message": "this is the bot's own reply echo",
                    }
                },
            }
        })

        assert called == [], "Echo of bot's own reply must be suppressed"
        # Consumed: timestamp must be removed from the ring
        assert 3000000000 not in adapter._recent_sent_timestamps

    @pytest.mark.asyncio
    async def test_group_sync_sent_promoted_to_inbound(self, monkeypatch):
        """User sends a message in a group from their primary phone; the
        linked device receives it as a sync-sent with destination=None and
        a groupInfo block. It must be treated as inbound so the agent can
        respond in groups when the user is the only human participant."""
        adapter = _make_signal_adapter(
            monkeypatch, account="+155****4567", group_allowed="abc123=="
        )
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****4567",
                "sourceUuid": "uuid-self",
                "timestamp": 4000000000,
                "syncMessage": {
                    "sentMessage": {
                        "destinationNumber": None,
                        "destination": None,
                        "timestamp": 4000000000,
                        "message": "ping the group",
                        "groupInfo": {
                            "groupId": "abc123==",
                            "type": "DELIVER",
                        },
                    }
                },
            }
        })

        assert "event" in captured, "Group sync-sent must reach handle_message"
        assert captured["event"].text == "ping the group"
        assert captured["event"].source.chat_id == "group:abc123=="

    @pytest.mark.asyncio
    async def test_group_sync_sent_echo_of_own_reply_is_suppressed(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, account="+155****4567")
        adapter._track_sent_timestamp({"timestamp": 5000000000})
        called = []

        async def fake_handle(event):
            called.append(event)

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****4567",
                "sourceUuid": "uuid-self",
                "timestamp": 5000000000,
                "syncMessage": {
                    "sentMessage": {
                        "destinationNumber": None,
                        "destination": None,
                        "timestamp": 5000000000,
                        "message": "bot's own group reply",
                        "groupInfo": {"groupId": "abc123==", "type": "DELIVER"},
                    }
                },
            }
        })

        assert called == [], "Group echo of bot's own reply must be suppressed"
        assert 5000000000 not in adapter._recent_sent_timestamps

    @pytest.mark.asyncio
    async def test_unrelated_sync_message_still_dropped(self, monkeypatch):
        """Read receipts / typing sync events have no sentMessage at all,
        or a sentMessage with non-self destination — must keep being filtered."""
        adapter = _make_signal_adapter(monkeypatch, account="+155****4567")
        called = []

        async def fake_handle(event):
            called.append(event)

        adapter.handle_message = fake_handle

        # No sentMessage at all
        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****4567",
                "timestamp": 6000000000,
                "syncMessage": {"readMessages": [{"sender": "+155****9999"}]},
            }
        })
        # sentMessage to a different contact (not self, not a group)
        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****4567",
                "timestamp": 6000000001,
                "syncMessage": {
                    "sentMessage": {
                        "destinationNumber": "+155****9999",
                        "destination": "+155****9999",
                        "timestamp": 6000000001,
                        "message": "outbound DM to someone else",
                    }
                },
            }
        })

        assert called == [], "Non-promotable sync messages must be filtered"


class TestRecentSentTimestampRing:
    """Verify the LRU+TTL behaviour of the echo-suppression ring."""

    def test_track_inserts_and_marks_most_recent(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._track_sent_timestamp({"timestamp": 1})
        adapter._track_sent_timestamp({"timestamp": 2})
        adapter._track_sent_timestamp({"timestamp": 1})  # touch
        # After touching 1, insertion order should be [2, 1]
        assert list(adapter._recent_sent_timestamps.keys()) == [2, 1]

    def test_consume_returns_true_and_removes(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._track_sent_timestamp({"timestamp": 42})
        assert adapter._consume_sent_timestamp(42) is True
        assert 42 not in adapter._recent_sent_timestamps
        assert adapter._consume_sent_timestamp(42) is False
        assert adapter._consume_sent_timestamp(None) is False

    def test_hard_cap_evicts_oldest(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._max_recent_timestamps = 3
        for ts in (1, 2, 3, 4):
            adapter._track_sent_timestamp({"timestamp": ts})
        # 1 should have been evicted (oldest); 2/3/4 retained in order
        assert list(adapter._recent_sent_timestamps.keys()) == [2, 3, 4]

    def test_ttl_evicts_stale_entries(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._recent_sent_ttl_seconds = 100.0

        # Drive time.monotonic deterministically.
        import gateway.platforms.signal as sig_mod
        fake_now = [1000.0]
        monkeypatch.setattr(sig_mod.time, "monotonic", lambda: fake_now[0])

        adapter._track_sent_timestamp({"timestamp": 1})
        fake_now[0] = 1050.0
        adapter._track_sent_timestamp({"timestamp": 2})
        fake_now[0] = 1200.0  # 200s elapsed since ts=1 (>TTL), 150s since ts=2 (>TTL)
        adapter._track_sent_timestamp({"timestamp": 3})
        # Both 1 and 2 should be evicted on TTL, only 3 remains
        assert list(adapter._recent_sent_timestamps.keys()) == [3]

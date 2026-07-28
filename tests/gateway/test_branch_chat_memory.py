"""Gateway /branch preserves the parent's frozen CHAT prompt opaquely."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.slash_commands import GatewaySlashCommandsMixin


@pytest.mark.asyncio
async def test_branch_inherits_parent_prompt_and_runtime_without_parsing_chat():
    handler = GatewaySlashCommandsMixin()
    parent_prompt = (
        "system head\n\n"
        "CHAT MEMORY [target: chat-aaaaaaaaaaaaaaaa] [1% — 10/4,200 chars]\n"
        "fact from A\n\n"
        "Model: parent-model\nProvider: openrouter\nPlatform: signal"
    )

    session_db = SimpleNamespace(
        get_session=AsyncMock(
            return_value={
                "system_prompt": parent_prompt,
                "model": "parent-model",
                "model_config": '{"provider":"openrouter","service_tier":"priority"}',
            }
        ),
        get_session_title=AsyncMock(return_value="Parent"),
        get_next_title_in_lineage=AsyncMock(return_value="Parent — branch 2"),
        create_session=AsyncMock(),
        append_message=AsyncMock(),
        set_session_title=AsyncMock(),
    )
    parent_entry = SimpleNamespace(session_id="parent-session")
    async_store = SimpleNamespace(
        get_or_create_session=AsyncMock(return_value=parent_entry),
        load_transcript=AsyncMock(
            return_value=[{"role": "user", "content": "continue A"}]
        ),
        switch_session=AsyncMock(return_value=SimpleNamespace(session_id="child")),
    )

    handler._session_db = session_db
    handler.async_session_store = async_store
    handler.config = {"model": {"default": "configured-default"}}
    handler._session_key_for_source = lambda _source: "agent:main:signal:group:B"
    handler._clear_session_boundary_security_state = MagicMock()
    handler._evict_cached_agent = MagicMock()

    source = SimpleNamespace(
        platform=SimpleNamespace(value="signal"),
        user_id="user-B",
        chat_id="group:B",
        chat_type="group",
        thread_id=None,
    )
    event = SimpleNamespace(source=source, get_command_args=lambda: "")

    await handler._handle_branch_command(event)

    kwargs = session_db.create_session.await_args.kwargs
    assert kwargs["parent_session_id"] == "parent-session"
    assert kwargs["system_prompt"] == parent_prompt
    assert "chat-aaaaaaaaaaaaaaaa" in kwargs["system_prompt"]
    assert "group:B" not in kwargs["system_prompt"]
    assert kwargs["model"] == "parent-model"
    assert kwargs["model_config"] == {
        "provider": "openrouter",
        "service_tier": "priority",
        "_branched_from": "parent-session",
    }
    # Routing metadata describes physical B but does not alter the opaque A prompt.
    assert kwargs["chat_id"] == "group:B"
    assert kwargs["session_key"] == "agent:main:signal:group:B"
    session_db.append_message.assert_awaited_once()
    async_store.switch_session.assert_awaited_once()

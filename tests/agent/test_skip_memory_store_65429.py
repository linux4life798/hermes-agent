"""Regression test for issue #65429.

An agent built with ``skip_memory=True`` AND ``enabled_toolsets=["memory"]``
used to get a memory tool wired to ``store=None`` (the built-in
``MemoryStore`` was skipped along with the external provider), so every
memory call failed silently and the main auto-capture path was dead.

The fix creates the built-in store whenever memory is enabled in config OR the
memory toolset is explicitly enabled, while the external-provider block stays
gated on ``skip_memory``.
"""

import pytest

from run_agent import AIAgent


class _FakeOpenAI:
    def __init__(self, **kw):
        self.api_key = kw.get("api_key", "test")
        self.base_url = kw.get("base_url", "http://test")

    def close(self):
        pass


def _make_agent(monkeypatch, enabled_toolsets=None, skip_memory=True, **kwargs):
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)
    return AIAgent(
        api_key="test-key",
        base_url="http://test",
        provider="openrouter",
        api_mode="chat_completions",
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=skip_memory,
        enabled_toolsets=enabled_toolsets,
        **kwargs,
    )


def test_skip_memory_with_memory_toolset_creates_store(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    agent = _make_agent(monkeypatch, enabled_toolsets=["memory"], skip_memory=True)
    assert agent._memory_store is not None, (
        "memory toolset enabled despite skip_memory=True must still build "
        "the built-in store (#65429)"
    )


def test_skip_memory_without_memory_toolset_has_no_store(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    agent = _make_agent(monkeypatch, enabled_toolsets=None, skip_memory=True)
    assert agent._memory_store is None, (
        "flush/background agent with skip_memory and no memory toolset must "
        "have no built-in store"
    )


def test_memory_toolset_without_skip_memory_creates_store(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    agent = _make_agent(monkeypatch, enabled_toolsets=["memory"], skip_memory=False)
    assert agent._memory_store is not None


def test_skip_memory_with_chat_origin_loads_only_chat_store(monkeypatch, tmp_path):
    """Cron-style agents suppress global memory but still receive one CHAT block."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    agent = _make_agent(
        monkeypatch,
        enabled_toolsets=None,
        skip_memory=True,
        platform="signal",
        chat_id="group:cron-origin",
    )

    assert agent._memory_store is not None
    assert agent._memory_enabled is False
    assert agent._user_profile_enabled is False
    assert agent._chat_memory_enabled is True
    assert agent._chat_memory_target.startswith("chat-")

    from agent.system_prompt import build_system_prompt_parts

    volatile = build_system_prompt_parts(agent)["volatile"]
    assert f"CHAT MEMORY [target: {agent._chat_memory_target}]" in volatile
    assert "PERSISTENT MEMORY (your personal notes)" not in volatile
    assert "USER PROFILE (who the user is)" not in volatile


def test_invalid_chat_key_does_not_disable_global_memory(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hm"
    mem_dir = hermes_home / "memories"
    mem_dir.mkdir(parents=True)
    (mem_dir / "MEMORY.md").write_text("Global fact survives.", encoding="utf-8")
    (mem_dir / ".chat-target-key").write_text("not-hex\n", encoding="ascii")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    agent = _make_agent(
        monkeypatch,
        enabled_toolsets=["memory"],
        skip_memory=False,
        platform="signal",
        chat_id="group:corrupt-key",
    )

    assert agent._memory_store is not None
    assert "Global fact survives." in agent._memory_store.memory_entries
    assert agent._chat_memory_enabled is False
    assert agent._chat_memory_target is None


def test_skip_memory_memory_tool_handler_works_and_provider_skipped(
    monkeypatch, tmp_path
):
    """End-to-end behavioral check for #65429.

    The memory tool handler must actually WORK (not return the
    "Memory is not available" store=None error) on a skip_memory=True agent
    with the memory toolset enabled, while the external memory provider
    sync/prefetch stays skipped (no MemoryManager is created).
    """
    import json

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    agent = _make_agent(monkeypatch, enabled_toolsets=["memory"], skip_memory=True)

    # Provider sync/prefetch must remain skipped: skip_memory still gates the
    # external memory provider block.
    assert agent._memory_manager is None, (
        "skip_memory=True must still skip the external memory provider"
    )

    # Dispatch through the same entry point the tool executor uses
    # (agent/tool_executor.py wires store=agent._memory_store).
    from tools.memory_tool import memory_tool

    raw = memory_tool(
        action="add",
        target="memory",
        content="User prefers concise answers.",
        store=agent._memory_store,
    )
    result = json.loads(raw)
    assert result.get("success") is True, (
        f"memory tool handler must work with skip_memory=True + memory "
        f"toolset (#65429), got: {raw}"
    )
    assert "Memory is not available" not in raw

    # The write must actually persist to the profile-scoped memories dir.
    memory_md = tmp_path / "hm" / "memories" / "MEMORY.md"
    assert memory_md.exists()
    assert "User prefers concise answers." in memory_md.read_text()

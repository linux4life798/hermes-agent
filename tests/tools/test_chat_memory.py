"""CHAT-scoped persistent-memory behavior and privacy contracts."""

from __future__ import annotations

import json
import stat
from concurrent.futures import ThreadPoolExecutor

import pytest

import tools.memory_tool as memory_module
from tools.memory_tool import (
    MemoryStore,
    derive_chat_memory_target,
    is_valid_chat_memory_target,
    memory_tool,
)


@pytest.fixture
def chat_memory_dir(monkeypatch, tmp_path):
    mem_dir = tmp_path / "memories"
    monkeypatch.setattr(memory_module, "get_memory_dir", lambda: mem_dir)
    # Deterministic local-only key: tests never depend on process randomness.
    mem_dir.mkdir(parents=True)
    (mem_dir / ".chat-target-key").write_text("11" * 32 + "\n", encoding="ascii")
    return mem_dir


def _target(platform: str, chat_id: str, thread_id: str | None = None) -> str:
    target = derive_chat_memory_target(platform, chat_id, thread_id)
    assert target is not None
    return target


def test_chat_target_is_stable_opaque_and_identity_scoped(chat_memory_dir):
    target = _target("signal", "group:stable-native-id")

    assert target == _target("SIGNAL", "group:stable-native-id")
    assert target != _target("signal", "group:other-native-id")
    assert target != _target(
        "signal", "group:stable-native-id", "thread-2"
    )
    assert target != _target("telegram", "group:stable-native-id")
    assert is_valid_chat_memory_target(target)
    assert "stable-native-id" not in target


def test_chat_target_key_is_private(chat_memory_dir):
    key_path = chat_memory_dir / ".chat-target-key"
    key_path.chmod(0o644)

    derive_chat_memory_target("signal", "dm:guessable-identifier")

    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600


def test_empty_existing_chat_key_fails_closed_without_recursion(
    monkeypatch, tmp_path
):
    mem_dir = tmp_path / "memories"
    mem_dir.mkdir()
    (mem_dir / ".chat-target-key").write_text("", encoding="ascii")
    monkeypatch.setattr(memory_module, "get_memory_dir", lambda: mem_dir)

    with pytest.raises(RuntimeError, match="Empty CHAT target key"):
        derive_chat_memory_target("signal", "group:race-or-crash")


def test_concurrent_first_derivation_converges_on_one_key(monkeypatch, tmp_path):
    mem_dir = tmp_path / "memories"
    monkeypatch.setattr(memory_module, "get_memory_dir", lambda: mem_dir)

    with ThreadPoolExecutor(max_workers=16) as pool:
        targets = list(
            pool.map(
                lambda _: derive_chat_memory_target("signal", "group:concurrent"),
                range(64),
            )
        )

    assert None not in targets
    assert len(set(targets)) == 1
    raw_key = (mem_dir / ".chat-target-key").read_text(encoding="ascii").strip()
    assert len(bytes.fromhex(raw_key)) == 32
    assert stat.S_IMODE((mem_dir / ".chat-target-key").stat().st_mode) == 0o600


def test_empty_chat_prompt_has_address_without_creating_memory_file(chat_memory_dir):
    target = _target("signal", "group:new-chat")
    store = MemoryStore(memory_char_limit=4200, chat_target=target)

    store.load_from_disk()

    block = store.format_for_system_prompt(target)
    assert block is not None
    assert f"CHAT MEMORY [target: {target}]" in block
    assert "0/4,200 chars" in block
    assert f"Use the exact target `{target}`" in block
    assert not (chat_memory_dir / "chat" / f"{target[5:]}.md").exists()


def test_chat_write_is_lazy_atomic_and_isolated(chat_memory_dir):
    target_a = _target("signal", "group:a")
    target_b = _target("signal", "group:b")
    store = MemoryStore(memory_char_limit=4200, chat_target=target_a)
    store.load_from_disk()

    result_a = json.loads(
        memory_tool(
            action="add",
            target=target_a,
            content="Chat A uses Eastern time.",
            store=store,
        )
    )
    result_b = json.loads(
        memory_tool(
            action="add",
            target=target_b,
            content="Chat B uses Pacific time.",
            store=store,
        )
    )

    assert result_a["success"] is True
    assert result_b["success"] is True
    path_a = chat_memory_dir / "chat" / f"{target_a[5:]}.md"
    path_b = chat_memory_dir / "chat" / f"{target_b[5:]}.md"
    assert path_a.read_text() == "Chat A uses Eastern time."
    assert path_b.read_text() == "Chat B uses Pacific time."
    assert stat.S_IMODE(path_a.stat().st_mode) == 0o600
    assert stat.S_IMODE(path_b.stat().st_mode) == 0o600
    assert not list((chat_memory_dir / "chat").glob("*.tmp"))

    # Automatic prompt injection remains narrow even though the tool may
    # explicitly address another syntactically valid CHAT target.
    next_a = MemoryStore(memory_char_limit=4200, chat_target=target_a)
    next_a.load_from_disk()
    prompt_a = next_a.format_for_system_prompt(target_a)
    assert prompt_a is not None
    assert "Chat A uses Eastern time." in prompt_a
    assert "Chat B uses Pacific time." not in prompt_a

    next_b = MemoryStore(memory_char_limit=4200, chat_target=target_b)
    next_b.load_from_disk()
    prompt_b = next_b.format_for_system_prompt(target_b)
    assert prompt_b is not None
    assert "Chat B uses Pacific time." in prompt_b
    assert "Chat A uses Eastern time." not in prompt_b


def test_chat_snapshot_stays_frozen_until_next_store(chat_memory_dir):
    target = _target("signal", "group:frozen")
    store = MemoryStore(memory_char_limit=4200, chat_target=target)
    store.load_from_disk()
    initial = store.format_for_system_prompt(target)

    result = json.loads(
        memory_tool(
            action="add",
            target=target,
            content="Durable fact written mid-session.",
            store=store,
        )
    )
    assert result["success"] is True
    assert store.format_for_system_prompt(target) == initial
    assert "Durable fact written mid-session." not in initial

    next_session = MemoryStore(memory_char_limit=4200, chat_target=target)
    next_session.load_from_disk()
    assert "Durable fact written mid-session." in next_session.format_for_system_prompt(
        target
    )


@pytest.mark.parametrize(
    "target",
    [
        "chat-../../MEMORY",
        "chat-ABCDEF0123456789",
        "chat-123",
        "chat-0123456789abcdef/extra",
        "chat-0123456789abcdef.md",
        "signal-group-native-id",
    ],
)
def test_malformed_chat_targets_cannot_escape_directory(chat_memory_dir, target):
    store = MemoryStore()

    result = json.loads(
        memory_tool(action="add", target=target, content="must not write", store=store)
    )

    assert result["success"] is False
    assert "Invalid target" in result["error"]
    assert not (chat_memory_dir / "MEMORY.md").exists()
    assert not (chat_memory_dir / "chat").exists()

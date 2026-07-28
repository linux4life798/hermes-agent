"""Tests for the `hermes memory reset` CLI command.

Covers:
- Reset all stores (MEMORY.md + USER.md + CHAT files)
- Reset individual stores (--target memory / --target user / --target chat)
- Preserve the private CHAT target-derivation key across content resets
- Skip confirmation with --yes
- Graceful handling when no memory files exist
- Profile-scoped reset (uses HERMES_HOME)
"""

from types import SimpleNamespace
from unittest.mock import patch as mock_patch

import pytest


@pytest.fixture
def memory_env(tmp_path, monkeypatch):
    """Set up a fake HERMES_HOME with memory files."""
    hermes_home = tmp_path / ".hermes"
    memories = hermes_home / "memories"
    memories.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Create sample memory files
    (memories / "MEMORY.md").write_text(
        "§\nHermes repo is at ~/.hermes/hermes-agent\n§\nUser prefers dark themes",
        encoding="utf-8",
    )
    (memories / "USER.md").write_text(
        "§\nUser is Teknium\n§\nTimezone: US Pacific",
        encoding="utf-8",
    )
    chat_dir = memories / "chat"
    chat_dir.mkdir()
    (chat_dir / "1111111111111111.md").write_text("Chat one", encoding="utf-8")
    (chat_dir / "2222222222222222.md").write_text("Chat two", encoding="utf-8")
    (memories / ".chat-target-key").write_text("11" * 32 + "\n", encoding="ascii")
    return hermes_home, memories


def _run_memory_reset(target="all", yes=False, monkeypatch=None, confirm_input="no"):
    """Invoke the production reset helper with deterministic confirmation."""
    from hermes_cli.main import _cmd_memory_reset

    args = SimpleNamespace(target=target, yes=yes)
    with mock_patch("builtins.input", return_value=confirm_input):
        return _cmd_memory_reset(args)


class TestMemoryReset:
    """Tests for `hermes memory reset` subcommand."""

    def test_reset_all_with_yes_flag(self, memory_env):
        """--yes flag should delete global, user, and CHAT content."""
        hermes_home, memories = memory_env
        assert (memories / "MEMORY.md").exists()
        assert (memories / "USER.md").exists()

        result = _run_memory_reset(target="all", yes=True)
        assert result == "deleted"
        assert not (memories / "MEMORY.md").exists()
        assert not (memories / "USER.md").exists()
        assert not list((memories / "chat").glob("*.md"))
        assert (memories / ".chat-target-key").exists()

    def test_reset_memory_only(self, memory_env):
        """--target memory should only delete MEMORY.md."""
        hermes_home, memories = memory_env

        result = _run_memory_reset(target="memory", yes=True)
        assert result == "deleted"
        assert not (memories / "MEMORY.md").exists()
        assert (memories / "USER.md").exists()

    def test_reset_user_only(self, memory_env):
        """--target user should only delete USER.md."""
        hermes_home, memories = memory_env

        result = _run_memory_reset(target="user", yes=True)
        assert result == "deleted"
        assert (memories / "MEMORY.md").exists()
        assert not (memories / "USER.md").exists()

    def test_reset_chat_only_preserves_global_user_and_key(self, memory_env):
        hermes_home, memories = memory_env

        result = _run_memory_reset(target="chat", yes=True)

        assert result == "deleted"
        assert (memories / "MEMORY.md").exists()
        assert (memories / "USER.md").exists()
        assert not list((memories / "chat").glob("*.md"))
        assert (memories / ".chat-target-key").exists()

    def test_reset_chat_refuses_symlinked_directory(self, memory_env, tmp_path):
        """A symlinked CHAT directory must not redirect deletion outside the profile."""
        _hermes_home, memories = memory_env
        chat_dir = memories / "chat"
        for path in chat_dir.iterdir():
            path.unlink()
        chat_dir.rmdir()

        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "3333333333333333.md"
        secret.write_text("do not delete", encoding="utf-8")
        try:
            chat_dir.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"directory symlinks unavailable: {exc}")

        result = _run_memory_reset(target="all", yes=True)

        assert result == "refused"
        assert secret.read_text(encoding="utf-8") == "do not delete"
        assert (memories / "MEMORY.md").exists()
        assert (memories / "USER.md").exists()
        assert (memories / ".chat-target-key").exists()

    def test_reset_chat_unlinks_symlink_entry_without_following_it(self, memory_env, tmp_path):
        """A valid-looking CHAT symlink is removed without touching its target."""
        _hermes_home, memories = memory_env
        chat_dir = memories / "chat"
        outside = tmp_path / "outside-entry.md"
        outside.write_text("external", encoding="utf-8")
        linked = chat_dir / "3333333333333333.md"
        try:
            linked.symlink_to(outside)
        except OSError as exc:
            pytest.skip(f"file symlinks unavailable: {exc}")

        result = _run_memory_reset(target="chat", yes=True)

        assert result == "deleted"
        assert not linked.exists()
        assert outside.read_text(encoding="utf-8") == "external"
        assert (memories / ".chat-target-key").exists()

    def test_reset_chat_preserves_unrelated_markdown(self, memory_env):
        """Only valid opaque CHAT filenames are reset."""
        _hermes_home, memories = memory_env
        unrelated = memories / "chat" / "notes.md"
        unrelated.write_text("keep", encoding="utf-8")

        result = _run_memory_reset(target="chat", yes=True)

        assert result == "deleted"
        assert unrelated.read_text(encoding="utf-8") == "keep"

    def test_reset_no_files_exist(self, tmp_path, monkeypatch):
        """Should return 'nothing' when no memory files exist."""
        hermes_home = tmp_path / ".hermes"
        (hermes_home / "memories").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        result = _run_memory_reset(target="all", yes=True)
        assert result == "nothing"

    def test_reset_confirmation_denied(self, memory_env):
        """Without --yes and without typing 'yes', should be cancelled."""
        hermes_home, memories = memory_env

        result = _run_memory_reset(target="all", yes=False, confirm_input="no")
        assert result == "cancelled"
        # Files should still exist
        assert (memories / "MEMORY.md").exists()
        assert (memories / "USER.md").exists()

    def test_reset_confirmation_accepted(self, memory_env):
        """Typing 'yes' should proceed with deletion."""
        hermes_home, memories = memory_env

        result = _run_memory_reset(target="all", yes=False, confirm_input="yes")
        assert result == "deleted"
        assert not (memories / "MEMORY.md").exists()
        assert not (memories / "USER.md").exists()

    def test_reset_profile_scoped(self, tmp_path, monkeypatch):
        """Reset should work on the active profile's HERMES_HOME."""
        profile_home = tmp_path / "profiles" / "myprofile"
        memories = profile_home / "memories"
        memories.mkdir(parents=True)
        (memories / "MEMORY.md").write_text("profile memory", encoding="utf-8")
        (memories / "USER.md").write_text("profile user", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(profile_home))

        result = _run_memory_reset(target="all", yes=True)
        assert result == "deleted"
        assert not (memories / "MEMORY.md").exists()
        assert not (memories / "USER.md").exists()

    def test_reset_partial_files(self, memory_env):
        """Reset should work when only one memory file exists."""
        hermes_home, memories = memory_env
        (memories / "USER.md").unlink()

        result = _run_memory_reset(target="all", yes=True)
        assert result == "deleted"
        assert not (memories / "MEMORY.md").exists()

    def test_reset_empty_memories_dir(self, tmp_path, monkeypatch):
        """No memories dir at all should report nothing."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True)
        # No memories dir
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        # The memories dir won't exist; get_hermes_home() / "memories" won't have files
        result = _run_memory_reset(target="all", yes=True)
        assert result == "nothing"

"""Tests for the standalone Hermes dashboard service helper script."""

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hermes-dashboard"


loader = SourceFileLoader("hermes_dashboard_script", str(SCRIPT_PATH))
spec = spec_from_loader(loader.name, loader)
assert spec is not None
hermes_dashboard = module_from_spec(spec)
loader.exec_module(hermes_dashboard)


def test_generate_systemd_unit_runs_dashboard_with_no_open():
    unit = hermes_dashboard.generate_systemd_unit()

    assert "ExecStart=" in unit
    assert "/.venv/bin/python" in unit
    assert "-m hermes_cli.main dashboard --no-open" in unit
    assert 'Environment="PATH=' in unit
    assert 'Environment="VIRTUAL_ENV=' in unit
    assert "/.venv" in unit
    assert "WantedBy=default.target" in unit
    assert "Restart=on-failure" in unit


def test_get_python_path_uses_dot_venv_without_legacy_fallback(monkeypatch, tmp_path):
    project_dir = tmp_path / "project"
    legacy_python = project_dir / "venv" / "bin" / "python"
    legacy_python.parent.mkdir(parents=True)
    legacy_python.touch()
    monkeypatch.setattr(hermes_dashboard, "PROJECT_DIR", project_dir)

    assert hermes_dashboard.get_python_path() == str(
        project_dir / ".venv" / "bin" / "python"
    )


def test_generate_systemd_unit_quotes_paths_with_spaces(monkeypatch, tmp_path):
    project_dir = tmp_path / "Hermes Dashboard"
    dot_python = project_dir / ".venv" / "bin" / "python"
    dot_python.parent.mkdir(parents=True)
    dot_python.touch()
    monkeypatch.setattr(hermes_dashboard, "PROJECT_DIR", project_dir)

    unit = hermes_dashboard.generate_systemd_unit()
    expected_exec = f'ExecStart="{project_dir / ".venv" / "bin" / "python"}" -m hermes_cli.main dashboard --no-open'
    expected_workdir = "WorkingDirectory=" + str(project_dir).replace(" ", r"\x20")
    expected_virtual_env = f'Environment="VIRTUAL_ENV={project_dir / ".venv"}"'

    assert expected_exec in unit
    assert expected_workdir in unit
    assert expected_virtual_env in unit

import json
import os

from canrun.config import DEFAULT_MODEL, Config, clear_key, config_path, set_key, set_model


def test_key_workflow_and_permissions(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    path = set_key("  secret-key  ")

    assert path == config_path()
    assert Config.load().api_key == "secret-key"
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())["api_key"] == "secret-key"

    assert clear_key() is True
    assert Config.load().api_key is None


def test_environment_key_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    set_key("stored")
    monkeypatch.setenv("GEMINI_API_KEY", "environment")
    assert Config.load().api_key == "environment"


def test_model_is_configurable(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    set_model("gemini-example")
    assert Config.load().model == "gemini-example"


def test_retired_default_model_is_migrated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    set_model("gemini-2.5-flash")

    assert Config.load().model == DEFAULT_MODEL

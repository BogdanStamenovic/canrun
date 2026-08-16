import io
import json

from canrun.cli import main
from canrun.config import Config


def test_config_set_key_reads_stdin_without_echoing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr("sys.stdin", io.StringIO("secret-value\n"))

    assert main(["config", "set-key"]) == 0
    output = capsys.readouterr().out
    assert "secret-value" not in output
    assert Config.load().api_key == "secret-value"


def test_specs_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(
        "canrun.cli.detect_hardware",
        lambda: {
            "operating_system": {},
            "cpu": {},
            "gpus": [],
            "memory": {},
            "storage": {},
            "display": {},
        },
    )

    assert main(["specs", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["gpus"] == []


def test_application_name_can_be_used_without_check_word(monkeypatch):
    captured = {}

    class FakeAnalyzer:
        def __init__(self, client, progress):
            pass

        def run(self, application, *args):
            captured["application"] = application
            return ({"summary": "ok", "component_results": {}}, True)

    monkeypatch.setenv("GEMINI_API_KEY", "key")
    monkeypatch.setattr("canrun.cli.detect_hardware", dict)
    monkeypatch.setattr("canrun.cli.write_json", lambda path, value: path)
    monkeypatch.setattr("canrun.cli.Analyzer", FakeAnalyzer)

    assert main(["Example", "Game"]) == 0
    assert captured["application"] == "Example Game"

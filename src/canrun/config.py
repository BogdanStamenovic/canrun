from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "gemini-2.5-flash"


class ConfigError(RuntimeError):
    pass


def config_home() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "canrun"


def config_path() -> Path:
    return config_home() / "config.json"


@dataclass(frozen=True)
class Config:
    api_key: str | None = None
    model: str = DEFAULT_MODEL

    @classmethod
    def load(cls) -> Config:
        path = config_path()
        data: dict = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigError(f"could not read {path}: {exc}") from exc
            if not isinstance(loaded, dict):
                raise ConfigError(f"{path} must contain a JSON object")
            data = loaded
        env_key = os.environ.get("GEMINI_API_KEY")
        return cls(
            api_key=env_key or data.get("api_key"),
            model=str(data.get("model") or DEFAULT_MODEL),
        )

    def require_key(self) -> str:
        if not self.api_key:
            raise ConfigError("no Gemini API key configured; run 'canrun config set-key'")
        return self.api_key


def _stored_data() -> dict:
    path = config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a JSON object")
    return data


def _save(data: dict) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(data, indent=2) + "\n")
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        temporary.replace(path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def set_key(api_key: str) -> Path:
    value = api_key.strip()
    if not value:
        raise ConfigError("API key cannot be empty")
    data = _stored_data()
    data["api_key"] = value
    return _save(data)


def clear_key() -> bool:
    data = _stored_data()
    existed = bool(data.pop("api_key", None))
    _save(data)
    return existed


def set_model(model: str) -> Path:
    value = model.strip()
    if not value or any(char.isspace() for char in value):
        raise ConfigError("model must be a non-empty model identifier without spaces")
    data = _stored_data()
    data["model"] = value
    return _save(data)


def masked_key(key: str | None) -> str:
    if not key:
        return "not configured"
    if len(key) <= 8:
        return "configured (hidden)"
    return f"{key[:4]}…{key[-4:]}"

import json

import pytest

from canrun.gemini import (
    GeminiClient,
    GeminiError,
    InvalidGeminiResponse,
    extract_json,
)


def _payload(text):
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": text}]},
                "groundingMetadata": {
                    "groundingChunks": [
                        {"web": {"title": "Official", "uri": "https://example.test"}}
                    ]
                },
            }
        ]
    }


def test_extract_json_accepts_plain_and_fenced_objects():
    assert extract_json('{"ok": true}') == {"ok": True}
    assert extract_json('```json\n{"ok": true}\n```') == {"ok": True}


def test_call_retries_until_a_valid_success(monkeypatch):
    client = GeminiClient("key", "model")
    responses = iter([_payload("not json"), _payload('{"ok": true}')])
    monkeypatch.setattr(client, "_request", lambda prompt: next(responses))
    monkeypatch.setattr("canrun.gemini.time.sleep", lambda delay: None)

    result = client.call_json("prompt", "stage", lambda data: None, max_attempts=3)

    assert result.data == {"ok": True}
    assert result.attempts == 2
    assert result.sources[0]["url"] == "https://example.test"


def test_call_stops_after_attempt_limit(monkeypatch):
    client = GeminiClient("key", "model")
    monkeypatch.setattr(client, "_request", lambda prompt: _payload("bad"))
    monkeypatch.setattr("canrun.gemini.time.sleep", lambda delay: None)

    with pytest.raises(GeminiError, match="failed after 2 attempts"):
        client.call_json("prompt", "stage", lambda data: None, max_attempts=2)


def test_invalid_non_object_json_is_rejected():
    with pytest.raises(InvalidGeminiResponse):
        extract_json(json.dumps([1, 2]))

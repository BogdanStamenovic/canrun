import json
import urllib.request
from io import BytesIO

import pytest

from canrun.gemini import (
    GeminiClient,
    GeminiError,
    InvalidGeminiResponse,
    extract_json,
)
from canrun.prompts import performance_prompt, requirements_prompt


def _payload(text):
    return {
        "steps": [
            {
                "type": "google_search_call",
                "id": "search-1",
                "arguments": {"queries": ["official requirements"]},
            },
            {
                "type": "model_output",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                        "annotations": [
                            {
                                "type": "url_citation",
                                "title": "Official",
                                "url": "https://example.test",
                            }
                        ],
                    }
                ],
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


def test_call_retries_with_search_specific_correction(monkeypatch):
    client = GeminiClient("key", "model")
    no_search = _payload('{"ok": true}')
    no_search["steps"] = [
        step for step in no_search["steps"] if step["type"] != "google_search_call"
    ]
    prompts = []
    responses = iter([no_search, _payload('{"ok": true}')])

    def request(prompt):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr("canrun.gemini.time.sleep", lambda delay: None)

    result = client.call_json("base prompt", "stage", lambda data: None)

    assert result.attempts == 2
    assert prompts[0] == "base prompt"
    assert "previous attempt answered without invoking Google Search" in prompts[1]


def test_call_retries_with_citation_specific_correction(monkeypatch):
    client = GeminiClient("key", "model")
    no_citations = _payload('{"ok": true}')
    no_citations["steps"][1]["content"][0]["annotations"] = []
    prompts = []
    responses = iter([no_citations, _payload('{"ok": true}')])
    monkeypatch.setattr(
        client,
        "_request",
        lambda prompt: prompts.append(prompt) or next(responses),
    )
    monkeypatch.setattr("canrun.gemini.time.sleep", lambda delay: None)

    client.call_json("base prompt", "stage", lambda data: None)

    assert "previous attempt searched but did not cite its evidence" in prompts[1]


def test_citations_without_search_invocation_are_rejected(monkeypatch):
    client = GeminiClient("key", "model")
    payload = _payload('{"ok": true}')
    payload["steps"] = [step for step in payload["steps"] if step["type"] != "google_search_call"]
    monkeypatch.setattr(client, "_request", lambda prompt: payload)
    monkeypatch.setattr("canrun.gemini.time.sleep", lambda delay: None)

    with pytest.raises(GeminiError, match="did not invoke Google Search"):
        client.call_json("prompt", "stage", lambda data: None, max_attempts=1)


def test_invalid_non_object_json_is_rejected():
    with pytest.raises(InvalidGeminiResponse):
        extract_json(json.dumps([1, 2]))


def test_both_stage_prompts_require_google_search():
    requirements = requirements_prompt("Demo", "Linux")
    performance = performance_prompt("Demo", {}, {}, {}, "1080p", "medium")

    for prompt in (requirements, performance):
        assert "MANDATORY: invoke the google_search tool" in prompt
        assert "URLs will be rejected" in prompt


def test_request_uses_interactions_api(monkeypatch):
    captured = {}

    class Response(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response(b'{"steps": []}')

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = GeminiClient("key", "gemini-3.6-flash", timeout=12)

    client._request("research this")

    request = captured["request"]
    assert request.full_url == "https://generativelanguage.googleapis.com/v1/interactions"
    assert captured["timeout"] == 12
    assert request.get_header("X-goog-api-key") == "key"
    assert json.loads(request.data) == {
        "model": "gemini-3.6-flash",
        "input": "research this",
        "tools": [{"type": "google_search", "search_types": ["web_search"]}],
    }

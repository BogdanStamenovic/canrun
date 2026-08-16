from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass


class GeminiError(RuntimeError):
    pass


class InvalidGeminiResponse(GeminiError):
    pass


@dataclass(frozen=True)
class GroundedResult:
    data: dict
    sources: list[dict[str, str]]
    attempts: int


def extract_json(text: str) -> dict:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise InvalidGeminiResponse("Gemini returned no JSON object") from None
        try:
            parsed = json.loads(value[start : end + 1])
        except json.JSONDecodeError as exc:
            raise InvalidGeminiResponse(f"Gemini returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise InvalidGeminiResponse("Gemini JSON response was not an object")
    return parsed


def _response_text(payload: dict) -> str:
    steps = payload.get("steps") or []
    text = "\n".join(
        str(block.get("text", ""))
        for step in steps
        if step.get("type") == "model_output"
        for block in step.get("content") or []
        if block.get("type") == "text" and block.get("text")
    )
    if text.strip():
        return text

    # Accept generateContent responses so cached fixtures and older callers remain readable.
    candidates = payload.get("candidates") or []
    if not candidates:
        feedback = payload.get("promptFeedback") or {}
        reason = feedback.get("blockReason") or "no candidate returned"
        raise InvalidGeminiResponse(f"Gemini produced no response: {reason}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "\n".join(str(part.get("text", "")) for part in parts if part.get("text"))
    if not text.strip():
        raise InvalidGeminiResponse("Gemini returned an empty response")
    return text


def _grounding_sources(payload: dict) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for step in payload.get("steps") or []:
        if step.get("type") != "model_output":
            continue
        for block in step.get("content") or []:
            for annotation in block.get("annotations") or []:
                if annotation.get("type") != "url_citation":
                    continue
                uri = str(annotation.get("url") or "").strip()
                if uri and uri not in seen:
                    seen.add(uri)
                    sources.append({"title": str(annotation.get("title") or uri), "url": uri})
    if sources:
        return sources

    # Fall back to the generateContent grounding shape for backwards compatibility.
    candidates = payload.get("candidates") or []
    metadata = candidates[0].get("groundingMetadata") or {} if candidates else {}
    for chunk in metadata.get("groundingChunks") or []:
        web = chunk.get("web") or {}
        uri = str(web.get("uri") or "").strip()
        if uri and uri not in seen:
            seen.add(uri)
            sources.append({"title": str(web.get("title") or uri), "url": uri})
    return sources


def validate_requirements(data: dict) -> None:
    required = ("application", "minimum", "recommended", "sources", "confidence", "warnings")
    missing = [key for key in required if key not in data]
    if missing:
        raise InvalidGeminiResponse(f"requirements response omitted: {', '.join(missing)}")
    if not isinstance(data["application"], dict):
        raise InvalidGeminiResponse("application must be an object")
    if not isinstance(data["application"].get("name"), str):
        raise InvalidGeminiResponse("application.name must be a string")
    for key in ("minimum", "recommended"):
        if data[key] is not None and not isinstance(data[key], dict):
            raise InvalidGeminiResponse(f"{key} must be an object or null")
        tier = data[key] or {}
        for field in ("ram_gb", "vram_gb", "storage_gb"):
            value = tier.get(field)
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool)
            ):
                raise InvalidGeminiResponse(f"{key}.{field} must be a number or null")
    if not isinstance(data["sources"], list) or not isinstance(data["warnings"], list):
        raise InvalidGeminiResponse("sources and warnings must be arrays")
    if not all(isinstance(item, dict) and item.get("url") for item in data["sources"]):
        raise InvalidGeminiResponse("each requirements source must contain a URL")
    if data["confidence"] not in {"high", "medium", "low"}:
        raise InvalidGeminiResponse("confidence must be high, medium, or low")


def validate_report(data: dict) -> None:
    required = (
        "compatibility",
        "will_launch",
        "confidence",
        "bottleneck",
        "estimated_performance",
        "component_results",
        "recommendations",
        "sources",
        "uncertainties",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise InvalidGeminiResponse(f"performance response omitted: {', '.join(missing)}")
    if not isinstance(data["component_results"], dict):
        raise InvalidGeminiResponse("component_results must be an object")
    missing_components = [
        key
        for key in ("cpu", "gpu", "vram", "ram", "storage", "os")
        if not isinstance(data["component_results"].get(key), dict)
    ]
    if missing_components:
        raise InvalidGeminiResponse(f"component_results omitted: {', '.join(missing_components)}")
    if data["compatibility"] not in {
        "cannot_run",
        "below_minimum",
        "meets_minimum",
        "meets_recommended",
        "unknown",
    }:
        raise InvalidGeminiResponse("invalid compatibility value")
    if data["confidence"] not in {"high", "medium", "low"}:
        raise InvalidGeminiResponse("confidence must be high, medium, or low")
    performance = data["estimated_performance"]
    if not isinstance(performance, dict):
        raise InvalidGeminiResponse("estimated_performance must be an object")
    for field in ("fps_low", "fps_typical", "fps_high"):
        value = performance.get(field)
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise InvalidGeminiResponse(f"estimated_performance.{field} must be numeric or null")
    for key in ("recommendations", "sources", "uncertainties"):
        if not isinstance(data[key], list):
            raise InvalidGeminiResponse(f"{key} must be an array")
    if not all(isinstance(item, dict) and item.get("url") for item in data["sources"]):
        raise InvalidGeminiResponse("each performance source must contain a URL")


class GeminiClient:
    def __init__(self, api_key: str, model: str, timeout: int = 90):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _request(self, prompt: str) -> dict:
        url = "https://generativelanguage.googleapis.com/v1/interactions"
        body = {
            "model": self.model,
            "input": prompt,
            "tools": [{"type": "google_search"}],
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                try:
                    payload = json.load(response)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise InvalidGeminiResponse("Gemini returned an invalid HTTP response") from exc
        except urllib.error.HTTPError as exc:
            detail = exc.reason
            try:
                error = json.load(exc).get("error") or {}
                detail = error.get("message") or detail
            except (ValueError, AttributeError):
                pass
            error = GeminiError(f"Gemini API error {exc.code}: {detail}")
            error.retryable = exc.code in {408, 429, 500, 502, 503, 504}  # type: ignore[attr-defined]
            raise error from exc
        except urllib.error.URLError as exc:
            error = GeminiError(f"could not reach Gemini: {exc.reason}")
            error.retryable = True  # type: ignore[attr-defined]
            raise error from exc
        except TimeoutError as exc:
            error = GeminiError("Gemini request timed out")
            error.retryable = True  # type: ignore[attr-defined]
            raise error from exc
        if not isinstance(payload, dict):
            raise InvalidGeminiResponse("unexpected Gemini response format")
        return payload

    def call_json(
        self,
        prompt: str,
        stage: str,
        validator: Callable[[dict], None],
        max_attempts: int = 3,
        progress: Callable[[str], None] | None = None,
    ) -> GroundedResult:
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            if progress:
                suffix = f" (attempt {attempt}/{max_attempts})" if attempt > 1 else ""
                progress(f"Gemini {stage}{suffix}…")
            try:
                payload = self._request(prompt)
                data = extract_json(_response_text(payload))
                validator(data)
                sources = _grounding_sources(payload)
                if not sources:
                    raise InvalidGeminiResponse(
                        "Gemini did not return Google Search grounding evidence"
                    )
                return GroundedResult(data, sources, attempt)
            except (GeminiError, InvalidGeminiResponse) as exc:
                last_error = exc
                retryable = isinstance(exc, InvalidGeminiResponse) or getattr(
                    exc, "retryable", False
                )
                if attempt >= max_attempts or not retryable:
                    break
                if progress:
                    progress(f"{stage.capitalize()} attempt failed: {exc}; retrying")
                time.sleep(min(2 ** (attempt - 1), 4))
        raise GeminiError(f"{stage} failed after {max_attempts} attempts: {last_error}")


def merge_sources(data: dict, grounded: list[dict[str, str]]) -> dict:
    sources = data.get("sources") if isinstance(data.get("sources"), list) else []
    seen = {str(item.get("url")) for item in sources if isinstance(item, dict) and item.get("url")}
    for item in grounded:
        if item["url"] not in seen:
            sources.append(item)
            seen.add(item["url"])
    data["sources"] = sources
    return data

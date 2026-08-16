from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .gemini import GeminiClient, merge_sources, validate_report, validate_requirements
from .paths import cache_home, read_json, slugify, write_json
from .prompts import performance_prompt, requirements_prompt


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def deterministic_checks(hardware: dict, requirements: dict) -> dict:
    minimum = requirements.get("minimum") or {}
    memory = hardware.get("memory") or {}
    storage = hardware.get("storage") or {}
    gpus = hardware.get("gpus") or []
    known_vram = [_number(gpu.get("vram_mb")) for gpu in gpus if isinstance(gpu, dict)]
    known_vram = [value for value in known_vram if value is not None]
    checks = {}

    def capacity(name: str, actual, required, unit: str, scale: float = 1) -> None:
        if actual is None or required is None:
            checks[name] = {
                "status": "unknown",
                "actual": actual,
                "required": required,
                "unit": unit,
            }
        else:
            checks[name] = {
                "status": "pass" if actual >= required * scale else "fail",
                "actual": actual,
                "required": required * scale,
                "unit": unit,
            }

    capacity("ram", _number(memory.get("total_mb")), _number(minimum.get("ram_gb")), "MB", 1024)
    capacity(
        "storage", _number(storage.get("available_gb")), _number(minimum.get("storage_gb")), "GB"
    )
    capacity(
        "vram", max(known_vram) if known_vram else None, _number(minimum.get("vram_gb")), "MB", 1024
    )
    required_type = minimum.get("storage_type")
    actual_type = storage.get("type")
    if required_type == "ssd" and actual_type:
        checks["storage_type"] = {
            "status": "pass" if actual_type == "ssd" else "fail",
            "actual": actual_type,
            "required": "ssd",
        }
    else:
        checks["storage_type"] = {
            "status": "unknown" if not actual_type else "pass",
            "actual": actual_type,
            "required": required_type,
        }
    return checks


def _fingerprint(application: str, hardware: dict, resolution: str, preset: str, model: str) -> str:
    stable_hardware = json.loads(json.dumps(hardware))
    stable_hardware.pop("collected_at", None)
    if isinstance(stable_hardware.get("memory"), dict):
        stable_hardware["memory"].pop("available_mb", None)
    if isinstance(stable_hardware.get("storage"), dict):
        available = stable_hardware["storage"].get("available_gb")
        if isinstance(available, (int, float)):
            stable_hardware["storage"]["available_gb"] = int(available // 10) * 10
    payload = [application.casefold(), stable_hardware, resolution, preset, model]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _fresh_enough(data: dict, days: int = 7) -> bool:
    try:
        created = datetime.fromisoformat(str(data["generated_at"]))
    except (KeyError, TypeError, ValueError):
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - created < timedelta(days=days)


class Analyzer:
    def __init__(
        self,
        client: GeminiClient,
        progress: Callable[[str], None] | None = None,
        cache_dir: Path | None = None,
    ):
        self.client = client
        self.progress = progress
        self.cache_dir = cache_dir or cache_home()

    def run(
        self,
        application: str,
        hardware: dict,
        resolution: str = "native",
        preset: str = "auto",
        fresh: bool = False,
        max_attempts: int = 3,
    ) -> tuple[dict, bool]:
        fingerprint = _fingerprint(application, hardware, resolution, preset, self.client.model)
        report_path = self.cache_dir / "reports" / f"{slugify(application)}-{fingerprint}.json"
        cached = read_json(report_path)
        if cached and not fresh and _fresh_enough(cached):
            if self.progress:
                self.progress("Using cached analysis (use --fresh to research again)")
            return cached, True

        platform_name = f"{platform.system()} {platform.release()}"
        req_result = self.client.call_json(
            requirements_prompt(application, platform_name),
            "requirements research",
            validate_requirements,
            max_attempts,
            self.progress,
        )
        requirements = merge_sources(req_result.data, req_result.sources)
        requirements["generated_at"] = datetime.now(timezone.utc).isoformat()
        requirements["model"] = self.client.model
        requirements["successful_stage"] = 1
        requirements_path = self.cache_dir / "requirements" / f"{slugify(application)}.json"
        write_json(requirements_path, requirements)

        checks = deterministic_checks(hardware, requirements)
        report_result = self.client.call_json(
            performance_prompt(application, requirements, hardware, checks, resolution, preset),
            "performance research",
            validate_report,
            max_attempts,
            self.progress,
        )
        report = merge_sources(report_result.data, report_result.sources)
        report.update(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "application_query": application,
                "model": self.client.model,
                "successful_calls": 2,
                "attempts": {
                    "requirements": req_result.attempts,
                    "performance": report_result.attempts,
                },
                "requirements_file": str(requirements_path),
                "hardware": hardware,
                "local_checks": checks,
            }
        )
        write_json(report_path, report)
        return report, False

from canrun.analysis import Analyzer, deterministic_checks
from canrun.gemini import GroundedResult

HARDWARE = {
    "schema_version": 1,
    "collected_at": "2026-01-01T00:00:00+00:00",
    "operating_system": {"name": "Test Linux"},
    "cpu": {"model": "Example CPU"},
    "gpus": [{"model": "Example GPU", "vram_mb": 8192}],
    "memory": {"total_mb": 16384},
    "storage": {"available_gb": 100, "type": "ssd"},
    "display": {"width": 1920, "height": 1080},
}

REQUIREMENTS = {
    "application": {"name": "Example App"},
    "minimum": {"ram_gb": 8, "vram_gb": 6, "storage_gb": 50, "storage_type": "ssd"},
    "recommended": {"ram_gb": 16},
    "sources": [],
    "confidence": "high",
    "warnings": [],
}

REPORT = {
    "compatibility": "meets_recommended",
    "will_launch": "yes",
    "confidence": "high",
    "summary": "It should run well.",
    "bottleneck": "none",
    "estimated_performance": {
        "resolution": "1080p",
        "preset": "high",
        "fps_low": 50,
        "fps_typical": 60,
        "fps_high": 70,
    },
    "component_results": {
        name: {"status": "pass", "explanation": "Meets the requirement"}
        for name in ("cpu", "gpu", "vram", "ram", "storage", "os")
    },
    "recommendations": [],
    "sources": [],
    "uncertainties": [],
}


class FakeClient:
    model = "fake-model"

    def __init__(self):
        self.calls = []

    def call_json(self, prompt, stage, validator, max_attempts, progress):
        self.calls.append(stage)
        data = dict(REQUIREMENTS if "requirements" in stage else REPORT)
        validator(data)
        return GroundedResult(data, [], 1)


def test_deterministic_capacity_checks():
    checks = deterministic_checks(HARDWARE, REQUIREMENTS)
    assert checks["ram"]["status"] == "pass"
    assert checks["vram"]["status"] == "pass"
    assert checks["storage"]["status"] == "pass"
    assert checks["storage_type"]["status"] == "pass"


def test_analyzer_makes_two_successful_stages_then_uses_cache(tmp_path):
    client = FakeClient()
    analyzer = Analyzer(client, cache_dir=tmp_path)

    report, cached = analyzer.run("Example App", HARDWARE)
    again, cached_again = analyzer.run("Example App", HARDWARE)

    assert cached is False
    assert cached_again is True
    assert report["successful_calls"] == 2
    assert again["summary"] == report["summary"]
    assert client.calls == ["requirements research", "performance research"]
    assert (tmp_path / "requirements" / "example-app.json").exists()


def test_fresh_forces_both_stages(tmp_path):
    client = FakeClient()
    analyzer = Analyzer(client, cache_dir=tmp_path)
    analyzer.run("Example App", HARDWARE)
    analyzer.run("Example App", HARDWARE, fresh=True)
    assert len(client.calls) == 4

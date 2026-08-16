from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import sys
from pathlib import Path

from . import __version__
from .analysis import Analyzer
from .config import Config, ConfigError, clear_key, config_path, masked_key, set_key, set_model
from .gemini import GeminiClient, GeminiError
from .hardware import detect_hardware
from .paths import cache_home, write_json

COMMANDS = {"check", "specs", "config", "cache", "doctor"}


def _add_check_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("application", nargs="+", help="application or game name")
    parser.add_argument("--resolution", default="native", help="target resolution, e.g. 1080p")
    parser.add_argument("--preset", default="auto", help="target quality preset")
    parser.add_argument("--fresh", action="store_true", help="ignore a cached report")
    parser.add_argument("--json", action="store_true", help="print the complete JSON report")
    parser.add_argument("--output", type=Path, help="also write the report to this file")
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        metavar="N",
        help="maximum attempts per successful Gemini stage (default: 3)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="canrun", description="Check how well an application will run on this computer"
    )
    parser.add_argument("--version", action="version", version=f"canrun {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="research and evaluate an application")
    _add_check_arguments(check)

    specs = sub.add_parser("specs", help="show locally detected hardware")
    specs.add_argument("--json", action="store_true", help="print machine-readable JSON")
    specs.add_argument("--output", type=Path, help="write specifications to this file")

    config = sub.add_parser("config", help="manage API and model configuration")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    config_sub.add_parser("set-key", help="securely prompt for and store a Gemini API key")
    config_sub.add_parser("clear-key", help="remove the stored API key")
    config_sub.add_parser("show", help="show configuration with the key masked")
    model = config_sub.add_parser("set-model", help="select a Gemini model")
    model.add_argument("model")

    cache = sub.add_parser("cache", help="inspect or clear cached reports")
    cache_sub = cache.add_subparsers(dest="cache_command", required=True)
    cache_sub.add_parser("path", help="print the cache directory")
    cache_sub.add_parser("clear", help="remove cached specifications and reports")

    sub.add_parser("doctor", help="check configuration and hardware detection tools")
    return parser


def _value(value) -> str:
    return "unknown" if value is None or value == "" else str(value)


def print_specs(specs: dict) -> None:
    operating_system = specs.get("operating_system") or {}
    cpu = specs.get("cpu") or {}
    memory = specs.get("memory") or {}
    storage = specs.get("storage") or {}
    display = specs.get("display") or {}
    print(
        f"OS:       {_value(operating_system.get('name'))} ({_value(operating_system.get('architecture'))})"
    )
    print(f"CPU:      {_value(cpu.get('model'))}")
    print(
        f"Cores:    {_value(cpu.get('physical_cores'))} cores / {_value(cpu.get('logical_threads'))} threads"
    )
    for index, gpu in enumerate(specs.get("gpus") or [], 1):
        vram = f"{gpu['vram_mb']} MB" if gpu.get("vram_mb") is not None else "VRAM unknown"
        label = "GPU" if index == 1 else f"GPU {index}"
        print(f"{label + ':':<9}{_value(gpu.get('model'))} ({vram})")
    print(f"RAM:      {_value(memory.get('total_mb'))} MB total")
    print(
        f"Storage:  {_value(storage.get('available_gb'))} GB free ({_value(storage.get('type'))})"
    )
    if display.get("width") and display.get("height"):
        refresh = f" at {display['refresh_rate_hz']} Hz" if display.get("refresh_rate_hz") else ""
        print(f"Display:  {display['width']}x{display['height']}{refresh}")


def _component_line(label: str, component: dict | None) -> str:
    component = component or {}
    status = str(component.get("status") or "unknown").upper()
    explanation = component.get("explanation") or "No explanation available"
    return f"{label:<10} {status:<9} {explanation}"


def print_report(report: dict, cached: bool) -> None:
    print(str(report.get("summary") or "Analysis complete."))
    print()
    print(f"Compatibility: {_value(report.get('compatibility')).replace('_', ' ')}")
    print(f"Will launch:   {_value(report.get('will_launch'))}")
    print(f"Confidence:    {_value(report.get('confidence'))}")
    print(f"Bottleneck:    {_value(report.get('bottleneck'))}")
    performance = report.get("estimated_performance") or {}
    if performance:
        low, typical, high = (
            performance.get(key) for key in ("fps_low", "fps_typical", "fps_high")
        )
        fps = "unknown"
        if typical is not None:
            fps = str(typical)
            if low is not None and high is not None:
                fps = f"{low}–{high} (typically {typical})"
        print(
            f"Estimate:      {_value(performance.get('resolution'))}, "
            f"{_value(performance.get('preset'))}, {fps} FPS"
        )
    print()
    for name in ("cpu", "gpu", "vram", "ram", "storage", "os"):
        print(_component_line(name.upper(), (report.get("component_results") or {}).get(name)))
    recommendations = report.get("recommendations") or []
    if recommendations:
        print("\nRecommendations:")
        for item in recommendations:
            print(f"  - {item}")
    uncertainties = report.get("uncertainties") or []
    if uncertainties:
        print("\nUncertainties:")
        for item in uncertainties:
            print(f"  - {item}")
    sources = [item for item in report.get("sources") or [] if isinstance(item, dict)]
    if sources:
        print("\nSources:")
        for item in sources[:8]:
            print(f"  - {_value(item.get('title'))}: {_value(item.get('url'))}")
    suffix = "cached report; 0 API calls" if cached else "2 successful Gemini calls"
    print(f"\nResult: {suffix}")
    if report.get("requirements_file"):
        print(f"Requirements: {report['requirements_file']}")


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _read_key() -> str:
    if sys.stdin.isatty():
        return getpass.getpass("Gemini API key: ")
    return sys.stdin.readline().strip()


def _remove_cache() -> int:
    path = cache_home()
    if not path.exists():
        print("Cache is already empty.")
        return 0
    resolved = path.resolve()
    if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
        raise RuntimeError(f"refusing to remove unsafe cache path: {path}")
    shutil.rmtree(path)
    print(f"Removed cache: {path}")
    return 0


def _doctor() -> int:
    config = Config.load()
    system = os.name
    print(f"Can Run: {__version__}")
    print(f"Platform: {sys.platform} ({system})")
    print(f"Gemini key: {masked_key(config.api_key)}")
    print(f"Gemini model: {config.model}")
    print(f"Config: {config_path()}")
    print(f"Cache: {cache_home()}")
    tools = {
        "linux": ("lspci", "nvidia-smi", "vulkaninfo", "glxinfo", "xrandr"),
        "win32": ("powershell", "pwsh"),
        "darwin": ("sysctl", "system_profiler"),
    }.get(sys.platform, ())
    if tools:
        print("Optional/native tools:")
        for tool in tools:
            print(f"  {tool}: {'found' if shutil.which(tool) else 'not found'}")
    try:
        specs = detect_hardware()
        print(f"Hardware scan: ok ({len(specs.get('gpus') or [])} GPU(s))")
    except RuntimeError as exc:
        print(f"Hardware scan: failed ({exc})")
        return 1
    return 0 if config.api_key else 1


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and not raw[0].startswith("-") and raw[0] not in COMMANDS:
        raw.insert(0, "check")
    args = build_parser().parse_args(raw)
    try:
        if args.command == "config":
            if args.config_command == "set-key":
                path = set_key(_read_key())
                print(f"Gemini API key saved to {path}; permissions restricted where supported.")
            elif args.config_command == "clear-key":
                print(
                    "Stored API key removed." if clear_key() else "No stored API key was present."
                )
            elif args.config_command == "set-model":
                path = set_model(args.model)
                print(f"Gemini model set to {args.model} in {path}")
            else:
                config = Config.load()
                source = (
                    "GEMINI_API_KEY environment variable"
                    if os.environ.get("GEMINI_API_KEY")
                    else str(config_path())
                )
                print(f"API key: {masked_key(config.api_key)}")
                print(f"Key source: {source if config.api_key else 'none'}")
                print(f"Model: {config.model}")
            return 0
        if args.command == "specs":
            specs = detect_hardware()
            default_path = cache_home() / "hardware.json"
            write_json(default_path, specs)
            if args.output:
                write_json(args.output, specs)
            if args.json:
                print(json.dumps(specs, indent=2))
            else:
                print_specs(specs)
                print(f"\nSaved: {default_path}")
            return 0
        if args.command == "cache":
            if args.cache_command == "path":
                print(cache_home())
                return 0
            return _remove_cache()
        if args.command == "doctor":
            return _doctor()
        if args.command == "check":
            if args.attempts < 1 or args.attempts > 10:
                raise RuntimeError("--attempts must be between 1 and 10")
            application = " ".join(args.application).strip()
            config = Config.load()
            api_key = config.require_key()
            hardware = detect_hardware()
            hardware_path = write_json(cache_home() / "hardware.json", hardware)
            _progress(f"Hardware saved to {hardware_path}")
            client = GeminiClient(api_key, config.model)
            report, cached = Analyzer(client, _progress).run(
                application,
                hardware,
                args.resolution,
                args.preset,
                args.fresh,
                args.attempts,
            )
            if args.output:
                write_json(args.output, report)
            if args.json:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                print_report(report, cached)
            return 0
        raise RuntimeError("unknown command")
    except (ConfigError, GeminiError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

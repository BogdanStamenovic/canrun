from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(command: list[str], timeout: int = 8) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _os_release() -> str:
    path = Path("/etc/os-release")
    if not path.exists():
        return platform.system()
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    return values.get("PRETTY_NAME") or values.get("NAME") or platform.system()


def _linux_cpu() -> dict[str, Any]:
    text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    blocks = [block for block in text.split("\n\n") if block.strip()]
    records = []
    for block in blocks:
        item = {}
        for line in block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                item[key.strip()] = value.strip()
        records.append(item)
    first = records[0] if records else {}
    pairs = {(item.get("physical id"), item.get("core id")) for item in records}
    pairs.discard((None, None))
    flags = (first.get("flags") or first.get("Features") or "").split()
    model = first.get("model name") or first.get("Hardware") or platform.processor() or "Unknown"
    return {
        "model": model,
        "physical_cores": len(pairs) or (os.cpu_count() or 1),
        "logical_threads": len(records) or (os.cpu_count() or 1),
        "architecture": platform.machine(),
        "instruction_sets": sorted(set(flags)),
    }


def _linux_memory() -> dict[str, int | None]:
    values: dict[str, int] = {}
    text = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        match = re.match(r"(\w+):\s+(\d+)", line)
        if match:
            values[match.group(1)] = int(match.group(2)) // 1024
    return {"total_mb": values.get("MemTotal"), "available_mb": values.get("MemAvailable")}


def _nvidia_gpus() -> list[dict[str, Any]]:
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    result = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            try:
                memory = int(float(parts[1]))
            except ValueError:
                memory = None
            result.append(
                {
                    "model": parts[0],
                    "type": "dedicated",
                    "vram_mb": memory,
                    "driver": parts[2],
                }
            )
    return result


def _same_gpu(left: str, right: str) -> bool:
    if left.casefold() in right.casefold() or right.casefold() in left.casefold():
        return True
    token_pattern = re.compile(r"[a-z]*\d+[a-z0-9]*", re.IGNORECASE)
    left_tokens = {token.casefold() for token in token_pattern.findall(left)}
    right_tokens = {token.casefold() for token in token_pattern.findall(right)}
    return bool(left_tokens & right_tokens)


def _linux_gpus() -> list[dict[str, Any]]:
    gpus = _nvidia_gpus()
    output = _run(["lspci", "-nnk"])
    for line in output.splitlines():
        if not re.search(
            r"(VGA compatible controller|3D controller|Display controller)", line, re.IGNORECASE
        ):
            continue
        model = line.split(": ", 1)[-1].strip()
        if not any(_same_gpu(gpu["model"], model) for gpu in gpus):
            integrated = bool(re.search(r"Intel|integrated|APU", model, re.IGNORECASE))
            gpus.append(
                {
                    "model": model,
                    "type": "integrated" if integrated else "unknown",
                    "vram_mb": None,
                    "driver": None,
                }
            )
    if not gpus:
        gpus.append({"model": "Unknown", "type": "unknown", "vram_mb": None, "driver": None})

    vulkan = _run(["vulkaninfo", "--summary"])
    match = re.search(r"apiVersion\s*=\s*([^\s]+)", vulkan)
    vulkan_version = match.group(1) if match else None
    gl = _run(["glxinfo", "-B"])
    match = re.search(r"OpenGL core profile version string:\s*([^\n]+)", gl)
    opengl_version = match.group(1).strip() if match else None
    for gpu in gpus:
        gpu["vulkan_version"] = vulkan_version
        gpu["opengl_version"] = opengl_version
    return gpus


def _linux_display() -> dict[str, int | None]:
    output = _run(["xrandr", "--current"])
    match = re.search(r"current\s+(\d+)\s+x\s+(\d+)", output)
    active = re.search(r"^\s*(\d+)x(\d+)\s+([0-9.]+)\*", output, re.MULTILINE)
    width = int(match.group(1)) if match else int(active.group(1)) if active else None
    height = int(match.group(2)) if match else int(active.group(2)) if active else None
    refresh = round(float(active.group(3))) if active else None
    return {"width": width, "height": height, "refresh_rate_hz": refresh}


def _storage(path: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
        total = round(usage.total / 1024**3, 1)
        available = round(usage.free / 1024**3, 1)
    except OSError:
        total = available = None
    kind = None
    if platform.system() == "Linux":
        source = _run(["findmnt", "-no", "SOURCE", str(path)])
        if source:
            parent = _run(["lsblk", "-ndo", "PKNAME", source]) or source
            rota = _run(["lsblk", "-ndo", "ROTA", parent])
            kind = "hdd" if rota == "1" else "ssd" if rota == "0" else None
    return {"total_gb": total, "available_gb": available, "type": kind}


def _powershell_json(script: str) -> Any:
    executable = shutil.which("powershell") or shutil.which("pwsh")
    if not executable:
        return None
    output = _run([executable, "-NoProfile", "-NonInteractive", "-Command", script], timeout=15)
    if not output:
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _windows_specs(scan_path: Path) -> dict[str, Any]:
    cpu_items = _as_list(
        _powershell_json(
            "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"
        )
    )
    gpu_items = _as_list(
        _powershell_json(
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress"
        )
    )
    os_item = (
        _powershell_json(
            "Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json -Compress"
        )
        or {}
    )
    display = (
        _powershell_json(
            "Get-CimInstance Win32_VideoController | Select-Object -First 1 CurrentHorizontalResolution,CurrentVerticalResolution,CurrentRefreshRate | ConvertTo-Json -Compress"
        )
        or {}
    )
    first_cpu = cpu_items[0] if cpu_items else {}
    total_kb = os_item.get("TotalVisibleMemorySize")
    free_kb = os_item.get("FreePhysicalMemory")
    normalized_gpus = [
        {
            "model": item.get("Name") or "Unknown",
            "type": "unknown",
            "vram_mb": round(item["AdapterRAM"] / 1024**2)
            if isinstance(item.get("AdapterRAM"), (int, float))
            else None,
            "driver": item.get("DriverVersion"),
            "directx_version": None,
        }
        for item in gpu_items
    ]
    # AdapterRAM is a 32-bit CIM field and is unreliable above 4 GB. Prefer the
    # vendor utility for NVIDIA cards when it is installed.
    for nvidia in _nvidia_gpus():
        numbered_tokens = [
            token
            for token in nvidia["model"].casefold().split()
            if any(char.isdigit() for char in token)
        ]
        match = next(
            (
                item
                for item in normalized_gpus
                if "nvidia" in item["model"].casefold()
                and any(token in item["model"].casefold() for token in numbered_tokens)
            ),
            None,
        )
        if match:
            match.update(nvidia)
        else:
            normalized_gpus.append(nvidia)
    return {
        "operating_system": {
            "name": os_item.get("Caption") or platform.platform(),
            "version": os_item.get("Version") or platform.version(),
            "architecture": platform.machine(),
        },
        "cpu": {
            "model": first_cpu.get("Name") or platform.processor() or "Unknown",
            "physical_cores": first_cpu.get("NumberOfCores") or os.cpu_count(),
            "logical_threads": first_cpu.get("NumberOfLogicalProcessors") or os.cpu_count(),
            "architecture": platform.machine(),
            "instruction_sets": [],
        },
        "gpus": normalized_gpus
        or [{"model": "Unknown", "type": "unknown", "vram_mb": None, "driver": None}],
        "memory": {
            "total_mb": round(total_kb / 1024) if isinstance(total_kb, (int, float)) else None,
            "available_mb": round(free_kb / 1024) if isinstance(free_kb, (int, float)) else None,
        },
        "storage": _storage(scan_path),
        "display": {
            "width": display.get("CurrentHorizontalResolution"),
            "height": display.get("CurrentVerticalResolution"),
            "refresh_rate_hz": display.get("CurrentRefreshRate"),
        },
    }


def _mac_specs(scan_path: Path) -> dict[str, Any]:
    cpu_model = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or _run(
        ["sysctl", "-n", "hw.model"]
    )
    physical = _run(["sysctl", "-n", "hw.physicalcpu"])
    logical = _run(["sysctl", "-n", "hw.logicalcpu"])
    memory_bytes = _run(["sysctl", "-n", "hw.memsize"])
    profile_text = _run(
        ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"], timeout=20
    )
    profile: dict = {}
    try:
        profile = json.loads(profile_text) if profile_text else {}
    except json.JSONDecodeError:
        pass
    hardware_rows = profile.get("SPHardwareDataType", [])
    hardware_info = hardware_rows[0] if hardware_rows else {}
    cpu_model = (
        cpu_model
        or hardware_info.get("chip_type")
        or hardware_info.get("cpu_type")
        or platform.processor()
        or "Unknown"
    )
    gpu_rows = profile.get("SPDisplaysDataType", [])
    gpus = []
    display = {"width": None, "height": None, "refresh_rate_hz": None}
    for row in gpu_rows:
        vram_text = str(row.get("spdisplays_vram") or row.get("spdisplays_vram_shared") or "")
        match = re.search(r"([0-9.]+)\s*(GB|MB)", vram_text, re.IGNORECASE)
        vram = None
        if match:
            amount = float(match.group(1))
            vram = round(amount * 1024) if match.group(2).upper() == "GB" else round(amount)
        gpus.append(
            {
                "model": row.get("sppci_model") or row.get("_name") or "Unknown",
                "type": "integrated" if "shared" in vram_text.casefold() else "unknown",
                "vram_mb": vram,
                "metal_support": row.get("spdisplays_metal"),
            }
        )
        for screen in row.get("spdisplays_ndrvs", []):
            resolution = str(screen.get("_spdisplays_resolution") or "")
            res_match = re.search(r"(\d+)\s*x\s*(\d+)", resolution)
            if res_match:
                display["width"], display["height"] = map(int, res_match.groups())
                break
    try:
        total_mb = round(int(memory_bytes) / 1024**2)
    except ValueError:
        total_mb = None
    return {
        "operating_system": {
            "name": f"macOS {platform.mac_ver()[0]}",
            "version": platform.mac_ver()[0],
            "architecture": platform.machine(),
        },
        "cpu": {
            "model": cpu_model,
            "physical_cores": int(physical) if physical.isdigit() else os.cpu_count(),
            "logical_threads": int(logical) if logical.isdigit() else os.cpu_count(),
            "architecture": platform.machine(),
            "instruction_sets": [],
        },
        "gpus": gpus or [{"model": "Unknown", "type": "unknown", "vram_mb": None}],
        "memory": {"total_mb": total_mb, "available_mb": None},
        "storage": _storage(scan_path),
        "display": display,
    }


def detect_hardware(scan_path: Path | None = None) -> dict[str, Any]:
    target = (scan_path or Path.cwd()).resolve()
    system = platform.system()
    if system == "Windows":
        data = _windows_specs(target)
    elif system == "Darwin":
        data = _mac_specs(target)
    elif system == "Linux":
        data = {
            "operating_system": {
                "name": _os_release(),
                "kernel": platform.release(),
                "architecture": platform.machine(),
            },
            "cpu": _linux_cpu(),
            "gpus": _linux_gpus(),
            "memory": _linux_memory(),
            "storage": _storage(target),
            "display": _linux_display(),
        }
    else:
        raise RuntimeError(f"unsupported operating system: {system}")
    return {
        "schema_version": 1,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        **data,
    }

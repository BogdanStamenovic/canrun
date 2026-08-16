import platform
from pathlib import Path

import pytest

from canrun import hardware


def test_windows_detector_normalizes_single_cim_objects(monkeypatch, tmp_path):
    values = iter(
        [
            {"Name": "CPU", "NumberOfCores": 6, "NumberOfLogicalProcessors": 12},
            {"Name": "GPU", "AdapterRAM": 4 * 1024**3, "DriverVersion": "1.2"},
            {
                "Caption": "Windows 11",
                "Version": "10.0",
                "TotalVisibleMemorySize": 16 * 1024**2,
                "FreePhysicalMemory": 8 * 1024**2,
            },
            {
                "CurrentHorizontalResolution": 1920,
                "CurrentVerticalResolution": 1080,
                "CurrentRefreshRate": 60,
            },
        ]
    )
    monkeypatch.setattr(hardware, "_powershell_json", lambda script: next(values))
    monkeypatch.setattr(hardware, "_storage", lambda path: {"available_gb": 100})
    monkeypatch.setattr(hardware, "_nvidia_gpus", list)

    specs = hardware._windows_specs(tmp_path)

    assert specs["cpu"]["model"] == "CPU"
    assert specs["gpus"][0]["vram_mb"] == 4096
    assert specs["memory"]["total_mb"] == 16384
    assert specs["display"]["width"] == 1920


@pytest.mark.skipif(platform.system() != "Linux", reason="requires Linux /proc")
def test_current_linux_machine_can_be_scanned():
    specs = hardware.detect_hardware(Path.cwd())
    assert specs["schema_version"] == 1
    assert specs["cpu"]["model"]
    assert specs["memory"]["total_mb"] > 0
    assert specs["gpus"]


def test_mac_detector_uses_system_profiler_chip_and_gpu(monkeypatch, tmp_path):
    commands = {
        ("sysctl", "-n", "machdep.cpu.brand_string"): "",
        ("sysctl", "-n", "hw.model"): "",
        ("sysctl", "-n", "hw.physicalcpu"): "8",
        ("sysctl", "-n", "hw.logicalcpu"): "8",
        ("sysctl", "-n", "hw.memsize"): str(16 * 1024**3),
        (
            "system_profiler",
            "SPHardwareDataType",
            "SPDisplaysDataType",
            "-json",
        ): '{"SPHardwareDataType":[{"chip_type":"Apple M2"}],'
        '"SPDisplaysDataType":[{"sppci_model":"Apple M2",'
        '"spdisplays_vram_shared":"Shared",'
        '"spdisplays_ndrvs":[{"_spdisplays_resolution":"2560 x 1600"}]}]}',
    }
    monkeypatch.setattr(
        hardware, "_run", lambda command, timeout=8: commands.get(tuple(command), "")
    )
    monkeypatch.setattr(hardware, "_storage", lambda path: {"available_gb": 100})

    specs = hardware._mac_specs(tmp_path)

    assert specs["cpu"]["model"] == "Apple M2"
    assert specs["memory"]["total_mb"] == 16384
    assert specs["gpus"][0]["model"] == "Apple M2"
    assert specs["display"]["width"] == 2560


def test_gpu_alias_matching_handles_vendor_prefixes():
    assert hardware._same_gpu(
        "NVIDIA GeForce RTX 4060 Laptop GPU",
        "NVIDIA Corporation AD107M [GeForce RTX 4060 Max-Q / Mobile]",
    )

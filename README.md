# Can Run

Can Run is a local, cross-platform CLI that answers a practical question: **how well will this
application or game run on this computer?** It detects the machine locally, uses two successful
Gemini calls with Google Search grounding, saves the evidence as JSON, and prints an explainable
compatibility report. There is no Can Run server or central application database.

```console
$ canrun "Cyberpunk 2077" --resolution 1080p --preset medium
Hardware saved to /home/me/.cache/canrun/hardware.json
Gemini requirements research…
Gemini performance research…
It should run, but the GPU is below the recommended tier.

Compatibility: meets minimum
Will launch:   likely
Confidence:    medium
Bottleneck:    gpu
Estimate:      1920x1080, low, 27–38 (typically 32) FPS
```

## How it works

One fresh analysis performs two successful stages:

1. Search for current minimum and recommended requirements, prioritizing official sources.
2. Search for relevant benchmarks and evaluate the locally detected hardware.

Failed API, invalid JSON, and incomplete-schema attempts are retried up to three times by default.
That means a run always needs two *valid stage results*, but a bad attempt can result in more than
two underlying API requests and may still count against provider quota. Change the bound with
`--attempts N`.

Completed reports are cached for seven days. A cache hit makes zero calls; `--fresh` forces both
research stages again.

## Install

Can Run requires Python 3.10 or newer.

### Ownbox (Linux, macOS, and Windows)

The included `ownbox.yaml` follows the manifest contract in the sibling Ownbox project:

```console
ownbox sync
ownbox install canrun
canrun config set-key
canrun "Elden Ring"
```

Ownbox now selects the matching setup and entry commands from this repository's platform-aware
manifest. On Windows it generates `canrun.cmd`; ensure Ownbox's command directory (normally
`%LOCALAPPDATA%\ownbox\bin`) is on `PATH`.

### pipx (Linux, macOS, and Windows)

From a checkout:

```console
pipx install .
canrun config set-key
```

For development:

```console
python -m venv .venv
# Linux/macOS
.venv/bin/pip install -e '.[dev]'
# Windows PowerShell
.venv\Scripts\pip install -e ".[dev]"
```

## Configure Gemini

Create a key in Google AI Studio, then run:

```console
canrun config set-key
Gemini API key:
```

Input is hidden in an interactive terminal. The key is saved to the platform configuration folder;
on POSIX systems the file mode is set to `0600`. `GEMINI_API_KEY` overrides the stored key and is
useful for ephemeral or managed environments.

```console
canrun config show
canrun config set-model gemini-3.6-flash
canrun config clear-key
```

The default is `gemini-3.6-flash`. Can Run uses Google's Interactions API with Search grounding.
Gemini availability, quotas, and pricing can change, so the model is configurable.
See Google's [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing) and
[Search grounding](https://ai.google.dev/gemini-api/docs/google-search) documentation for current
availability and limits.

## Commands

```text
canrun "Application Name" [--resolution 1440p] [--preset high]
canrun check "Application Name"
canrun specs [--json] [--output FILE]
canrun config set-key
canrun config show
canrun config set-model MODEL
canrun cache path
canrun cache clear
canrun doctor
```

Use `--json` for automation, `--output report.json` for a copy of the report, and `--fresh` to
ignore the cached result.

## Files

Can Run writes only local configuration and cache data:

```text
config.json                       # API key and selected model
hardware.json                     # privacy-filtered local scan
requirements/<application>.json  # grounded requirements from stage 1
reports/<application>-<hash>.json # complete stage-2 report
```

On Linux these normally live under `~/.config/canrun` and `~/.cache/canrun`. macOS uses the same
XDG-compatible defaults. Windows uses `%APPDATA%\canrun` and `%LOCALAPPDATA%\canrun`.

The scan includes the OS, CPU, core/thread counts, GPU, available VRAM when detectable, RAM,
available storage, display resolution, drivers, and graphics APIs when tools expose them. It does
not collect usernames, hostnames, serial numbers, network addresses, file lists, environment
variables, or running processes.

Detection uses `/proc` and common graphics utilities on Linux, PowerShell/CIM on Windows, and
`sysctl` plus `system_profiler` on macOS. Missing optional tools produce unknown fields rather than
preventing an analysis.

## Accuracy

RAM, VRAM, free storage, and storage-type comparisons are performed locally and passed to Gemini as
deterministic checks. CPU/GPU equivalence and FPS still depend on published requirements and
benchmarks. Estimates are ranges, not guarantees; laptop power limits, drivers, cooling, game
versions, compatibility layers, and scene complexity can materially change performance.

## Development

```console
pytest
ruff check .
python -m build
```

Tests mock Gemini; they do not require a key or consume API quota.

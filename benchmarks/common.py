"""Shared helpers for the Phase 0b benchmarks."""
from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

# WINDOWS DLL NOTE: sherpa_onnx (needs sherpa-onnx-core) and the onnxruntime wheel each ship an
# "onnxruntime.dll" of a different version, and C:\Windows\System32 has an old third one (1.17.1).
# Whichever loads first in a process wins by name, so DO NOT mix sherpa_onnx and onnxruntime-based
# engines (Kokoro, Moonshine, faster-whisper) in one process. The benchmarks run one engine family
# per process; the voice pipeline will need the same split (see docs/PHASE0B_REPORT.md).

ROOT =Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RESULTS = ROOT / "benchmarks" / "results"
RECORDINGS = ROOT / "benchmarks" / "recordings"
HOTWORDS = ROOT / "config" / "hotwords.txt"


def now() -> float:
    return time.perf_counter()


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "median": round(statistics.median(values), 1) if values else None,
        "p95": round(pct(values, 95), 1) if values else None,
        "min": round(min(values), 1) if values else None,
    }


def load_wav_16k(path: Path) -> np.ndarray:
    """Read a wav, mix to mono, resample to 16 kHz float32."""
    import soundfile as sf
    import soxr

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != 16000:
        mono = soxr.resample(mono, sr, 16000)
    return mono.astype(np.float32)


def hotword_terms() -> list[str]:
    return [ln.strip() for ln in HOTWORDS.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def system_info() -> dict:
    info = {
        "machine": platform.platform(),
        "python": platform.python_version(),
        "cpu_logical": os.cpu_count(),
    }
    try:
        info["power_scheme"] = subprocess.run(
            ["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        batt = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Battery | Select-Object -First 1 BatteryStatus,EstimatedChargeRemaining | ConvertTo-Json -Compress)"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        info["battery"] = batt or "no battery info"
    except Exception as e:  # informational only
        info["sysinfo_error"] = str(e)
    return info


def save_result(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"{name}.json"
    payload = {"system": system_info(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p

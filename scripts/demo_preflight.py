r"""Pre-demo checklist: is this machine in a state where the demo will run smoothly? Prints OK / WARN / FAIL per item.

  .venv\Scripts\python.exe scripts\demo_preflight.py          (or double-click demo_check.cmd)

It only READS: it never stops or changes anything.
"""
from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
results: list[tuple[str, str, str]] = []


def add(level: str, what: str, detail: str = "") -> None:
    results.append((level, what, detail))
    print(f"[{level:4s}] {what}" + (f"  -  {detail}" if detail else ""), flush=True)


def main() -> int:
    # 1. power
    b = psutil.sensors_battery()
    if b is None:
        add("OK", "power", "no battery (desktop)")
    elif b.power_plugged:
        add("OK", "AC power connected", f"battery {b.percent:.0f}%")
    else:
        add("FAIL", "ON BATTERY", "plug in AC and set Windows power mode to Best performance: battery cuts sustained speed to ~30%")

    # 2. competing jobs (the iGPU shares the package power budget with the CPU)
    me = psutil.Process().pid
    heavy = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["name"] and p.info["name"].lower().startswith("python") and p.info["pid"] != me:
                cmd = " ".join(p.info["cmdline"] or [])
                if "dealer_core.labeling" in cmd:          # the labeling page is idle; it is reported separately below
                    continue
                if any(k in cmd for k in ("benchmarks", "bench_", "score_", "optimum", "compare", "dealer_core.cli", "run_llm")):
                    heavy.append((p.info["pid"], cmd[:110]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if heavy:
        add("WARN", f"{len(heavy)} benchmark/conversion job(s) running", "; ".join(f"pid {pid}: {c}" for pid, c in heavy)
            + "  (they steal CPU and power from the demo: let them finish or stop them)")
    else:
        add("OK", "no benchmark or conversion jobs running")
    load = psutil.cpu_percent(interval=2.0)
    add("OK" if load < 25 else "WARN", f"CPU load {load:.0f}% (idle machine should be under ~20%)")

    # 3. model files and devices
    cfg_model = "Qwen3-4B-Instruct-2507-int4-ov-nocal"
    mdir = ROOT / "models" / "ov" / cfg_model
    add("OK" if (mdir / "openvino_model.xml").exists() else "FAIL", f"model files {cfg_model}", str(mdir))
    try:
        import openvino as ov

        devs = ov.Core().available_devices
        add("OK" if "GPU" in devs else "FAIL", "OpenVINO devices", ", ".join(devs) + ("" if "GPU" in devs else "  (no iGPU: use --backend ollama)"))
    except Exception as e:                                           # pragma: no cover - environment dependent
        add("FAIL", "OpenVINO", str(e))

    # 4. the fallback (Ollama) is reachable and has the model
    try:
        import ollama

        names = {m.model for m in ollama.Client(host="http://127.0.0.1:11434", timeout=3).list().models}
        want = "qwen3:4b-instruct-2507-q4_K_M"
        add("OK" if want in names else "WARN", "Ollama fallback", f"{want} {'present' if want in names else 'NOT pulled (fallback unavailable)'}")
    except Exception as e:
        add("WARN", "Ollama fallback not reachable", f"{type(e).__name__}: demo still works on the iGPU; no fallback")

    # 5. leftovers
    s = socket.socket()
    s.settimeout(0.5)
    if s.connect_ex(("127.0.0.1", 8765)) == 0:
        add("WARN", "the labeling helper is still running on port 8765", "harmless (idle) but stop it if you want a quiet machine")
    s.close()
    free_gb = psutil.virtual_memory().available / 1e9
    add("OK" if free_gb > 8 else "WARN", f"free memory {free_gb:.1f} GB")

    fails = [r for r in results if r[0] == "FAIL"]
    warns = [r for r in results if r[0] == "WARN"]
    print(f"\n{'NOT READY' if fails else 'READY' if not warns else 'READY WITH WARNINGS'}: {len(fails)} fail, {len(warns)} warning(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

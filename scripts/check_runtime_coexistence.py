"""Which native runtimes can share ONE process on this machine? Runs every ordering in a fresh subprocess, with a real smoke test
per runtime (not just an import):
  ov     : compile an OpenVINO model (whisper-tiny encoder) on CPU
  ort    : create an onnxruntime InferenceSession on a real .onnx file
  sherpa : build a sherpa-onnx Piper TTS and synthesise one word
Result on this machine (sherpa-onnx 1.13.8 + sherpa-onnx-core, onnxruntime 1.24.4, openvino 2026.4.1): all orderings work. An earlier belief that sherpa
and onnxruntime clash was caused by sherpa-onnx-core being absent; re-run this after any runtime upgrade.

  uv run python scripts/check_runtime_coexistence.py
"""
from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = {
    "ov": f"""
import openvino as ov
core = ov.Core()
m = core.read_model(r"{ROOT / 'models/ov/whisper-tiny-int8-ov/openvino_encoder_model.xml'}")
core.compile_model(m, "CPU")
""",
    "ort": f"""
import onnxruntime as ort
ort.InferenceSession(r"{ROOT / 'models/kokoro-v1.0/kokoro-v1.0.int8.onnx'}", providers=["CPUExecutionProvider"])
""",
    "sherpa": f"""
import pathlib, sherpa_onnx
d = next(p for p in pathlib.Path(r"{ROOT / 'models/vits-piper-en_US-joe-medium'}").iterdir() if p.is_dir())
cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(vits=sherpa_onnx.OfflineTtsVitsModelConfig(
    model=str(next(d.glob("*.onnx"))), tokens=str(d / "tokens.txt"), data_dir=str(d / "espeak-ng-data")), num_threads=2))
a = sherpa_onnx.OfflineTts(cfg).generate("hello", sid=0, speed=1.0)
assert len(a.samples) > 1000
""",
}


def run(order: tuple[str, ...]) -> tuple[bool, str]:
    code = "\n".join(SMOKE[k] for k in order) + "\nprint('ok')"
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    if p.returncode == 0 and p.stdout.strip().endswith("ok"):
        return True, ""
    err = (p.stderr or p.stdout).strip().splitlines()
    return False, (err[-1] if err else f"exit {p.returncode}")[:150]


def main() -> int:
    rows = []
    for n in (2, 3):
        for order in itertools.permutations(SMOKE, n):
            ok, why = run(order)
            rows.append((" -> ".join(order), ok, why))
            print(f"{' -> '.join(order):24s} {'OK  ' if ok else 'FAIL'} {why}", flush=True)
    bad = [r for r in rows if not r[1]]
    print(f"\n{len(rows) - len(bad)} of {len(rows)} orderings work in one process.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

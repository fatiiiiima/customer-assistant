"""Parakeet-TDT 0.6B v2 (NVIDIA, English) on the Mac's GPU (MLX), the whole clip after the end of speech. No word list or hint: this model takes none. docs/ARCH_REVIEW.md section 85.

  uv venv benchmarks/results/asr_shootout/qwen3-venv   # the private MLX environment of scripts/qwen3_asr_server.py (mlx-audio, soundfile)
  benchmarks/results/asr_shootout/qwen3-venv/bin/python scripts/asr_shootout/run_parakeet.py [--wavs DIR --out FILE]

Default: the 252 turns that prepare.py saved, written to res_parakeet.json (score.py reads it). With --wavs, every WAV in DIR (a call's stretches of speech, say) is read instead.
The weights (mlx-community/parakeet-tdt-0.6b-v2, 2.47 GB, CC-BY-4.0) are fetched by Hugging Face on first use. Time = the transcription, all of it after the end of speech.
"""
import argparse
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.stt import load

ROOT_ = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("ASR_WORK", ROOT_ / "benchmarks" / "results" / "asr_shootout"))
REPO = "mlx-community/parakeet-tdt-0.6b-v2"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wavs", type=Path, help="read every WAV in this folder instead of the shoot-out's turns")
    ap.add_argument("--out", type=Path, help="where to write the results (default: res_parakeet.json in the work folder)")
    a = ap.parse_args()
    model = load(REPO)

    def hear(x: np.ndarray) -> tuple[str, float]:
        t = time.perf_counter()
        text = str(model.generate(mx.array(x.astype(np.float32))).text).strip()
        return text, (time.perf_counter() - t) * 1000

    hear(sf.read(next((a.wavs or WORK / "turns").glob("*.wav")), dtype="float32")[0])           # warm-up: compile, so the first clip does not pay for it
    files = sorted((a.wavs or WORK / "turns").glob("*.wav"))
    out = []
    for f in files:
        x, rate = sf.read(f, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        assert rate == 16000, f"{f.name}: expects 16 kHz"
        text, ms = hear(x)
        out.append({"i": int(f.stem) if f.stem.isdigit() else f.stem, "text": text, "after_end_ms": round(ms)})
    dest = a.out or WORK / "res_parakeet.json"
    dest.write_text(json.dumps(out, indent=1))
    ms = sorted(r["after_end_ms"] for r in out)
    print(f"parakeet done: {len(out)} clips, median {ms[len(ms) // 2]} ms, p90 {ms[int(len(ms) * 0.9)]} ms -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

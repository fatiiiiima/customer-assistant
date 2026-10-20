"""Make SYNTHETIC test clips from script.tsv with sherpa-onnx Piper (lessac-medium).

Interim stand-in until real recordings exist. Synthetic speech is clean and unaccented, so
WER on it is optimistic: use it for speed and as a smoke test, never as the accuracy verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import MODELS, RESULTS  # noqa: E402  (must come before sherpa_onnx: see common.py)

import numpy as np  # noqa: E402
import sherpa_onnx  # noqa: E402
import soundfile as sf  # noqa: E402
import soxr  # noqa: E402


def main() -> None:
    d = MODELS / "vits-piper-en_US-lessac-medium"
    sub = next(p for p in d.iterdir() if p.is_dir())
    cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(next(sub.glob("*.onnx"))), tokens=str(sub / "tokens.txt"),
            data_dir=str(sub / "espeak-ng-data")), num_threads=4, provider="cpu"))
    tts = sherpa_onnx.OfflineTts(cfg)
    out = RESULTS / "synth"
    out.mkdir(parents=True, exist_ok=True)
    rows = (Path(__file__).parent / "recordings" / "script.tsv").read_text(encoding="utf-8").splitlines()[1:]
    for row in rows:
        cid, text = row.split("\t", 1)
        audio = tts.generate(text, sid=0, speed=1.0)
        x = soxr.resample(np.asarray(audio.samples, dtype=np.float32), audio.sample_rate, 16000)
        pad = np.zeros(8000, dtype=np.float32)  # 0.5 s silence each side, like the recording guide
        sf.write(out / f"{cid}.wav", np.concatenate([pad, x, pad]), 16000, subtype="PCM_16")
    print(f"wrote {len(rows)} synthetic clips to {out}")


if __name__ == "__main__":
    main()

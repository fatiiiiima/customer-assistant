"""Denoise the scripted clips with noisereduce into benchmarks/results/denoised/ (scratch, gitignored).

Run it with the ISOLATED analysis environment, never the runtime venv:
    benchmarks\.analysis-venv\Scripts\python.exe benchmarks\denoise_clips.py
(create it with: python -m uv venv benchmarks/.analysis-venv --python .venv/Scripts/python.exe
                 python -m uv pip install --python benchmarks/.analysis-venv/Scripts/python.exe numpy scipy soundfile noisereduce)
noisereduce (MIT) pulls matplotlib & friends, which is why it stays out of the runtime environment.
Setting: stationary spectral gating, library defaults (one fixed setting, no sweeps).
"""
from pathlib import Path

import noisereduce as nr
import numpy as np
import soundfile as sf

BENCH = Path(__file__).resolve().parent
OUT = BENCH / "results" / "denoised"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / ".gitignore").write_text("*\n", encoding="utf-8")
n = 0
for wav in sorted((BENCH / "recordings").glob("*.wav")):
    stem = wav.stem
    if stem.endswith(("_noisy", "_spk2", "_quiet", "_headset", "_raw")):
        continue  # only the office-noise scripted clips are used for the mitigation test
    x, sr = sf.read(str(wav), dtype="float32")
    y = nr.reduce_noise(y=x, sr=sr, stationary=True)
    sf.write(str(OUT / wav.name), np.clip(y, -1, 1), sr, subtype="PCM_16")
    n += 1
print(f"denoised {n} clips -> {OUT}")

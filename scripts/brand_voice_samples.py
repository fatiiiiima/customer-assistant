r"""Render the car brands and models in every locally installed voice, so you can LISTEN and choose.

  .venv\Scripts\python.exe scripts\brand_voice_samples.py
Writes logs/brand_voices/<voice>.wav (each name spoken once, with a pause) and prints the list of names in order.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from dealer_core.voice.tts import VOICES, PiperSherpa, pronounce  # noqa: E402

NAMES = ["Toyota", "Land Cruiser", "Prado", "Hilux", "Camry", "Nissan", "Patrol", "Altima", "Hyundai", "Santa Fe", "Kia", "Sorento", "Sportage",
         "Mitsubishi", "Pajero", "Range Rover", "Lexus", "Honda", "Accord", "Mercedes", "Ford", "Mustang", "Bentley", "Porsche", "Peugeot", "Mazda",
         "Suzuki", "Subaru", "Isuzu", "Volkswagen", "Citroen", "Chevrolet", "Jaguar", "Infiniti", "A G M C", "G C C"]
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    out = ROOT / "logs" / "brand_voices"
    out.mkdir(parents=True, exist_ok=True)
    print("order:", ", ".join(f"{i + 1}.{n}" for i, n in enumerate(NAMES)))
    for voice, folder in VOICES.items():
        if not (ROOT / "models" / folder).exists() or voice == "lessac":                 # lessac is research-only: never used
            continue
        for sp in ((1,) if voice == "libritts_r" else (0,)):
            try:
                tts = PiperSherpa(ROOT / "models", voice, speaker=sp)
            except Exception as e:
                print(f"skip {voice}: {type(e).__name__}")
                continue
            gap = np.zeros(int(0.6 * tts.sample_rate), dtype=np.float32)
            pieces = []
            for n in NAMES:
                pieces += [tts.synthesize(pronounce(n) if n not in ("A G M C", "G C C") else n), gap]
            path = out / f"{voice}{'_speaker' + str(sp) if voice == 'libritts_r' else ''}.wav"
            sf.write(path, np.concatenate(pieces), tts.sample_rate)
            print("wrote", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())

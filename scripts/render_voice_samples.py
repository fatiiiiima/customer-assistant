"""Render the same sentences in each candidate Piper voice, as WAV files you can listen to and choose from.

Runs in its own process and imports only sherpa_onnx (no onnxruntime engines): see the DLL note in README.md.
Both voices are permissively licensed (libritts_r = CC BY 4.0, joe = CC0). libritts_r is a multi-speaker model; speaker 0 is
rendered here and other speaker ids can be tried with --libritts-sid.
"""
import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

ROOT = Path(__file__).resolve().parents[1]
SENTENCES = [
    "Hello, welcome to the dealership. How can I help you today?",
    "I found two Toyota Land Cruisers available at our Dubai branch.",
    "That car has eighty seven thousand kilometres and is priced at one hundred and sixty thousand dirhams, subject to confirmation.",
    "Let me confirm: Saturday at four thirty, for Mohammed, phone zero five zero, one two three, four five six seven. Is that right?",
    "I'm sorry, I can't help with that, but I can connect you with a colleague.",
]
VOICES = {"joe": "vits-piper-en_US-joe-medium", "libritts_r": "vits-piper-en_US-libritts_r-medium"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libritts-sid", type=int, nargs="*", default=[0])
    ap.add_argument("--out", type=Path, default=ROOT / "samples" / "voices")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name, folder in VOICES.items():
        d = next(p for p in (ROOT / "models" / folder).iterdir() if p.is_dir())
        cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(next(d.glob("*.onnx"))), tokens=str(d / "tokens.txt"), data_dir=str(d / "espeak-ng-data")), num_threads=4))
        tts = sherpa_onnx.OfflineTts(cfg)
        sids = a.libritts_sid if name == "libritts_r" else [0]
        for sid in sids:
            tag = f"{name}" + (f"_speaker{sid}" if name == "libritts_r" else "")
            for i, text in enumerate(SENTENCES, 1):
                audio = tts.generate(text, sid=sid, speed=1.0)
                pcm = (np.clip(np.asarray(audio.samples, dtype=np.float32), -1, 1) * 32767).astype(np.int16)
                path = a.out / f"{tag}_{i}.wav"
                with wave.open(str(path), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(audio.sample_rate)
                    w.writeframes(pcm.tobytes())
    (a.out / "README.txt").write_text(
        "Same 5 sentences, two permissively licensed Piper voices.\n"
        "  joe_1..5.wav          en_US joe   - CC0 (no attribution needed)   <- default unless you choose otherwise\n"
        "  libritts_r_speaker0_1..5.wav  en_US libritts_r speaker 0 - CC BY 4.0 (attribution required)\n"
        "Sentences: 1 greeting, 2 car found, 3 price (subject to confirmation), 4 read-back of a booking, 5 handoff.\n", encoding="utf-8")
    print("wrote", len(list(a.out.glob("*.wav"))), "files to", a.out)


if __name__ == "__main__":
    sys.exit(main())

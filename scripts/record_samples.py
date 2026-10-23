r"""Record your own voice for the end-of-turn study (scripts/vad_study.py): 12 short sentences, each with a natural pause in the middle, two takes each.

  .venv\Scripts\python.exe scripts\record_samples.py --list-devices
  .venv\Scripts\python.exe scripts\record_samples.py --input-device 2          # the headset microphone from the list

For each sentence the script shows it; press Enter to start recording, say it, press Enter to stop. Say it the way you would on a call: where the text shows "..." make a natural
pause (a half second to a second, like thinking), then carry on. Nothing is sent anywhere. Files go to recordings/ (git-ignored) as 16 kHz mono WAV.
Use a quiet room and the headset you will really use.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SENTENCES = [
    "I'm looking for... a white SUV",
    "Can I book... a test drive for tomorrow afternoon",
    "How much is... the Land Cruiser",
    "Do you have... anything under one hundred and fifty thousand",
    "Tell me about... the second one",
    "I want to see... the BMW X5, the grey one",
    "My name is... Ahmed Khan",
    "My number is zero five zero... one two three... four five six seven",
    "Is it... a GCC spec car",
    "What about... the kilometres, how many has it done",
    "Hello... I saw a car on your website",
    "Yes... that is right",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--input-device", type=int)
    ap.add_argument("--takes", type=int, default=2)
    a = ap.parse_args()
    import pyaudio

    pa = pyaudio.PyAudio()
    if a.list_devices:
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if d["maxInputChannels"]:
                print(f"{i:3d}  {d['name']}")
        return 0
    out = ROOT / "recordings"
    out.mkdir(exist_ok=True)
    meta = {}
    rate = 16000
    for n, sentence in enumerate(SENTENCES, 1):
        for take in range(1, a.takes + 1):
            input(f"\n[{n}/{len(SENTENCES)} take {take}]  Say:  {sentence}\n   press Enter to START recording...")
            stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, input_device_index=a.input_device, frames_per_buffer=320)
            frames: list[bytes] = []
            import threading

            stop = threading.Event()
            threading.Thread(target=lambda: (input("   recording... press Enter to STOP "), stop.set()), daemon=True).start()
            while not stop.is_set():
                frames.append(stream.read(320, exception_on_overflow=False))
            stream.stop_stream()
            stream.close()
            name = f"s{n:02d}_t{take}.wav"
            audio = np.frombuffer(b"".join(frames), dtype=np.int16)
            with wave.open(str(out / name), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(audio.tobytes())
            meta[name] = {"sentence": sentence, "seconds": round(len(audio) / rate, 2), "peak": int(np.abs(audio).max())}
            print(f"   saved {name}: {len(audio) / rate:.1f} s, peak level {int(np.abs(audio).max())} (aim for 3000-20000)")
    (out / "index.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"\nDone: {len(meta)} recordings in {out}. Now run:  .venv\\Scripts\\python.exe scripts\\vad_study.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

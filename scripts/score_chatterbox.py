r"""Chatterbox Turbo (Resemble AI, through mlx-audio on the Mac's GPU) on the sentences of scripts/score_voices.py: first sound streaming and whole, the time to make
each against its length, the pauses inside, and the audio to listen to. Runs in the MLX environment (mlx-audio). docs/ARCH_REVIEW.md section 38.

  uv run python scripts/score_voices.py --texts /tmp/texts.json                     # the sentences, numbers in words, as the voice is given them
  ~/mlxenv312/bin/python scripts/score_chatterbox.py /tmp/texts.json                 # -> logs/voice_samples/compare/*_chatterbox*.wav and rows JSON
  uv run python scripts/score_voices.py --extra chatterbox=logs/voice_samples/compare/chatterbox_stream.json ...

Its built-in voice (no reference sample). Streaming: audio is sent every `--interval` seconds of speech made (mlx-audio: 25 sound tokens a second, at least 10).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from score_voices import pauses  # noqa: E402  (numpy only)


def save(path: Path, x: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("texts", type=Path)
    ap.add_argument("--model", default="mlx-community/chatterbox-turbo-8bit")
    ap.add_argument("--interval", type=float, nargs="+", default=[0.4, 1.0], help="streaming intervals to try (seconds of speech per chunk)")
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "voice_samples" / "compare")
    a = ap.parse_args()
    from mlx_audio.tts.utils import load_model

    texts = json.loads(a.texts.read_text())
    t0 = time.perf_counter()
    model = load_model(a.model)
    for _ in model.generate(text="Warming up the voice."):
        pass
    print(f"loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    for label, stream, interval in [("whole", False, 0.0)] + [(f"stream{iv}", True, iv) for iv in a.interval]:
        rows = []
        for i, text in enumerate(texts):
            t0 = time.perf_counter()
            chunks, first, rate = [], None, 24000
            for r in model.generate(text=text, stream=stream, streaming_interval=interval or 2.0):
                first = first if first is not None else (time.perf_counter() - t0) * 1000
                chunks.append(np.asarray(r.audio, dtype=np.float32).reshape(-1))
                rate = getattr(r, "sample_rate", rate) or rate
            made = (time.perf_counter() - t0) * 1000
            audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
            save(a.out / f"{i:02d}_chatterbox-{label}.wav", audio, rate)
            rows.append({"first": first, "made": made, "length": len(audio) / rate * 1000, "pauses": pauses(audio, rate), "chunks": len(chunks)})
        (a.out / f"chatterbox_{label}.json").write_text(json.dumps(rows))
        print(f"{label}: first sound median {np.median([r['first'] for r in rows]):.0f} ms, worst {max(r['first'] for r in rows):.0f} ms", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

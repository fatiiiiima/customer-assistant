r"""Kokoro (as the worker plays it) against another voice behind an OpenAI-style speech server (Magpie-TTS through soniqo's speech-server): how soon a sentence
starts, how long it takes to make against how long it is, and the pauses inside it (the owner, 29 Oct: "Kokoro has too many pauses"). docs/ARCH_REVIEW.md section 37.

  speech-server --port 8020                                                 # soniqo's toolkit (brew install speech)
  uv run python scripts/score_voices.py --server magpie=http://127.0.0.1:8020/v1

Kokoro: each sentence cut into pieces and made piece by piece, as voice/kokoro_tts.py does live (B, openings ready in advance), the pieces' audio played back to back.
A server voice: the whole sentence in one request (soniqo's server does not stream Magpie), so its first sound is when the whole sentence is ready.
Pauses: silent stretches of 120 ms or more inside the sentence. The audio is written to logs/voice_samples/compare/ to listen to.
"""
from __future__ import annotations

import argparse
import io
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from score_first_sound import SENTENCES  # noqa: E402


def pauses(x: np.ndarray, rate: int, min_ms: int = 120) -> list[int]:
    """Silent stretches (10 ms frames well below the sentence's level) of at least `min_ms`, between its first and last sound."""
    f = int(rate * 0.01)
    n = len(x) // f
    if not n:
        return []
    lvl = np.sqrt(np.mean(x[: n * f].reshape(n, f) ** 2, axis=1))
    quiet = lvl < max(0.004, 0.05 * float(lvl.max()))
    voiced = np.where(~quiet)[0]
    if not len(voiced):
        return []
    out, run = [], 0
    for q in quiet[voiced[0]: voiced[-1] + 1]:
        if q:
            run += 1
        else:
            if run * 10 >= min_ms:
                out.append(run * 10)
            run = 0
    return out


def save(path: Path, x: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def kokoro_rows(out: Path, join_ms: int = 0, tag: str = "kokoro") -> list[dict]:
    from dealer_core.config import load_config
    from dealer_core.voice.kokoro_tts import KokoroOV
    from dealer_core.voice.pipeline import common_phrases, slot_offer_openings
    from dealer_core.voice.tts import pronounce  # noqa: F401  (the same text step the worker uses, inside KokoroOV)

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    k = KokoroOV(ROOT / "models", cfg.tts.kokoro_voice, runtime=cfg.tts.kokoro_runtime, threads=cfg.tts.threads, opening_first=cfg.voice.opening_first,
                 join_silence_ms=join_ms)
    k.preload(common_phrases(cfg), first_only=slot_offer_openings(k.pieces))
    ready = dict(k._cache)
    rows = []
    for i, text in enumerate(SENTENCES):
        k._cache.clear()
        k._cache.update(ready)
        t0 = time.perf_counter()
        parts, first = [], None
        for audio in k._joined(k.pieces(text)):                        # as the worker plays it (joins trimmed when join_ms)
            parts.append(audio)
            first = first if first is not None else (time.perf_counter() - t0) * 1000
        made = (time.perf_counter() - t0) * 1000
        audio = np.concatenate(parts)
        save(out / f"{i:02d}_{tag}.wav", audio, k.sample_rate)
        rows.append({"first": first, "made": made, "length": len(audio) / k.sample_rate * 1000, "pauses": pauses(audio, k.sample_rate)})
    return rows


def server_rows(name: str, url: str, voice: str, out: Path) -> list[dict]:
    import httpx

    client = httpx.Client(timeout=120)
    client.post(f"{url}/audio/speech", json={"model": name, "input": "Warming up.", "voice": voice, "response_format": "wav"}).raise_for_status()
    rows = []
    for i, text in enumerate(SENTENCES):
        from dealer_core.voice.tts import pronounce

        t0 = time.perf_counter()
        r = client.post(f"{url}/audio/speech", json={"model": name, "input": pronounce(text), "voice": voice, "response_format": "wav"})
        r.raise_for_status()
        made = (time.perf_counter() - t0) * 1000
        with wave.open(io.BytesIO(r.content)) as w:
            rate = w.getframerate()
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        save(out / f"{i:02d}_{name}.wav", audio, rate)
        rows.append({"first": made, "made": made, "length": len(audio) / rate * 1000, "pauses": pauses(audio, rate)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", action="append", default=[], help="name=url of a speech server, e.g. magpie=http://127.0.0.1:8020/v1 (repeatable)")
    ap.add_argument("--voice", default="default", help="the voice field sent to the server (soniqo's server uses Magpie's default speaker whatever it says)")
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "voice_samples" / "compare")
    ap.add_argument("--trim", type=int, default=40, help="also Kokoro with the joins trimmed to about this many ms (0 = not)")
    ap.add_argument("--texts", type=Path, help="write the sentences as the voice is given them (numbers in words) to this JSON file, for another program, and stop")
    ap.add_argument("--extra", action="append", default=[], help="name=rows.json from another program (scripts/score_chatterbox.py), shown in the same table")
    a = ap.parse_args()
    if a.texts:
        import json

        from dealer_core.voice.tts import pronounce
        a.texts.write_text(json.dumps([pronounce(t) for t in SENTENCES], indent=1))
        return 0
    results = {"kokoro": kokoro_rows(a.out)}
    if a.trim:
        results["kokoro-trim"] = kokoro_rows(a.out, a.trim, "kokoro-trim")
    for spec in a.extra:
        import json

        name, path = spec.split("=", 1)
        results[name] = json.loads(Path(path).read_text())
    for spec in a.server:
        name, url = spec.split("=", 1)
        results[name] = server_rows(name, url.rstrip("/"), a.voice, a.out)
    print(f"{len(SENTENCES)} sentences; audio in {a.out.relative_to(ROOT)}\n")
    print(f"{'voice':10s} {'first sound: median / worst':>28s} {'made in (x its length)':>24s} {'pauses >= 120 ms':>18s} {'longest pause':>14s}")
    for name, rows in results.items():
        firsts = [r["first"] for r in rows]
        ratio = [r["made"] / r["length"] for r in rows if r["length"]]
        ps = [p for r in rows for p in r["pauses"]]
        print(f"{name:10s} {statistics.median(firsts):13.0f} / {max(firsts):5.0f} ms {statistics.median(ratio):20.2f} x {len(ps):12d} in {len(rows)} {max(ps) if ps else 0:10d} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

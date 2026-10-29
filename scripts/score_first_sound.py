r"""How soon the voice starts a sentence, A (a first piece of 4 words, or up to 7 to a comma) against B (`voice.opening_first`: a known opening rendered at start-up and
played at once, else a first piece of 3 words). Section 33.

  uv run python scripts/score_first_sound.py

The real Kokoro, loaded as the voice server loads it (the phrases and openings it renders at start-up). Each sentence is then spoken from a cold start (pieces made for an
earlier sentence are forgotten): the first sound is the time to the first piece; the pause is how long the second piece is late after the first has finished playing (the
next pieces are made while it plays). The sentences are the code's own first sentences from the owner's calls and the sample calls.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SENTENCES = ["There are 4 Toyotas in stock right now.",
             "There's the Corolla at 62,500 dirhams, the Camry at 74,500, the RAV4 at 98,500 and the Land Cruiser at 189,000.",
             "There's the Macan at 149,000 dirhams.", "There are 4 BMWs in stock right now.",
             "There's the BMW 320i at 119,000 dirhams, the BMW 530i at 189,000, the BMW X5 at 209,000 and the BMW M3 at 289,000.",
             "It's the 2019 BMW X5 in white, with 71,000 kilometres on it, at 209,000 dirhams.",
             "That's a 2016 Macan, black, 91,000 kilometres, and it's 149,000 dirhams.",
             "Here's the 2018 white Land Cruiser with 92,000 km, and it's 189,000 dirhams.",
             "The best for a family we have is the Tucson, with 5 seats and a 620-litre boot.",
             "It has a 3.0-litre V6 twin-turbo engine with 340 horsepower.", "It's accident-free and passed 146 of our 150 inspection checks.",
             "The boot holds 500 litres.", "Our prices are fixed, so the 2019 BMW X5 is 209,000 dirhams.", "It uses about 9.8 litres per hundred kilometres.",
             "Today I've got 9 in the morning or 2 in the afternoon.", "Tomorrow I can do 11 in the morning or 3 in the afternoon.",
             "So that's the 2022 CX-5 on Wednesday, 28 October at 2:00 PM, at our AGMC showroom on Sheikh Zayed Road.",
             "If you'd like something nearer 90,000 dirhams, there's the 2022 Mazda CX-5 at 89,900 dirhams and the 2022 Hyundai Tucson at 79,900 dirhams."]


def measure(opening_first: bool) -> list[tuple[float, float]]:
    from dealer_core.config import load_config
    from dealer_core.voice.kokoro_tts import KokoroOV
    from dealer_core.voice.pipeline import common_phrases, slot_offer_openings

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    k = KokoroOV(ROOT / "models", cfg.tts.kokoro_voice, runtime=cfg.tts.kokoro_runtime, threads=cfg.tts.threads, opening_first=opening_first)
    k.preload(common_phrases(cfg), first_only=slot_offer_openings(k.pieces))
    ready = dict(k._cache)                                       # what the voice has at the start of a call
    rows = []
    for text in SENTENCES:
        k._cache.clear()
        k._cache.update(ready)
        pieces = k.pieces(text)
        t0 = time.perf_counter()
        first = k._say(pieces[0])
        first_ms = (time.perf_counter() - t0) * 1000
        pause = 0.0
        if len(pieces) > 1:
            t1 = time.perf_counter()
            k._say(pieces[1])
            pause = max(0.0, (time.perf_counter() - t1) * 1000 - len(first) / k.sample_rate * 1000)
        rows.append((first_ms, pause))
        print(f"  {'B' if opening_first else 'A'}  first sound {first_ms:5.0f} ms  pause {pause:4.0f} ms  [{pieces[0]}]  {text[:44]}", flush=True)
    return rows


def main() -> int:
    out = {}
    for label, b in (("A", False), ("B", True)):
        rows = measure(b)
        firsts, pauses = [r[0] for r in rows], [r[1] for r in rows]
        out[label] = (statistics.median(firsts), max(firsts), sum(1 for f in firsts if f < 50), sum(1 for p in pauses if p > 0), max(pauses))
        print()
    for label, (med, worst, instant, paused, max_pause) in out.items():
        print(f"{label}: first sound median {med:.0f} ms, worst {worst:.0f} ms, instant in {instant} of {len(SENTENCES)}; a pause after the first piece in {paused}, "
              f"longest {max_pause:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

r"""Which recogniser hears callers best, and how fast: Whisper base and small.en (OpenVINO, CPU, as the worker runs them) against any server with the OpenAI
transcription API (Canary-Qwen on the Mac's GPU: scripts/canary_server.py). docs/ARCH_REVIEW.md section 36.

  uv run python scripts/score_recognisers.py --server canary=http://127.0.0.1:8010/v1

Two sets of turns:
  recorded   the owner's recorded calls that have reference lines (logs/audio_dump/*.ref.txt): real voices, real microphones
  synthetic  phrasings of the sweep (tests/data/phrasing_sweep.yaml) spoken by three Kokoro voices (American, British, Indian English), each heard
             clean and as a Bluetooth headset microphone sends it (telephone band, 300-3400 Hz, at 8 kHz, and the quiet between words set to exact zero,
             as in the owner's 29 Oct 16:14 call)
Each recogniser hears the same audio. Scores: words wrong (word edits / words said; '9' against 'nine' counts as wrong for every recogniser alike), turns
heard exactly, median time per turn. Whisper gets the hint it gets live for a turn with no special context; Canary-Qwen takes no hint.
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from hearing_check import lines, load_turns, wer  # noqa: E402

VOICES = ["am_michael", "bf_emma", "hf_alpha"]


BRITISH = {"color": "colour", "colors": "colours", "favorite": "favourite", "center": "centre", "kilometer": "kilometre", "kilometers": "kilometres",
           "liter": "litre", "liters": "litres", "gray": "grey"}


def read_wer(ref: str, hyp: str) -> tuple[int, int]:
    """Word edits between the two as the agent's code reads them (see --as-read)."""
    from rapidfuzz.distance import Levenshtein

    from dealer_core.nlu.extractors import tokens_of

    r = [BRITISH.get(w, w) for w in tokens_of(ref)]
    return Levenshtein.distance(r, [BRITISH.get(w, w) for w in tokens_of(hyp)]), len(r)


def phone_band(x: np.ndarray, rate: int = 16000) -> np.ndarray:
    """What a Bluetooth headset microphone sends: 8 kHz telephone band, and digital silence where it is quiet."""
    import soxr

    y = soxr.resample(soxr.resample(x, rate, 8000), 8000, rate)
    spec = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1 / rate)
    spec[(f < 300) | (f > 3400)] = 0
    y = np.fft.irfft(spec, len(y)).astype(np.float32)
    frames = y[: len(y) // 320 * 320].reshape(-1, 320)
    quiet = np.sqrt((frames ** 2).mean(1)) < 0.006
    frames[quiet] = 0
    return frames.reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", action="append", default=[], help="name=url of a transcription server, e.g. canary=http://127.0.0.1:8010/v1 (repeatable)")
    ap.add_argument("--whisper", nargs="*", default=["whisper-base-int8-ov", "whisper-small.en-int8-ov"])
    ap.add_argument("--lines", type=int, default=40, help="sweep phrasings to speak (every one is said by each voice, clean and on a headset mic)")
    ap.add_argument("--show", action="store_true", help="print every turn a recogniser got wrong")
    ap.add_argument("--as-read", action="store_true", help="compare the words as the agent's code reads them (extractors.tokens_of: '10' and 'ten', '050 ...' and 'zero five zero ...', "
                    "'RAV4' and 'RAV four' are the same), with American spellings made British: what matters to the agent, not how the words are written")
    a = ap.parse_args()

    import soxr
    import yaml

    from dealer_core.config import load_config
    from dealer_core.dialogue.states import State
    from dealer_core.voice.kokoro_tts import KokoroOV
    from dealer_core.voice.prompts import hotword_groups, stt_prompt
    from dealer_core.voice.services import clean_transcript
    from dealer_core.voice.stt import RemoteWhisper, WhisperOpenVino, hotword_terms

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    hot = hotword_terms(ROOT / cfg.stt.hotwords_file)
    from dealer_core.app import build_app

    site = ROOT.parent / "bmw-portal" / "site" / "data.js"
    app = build_app(cfg, inventory="portal", portal_data=site, reply_mode="template", classifier=type("Stub", (), {"name": "stub", "classify": lambda *x, **k: None})())
    hint = stt_prompt(State.ANSWERING, "", hotword_groups(ROOT / cfg.stt.hotwords_file), app.inventory.vocabulary())

    engines: list[tuple[str, object, bool]] = []
    for m in a.whisper:
        w = WhisperOpenVino(ROOT / "models" / "ov" / m, ROOT / "models" / "ov_cache", "CPU", hot, precision=cfg.voice.stt_precision)
        w.warm_up()
        engines.append((m.replace("-int8-ov", ""), w, True))
    for spec in a.server:
        name, url = spec.split("=", 1)
        r = RemoteWhisper(url, name, timeout_s=30.0)
        r.warm_up()
        engines.append((name, r, False))

    turns: list[tuple[str, str, np.ndarray]] = []                              # (set, what was said, 16 kHz audio)
    folder = ROOT / "logs" / "audio_dump"
    for ref in sorted(folder.glob("*.ref.txt")):
        stem = ref.name[: -len(".ref.txt")]
        clips = load_turns(stem, folder, None)
        for said, clip in zip(lines(ref), clips):
            turns.append(("recorded", said, clip))
    phrasings = [c["say"] for c in yaml.safe_load((ROOT / "tests" / "data" / "phrasing_sweep.yaml").read_text())]
    phrasings = list(dict.fromkeys(p for p in phrasings if len(p.split()) >= 2))
    pick = phrasings[:: max(1, len(phrasings) // a.lines)][: a.lines]
    for voice in VOICES:
        tts = KokoroOV(ROOT / "models", voice)
        for said in pick:
            audio = soxr.resample(np.asarray(tts.synthesize(said), dtype=np.float32), tts.sample_rate, 16000)
            turns.append((f"synthetic, clean", said, audio))
            turns.append((f"synthetic, headset mic", said, phone_band(audio)))
    print(f"{len(turns)} turns: " + ", ".join(f"{k} {v}" for k, v in sorted({s: sum(1 for t in turns if t[0] == s) for s, _, _ in turns}.items())) + "\n", flush=True)

    tally = defaultdict(lambda: [0, 0, 0, 0, []])                               # (engine, set) -> [edits, words, exact, turns, ms]
    for name, engine, hinted in engines:
        for kind, said, audio in turns:
            t0 = time.perf_counter()
            heard = clean_transcript(engine.transcribe_float(audio, hint if hinted else None))
            ms = (time.perf_counter() - t0) * 1000
            edits, n = read_wer(said, heard) if a.as_read else wer(said, heard)
            t = tally[(name, kind)]
            t[0] += edits
            t[1] += n
            t[2] += edits == 0
            t[3] += 1
            t[4].append(ms)
            if a.show and edits:
                print(f"  {name:18s} [{kind}] said {said!r}\n  {'':18s} heard {heard!r}")
        print(f"{name} done", flush=True)
    print(f"\n{'recogniser':20s} {'turns':26s} {'words wrong':>12s} {'exact':>9s} {'median ms':>10s}")
    for (name, kind), (edits, n, exact, count, ms) in sorted(tally.items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"{name:20s} {kind:26s} {100 * edits / max(1, n):11.1f}% {exact:4d}/{count:<4d} {statistics.median(ms):9.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

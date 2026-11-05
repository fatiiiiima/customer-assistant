"""Which context should the worker give Qwen3-ASR? Sends the shoot-out's 252 turns (prepare.py) to a running scripts/qwen3_asr_server.py once for each way of giving it
the car names, and scores the words wrong "as the agent's code reads them" (scripts/score_recognisers.py --as-read). docs/ARCH_REVIEW.md section 81.

  benchmarks/results/asr_shootout/qwen3-venv/bin/python scripts/qwen3_asr_server.py --port 8011 &
  uv run python scripts/asr_shootout/context_ab.py [--url http://127.0.0.1:8011/v1]

Ways: none; the stock's names as a sentence ("names"); the worker's own hint exactly as it sends it (stt_prompt, the answering state, with the agent's last line for the recorded
turns: "live"); a short list ("short"); the stock names as the model's hot-word list instead of a prompt ("hotwords"); the live hint plus the hot-words ("live+hotwords").
Then the turns that carry digits or a time, with the hint the worker uses for those states (stt_prompt in the contact and slot states).
"""
import argparse
import json
import os
import re
import statistics as st
import sys
import time
from pathlib import Path

import httpx

ROOT_ = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("ASR_WORK", ROOT_ / "benchmarks" / "results" / "asr_shootout"))   # the shoot-out's turns and results (git-ignored)
sys.path.insert(0, str(ROOT_ / "src"))
sys.path.insert(0, str(ROOT_ / "scripts"))
from hearing_check import lines
from score_recognisers import read_wer

from dealer_core.app import build_app
from dealer_core.config import load_config
from dealer_core.dialogue.states import State
from dealer_core.voice.prompts import hotword_groups, stt_prompt
from dealer_core.voice.services import clean_transcript

DIGITS = re.compile(r"\d")
TIMEY = re.compile(r"\b(?:o'clock|pm|am|tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday|morning|afternoon|evening|\d{1,2}(?::\d\d)?\s*(?:am|pm))\b", re.IGNORECASE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8011/v1")
    a = ap.parse_args()
    cfg = load_config(overlay=ROOT_ / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=ROOT_.parent / "bmw-portal" / "site" / "data.js", reply_mode="template",
                    classifier=type("Stub", (), {"name": "stub", "classify": lambda *x, **k: None})())
    groups = hotword_groups(ROOT_ / cfg.stt.hotwords_file)
    vocab = app.inventory.vocabulary()
    names = [x for mk, models in vocab.items() for x in (mk.title(), *[m.title() if not any(c.isdigit() for c in m) else m.upper() for m in models])]
    names = list(dict.fromkeys(names))[:45]
    meta = {m["i"]: m for m in json.loads((WORK / "turns.json").read_text())}
    agent, i = {}, 0                                                         # what the agent had just said before each recorded turn (it shapes the live hint)
    folder = ROOT_ / "logs" / "audio_dump"
    for ref in sorted(folder.glob("*.ref.txt")):
        stem = ref.name[: -len(".ref.txt")]
        ag = folder / f"{stem}.agent.txt"
        al = ag.read_text().splitlines() if ag.exists() else []
        for k, _ in enumerate(lines(ref)):
            agent[i] = al[k] if k < len(al) else ""
            i += 1

    def live(i, state=State.ANSWERING, last=None):
        return stt_prompt(state, agent.get(i, "") if last is None else last, groups, vocab)

    ways = {
        "none": lambda i: {},
        "names": lambda i: {"prompt": "Used-car dealership in Dubai. Makes and models: " + ", ".join(names) + "."},
        "live": lambda i: {"prompt": live(i)},
        "short": lambda i: {"prompt": ", ".join(names[:20])},
        "hotwords": lambda i: {"hotwords": ", ".join(names)},
        "live+hotwords": lambda i: {"prompt": live(i), "hotwords": ", ".join(names)},
    }
    client = httpx.Client(timeout=30.0)

    def recognise(i, extra):
        wav = (WORK / "turns" / f"{i:03d}.wav").read_bytes()
        t = time.perf_counter()
        r = client.post(f"{a.url}/audio/transcriptions", data={"model": "qwen3-asr", "language": "en", **extra}, files={"file": ("turn.wav", wav, "audio/wav")})
        r.raise_for_status()
        return clean_transcript(r.json()["text"]), (time.perf_counter() - t) * 1000

    sets = ["recorded", "synthetic, clean", "synthetic, headset mic"]
    out, texts = {}, {}
    print(f"{'way':16s} {'recorded (12)':>22s} {'clean (120)':>22s} {'headset (120)':>22s} {'all 252':>22s} {'median ms':>10s}")
    for name, make in ways.items():
        res = {i: recognise(i, make(i)) for i in meta}
        texts[name] = {i: r[0] for i, r in res.items()}
        cells = []
        for s in sets + ["all"]:
            ids = [i for i in meta if s == "all" or meta[i]["set"] == s]
            e = n = x = 0
            for i in ids:
                ed, w = read_wer(meta[i]["said"], res[i][0])
                e += ed
                n += w
                x += ed == 0
            cells.append(f"{100 * e / n:5.1f}% / {x:3d} exact")
            out[(name, s)] = (round(100 * e / n, 1), x)
        print(f"{name:16s} " + " ".join(f"{c:>22s}" for c in cells) + f" {st.median(r[1] for r in res.values()):>10.0f}")

    print("\nTurns with digits or a time, with the hint the worker uses in those states:")
    special = {
        "digits": [i for i in meta if DIGITS.search(meta[i]["said"])],
        "times": [i for i in meta if TIMEY.search(meta[i]["said"]) and not DIGITS.search(meta[i]["said"])],
    }
    hints = {"digits": ("contact", State.CONTACT, "What is your phone number?"), "times": ("slot", State.SLOT, "Which day and time suit you? I can do tomorrow at 9 AM or 2 PM.")}
    for kind, ids in special.items():
        if not ids:
            continue
        label, state, last = hints[kind]
        for name, make in (("none", lambda i: {}), ("live (answering state)", lambda i: {"prompt": live(i)}), (f"live ({label} state)", lambda i, s=state, la=last: {"prompt": live(i, s, la)})):
            e = n = x = 0
            for i in ids:
                ed, w = read_wer(meta[i]["said"], recognise(i, make(i))[0])
                e += ed
                n += w
                x += ed == 0
            print(f"  {kind:7s} ({len(ids):3d} turns) {name:28s} {100 * e / n:5.1f}% wrong, {x} exact")
    print("\nrecorded turns that the live hint still gets wrong:")
    for i in meta:
        if meta[i]["set"] == "recorded" and read_wer(meta[i]["said"], texts["live"][i])[0]:
            print(f"  said {meta[i]['said']!r}\n    none  {texts['none'][i]!r}\n    live  {texts['live'][i]!r}\n    names {texts['names'][i]!r}")
    (WORK / "context_ab.json").write_text(json.dumps({"|".join(k): v for k, v in out.items()}, indent=1))
    (WORK / "context_ab_texts.json").write_text(json.dumps({n: {str(i): t for i, t in d.items()} for n, d in texts.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

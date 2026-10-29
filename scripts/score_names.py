r"""How well is a caller's name understood in the name step, and does the common-names correction (dealer_core/nlu/names.py) help? Offline, nothing spoken by a person.

  uv run python scripts/score_names.py                          # 60 names x 6 voices x 2 ways of saying it, Whisper base
  uv run python scripts/score_names.py --model whisper-small.en-int8-ov --voices am_michael hf_alpha

Each name is spoken by Kokoro ("My name is Ahmed." and "Ahmed.", as callers answer "And your name, please?"), recognised by the real Whisper with the name step's hint,
read by the real name reader (extractors.contact), and compared with the name said, before and after the correction. Kokoro has American, British and Hindi
voices (the Hindi ones read English with an Indian accent); there is no Arabic voice, so Arab names are heard in those accents only. Synthetic speech is cleaner
than a caller on a phone: read the numbers as a comparison of before and after, not as the rate a real caller will see.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NAMES = {
    "Arab": ["Ahmed", "Mohammed", "Omar", "Ali", "Khalid", "Yousef", "Hassan", "Hussein", "Ibrahim", "Abdullah",
             "Fatima", "Aisha", "Mariam", "Layla", "Noor", "Huda", "Zainab", "Salma", "Rashid", "Saeed"],
    "English": ["James", "John", "Michael", "David", "Daniel", "Thomas", "Robert", "William", "Sarah", "Emma",
                "Olivia", "Emily", "Jessica", "Laura", "Rachel", "Hannah", "Chris", "Mark", "Paul", "Kate"],
    "Indian": ["Rahul", "Priya", "Amit", "Anjali", "Arjun", "Deepak", "Sanjay", "Vikram", "Pooja", "Neha",
               "Ravi", "Suresh", "Ramesh", "Lakshmi", "Divya", "Kiran", "Anil", "Sunil", "Krishna", "Imran"],
}
VOICES = ["am_michael", "af_heart", "bm_george", "bf_emma", "hm_omega", "hf_alpha"]
FORMS = ["My name is {}.", "{}."]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="recogniser folder under models/ov (default voice.stt_model)")
    ap.add_argument("--voices", nargs="+", default=VOICES)
    ap.add_argument("--show", action="store_true", help="print every case the correction changed, and every miss")
    ap.add_argument("--hint", choices=["step", "names"], default="step", help="step: the name step's own hint (as live); names: common names given to the recogniser as a hint (an experiment)")
    ap.add_argument("--cache", type=Path, default=None, help="keep the synthesised clips here, so every variant is measured on the same audio")
    a = ap.parse_args()

    import soxr

    from dealer_core.app import build_app
    from dealer_core.config import load_config
    from dealer_core.dialogue.states import State
    from dealer_core.nlu import extractors as ex
    from dealer_core.nlu import names as N
    from dealer_core.voice.kokoro_tts import KokoroOV
    from dealer_core.voice.prompts import hotword_groups, stt_prompt
    from dealer_core.voice.services import clean_transcript
    from dealer_core.voice.stt import WhisperOpenVino, hotword_terms

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    hotwords, groups = hotword_terms(ROOT / cfg.stt.hotwords_file), hotword_groups(ROOT / cfg.stt.hotwords_file)
    site = ROOT.parent / "bmw-portal" / "site" / "data.js"
    app = build_app(cfg, inventory="portal" if site.exists() else "demo", portal_data=site if site.exists() else None, reply_mode="template",
                    classifier=type("Stub", (), {"name": "stub", "classify": lambda *x, **k: None})())
    hint = stt_prompt(State.CONTACT, "And your name, please?", groups, app.inventory.vocabulary())
    if a.hint == "names":                                                # one sentence of names, as the agent's question would be followed by an answer
        hint = "And your name, please? " + ", ".join(n for g in ("Arab", "English", "Indian") for n in NAMES[g][:10]) + "."
    model = a.model or cfg.voice.stt_model
    stt = WhisperOpenVino(ROOT / "models" / "ov" / model, ROOT / "models" / "ov_cache", "CPU", hotwords, precision=cfg.voice.stt_precision)
    stt.warm_up()
    print(f"recogniser {model}, name-step hint {hint!r}\n")

    tally = defaultdict(lambda: [0, 0, 0, 0])            # key -> [cases, heard right, read right before, read right after]
    changed = []
    for voice in a.voices:
        tts = None
        for group, names in NAMES.items():
            for name in names:
                for form in FORMS:
                    clip = (a.cache / f"{voice}_{name}_{FORMS.index(form)}.npy") if a.cache else None
                    if clip is not None and clip.exists():
                        pcm = np.load(clip)
                    else:
                        tts = tts or KokoroOV(ROOT / "models", voice)
                        audio = tts.synthesize(form.format(name))
                        pcm = soxr.resample(np.asarray(audio, dtype=np.float32), tts.sample_rate, 16000)
                        if clip is not None:
                            clip.parent.mkdir(parents=True, exist_ok=True)
                            np.save(clip, pcm)
                    text = clean_transcript(stt.transcribe_float(pcm, hint))
                    before = ex.contact(text).name
                    after = N.correct(before)
                    heard = name.lower() in text.lower()
                    listy = sum(1 for g in NAMES.values() for n in g if n.lower() in text.lower() and n.lower() != name.lower())
                    tally[("names written that were not said",)][0] += 1
                    tally[("names written that were not said",)][1] += listy > 0
                    ok_b = bool(before) and before.split()[0].lower() == name.lower()
                    ok_a = bool(after) and after.split()[0].lower() == name.lower()
                    in_hint = name.lower() in hint.lower()
                    for key in (("all",), ("group", group), ("voice", voice), ("form", form), ("in the hint", "yes" if in_hint else "no")):
                        t = tally[key]
                        t[0] += 1
                        t[1] += heard
                        t[2] += ok_b
                        t[3] += ok_a
                    if before != after:
                        changed.append((voice, name, text, before, after, ok_a))
                    if a.show and not ok_a:
                        print(f"  miss  [{voice}] {name:10s} heard {text!r:32s} read {before!r} -> {after!r}")

    def row(label, t):
        n = max(1, t[0])
        print(f"{label:22s} {t[0]:4d}   heard right {100 * t[1] / n:5.1f}%   name read right: before {100 * t[2] / n:5.1f}%   after {100 * t[3] / n:5.1f}%")

    row("all", tally[("all",)])
    for kind in ("group", "voice", "form", "in the hint"):
        for key in sorted(k for k in tally if k[0] == kind):
            row(f"  {key[1]}", tally[key])
    t = tally[("names written that were not said",)]
    print(f"\nanother common name written that was not said: {t[1]} of {t[0]} readings")
    good = sum(1 for c in changed if c[5])
    print(f"\nthe correction changed {len(changed)} names: {good} to the right one, {len(changed) - good} to a wrong one")
    for voice, name, text, before, after, ok in changed if a.show else []:
        print(f"  {'right' if ok else 'WRONG'}  [{voice}] said {name:10s} heard {text!r:30s} {before!r} -> {after!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

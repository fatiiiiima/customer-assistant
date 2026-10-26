r"""How well does each speech recognizer hear YOUR recordings? Key words right (car, colour, name, digits, day) and time per clip.

  .venv\Scripts\python.exe scripts\study_hearing.py                    # recordings/*.wav + recordings/index.json (made with scripts/record_samples.py)

Each recording is transcribed whole (no VAD), once per model and per hint: "none", "fixed" (the old list of car words) and "question" (the hint
the live agent uses for what it just asked). A clip passes when every key word of its sentence is in the transcript.
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
import time
import wave
from pathlib import Path

import numpy as np
import soxr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KEYS = {  # sentence start -> (key words that must be heard, the state the agent would be in, what the agent just asked)
    "Can I book": (["test drive", "tomorrow", "afternoon"], "ANSWERING", "Would you like to book a test drive?"),
    "Do you have": (["under", "150000"], "INTEREST", "What is your budget?"),
    "Hello": (["saw", "car", "website"], "GREETING", "What are you looking for today?"),
    "How much": (["how much", "land cruiser"], "INTEREST", "Which car would you like to know about?"),
    "I want to see": (["bmw x5", "grey"], "INTEREST", "Which car would you like to see?"),
    "I'm looking for": (["white", "suv"], "GREETING", "What are you looking for today?"),
    "Is it": (["gcc"], "ANSWERING", "Anything else about this car?"),
    "My name is": (["ahmed", "khan"], "CONTACT", "Can I have your name?"),
    "My number is": (["0501234567"], "CONTACT", "Thanks. And your mobile number?"),
    "Tell me about": (["second one"], "INTEREST", "Which one would you like to hear about?"),
    "What about": (["kilometres", "how many"], "ANSWERING", "Anything else about this car?"),
    "Yes": (["yes", "right"], "CONFIRM", "Shall I confirm the test drive for tomorrow at four?"),
}
DIGIT = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}


def norm(text: str) -> str:
    t = text.lower().replace("kilometers", "kilometres").replace("km", "kilometres").replace("hundred and fifty thousand", "150000")
    t = re.sub(r"one 150000|1 150000", "150000", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\b(" + "|".join(DIGIT) + r")\b", lambda m: DIGIT[m.group(1)], t)
    t = re.sub(r"(?<=\d)\s+(?=\d)", "", t)                      # "050 123 4567" -> "0501234567"
    t = re.sub(r"\bx 5\b", "x5", t)
    return re.sub(r"\s+", " ", t).strip()


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        return soxr.resample(x, w.getframerate(), 16000) if w.getframerate() != 16000 else x


def main() -> int:
    from dealer_core.config import load_config
    from dealer_core.dialogue.states import State
    from dealer_core.voice.prompts import hotword_groups, stt_prompt
    from dealer_core.voice.stt import WhisperOpenVino, hotword_terms

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    from dealer_core.app import build_app

    vocab = build_app(cfg, inventory="portal", portal_data=Path("C:/dev/bmw-portal/site/data.js"), reply_mode="template").inventory.vocabulary()
    meta = json.loads((ROOT / "recordings" / "index.json").read_text(encoding="utf-8"))
    clips = {n: read_wav(ROOT / "recordings" / n) for n in sorted(meta)}
    groups = hotword_groups(ROOT / cfg.stt.hotwords_file)
    fixed = ", ".join(hotword_terms(ROOT / cfg.stt.hotwords_file)[:60])
    models = sys.argv[1:] or ["whisper-base-int8-ov", "whisper-small.en-int8-ov", "whisper-small-int8-ov"]
    out = []
    for model in models:
        name, _, device = model.partition("@")                       # "whisper-small.en-int8-ov@NPU" runs it on the NPU
        eng = WhisperOpenVino(ROOT / "models" / "ov" / name, ROOT / "models" / "ov_cache", device or "CPU")
        eng.warm_up()
        for hint in ("none", "fixed", "question"):
            passed, ms, misses = 0, [], []
            for name, x in clips.items():
                sentence = meta[name]["sentence"]
                keys, state, asked = next(v for k, v in KEYS.items() if sentence.startswith(k))
                prompt = "" if hint == "none" else fixed if hint == "fixed" else stt_prompt(State[state], asked, groups, vocab)
                t0 = time.perf_counter()
                text = eng.transcribe_float(x, prompt)
                ms.append((time.perf_counter() - t0) * 1000)
                heard = norm(text)
                ok = all(norm(k) in heard for k in keys)
                passed += ok
                if not ok:
                    misses.append(f"{sentence!r} -> {text!r}")
            row = {"model": model, "hint": hint, "passed": passed, "clips": len(clips), "ms_median": round(st.median(ms)), "ms_p90": round(float(np.percentile(ms, 90))), "misses": misses}
            out.append(row)
            print(f"{model:26s} hint={hint:9s} key words right in {passed}/{len(clips)} clips; {row['ms_median']} ms median, {row['ms_p90']} ms p90", flush=True)
            for m in misses:
                print("     miss:", m, flush=True)
    (ROOT / "logs" / "study_hearing.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

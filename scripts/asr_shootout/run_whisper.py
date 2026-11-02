"""Whisper base int8 (OpenVINO, CPU, f32) with the live hint, as the worker runs it: the whole clip after the endpoint. Time = the transcription, all of it after the end of speech."""
import os
from pathlib import Path as _P
ROOT_ = _P(__file__).resolve().parents[2]
WORK = _P(os.environ.get("ASR_WORK", ROOT_ / "benchmarks" / "results" / "asr_shootout")); WORK.mkdir(parents=True, exist_ok=True)   # audio, models and results; git-ignored
import json, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
ROOT = ROOT_; sys.path.insert(0, str(ROOT / "src"))
from dealer_core.app import build_app
from dealer_core.config import load_config
from dealer_core.dialogue.states import State
from dealer_core.voice.prompts import hotword_groups, stt_prompt
from dealer_core.voice.services import clean_transcript
from dealer_core.voice.stt import WhisperOpenVino, hotword_terms

here = WORK
cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
hot = hotword_terms(ROOT / cfg.stt.hotwords_file)
app = build_app(cfg, inventory="portal", portal_data=ROOT.parent / "bmw-portal" / "site" / "data.js", reply_mode="template", classifier=type("Stub", (), {"name": "stub", "classify": lambda *x, **k: None})())
hint = stt_prompt(State.ANSWERING, "", hotword_groups(ROOT / cfg.stt.hotwords_file), app.inventory.vocabulary())
w = WhisperOpenVino(ROOT / "models" / "ov" / "whisper-base-int8-ov", here / "ov_cache_fresh", "CPU", hot, precision=cfg.voice.stt_precision); w.warm_up()
out = []
for m in json.load(open(here / "turns.json")):
    x, _ = sf.read(here / "turns" / f"{m['i']:03d}.wav", dtype="float32")
    t = time.perf_counter(); text = clean_transcript(w.transcribe_float(x, hint)); ms = (time.perf_counter() - t) * 1000
    out.append({"i": m["i"], "text": text, "after_end_ms": round(ms)})
json.dump(out, open(here / "res_whisper.json", "w"), indent=1)
print("whisper done", len(out), "median ms", sorted(o["after_end_ms"] for o in out)[len(out) // 2])

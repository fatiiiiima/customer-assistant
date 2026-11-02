"""Moonshine streaming small (quantised, its own runtime). Same real-speed feeding in 160 ms chunks; the lag is the time stop() takes after the last chunk."""
import os
from pathlib import Path as _P
ROOT_ = _P(__file__).resolve().parents[2]
WORK = _P(os.environ.get("ASR_WORK", ROOT_ / "benchmarks" / "results" / "asr_shootout")); WORK.mkdir(parents=True, exist_ok=True)   # audio, models and results; git-ignored
import json, time
from pathlib import Path
import numpy as np, soundfile as sf
import moonshine_voice as mv
from moonshine_voice.transcriber import Transcriber
here = WORK
path, arch = mv.get_model_for_language("en", mv.ModelArch.SMALL_STREAMING)
tr = Transcriber(path, arch, update_interval=0.5)
def text_of(tx): return " ".join(l.text.strip() for l in tx.lines if l.text and l.text.strip()).strip()
def run(x, paced):
    st = tr.create_stream(update_interval=0.5); st.start(); step = 2560; t0 = time.perf_counter(); first = None
    for k in range(0, len(x), step):
        if paced: time.sleep(max(0.0, t0 + (k + step) / 16000 - time.perf_counter()))
        st.add_audio(x[k:k + step].tolist(), 16000)
    t_end = time.perf_counter()
    tx = st.stop()
    lag = (time.perf_counter() - t_end) * 1000
    st.close()
    return text_of(tx) if tx is not None else "", lag
run(sf.read(here / "turns" / "000.wav", dtype="float32")[0], False)                                  # warm-up
out = []
for m in json.load(open(here / "turns.json")):
    x, _ = sf.read(here / "turns" / f"{m['i']:03d}.wav", dtype="float32")
    text, lag = run(x, True)
    out.append({"i": m["i"], "text": text, "after_end_ms": round(lag)})
json.dump(out, open(here / "res_moonshine.json", "w"), indent=1)
print("moonshine done", len(out), "median lag ms", sorted(o["after_end_ms"] for o in out)[len(out) // 2])

"""NVIDIA Nemotron Speech Streaming 0.6B, 160 ms chunks, int8, through sherpa-onnx (CPU, 4 threads). The audio is fed at REAL speed in 160 ms chunks (so the work during speech
competes as it would live); after the last sample of the clip (the endpoint) a short silent tail is fed, the stream is closed, and the time to the final text is the lag."""
import os
from pathlib import Path as _P
ROOT_ = _P(__file__).resolve().parents[2]
WORK = _P(os.environ.get("ASR_WORK", ROOT_ / "benchmarks" / "results" / "asr_shootout")); WORK.mkdir(parents=True, exist_ok=True)   # audio, models and results; git-ignored
import json, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, sherpa_onnx
here = WORK; M = here / "models" / "nemotron160"
r = sherpa_onnx.OnlineRecognizer.from_transducer(tokens=str(M / "tokens.txt"), encoder=str(M / "encoder.int8.onnx"), decoder=str(M / "decoder.int8.onnx"), joiner=str(M / "joiner.int8.onnx"),
                                                 num_threads=4, sample_rate=16000, decoding_method="greedy_search")
def run(x, paced):
    s = r.create_stream(); step = 2560                                   # 160 ms
    t0 = time.perf_counter(); first = None
    for k in range(0, len(x), step):
        if paced: time.sleep(max(0.0, t0 + (k + step) / 16000 - time.perf_counter()))   # the chunk exists only once it has been spoken
        s.accept_waveform(16000, x[k:k + step])
        while r.is_ready(s): r.decode_stream(s)
        if first is None and r.get_result(s).strip(): first = (time.perf_counter() - t0) * 1000
    t_end = time.perf_counter()                                          # the endpoint: the last sample is in
    s.accept_waveform(16000, np.zeros(5120, dtype=np.float32)); s.input_finished()   # 320 ms of silence pads the last chunk
    while r.is_ready(s): r.decode_stream(s)
    return r.get_result(s).strip(), (time.perf_counter() - t_end) * 1000, first
sf_ = sf.read(M / "test_wavs" / "0.wav", dtype="float32")[0]; run(sf_[:16000], False)       # warm-up
out = []
for m in json.load(open(here / "turns.json")):
    x, _ = sf.read(here / "turns" / f"{m['i']:03d}.wav", dtype="float32")
    text, lag, first = run(x, True)
    out.append({"i": m["i"], "text": text, "after_end_ms": round(lag), "first_text_ms": None if first is None else round(first)})
json.dump(out, open(here / "res_nemotron.json", "w"), indent=1)
print("nemotron done", len(out), "median lag ms", sorted(o["after_end_ms"] for o in out)[len(out) // 2])

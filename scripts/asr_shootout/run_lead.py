"""The streaming recognisers again with 0.5 s of quiet in front of the speech, as a live stream has (the microphone is open before the caller speaks). Accuracy only, so not paced."""
import os
from pathlib import Path as _P
ROOT_ = _P(__file__).resolve().parents[2]
WORK = _P(os.environ.get("ASR_WORK", ROOT_ / "benchmarks" / "results" / "asr_shootout")); WORK.mkdir(parents=True, exist_ok=True)   # audio, models and results; git-ignored
import json, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
here = WORK; LEAD = np.zeros(8000, dtype=np.float32)
which = sys.argv[1]
if which == "nemotron":
    import sherpa_onnx
    M = here / "models" / "nemotron160"
    r = sherpa_onnx.OnlineRecognizer.from_transducer(tokens=str(M / "tokens.txt"), encoder=str(M / "encoder.int8.onnx"), decoder=str(M / "decoder.int8.onnx"), joiner=str(M / "joiner.int8.onnx"), num_threads=4, sample_rate=16000)
    def run(x):
        s = r.create_stream(); s.accept_waveform(16000, np.concatenate([LEAD, x, np.zeros(5120, dtype=np.float32)])); s.input_finished()
        while r.is_ready(s): r.decode_stream(s)
        return r.get_result(s).strip()
else:
    import moonshine_voice as mv
    from moonshine_voice.transcriber import Transcriber
    path, arch = mv.get_model_for_language("en", mv.ModelArch.SMALL_STREAMING); tr = Transcriber(path, arch, update_interval=0.5)
    def run(x):
        st = tr.create_stream(update_interval=0.5); st.start(); y = np.concatenate([LEAD, x])
        for k in range(0, len(y), 2560): st.add_audio(y[k:k + 2560].tolist(), 16000)
        tx = st.stop(); st.close()
        return " ".join(l.text.strip() for l in tx.lines if l.text and l.text.strip()).strip() if tx is not None else ""
out = []
for m in json.load(open(here / "turns.json")):
    x, _ = sf.read(here / "turns" / f"{m['i']:03d}.wav", dtype="float32"); out.append({"i": m["i"], "text": run(x), "after_end_ms": None})
json.dump(out, open(here / f"res_{which}_lead.json", "w"), indent=1); print(which, "lead-in done", len(out))

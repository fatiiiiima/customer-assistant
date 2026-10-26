r"""Can Kokoro-82M (Apache-2.0) speak fast enough on THIS laptop? Time to the audio of one sentence, on each way of running it.

  .venv\Scripts\python.exe scripts\bench_kokoro.py

Runtimes tried: ONNX Runtime on the CPU (what Phase 0b measured: too slow), OpenVINO on the CPU, the iGPU and the NPU (the NPU needs a
fixed input size, so the phonemes are padded to a bucket). Writes a sample WAV per runtime to logs/kokoro_*.wav for listening.
"""
from __future__ import annotations

import statistics as st
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SENTENCES = ["Sure, let me check that for you.",
             "The RAV4 is a 2021 in white, with thirty three thousand kilometres on it.",
             "Tomorrow I can do nine in the morning or two in the afternoon. Which suits you better?"]


class OVSession:
    """Looks like an onnxruntime session to kokoro-onnx, runs on OpenVINO."""

    def __init__(self, model_path: Path, device: str, bucket: int | None = None) -> None:
        import openvino as ov

        core = ov.Core()
        m = core.read_model(str(model_path))
        self.bucket = bucket
        if bucket:                                               # NPU: one fixed phoneme length, padded
            m.reshape({"tokens": [1, bucket], "style": [1, 256], "speed": [1]})
        self.c = core.compile_model(m, device, {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": str(ROOT / "models" / "ov_cache")})
        self._model_path = str(model_path)
        import onnxruntime as rt

        self._meta = rt.InferenceSession(str(model_path), providers=["CPUExecutionProvider"]).get_modelmeta()   # the vocabulary lives in the model's metadata

    def get_modelmeta(self):
        return self._meta

    def run(self, _names, inputs):
        tok = inputs["tokens"]
        if self.bucket:
            n = tok.shape[1]
            if n > self.bucket:
                raise ValueError("sentence longer than the NPU bucket")
            tok = np.pad(tok, ((0, 0), (0, self.bucket - n)))
        out = self.c({"tokens": tok, "style": inputs["style"], "speed": inputs["speed"]})
        return [next(iter(out.values()))]

    def get_inputs(self):
        class I:
            def __init__(s, name, t):
                s.name, s.type = name, t
        return [I("tokens", "tensor(int64)"), I("style", "tensor(float)"), I("speed", "tensor(float)")]

    def get_outputs(self):
        class O:
            name = "audio"
        return [O()]


def save(path: Path, audio: np.ndarray, rate: int = 24000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def main() -> int:
    import onnxruntime as rt
    from kokoro_onnx import Kokoro

    model = ROOT / "models" / "kokoro-v1.0" / "kokoro-v1.0.onnx"
    voices = ROOT / "models" / "kokoro-v1.0" / "voices-v1.0.bin"
    voice = sys.argv[1] if len(sys.argv) > 1 else "am_michael"
    runtimes = [("onnxruntime CPU", lambda: rt.InferenceSession(str(model), providers=["CPUExecutionProvider"])),
                ("openvino CPU", lambda: OVSession(model, "CPU")),
                ("openvino GPU", lambda: OVSession(model, "GPU")),
                ("openvino NPU", lambda: OVSession(model, "NPU", bucket=160))]
    for label, make in runtimes:
        try:
            k = Kokoro.from_session(make(), str(voices))
            k.create("Warm up.", voice=voice, speed=1.0, lang="en-us")
            times, audio_s = [], []
            for text in SENTENCES * 2:
                t0 = time.perf_counter()
                audio, rate = k.create(text, voice=voice, speed=1.0, lang="en-us")
                times.append((time.perf_counter() - t0) * 1000)
                audio_s.append(len(audio) / rate)
            save(ROOT / "logs" / f"kokoro_{label.replace(' ', '_')}.wav", audio, rate)
            per = [f"{t:.0f}" for t in times[:3]]
            print(f"{label:16s} ms per sentence (short / medium / long): {' / '.join(per)};  real-time factor {sum(times) / 1000 / sum(audio_s):.2f}", flush=True)
        except Exception as e:
            print(f"{label:16s} failed: {type(e).__name__}: {' | '.join(l for l in str(e).splitlines() if l.strip())[:300]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

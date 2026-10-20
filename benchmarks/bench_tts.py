"""TTS benchmark: time to first audio (TTFA) and real-time factor for Piper voices and Kokoro.

Piper voices run through sherpa-onnx (Apache-2.0 runtime). Kokoro via kokoro-onnx.
Both split text into sentences; TTFA is the time until the FIRST sentence is audible,
so short first clauses matter more than total length.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import MODELS, now, save_result, summarize  # noqa: E402

SHORT = "Sure, let me check that for you."
MEDIUM = "I found a Toyota Land Cruiser from twenty twenty one with eighty seven thousand kilometres."
TWO_SENT = ("That car is available at our Dubai branch. The price is one hundred and sixty thousand dirhams, "
            "subject to confirmation.")
TEXTS = {"short(7w)": SHORT, "medium(15w)": MEDIUM, "two_sentences": TWO_SENT}


class SherpaPiper:
    def __init__(self, dirname: str, threads: int) -> None:
        import sherpa_onnx

        d = next(p for p in (MODELS / dirname).iterdir() if p.is_dir())
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(next(d.glob("*.onnx"))), tokens=str(d / "tokens.txt"),
                    data_dir=str(d / "espeak-ng-data")),
                num_threads=threads, provider="cpu"),
            max_num_sentences=1)  # callback after each sentence
        self.tts = sherpa_onnx.OfflineTts(cfg)
        self.name = f"piper:{dirname.replace('vits-piper-', '')} (sherpa-onnx)"
        self.sr = self.tts.sample_rate

    def run(self, text: str) -> tuple[float, float, float]:
        """Return (ttfa_ms, total_ms, audio_seconds)."""
        first: list[float] = []
        t0 = now()

        def cb(samples, progress):
            if not first:
                first.append((now() - t0) * 1000)
            return 1  # continue

        audio = self.tts.generate(text, sid=0, speed=1.0, callback=cb)
        total = (now() - t0) * 1000
        return first[0] if first else total, total, len(audio.samples) / self.sr


class Kokoro:
    def __init__(self, model_file: str, threads: int, voice: str = "af_heart") -> None:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro as K

        d = MODELS / "kokoro-v1.0"
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        sess = ort.InferenceSession(str(d / model_file), sess_options=so, providers=["CPUExecutionProvider"])
        self.k = K.from_session(sess, str(d / "voices-v1.0.bin"))
        self.voice = voice
        self.name = f"kokoro:{model_file.replace('kokoro-v1.0.', '').replace('.onnx', '') or 'fp32'} voice={voice}"

    def run(self, text: str) -> tuple[float, float, float]:
        async def go():
            t0 = now()
            first = None
            secs = 0.0
            async for samples, sr in self.k.create_stream(text, voice=self.voice, speed=1.0, lang="en-us"):
                if first is None:
                    first = (now() - t0) * 1000
                secs += len(samples) / sr
            return first or 0.0, (now() - t0) * 1000, secs

        return asyncio.run(go())


def build(spec: str, threads: int):
    kind, _, arg = spec.partition(":")
    if kind == "piper":
        return SherpaPiper(arg, threads)
    if kind == "kokoro":
        return Kokoro(arg, threads)
    raise SystemExit(f"unknown tts spec {spec}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", required=True,
                    help="piper:<sherpa model dir> | kokoro:<kokoro-v1.0.int8.onnx|kokoro-v1.0.onnx>")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--tag", default="tts")
    a = ap.parse_args()
    results = []
    for spec in a.engines:
        print(f"== {spec} threads={a.threads}", flush=True)
        try:
            t0 = now()
            eng = build(spec, a.threads)
            load_s = now() - t0
            eng.run(SHORT)  # warm up
            r: dict = {"engine": eng.name, "load_s": round(load_s, 1), "texts": {}}
            for label, text in TEXTS.items():
                runs = [eng.run(text) for _ in range(a.repeat)]
                ttfa = summarize([x[0] for x in runs])
                tot = summarize([x[1] for x in runs])
                audio = np.median([x[2] for x in runs])
                r["texts"][label] = {"ttfa_ms": ttfa, "total_ms": tot, "audio_s": round(float(audio), 2),
                                     "rtf": round(tot["median"] / 1000 / float(audio), 3)}
                print(f"   {label:14s} TTFA median {ttfa['median']}ms p95 {ttfa['p95']}ms | total {tot['median']}ms "
                      f"| audio {audio:.1f}s RTF {r['texts'][label]['rtf']}")
        except Exception as e:
            r = {"engine": spec, "error": f"{type(e).__name__}: {e}"}
            print("   ERROR", r["error"])
        results.append(r)
        save_result(a.tag, {"threads": a.threads, "results": results})


if __name__ == "__main__":
    main()

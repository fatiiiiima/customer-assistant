"""Realistic SEQUENTIAL turns: STT finishes, then NLU, then TTS. Reports T0 -> T2 per turn type.

  T0  caller's audio is complete (endpointing/VAD latency is NOT included: it is unmeasured, Phase 3)
  T1  reply text is ready (STT final + NLU + policy stub)
  T2  first TTS audio chunk is available
Process layout under test (the layout proposed for Phase 1):
  main process : OpenVINO Whisper (STT) + OpenVINO Qwen3 (NLU) + rule fast path     [no sherpa / no onnxruntime]
  TTS process  : sherpa-onnx Piper, reached over a multiprocessing pipe              [owns its onnxruntime.dll]
Turn types: 'rule' (fast path answered), 'llm' (rules declined -> intent-only constrained call),
            'open' (free-form 1-2 sentence answer; T2 = first sentence's first audio).

  uv run python benchmarks/bench_turns.py --make-clips            # once, separate process (uses sherpa)
  uv run python benchmarks/bench_turns.py --llm ov:Qwen3-4B-int4-ov@GPU --stt ovw:whisper-tiny-int8-ov@CPU+hot
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import MODELS, RESULTS, now, save_result, summarize  # noqa: E402

CLIPS = RESULTS / "turns"
REPLY = "I found two Toyota Land Cruisers available at our Dubai branch."  # stand-in for a template reply
OPEN_QUESTIONS = [
    "how long is the warranty period", "what happens if I change my mind after buying",
    "is the car sold with a service contract", "do you deliver to Abu Dhabi",
    "can I return the car within a week", "tell me about your inspection process",
    "what is the difference between your two branches",
]
OPEN_SYSTEM = ("You are a friendly voice assistant at a used car dealership in the UAE. Answer the caller in one or two "
               "short spoken sentences. No lists, no markdown, no links.")


# ---------------- TTS worker process (sherpa-onnx lives ONLY here) ----------------
def tts_proc(conn, voice_dir: str, threads: int) -> None:
    import sherpa_onnx  # noqa: PLC0415

    d = next(p for p in (MODELS / voice_dir).iterdir() if p.is_dir())
    cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=str(next(d.glob("*.onnx"))), tokens=str(d / "tokens.txt"),
                                                   data_dir=str(d / "espeak-ng-data")), num_threads=threads, provider="cpu"),
        max_num_sentences=1)
    tts = sherpa_onnx.OfflineTts(cfg)
    tts.generate("warm up", sid=0, speed=1.0)
    conn.send("ready")
    while True:
        msg = conn.recv()
        if msg is None:
            break
        if msg == "ping":
            conn.send(("pong", time.perf_counter()))
            continue
        sent = []

        def cb(samples, progress):
            if not sent:
                sent.append(1)
                conn.send(("first", time.perf_counter()))
            return 1

        tts.generate(msg, sid=0, speed=1.0, callback=cb)
        conn.send(("done", time.perf_counter()))


class TtsClient:
    def __init__(self, voice: str, threads: int) -> None:
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.p = ctx.Process(target=tts_proc, args=(child, voice, threads), daemon=True)
        self.p.start()
        assert self.conn.recv() == "ready"

    def first_audio(self, text: str) -> float:
        """Send text, return perf_counter() when the first audio chunk is reported."""
        self.conn.send(text)
        kind, t = self.conn.recv()
        assert kind == "first"
        assert self.conn.recv()[0] == "done"
        return t

    def rtt_ms(self) -> float:
        t = now()
        self.conn.send("ping")
        self.conn.recv()
        return (now() - t) * 1000

    def close(self) -> None:
        self.conn.send(None)
        self.p.join(timeout=5)


# ---------------- NLU / LLM (OpenVINO) ----------------
class OvNlu:
    def __init__(self, spec: str, static: bool = False) -> None:
        import openvino_genai as ov  # noqa: PLC0415
        from nlu_cases import SYSTEM, state_schema, state_system  # noqa: PLC0415

        self.ov, self.state_schema, self.state_system = ov, state_schema, state_system
        self.static_prompt = static
        self.SYSTEM = SYSTEM
        model, _, device = spec[3:].partition("@")
        self.name = f"openvino:{model}@{device} ({'static prompt' if static else 'per-state prompt'})"
        self.pipe = ov.LLMPipeline(str(MODELS / "ov" / model), device, CACHE_DIR=str(MODELS / "ov_cache"))
        self.tok = self.pipe.get_tokenizer()
        self.cfgs = {}

    def _cfg(self, state: str):
        if state not in self.cfgs:
            c = self.ov.GenerationConfig()
            c.max_new_tokens, c.do_sample = 14, False
            sch = self.state_schema(state)
            c.structured_output_config = self.ov.StructuredOutputConfig(json_schema=json.dumps(
                {"type": "object", "properties": {"intent": sch["properties"]["intent"]}, "required": ["intent"]}))
            self.cfgs[state] = c
        return self.cfgs[state]

    def intent(self, state: str, car: str | None, text: str) -> str:
        prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": self.SYSTEM if self.static_prompt else self.state_system(state)},
             {"role": "user", "content": f"STATE: {state}\nCAR_IN_FOCUS: {car or 'none'}\nCUSTOMER: {text}"}],
            add_generation_prompt=True, extra_context={"enable_thinking": False})
        out = str(self.pipe.generate([prompt], self._cfg(state)).texts[0])
        try:
            return json.loads(out)["intent"]
        except Exception:
            return "?"

    def first_sentence(self, question: str) -> tuple[str, float]:
        """Free-form answer; stop at the end of the first sentence. Returns (text, ms to first sentence)."""
        ov = self.ov
        prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": OPEN_SYSTEM}, {"role": "user", "content": question}],
            add_generation_prompt=True, extra_context={"enable_thinking": False})
        cfg = ov.GenerationConfig()
        cfg.max_new_tokens, cfg.do_sample = 40, False
        buf: list[str] = []
        t0 = now()

        def streamer(piece: str):
            buf.append(piece)
            txt = "".join(buf)
            if re.search(r"[.!?]\s*$", txt) and len(txt.split()) >= 4:
                return ov.StreamingStatus.STOP
            return ov.StreamingStatus.RUNNING

        self.pipe.generate(prompt, cfg, streamer)
        return "".join(buf).strip(), (now() - t0) * 1000


# ---------------- clip generation (run as its own process; sherpa only) ----------------
def make_clips() -> None:
    from common import MODELS as M  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import sherpa_onnx  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415
    from scipy.signal import resample_poly  # noqa: PLC0415
    from nlu_cases import FRESH, HELDOUT  # noqa: PLC0415

    d = next(p for p in (M / "vits-piper-en_US-ljspeech-medium").iterdir() if p.is_dir())  # public-domain caller voice
    cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(vits=sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(next(d.glob("*.onnx"))), tokens=str(d / "tokens.txt"), data_dir=str(d / "espeak-ng-data")), num_threads=4))
    tts = sherpa_onnx.OfflineTts(cfg)
    CLIPS.mkdir(parents=True, exist_ok=True)
    items = [(f"h{i:02d}", c[2]) for i, c in enumerate(HELDOUT)] + [(f"f{i:02d}", c[2]) for i, c in enumerate(FRESH)] \
        + [(f"o{i}", q) for i, q in enumerate(OPEN_QUESTIONS)]
    for cid, text in items:
        a = tts.generate(text, sid=0, speed=1.0)
        x = resample_poly(np.asarray(a.samples, dtype=np.float32), 16000, a.sample_rate).astype(np.float32)
        sf.write(CLIPS / f"{cid}.wav", np.concatenate([np.zeros(8000, np.float32), x, np.zeros(8000, np.float32)]), 16000)
    print(f"wrote {len(items)} caller clips to {CLIPS}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-clips", action="store_true")
    ap.add_argument("--stt", default="ovw:whisper-tiny-int8-ov@CPU+hot")
    ap.add_argument("--llm", default="ov:Qwen3-4B-int4-ov@GPU")
    ap.add_argument("--voice", default="vits-piper-en_US-libritts_r-medium")
    ap.add_argument("--tts-threads", type=int, default=4)
    ap.add_argument("--prompt-mode", choices=["state", "static"], default="state",
                    help="static = one system prompt for every state (prefix stays cached); state enforced by grammar only")
    ap.add_argument("--tag", default="turns")
    a = ap.parse_args()
    if a.make_clips:
        return make_clips()

    from bench_stt import build  # noqa: PLC0415  (OpenVINO Whisper wrapper)
    from common import load_wav_16k  # noqa: PLC0415
    from fastpath import classify  # noqa: PLC0415
    from nlu_cases import FRESH, HELDOUT  # noqa: PLC0415

    stt = build(a.stt, 0)
    nlu = OvNlu(a.llm, static=a.prompt_mode == "static")
    tts = TtsClient(a.voice, a.tts_threads)
    # warm every compiled path once so we time steady state, not compilation
    warm = load_wav_16k(CLIPS / "h00.wav")
    stt.transcribe(warm)
    for st in ("greeting", "interest", "answering", "negotiating", "slot", "contact", "confirm"):
        nlu.intent(st, "S1042", "hello there")
    nlu.first_sentence("do you deliver")
    tts.first_audio(REPLY)
    rtt = summarize([tts.rtt_ms() for _ in range(30)])

    rows = []
    cases = [(f"h{i:02d}", c) for i, c in enumerate(HELDOUT)] + [(f"f{i:02d}", c) for i, c in enumerate(FRESH)]
    for cid, (state, car, utt, gold) in cases:
        audio = load_wav_16k(CLIPS / f"{cid}.wav")
        t0 = now()
        text, _ = stt.transcribe(audio)
        t_stt = now()
        fp = classify(state, car, text)
        if fp is not None:
            path = "rule"
        else:
            path = "llm"
            nlu.intent(state, car, text)
        t1 = now()                                   # reply text ready (policy/tools ~0 in this stub)
        t2 = tts.first_audio(REPLY)
        rows.append({"id": cid, "path": path, "stt_ms": (t_stt - t0) * 1000, "nlu_ms": (t1 - t_stt) * 1000,
                     "tts_ms": (t2 - t1) * 1000, "t0_t2_ms": (t2 - t0) * 1000})
    for i, q in enumerate(OPEN_QUESTIONS):
        audio = load_wav_16k(CLIPS / f"o{i}.wav")
        t0 = now()
        text, _ = stt.transcribe(audio)
        t_stt = now()
        sentence, llm_ms = nlu.first_sentence(text)
        t1 = now()
        t2 = tts.first_audio(sentence or REPLY)
        rows.append({"id": f"o{i}", "path": "open", "stt_ms": (t_stt - t0) * 1000, "nlu_ms": (t1 - t_stt) * 1000,
                     "tts_ms": (t2 - t1) * 1000, "t0_t2_ms": (t2 - t0) * 1000, "first_sentence": sentence})
    tts.close()

    out = {"stt": stt.name, "llm": nlu.name, "tts": a.voice, "ipc_rtt_ms": rtt, "by_path": {}}
    print(f"STT {stt.name} | NLU {nlu.name} | TTS {a.voice} (separate process, pipe RTT median {rtt['median']} ms)")
    print(f"{'turn type':10s} {'n':>3s} | {'STT':>11s} {'NLU/LLM':>13s} {'TTS 1st audio':>14s} | {'T0->T2 p50':>10s} {'p95':>6s}")
    for path in ("rule", "llm", "open"):
        rs = [r for r in rows if r["path"] == path]
        if not rs:
            continue
        s = {k: summarize([r[k] for r in rs]) for k in ("stt_ms", "nlu_ms", "tts_ms", "t0_t2_ms")}
        out["by_path"][path] = {"n": len(rs), **s}
        print(f"{path:10s} {len(rs):3d} | {s['stt_ms']['median']:5.0f}/{s['stt_ms']['p95']:<5.0f} {s['nlu_ms']['median']:6.0f}/{s['nlu_ms']['p95']:<6.0f} "
              f"{s['tts_ms']['median']:7.0f}/{s['tts_ms']['p95']:<6.0f} | {s['t0_t2_ms']['median']:10.0f} {s['t0_t2_ms']['p95']:6.0f}")
    share = sum(r["path"] == "rule" for r in rows if r["path"] != "open") / max(1, sum(r["path"] != "open" for r in rows))
    out["rule_share"] = round(share, 3)
    print(f"rule share of the {sum(r['path'] != 'open' for r in rows)} scripted turns: {share:.0%}")
    out["rows"] = rows
    save_result(a.tag, out)


if __name__ == "__main__":
    main()

r"""Where does a FULL-quality turn spend its time? Each step of a turn timed on THIS machine, with the real served model.

  uv run python scripts/trace_full_turn.py                       # the owner's 28 Oct test call and the "tell me more" call, model at gpu_server
  uv run python scripts/trace_full_turn.py --only testcall --show

A turn at the full level, in order (agent/engine.py handle_stream):
  1. turn end      the caller stops; the VAD waits for silence and Smart Turn judges the turn finished (config values, not measured here)
  2. hearing       Whisper writes the words (measured: the caller's line spoken by Kokoro, then recognised by the configured Whisper on the CPU)
  3. rules         the code's own routing (_fast_decide): when it knows the action, the model is not asked to decide
  4. decide        model call 1: which tool (and car). Prompt read ("prefill") + a few words written
  5. tool          the code runs the tool (search, car facts, slots)
  6. write         model call 2: the reply, streamed. Prompt read + words written until the first full sentence, then the rest
  7. voice         Kokoro turns the first sentence into audio (measured)
The live deadlines (decision 1.2 s, first sentence 1.5 s, whole reply 4.5 s) are LIFTED here, so the natural time of each step is seen; the summary says how many
turns would have hit them. Text only through the engine: steps 2 and 7 are timed separately on the same words. Prompt sizes are counted by asking the server.
Nothing else should use the model server while this runs (a live call on the worker would share it).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

NOW = datetime(2026, 10, 27, 11, 0, tzinfo=ZoneInfo("Asia/Dubai"))
STEPS = ["hearing", "rules", "decide", "tool", "write_first", "voice"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="gpu_server")
    ap.add_argument("--portal-data", type=Path, default=ROOT.parent / "bmw-portal" / "site" / "data.js")
    ap.add_argument("--only", default="", help="testcall, tell-me-more or family")
    ap.add_argument("--no-audio", action="store_true", help="skip hearing and voice (steps 2 and 7)")
    ap.add_argument("--show", action="store_true", help="print each turn's reply")
    ap.add_argument("--speak", action="store_true", help="after each turn, wait as long as the reply takes to say (about 2.5 words a second), as a call does: the model is idle then "
                    "(and dialogue.prefetch uses that time)")
    ap.add_argument("--no-prefetch", action="store_true", help="dialogue.prefetch off, to compare")
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "trace_full_turn.json")
    a = ap.parse_args()

    from replay_model_speaks import TELL_ME, TEST_CALL

    from dealer_core.app import build_app
    from dealer_core.cli.chat import start
    from dealer_core.config import load_config
    from dealer_core.voice.pipeline import apply_voice_profile

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=a.portal_data, engine="agent", backend=a.backend, quality="full", clock=lambda: NOW.astimezone(timezone.utc))
    start(app, quiet=True)
    apply_voice_profile(app, cfg)
    e = app.engine
    if a.no_prefetch:
        e.prefetch_on = False
    live = {"decide_s": e.decide_deadline_s, "first_s": e.reply_writer.first_deadline_s, "reply_s": e.reply_writer.deadline_s}
    e.decide_deadline_s = 15.0
    e.reply_writer.first_deadline_s, e.reply_writer.deadline_s = None, 15.0

    calls: list[dict] = []                      # every request to the model server, in order
    phase = {"now": ""}
    turn_marks: dict = {}

    import threading

    def _phase() -> str:
        return "prefetch" if threading.current_thread().name.startswith("prefetch") else (phase["now"] or "other")

    def wrap_backend(b):
        for inner in getattr(b, "backends", []):              # a chain of backends: each one that talks to a server
            wrap_backend(inner)
        if getattr(b, "_traced", False) or not hasattr(b, "_post"):
            return
        real_post, real_stream = b._post, b._stream

        def post(body, timeout):
            t0 = time.perf_counter()
            out = real_post(body, timeout)
            calls.append({"phase": _phase(), "stream": False, "body": body, "ms": (time.perf_counter() - t0) * 1000,
                          "first_ms": None, "usage": out.get("usage") or {}, "text": (out.get("choices") or [{}])[0].get("message", {}).get("content", "")})
            return out

        def stream(body, deadline_s):
            t0 = time.perf_counter()
            rec = {"phase": _phase(), "stream": True, "body": body, "ms": None, "first_ms": None, "usage": {}, "pieces": 0, "text": ""}
            calls.append(rec)
            try:
                for piece in real_stream(body, deadline_s):
                    if rec["first_ms"] is None:
                        rec["first_ms"] = (time.perf_counter() - t0) * 1000
                    rec["pieces"] += 1
                    rec["text"] += piece
                    yield piece
            finally:
                rec["ms"] = (time.perf_counter() - t0) * 1000           # also when the reader stops early (the writer closes the stream)

        b._post, b._stream, b._traced = post, stream, True

    wrap_backend(e.backend)
    wrap_backend(e.reply_writer.backend)

    def timed(name, fn, ph=None):
        def inner(*args, **kw):
            prev = phase["now"]
            if ph:
                phase["now"] = ph
            t0 = time.perf_counter()
            try:
                return fn(*args, **kw)
            finally:
                turn_marks[name] = turn_marks.get(name, 0.0) + (time.perf_counter() - t0) * 1000
                phase["now"] = prev
        return inner

    e._fast_decide = timed("rules", e._fast_decide)
    e._decide = timed("decide", e._decide, "decide")
    e._run = timed("tool", e._run)
    real_writer_stream = e.reply_writer.stream

    def writer_stream(*args, **kw):
        prev, phase["now"] = phase["now"], "write"
        try:
            yield from real_writer_stream(*args, **kw)
        finally:
            phase["now"] = prev
    e.reply_writer.stream = writer_stream

    stt = tts = None
    if not a.no_audio:
        import numpy as np
        import soxr

        from dealer_core.voice.kokoro_tts import KokoroOV
        from dealer_core.voice.stt import WhisperOpenVino, hotword_terms

        hot = hotword_terms(ROOT / cfg.stt.hotwords_file)
        stt = WhisperOpenVino(ROOT / "models" / "ov" / cfg.voice.stt_model, ROOT / "models" / "ov_cache", cfg.voice.stt_device, hot, precision=cfg.voice.stt_precision)
        stt.warm_up()
        tts = KokoroOV(ROOT / "models", cfg.tts.kokoro_voice)
        tts.synthesize("Warming up.")

        def hear(text: str) -> float:
            pcm = soxr.resample(np.asarray(tts.synthesize(text), dtype=np.float32), tts.sample_rate, 16000)
            t0 = time.perf_counter()
            stt.transcribe_float(pcm, "")
            return (time.perf_counter() - t0) * 1000

        def speak(text: str) -> float:
            return tts.first_audio_ms(text)                           # as live: the sentence is split and its short first piece is played while the rest is made

    from sample_calls import CALLS

    conversations = [("testcall", TEST_CALL), ("tell-me-more", [t for t, _ in TELL_ME]), ("family", CALLS["Family SUV, haggling, booking"])]
    conversations = [c for c in conversations if c[0].startswith(a.only)]
    print(f"machine: this Mac; model {cfg.llm.backends[a.backend].model} at {cfg.llm.backends[a.backend].host}; Whisper {cfg.voice.stt_model} on {cfg.voice.stt_device}; "
          f"Kokoro {cfg.tts.kokoro_voice}; model_speaks {'on' if e.model_speaks else 'off'}; live deadlines lifted (were decide {live['decide_s']} s, "
          f"first sentence {live['first_s']} s, reply {live['reply_s']} s)\n", flush=True)

    turns = []
    for name, lines in conversations:
        s = e.new_session()
        for text in lines:
            turn_marks.clear()
            first_call = len(calls)
            t0 = time.perf_counter()
            first_out = first_model = None
            sentences = []
            for out in e.handle_stream(s, text):
                if isinstance(out, str):
                    now = (time.perf_counter() - t0) * 1000
                    first_out = first_out if first_out is not None else now
                    if first_model is None and any(c["phase"] == "write" and c["pieces"] for c in calls[first_call:]):
                        first_model = now
                    sentences.append(out)
                else:
                    result = out
            total = (time.perf_counter() - t0) * 1000
            mine = calls[first_call:]
            dec = [c for c in mine if c["phase"] == "decide"]
            wri = [c for c in mine if c["phase"] == "write"]
            row = {"call": name, "said": text, "tool": result.debug.intent, "via": result.debug.via, "note": result.debug.note,
                   "rules": turn_marks.get("rules", 0.0), "decide": sum(c["ms"] for c in dec), "tool_ms": turn_marks.get("tool", 0.0),
                   "write_first": result.debug.lead_ms or None, "write_all": sum(c["ms"] or 0 for c in wri) or None,
                   "write_prefill": wri[0]["first_ms"] if wri else None, "write_pieces": sum(c["pieces"] for c in wri),
                   "decide_calls": len(dec), "write_calls": len(wri), "first_heard": first_out, "first_model_sentence": first_model, "total": total,
                   "reply": " ".join(sentences), "first_sentence": next((x for x in sentences if len(x.split()) > 2), sentences[0] if sentences else ""),
                   "call_ids": list(range(first_call, len(calls)))}
            if stt is not None:
                row["hearing"] = hear(text)
                row["voice"] = speak(row["first_sentence"]) if row["first_sentence"] else 0.0
            turns.append(row)
            print(f"[{name}] {text!r}\n   tool {row['tool']:16s} rules {row['rules']:5.0f}  decide {row['decide']:5.0f}  tool {row['tool_ms']:4.0f}  "
                  f"write: first sentence {row['write_first'] or 0:5.0f} (prompt read {row['write_prefill'] or 0:5.0f}), all {row['write_all'] or 0:5.0f}  "
                  + (f"hearing {row['hearing']:4.0f}  voice {row['voice']:4.0f}  " if stt else "") + f"| first heard {row['first_heard'] or 0:5.0f}  total {total:5.0f} ms", flush=True)
            if a.show:
                print(f"   -> {row['reply']}")
            if a.speak:
                time.sleep(len(row["reply"].split()) / 2.5)

    # Prompt sizes: the same request again with one token to write, for the server's own count (a repeated prompt is answered from its cache, so its time is not used).
    from dealer_core.nlu.openai_compat import OpenAICompatBackend
    b = next(x for x in [e.reply_writer.backend, e.backend, *getattr(e.reply_writer.backend, "backends", []), *getattr(e.backend, "backends", [])]
             if isinstance(x, OpenAICompatBackend))
    for c in calls:
        body = {k: v for k, v in c["body"].items() if k not in ("stream", "stream_options", "structured_outputs")}
        body["max_tokens"] = 1
        try:
            c["prompt_tokens"] = (b._post.__wrapped__ if hasattr(b._post, "__wrapped__") else _raw_post(b))(body, 30).get("usage", {}).get("prompt_tokens")
        except Exception:
            c["prompt_tokens"] = None
    for t in turns:
        mine = [calls[i] for i in t["call_ids"]]
        t["decide_prompt_tokens"] = next((c["prompt_tokens"] for c in mine if c["phase"] == "decide"), None)
        t["write_prompt_tokens"] = next((c["prompt_tokens"] for c in mine if c["phase"] == "write"), None)

    def med(key, rows):
        v = [r[key] for r in rows if r.get(key)]
        return round(statistics.median(v)) if v else None

    model_turns = [t for t in turns if t["decide_calls"] or t["write_calls"]]
    print(f"\nprefetch {'on' if e.prefetch_on else 'off'}: {sum(1 for c in calls if c['phase'] == 'prefetch')} prompt starts sent ahead")
    print(f"{len(turns)} turns, {sum(1 for t in turns if t['decide_calls'])} decided by the model, {sum(1 for t in turns if t['write_calls'])} written by the model")
    summary = {k: med(k, turns) for k in ("hearing", "rules", "tool_ms", "voice", "first_heard", "total")}
    summary.update({k: med(k, [t for t in turns if t["decide_calls"]]) for k in ("decide", "decide_prompt_tokens")})
    summary.update({k: med(k, [t for t in turns if t["write_calls"]]) for k in ("write_prefill", "write_first", "write_all", "write_prompt_tokens", "write_pieces")})
    summary["first_model_sentence"] = med("first_model_sentence", model_turns)
    summary["over_live_deadlines"] = {"decide": sum(1 for t in turns if t["decide"] > live["decide_s"] * 1000),
                                      "first_sentence": sum(1 for t in turns if t["write_first"] and t["write_first"] > (live["first_s"] or 99) * 1000)}
    for k, v in summary.items():
        print(f"  median {k:22s} {v}")
    a.out.parent.mkdir(exist_ok=True)
    for c in calls:
        c["system_chars"] = len(c["body"]["messages"][0]["content"]) if c["body"].get("messages") else None
        c["user_chars"] = sum(len(m["content"]) for m in c["body"].get("messages", [])[1:])
        del c["body"]
    a.out.write_text(json.dumps({"live_deadlines": live, "summary": summary, "turns": turns, "calls": calls}, indent=1, default=str), encoding="utf-8")
    print(f"\nwritten {a.out.relative_to(ROOT)}")
    return 0


def _raw_post(b):
    """The backend's own request, without the trace wrapper."""
    import types

    from dealer_core.nlu.openai_compat import OpenAICompatBackend
    return types.MethodType(OpenAICompatBackend._post, b)


if __name__ == "__main__":
    raise SystemExit(main())

r"""How much can ONE copy of each shared service carry? The voice server, the recogniser and the model, each alone, at rising concurrency (ARCH_REVIEW section 101).

  uv run python scripts/service_capacity.py voice      --url http://127.0.0.1:8040/v1
  uv run python scripts/service_capacity.py recogniser --url http://127.0.0.1:8011/v1 --model qwen3-asr --names      (the worker's hot-word list is sent)
  uv run python scripts/service_capacity.py recogniser --url http://127.0.0.1:8012/v1 --model parakeet
  VOICE_AGENT_ALLOW_CLOUD=1 uv run python scripts/service_capacity.py model --dry-run          (what would be sent; nothing is)
  VOICE_AGENT_ALLOW_CLOUD=1 uv run python scripts/service_capacity.py model --backend bedrock_nemotron     (COSTS MONEY: about 800 requests)

Each level runs N clients that ask again the moment they are answered (a closed loop) for `--seconds`, so the service is held at N requests at once. Per level: requests
answered, errors, requests a second, the wait for the first byte (or first text) and for the whole answer, median / 90th / 99th percentile.

From the busiest level that answered every request, the callers ONE copy carries are worked out with the planning rule of the autoscaler (a copy is kept at 70% of what it can
do) and the turn rhythm of the recorded calls: a caller takes a turn about every 15 s, and one turn asks the voice server for about 3 sentences, the recogniser for 1 (the final
reading) to 3 (with the early start's extra readings) clips, and the model for about 2.5 requests. Those three numbers are assumptions you can change (--turn-every,
--per-turn); everything else is measured. Run one service at a time and leave the machine otherwise quiet.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import statistics as st
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TARGET = 0.70                                       # the share of a copy's capacity the autoscaler keeps in use (deploy/k8s/voice.yaml)
SENTENCES = [                                       # what the agent really said (3 Nov calls): a short clause up to a long sentence
    "I've got five Hyundais,", "Sure,", "What budget do you have in mind, or is there a model you like?", "It had one previous owner and comes with an AGMC warranty of 12 months or 20,000 kilometres.",
    "like a 2022 Elantra in blue at 58,000 dirhams or a 2022 Tucson in white at 79,900 dirhams.", "Which one would you like to hear more about, or shall I go through the rest?",
    "The 2020 Porsche Cayenne is white,", "It's got 47,000 kilometres, automatic gearbox, and full service history.", "Would you like to come and see it?", "I don't have a Ferrari in stock right now."]
CLIPS = ["Porsche.", "Do you have a Santa Fe?", "Tell me more about the Land Cruiser please.", "I'm looking for a family SUV under one hundred thousand dirhams.",
         "How many kilometres has it done and has it had any accidents?", "Can I book a test drive for tomorrow at four in the afternoon?", "My name is Sara Khan.", "Yes, that is right."]


def pct(xs: list[float], p: float) -> float | None:
    return round(float(np.percentile(xs, p)), 3) if xs else None


def run_level(n: int, seconds: float, once, stop_error_share: float = 0.5) -> dict:
    """n clients, each asking again as soon as it is answered, for `seconds`. `once(client_index, request_index)` returns {"first": s, "total": s, **extra} or raises."""
    records: list[dict] = []
    errors: list[str] = []
    lock = threading.Lock()
    t_end = time.monotonic() + seconds
    t0 = time.monotonic()

    def client(i: int) -> None:
        k = 0
        while time.monotonic() < t_end:
            try:
                rec = once(i, k)
                with lock:
                    records.append(rec)
            except Exception as e:  # noqa: BLE001 - counted, not hidden
                with lock:
                    errors.append(f"{type(e).__name__}: {str(e)[:70]}")
                time.sleep(0.2)                                                    # a refused request is not retried in a tight loop
                if len(errors) > 8 and len(errors) / max(1, len(errors) + len(records)) > stop_error_share:
                    return
            k += 1

    threads = [threading.Thread(target=client, args=(i,), daemon=True) for i in range(n)]
    [t.start() for t in threads]
    [t.join(timeout=seconds + 60) for t in threads]
    wall = time.monotonic() - t0
    first, total = [r["first"] for r in records], [r["total"] for r in records]
    out = {"concurrency": n, "answered": len(records), "errors": len(errors), "error_kinds": sorted(set(errors))[:3], "per_second": round(len(records) / wall, 2),
           "first_p50": pct(first, 50), "first_p90": pct(first, 90), "first_p99": pct(first, 99), "total_p50": pct(total, 50), "total_p90": pct(total, 90)}
    if records and "audio_s" in records[0]:
        out["audio_seconds_per_second"] = round(sum(r["audio_s"] for r in records) / wall, 2)
    return out


def carried(rows: list[dict], per_turn: float, turn_every: float, target: float = TARGET, audio_per_turn: float | None = None) -> dict:
    """The callers one copy carries: the best rate it kept up at a level that answered everything, at the autoscaler's 70%, over what one caller asks of it.
    `audio_per_turn` (the voice server): count in seconds of speech made, which is what the server's time goes on, instead of in sentences (sentences differ a lot in length)."""
    clean = [r for r in rows if r["answered"] and not r["errors"]]
    if not clean:
        return {"callers": None, "why": "no level answered every request"}
    best = max(clean, key=lambda r: r["per_second"])
    rate = best["per_second"] * target
    out = {"saturated_per_second": best["per_second"], "at_concurrency": best["concurrency"], "planning_per_second": round(rate, 2),
           "callers": int(rate * turn_every / per_turn), "per_turn": per_turn, "turn_every_s": turn_every}
    if audio_per_turn and best.get("audio_seconds_per_second"):
        audio_rate = best["audio_seconds_per_second"] * target
        out.update({"saturated_audio_seconds_per_second": best["audio_seconds_per_second"], "callers": int(audio_rate * turn_every / audio_per_turn), "audio_seconds_per_turn": audio_per_turn,
                    "callers_if_sentences_are_these_long": int(rate * turn_every / per_turn)})
    return out


# ------------------------------------------------------------------------------------------------ the voice server
def voice_once(url: str, model: str, voice: str, rate: int):
    import httpx

    clients: dict[int, httpx.Client] = {}

    def once(i: int, k: int) -> dict:
        if i not in clients:
            clients[i] = httpx.Client(timeout=httpx.Timeout(10.0, read=20.0))
        c = clients[i]
        body = {"model": model, "input": SENTENCES[(i + k) % len(SENTENCES)], "stream": True, "response_format": "pcm", **({"voice": voice} if voice else {})}
        t = time.perf_counter()
        first, size = None, 0
        with c.stream("POST", f"{url.rstrip('/')}/audio/speech", json=body) as r:
            r.raise_for_status()
            for chunk in r.iter_bytes():
                if chunk and first is None:
                    first = time.perf_counter() - t
                size += len(chunk)
        if first is None:
            raise RuntimeError("no audio came back")
        return {"first": first, "total": time.perf_counter() - t, "audio_s": size / 2 / rate}
    return once


# ------------------------------------------------------------------------------------------------ the recogniser
def clip_wavs(voices: list[str]) -> list[bytes]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from portal_call import caller_audio

    out = []
    for k, line in enumerate(CLIPS):
        pcm = caller_audio([line], voices[k % len(voices)])[0]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm.astype("<i2").tobytes())
        out.append(buf.getvalue())
    return out


def stock_names() -> str:
    """The names the worker sends: every make and model of the stock database, as the hot-word list."""
    import sqlite3

    from dealer_core.voice.prompts import hotword_names

    vocab: dict[str, list[str]] = {}
    for mk, mo in sqlite3.connect(ROOT / "data" / "stock.db").execute("select distinct make, model from cars"):
        vocab.setdefault(mk.lower(), []).append(mo.lower())
    return ", ".join(hotword_names(vocab, (), 200))


def recogniser_once(url: str, model: str, wavs: list[bytes], names: str):
    import httpx

    clients: dict[int, httpx.Client] = {}

    def once(i: int, k: int) -> dict:
        if i not in clients:
            clients[i] = httpx.Client(timeout=15.0)
        c = clients[i]
        data = {"model": model, "language": "en", "response_format": "json", **({"hotwords": names} if names else {})}
        t = time.perf_counter()
        r = c.post(f"{url.rstrip('/')}/audio/transcriptions", data=data, files={"file": ("turn.wav", wavs[(i + k) % len(wavs)], "audio/wav")})
        r.raise_for_status()
        dt = time.perf_counter() - t
        return {"first": dt, "total": dt, "text": r.json().get("text", "")}
    return once


# ------------------------------------------------------------------------------------------------ the model
def model_once(backend, prompts: list[dict]):
    def once(i: int, k: int) -> dict:
        p = prompts[(i * 7 + k) % len(prompts)]
        t = time.perf_counter()
        first, chars = None, 0
        for piece in backend.lead_in_stream(p["system"], p["user"], p.get("max_tokens", 200), 0.2, 20.0):
            if first is None:
                first = time.perf_counter() - t
            chars += len(piece)
        if first is None:
            raise RuntimeError("the model wrote nothing")
        return {"first": first, "total": time.perf_counter() - t, "chars": chars}
    return once


def table(rows: list[dict], extra: str = "") -> None:
    print(f"{'at once':>8} {'answered':>9} {'errors':>7} {'a second':>9} {'first p50':>10} {'p90':>7} {'p99':>7} {'whole p50':>10} {'p90':>7}" + (f" {extra:>10}" if extra else ""))
    for r in rows:
        print(f"{r['concurrency']:>8} {r['answered']:>9} {r['errors']:>7} {r['per_second']:>9} {r['first_p50']!s:>10} {r['first_p90']!s:>7} {r['first_p99']!s:>7} {r['total_p50']!s:>10} {r['total_p90']!s:>7}"
              + (f" {r.get('audio_seconds_per_second', ''):>10}" if extra else ""), flush=True)
        if r["error_kinds"]:
            print("          errors:", "; ".join(r["error_kinds"]), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["voice", "recogniser", "model"])
    ap.add_argument("--url", default="")
    ap.add_argument("--model", default="", help="voice: the model name the worker sends ('pocket'); recogniser: qwen3-asr or parakeet")
    ap.add_argument("--voice", default="eve")
    ap.add_argument("--names", action="store_true", help="recogniser: send the stock's make and model names as hot-words, as the worker does")
    ap.add_argument("--backend", default="bedrock_nemotron", help="model: the backend of config/default.yaml")
    ap.add_argument("--levels", type=int, nargs="+", help="requests at once (default: voice 1 2 4 8 16; recogniser 1 2 4 8 16; model 1 2 4 8 16 32)")
    ap.add_argument("--seconds", type=float, default=0.0, help="how long each level runs (default: voice 15, recogniser 15, model 12)")
    ap.add_argument("--turn-every", type=float, default=15.0, help="seconds between one caller's turns (the recorded calls: about 15)")
    ap.add_argument("--per-turn", type=float, default=0.0, help="requests one turn makes of this service (measured, section 102: voice 1.8, recogniser 2.7, model 1.6)")
    ap.add_argument("--audio-per-turn", type=float, default=5.5, help="voice: seconds of speech the voice server makes for one turn (the worker's counters, 3 Nov: 5.4 to 5.8 s)")
    ap.add_argument("--caller-voices", default="say:Samantha,say:Daniel,say:Karen", help="recogniser: the voices that speak the clips")
    ap.add_argument("--dry-run", action="store_true", help="say what would be sent and how many requests, send nothing")
    ap.add_argument("--out", default="", help="write the rows as JSON (default logs/service_capacity_<what>.json)")
    a = ap.parse_args()

    levels = a.levels or ([1, 2, 4, 8, 16, 32] if a.what == "model" else [1, 2, 4, 8, 16])
    seconds = a.seconds or (12.0 if a.what == "model" else 15.0)
    per_turn = a.per_turn or {"voice": 1.8, "recogniser": 2.7, "model": 1.6}[a.what]

    if a.what == "model":
        from dealer_core.config import load_config
        from dealer_core.nlu.chain import make_backend

        cfg = load_config()                                                         # as the worker does: the key is read from .env
        prompts = json.loads((ROOT / "benchmarks" / "bedrock_full_reply_prompts.json").read_text())
        est = sum(n * seconds / 1.3 for n in levels)                                  # a reply takes about 1.3 s: how many requests the levels make
        chars = sum(len(p["system"]) + len(p["user"]) for p in prompts) / len(prompts)
        print(f"model {a.backend}: levels {levels} for {seconds:.0f} s each, about {est:.0f} requests of about {chars / 4:.0f} input tokens: about {est * chars / 4 / 1e6:.2f} million input tokens (COSTS MONEY)", flush=True)
        if a.dry_run:
            return 0
        backend = make_backend(a.backend, cfg.llm.backends[a.backend], cfg.llm, ROOT / "models")
        warm = getattr(backend, "warm_connection", None)
        if warm:
            warm()
        once = model_once(backend, prompts)
    elif a.what == "voice":
        url = a.url or "http://127.0.0.1:8040/v1"
        print(f"voice server {url}: levels {levels} for {seconds:.0f} s each, sentences of 1 to 18 words", flush=True)
        if a.dry_run:
            return 0
        once = voice_once(url, a.model or "pocket", a.voice, 24000)
    else:
        url = a.url or "http://127.0.0.1:8011/v1"
        print(f"recogniser {url} ({a.model or 'qwen3-asr'}{', hot-words sent' if a.names else ''}): levels {levels} for {seconds:.0f} s each, 8 clips of 1 to 5 s", flush=True)
        if a.dry_run:
            return 0
        wavs = clip_wavs([v.strip() for v in a.caller_voices.split(",")])
        once = recogniser_once(url, a.model or "qwen3-asr", wavs, stock_names() if a.names else "")

    once(0, 0)                                                                         # the first request after a quiet spell is not a measure
    rows = []
    for n in levels:
        r = run_level(n, seconds, once)
        rows.append(r)
        if r["errors"] and r["errors"] > r["answered"]:
            print(f"stopped at {n} at once: more refusals than answers ({r['error_kinds']})", flush=True)
            break
        time.sleep(2)
    table(rows, "audio s/s" if a.what == "voice" else "")
    cap = carried(rows, per_turn, a.turn_every, audio_per_turn=a.audio_per_turn if a.what == "voice" else None)
    print(f"\none copy carries about {cap.get('callers')} callers: {json.dumps(cap)}", flush=True)
    out = Path(a.out) if a.out else ROOT / "logs" / f"service_capacity_{a.what}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"service": a.what, "url": a.url, "rows": rows, "carried": cap}, indent=1), encoding="utf-8")
    print("log:", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

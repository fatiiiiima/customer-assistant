r"""Several synthetic callers at once through the real WebSocket door (tap-to-talk, the way the page talks).

  .venv\Scripts\python.exe scripts\load_test.py --levels 1 2 3 4 --url ws://127.0.0.1:8766/voice

Start the server first with --max-calls at least as large as the biggest level and on its own port. Each level runs that many identical
conversations at the same moment (the worst case: every caller is in the same phase of the script). Per level: failures, the time from the end
of the caller's speech to the first sound (median, 90th percentile, worst), the share over 2 s, and how long the mic took to close.
Callers' speech is clean synthetic audio, so recognition is optimistic; the timing is real.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
from portal_call import DEFAULT_SCRIPT, caller_audio, one_call_hf, one_call_tap  # noqa: E402


MIX = [DEFAULT_SCRIPT,
       ["Hi, I'm looking for a family SUV under one hundred thousand dirhams with low mileage", "What is the difference between the first two you mentioned?", "Which of them has the better service history and is it still under warranty?",
        "Okay, can I come and see the first one on Saturday morning?", "Ten in the morning is fine", "My name is Omar Hassan", "Zero five five nine eight seven six five four three", "Yes that is right"],
       ["Do you have any BMW X5", "Tell me everything about it, the colour, mileage, engine and price", "Is the price negotiable?", "What other cars do you have that are similar to it?", "Thank you that is all for now"],
       ["Good afternoon, I want a Toyota Camry or something similar, not too old", "Show me the cheapest one", "How many kilometres has it done and has it had any accidents?",
        "Can you book me a test drive tomorrow at five in the afternoon", "My name is Layla Ahmed", "My number is zero five zero three three three two two two one", "Yes please confirm it"],
       ["What time are you open today", "Do you have any Toyota", "Compare the Camry and the RAV4", "Tell me about the RAV4", "Does it have a warranty",
        "Has it been in an accident", "How big is the boot", "Can you repeat that", "Thank you"]]


def pct(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else float("nan")


GATEWAY = {"url": ""}


async def staggered(url, origin, variant, i: int, stagger: float, mode: str = "tap"):
    clips, lines = variant
    await asyncio.sleep(i * stagger)
    if GATEWAY["url"]:                                       # like the page: ask the router which worker takes this call
        import urllib.request

        def assign():
            with urllib.request.urlopen(GATEWAY["url"].rstrip("/") + "/assign", timeout=5) as r:
                return json.loads(r.read())["url"]
        url = await asyncio.to_thread(assign)
    res = await (one_call_hf(url, clips, None, f"c{i + 1}", origin, lines=lines) if mode == "handsfree" else one_call_tap(url, clips, None, f"c{i + 1}", origin, None))
    res["worker_url"] = url                                  # which worker took this call (through the router: the one it assigned)
    return res


GROUPS = (("worker", ("dealer_core.server",)), ("recogniser (Qwen3, MLX)", ("qwen3_asr_server",)), ("recogniser (Parakeet, MLX)", ("parakeet_asr_server",)),
          ("voice server (Pocket)", ("pocket_server",)), ("router", ("dealer_core.gateway",)), ("these callers (this script)", ("load_test.py",)))


def group_of(cmdline: list[str]) -> str | None:
    """Which part of the system a process is, from its command line (None: not one of ours)."""
    text = " ".join(cmdline or [])
    for name, marks in GROUPS:
        if any(m in text for m in marks):
            return name
    return None


async def sample_machine(into: list, by_process: dict | None = None) -> None:
    import psutil

    psutil.cpu_percent()
    watched: dict[int, tuple[str, psutil.Process]] = {}
    while True:
        await asyncio.sleep(2)
        into.append((psutil.cpu_percent(), psutil.virtual_memory().percent))
        if by_process is None:
            continue
        for p in psutil.process_iter(["pid", "cmdline"]):                       # the processes of ours, found afresh (workers start and stop)
            g = group_of(p.info["cmdline"])
            if g and p.info["pid"] not in watched:
                watched[p.info["pid"]] = (g, p)
                try:
                    p.cpu_percent(None)
                except psutil.Error:
                    pass
        now: dict[str, float] = {}
        for pid, (g, p) in list(watched.items()):
            try:
                now[g] = now.get(g, 0.0) + p.cpu_percent(None) / 100.0                 # in cores: 1.0 = one core fully busy
            except psutil.Error:
                watched.pop(pid, None)
        for g, cores in now.items():
            by_process.setdefault(g, []).append(cores)


def http_usage_url(ws_url: str) -> str:
    """ws://host:port/voice -> http://host:port/usage (the worker's counters, shared/usage.py)."""
    return ws_url.replace("wss://", "https://").replace("ws://", "http://").rsplit("/", 1)[0] + "/usage"


async def worker_urls(url: str, gateway: str) -> list[str]:
    """The workers that will take the calls: the one given, or every one the router lists."""
    if not gateway:
        return [url]

    def fetch():
        import urllib.request

        with urllib.request.urlopen(gateway.rstrip("/") + "/workers", timeout=5) as r:
            return [w["url"] for w in json.loads(r.read()).get("workers", [])]
    try:
        return await asyncio.to_thread(fetch)
    except Exception:  # noqa: BLE001 - the counts are a bonus: the run still goes ahead
        return []


async def fetch_usage(urls: list[str]) -> dict[str, float]:
    """The counters of all the workers added up (empty when one cannot be read)."""
    def one(u: str) -> dict[str, float]:
        import urllib.request

        try:
            with urllib.request.urlopen(http_usage_url(u), timeout=5) as r:
                return json.loads(r.read()).get("counts", {})
        except Exception:  # noqa: BLE001
            return {}
    total: dict[str, float] = {}
    for counts in await asyncio.gather(*[asyncio.to_thread(one, u) for u in urls]):
        for k, v in counts.items():
            total[k] = total.get(k, 0.0) + v
    return total


def usage_line(per_turn: dict[str, float], turns: float) -> str:
    g = per_turn.get
    return (f"per turn ({turns:.0f} answered): model replies {g('model.stream', 0)} (stopped before their end {g('model.stream_cut', 0)}), other model calls {g('model.post', 0)}, "
            f"model errors {g('model.error', 0)}, about {g('model.input_chars', 0) / 4:.0f} input tokens; recogniser clips {g('stt.request', 0)} ({g('spec.start_early', 0)} early starts, "
            f"{g('spec.start_hold', 0)} held, {g('spec.skipped', 0)} skipped, {g('stt.final', 0)} full readings, {g('stt.final_from_early', 0)} taken from an early one), {g('stt.audio_s', 0)} s of audio; "
            f"voice sentences {g('tts.request', 0)} (+{g('tts.cache_hit', 0)} from the cache, {g('tts.fallback', 0)} by the fallback voice), {g('tts.audio_s', 0)} s of speech; "
            f"{g('turn.sentences', 0)} sentences and {g('turn.reply_words', 0)} words said")


def _words(text) -> list[str]:
    import re

    return re.findall(r"[a-z0-9']+", (text or "").lower())


def server_steps(turns: list[dict]) -> dict:
    """What the server's own messages say (hands-free): the median of each step of the latency table, how many replies were ready ahead, and how many turns were heard as said."""
    def med(vals):
        vals = [v for v in vals if v is not None]
        return round(float(np.median(vals)), 3) if vals else None

    srv = [t["server"] for t in turns if t.get("server")]
    det = [t["detail"] for t in turns if t.get("detail")]
    steps = [d["steps"] for d in det if d.get("steps")]
    heard = [t for t in turns if "heard" in t and t.get("said")]
    out = {k: med([s_.get(k) for s_ in srv]) for k in ("wait", "check", "hear", "reply", "voice")}
    out.update({"decision": med([d.get("decision") for d in det]), "ahead_share": round(sum(1 for d in det if d.get("ahead") is True) / len(det), 2) if det else None,
                "by_code_share": round(sum(1 for d in det if d.get("by_code")) / len(det), 2) if det else None,
                "model_sent": med([x.get("sent") for x in steps]), "model_first_text": med([x.get("token") for x in steps]), "model_first_words": med([x.get("first") for x in steps]),
                "network_est": med([x.get("trip") for x in steps]),
                "heard_as_said": round(sum(1 for t in heard if _words(t["heard"]) == _words(t["said"])) / len(heard), 2) if heard else None, "turns_with_server_steps": len(srv)})
    return out


async def level(url: str, origin: str, scripts, n: int, stagger: float, mode: str = "tap", profile: bool = False, usage_urls: list[str] | None = None) -> dict:
    t0 = time.monotonic()
    machine: list = []
    cores: dict[str, list[float]] = {}
    sampler = asyncio.create_task(sample_machine(machine, cores if profile else None))
    before = await fetch_usage(usage_urls) if usage_urls else {}
    res = await asyncio.gather(*[staggered(url, origin, scripts[i % len(scripts)], i, stagger, mode) for i in range(n)], return_exceptions=True)
    sampler.cancel()
    after_counts = {}
    if usage_urls:
        await asyncio.sleep(1.5)                                  # the last replies finish their work
        after_counts = await fetch_usage(usage_urls)
    failed = [r for r in res if isinstance(r, Exception)]
    ok = [r for r in res if not isinstance(r, Exception)]
    first = [t["silence_s"] for r in ok for t in r["turns"] if t["silence_s"] is not None]
    whole = [t["reply_s"] for r in ok for t in r["turns"] if t.get("reply_s")]
    overlapped = sum(1 for r in ok for t in r["turns"] if t.get("overlap"))                  # sound was still playing as the caller finished: not a measure
    silent = sum(1 for r in ok for t in r["turns"] if t["silence_s"] is None and not t.get("overlap"))
    closed = [t["closed_s"] for r in ok for t in r["turns"] if t["closed_s"] is not None]
    taps = sum(1 for r in ok for t in r["turns"] if t["mic_closed_by"] == "tap")
    steps = server_steps([t for r in ok for t in r["turns"]])
    out_ = {"callers": n, "failed_calls": len(failed), "errors": sorted({type(e).__name__ + ": " + str(e)[:80] for e in failed}), "turns": len(first) + silent, "overlapped_turns": overlapped,
            "silent_turns": silent, "first_sound_median": round(pct(first, 50), 2), "p90": round(pct(first, 90), 2), "worst": round(max(first), 2) if first else None,
            "over_2s": round(sum(x > 2 for x in first) / len(first), 2) if first else None, "mic_close_median": round(st.median(closed), 2) if closed else None,
            "mic_closed_by_tap": taps, "cpu_mean": round(float(np.mean([c for c, _ in machine])), 0) if machine else None, "cpu_max": max((c for c, _ in machine), default=None), "ram_max": max((m for _, m in machine), default=None), "whole_reply_median": round(pct(whole, 50), 1), "whole_reply_p90": round(pct(whole, 90), 1), "greeting_s": round(st.median([r["greeting_s"] for r in ok]), 1) if ok else None, "wall_s": round(time.monotonic() - t0)}
    out_.update(steps)
    if usage_urls and after_counts:
        from dealer_core.shared import usage as _usage

        counts = _usage.delta(before, after_counts)
        out_["usage_total"], out_["usage_per_turn"] = counts, _usage.per_turn(counts)
        out_["usage_turns"] = counts.get("turn.answered", 0.0)
    if profile:
        out_["cpu_cores_by_process"] = {g: {"mean": round(float(np.mean(v)), 2), "max": round(max(v), 2)} for g, v in sorted(cores.items())}
    out_["calls_per_worker"] = {u: sum(1 for r in ok if r.get("worker_url") == u) for u in sorted({r.get("worker_url") for r in ok})}
    out_["detail"] = [{"caller": r["tag"], **t} for r in ok for t in r["turns"]]       # every turn, for the log file
    return out_


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://127.0.0.1:8766/voice")
    ap.add_argument("--gateway", default="", help="ask this router for a worker per call (e.g. http://127.0.0.1:8760), like the page does")
    ap.add_argument("--origin", default="http://localhost:8080")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--voice", default="amy", help="the callers' voice; several, comma separated, are shared out among the callers. say:Samantha uses the Mac's own voices (say:Samantha,say:Daniel,say:Karen)")
    ap.add_argument("--mode", choices=["tap", "handsfree"], default="tap", help="tap = tap-to-talk (as before); handsfree = the microphone stays open and the server decides when a turn is over (how the page is used hands-free)")
    ap.add_argument("--repeat", type=int, default=1, help="each level this many times")
    ap.add_argument("--profile", action="store_true", help="also report the CPU each of our processes used (the worker, the recogniser, the voice server, these callers), in cores: which part runs out first")
    ap.add_argument("--out", default="", help="where to write the per-turn log (default logs/load_test.json, or logs/load_test_handsfree.json)")
    ap.add_argument("--mix", action="store_true", help="five different conversations (longer questions, comparisons, a negotiation question) instead of one")
    ap.add_argument("--stagger", type=float, default=0.0, help="seconds between caller starts")
    a = ap.parse_args()
    GATEWAY["url"] = a.gateway
    print("synthesising caller lines...", flush=True)
    sets = MIX if a.mix else [DEFAULT_SCRIPT]
    voices = [v.strip() for v in a.voice.split(",") if v.strip()]
    scripts = []
    for k in range(max(len(sets), len(voices))):                       # script k is spoken in voice k (shared out when there are several voices)
        lines = sets[k % len(sets)]
        scripts.append((caller_audio(lines, voices[k % len(voices)]), lines))
    urls = await worker_urls(a.url, a.gateway)
    rows = []
    for n, rep in [(n, r_) for n in a.levels for r_ in range(a.repeat)]:
        r = await level(a.url, a.origin, scripts, n, a.stagger, a.mode, a.profile, urls)
        r["repeat"] = rep + 1
        rows.append(r)
        print(f"{n} caller(s): {r['turns']} turns, {r['failed_calls']} failed calls, {r['silent_turns']} silent; first sound median {r['first_sound_median']} s, p90 {r['p90']} s, worst {r['worst']} s, "
              f"over 2 s: {r['over_2s']}; whole reply median {r['whole_reply_median']} s, p90 {r['whole_reply_p90']} s; CPU mean {r['cpu_mean']}% max {r['cpu_max']}%, RAM max {r['ram_max']}%; mic closes after {r['mic_close_median']} s; greeting {r['greeting_s']} s of speech; {r['wall_s']} s total {r['errors'] or ''}", flush=True)
        if r.get("turns_with_server_steps"):
            print(f"    server steps (median s): silence {r['wait']}, turn check {r['check']}, hearing {r['hear']}, reply words {r['reply']} (decision {r['decision']}; model request sent at {r['model_sent']}, "
                  f"first text at {r['model_first_text']}, first words at {r['model_first_words']}; network est. {r['network_est']}), voice {r['voice']}; ready ahead {r['ahead_share']}, picked by the code {r['by_code_share']}; "
                  f"heard exactly as said {r['heard_as_said']}", flush=True)
        if r.get("usage_per_turn"):
            print("    " + usage_line(r["usage_per_turn"], r["usage_turns"]), flush=True)
        if r.get("cpu_cores_by_process"):
            print("    CPU by process, in cores (mean / max; this machine has " + str(__import__("os").cpu_count()) + "): " + "; ".join(f"{g} {v['mean']} / {v['max']}" for g, v in r["cpu_cores_by_process"].items()), flush=True)
        if len(r.get("calls_per_worker", {})) > 1:
            print(f"    calls per worker: {r['calls_per_worker']}", flush=True)
        await asyncio.sleep(3)
    (ROOT / "logs").mkdir(exist_ok=True)
    out = Path(a.out) if a.out else ROOT / "logs" / ("load_test_handsfree.json" if a.mode == "handsfree" else "load_test.json")
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("per-turn log:", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

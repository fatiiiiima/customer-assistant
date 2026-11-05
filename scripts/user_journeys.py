r"""Many kinds of user at once, the way the website is really used: most people only browse, some talk to the agent, some book a test drive.

  .venv/bin/python scripts/user_journeys.py --browse 940 --talk 40 --book 20 --duration-secs 300 --ramp-secs 60

Each agent is one simulated person, running as long as --duration-secs says:
  browse   opens the site's pages one after another with a pause to read each (2 to 8 s); on a first visit it also fetches the page's own scripts, styles and pictures,
           like a browser with an empty cache, and after that only the page, like one with a warm cache.
  talk     calls the voice agent (asks the router for a worker, opens the WebSocket, microphone open, one of the load test's conversations: searches, a comparison, questions),
           hangs up, browses for 10 to 30 s, and calls again.
  book     the same, with a conversation that ends in a test-drive booking (its own day, time, name and a number starting 05599, made fresh for each call); a call counts as
           booked only when the server sent its booking message.
Reported per kind: pages (how long, how many failed), calls (how many went through, were refused as busy or failed, the wait from the end of the caller's words to the first
sound: median, 90th percentile, worst, share over 2 s), bookings made against attempted, how many calls were open at once at most, and the processor and memory of our processes
while it ran (the worker, the router, these agents). The site and the router are given as addresses, so this can run against another machine's stack.
With the real recogniser a booking is made when the words are heard right; with the stand-in recogniser (scripts/stub_services.py) it never is, so use the stand-in only for load.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import httpx  # noqa: E402
import numpy as np  # noqa: E402
from load_test import MIX, pct  # noqa: E402
from portal_call import caller_audio, one_call_hf  # noqa: E402
from soak_test import assign_url, watch, worker_health  # noqa: E402

PAGES = ["/", "/cars.html", "/brand.html", "/car.html"]
TALK_SCRIPTS = [MIX[0], MIX[2], MIX[4]]                       # questions, a comparison, no booking
NAMES = ["Sara Khan", "Omar Hassan", "Layla Ahmed", "Yusuf Ali", "Mariam Saeed", "Khalid Nasser", "Noor Rahman", "Hamad Salem", "Aisha Malik", "Tariq Aziz", "Dina Farouk", "Rashid Karim"]
DAYS = ["tomorrow", "on Saturday", "on Monday", "on Tuesday", "on Wednesday", "on Thursday"]
HOURS = ["nine in the morning", "ten in the morning", "eleven in the morning", "twelve noon", "one in the afternoon", "two in the afternoon", "three in the afternoon",
         "four in the afternoon", "five in the afternoon"]
DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TEST_PHONE_PREFIX = "05599"                                   # every booking made by this script has a number starting 05599, so the rows can be told apart and removed


def unique_booking_lines(rng: random.Random) -> list[str]:
    """A booking conversation nobody else has had: its own day, time, name and number (the server refuses a second test drive for the same car, time and caller)."""
    tail = " ".join(rng.choice(DIGITS) for _ in range(5))
    number = " ".join(DIGITS[int(d)] for d in TEST_PHONE_PREFIX) + " " + tail
    return ["Hello, do you have any Toyota SUVs?", "How much is the first one?", "Can I book a test drive for it?", f"{rng.choice(DAYS)} at {rng.choice(HOURS)}",
            f"My name is {rng.choice(NAMES)}", f"My number is {number}", "Yes that is right"]


def local_assets(html: str) -> list[str]:
    """The same-site files a page asks for (scripts, styles, pictures): not other sites, not data: addresses."""
    found = re.findall(r'(?:src|href)="([^"#]+)"', html)
    return sorted({a for a in found if not re.match(r"^(?:[a-z]+:|//)", a, re.I) and not a.endswith(".html")})


def car_ids(html: str) -> list[str]:
    """The stock numbers the home page shows (pictures named img/AG-122.jpg), to open car pages."""
    return sorted(set(re.findall(r"img/([A-Za-z]+-\d+)\.jpg", html)))


class Log:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.calls_now = 0
        self.calls_peak = 0

    def add(self, kind: str, what: str, seconds: float, ok: bool, **extra) -> None:
        self.rows.append({"kind": kind, "what": what, "s": seconds, "ok": ok, **extra})


async def get(client: httpx.AsyncClient, log: Log, url: str, kind: str) -> str:
    t0 = time.perf_counter()
    try:
        r = await client.get(url)
        log.add(kind, url.split("?")[0].rsplit("/", 1)[-1] or "/", time.perf_counter() - t0, r.status_code == 200)
        return r.text if r.status_code == 200 and "text" in r.headers.get("content-type", "") else ""
    except Exception as e:  # noqa: BLE001 - a failed page is a result
        log.add(kind, url, time.perf_counter() - t0, False, error=type(e).__name__)
        return ""


async def browse_once(client: httpx.AsyncClient, log: Log, site: str, seen: set[str], ids: list[str]) -> None:
    page = random.choice(PAGES)
    path = f"{page}?id={random.choice(ids)}" if page == "/car.html" and ids else page
    html = await get(client, log, site + path, "page")
    for a in local_assets(html):
        if a not in seen:                                           # a first visit loads the page's files; later ones come from the browser's cache
            seen.add(a)
            await get(client, log, site + "/" + a.lstrip("/"), "asset")


async def browse_agent(n: int, ctx: dict) -> None:
    await asyncio.sleep(ctx["start_delay"](n))
    seen: set[str] = set()
    while time.monotonic() < ctx["end"]:
        await browse_once(ctx["client"], ctx["log"], ctx["site"], seen, ctx["ids"])
        await asyncio.sleep(random.uniform(2, 8))


async def call_agent(n: int, ctx: dict, kind: str, scripts: list) -> None:
    await asyncio.sleep(ctx["start_delay"](n))
    log: Log = ctx["log"]
    seen: set[str] = set()
    k = 0
    while time.monotonic() < ctx["end"]:
        if kind == "book":                                          # a booking call gets its own details, made now
            lines = unique_booking_lines(random)
            clips = await asyncio.to_thread(caller_audio, lines, ctx["voices"][k % len(ctx["voices"])])
        else:
            clips, lines = scripts[(n + k) % len(scripts)]
        k += 1
        url, assign_s, why = await asyncio.to_thread(assign_url, ctx["gateway"])
        if url is None:
            log.add(kind, "assign", assign_s, False, error=why)
            await asyncio.sleep(random.uniform(5, 15))
            continue
        log.calls_now += 1
        log.calls_peak = max(log.calls_peak, log.calls_now)
        t0 = time.perf_counter()
        try:
            res = await one_call_hf(url, clips, None, f"{kind}{n}", ctx["origin"], lines=lines)
            turns = [t["silence_s"] for t in res["turns"] if t.get("silence_s") is not None]
            log.add(kind, "call", time.perf_counter() - t0, res["status"] == "ok", status=res["status"], waits=turns, booked=bool(res.get("bookings")), turns=len(res["turns"]))
        except Exception as e:  # noqa: BLE001
            log.add(kind, "call", time.perf_counter() - t0, False, status="error", error=type(e).__name__, waits=[], booked=False, turns=0)
        finally:
            log.calls_now -= 1
        for _ in range(random.randint(1, 3)):                       # a look at the site before the next call
            if time.monotonic() >= ctx["end"]:
                break
            await browse_once(ctx["client"], log, ctx["site"], seen, ctx["ids"])
            await asyncio.sleep(random.uniform(3, 10))


def table(rows: list[dict]) -> dict:
    out: dict = {}
    for kind in ("page", "asset"):
        xs = [r for r in rows if r["kind"] == kind]
        if xs:
            s = [r["s"] for r in xs]
            out[kind] = {"requests": len(xs), "failed": sum(1 for r in xs if not r["ok"]), "median_s": round(st.median(s), 3), "p90_s": round(pct(s, 90), 3), "p99_s": round(pct(s, 99), 3), "worst_s": round(max(s), 3)}
    for kind in ("talk", "book"):
        xs = [r for r in rows if r["kind"] == kind]
        if not xs:
            continue
        calls = [r for r in xs if r["what"] == "call"]
        w = [x for r in calls for x in r.get("waits", [])]
        out[kind] = {"calls": len(calls), "completed": sum(1 for r in calls if r["ok"]), "busy": sum(1 for r in calls if r.get("status") == "busy"),
                     "failed": sum(1 for r in calls if not r["ok"] and r.get("status") != "busy") + sum(1 for r in xs if r["what"] == "assign"),
                     "turns_timed": len(w), "wait_median_s": round(st.median(w), 2) if w else None, "wait_p90_s": round(pct(w, 90), 2) if w else None,
                     "wait_worst_s": round(max(w), 2) if w else None, "over_2s": round(sum(1 for x in w if x > 2) / len(w), 3) if w else None,
                     "booked": sum(1 for r in calls if r.get("booked")), "call_length_median_s": round(st.median([r["s"] for r in calls]), 1) if calls else None}
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="http://localhost:8080", help="the website")
    ap.add_argument("--gateway", default="http://127.0.0.1:8760", help="the router")
    ap.add_argument("--origin", default="http://localhost:8080", help="the page address the workers allow")
    ap.add_argument("--browse", type=int, default=100, help="agents that only browse")
    ap.add_argument("--talk", type=int, default=5, help="agents that call and ask questions")
    ap.add_argument("--book", type=int, default=2, help="agents that call and book a test drive")
    ap.add_argument("--duration-secs", type=float, default=120.0)
    ap.add_argument("--ramp-secs", type=float, default=30.0)
    ap.add_argument("--voice", default="say:Samantha,say:Daniel,say:Karen", help="the callers' voices, shared out")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    voices = a.voice.split(",")
    mk = lambda scripts: [(caller_audio(lines, voices[i % len(voices)]), lines) for i, lines in enumerate(scripts)]  # noqa: E731
    talk_scripts, book_scripts = (mk(TALK_SCRIPTS) if a.talk else []), []
    log = Log()
    limits = httpx.Limits(max_connections=max(200, a.browse + 50), max_keepalive_connections=max(100, a.browse))
    async with httpx.AsyncClient(timeout=30, limits=limits, follow_redirects=True) as client:
        home = await get(client, Log(), a.site + "/", "page")
        if not home:
            print(f"the site does not answer at {a.site}")
            return 1
        total = a.browse + a.talk + a.book
        start = time.monotonic()
        ctx = {"client": client, "log": log, "site": a.site.rstrip("/"), "gateway": a.gateway, "origin": a.origin, "ids": car_ids(home), "end": start + a.ramp_secs + a.duration_secs, "voices": voices,
               "start_delay": lambda n: n * a.ramp_secs / max(1, total)}
        samples: list[dict] = []
        watcher = asyncio.create_task(watch(samples))
        tasks = [asyncio.create_task(browse_agent(n, ctx)) for n in range(a.browse)]
        tasks += [asyncio.create_task(call_agent(a.browse + n, ctx, "talk", talk_scripts)) for n in range(a.talk)]
        tasks += [asyncio.create_task(call_agent(a.browse + a.talk + n, ctx, "book", book_scripts)) for n in range(a.book)]
        print(f"{a.browse} browsing, {a.talk} talking, {a.book} booking = {total} people; {a.ramp_secs:.0f} s to arrive, {a.duration_secs:.0f} s of use", flush=True)
        await asyncio.gather(*tasks, return_exceptions=True)
        watcher.cancel()
    res = table(log.rows)
    res["most_calls_open_at_once"] = log.calls_peak
    steady = [s for s in samples if start + a.ramp_secs <= s["t"]]
    groups = sorted({g for s in steady for g in s["groups"]})
    res["processes"] = {g: {"cores_avg": round(float(np.mean([s["groups"].get(g, (0, 0))[1] for s in steady])), 2), "cores_max": round(max(s["groups"].get(g, (0, 0))[1] for s in steady), 2),
                            "memory_gb_max": round(max(s["groups"].get(g, (0, 0))[0] for s in samples), 2)} for g in groups} if steady else {}
    res["machine_cpu_avg_pct"] = round(float(np.mean([s["machine_cpu"] for s in steady])), 0) if steady else None
    res["workers_after"] = worker_health(a.gateway)
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps({"settings": vars(a) | {"out": str(a.out)}, "result": res, "rows": log.rows}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

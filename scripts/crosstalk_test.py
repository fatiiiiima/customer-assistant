r"""Does every caller get only their own conversation? Many booking calls at once through the router, across all the workers, each with details nobody else has.

  .venv/bin/python scripts/crosstalk_test.py --gateway http://127.0.0.1:8760 --callers 24

Every caller books a test drive with its own full name, its own number (05599 and five digits no one else has) and its own day and time, under its own call id. Afterwards:

  1. what the server sent back on a caller's connection (the booking, the words it heard, what the agent said) is searched for any OTHER caller's name or number: that is a cross-talk
     ("crossed"). A booking with neither the caller's own details nor anyone else's is a recogniser mistake ("misheard"), not a mix-up.
  2. each caller's saved conversation in Redis (call:<its id>) is searched the same way: it may hold its own details and nobody else's.
  3. a few finished calls are opened again on a DIFFERENT worker with the same call id: the server must say it carries on (resumed), and a call id nobody used must say it does not.

The recogniser matters: with the stand-in recogniser (scripts/stub_services.py) nothing is understood and no booking is made, so use the real one. The router, the workers, Redis and
a recogniser must be running (scripts/run_cluster.py).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

FIRST = ["Sara", "Omar", "Layla", "Yusuf", "Mariam", "Khalid", "Noor", "Hamad", "Aisha", "Tariq", "Dina", "Rashid"]
LAST = ["Khan", "Hassan", "Ahmed", "Ali", "Saeed", "Nasser", "Rahman", "Salem", "Malik", "Aziz", "Farouk", "Karim"]
DAYS = ["tomorrow", "on Saturday", "on Monday", "on Tuesday", "on Wednesday", "on Thursday"]
HOURS = ["nine in the morning", "ten in the morning", "eleven in the morning", "twelve noon", "one in the afternoon", "two in the afternoon", "three in the afternoon",
         "four in the afternoon", "five in the afternoon"]
DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
PREFIX = "05599"


def make_callers(n: int, run: str, seed: int = 1) -> list[dict]:
    """n callers whose details are all different: the full name, the number and the day-and-time (so the server has no reason to refuse any of them)."""
    rng = random.Random(seed)
    slots = [(d, h) for d in DAYS for h in HOURS]
    if n > min(len(slots), len(FIRST) * len(LAST)):
        raise ValueError(f"at most {min(len(slots), len(FIRST) * len(LAST))} callers with different details")
    rng.shuffle(slots)
    pairs = [(f, l) for f in FIRST for l in LAST]
    rng.shuffle(pairs)
    tails: set[str] = set()
    out = []
    for i in range(n):
        while True:
            tail = "".join(rng.choice("0123456789") for _ in range(5))
            if tail not in tails:
                tails.add(tail)
                break
        first, last = pairs[i]
        day, hour = slots[i]
        spoken = " ".join(DIGITS[int(d)] for d in PREFIX + tail)
        out.append({"n": i, "call_id": f"xt-{run}-{i}", "name": f"{first} {last}", "last": last, "phone": PREFIX + tail, "tail": tail,
                    "lines": ["Hello, do you have any Toyota SUVs?", "How much is the first one?", "Can I book a test drive for it?", f"{day} at {hour}",
                              f"My name is {first} {last}", f"My number is {spoken}", "Yes that is right"]})
    return out


def digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def phone9(text: str) -> str:
    """The nine digits that tell a UAE mobile number apart, however it is written: 0559912345, +971559912345, 971 55 991 2345 (the server saves it as +971...)."""
    d = digits(text)
    return d[-9:] if len(d) >= 9 else d


def mentions(text: str, other: dict) -> str | None:
    """How `text` shows another caller's details (their full name, or their five digits as digits or as words), or None."""
    low = (text or "").lower()
    if other["name"].lower() in low:
        return "name " + other["name"]
    words = re.sub(r"[^a-z ]", " ", low).split()
    spoken = [DIGITS[int(d)] for d in PREFIX + other["tail"]]
    for i in range(len(words) - len(spoken) + 1):
        if words[i:i + len(spoken)] == spoken:
            return "number " + other["phone"]
    if phone9(other["phone"]) in digits(text):
        return "number " + other["phone"]
    return None


def classify_booking(booking: dict, me: dict, callers: list[dict]) -> str:
    """own: this caller's number (the number is the key: nobody else has it; the name may be misheard by the recogniser: `name_exact` says whether it was right);
    crossed: another caller's number or exact name; misheard: neither (the recogniser's mistake)."""
    name_ok = (booking.get("name") or "").strip().lower() == me["name"].lower()
    phone_ok = phone9(booking.get("phone")) == phone9(me["phone"])
    for other in callers:
        if other is me:
            continue
        if phone9(booking.get("phone")) == phone9(other["phone"]) or (booking.get("name") or "").strip().lower() == other["name"].lower():
            return "crossed"
    if phone_ok:
        return "own" if name_ok else "own_name_misheard"
    return "misheard"


def check_texts(texts: list[str], me: dict, callers: list[dict]) -> list[str]:
    """Every sign of another caller's details in texts that belong to `me` (what the server heard, what the agent said, the saved conversation)."""
    found = []
    for t in texts:
        for other in callers:
            if other is not me:
                hit = mentions(t, other)
                if hit:
                    found.append(f"caller {me['n']} got {hit} (caller {other['n']}'s) in: {t[:80]}")
    return found


def worker_of(url: str) -> str:
    m = re.search(r":(\d+)", url)
    return m.group(1) if m else url


async def one_caller(me: dict, callers: list[dict], gateway: str, origin: str, delay: float, voice: str) -> dict:
    from portal_call import caller_audio, one_call_hf
    from soak_test import assign_url

    await asyncio.sleep(delay)
    clips = await asyncio.to_thread(caller_audio, me["lines"], voice)
    url, _, why = await asyncio.to_thread(assign_url, gateway)
    if url is None:
        return {"n": me["n"], "status": "no worker: " + why, "worker": None, "bookings": [], "problems": []}
    res = await one_call_hf(url, clips, None, f"xt{me['n']}", origin, call_id=me["call_id"], lines=me["lines"])
    heard = [t["heard"] for t in res["turns"] if t.get("heard")]
    said = list(res.get("agent_lines", [])) + [res.get("greeting", "")]
    bookings = res.get("bookings", [])
    problems = check_texts(heard + said, me, callers)
    kinds = [classify_booking(b, me, callers) for b in bookings]
    problems += [f"caller {me['n']}: booking {k}: {b.get('name')} / {b.get('phone')}" for b, k in zip(bookings, kinds) if k == "crossed"]
    return {"n": me["n"], "status": res["status"], "worker": worker_of(url), "url": url, "bookings": bookings, "kinds": kinds, "problems": problems, "heard": heard, "said": said}


def check_redis(redis_url: str, callers: list[dict]) -> tuple[list[str], dict]:
    """The saved conversation of each caller: its own details at most, nobody else's."""
    import redis

    r = redis.Redis.from_url(redis_url)
    problems, seen = [], {"saved": 0, "missing": 0, "own_details": 0}
    for me in callers:
        raw = r.get(f"call:{me['call_id']}")
        if raw is None:
            seen["missing"] += 1
            continue
        seen["saved"] += 1
        text = raw.decode() if isinstance(raw, bytes) else raw
        snap = json.loads(text)
        if phone9(me["phone"]) in digits(json.dumps(snap.get("draft_phone") or "")) or (snap.get("draft_name") or "").lower() == me["name"].lower():
            seen["own_details"] += 1
        problems += check_texts([text], me, callers)
    return problems, seen


async def resume_check(url: str, call_id: str) -> dict:
    """Open the call again at `url` (another worker) and read what the server says: resumed or not."""
    import websockets

    async with websockets.connect(url, max_size=None, origin="http://localhost:8080") as ws:
        await ws.send(json.dumps({"type": "hello", "car": None, "call_id": call_id, "returning": True}))
        t0 = time.monotonic()
        while time.monotonic() - t0 < 10:
            m = await asyncio.wait_for(ws.recv(), 10)
            if isinstance(m, bytes):
                continue
            e = json.loads(m)
            if e.get("type") == "ready":
                return {"resumed": bool(e.get("resumed")), "worker": worker_of(url)}
    return {"resumed": None, "worker": worker_of(url)}


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gateway", default="http://127.0.0.1:8760")
    ap.add_argument("--origin", default="http://localhost:8080")
    ap.add_argument("--redis-url", default="redis://127.0.0.1:6390/0")
    ap.add_argument("--callers", type=int, default=24)
    ap.add_argument("--ramp-secs", type=float, default=10.0, help="the callers start spread over this time")
    ap.add_argument("--voice", default="say:Samantha,say:Daniel,say:Karen")
    ap.add_argument("--resume", type=int, default=4, help="how many finished calls to open again on another worker")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    run = str(int(time.time()))[-6:]
    callers = make_callers(a.callers, run, seed=int(run))                       # a different day-and-time for each caller on every run: an earlier run's bookings still hold theirs
    voices = a.voice.split(",")
    print(f"{a.callers} callers at once, run {run}", flush=True)
    results = await asyncio.gather(*[one_caller(c, callers, a.gateway, a.origin, c["n"] * a.ramp_secs / a.callers, voices[c["n"] % len(voices)]) for c in callers], return_exceptions=True)
    rows = [r if isinstance(r, dict) else {"n": -1, "status": f"error {type(r).__name__}: {r}", "worker": None, "bookings": [], "kinds": [], "problems": []} for r in results]
    workers: dict[str, int] = {}
    for r in rows:
        workers[r["worker"]] = workers.get(r["worker"], 0) + 1
    kinds = [k for r in rows for k in r.get("kinds", [])]
    problems = [p for r in rows for p in r["problems"]]
    print(f"calls by worker: {workers}")
    print(f"statuses: { {s: sum(1 for r in rows if r['status'] == s) for s in sorted({r['status'] for r in rows})} }")
    refused = sum(1 for r in rows if not r["bookings"] and any("slot isn't free" in line for line in r.get("said", [])))
    print(f"calls told \"That slot isn't free.\" (a day and time an earlier run already booked; no booking, not a mix-up): {refused}")
    print(f"bookings: {len(kinds)}  own number and name {kinds.count('own')}  own number, name misheard {kinds.count('own_name_misheard')}  neither (misheard) {kinds.count('misheard')}  CROSSED {kinds.count('crossed')}")
    rp, seen = check_redis(a.redis_url, callers)
    problems += rp
    print(f"saved conversations in Redis: {seen['saved']} found, {seen['missing']} missing, {seen['own_details']} hold the caller's own details; other callers' details found: {len(rp)}")
    resumed = []
    done = [r for r in rows if r.get("url")]
    for r in done[:a.resume]:
        other = next((u["url"] for u in done if u["worker"] != r["worker"]), None)
        if other:
            resumed.append(await resume_check(other, callers[r["n"]]["call_id"]))
    fresh = await resume_check(done[0]["url"], f"xt-{run}-never-used") if done else None
    print(f"opened again on a different worker: {resumed}; a call id nobody used: {fresh}")
    for p in problems:
        print("PROBLEM:", p)
    verdict = not problems and all(x["resumed"] for x in resumed) and (fresh is None or fresh["resumed"] is False)
    print("RESULT:", "no caller got another caller's details; resume works across workers" if verdict else "CHECK THE PROBLEMS ABOVE")
    if a.out:
        a.out.write_text(json.dumps({"rows": rows, "redis": seen, "resumed": resumed, "fresh": fresh, "problems": problems}, indent=1, default=str))
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

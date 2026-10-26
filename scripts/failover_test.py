r"""A worker dies in the middle of a call: does the call carry on with another worker, booking intact?

  .venv\Scripts\python.exe scripts\failover_test.py --gateway http://127.0.0.1:8760      (with scripts\run_cluster.py running)

1. A caller gets a worker from the router and starts booking a test drive (car, day, time).
2. That worker is killed.
3. The caller reconnects through the router with the same call id (as the page does), gets another worker, and gives name and number.
Pass: the second worker says it carried the call on, reads the booking back with the car and time from before the crash, and books it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from portal_call import caller_audio, one_call_tap   # noqa: E402


def assign(gw: str) -> dict:
    with urllib.request.urlopen(gw.rstrip("/") + "/assign", timeout=5) as r:
        return json.loads(r.read())


def kill_port(port: int) -> None:
    subprocess.run(["powershell", "-c", f"$c=Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue; "
                    "if ($c) { taskkill /PID $c.OwningProcess /T /F }"], capture_output=True)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gateway", default="http://127.0.0.1:8760")
    ap.add_argument("--origin", default="http://localhost:8080")
    a = ap.parse_args()
    call = uuid.uuid4().hex
    before = caller_audio(["Toyota", "the Corolla", "yes", "tomorrow", "9"], "amy")
    after = caller_audio(["my name is Sara, 050 111 2222", "yes"], "amy")
    w1 = assign(a.gateway)
    print(f"call {call[:8]} -> {w1['worker']} ({w1['url']})", flush=True)
    r1 = await one_call_tap(w1["url"], before, None, "before", a.origin, None, call_id=call)
    for who, text in r1["lines"]:
        print(f"   {'YOU  ' if who == 'you' else 'AGENT'} {text}", flush=True)
    port = int(w1["url"].rsplit(":", 1)[1].split("/")[0])
    kill_port(port)
    print(f"*** killed {w1['worker']} (port {port}) ***", flush=True)
    await asyncio.sleep(7)                                         # its registry entry expires; the router stops offering it
    w2 = assign(a.gateway)
    print(f"reconnect -> {w2['worker']} ({w2['url']})", flush=True)
    r2 = await one_call_tap(w2["url"], after, None, "after", a.origin, None, call_id=call)
    for who, text in r2["lines"]:
        print(f"   {'YOU  ' if who == 'you' else 'AGENT'} {text}", flush=True)
    said = " ".join(t for w, t in r2["lines"] if w == "agent")
    ok = w2["worker"] != w1["worker"] and r2.get("resumed") and "Corolla" in said and ("9" in said or "nine" in said.lower()) and "booked" in said.lower()
    print("PASS: the call carried on with another worker and the booking was made" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

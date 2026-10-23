r"""A synthetic caller for the website's voice server: speaks a script into ws://.../voice in real time and prints what the page would show.

  .venv\Scripts\python.exe scripts\portal_call.py                      one call, default script
  .venv\Scripts\python.exe scripts\portal_call.py --calls 3            three callers at once (load check through the real door)
  .venv\Scripts\python.exe scripts\portal_call.py --car AG-122         the caller is on that car's page

The caller is a synthetic voice (clean audio), so recognition here is optimistic. Everything after the WebSocket is the real server.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import soxr
import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_SCRIPT = ["Hello, do you have any Toyota SUVs?", "How much is the first one?", "Can I book a test drive for it?",
                  "Tomorrow at four in the afternoon", "My name is Sara Khan", "My number is zero five zero one two three four five six seven", "Yes that is right"]
FRAME = 320                                                # 20 ms of 16 kHz


def caller_audio(lines: list[str], voice: str) -> list[np.ndarray]:
    from dealer_core.voice.tts import PiperSherpa

    tts = PiperSherpa(ROOT / "models", voice)
    return [(np.clip(soxr.resample(tts.synthesize(t), tts.sample_rate, 16000), -1, 1) * 32767).astype(np.int16) for t in lines]


async def one_call(url: str, clips: list[np.ndarray], car: str | None, tag: str, quiet: bool, origin: str, barge: bool = False) -> dict:
    out = {"tag": tag, "lines": [], "timing": [], "cars": [], "booking": None, "audio_s": 0.0, "status": "ok"}
    async with websockets.connect(url, max_size=None, origin=origin) as ws:
        await ws.send(json.dumps({"type": "hello", "car": car}))
        last_audio = [time.monotonic()]
        state = {"ready": None}

        async def reader() -> None:
            async for m in ws:
                if isinstance(m, bytes):
                    out["audio_s"] += len(m) / 2 / (state["ready"] or 22050)
                    last_audio[0] = time.monotonic()
                    continue
                e = json.loads(m)
                if e["type"] == "ready":
                    state["ready"] = e["sample_rate"]
                elif e["type"] == "line":
                    out["lines"].append((e["who"], e["text"]))
                    if not quiet:
                        print(f"[{tag}] {'YOU  ' if e['who'] == 'you' else 'AGENT'} {e['text']}", flush=True)
                elif e["type"] == "timing":
                    out["timing"].append(e["seconds"])
                    if not quiet:
                        print(f"[{tag}] TIME  {e['seconds']:.2f} s", flush=True)
                elif e["type"] == "cars":
                    out["cars"].append(e["ids"])
                elif e["type"] == "booking":
                    out["booking"] = e
                    if not quiet:
                        print(f"[{tag}] BOOKING {e}", flush=True)
                elif e["type"] == "stop":
                    out["stops"] = out.get("stops", 0) + 1
                    if not quiet:
                        print(f"[{tag}] STOP  (the server cut the agent off)", flush=True)
                elif e["type"] == "busy":
                    out["status"] = "busy"

        task = asyncio.create_task(reader())
        silence = np.zeros(FRAME, dtype=np.int16).tobytes()

        async def pace(chunk: bytes) -> None:
            await ws.send(chunk)
            await asyncio.sleep(0.02)

        try:
            for _ in range(250):                             # listen to the opening line: 5 s of silence, or until it has finished
                await pace(silence)
                if out["audio_s"] > 0.5 and time.monotonic() - last_audio[0] > 1.2:
                    break
            for index, clip in enumerate(clips):
                heard_before = len(out["timing"])
                for k in range(0, len(clip), FRAME):
                    part = clip[k:k + FRAME]
                    await pace((part if len(part) == FRAME else np.pad(part, (0, FRAME - len(part)))).tobytes())
                for _ in range(1500):                        # up to 30 s for the reply to start, then wait for it to finish
                    await pace(silence)
                    if barge and index < len(clips) - 1 and len(out["timing"]) > heard_before:
                        for _ in range(60):                  # the agent has started to speak: keep listening for 1.2 s, then talk over it
                            await pace(silence)
                        break
                    if len(out["timing"]) > heard_before and time.monotonic() - last_audio[0] > 1.0:
                        break
        except websockets.ConnectionClosed:
            out["status"] = out["status"] if out["status"] != "ok" else "closed"
        await asyncio.sleep(4)                                # events for the last turn (the booking) arrive after its audio
        await ws.close()
        task.cancel()
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://127.0.0.1:8765/voice")
    ap.add_argument("--origin", default="http://localhost:8080")
    ap.add_argument("--calls", type=int, default=1)
    ap.add_argument("--car", help="stock id of the car page the caller is on, e.g. AG-122")
    ap.add_argument("--voice", default="amy", help="the synthetic caller's voice (different from the agent's)")
    ap.add_argument("--barge", action="store_true", help="speak the next line while the agent is still answering the last one")
    ap.add_argument("--say", action="append", help="a line to say (repeatable); default: a booking conversation")
    a = ap.parse_args()
    lines = a.say or DEFAULT_SCRIPT
    print(f"synthesising {len(lines)} caller lines...", flush=True)
    clips = caller_audio(lines, a.voice)
    t0 = time.monotonic()
    results = await asyncio.gather(*[one_call(a.url, clips, a.car, f"c{i + 1}", a.calls > 1, a.origin, a.barge) for i in range(a.calls)])
    print(f"\n{a.calls} call(s) finished in {time.monotonic() - t0:.0f} s")
    for r in results:
        t = r["timing"]
        print(f"  {r['tag']}: {r['status']}, {len(t)} replies" + (f", first sound median {st.median(t):.2f} s, worst {max(t):.2f} s" if t else "") +
              f", {r['audio_s']:.0f} s of speech back" + (", booked " + r["booking"]["date"] + " " + r["booking"]["time"] if r["booking"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

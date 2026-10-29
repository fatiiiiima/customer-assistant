r"""How long the caller waited on each turn of a recorded call, and where the time went.

  VA_DUMP_AUDIO_DIR=logs/audio_dump <the worker's usual command>      # record a call (the caller's voice is kept: local only, delete after use)
  uv run python scripts/call_latency.py                                 # the latest recorded call
  uv run python scripts/call_latency.py logs/audio_dump/20261028_133031_0048_in.txt

From the recorder's files (voice/audiotap.py): the caller's audio with a time per 20 ms frame, the notes ("stt" when the words were recognised, "said" when the
agent had a sentence for the voice) and the agent's audio with a time per chunk sent to the page. Per turn:
  turn end             from the caller's last loud frame to the turn decided over (the silence the detector waits, then Smart Turn)
  hearing              from then to the words recognised (Whisper)
  (calls recorded before 29 Oct have no "turn_end" note: the two are shown together under turn end)
  agent                from the words to the agent's first sentence (rules, the model's decision and its first sentence)
  voice                from that sentence to the first audio sent to the page (Kokoro)
  total                from the caller's last loud frame to the first audio: what the caller waits
The caller's last loud frame is the last one well above the call's background level before the words were recognised (an estimate: a soft last syllable can be
missed, making the wait look about 0.1 s shorter). Turns with no reply audio (the caller spoke over the agent, or the call ended) are left out of the medians.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(notes_path: Path):
    base = notes_path.name[: -len("_in.txt")]
    stamp = "_".join(base.split("_")[:2])
    folder = notes_path.parent
    notes = []
    for line in notes_path.read_text(encoding="utf-8").splitlines():
        t, kind, text = (line.split("\t", 2) + ["", ""])[:3]
        notes.append((float(t), kind, text))
    times = [float(r["unix_time"]) for r in csv.DictReader(open(folder / f"{base}_in.csv"))]
    pcm = np.frombuffer((folder / f"{base}_in.pcm").read_bytes(), dtype=np.int16)
    frames = [pcm[i * 320:(i + 1) * 320] for i in range(len(times))]       # the recorder writes 20 ms frames at 16 kHz
    level = np.array([float(np.sqrt(np.mean((f.astype(np.float32) / 32768) ** 2))) if len(f) else 0.0 for f in frames])
    outs = sorted(folder.glob(f"{stamp}_*_out.csv"))
    out_times = [float(r["unix_time"]) for p in outs for r in csv.DictReader(open(p))]
    return notes, np.array(times), level, sorted(out_times)


def turns(notes, times, level, out_times):
    loud = max(0.01, 4 * float(np.percentile(level, 20)))           # well above the quietest fifth of the call (its background)
    rows, last_reply = [], 0.0
    stts = [(t, text) for t, kind, text in notes if kind == "stt"]
    for i, (t_stt, text) in enumerate(stts):
        window = (times > last_reply) & (times < t_stt) & (level > loud)
        if not window.any():
            continue
        spoke_end = float(times[window][-1]) + 0.02
        nxt = stts[i + 1][0] if i + 1 < len(stts) else float("inf")
        said = next((t for t, kind, _ in notes if kind == "said" and t_stt <= t < nxt), None)
        audio = next((t for t in out_times if (said or t_stt) <= t < nxt), None)
        ended = next((t for t, kind, _ in notes if kind == "turn_end" and spoke_end - 0.05 <= t <= t_stt), None)
        rows.append({"words": text, "turn_end": (ended if ended else t_stt) - spoke_end, "hearing": (t_stt - ended) if ended else None, "agent": (said - t_stt) if said else None,
                     "voice": (audio - said) if (audio and said) else None, "total": (audio - spoke_end) if audio else None})
        last_reply = audio or t_stt
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notes", nargs="?", type=Path, help="a recorded call's *_in.txt (default: the latest in logs/audio_dump)")
    a = ap.parse_args()
    folder = ROOT / "logs" / "audio_dump"
    heard = lambda p: p.with_name(p.name[: -len(".txt")] + ".pcm").exists() and p.with_name(p.name[: -len(".txt")] + ".pcm").stat().st_size > 0
    if a.notes is None:                                         # the latest call in which the caller was recorded (a call that ended at the greeting has no caller audio)
        calls = sorted((p for p in folder.glob("*_in.txt") if heard(p)), key=lambda p: p.stat().st_mtime) if folder.exists() else []
        if not calls:
            print(f"no recorded call with the caller's voice in {folder}: start the worker with VA_DUMP_AUDIO_DIR=logs/audio_dump in front of its command, make a call, then run this")
            return 1
        path = calls[-1]
    else:
        path = a.notes
        if not heard(path):
            print(f"{path.name}: the caller's audio is empty (the call ended before anyone spoke); nothing to time")
            return 1
    rows = turns(*load(path))
    when = datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %H:%M")
    print(f"{path.name} (recorded {when}): {len(rows)} turns\n")
    print(f"{'turn end':>8s} {'hearing':>8s} {'agent':>7s} {'voice':>7s} {'total':>7s}   the caller's words")
    f = lambda v: f"{v:6.2f}s" if v is not None else "      -"
    for r in rows:
        print(f"{f(r['turn_end']):>8s} {f(r['hearing']):>8s} {f(r['agent'])} {f(r['voice'])} {f(r['total'])}   {r['words'][:60]}")
    print()
    for key, label in (("turn_end", "turn end"), ("hearing", "hearing"), ("agent", "agent"), ("voice", "voice"), ("total", "TOTAL wait")):
        v = [r[key] for r in rows if r[key] is not None and r["total"] is not None]
        if v:
            print(f"  {label:20s} median {statistics.median(v):5.2f} s   worst {max(v):5.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score every recogniser on the same turns: words wrong ("as the agent's code reads them", scripts/score_recognisers.py --as-read), turns heard exactly, and the time after the end of speech."""
import os
from pathlib import Path as _P
ROOT_ = _P(__file__).resolve().parents[2]
WORK = _P(os.environ.get("ASR_WORK", ROOT_ / "benchmarks" / "results" / "asr_shootout")); WORK.mkdir(parents=True, exist_ok=True)   # audio, models and results; git-ignored
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = ROOT_
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from score_recognisers import read_wer  # noqa: E402

here = WORK
meta = {m["i"]: m for m in json.load(open(here / "turns.json"))}
engines: dict[str, dict[int, dict]] = {}


def load_json(name, label):
    p = here / f"res_{name}.json"
    if p.exists():
        engines[label] = {r["i"]: r for r in json.load(open(p))}


load_json("whisper", "Whisper base (now)")
load_json("nemotron", "Nemotron stream 160ms")
load_json("moonshine", "Moonshine stream small")
if (here / "res_nemotron_lead.json").exists():                    # the same models with 0.5 s of quiet before the speech, as a live stream has; the timing is the paced run's
    for name, label in (("nemotron", "Nemotron stream 160ms"), ("moonshine", "Moonshine stream small")):
        lead = {r["i"]: r for r in json.load(open(here / f"res_{name}_lead.json"))}
        engines[label + " +0.5s lead"] = {i: {**lead[i], "after_end_ms": engines[label][i]["after_end_ms"]} for i in lead}
p = here / "res_qwen3.jsonl"
if p.exists():
    rows = {}
    for line in p.read_text().splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            name = Path(r.get("file") or r.get("path") or r.get("name") or "").stem
            if name.isdigit():
                rows[int(name)] = {"i": int(name), "text": r.get("text") or r.get("transcript") or "", "after_end_ms": round(1000 * float(r.get("time") or r.get("elapsed") or r.get("seconds") or 0))}
    engines["Qwen3-ASR 0.6B"] = rows
p = here / "res_kyutai.txt"
if p.exists():
    rows = {}
    for line in p.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].isdigit():
            t = re.search(r"Time: ([\d.]+)s", parts[2]) if len(parts) > 2 else None
            rows[int(parts[0])] = {"i": int(parts[0]), "text": parts[1], "whole_clip_ms": round(1000 * float(t.group(1))) if t else None}
    engines["Kyutai STT 1B 4-bit"] = rows

sets = ["recorded", "synthetic, clean", "synthetic, headset mic"]
print(f"{'recogniser':24s} {'turns set':24s} {'n':>4s} {'words wrong':>12s} {'exact':>9s} {'after end of speech: median / p90':>34s}")
summary = []
for label, rows in engines.items():
    for s in sets + ["ALL"]:
        ids = [i for i in rows if s == "ALL" or meta[i]["set"] == s]
        if not ids:
            continue
        e = n = ex = 0
        for i in ids:
            ed, w = read_wer(meta[i]["said"], rows[i]["text"]); e += ed; n += w; ex += ed == 0
        ms = [rows[i]["after_end_ms"] for i in ids if rows[i].get("after_end_ms") is not None]
        lat = f"{round(st.median(ms))} / {sorted(ms)[max(0, int(len(ms) * .9) - 1)]} ms" if ms else "-"
        print(f"{label:24s} {s:24s} {len(ids):>4d} {100 * e / max(1, n):>11.1f}% {ex:>4d}/{len(ids):<4d} {lat:>34s}")
        summary.append({"engine": label, "set": s, "turns": len(ids), "wrong_pct": round(100 * e / max(1, n), 1), "exact": ex, "latency": lat})
# the same turns for everyone: the core set (all engines have it)
core = [i for i, m in meta.items() if m["core"] and all(i in rows for rows in engines.values())]
print(f"\nSame {len(core)} core turns for every recogniser:")
for label, rows in engines.items():
    e = n = ex = 0
    for i in core:
        ed, w = read_wer(meta[i]["said"], rows[i]["text"]); e += ed; n += w; ex += ed == 0
    print(f"  {label:24s} words wrong {100 * e / max(1, n):5.1f}%  exact {ex}/{len(core)}")
json.dump(summary, open(here / "summary.json", "w"), indent=1)
if "--show" in sys.argv:
    for label, rows in engines.items():
        for i in sorted(rows):
            ed, _ = read_wer(meta[i]["said"], rows[i]["text"])
            if ed and meta[i]["set"] == "recorded":
                print(f"  {label:24s} said {meta[i]['said']!r}\n  {'':24s} heard {rows[i]['text']!r}")

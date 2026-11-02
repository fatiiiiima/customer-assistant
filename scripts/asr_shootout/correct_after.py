"""Offline: how far can 'correct the car words afterwards' go on Qwen3-ASR's transcripts without a context? Replaces a span of the transcript by a hot-word when it looks very like it."""
import os
from pathlib import Path as _P
ROOT_ = _P(__file__).resolve().parents[2]
WORK = _P(os.environ.get("ASR_WORK", ROOT_ / "benchmarks" / "results" / "asr_shootout"))   # the shoot-out's turns and results (git-ignored)
import json, re, sys
from pathlib import Path
from rapidfuzz import fuzz
sys.path.insert(0, str(ROOT_ / "src")); sys.path.insert(0, str(ROOT_ / "scripts"))
from score_recognisers import read_wer
from hearing_check import lines
here = WORK
meta = {m["i"]: m for m in json.load(open(here / "turns.json"))}
q = {}
for line in open(here / "res_qwen3.jsonl"):
    if line.startswith("{"): r = json.loads(line); q[int(r["file"])] = r["text"]
terms = [t.strip() for t in open(ROOT_ / "config" / "hotwords.txt") if t.strip() and not t.startswith("#")]
# what the agent had just said before each recorded turn (it shapes the live hint)
agent_lines = {}
folder = ROOT_ / "logs" / "audio_dump"; i = 0
for ref in sorted(folder.glob("*.ref.txt")):
    stem = ref.name[: -len(".ref.txt")]; ag = folder / f"{stem}.agent.txt"
    al = ag.read_text().splitlines() if ag.exists() else []
    for k, _ in enumerate(lines(ref)):
        agent_lines[i] = al[k] if k < len(al) else ""; i += 1
words = lambda s: re.findall(r"[A-Za-z0-9'\-]+", s)
def correct(text, pool, threshold):
    toks = words(text); out = list(toks); changed = []
    for term in pool:
        n = len(term.split())
        for k in range(len(toks) - n + 1):
            span = " ".join(toks[k:k + n])
            if span.lower() == term.lower(): continue
            if fuzz.ratio(span.lower(), term.lower()) >= threshold and len(span) >= 4:
                out[k:k + n] = [term] + [""] * (n - 1); changed.append((span, term))
    return " ".join(w for w in out if w), changed
def score(fn):
    res = {}
    for name, sel in (("recorded 12", lambda m: m["set"] == "recorded"), ("all 252", lambda m: True)):
        e = n = ex = 0; fixed = broke = 0
        for i, m in meta.items():
            if not sel(m): continue
            base, changed = fn(i); ed0, _ = read_wer(m["said"], q[i]); ed, w = read_wer(m["said"], base)
            e += ed; n += w; ex += ed == 0; fixed += ed < ed0; broke += ed > ed0
        res[name] = f"{100*e/n:5.1f}% words wrong, exact {ex}, fixed {fixed}, made worse {broke}"
    return res
print("no correction:        ", score(lambda i: (q[i], [])))
for t in (90, 85, 80, 75, 70):
    print(f"global hot-words >= {t}:", score(lambda i, t=t: correct(q[i], terms, t)))
# the agent's last line as a tiny pool with a looser threshold (recorded turns only have that line)
def with_agent(i, t=80, loose=55):
    text, ch = correct(q[i], terms, t)
    pool = [w for w in dict.fromkeys(words(agent_lines.get(i, ""))) if w[:1].isupper() or w.lower() in {x.lower() for x in terms}]
    pool = [p for p in pool if len(p) >= 4]
    t2, ch2 = correct(text, pool, loose) if pool else (text, [])
    return t2, ch + ch2
print("global >=80 + the agent's last-line words >= 55:", score(with_agent))
for i, m in meta.items():
    if m["set"] != "recorded": continue
    a, c = with_agent(i)
    if a != q[i]: print(f"  said {m['said']!r}\n    qwen3 {q[i]!r}\n    fixed {a!r}  {c}")

print("\nFALSE CORRECTIONS at hot-words >= 80 (a correct transcript made worse):")
seen = 0
for i, m in meta.items():
    ed0, _ = read_wer(m["said"], q[i]); fixed, ch = correct(q[i], terms, 80); ed, _ = read_wer(m["said"], fixed)
    if ed > ed0 and seen < 8: print(f"  said {m['said']!r}: Qwen3 {q[i]!r} -> {fixed!r} {ch}"); seen += 1

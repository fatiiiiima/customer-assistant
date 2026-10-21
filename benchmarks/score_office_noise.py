"""Score recorded STT clips, one recording CONDITION at a time (never pooled).

Conditions (by file suffix; each is reported separately and labelled in every table):
  office noise   <id>.wav            recorded in an office with audible background noise (NOT a clean baseline)
  quiet          <id>_quiet.wav      quiet-room subset
  headset        <id>_headset.wav    headset microphone
Spontaneous clips live in benchmarks/spontaneous/ (spontNNN[suffix].wav + typed .txt) and are a
HELD-OUT set: reported, never used for tuning or for choosing a mitigation.
Non-speech probes (PROBE_IDS, e.g. spont008 "blah blah blah") are scored separately and are
excluded from the WER set.

Usage:  .venv\\Scripts\\python.exe benchmarks\\score_office_noise.py [--conditions "office noise" quiet headset]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from stt_variants import build_ext as build  # noqa: E402  (adds fwt0:/ovg: variants; others pass through)
from common import RECORDINGS, RESULTS, hotword_terms, load_wav_16k, save_result  # noqa: E402
from stt_eval import canon_numbers, normalise, term_recall, wer  # noqa: E402

SPONT = Path(__file__).parent / "spontaneous"
CONDITIONS = {"office noise": "", "quiet": "_quiet", "headset": "_headset", "raw": "_raw"}
PROBE_IDS = {"spont008"}  # "blah blah blah": non-speech probe, not part of the WER set
# Fixed engine list = the Phase 0b recommended STT (primary/fallback) and the Ollama fallback.
DEFAULT_ENGINES = ["ovw:whisper-tiny-int8-ov@CPU+hot", "fw:tiny.en+hot"]
NUMERIC_GROUPS = ("num", "plate", "phone", "date")


def _refs() -> dict[str, str]:
    refs = {}
    for line in (RECORDINGS / "script.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        cid, text = line.split("\t", 1)
        refs[cid] = text.strip()
    return refs


def load_scripted(condition: str = "office noise") -> list[tuple[str, Path, str]]:
    suf = CONDITIONS[condition]
    return [(cid, RECORDINGS / f"{cid}{suf}.wav", ref) for cid, ref in _refs().items()
            if (RECORDINGS / f"{cid}{suf}.wav").exists()]


def load_spont(condition: str = "office noise", include_probes: bool = False) -> list[tuple[str, Path, str]]:
    suf = CONDITIONS[condition]
    out = []
    for wav in sorted(SPONT.glob(f"spont*{suf}.wav")):
        base = wav.stem[: -len(suf)] if suf else wav.stem
        if not re.fullmatch(r"spont\d+", base):
            continue  # a plain-condition glob must not pick up _quiet/_headset files
        if base in PROBE_IDS and not include_probes:
            continue
        txt = wav.with_name(f"{base}.txt")
        if txt.exists():
            out.append((base, wav, txt.read_text(encoding="utf-8").strip()))
    return out


def load_probes(condition: str = "office noise") -> list[tuple[str, Path, str]]:
    return [c for c in load_spont(condition, include_probes=True) if c[0] in PROBE_IDS]


def score(eng, clips, terms, fx=None) -> dict:
    """fx: optional audio transform (float32 16 kHz -> float32 16 kHz) applied before the engine."""
    per, rows = {}, []
    num_ok = num_n = tf = te = 0
    for cid, path, ref in clips:
        audio = load_wav_16k(path)
        if fx is not None:
            audio = fx(audio).astype(np.float32)
        hyp, _ = eng.transcribe(audio)
        w = wer(ref, hyp)
        grp = cid.rstrip("0123456789")
        per.setdefault(grp, []).append(w)
        row = {"id": cid, "wer": round(w, 3), "ref": ref, "hyp": hyp}
        if grp in NUMERIC_GROUPS:
            ok = canon_numbers(ref) == canon_numbers(hyp)
            num_n += 1
            num_ok += int(ok)
            row.update(number_ok=ok, ref_numbers=canon_numbers(ref), hyp_numbers=canon_numbers(hyp))
        f, e = term_recall(ref, hyp, terms)
        tf, te = tf + f, te + e
        rows.append(row)
    all_w = [r["wer"] for r in rows]
    return {
        "clips": len(rows),
        "wer_mean": round(float(np.mean(all_w)), 3) if all_w else None,
        "wer_median": round(float(np.median(all_w)), 3) if all_w else None,
        # per-clip WER capped at 1.0: stops a repetition loop (WER 20+) dominating the mean
        "wer_capped_mean": round(float(np.mean(np.minimum(all_w, 1.0))), 3) if all_w else None,
        "runaway_clips": [r["id"] for r in rows if r["wer"] > 2.0],
        "wer_by_group": {k: round(float(np.mean(v)), 3) for k, v in per.items() if k != "spont"},
        "number_exact": f"{num_ok}/{num_n}" if num_n else "n/a",
        "term_recall": f"{tf}/{te}",
        "per_clip": rows,
    }


def probe(eng, clips) -> list[dict]:
    """Non-speech / nonsense probe: does the engine hallucinate or loop?"""
    out = []
    for cid, path, ref in clips:
        hyp, _ = eng.transcribe(load_wav_16k(path))
        toks = normalise(hyp).split()
        top = max((toks.count(t) for t in set(toks)), default=0)
        out.append({"id": cid, "typed_ref": ref, "hyp": hyp[:300], "hyp_words": len(toks),
                    "ref_words": len(normalise(ref).split()), "most_repeated_word_count": top,
                    "looped": len(toks) > 3 * max(len(normalise(ref).split()), 3) or top >= 8})
    return out


def run(engine_specs, conditions, threads, tag) -> dict:
    terms = hotword_terms()
    results = []
    for spec in engine_specs:
        print(f"== {spec}", flush=True)
        try:
            eng = build(spec, threads)
            first = next((load_scripted(c) for c in conditions if load_scripted(c)), None)
            eng.transcribe(load_wav_16k(first[0][1]))  # warm-up
            r = {"engine": eng.name, "conditions": {}}
            for cond in conditions:
                sc, sp, pr = load_scripted(cond), load_spont(cond), load_probes(cond)
                if not (sc or sp):
                    continue
                r["conditions"][cond] = {"scripted": score(eng, sc, terms) if sc else None,
                                         "spontaneous": score(eng, sp, terms) if sp else None,
                                         "probe": probe(eng, pr) if pr else []}
                s, p = r["conditions"][cond]["scripted"], r["conditions"][cond]["spontaneous"]
                print(f"   [{cond}] scripted WER {s and s['wer_mean']} (cap {s and s['wer_capped_mean']}) "
                      f"numbers {s and s['number_exact']} | spontaneous WER {p and p['wer_mean']}", flush=True)
        except Exception as e:  # noqa: BLE001
            r = {"engine": spec, "error": f"{type(e).__name__}: {e}"}
            print("   ERROR", r["error"])
        results.append(r)
    payload = {"note": "Conditions are never pooled. 'office noise' = recorded in an office with audible background "
                       "noise, NOT a clean baseline. Spontaneous clips are held out (no tuning).", "results": results}
    save_result(tag, payload)
    return payload


def to_markdown(payload: dict, title: str) -> str:
    md = [f"# {title}", "",
          "Conditions are reported separately and never pooled. **office noise** = recorded in an office with "
          "audible background noise; it is not the clean baseline. Spontaneous clips are a held-out set (never used "
          "for tuning); non-speech probes are excluded from the WER set. "
          f"Generated {time.strftime('%Y-%m-%d %H:%M')}.", "",
          "**Audio provenance:** every clip was recorded directly as 16 kHz mono 16-bit WAV with the recording helper "
          "(`benchmarks/record_clips.py`); there is no .m4a (or other) conversion step. Verified from the file headers "
          "(2026-10-20): 75 of 75 clips are WAV/PCM_16, 16000 Hz, 1 channel, and no .m4a files exist.", "",
          "| Engine | Condition | Set | Clips | WER mean | WER mean (capped 1/clip) | WER median | Runaway (WER>2) | Numbers exact | Car-term recall |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in payload["results"]:
        if "error" in r:
            md.append(f"| {r['engine']} | - | - | - | ERROR: {r['error']} | | | | | |")
            continue
        for cond, d in r["conditions"].items():
            for key, label in (("scripted", "scripted"), ("spontaneous", "spontaneous (held out)")):
                s = d[key]
                if s:
                    md.append(f"| {r['engine']} | {cond} | {label} | {s['clips']} | {s['wer_mean']} | {s['wer_capped_mean']} | "
                              f"{s['wer_median']} | {', '.join(s['runaway_clips']) or '-'} | {s['number_exact']} | {s['term_recall']} |")
    md += ["", "## Scripted WER by group", ""]
    for r in payload["results"]:
        for cond, d in r.get("conditions", {}).items():
            if d["scripted"]:
                md.append(f"- {r['engine']} [{cond}]: " + ", ".join(f"{k} {v}" for k, v in d["scripted"]["wer_by_group"].items()))
    md += ["", "## Non-speech probe (excluded from WER)", "",
           "| Engine | Condition | Clip | Typed text | Engine output | Words out/typed | Looped? |", "|---|---|---|---|---|---|---|"]
    for r in payload["results"]:
        for cond, d in r.get("conditions", {}).items():
            for p in d["probe"]:
                md.append(f"| {r['engine']} | {cond} | {p['id']} | {p['typed_ref']} | {p['hyp'][:90]} | "
                          f"{p['hyp_words']}/{p['ref_words']} | {'YES' if p['looped'] else 'no'} |")
    return "\n".join(md) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=DEFAULT_ENGINES)
    ap.add_argument("--conditions", nargs="+", default=["office noise", "quiet", "headset", "raw"], choices=list(CONDITIONS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="stt_office_noise")
    a = ap.parse_args()
    payload = run(a.engines, a.conditions, a.threads, a.tag)
    out = RESULTS / f"{a.tag}.md"
    out.write_text(to_markdown(payload, "STT accuracy by recording condition"), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()

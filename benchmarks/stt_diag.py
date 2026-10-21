"""Analysis helpers for the office-noise STT results. Accuracy only: nothing here measures speed.

    stt_diag.py audio                       per-clip audio diagnostics -> results/stt_audio_diag.{csv,md}
    stt_diag.py worst  [--json X --n 10]    worst clips per engine, ref vs hyp, number-scorer view
    stt_diag.py scorer                      sanity checks of the scorer itself (numbers, normalisation)
    stt_diag.py config                      Whisper decoding settings actually in use
    stt_diag.py mitigate [--engines ...]    offline mitigations, ONE at a time, on the SCRIPTED clips only
                                            (the held-out spontaneous set is never used to pick a mitigation)
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from common import RECORDINGS, RESULTS, hotword_terms, load_wav_16k, save_result  # noqa: E402
from score_office_noise import (PROBE_IDS, SPONT, CONDITIONS, load_scripted, load_spont, score)  # noqa: E402
from stt_eval import canon_numbers, normalise, wer  # noqa: E402

SR = 16000
FRAME = int(0.02 * SR)


# ---------------------------------------------------------------- lenient digit-stream matcher
_ORD = {"first": "one", "second": "two", "third": "three", "fourth": "four", "fifth": "five", "sixth": "six",
        "seventh": "seven", "eighth": "eight", "ninth": "nine", "tenth": "ten", "twentieth": "twenty",
        "thirtieth": "thirty"}


def digit_stream(text: str) -> str:
    """Wording-independent digits: written digits are kept as is, ordinals -> cardinals, magnitude phrases -> value,
    other number-word runs -> their digits concatenated ('twelve eight nine zero one' -> 128901, 'four thirty' -> 430).
    The strict scorer (stt_eval.canon_numbers) is unchanged; this is an extra, more forgiving view of the same hypotheses."""
    import re
    from stt_eval import NUM_WORDS, ONES, TENS, _value
    t = re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())
    t = re.sub(r"(?<=\d)(st|nd|rd|th)", "", t)
    toks = [_ORD.get(w, w) for w in re.findall(r"\d+|[a-z']+", t)]
    out, i = [], 0
    while i < len(toks):
        if toks[i].isdigit():
            out.append(toks[i])
            i += 1
        elif toks[i] in NUM_WORDS and toks[i] != "and":
            j = i
            while j < len(toks) and toks[j] in NUM_WORDS:
                j += 1
            run = toks[i:j]
            if {"hundred", "thousand", "million"} & set(run):
                out.append(str(_value(run)))
            else:
                k = 0
                while k < len(run):
                    w = run[k]
                    if w in TENS and k + 1 < len(run) and 0 < ONES.get(run[k + 1], 99) < 10:
                        out.append(str(TENS[w] + ONES[run[k + 1]]))
                        k += 2
                    else:
                        out.append(str(TENS.get(w, ONES.get(w, ""))))
                        k += 1
            i = j
        else:
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------------- audio diagnostics
def frame_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    f = x[: n * FRAME].reshape(n, FRAME).astype(np.float64)
    return 20 * np.log10(np.sqrt((f ** 2).mean(axis=1)) + 1e-9)


def diag_clip(path: Path) -> dict:
    info = sf.info(str(path))
    x = load_wav_16k(path)
    db = frame_db(x)
    noise, speech = float(np.percentile(db, 10)), float(np.percentile(db, 90))
    active = np.where(db > speech - 20)[0]  # frames within 20 dB of the typical speech level
    lead = active[0] * 0.02 if active.size else float("nan")
    trail = (len(db) - 1 - active[-1]) * 0.02 if active.size else float("nan")
    gated = float((db < -80).mean())  # frames at/near digital silence (16-bit dither floor is ~ -96 dBFS)
    onset = float(db[active[0]] - db[active[0] - 1]) if active.size and active[0] > 0 else float("nan")
    ambient = db[(db > -80) & (db < speech - 20)]  # non-gated, non-speech frames: the audible noise bed, if any
    peak = float(np.abs(x).max())
    raw = np.abs((x * 32768).round()) >= 32767
    stem = path.stem
    m4a = path.with_suffix(".m4a")
    return {
        "file": path.name, "format": f"{info.format}/{info.subtype}", "orig_sr": info.samplerate, "channels": info.channels,
        "conversion": "from .m4a via PyAV" if m4a.exists() else "none (recorded directly by record_clips.py)",
        "dur_s": round(len(x) / SR, 2),
        "peak_dBFS": round(20 * np.log10(peak + 1e-9), 1),
        "rms_dBFS": round(20 * np.log10(np.sqrt((x.astype(np.float64) ** 2).mean()) + 1e-9), 1),
        "noise_floor_dBFS": round(noise, 1), "speech_dBFS": round(speech, 1),
        "snr_dB": round(speech - noise, 1),
        "gated_frac": round(gated, 2),
        "ambient_bed_dBFS": round(float(np.median(ambient)), 1) if ambient.size >= 5 else "none",
        "onset_jump_dB": round(onset, 0),
        "clipped_samples": int(raw.sum()),
        "lead_silence_s": round(lead, 2), "trail_silence_s": round(trail, 2),
    }


def cmd_audio(_a) -> None:
    files = sorted(p for p in RECORDINGS.glob("*.wav")) + sorted(SPONT.glob("spont*.wav"))
    rows = [diag_clip(p) for p in files]
    out = RESULTS / "stt_audio_diag"
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = ["file", "dur_s", "peak_dBFS", "rms_dBFS", "speech_dBFS", "noise_floor_dBFS", "gated_frac", "ambient_bed_dBFS",
            "onset_jump_dB", "clipped_samples", "lead_silence_s", "trail_silence_s"]
    md = ["# Audio diagnostics (condition: office noise)", "",
          f"All {len(rows)} files: {set(r['format'] for r in rows)}, sample rate {set(r['orig_sr'] for r in rows)}, "
          f"channels {set(r['channels'] for r in rows)}, conversion {set(r['conversion'] for r in rows)}.",
          "noise_floor = 10th-percentile 20 ms frame level; gated_frac = share of frames below -80 dBFS (digital silence); "
          "ambient_bed = median level of frames that are neither gated nor speech (the audible noise bed); onset_jump = level rise "
          "in the single 20 ms frame where speech starts. Silence = frames more than 20 dB below the typical speech level.", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    g = np.array([r["gated_frac"] for r in rows])
    nf = np.array([r["noise_floor_dBFS"] for r in rows])
    sp = np.array([r["speech_dBFS"] for r in rows])
    amb = np.array([r["ambient_bed_dBFS"] for r in rows if r["ambient_bed_dBFS"] != "none"])
    oj = np.array([r["onset_jump_dB"] for r in rows if r["onset_jump_dB"] == r["onset_jump_dB"]])
    summ = (f"\nNoise floor dBFS: min {nf.min():.0f}, median {np.median(nf):.0f}, max {nf.max():.0f}. Gated (digital-silence) frames: "
            f"median {np.median(g):.0%}. Speech level dBFS: median {np.median(sp):.0f}. Ambient bed (non-gated, non-speech frames) median "
            f"{np.median(amb):.0f} dBFS in {len(amb)}/{len(rows)} files => speech-to-bed about {np.median(sp) - np.median(amb):.0f} dB. "
            f"Onset jump dB: median {np.median(oj):.0f}, >=30 dB for {int((oj >= 30).sum())}/{len(oj)} files. "
            f"Files with clipping: {sum(r['clipped_samples'] > 0 for r in rows)}. "
            f"Files with < 0.3 s leading silence: {sum(r['lead_silence_s'] < 0.3 for r in rows)}; "
            f"< 0.3 s trailing: {sum(r['trail_silence_s'] < 0.3 for r in rows)}.\n")
    out.with_suffix(".md").write_text("\n".join(md) + summ, encoding="utf-8")
    print(summ)
    print("saved", out.with_suffix(".md"))


# ------------------------------------------------------------------------------ onset ramp
def onset_stats(path: Path) -> dict:
    """How fast does the signal rise at speech onset? 5 ms frames; levels relative to the clip's typical speech level S.
    ramp = time from first frame above S-30 dB to first frame above S-12 dB. A natural word start (weak consonant,
    then the vowel) takes tens of ms; a gate/suppressor that opens late shows up as a very short ramp."""
    x = load_wav_16k(path)
    f = int(0.005 * SR)
    n = len(x) // f
    db = 20 * np.log10(np.sqrt((x[: n * f].reshape(n, f).astype(np.float64) ** 2).mean(axis=1)) + 1e-9)
    S = float(np.percentile(db, 90))
    a = np.argmax(db > S - 30)
    b = np.argmax(db > S - 12)
    c = np.argmax(db > S - 6)
    pre = db[max(0, b - 10):b + 1]
    return {"file": path.name, "speech_dB": round(S, 1), "ramp_ms": int((b - a) * 5), "to_full_ms": int((c - a) * 5),
            "max_5ms_rise_dB": round(float(np.max(np.diff(pre))) if len(pre) > 1 else 0.0, 0),
            "level_before_onset_dB": round(float(db[max(0, a - 3)]), 0)}


def cmd_onset(_a) -> None:
    files = sorted(RECORDINGS.glob("*.wav")) + sorted(SPONT.glob("spont*.wav"))
    rows = [onset_stats(p) for p in files]
    r = np.array([x["ramp_ms"] for x in rows])
    rise = np.array([x["max_5ms_rise_dB"] for x in rows])
    md = ["# Onset ramp (condition: office noise)", "",
          "ramp = ms from the first 5 ms frame above (speech level - 30 dB) to the first above (speech level - 12 dB); "
          "to_full = to within 6 dB of speech level; max_5ms_rise = largest level jump between two 5 ms frames just before that.", "",
          f"Median ramp {np.median(r):.0f} ms (p10 {np.percentile(r, 10):.0f}, p90 {np.percentile(r, 90):.0f}). "
          f"Clips with ramp <= 10 ms: {int((r <= 10).sum())}/{len(r)}. Clips with a >= 20 dB jump inside one 5 ms frame: "
          f"{int((rise >= 20).sum())}/{len(r)}. Median level just before onset: "
          f"{np.median([x['level_before_onset_dB'] for x in rows]):.0f} dBFS.", "",
          "| file | speech dBFS | ramp ms | to full ms | max 5 ms rise dB | level before onset dBFS |", "|---|---|---|---|---|---|"]
    md += [f"| {x['file']} | {x['speech_dB']} | {x['ramp_ms']} | {x['to_full_ms']} | {x['max_5ms_rise_dB']} | {x['level_before_onset_dB']} |" for x in rows]
    out = RESULTS / "stt_onset.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[2:5]))
    print("saved", out)


# ----------------------------------------------------------------------------------- worst clips
def cmd_worst(a) -> None:
    d = json.loads(Path(a.json).read_text(encoding="utf-8"))
    md = ["# Worst clips per engine (condition: office noise)", "",
          "Ranked by per-clip WER (uncapped). `norm` rows are what the scorer actually compares; `nums` are the "
          "canonical numbers it extracted, so a numeric miss can be judged as audio vs scorer. Non-speech probe excluded.", ""]
    for r in d["results"]:
        if "error" in r:
            continue
        cd = r["conditions"].get("office noise")
        if not cd:
            continue
        clips = [("scripted", c) for c in cd["scripted"]["per_clip"]] + [("spont", c) for c in cd["spontaneous"]["per_clip"]]
        clips.sort(key=lambda t: -t[1]["wer"])
        md += [f"## {r['engine']}", ""]
        for kind, c in clips[: a.n]:
            md += [f"**{c['id']}** ({kind}) WER {c['wer']}", f"- ref : {c['ref']}", f"- hyp : {c['hyp'][:220]}",
                   f"- norm ref: {normalise(c['ref'])}", f"- norm hyp: {normalise(c['hyp'])[:220]}"]
            if "ref_numbers" in c:
                md.append(f"- nums ref {c['ref_numbers']} vs hyp {c['hyp_numbers']} -> {'OK' if c['number_ok'] else 'MISS'}")
            md.append("")
        # all numeric misses, since numbers are the main concern
        miss = [c for c in cd["scripted"]["per_clip"] if c.get("number_ok") is False]
        md += [f"### {r['engine']}: all {len(miss)} numeric misses (scripted)", "",
               "| id | ref numbers | hyp numbers | lenient digit-stream ref / hyp | lenient match | hyp text |", "|---|---|---|---|---|---|"]
        md += [f"| {c['id']} | {c['ref_numbers']} | {c['hyp_numbers']} | {digit_stream(c['ref'])} / {digit_stream(c['hyp'])} | "
               f"{'yes' if digit_stream(c['ref']) == digit_stream(c['hyp']) else 'no'} | {c['hyp'][:100]} |" for c in miss]
        numeric = [c for c in cd["scripted"]["per_clip"] if "number_ok" in c]
        md += ["", f"Strict numbers exact {sum(c['number_ok'] for c in numeric)}/{len(numeric)}; lenient digit-stream "
               f"{sum(digit_stream(c['ref']) == digit_stream(c['hyp']) for c in numeric)}/{len(numeric)} "
               f"(the difference is clips the strict scorer rejects on wording/ordinal/grouping, not on the digits)."]
        md.append("")
    out = RESULTS / "stt_worst10.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("saved", out)


# -------------------------------------------------------------------------------- scorer sanity
def cmd_scorer(_a) -> None:
    refs = {c[0]: c[2] for c in load_scripted()}
    md = ["# Scorer sanity checks", "", "## 1. Reference text vs itself (must be WER 0 and numbers match)", "",
          "| id | extracted numbers from the reference | self WER |", "|---|---|---|"]
    bad = 0
    for cid, ref in refs.items():
        if cid.rstrip("0123456789") in ("num", "plate", "phone", "date"):
            md.append(f"| {cid} | {canon_numbers(ref)} | {wer(ref, ref)} |")
            bad += wer(ref, ref) != 0
    cases = [  # (reference id, a plausible CORRECT Whisper-style rendering) -> numbers must be equal
        ("num01", "My budget is 55,000 dirhams."), ("num04", "I can pay 120,000."),
        ("num05", "The price is 162,900."), ("num06", "Under 35,000 and less than 60,000 kilometres."),
        ("num08", "A 2019 model with 40,000 kilometres."), ("car02", "Do you have a Nissan Patrol from 2021?"),
        ("phone01", "My number is 050 123 4567."), ("phone03", "+971 50 444 3322"),
        ("plate01", "The plate is Dubai A 12345."), ("plate02", "Plate number Sharjah 34567."),
        ("date01", "Can I come on Saturday at 4:30 in the afternoon?"), ("date03", "Let us do the 25th of November at 2 p.m."),
    ]
    md += ["", "## 2. Correct digit-style renderings (Whisper style) must compare equal to the reference", "",
           "| id | digit rendering | ref numbers | hyp numbers | equal |", "|---|---|---|---|---|"]
    for cid, hyp in cases:
        ref = refs[cid]
        eq = canon_numbers(ref) == canon_numbers(hyp)
        md.append(f"| {cid} | {hyp} | {canon_numbers(ref)} | {canon_numbers(hyp)} | {'yes' if eq else '**NO (scorer issue)**'} |")
    out = RESULTS / "stt_scorer_check.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("self-WER failures:", bad)


# ------------------------------------------------------------------------------ decoding config
def cmd_config(_a) -> None:
    import faster_whisper
    from faster_whisper import WhisperModel

    sig = inspect.signature(WhisperModel.transcribe)
    keys = ["beam_size", "best_of", "patience", "length_penalty", "repetition_penalty", "no_repeat_ngram_size",
            "temperature", "compression_ratio_threshold", "log_prob_threshold", "no_speech_threshold",
            "condition_on_previous_text", "max_new_tokens", "without_timestamps", "vad_filter", "hotwords", "language"]
    print("faster-whisper", faster_whisper.__version__, {k: sig.parameters[k].default for k in keys if k in sig.parameters})
    import openvino_genai as ov
    from common import MODELS
    for m in ("whisper-tiny-int8-ov", "whisper-base-int8-ov"):
        gc = json.loads((MODELS / "ov" / m / "generation_config.json").read_text())
        print(m, {k: gc.get(k) for k in ("max_length", "forced_decoder_ids", "is_multilingual", "suppress_tokens")} if False else
              {k: gc.get(k) for k in ("max_length", "is_multilingual", "no_timestamps_token_id")})
    p = ov.WhisperPipeline(str(MODELS / "ov" / "whisper-tiny-int8-ov"), "CPU")
    c = p.get_generation_config()
    print("OV default: language", c.language, "| task", c.task, "| max_length", c.max_length, "| max_new_tokens",
          c.max_new_tokens, "| repetition_penalty", c.repetition_penalty, "| no_repeat_ngram_size",
          c.no_repeat_ngram_size, "| initial_prompt", c.initial_prompt, "| temperature", c.temperature)


# -------------------------------------------------------------------------------- mitigations
def fx_gain(x: np.ndarray) -> np.ndarray:
    """Peak-normalise to -1 dBFS (gain only; the noise rises with the speech)."""
    p = np.abs(x).max()
    return x * (0.89 / p) if p > 1e-4 else x


def fx_highpass(x: np.ndarray) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    return sosfiltfilt(butter(4, 100, btype="highpass", fs=SR, output="sos"), x)


def fx_vad_trim(x: np.ndarray) -> np.ndarray:
    """Energy VAD (no model download): keep first..last active frame with 0.2 s / 0.3 s margins."""
    db = frame_db(x)
    noise, speech = np.percentile(db, 10), np.percentile(db, 90)
    active = np.where(db > noise + 0.35 * (speech - noise))[0]
    if active.size == 0:
        return x
    a = max(0, int((active[0] * 0.02 - 0.2) * SR))
    b = min(len(x), int(((active[-1] + 1) * 0.02 + 0.3) * SR))
    return x[a:b]


DENOISED = RESULTS / "denoised"  # written by denoise_clips.py in the isolated analysis env (noisereduce is not a runtime dep)


MITIGATIONS = {"baseline (none)": None, "gain normalisation": fx_gain, "high-pass 100 Hz": fx_highpass,
               "VAD trim (energy)": fx_vad_trim, "denoise (noisereduce, isolated env)": "precomputed"}


def cmd_mitigate(a) -> None:
    from bench_stt import build
    terms = hotword_terms()
    clips = load_scripted("office noise")
    res = []
    for spec in a.engines:
        eng = build(spec, a.threads)
        eng.transcribe(load_wav_16k(clips[0][1]))
        rows = {}
        for name, fx in MITIGATIONS.items():
            if name not in a.only and a.only:
                continue
            use = clips
            if fx == "precomputed":
                if not DENOISED.exists():
                    sys.exit("run benchmarks/denoise_clips.py in benchmarks/.analysis-venv first")
                use, fx = [(c, DENOISED / p.name, r) for c, p, r in clips], None
            s = score(eng, use, terms, fx)
            rows[name] = {k: s[k] for k in ("clips", "wer_mean", "wer_capped_mean", "wer_median", "runaway_clips",
                                            "number_exact", "term_recall", "wer_by_group")}
            rows[name]["per_clip"] = [{k: c[k] for k in ("id", "wer", "hyp")} for c in s["per_clip"]]
            print(f"[{eng.name}] {name}: cap-WER {s['wer_capped_mean']} median {s['wer_median']} nums {s['number_exact']} "
                  f"terms {s['term_recall']} runaway {s['runaway_clips']}", flush=True)
        res.append({"engine": eng.name, "mitigations": rows})
    save_result(a.tag, {"condition": "office noise", "set": "scripted only (37)", "results": res})
    md = ["# Offline mitigations (condition: office noise, scripted clips only)", "",
          "Applied one at a time to the same 37 recordings, each with one fixed setting (no parameter sweeps). "
          "The held-out spontaneous set was not used. Differences of a few points on 37 clips are within noise.", "",
          "| Engine | Mitigation | WER mean (capped) | WER median | Runaway | Numbers exact | Car terms |", "|---|---|---|---|---|---|---|"]
    for r in res:
        for n, d in r["mitigations"].items():
            md.append(f"| {r['engine']} | {n} | {d['wer_capped_mean']} | {d['wer_median']} | "
                      f"{', '.join(d['runaway_clips']) or '-'} | {d['number_exact']} | {d['term_recall']} |")
    (RESULTS / f"{a.tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("saved", RESULTS / f"{a.tag}.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audio")
    sub.add_parser("onset")
    w = sub.add_parser("worst")
    w.add_argument("--json", default=str(RESULTS / "stt_engines.json"))
    w.add_argument("--n", type=int, default=10)
    sub.add_parser("scorer")
    sub.add_parser("config")
    m = sub.add_parser("mitigate")
    m.add_argument("--engines", nargs="+", default=["fw:tiny.en+hot", "ovw:whisper-tiny-int8-ov@CPU+hot"])
    m.add_argument("--threads", type=int, default=2)
    m.add_argument("--only", nargs="*", default=[], help="mitigation names to run (default all)")
    m.add_argument("--tag", default="stt_mitigations")
    a = ap.parse_args()
    {"audio": cmd_audio, "onset": cmd_onset, "worst": cmd_worst, "scorer": cmd_scorer, "config": cmd_config, "mitigate": cmd_mitigate}[a.cmd](a)


if __name__ == "__main__":
    main()

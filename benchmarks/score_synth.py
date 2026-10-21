"""Re-score the Phase 0b SYNTHETIC clips with the fixed (lenient digit-stream) number matching, old vs new side by side,
next to the same engines on the real office-noise recordings, so the synthetic baseline and the real recordings compare.

    old strict  = number_exact stored by the Phase 0b run (stt_eval.canon_numbers, unchanged)
    new strict  = same scorer, re-run now on the same synthetic clips
    new lenient = digit-stream matching (stt_diag.digit_stream): ordinals, digit-by-digit plates/phones, grouping
Accuracy only; no timing.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, hotword_terms, save_result  # noqa: E402
from score_office_noise import _refs, build, score  # noqa: E402
from stt_diag import digit_stream  # noqa: E402

SPECS = {  # spec -> (Phase 0b stored engine name, office-noise engine name in stt_engines.json)
    "fw:tiny.en+hot": ("faster-whisper:tiny.en-int8+hotwords", "faster-whisper:tiny.en-int8+hotwords"),
    "fw:base.en+hot": ("faster-whisper:base.en-int8+hotwords", "faster-whisper:base.en-int8+hotwords"),
    "moonshine:base": ("moonshine:base", "moonshine:base"),
    "ovw:whisper-tiny-int8-ov@CPU+hot": (None, "openvino-whisper:whisper-tiny-int8-ov@CPU+prompt"),
    "ovw:whisper-base-int8-ov@CPU+hot": (None, "openvino-whisper:whisper-base-int8-ov@CPU+prompt"),
    "sherpa:sherpa-onnx-streaming-zipformer-en-2023-06-26+hot": ("sherpa:en-2023-06-26+hotwords", None),
}


def lenient(per_clip) -> str:
    nums = [c for c in per_clip if "number_ok" in c]
    return f"{sum(digit_stream(c['ref']) == digit_stream(c['hyp']) for c in nums)}/{len(nums)}"


def main() -> None:
    old = {}
    for f in RESULTS.glob("stt_*.json"):
        try:
            for r in json.loads(f.read_text(encoding="utf-8")).get("results", []):
                if "accuracy" in r:
                    old.setdefault(r["engine"], r["accuracy"])
        except Exception:  # noqa: BLE001
            pass
    real = {}
    p = RESULTS / "stt_engines.json"
    if p.exists():
        for r in json.loads(p.read_text(encoding="utf-8"))["results"]:
            d = r.get("conditions", {}).get("office noise")
            if d and d["scripted"]:
                real[r["engine"]] = d["scripted"]
    refs = _refs()
    clips = [(cid, RESULTS / "synth" / f"{cid}.wav", refs[cid]) for cid in refs if (RESULTS / "synth" / f"{cid}.wav").exists()]
    terms = hotword_terms()
    rows, out = [], []
    for spec, (oldname, realname) in SPECS.items():
        try:
            eng = build(spec, 2)
            eng.transcribe(__import__("common").load_wav_16k(clips[0][1]))
            s = score(eng, clips, terms)
        except Exception as e:  # noqa: BLE001
            print(spec, "ERROR", e)
            continue
        o = old.get(oldname) if oldname else None
        rr = real.get(realname) if realname else None
        row = {"engine": eng.name, "synthetic_old_strict": o["number_exact"] if o else "n/a",
               "synthetic_new_strict": s["number_exact"], "synthetic_new_lenient": lenient(s["per_clip"]),
               "synthetic_wer_old": o["wer_mean"] if o else "n/a", "synthetic_wer_new": s["wer_mean"],
               "office_noise_strict": rr["number_exact"] if rr else "n/a",
               "office_noise_lenient": lenient(rr["per_clip"]) if rr else "n/a",
               "office_noise_wer_capped": rr["wer_capped_mean"] if rr else "n/a",
               "synthetic_wer_capped_new": s["wer_capped_mean"]}
        rows.append(row)
        print(row, flush=True)
    save_result("stt_synth_rescore", {"rows": rows})
    md = ["# Synthetic baseline vs real recordings, number matching old vs fixed", "",
          "Synthetic = Phase 0b TTS clips (optimistic, clean). office noise = your 37 scripted recordings. "
          "Numbers exact is out of 17 numeric clips (num, plate, phone, date). 'old strict' is the value stored by the Phase 0b run; "
          "'new strict' re-runs the same scorer now (differences = run-to-run variation, e.g. sampling fallback); "
          "'new lenient' is the fixed digit matching.", "",
          "| Engine | synth numbers: old strict | new strict | new lenient | synth WER old | synth WER new | office-noise numbers strict | lenient | office-noise WER (capped) |",
          "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['engine']} | {r['synthetic_old_strict']} | {r['synthetic_new_strict']} | {r['synthetic_new_lenient']} | "
           f"{r['synthetic_wer_old']} | {r['synthetic_wer_new']} | {r['office_noise_strict']} | {r['office_noise_lenient']} | "
           f"{r['office_noise_wer_capped']} |" for r in rows]
    (RESULTS / "stt_synth_rescore.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("saved")


if __name__ == "__main__":
    main()

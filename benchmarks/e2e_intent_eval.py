"""End-to-end intent evaluation: audio -> STT -> NLU intent, against YOUR gold labels, next to the typed-text path.

Answers "how much accuracy do STT errors cost?" by running the SAME labelled utterances twice:
  typed  path: NLU on the typed utterance text                       (upper bound: perfect STT)
  spoken path: NLU on the STT transcript of the recorded audio       (what the live agent gets)
and reporting both accuracies, the cost (typed minus spoken) and every utterance that flipped.

Inputs
  --labels     CSV exported by the labeller: id,state,car_in_focus,utterance,gold_intent,gold_slots_json,source,notes
  --audio-dir  folder with <id>.wav for the labelled ids (ids without audio are skipped and counted)
  --condition  REQUIRED label printed on every table, e.g. "office noise" or "quiet"
  --stt        engine spec as in bench_stt.py (default: the Phase 0b primary, OpenVINO Whisper tiny + car-term prompt)
  --backend    NLU backend name from config/default.yaml (default: the config's primary)

This script is NOT run while other jobs are using the iGPU/CPU: the NLU call loads an LLM. It does not edit src/.
Spoken-path rows whose STT output hit a runaway repetition are flagged, since they would be a decoding bug rather than
an acoustic error.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--audio-dir", required=True, type=Path)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--stt", default="ovw:whisper-tiny-int8-ov@CPU+hot")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results" / "e2e_intent.md")
    a = ap.parse_args()

    from bench_stt import build
    from common import load_wav_16k
    from dealer_core.config import load_config
    from dealer_core.eval.datasets import load_csv_set
    from dealer_core.nlu.chain import BackendChain

    examples = load_csv_set(a.labels)
    with_audio = [(e, a.audio_dir / f"{e.id}.wav") for e in examples if (a.audio_dir / f"{e.id}.wav").exists()]
    print(f"{len(examples)} labelled, {len(with_audio)} with audio in {a.audio_dir} (condition: {a.condition})")
    if not with_audio:
        sys.exit("no labelled utterance has audio; nothing to compare")

    stt = build(a.stt, a.threads)
    stt.transcribe(load_wav_16k(with_audio[0][1]))  # warm-up
    cfg = load_config()
    chain = BackendChain.from_config(cfg, a.backend)
    print(chain.start())

    rows = []
    for ex, wav in with_audio:
        text, _ = stt.transcribe(load_wav_16k(wav))
        typed = chain.classify(ex.state, ex.car, ex.utterance).intent
        spoken = chain.classify(ex.state, ex.car, text).intent
        rows.append({"id": ex.id, "gold": ex.gold_intent, "typed_text": ex.utterance, "stt_text": text,
                     "typed_ok": typed == ex.gold_intent, "spoken_ok": spoken == ex.gold_intent,
                     "typed_pred": typed.value, "spoken_pred": spoken.value,
                     "runaway": len(text.split()) > 3 * max(len(ex.utterance.split()), 3)})

    n = len(rows)
    t_ok, s_ok = sum(r["typed_ok"] for r in rows), sum(r["spoken_ok"] for r in rows)
    lost = [r for r in rows if r["typed_ok"] and not r["spoken_ok"]]
    gained = [r for r in rows if not r["typed_ok"] and r["spoken_ok"]]
    per = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        p = per[r["gold"].value]
        p[0] += 1
        p[1] += r["typed_ok"]
        p[2] += r["spoken_ok"]

    md = [f"# End-to-end intent accuracy, condition: {a.condition}", "",
          f"STT: `{stt.name}`. NLU backend: `{chain.name}`. {n} labelled utterances with audio "
          f"({len(examples) - n} labelled without audio were skipped). Generated {time.strftime('%Y-%m-%d %H:%M')}.", "",
          "| Path | Correct | Accuracy |", "|---|---|---|",
          f"| typed text | {t_ok}/{n} | {t_ok / n:.1%} |",
          f"| audio -> STT | {s_ok}/{n} | {s_ok / n:.1%} |",
          f"| **cost of STT errors** | {t_ok - s_ok:+d} clips | {(t_ok - s_ok) / n:.1%} points |", "",
          f"Lost to STT (typed right, spoken wrong): {len(lost)}. Gained (spoken right, typed wrong): {len(gained)}. "
          f"Spoken rows with a runaway STT repetition: {sum(r['runaway'] for r in rows)}.", "",
          "## By gold intent", "", "| intent | n | typed | spoken |", "|---|---|---|---|"]
    md += [f"| {k} | {v[0]} | {v[1]}/{v[0]} | {v[2]}/{v[0]} |" for k, v in sorted(per.items())]
    md += ["", "## Utterances lost to STT", "", "| id | gold | typed text | STT text | spoken pred |", "|---|---|---|---|---|"]
    md += [f"| {r['id']} | {r['gold'].value} | {r['typed_text']} | {r['stt_text'][:100]} | {r['spoken_pred']} |" for r in lost]
    md += ["", "Confusions on the spoken path: " + ", ".join(
        f"{g}->{p} x{c}" for (g, p), c in Counter((r['gold'].value, r['spoken_pred']) for r in rows if not r['spoken_ok']).most_common(8))]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:12]))
    print("saved", a.out)


if __name__ == "__main__":
    main()

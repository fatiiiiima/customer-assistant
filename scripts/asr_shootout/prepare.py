"""Build the turn set once (the same as scripts/score_recognisers.py) and save it as 16 kHz WAV files, so every recogniser hears identical audio.
recorded: the owner's calls with reference lines (logs/audio_dump/*.ref.txt); synthetic: 40 phrasings x 3 Kokoro voices, each clean and as a Bluetooth headset mic sends it.
`core` marks the subset used for the real-time latency runs and for the engines that load slowly: all recorded turns plus every 5th synthetic one."""
import os
from pathlib import Path as _P
ROOT_ = _P(__file__).resolve().parents[2]
WORK = _P(os.environ.get("ASR_WORK", ROOT_ / "benchmarks" / "results" / "asr_shootout")); WORK.mkdir(parents=True, exist_ok=True)   # audio, models and results; git-ignored
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = ROOT_
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hearing_check import lines, load_turns  # noqa: E402
from score_recognisers import VOICES, phone_band  # noqa: E402

import soxr  # noqa: E402
import yaml  # noqa: E402

from dealer_core.voice.kokoro_tts import KokoroOV  # noqa: E402

OUT = WORK / "turns"
OUT.mkdir(exist_ok=True)
turns = []
folder = ROOT / "logs" / "audio_dump"
for ref in sorted(folder.glob("*.ref.txt")):
    stem = ref.name[: -len(".ref.txt")]
    for said, clip in zip(lines(ref), load_turns(stem, folder, None)):
        turns.append(("recorded", said, clip))
phrasings = [c["say"] for c in yaml.safe_load((ROOT / "tests" / "data" / "phrasing_sweep.yaml").read_text())]
phrasings = list(dict.fromkeys(p for p in phrasings if len(p.split()) >= 2))
pick = phrasings[:: max(1, len(phrasings) // 40)][:40]
for voice in VOICES:
    tts = KokoroOV(ROOT / "models", voice)
    for said in pick:
        audio = soxr.resample(np.asarray(tts.synthesize(said), dtype=np.float32), tts.sample_rate, 16000)
        turns.append(("synthetic, clean", said, audio))
        turns.append(("synthetic, headset mic", said, phone_band(audio)))
meta, seen = [], {}
for i, (kind, said, audio) in enumerate(turns):
    n = seen[kind] = seen.get(kind, -1) + 1
    sf.write(OUT / f"{i:03d}.wav", np.clip(audio, -1, 1).astype(np.float32), 16000, subtype="PCM_16")
    meta.append({"i": i, "set": kind, "said": said, "seconds": round(len(audio) / 16000, 2), "core": kind == "recorded" or n % 5 == 0})
json.dump(meta, open(WORK / "turns.json", "w"), indent=1)
print(len(meta), "turns:", {k: sum(m["set"] == k for m in meta) for k in dict.fromkeys(m["set"] for m in meta)}, "| core:", sum(m["core"] for m in meta),
      "| audio seconds:", round(sum(m["seconds"] for m in meta)), "| mean", round(np.mean([m["seconds"] for m in meta]), 1))

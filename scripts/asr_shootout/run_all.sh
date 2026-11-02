#!/bin/zsh
# Runs every recogniser on the turns that prepare.py saved, one at a time so none competes with another for the CPU. See docs/ARCH_REVIEW.md section 79.
#   1. uv run python scripts/asr_shootout/prepare.py        builds $WORK/turns (252 WAVs) and turns.json. WORK is benchmarks/results/asr_shootout (git-ignored), or $ASR_WORK
#   2. models:
#        Nemotron streaming 160 ms int8 for sherpa-onnx (663 MB):
#          huggingface_hub.snapshot_download("csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25", local_dir="$WORK/models/nemotron160")
#        Moonshine, in a private venv (its models download on first use, 140 MB):
#          uv venv $WORK/moon-venv && uv pip install --python $WORK/moon-venv/bin/python moonshine-voice soundfile numpy
#        Qwen3-ASR 0.6B (680 MB) and Kyutai STT 1B 4-bit (1.1 GB) come with the installed `speech` tool and download on first use
#   3. this script, then  uv run python scripts/asr_shootout/score.py
cd "$(dirname "$0")/../.."
WORK=${ASR_WORK:-benchmarks/results/asr_shootout}
PY=.venv/bin/python
export ASR_WORK=$WORK
mkdir -p $WORK/turns_core
$PY -c "
import json, os
w = '$WORK'
for m in json.load(open(f'{w}/turns.json')):
    dst = f'{w}/turns_core/{m[\"i\"]:03d}.wav'
    if m['core'] and not os.path.exists(dst): os.symlink(os.path.abspath(f'{w}/turns/{m[\"i\"]:03d}.wav'), dst)
"
$PY scripts/asr_shootout/run_whisper.py
$PY scripts/asr_shootout/run_nemotron.py
$WORK/moon-venv/bin/python scripts/asr_shootout/run_moonshine.py
$PY scripts/asr_shootout/run_lead.py nemotron
$WORK/moon-venv/bin/python scripts/asr_shootout/run_lead.py moonshine
speech transcribe-batch $WORK/turns --engine qwen3 --language en --jsonl > $WORK/res_qwen3.jsonl 2> $WORK/log_qwen3.txt
: > $WORK/res_kyutai.txt
for f in $WORK/turns_core/*.wav; do        # the tool cannot batch Kyutai: one file per call
  out=$(speech transcribe --engine kyutai -m int4 --stream $f 2>&1)
  r=$(printf "%s" "$out" | grep '^Result:' | head -1 | sed 's/^Result: //'); t=$(printf "%s" "$out" | grep -o 'Time: [0-9.]*s, RTF: [0-9.]*' | head -1)
  printf "%s\t%s\t%s\n" "${f:t:r}" "$r" "$t" >> $WORK/res_kyutai.txt
done

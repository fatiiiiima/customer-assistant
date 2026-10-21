# STT accuracy, condition: office noise

**Condition: office noise.** Both sets were recorded in an office with audible background noise. This is not the clean-room baseline (a quiet-room subset may be recorded later for comparison). Spontaneous clips are a held-out set and were not used for tuning.

**Audio provenance:** every clip was recorded directly as 16 kHz mono 16-bit WAV with the recording helper (`benchmarks/record_clips.py`); there is no .m4a (or other) conversion step. Verified from the file headers on 2026-10-20: 75 of 75 clips are WAV/PCM_16, 16000 Hz, 1 channel, and no .m4a files exist in `benchmarks/recordings` or `benchmarks/spontaneous`.

Scripted clips: 37 (script.tsv). Spontaneous clips: 38 (typed references). Generated 2026-10-20 15:01.

| Engine | Condition | Set | Clips | WER mean | WER mean (capped at 1/clip) | WER median | Runaway repeats | Numbers exact | Car-term recall |
|---|---|---|---|---|---|---|---|---|---|
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt | office noise | scripted | 37 | 1.261 | 0.552 | 0.538 | num06 | 3/17 | 20/37 |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt | office noise | spontaneous (held out) | 38 | 1.403 | 0.449 | 0.354 | spont008, spont029 | n/a | 0/0 |
| faster-whisper:tiny.en-int8+hotwords | office noise | scripted | 37 | 0.54 | 0.534 | 0.5 | - | 6/17 | 13/37 |
| faster-whisper:tiny.en-int8+hotwords | office noise | spontaneous (held out) | 38 | 0.466 | 0.384 | 0.333 | spont028 | n/a | 0/0 |

## Scripted WER by group (office noise)

- openvino-whisper:whisper-tiny-int8-ov@CPU+prompt: car 0.528, num 3.741, plate 0.814, phone 0.769, date 0.636, name 0.778, place 0.338, flow 0.359
- faster-whisper:tiny.en-int8+hotwords: car 0.513, num 0.506, plate 0.83, phone 0.59, date 0.347, name 0.656, place 0.815, flow 0.274

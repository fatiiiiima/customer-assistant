# STT accuracy by recording condition

Conditions are reported separately and never pooled. **office noise** = recorded in an office with audible background noise; it is not the clean baseline. Spontaneous clips are a held-out set (never used for tuning); non-speech probes are excluded from the WER set. Generated 2026-10-20 16:06.

**Audio provenance:** every clip was recorded directly as 16 kHz mono 16-bit WAV with the recording helper (`benchmarks/record_clips.py`); there is no .m4a (or other) conversion step. Verified from the file headers (2026-10-20): 75 of 75 clips are WAV/PCM_16, 16000 Hz, 1 channel, and no .m4a files exist.

| Engine | Condition | Set | Clips | WER mean | WER mean (capped 1/clip) | WER median | Runaway (WER>2) | Numbers exact | Car-term recall |
|---|---|---|---|---|---|---|---|---|---|
| openvino-whisper:whisper-small.en-int8-ov@CPU+prompt | office noise | scripted | 37 | 0.354 | 0.347 | 0.286 | - | 8/17 | 23/37 |
| openvino-whisper:whisper-small.en-int8-ov@CPU+prompt | office noise | spontaneous (held out) | 37 | 0.233 | 0.225 | 0.133 | - | n/a | 0/0 |

## Scripted WER by group

- openvino-whisper:whisper-small.en-int8-ov@CPU+prompt [office noise]: car 0.408, num 0.305, plate 0.614, phone 0.385, date 0.182, name 0.611, place 0.366, flow 0.028

## Non-speech probe (excluded from WER)

| Engine | Condition | Clip | Typed text | Engine output | Words out/typed | Looped? |
|---|---|---|---|---|---|---|
| openvino-whisper:whisper-small.en-int8-ov@CPU+prompt | office noise | spont008 | blah blah blah blah blah | La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La | 141/5 | YES |

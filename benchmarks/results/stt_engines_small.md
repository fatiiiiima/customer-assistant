# STT accuracy by recording condition

Conditions are reported separately and never pooled. **office noise** = recorded in an office with audible background noise; it is not the clean baseline. Spontaneous clips are a held-out set (never used for tuning); non-speech probes are excluded from the WER set. Generated 2026-10-20 15:39.

**Audio provenance:** every clip was recorded directly as 16 kHz mono 16-bit WAV with the recording helper (`benchmarks/record_clips.py`); there is no .m4a (or other) conversion step. Verified from the file headers (2026-10-20): 75 of 75 clips are WAV/PCM_16, 16000 Hz, 1 channel, and no .m4a files exist.

| Engine | Condition | Set | Clips | WER mean | WER mean (capped 1/clip) | WER median | Runaway (WER>2) | Numbers exact | Car-term recall |
|---|---|---|---|---|---|---|---|---|---|
| openvino-whisper:whisper-small-int8-ov@CPU+prompt | office noise | scripted | 37 | 0.323 | 0.323 | 0.222 | - | 8/17 | 29/37 |
| openvino-whisper:whisper-small-int8-ov@CPU+prompt | office noise | spontaneous (held out) | 37 | 0.262 | 0.256 | 0.2 | - | n/a | 0/0 |
| ovw:whisper-small.en-int8-ov@CPU+hot | - | - | - | ERROR: RuntimeError: Check '!language.has_value()' failed at C:\Jenkins\workspace\private-ci\ie\build-windows-vs2022\b\repos\openvino.genai\src\cpp\src\whisper\generation_config.cpp:98:
Cannot specify 'language' for not multilingual model.
 | | | | | |
| faster-whisper:small.en-int8+hotwords | office noise | scripted | 37 | 0.351 | 0.347 | 0.3 | - | 6/17 | 23/37 |
| faster-whisper:small.en-int8+hotwords | office noise | spontaneous (held out) | 37 | 0.25 | 0.222 | 0.133 | - | n/a | 0/0 |

## Scripted WER by group

- openvino-whisper:whisper-small-int8-ov@CPU+prompt [office noise]: car 0.329, num 0.205, plate 0.549, phone 0.462, date 0.121, name 0.611, place 0.472, flow 0.097
- faster-whisper:small.en-int8+hotwords [office noise]: car 0.403, num 0.303, plate 0.614, phone 0.359, date 0.316, name 0.611, place 0.236, flow 0.028

## Non-speech probe (excluded from WER)

| Engine | Condition | Clip | Typed text | Engine output | Words out/typed | Looped? |
|---|---|---|---|---|---|---|
| openvino-whisper:whisper-small-int8-ov@CPU+prompt | office noise | spont008 | blah blah blah blah blah | Raman, | 1/5 | no |
| faster-whisper:small.en-int8+hotwords | office noise | spont008 | blah blah blah blah blah | Leblada Leblada Leblada Leblada... | 4/5 | no |

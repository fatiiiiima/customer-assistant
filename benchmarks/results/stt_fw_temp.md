# STT accuracy by recording condition

Conditions are reported separately and never pooled. **office noise** = recorded in an office with audible background noise; it is not the clean baseline. Spontaneous clips are a held-out set (never used for tuning); non-speech probes are excluded from the WER set. Generated 2026-10-20 16:02.

**Audio provenance:** every clip was recorded directly as 16 kHz mono 16-bit WAV with the recording helper (`benchmarks/record_clips.py`); there is no .m4a (or other) conversion step. Verified from the file headers (2026-10-20): 75 of 75 clips are WAV/PCM_16, 16000 Hz, 1 channel, and no .m4a files exist.

| Engine | Condition | Set | Clips | WER mean | WER mean (capped 1/clip) | WER median | Runaway (WER>2) | Numbers exact | Car-term recall |
|---|---|---|---|---|---|---|---|---|---|
| faster-whisper:tiny.en-int8+hotwords | office noise | scripted | 37 | 0.528 | 0.52 | 0.5 | - | 8/17 | 14/37 |
| faster-whisper:tiny.en-int8+hotwords | office noise | spontaneous (held out) | 37 | 0.467 | 0.374 | 0.267 | spont028 | n/a | 0/0 |
| faster-whisper:tiny.en-int8+hotwords | office noise | scripted | 37 | 0.488 | 0.481 | 0.455 | - | 7/17 | 15/37 |
| faster-whisper:tiny.en-int8+hotwords | office noise | spontaneous (held out) | 37 | 0.451 | 0.384 | 0.3 | spont028 | n/a | 0/0 |
| faster-whisper:tiny.en-int8 temp=0+hotwords | office noise | scripted | 37 | 0.579 | 0.568 | 0.538 | - | 6/17 | 13/37 |
| faster-whisper:tiny.en-int8 temp=0+hotwords | office noise | spontaneous (held out) | 37 | 0.479 | 0.394 | 0.3 | spont028 | n/a | 0/0 |
| faster-whisper:base.en-int8+hotwords | office noise | scripted | 37 | 0.515 | 0.431 | 0.375 | num07 | 7/17 | 16/37 |
| faster-whisper:base.en-int8+hotwords | office noise | spontaneous (held out) | 37 | 0.463 | 0.345 | 0.333 | spont028 | n/a | 0/0 |
| faster-whisper:base.en-int8+hotwords | office noise | scripted | 37 | 0.438 | 0.417 | 0.364 | - | 8/17 | 16/37 |
| faster-whisper:base.en-int8+hotwords | office noise | spontaneous (held out) | 37 | 0.429 | 0.356 | 0.333 | spont028 | n/a | 0/0 |
| faster-whisper:base.en-int8 temp=0+hotwords | office noise | scripted | 37 | 0.467 | 0.442 | 0.375 | - | 6/17 | 15/37 |
| faster-whisper:base.en-int8 temp=0+hotwords | office noise | spontaneous (held out) | 37 | 0.367 | 0.349 | 0.333 | - | n/a | 0/0 |
| faster-whisper:small.en-int8+hotwords | office noise | scripted | 37 | 0.336 | 0.332 | 0.286 | - | 7/17 | 24/37 |
| faster-whisper:small.en-int8+hotwords | office noise | spontaneous (held out) | 37 | 0.23 | 0.218 | 0.133 | - | n/a | 0/0 |
| faster-whisper:small.en-int8 temp=0+hotwords | office noise | scripted | 37 | 0.342 | 0.338 | 0.3 | - | 6/17 | 23/37 |
| faster-whisper:small.en-int8 temp=0+hotwords | office noise | spontaneous (held out) | 37 | 0.456 | 0.222 | 0.133 | spont029 | n/a | 0/0 |

## Scripted WER by group

- faster-whisper:tiny.en-int8+hotwords [office noise]: car 0.489, num 0.446, plate 0.907, phone 0.59, date 0.347, name 0.722, place 0.81, flow 0.233
- faster-whisper:tiny.en-int8+hotwords [office noise]: car 0.5, num 0.52, plate 0.527, phone 0.59, date 0.31, name 0.6, place 0.625, flow 0.233
- faster-whisper:tiny.en-int8 temp=0+hotwords [office noise]: car 0.518, num 0.516, plate 0.968, phone 0.59, date 0.606, name 0.722, place 0.847, flow 0.233
- faster-whisper:base.en-int8+hotwords [office noise]: car 0.458, num 0.663, plate 0.938, phone 0.846, date 0.451, name 0.6, place 0.19, flow 0.028
- faster-whisper:base.en-int8+hotwords [office noise]: car 0.458, num 0.334, plate 0.795, phone 0.846, date 0.451, name 0.667, place 0.19, flow 0.028
- faster-whisper:base.en-int8 temp=0+hotwords [office noise]: car 0.48, num 0.421, plate 0.843, phone 0.846, date 0.451, name 0.667, place 0.19, flow 0.028
- faster-whisper:small.en-int8+hotwords [office noise]: car 0.403, num 0.261, plate 0.614, phone 0.359, date 0.316, name 0.544, place 0.236, flow 0.028
- faster-whisper:small.en-int8 temp=0+hotwords [office noise]: car 0.392, num 0.303, plate 0.614, phone 0.359, date 0.316, name 0.544, place 0.236, flow 0.028

## Non-speech probe (excluded from WER)

| Engine | Condition | Clip | Typed text | Engine output | Words out/typed | Looped? |
|---|---|---|---|---|---|---|
| faster-whisper:tiny.en-int8+hotwords | office noise | spont008 | blah blah blah blah blah | Black, da, da, da, just, yeah. | 6/5 | no |
| faster-whisper:tiny.en-int8+hotwords | office noise | spont008 | blah blah blah blah blah | Blah, blah, blah, blah, blah, blah | 6/5 | no |
| faster-whisper:tiny.en-int8 temp=0+hotwords | office noise | spont008 | blah blah blah blah blah | Alai, Blah, Blah, Blah, Blah, Blah, Blah, Blah, Blah, Blah, Blah, Blah, Blah, Blah, Blah,  | 75/5 | YES |
| faster-whisper:base.en-int8+hotwords | office noise | spont008 | blah blah blah blah blah | Al, Dada Dada, Dada, Dada, Dada Dada Dada. | 8/5 | no |
| faster-whisper:base.en-int8+hotwords | office noise | spont008 | blah blah blah blah blah | La, Gla, Gla, Gla, Gla, Gla, Gla, Gla. | 8/5 | no |
| faster-whisper:base.en-int8 temp=0+hotwords | office noise | spont008 | blah blah blah blah blah | and the other. | 3/5 | no |
| faster-whisper:small.en-int8+hotwords | office noise | spont008 | blah blah blah blah blah | La, Blah, Blah, Blah, Blah, Blah, Blah Blah. | 8/5 | no |
| faster-whisper:small.en-int8 temp=0+hotwords | office noise | spont008 | blah blah blah blah blah | La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La | 112/5 | YES |

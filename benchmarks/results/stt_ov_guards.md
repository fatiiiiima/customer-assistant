# STT accuracy by recording condition

Conditions are reported separately and never pooled. **office noise** = recorded in an office with audible background noise; it is not the clean baseline. Spontaneous clips are a held-out set (never used for tuning); non-speech probes are excluded from the WER set. Generated 2026-10-20 15:43.

**Audio provenance:** every clip was recorded directly as 16 kHz mono 16-bit WAV with the recording helper (`benchmarks/record_clips.py`); there is no .m4a (or other) conversion step. Verified from the file headers (2026-10-20): 75 of 75 clips are WAV/PCM_16, 16000 Hz, 1 channel, and no .m4a files exist.

| Engine | Condition | Set | Clips | WER mean | WER mean (capped 1/clip) | WER median | Runaway (WER>2) | Numbers exact | Car-term recall |
|---|---|---|---|---|---|---|---|---|---|
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt | office noise | scripted | 37 | 1.259 | 0.55 | 0.5 | num06 | 3/17 | 20/37 |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt | office noise | spontaneous (held out) | 37 | 0.857 | 0.434 | 0.333 | spont029 | n/a | 0/0 |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt [cap] | office noise | scripted | 37 | 0.689 | 0.55 | 0.5 | num06 | 3/17 | 20/37 |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt [cap] | office noise | spontaneous (held out) | 37 | 0.519 | 0.434 | 0.333 | spont029 | n/a | 0/0 |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt [cap] [rep] | office noise | scripted | 37 | 0.599 | 0.584 | 0.625 | - | 4/17 | 13/37 |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt [cap] [rep] | office noise | spontaneous (held out) | 37 | 0.425 | 0.425 | 0.4 | - | n/a | 0/0 |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt [cap] [chk] | office noise | scripted | 37 | 0.565 | 0.55 | 0.5 | - | 3/17 | 20/37 |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt [cap] [chk] | office noise | spontaneous (held out) | 37 | 0.442 | 0.434 | 0.333 | - | n/a | 0/0 |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt [cap] [rep] [chk] | office noise | scripted | 37 | 0.599 | 0.584 | 0.625 | - | 4/17 | 13/37 |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt [cap] [rep] [chk] | office noise | spontaneous (held out) | 37 | 0.425 | 0.425 | 0.4 | - | n/a | 0/0 |
| openvino-whisper:whisper-base-int8-ov@CPU+prompt | office noise | scripted | 37 | 0.695 | 0.426 | 0.4 | phone02 | 7/17 | 24/37 |
| openvino-whisper:whisper-base-int8-ov@CPU+prompt | office noise | spontaneous (held out) | 37 | 0.362 | 0.338 | 0.333 | - | n/a | 0/0 |
| openvino-whisper:whisper-base-int8-ov@CPU+prompt [cap] [chk] | office noise | scripted | 37 | 0.445 | 0.426 | 0.4 | - | 7/17 | 24/37 |
| openvino-whisper:whisper-base-int8-ov@CPU+prompt [cap] [chk] | office noise | spontaneous (held out) | 37 | 0.362 | 0.338 | 0.333 | - | n/a | 0/0 |
| openvino-whisper:whisper-base-int8-ov@CPU+prompt [cap] [rep] [chk] | office noise | scripted | 37 | 0.452 | 0.428 | 0.429 | - | 6/17 | 22/37 |
| openvino-whisper:whisper-base-int8-ov@CPU+prompt [cap] [rep] [chk] | office noise | spontaneous (held out) | 37 | 0.384 | 0.368 | 0.364 | - | n/a | 0/0 |

## Scripted WER by group

- openvino-whisper:whisper-tiny-int8-ov@CPU+prompt [office noise]: car 0.528, num 3.757, plate 0.814, phone 0.769, date 0.576, name 0.778, place 0.338, flow 0.359
- openvino-whisper:whisper-tiny-int8-ov@CPU+prompt [cap] [office noise]: car 0.528, num 1.12, plate 0.814, phone 0.769, date 0.576, name 0.778, place 0.338, flow 0.359
- openvino-whisper:whisper-tiny-int8-ov@CPU+prompt [cap] [rep] [office noise]: car 0.605, num 0.526, plate 0.814, phone 0.974, date 0.606, name 0.833, place 0.264, flow 0.359
- openvino-whisper:whisper-tiny-int8-ov@CPU+prompt [cap] [chk] [office noise]: car 0.528, num 0.545, plate 0.814, phone 0.769, date 0.576, name 0.778, place 0.338, flow 0.359
- openvino-whisper:whisper-tiny-int8-ov@CPU+prompt [cap] [rep] [chk] [office noise]: car 0.605, num 0.526, plate 0.814, phone 0.974, date 0.606, name 0.833, place 0.264, flow 0.359
- openvino-whisper:whisper-base-int8-ov@CPU+prompt [office noise]: car 0.512, num 0.334, plate 0.544, phone 4.026, date 0.256, name 0.611, place 0.44, flow 0.069
- openvino-whisper:whisper-base-int8-ov@CPU+prompt [cap] [chk] [office noise]: car 0.512, num 0.334, plate 0.544, phone 0.949, date 0.256, name 0.611, place 0.44, flow 0.069
- openvino-whisper:whisper-base-int8-ov@CPU+prompt [cap] [rep] [chk] [office noise]: car 0.467, num 0.34, plate 0.544, phone 1.0, date 0.256, name 0.667, place 0.551, flow 0.069

## Non-speech probe (excluded from WER)

| Engine | Condition | Clip | Typed text | Engine output | Words out/typed | Looped? |
|---|---|---|---|---|---|---|
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt | office noise | spont008 | blah blah blah blah blah | Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi,  | 108/5 | YES |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt [cap] | office noise | spont008 | blah blah blah blah blah | Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi,  | 20/5 | YES |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt [cap] [rep] | office noise | spont008 | blah blah blah blah blah | Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi,  | 20/5 | YES |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt [cap] [chk] | office noise | spont008 | blah blah blah blah blah |  | 0/5 | no |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt [cap] [rep] [chk] | office noise | spont008 | blah blah blah blah blah |  | 0/5 | no |
| openvino-whisper:whisper-base-int8-ov@CPU+prompt | office noise | spont008 | blah blah blah blah blah | La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La | 134/5 | YES |
| openvino-whisper:whisper-base-int8-ov@CPU+prompt [cap] [chk] | office noise | spont008 | blah blah blah blah blah |  | 0/5 | no |
| openvino-whisper:whisper-base-int8-ov@CPU+prompt [cap] [rep] [chk] | office noise | spont008 | blah blah blah blah blah | La Blah, La Blah, La Blah. | 6/5 | no |

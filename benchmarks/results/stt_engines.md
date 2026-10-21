# STT accuracy by recording condition

Conditions are reported separately and never pooled. **office noise** = recorded in an office with audible background noise; it is not the clean baseline. Spontaneous clips are a held-out set (never used for tuning); non-speech probes are excluded from the WER set. Generated 2026-10-20 15:17.

| Engine | Condition | Set | Clips | WER mean | WER mean (capped 1/clip) | WER median | Runaway (WER>2) | Numbers exact | Car-term recall |
|---|---|---|---|---|---|---|---|---|---|
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt | office noise | scripted | 37 | 1.259 | 0.55 | 0.5 | num06 | 3/17 | 20/37 |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt | office noise | spontaneous (held out) | 37 | 0.857 | 0.434 | 0.333 | spont029 | n/a | 0/0 |
| openvino-whisper:whisper-tiny-int8-ov@CPU | office noise | scripted | 37 | 1.456 | 0.599 | 0.636 | phone01, phone03 | 5/17 | 3/37 |
| openvino-whisper:whisper-tiny-int8-ov@CPU | office noise | spontaneous (held out) | 37 | 0.404 | 0.381 | 0.333 | - | n/a | 0/0 |
| openvino-whisper:whisper-base-int8-ov@CPU+prompt | office noise | scripted | 37 | 0.695 | 0.426 | 0.4 | phone02 | 7/17 | 24/37 |
| openvino-whisper:whisper-base-int8-ov@CPU+prompt | office noise | spontaneous (held out) | 37 | 0.362 | 0.338 | 0.333 | - | n/a | 0/0 |
| faster-whisper:tiny.en-int8+hotwords | office noise | scripted | 37 | 0.577 | 0.548 | 0.5 | - | 6/17 | 13/37 |
| faster-whisper:tiny.en-int8+hotwords | office noise | spontaneous (held out) | 37 | 0.466 | 0.381 | 0.333 | spont028 | n/a | 0/0 |
| faster-whisper:tiny.en-int8 | office noise | scripted | 37 | 0.601 | 0.58 | 0.6 | - | 6/17 | 3/37 |
| faster-whisper:tiny.en-int8 | office noise | spontaneous (held out) | 37 | 0.397 | 0.336 | 0.25 | spont028 | n/a | 0/0 |
| faster-whisper:base.en-int8+hotwords | office noise | scripted | 37 | 0.451 | 0.427 | 0.375 | - | 9/17 | 14/37 |
| faster-whisper:base.en-int8+hotwords | office noise | spontaneous (held out) | 37 | 0.51 | 0.341 | 0.25 | spont028 | n/a | 0/0 |
| moonshine:base | office noise | scripted | 37 | 0.385 | 0.366 | 0.333 | - | 6/17 | 14/37 |
| moonshine:base | office noise | spontaneous (held out) | 37 | 0.561 | 0.448 | 0.333 | spont028 | n/a | 0/0 |

## Scripted WER by group

- openvino-whisper:whisper-tiny-int8-ov@CPU+prompt [office noise]: car 0.528, num 3.757, plate 0.814, phone 0.769, date 0.576, name 0.778, place 0.338, flow 0.359
- openvino-whisper:whisper-tiny-int8-ov@CPU [office noise]: car 0.624, num 0.575, plate 0.67, phone 11.359, date 0.488, name 0.856, place 0.806, flow 0.127
- openvino-whisper:whisper-base-int8-ov@CPU+prompt [office noise]: car 0.512, num 0.334, plate 0.544, phone 4.026, date 0.256, name 0.611, place 0.44, flow 0.069
- faster-whisper:tiny.en-int8+hotwords [office noise]: car 0.498, num 0.525, plate 0.877, phone 0.59, date 0.347, name 0.989, place 0.889, flow 0.274
- faster-whisper:tiny.en-int8 [office noise]: car 0.761, num 0.417, plate 0.72, phone 0.564, date 0.697, name 0.656, place 0.847, flow 0.212
- faster-whisper:base.en-int8+hotwords [office noise]: car 0.512, num 0.309, plate 0.83, phone 0.744, date 0.451, name 0.667, place 0.301, flow 0.028
- moonshine:base [office noise]: car 0.334, num 0.35, plate 0.502, phone 0.641, date 0.444, name 0.744, place 0.333, flow 0.028

## Non-speech probe (excluded from WER)

| Engine | Condition | Clip | Typed text | Engine output | Words out/typed | Looped? |
|---|---|---|---|---|---|---|
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt | office noise | spont008 | blah blah blah blah blah | Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi, Al Bhabi,  | 108/5 | YES |
| openvino-whisper:whisper-tiny-int8-ov@CPU | office noise | spont008 | blah blah blah blah blah | blah blah blah blah | 4/5 | no |
| openvino-whisper:whisper-base-int8-ov@CPU+prompt | office noise | spont008 | blah blah blah blah blah | La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La, La | 134/5 | YES |
| faster-whisper:tiny.en-int8+hotwords | office noise | spont008 | blah blah blah blah blah | Si, Power Flag, horns, the Japanese as Zen, M Graphs is the lead, the open, heard, the, ef | 57/5 | YES |
| faster-whisper:tiny.en-int8 | office noise | spont008 | blah blah blah blah blah |  | 0/5 | no |
| faster-whisper:base.en-int8+hotwords | office noise | spont008 | blah blah blah blah blah | and the other one. | 4/5 | no |
| moonshine:base | office noise | spont008 | blah blah blah blah blah |  | 0/5 | no |

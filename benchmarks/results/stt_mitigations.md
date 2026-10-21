# Offline mitigations (condition: office noise, scripted clips only)

Applied one at a time to the same 37 recordings, each with one fixed setting (no parameter sweeps). The held-out spontaneous set was not used. Differences of a few points on 37 clips are within noise.

| Engine | Mitigation | WER mean (capped) | WER median | Runaway | Numbers exact | Car terms |
|---|---|---|---|---|---|---|
| faster-whisper:tiny.en-int8+hotwords | baseline (none) | 0.546 | 0.538 | - | 7/17 | 15/37 |
| faster-whisper:tiny.en-int8+hotwords | gain normalisation | 0.547 | 0.5 | - | 6/17 | 14/37 |
| faster-whisper:tiny.en-int8+hotwords | high-pass 100 Hz | 0.528 | 0.5 | - | 7/17 | 13/37 |
| faster-whisper:tiny.en-int8+hotwords | VAD trim (energy) | 0.576 | 0.538 | - | 5/17 | 12/37 |
| faster-whisper:tiny.en-int8+hotwords | denoise (noisereduce) | 0.681 | 0.7 | - | 2/17 | 12/37 |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt | baseline (none) | 0.55 | 0.5 | num06 | 3/17 | 20/37 |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt | gain normalisation | 0.569 | 0.556 | - | 5/17 | 20/37 |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt | high-pass 100 Hz | 0.551 | 0.538 | - | 3/17 | 20/37 |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt | VAD trim (energy) | 0.562 | 0.571 | phone02 | 4/17 | 19/37 |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt | denoise (noisereduce) | 0.773 | 0.818 | phone01 | 0/17 | 11/37 |

# Offline mitigations (condition: office noise, scripted clips only)

Applied one at a time to the same 37 recordings, each with one fixed setting (no parameter sweeps). The held-out spontaneous set was not used. Differences of a few points on 37 clips are within noise.

| Engine | Mitigation | WER mean (capped) | WER median | Runaway | Numbers exact | Car terms |
|---|---|---|---|---|---|---|
| faster-whisper:tiny.en-int8+hotwords | denoise (noisereduce, isolated env) | 0.738 | 0.8 | - | 2/17 | 10/37 |

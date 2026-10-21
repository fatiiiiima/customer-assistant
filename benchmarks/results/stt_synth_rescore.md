# Synthetic baseline vs real recordings, number matching old vs fixed

Synthetic = Phase 0b TTS clips (optimistic, clean). office noise = your 37 scripted recordings. Numbers exact is out of 17 numeric clips (num, plate, phone, date). 'old strict' is the value stored by the Phase 0b run; 'new strict' re-runs the same scorer now (differences = run-to-run variation, e.g. sampling fallback); 'new lenient' is the fixed digit matching.

| Engine | synth numbers: old strict | new strict | new lenient | synth WER old | synth WER new | office-noise numbers strict | lenient | office-noise WER (capped) |
|---|---|---|---|---|---|---|---|---|
| faster-whisper:tiny.en-int8+hotwords | 14/17 | 14/17 | 15/17 | 0.144 | 0.135 | 6/17 | 6/17 | 0.548 |
| faster-whisper:base.en-int8+hotwords | 13/17 | 13/17 | 15/17 | 0.119 | 0.119 | 9/17 | 10/17 | 0.427 |
| moonshine:base | 17/17 | 17/17 | 17/17 | 0.121 | 0.121 | 6/17 | 7/17 | 0.366 |
| openvino-whisper:whisper-tiny-int8-ov@CPU+prompt | n/a | 13/17 | 16/17 | n/a | 0.139 | 3/17 | 3/17 | 0.55 |
| openvino-whisper:whisper-base-int8-ov@CPU+prompt | n/a | 14/17 | 17/17 | n/a | 0.134 | 7/17 | 9/17 | 0.426 |
| sherpa:en-2023-06-26+hotwords | 14/17 | 14/17 | 15/17 | 0.212 | 0.215 | n/a | n/a | n/a |

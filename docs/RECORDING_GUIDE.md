# Recording guide: STT accuracy clips (Phase 0b)

These clips measure word error rate (WER) on car names, plates and spoken numbers.
They stay on this laptop: `benchmarks/recordings/*.wav` is gitignored.

## What to record
- The lines in `benchmarks/recordings/script.tsv`: **39 clips**, each about 3-6 seconds.
- Read each line **exactly as written**, in your normal speaking voice and accent.
  Numbers are written as words so the reference transcript is unambiguous.

## Format and file names
- **WAV, mono, 16 kHz, 16-bit PCM** (any recorder works; Windows Voice Recorder output
  is converted by the benchmark script if it is not already 16 kHz mono).
- Name each file by its id: `car01.wav`, `num03.wav`, `plate02.wav`, and so on.
- Put all files in `C:\dev\voice-agent\benchmarks\recordings\`.

## Tips for useful results
- Record in the room and with the microphone you will actually use for the demo.
- Leave about half a second of silence at the start and end of each clip.
- Do one take per clip. Don't re-record to be "clearer": we want realistic speech.
- Optional but valuable: record the 10 `car*` and 8 `num*` lines a second time with
  mild background noise (a fan or a TV low), and name them `car01_noisy.wav`, etc.
- Optional: a second speaker reading the same lines, named `car01_spk2.wav`.

If some clips are missing, the benchmark skips them and reports how many were scored.

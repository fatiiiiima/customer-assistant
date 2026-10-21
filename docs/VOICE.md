# Voice (Phase 3): how it works, how fast it is, how to run it

Everything runs on this laptop: no cloud call anywhere in the audio path.

```
microphone -> [half-duplex gate] -> Silero VAD -> Whisper (OpenVINO, CPU) -> DealerBrain -> Piper (sherpa-onnx, CPU) -> speaker
                                                         |
                                  DialogueEngine: intent (rules or Qwen3-4B on the iGPU), tools, pricing engine,
                                  state machine, model-written reply streamed sentence by sentence and checked
```

| Piece | Choice | Why |
|---|---|---|
| Framework | Pipecat 1.13 (BSD-2), `PipelineWorker` / `WorkerRunner` | the brief; checked against the installed 1.13 API (the old `PipelineTask` / `PipelineRunner` are deprecated) |
| VAD | Silero (bundled in Pipecat), `stop_secs` 0.3 | local, fast; Smart Turn v3 (also bundled) is NOT wired in yet |
| STT | OpenVINO Whisper **base** int8 on the **CPU** (`voice.stt_model`; tiny is ~0.1 s faster but mishears more), car-term prompt | Phase 0b winner; never on the iGPU (it would fight the language model) |
| Brain | `DialogueEngine.handle_stream` | same engine as the text chat |
| TTS | Piper **libritts_r, speaker 1** (owner choice; `tts.voice` / `tts.speaker` in config) fp32 via sherpa-onnx, CPU | Phase 0b: fastest first audio (40-80 ms here). **CC BY 4.0: attribution required** (LibriTTS-R). espeak-ng phoneme data is GPL-3: internal prototype only |
| Microphone / speaker | Pipecat `LocalAudioTransport` (PyAudio) | |

## Run it
```
demo_voice.cmd                          (microphone + speaker; add --headset if you wear one; --debug shows intent + who wrote the reply)
.venv\Scripts\python.exe -m dealer_core.cli.voice --list-devices
.venv\Scripts\python.exe scripts\voice_offline_bench.py --show-text     (no microphone: a synthetic caller, real-time, every stage timed)
```
Plug in AC and prefer a **headset**. With the laptop speaker the microphone hears the agent, so while the agent speaks (and 0.5 s after) the microphone is
ignored: you cannot interrupt it (no barge-in yet).

## Latency (end of the caller's speech -> first audio they hear)
Measured with `scripts/voice_offline_bench.py`, 6 scripted turns, real-time, iGPU language model, caller = synthetic clean voice (so STT is optimistic):

| Stage | Median |
|---|---|
| Endpointing (VAD stop) | ~0.3 s |
| Speech-to-text | ~0.15 s |
| Dialogue engine up to the first sentence | ~0.65 s |
| Text-to-speech first audio | ~0.1-0.5 s |
| **Total** | **median ~1.3-1.5 s, range 1.0-1.8 s** (first working version: 3.7 s) |

What got it there (each measured): the language model decodes at only ~18 tokens/s, so waiting for a whole reply costs 2-3 s. So in voice
1. the reply is **streamed**: each sentence is checked (`verify_sentence`) and spoken as soon as it is complete, while the rest is still being written;
2. the reply opens with a **very short reaction** (about 5 words) that is ready in under a second;
3. a sentence the model did not finish is never spoken; a fact of the standard wording that the model left out is added as it is; the question the
   system is waiting on is always asked (by the model, or by the system if the model did not);
4. **price, booking, read-back and hand-off turns are spoken from the exact standard wording** at once (no model-written wording in voice for those);
5. reply length is capped (~20 words, 48 tokens) and the model is stopped at a 2.2 s deadline.
Config: `voice:` section (`reply_deadline_ms`, `reply_max_words`, `reply_max_tokens`, `opener`, `stop_secs`); text chat keeps its longer replies.

## Privacy
The voice console and the turn log carry timings, the detected intent and flags only. What the caller says and what the agent says is not printed or written
unless you pass `--show-text` (agent side only; nothing the caller says is ever printed).

## Known limits
* No barge-in (half duplex); a headset allows `--headset` but still no interruption handling.
* Smart Turn v3 (better end-of-turn detection than a fixed silence) is bundled but not wired in; endpointing is plain VAD silence (0.3 s).
* STT numbers are from clean synthetic speech; real accents/noise will cost accuracy and some time. "four thirty" can come back as "430".
* One caller per process. Handling many callers at once is the next step (the iGPU model and the CPU speech models are shared resources).
* Arabic is detected and a human is offered; no Arabic speech recognition.
* Streaming checks sentences one at a time; the whole-reply checks (list order, kept counts) are diagnostics only in voice (shown in `--debug` notes).

## Transcript, pronunciation and fillers (added after first live test)
* `demo_voice.cmd` shows a live transcript of both sides (`YOU` = what the speech recognizer HEARD, which is also how you spot mishearing; `AGENT` = what the agent says,
  sentence by sentence). It is not written to disk unless you add `--save-transcript FILE`. Without `--transcript` the console shows only timings and intents.
* Spoken forms: `AGMC` is spelled "eigh gee em see" for the voice (Piper read it as a word, "aagmic"). Chosen by listening-test: a listener model heard "A G M C"
  correctly for 3 different speakers with this spelling, but "I G M C" with "ay". Also `GCC` and `SUV(s)`. The transcript keeps the written form (`voice/tts.py: PRONUNCIATIONS`).
* Interjections ("Ah", "Oh", "Ooh", "Hmm", "Ha", "Wow") at the start of a sentence are removed in code, and the prompts no longer teach them.

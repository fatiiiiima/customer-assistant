# Voice (Phase 3): how it works, how fast it is, how to run it

> **DISCARDED numbers.** Every latency figure in this file that is labelled "on battery" (the Smart Turn comparison, the Qwen3-8B timing, the agent-engine timing and the model-comparison table) was measured with the laptop on battery, which roughly halves speed. Do not use those numbers. They are being re-measured on AC power; the valid figures are in `logs/e2e_*.json` and in the section "Re-measured on AC power" below once it exists. The relative findings (which setting cut fewer sentences, which model passed more safety checks) are kept, the absolute milliseconds are not.


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
| TTS | Piper **amy_medium** (owner choice: Amy; `tts.voice` / `tts.speaker`), fp32, 22 kHz, via sherpa-onnx, CPU. First audio 30-150 ms, ~0.04 real-time factor. (`amy` = the older low-quality int8 build, ~3x slower) | **Licence unconfirmed ("See URL", Mimic3) and the medium build is finetuned from lessac (research-only): internal prototype ONLY, do not ship.** Safe fallbacks: libritts_r speaker 1 (CC BY 4.0), joe (CC0). espeak-ng data is GPL-3: internal prototype only |
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
Config: `voice:` section (`reply_deadline_ms`, `reply_max_words`, `reply_max_tokens`, `model_only`, `smart_turn`, `barge_in`, ...). The old short model-written "opener" (`voice.opener`) is OFF: replies start with the answer. Text chat keeps its longer replies.

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

## Timing in the transcript
With `--transcript` the call shows a `TIME` line per turn, e.g. `TIME : 1.4 s from the end of your speech to the first sound (waiting for you to finish 0.30 s,
speech recognition 0.21 s, reply 0.66 s, voice 0.18 s)`. Without `--transcript`, `--debug` prints the same as `[first sound 1.4 s after you finished: ...]`.
"End of your speech" is the moment the voice detector stopped plus its 0.3 s of silence, so the figure is what you would feel as the wait.

## Measuring: be on AC power
On battery the CPU is capped (about 2.0 GHz here) and every CPU stage slows down: Whisper base takes ~360 ms instead of ~190 ms. Always plug in and set Windows power mode to
Best performance before timing or demoing. (`scripts/demo_preflight.py` / `demo_check.cmd` checks this.)

## The website's voice button (AGMC portal)

`serve_portal.cmd` starts a WebSocket voice server (`python -m dealer_core.server`, port 8765) that the portal's mic button (`site/voice.js`) talks to.
One Pipecat pipeline and one dialogue Session per call; the models (LLM lane, Whisper, Piper) are shared, each behind a lock; calls past `--max-calls`
(default 2) are told "busy" and the page falls back to text chat. The agent sells the site's own cars (read from `site/data.js`), one showroom open 9 AM-8 PM,
fixed prices (no negotiation), nothing sent by message (`config/portal.yaml`). The page applies what the server says: transcript lines with response time,
car cards, and the booking (saved by the site's own `bookTestDrive`). Wire format: `src/dealer_core/server/protocol.py`.
`scripts/portal_call.py` is a synthetic caller (`--calls N` for several at once). Known limits: no barge-in yet; navigating to another page ends the call;
HTTPS pages need `wss://` (a TLS front); the reply writer keeps a little per-engine state shared between calls (to fix before the load test).

## Smart Turn (end-of-turn detection)

On by default (`voice.smart_turn`); `--fixed-silence` (voice CLI and server) turns it off. A short VAD silence (0.2 s) proposes the end of your turn, Pipecat's
bundled smart-turn v3 model (BSD-2, local, CPU) decides "finished" or "still going", and a held turn ends anyway after 1.0 s of silence.
Measured with the synthetic caller, on battery, 6 turns each (`scripts/voice_offline_bench.py --smart-turn [--pauses]`): median end-of-speech to first sound
1375 ms (fixed 0.35 s silence) -> 1195 ms. With a 0.7 s pause inside the sentence, fixed silence cut 2 of 3 sentences in half ("Hello, I'm looking for" was
answered as a greeting); Smart Turn kept 2 of 3 whole (it ended "Can I book a test drive?" before "for tomorrow afternoon", a complete question). Not yet
tried with a real voice and microphone.

## Keeping misunderstandings from coming back

* Short replies are read against the question the agent just asked (`nlu/expected.py`): "any budget", "doesn't matter", "100,000", "a white one" after a budget/make/colour/mileage question
  are decided by code, before the model is asked.
* A few unclear words are answered by asking the last question again ("Sorry, I didn't quite catch that. ..."), not by offering a colleague.
* `tests/data/real_utterances.yaml` holds the lines that went wrong in real calls, with the question that came before. Add every new miss there.
  `tests/unit/test_real_utterances.py` checks the code-decided ones; `scripts/replay_utterances.py --times 3` runs them all against the real model, three times each.

## The model drafts every reply (voice)

`voice.high_risk_drafts` (default true): price, booking and hand-off turns are worded by the model too, like everything else. Code still decides every fact
(the pricing engine, the slots, the booking) and checks every sentence before it is spoken: no new number, name or claim, and no "booked / confirmed / all set"
unless the system said so. Fallbacks to the standard wording: a sentence that fails a check, a missed deadline, a missing detail (the standard sentence is
appended, e.g. the read-back with the phone number). Cost measured on the synthetic caller: booking turns reach the first sound in about 2.0 s (was 1.3-1.5 s).

## Bigger model (Qwen3-8B), measured

`--backend openvino_gpu_8b` (OpenVINO/Qwen3-8B-int4-ov, Apache-2.0, 4.9 GB) works (thinking switched off, empty think block stripped) but is not the default:
on the iGPU, on battery, the first spoken sentence took 1.6-7.4 s (median ~3.7 s) against the 4B's 1.0-4.2 s on the same six-turn conversation; its replies were
mostly the same as the 4B's. Speech starts on the first clause (6+ words up to a comma), not the first full sentence.

## The agent engine (experimental): `serve_portal.cmd --agent`

`dialogue.engine: agent` (or `--agent` on the server / voice CLI). Each turn the model NAMES the next tool (search_cars, get_car, check_slots, prepare_booking,
confirm_booking, cancel_draft, handoff, end_call, none) and, for car tools, the stock id: about 5 tokens, 0.4-0.5 s. CODE then reads the arguments (filters, name,
phone, day and time) out of the caller's words with the same extractors the state machine uses, runs the tool against the real inventory/slots/bookings, and the model
speaks the result through the same reply writer and sentence checks. A booking needs a read-back and a yes (`confirm_booking`); code re-checks the car and the slot at that
moment. Asking the model for a full JSON object per turn was tried first: 1.6-3.8 s per decision (decoding is ~18 tokens/s), so the model only names the tool. A 1.7B
model as the decision-maker was tried and picked the wrong tool most turns.
Measured on the synthetic six-turn chat (battery, engine only): decision 0.4-0.6 s, first spoken sound 1.7-3.0 s (state machine: 1.0-2.5 s).

## Barge-in (talking over the agent)

On by default for the website server (`voice.barge_in`, `--no-barge-in` to turn off) and for the voice demo with `--headset`. When you start talking while the agent is
speaking, it stops at once: the voice stops synthesizing, the page is sent `{"type": "stop"}` and drops the audio it had queued, and the reply still being produced is
abandoned (the model stops). What you said goes through recognition as a normal turn; the history keeps the interrupted reply marked "(interrupted)". It needs echo
control: a headset, or the browser's echo cancellation. On open laptop speakers the agent would hear itself and interrupt itself, so the demo leaves it off without `--headset`.
Checked with the synthetic caller (`scripts/portal_call.py --barge`): three interruptions, three clean new turns.

## Model comparison (same prompts, this laptop, on battery)

`scripts/capture_prompts.py` records the exact prompts of a scripted 11-turn booking chat; `scripts/compare_models.py` replays them on each model.
Speed = tokens/s on a short prompt (`scripts/bench_llm_speed.py`); checks = share of the model's sentences the writer's safety checks accept; tool = agreement with
the current model's tool choice on the 11 decision prompts.

| model | tokens/s | sentences passing checks | tool choice | notes |
|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 int4 (current) | 19 | 79% | reference | repeats car details |
| Phi-4-mini-instruct int4 (MIT) `openvino_gpu_phi` | 22 | 90% | 9/11 | shortest, most compliant booking wording; weaker on lists |
| Qwen3-8B int4 | 9 | 90% | 11/11 | best wording, half the speed |
| Gemma-3-4B (Ollama; the OpenVINO build is gated on Hugging Face) | 12 | 81% | 10/11 | wordiest |
| Qwen3.5-4B (OpenVINO) | - | - | - | does not run on the installed OpenVINO GenAI (input port error) |

Speculative decoding (Qwen3-0.6B drafting for the 4B): +30-50% tokens/s but +0.15 s to the first token, so almost nothing for 10-25 token clauses: not enabled.
Prefix caching already works: back-to-back replies reach their first token in 0.34-0.55 s (the long system prompt is not re-read).

## End-of-turn settings, measured on AC power (Best performance), synthetic callers

36 utterances with a mid-sentence pause (pauses 0.45-0.7 s as the detector sees them) + 10 short answers; each setting run in the real pipeline. Same-setting repeat runs differ by about
+-3 cut sentences, so differences smaller than that are noise.

| setting | cut mid-sentence (of 36) | wait median / p90 | short answers: median / max wait |
|---|---|---|---|
| VAD 0.20 s, Smart Turn fallback 1.0 s (old) | 7 (4 in the repeat) | 281 / 1750 ms | 274 / 1796 ms |
| **VAD 0.20 s, fallback 0.6 s (chosen)** | 8 | 281 / **1113 ms** | 305 / **1265 ms** |
| 0.20 s, fallback 0.6 s + words rule | 11 | 266 / 1132 ms | 312 / 1157 ms |
| 0.15 s, fallback 0.6 s + words rule | 5 | 672 / 1048 ms | 298 / 1156 ms |
| 0.35 s, fallback 1.0 s | 5 | 547 / 1992 ms | 508 / 2078 ms |
| 0.35 s, fallback 0.6 s + words rule | 6 | 547 / 1391 ms | 462 / 1422 ms |
| 0.60 s, fallback 0.6 s + words rule | 3 | 867 / 1741 ms | 844 / 1765 ms |
| 0.20 s + length rule (rejected) | **20** | 250 / 725 ms | 203 / 1187 ms |

Findings: the shorter fallback cuts the slow tail by about 0.6 s at no measurable cost; a length rule is harmful ("Toyota" and "Can I book" are both short); the words rule (`voice.smart_peek`)
gave no gain on these voices and is off until real recordings show one. Speculative start (emulation on the same audio): beginning the work at the first 0.15 s of silence is wrong 4% of
the time with a 0.35 s confirmation window (0.6 s of work wasted over 24 starts) and 12% with 0.6 s; each correct start answers 0.20 s (0.45 s) sooner.

## Tap to talk (website default)

Tap the mic -> speak -> the server decides the turn is over and closes the mic (the browser releases the microphone: the recording indicator goes off) -> Wazir answers with the mic closed ->
tap to reply, or tap during the answer to interrupt it. `?voicemode=handsfree` (or `CONFIG.voiceMode = "handsfree"`, or `serve_portal.cmd --hands-free` as the server default) keeps the old open-mic call.
Measured with the synthetic caller streaming silence like a real mic: the server closes the mic 0.1-0.3 s after you stop, the first sound comes 0.6-0.8 s after you stop (agent + prefix; the
first turn of a call 1.1-1.8 s), and a tap over the reply interrupts it every time (9 s of speech came back instead of 36 s). Checked in the page with a fake microphone: listening -> thinking (mic released)
-> speaking -> waiting (green, "tap to reply") -> tap -> listening; tap while speaking -> listening again. No spoken greeting in this mode (the mic is open at once): the greeting is shown as text.

## Your own recordings (24 takes, headset or laptop mic, natural pauses)

Your pauses: median 0.27 s, 3 of 24 over the 0.5 s reference. Through the real pipeline: VAD 0.20 s / fallback 1.0 s (old) cut 5 sentences, VAD 0.20 s / fallback 0.6 s (current) cut 6, with the
words rule 6, VAD 0.35 s cut 5, VAD 0.60 s cut 0 but waits 953 ms (median) instead of 335 ms. The cuts are phone numbers dictated in groups ("050" | "123" | "4567"), "Is it..." with a 0.22 s pause, and
"I'm looking for..." with a 0.56 s pause. Whisper heard "Ahmed Khan" as "Edmmer Traun" once and "BMW X5, the grey one" as "BMW X5 Degrelo".

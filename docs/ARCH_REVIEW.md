# Architecture review (27 Oct 2026, read-only)

Sources: README, docs/, the code, and the newest report in `logs/` (`logs/cluster/*.log`, today 17:15–17:51: 3 workers, router, PostgreSQL, Ollama, this MacBook). Nothing was benchmarked. A number not in those sources is **unknown**.

## 1. What the system is

1. One WebSocket per call; the page sends 16 kHz PCM from an AudioWorklet and plays PCM back (`bmw-portal/site/voice.js`).
2. Per call, a Pipecat pipeline: MicGate (tap-to-talk) → Silero VAD → Smart Turn v3 → Whisper (OpenVINO, in-process) → DealerBrain → Kokoro/Piper TTS (`voice/pipeline.py`).
3. The agent engine routes most turns in code; the model only names a tool and writes wording, every sentence safety-checked (`agent/engine.py`, `dialogue/writer.py`).
4. A quality governor picks full / prefix / code wording per turn from the model's measured speed and sheds model calls when decisions slow down (`agent/governor.py`).
5. Each worker's call cap follows its measured p90 answer time (`server/admission.py`); the router hands new calls to a worker with room (`gateway/__main__.py`).
6. Call snapshots, the worker registry and bookings live in PostgreSQL, so any worker can resume any call (`shared/state.py`, `repos/pg_store.py`).
7. The model sits behind one interface: OpenVINO on an Intel iGPU, an OpenAI-compatible server (vLLM in production), or Ollama (`nlu/chain.py`).
8. Production shape: stateless workers of ~16 calls, one batched model server, one batched GPU recognizer (`voice.stt_url`), autoscaled on demand (`deploy/`, `docs/PRODUCTION.md`).
9. Budget: end of speech → first sound < 1 s (end-of-turn 0.2–0.35 s + recognition + decision + first sentence + first audio).
10. Today's run on this Mac: `run_cluster.py --workers 3`, each worker `--agent --prefix --backend ollama` (decisions on Ollama too), Whisper **base** int8 on CPU (`.env` overrides the portal profile; `models/ov` holds only `whisper-base-int8-ov`), Kokoro on ONNX Runtime, tap-to-talk.

## 2. Top 5 causes

**C1. Near-silence is transcribed and acted on, including hanging up** (problem 2, part of 1).
`voice/services.py` passes any segment whose loudest 0.3 s is above `MIN_RMS = 0.005`; `is_phantom` only catches a repeated word. Real speech in the log sits at level 0.09–0.26, yet segments at 0.008 / 0.013 / 0.015 produced 2–5 "words". Two became `end_call`: `worker3.log` 17:49:29 (level 0.015 → "Thank you for calling AGMC. Goodbye!", pipeline cancelled) and 17:49:59 (level 0.013, first turn of a call opened 4 s earlier → goodbye). Hang-ups MEASURED; that the words were Whisper inventions on room noise is INFERRED (transcripts are not logged). A room mic without earphones produces exactly such segments.

**C2. In tap mode an early end-of-turn loses the rest of the sentence for good.**
When the turn ends the server closes the mic (`server/app.py: turn_over` → `MicGate.close_after_turn`) and the page stops the track; speech after a 0.35 s pause (`voice.tap_stop_secs`) that Smart Turn calls complete is never heard, unlike hands-free where it becomes the next turn. In the log, 10 of 16 recognised turns are **1 word from 1.7–2.4 s of audio** (≈1 s of speech after pre-roll and stop silence); Smart Turn said COMPLETE in 15 of 16. Whether you were still talking: unknown. Your recordings showed 0.35 s cutting 2 of 24 sentences in quiet (`docs/VOICE.md`); a distant mic makes trailing speech quieter, so the VAD stops sooner (INFERRED). Files: `voice/endpoint.py`, `voice/micgate.py`.

**C3. The weakest recognizer, primed to guess.**
Whisper base int8's best measured result is 20/24 key words on your clean, close-mic recordings (`docs/SCALING.md`): one miss in six before any room noise. Its hint is 45 car names plus the agent's last sentence (`voice/prompts.py: stt_prompt`); on unclear audio Whisper copies the hint. The browser sends `noiseSuppression` and `autoGainControl` on (`voice.js: openMic`); AGC lifts room noise between words, feeding C1 (INFERRED). Accuracy without earphones: unknown, never measured.

**C4. On this Mac the model path is Ollama, and it blows the budget with one caller** (problem 1).
Model-decided turns with a single tab took 1.0–2.9 s to the first sentence: `worker2.log` 17:17:38 **2933 ms**, 17:48:04 1364, 17:48:13 1468, 17:50:12 1082; `worker1.log` 17:51:04 1333. Code-decided turns: 1–5 ms. Budget: 700 ms. Three workers interleave two system prompts (~540 tokens decision, ~1000 writer) on one Ollama, so its prompt cache rarely hits; its parallel-slot setting is unknown (nothing sets `OLLAMA_NUM_PARALLEL`). With three tabs the decisions queue and the 1.2 s deadline becomes the norm (INFERRED). The site's text chat loads a second model (`data.js: CONFIG.model = qwen2.5:3b`) if anyone types. Files: `nlu/ollama_backend.py`, `scripts/run_cluster.py: worker_env`, `config/default.yaml`.

**C5. Three tabs on one laptop = three full audio stacks, and three voices into one mic.**
The router spreads calls, so 3 tabs lit all 3 processes, each with its own Whisper, Kokoro (ONNX Runtime, default thread pool = all 12 cores), Silero, Smart Turn and event loop, beside Ollama, PostgreSQL and Chrome. Oversubscribed threads make output frames late, and the page's jitter cushion is 120 ms (`voice.js: play`), so late frames are gaps (INFERRED; CPU/RAM during the test unknown; `docs/PRODUCTION.md` saw RAM at 100% with this layout on a 32 GB laptop). Each tab also plays its agent out loud into the one open mic; browser echo cancellation reliably removes only that tab's own playback (GUESS).

## 3. Fixes, smallest first

1. **Never hang up on a whisper** (C1). **DONE 27 Oct.** Correction to the first draft of this review: it proposed raising `MIN_RMS` to 0.02–0.03. That was wrong. `docs/SCALING.md` measured that a gate of 0.012 turned the owner's 10%-volume clips from 20/24 heard to 0/24, and quiet or distant speech is problem 2, so a higher gate would make hearing worse. The two false goodbyes were at 0.013 and 0.015, inside that range. So the gate stays at 0.005 (now the setting `voice.min_rms`), and the fix is a guard instead: an `end_call` from a segment below `voice.end_call_min_rms` = 0.03 becomes "Sorry, are you still there?" (real turns in the log: 0.065–0.26, n=17: MEASURED gap). Not blocked on word count, which would stop every plain "bye". Side effect: a goodbye said very quietly must be said once more.
2. **Tap mode: don't close the mic on the first pause** (C2). **DONE 27 Oct as the simple version:** `voice.tap_stop_secs` 0.35 → 0.6 (your recordings: 0 cuts at 0.6 s vs 2 at 0.35 s, MEASURED). Cost, corrected from the first draft: about +0.3 s on every turn (synthetic callers, median 547 → 867 ms), which works against the < 1 s target; `VA__VOICE__TAP_STOP_SECS=0.35` puts it back. Still open and better: keep the gate open until the first reply audio and merge a second segment within ~1 s into the turn, which fixes the cut without the wait (INFERRED; touches the website's `voice.js`).
3. **Browser capture: try `noiseSuppression: false, autoGainControl: false`**, keep echo cancellation; one line in `voice.js` (GUESS until measured). **BUILT as an opt-in switch, not the default (27 Oct):** `?mic=raw` on the page URL turns off the browser's noise suppression and automatic gain (`voice.js` in the website repo, uncommitted there). Not made the default because every level logged so far (median 0.19) was measured WITH automatic gain, and the quiet-goodbye threshold (0.03) and the detector's volume gate (0.1) were set against those levels; compare the logged levels first.
4. **Measure hearing without earphones** with `scripts/study_hearing.py` on 24 room-mic takes, base vs small.en, hint on/off. That unknown decides C3. **DONE in part (27 Oct):** `scripts/hearing_check.py` replays a recorded call and scores it. On the 9 real room-mic turns of the Chrome call (no earphones): old 16-bit math 90% of words wrong without the hint and 15% with it; 32-bit math 16% without and 4% with. So the hint is worth keeping (4x fewer errors). NOT done: the 24 scripted room-mic takes are not on this machine, and `whisper-small.en` is not downloaded (its comparison needs a download, which needs your OK). Sample of 9: read differences of a few points as noise.
5. **Laptop layout: one worker, not three** (`--workers 1 --calls-per-worker 16`), `OLLAMA_NUM_PARALLEL=4`, and the text chat on the voice model (INFERRED, C4/C5).
6. **Cap threads per worker** (C5): ONNX Runtime intra-op threads for Kokoro / Silero / Smart Turn and OpenVINO `INFERENCE_NUM_THREADS` to 2–4 each (INFERRED). **MEASURED, NOT ADOPTED (27 Oct):** one turn of Whisper takes 242 ms uncapped, 296 ms at 4 threads, 382 at 3, 547 at 2; Kokoro one sentence 629 ms uncapped, 798 at 4, 975 at 3, 1350 at 2. With THREE recognizers at once: 595 ms uncapped vs 617 ms capped to 3 each, so capping gained nothing (the CPU itself is the limit: three simultaneous turns each take ~2.5x one turn). Silero already uses 1 thread and Smart Turn 1, so the review was wrong on those. Settings exist and are off by default: `voice.stt_threads`, `tts.threads`.
7. **Recognizer as a service here too** (C3/C5): `voice.stt_url` → a local OpenAI-compatible Whisper server (whisper.cpp / faster-whisper, `small.en` or `large-v3-turbo`); no worker change, the same path as 500 and 3000 callers (INFERRED). **BUILT (27 Oct):** `python -m dealer_core.sttserver` (the recognizer already on disk, behind the OpenAI transcription API; no download, no extra package), and `scripts/run_cluster.py --stt-server` points every worker at it. Checked with the workers' own client on 9 real turns: identical text to the in-worker recognizer (9/9), HTTP overhead not measurable (193 ms vs 196 ms), two turns at once run in parallel. It saves memory and start-up time with several workers; with one worker there is no gain. Opt-in.
8. **Model as a service instead of Ollama** (C4): llama.cpp server or MLX behind `runtime: openai`, `prompt_mode: static`, so both prompts share one prefix cache and the constrained `choose` works (INFERRED; first-token time here unknown). **WIRED, NOT RUN (27 Oct):** config entry `local_server` and `run_cluster.py --backend local` (checks the server, fills in its model name). Nothing to run it with is installed here (no llama.cpp, no MLX), and installing one means downloading it and a model, which needs your OK. Evidence for doing it is in section 7b.
9. **Beyond 500**: nothing new. Hold latency by adding workers and GPU replicas; never raise per-worker caps the probe did not measure (`docs/PRODUCTION.md §8`, simulated, not measured).

## 4. What NOT to change

* The governor / admission / shedding / router mechanisms and thresholds: they hold < 1 s at 500+; laptop symptoms are their inputs, not their faults.
* Tap-to-talk as the website default and barge-in off in tap mode; hands-free VAD barge-in on open speakers makes the agent interrupt itself.
* The model's role (tool name + wording; facts, prices and bookings by code; every sentence checked). Bigger models measured half the speed for the same words.
* Smart Turn with a short VAD and the phone-number silence rule; fixed silence and the length rule measured worse. Fix tap mode's close timing, not the detector.
* Whisper sharing a GPU with the language model (measured 0.9 → 1.6 s). Recognition gets its own batched service.
* The no-transcript logging policy; add levels, word counts and WER runs instead.
* Treating any laptop number as a production number: the capacity probe and load test on the real GPUs are the only sizing inputs.

## 5. The 500+ design: spread or pack, and what to change

**Keep spreading calls across workers; do not fill a worker to 16 before adding another.** Every call costs continuous CPU (per-frame VAD, Smart Turn, Whisper, Kokoro in one process and one event loop), so a worker at 16 calls is slower for all 16 than one at 5 (MEASURED on the laptop: the knee at 8–12 calls, the event loop behind at 32). Packing puts each new caller on the busiest process, where the tail latency lives; a packed worker that dies drops 16 calls; and a new worker takes ~45 s to start, so the caller who needs it waits. Least-loaded assignment plus an autoscaler at 70% headroom, and packing only on the way down (drain the emptiest workers: `deploy/k8s/voice.yaml` already has a 300 s scale-down window, 2 pods/min, 900 s termination grace) is the standard pattern and is what the code does.

Changes worth making, in order of impact:

1. **A `draining` flag** in the worker's registry report, set on SIGTERM, so the router stops assigning to a pod during its 900 s grace period instead of only after its heartbeat stops (`server/app.py`, `gateway/__main__.py: Assigner.pick`). Small, and it is what makes scale-in clean. INFERRED.
2. **A thinner worker.** The fleet size is set by CPU per call, mostly TTS and Whisper. STT already has a server path (`voice.stt_url`); give TTS the same (a batched GPU Kokoro/Piper service). A worker then does VAD, turn detection and orchestration and can carry far more than 16 calls, so 3000 callers means more GPU replicas, not hundreds of CPU pods. INFERRED, unmeasured.
3. **WebRTC (LiveKit) before real users.** Raw PCM over a WebSocket has no jitter buffer, loss concealment or codec; on a phone network that is where "voice breaking" will come from whatever the server does. Already planned; move it ahead of worker tuning.
4. **The model server is the bottleneck and the risk.** Two vLLM replicas, one spare GPU, prompts with a long shared prefix and short tail, 5-token constrained decisions, and the shedding already built. Alert on decision time, not only reply time.
5. **Redis for call snapshots and the registry** at thousands of callers (a few hundred small writes per second); PostgreSQL for bookings only. The code already supports both. INFERRED.
6. **Caps stay measured, not configured.** The 16 is a laptop number; the admission controller and the capacity probe make the design independent of it. Keep that property.

Not important yet: per-worker thread tuning, worker count arithmetic from laptop runs, and anything that reasons from the three-workers-on-one-Mac layout, which is the opposite of the production layout.

## 6. Under 1 s and conversational: what is left to win

The stages run one after another today: end of turn (0.2–0.35 s) → recognition (0.2 s) → decision (0.25–0.4 s) → first sentence (prefix 0 s, full 0.5–1.2 s) → first audio (0.03–0.15 s), all laptop figures. The remaining wins come from overlapping them, not from speeding one up. In order:

1. **Start every turn's work at the first silence, commit at the confirmed end of turn.** `voice/speculate.py` already does this when Smart Turn holds a pause; do it on every turn: at the 0.2 s VAD stop, recognise and ask for the decision; when Smart Turn confirms, the answer is half done. Measured wrong-start rate 4% with a 0.35 s window (`docs/VOICE.md`); a wrong start only wastes work. End of turn then overlaps processing. INFERRED gain ≈ recognition + decision on most turns.
2. **Streaming recognition** (sherpa-onnx zipformer, Parakeet, or chunked Whisper on the held audio): partial words while the caller speaks, so recognition ends ~0.1 s after speech does and the model's prefill can start early. Also helps hearing: the hint and the quiet gate work on growing text. INFERRED.
3. **One model call instead of two** on vLLM: a constrained tool name first, then the sentence streamed in the same generation, the tool run while the opener plays. Saves the second prefill and round trip (most of the 0.25–0.4 s decision time is overhead for 5 tokens). Probe on the real GPU; the two-call split exists because small models picked tools badly in JSON, which is a different failure. INFERRED.
4. **Prefill ahead of time**: send the prompt with the partial transcript so the prefix cache is warm and the final call prefills only the last words. Cheap once 2 exists. INFERRED.
5. **Fillers stay rare and measured** (the current rule: only when the expected wait says so, never twice in a row). Backchannels on long caller turns later.
6. **Make the code level sound human.** Under load most turns are prefix or code, so the ceiling on "conversational" at 500 callers is the phrasebook: variety, short sentences, a question at the end, no repeated facts. Writing work, scales for free.

Alternatives to the cascade: speech-to-speech models (300–500 ms, more alive) cannot have a sentence checked before it is spoken, still need tools for facts, are heavy per stream and mostly not local: not for an agent that must never invent a price. Audio-in language models (Ultravox / Qwen-Omni / Voxtral class) are the interesting middle: audio straight to the model, text out, the TTS and sentence checks kept; they remove the recognition stage and its error cascade, at the cost of a 7–8B model on the GPU. One probe when the hardware arrives, not before.

Verdict: the cascade with the governor, code-routed turns, verified wording and measured fillers is the right design. With 1–4 done on a datacenter GPU, a model-routed turn lands around 0.5–0.7 s (estimate, unmeasured); code-routed turns are already there. The floor that cannot be removed is the 0.2–0.35 s of silence that says the caller has finished.

## 7. Live test findings (27 Oct evening, MacBook, one worker, recorder on)

Added for diagnosis: `VA_DUMP_AUDIO_DIR` (records what the detector receives and what is sent; off by default; holds the caller's voice, local only, delete after use), the browser name in the per-call log line, and an always-on one-line warning per turn when the microphone audio arrives damaged (`voice/mic_health.py`).

**What the recordings show (MEASURED).**
* Server output is not the cause of the voice breaking up: replies are sent whole and ahead of time, no clipping, nothing late. Playback breaks in the browser (INFERRED).
* In one call, the first three turns lost 28–46% of the audio before it reached the server: pieces of ~0.3 s with ~0.4 s missing between them, no catch-up burst, and 0.7–1.0 s of exact digital silence after the tap. The last two turns of the same call arrived 100% complete. That is what produced "Yeah." for a whole sentence.
* The earlier "Yeah." call (21:33) had 0.7 s of detected speech at level 0.030. Its audio was not recorded.
* Controlled test, one tab at a time, fresh load: Safari and Chrome both delivered 100% of the audio, steady 20 ms cadence, level 0.19, and were understood first time. **Safari is not bad by itself**, and the damage is intermittent.

**Ruled out.** The server's quiet gate and goodbye guard (levels were 0.07–0.25, the guard never fired). Other calls on the worker: the 21:33 bad call had none open. The Python event loop: the damaged call had no catch-up bursts, so the audio never arrived.

**Still unknown.** What triggers it. It happened in the 21:33 call, 10–25 s into the 21:39 call (while an older page was still connected), and not in the controlled test. Candidates, none proven: the page releasing and re-acquiring the microphone on every tap, a page not in front or throttled, another page holding the audio device, CPU load on the laptop at that moment. The next occurrence will now show up in the worker log as a "mic audio damaged" line next to a "call N from <browser>" line.

**Not fixed, and why.** A fix to `voice.js` (keep the microphone stream open for the whole call and only gate sending, or wait for the first live audio before showing "Listening") is plausible but is a product decision, since the page deliberately releases the microphone between turns, and it would be a guess until the trigger is known.

### 7a. Root cause found for the mishearing on this Mac, and a routing bug (27 Oct, late evening)

**Whisper ran in float16 (MEASURED, important).** OpenVINO's CPU plugin defaults to float16 on this M3 Pro. Whisper base int8 then breaks on many inputs: it writes a word or two and then token 0 ("!") until its 448-token limit. Replaying the nine real turns of one call from the recorder, with no hint, all nine looped in float16 and all nine were transcribed correctly in float32 (about 50–100 ms slower per turn). Live, the hint masked most of it, so only one turn in nine looped: "No, I want to know something else about the Macan" came back as "No" and ~400 exclamation marks after 2.8 s. Changing the silence, level, hint, or splitting the audio did not help; only the precision did. This very likely explains a large part of problem 2 ("could not understand 60% of what I said") on this machine, and corrects section 2, C3, which blamed the model size and the hint. Not tested: the Windows/Intel path, which has a different default. Fix: `voice.stt_precision` = `f32` (default), applied to the CPU recognizer only (`voice/stt.py`).

**Safety net (built).** Output length is now capped by the audio length, so any loop costs ~0.4 s instead of 2.8 s, and a looped result is never passed to the agent as words: the caller hears "Sorry, I didn't catch that."

**Routing bug (reproduced, fixed).** "What do you have?" matched a "show the whole showroom by make" rule that ran before the active search was considered, so after "There are 4 Toyotas. Any specific model?" the caller was told about BMW, Audi and nine other makes. Now, with a search under way and no car in focus, "what do you have / what have you got / what are my options / show me what you have" lists the matches of that search. With no search under way, or with a car in focus, it is still the showroom. "I'm not looking for any specific model" is now a no-preference answer. Tests: `tests/unit/test_stock_routing.py` (the exact dialogue, on the demo inventory and on the website's own cars).

**Still open, seen in the same call.** After "tell me more about the Macan" the agent offered a visit or test drive in three replies in a row and repeated the same sentence twice. Not addressed here.

### 7b. Ollama and the prompt cache (MEASURED 27 Oct, M3 Pro, one caller, the real engine)

Every model call of a replayed conversation, from Ollama's own timing fields:

| Turn | Prompt read | Time to read it | Whole turn |
|---|---|---|---|
| Short prompts (5 of 8 turns) | 229–392 tokens | 0.3–0.6 s | 0.5–0.9 s |
| "Tell me about the RAV4" (fact sheet) | 1,726 tokens | 1.3 s | 2.7 s |
| "How does it compare for fuel economy" | 1,838 tokens | 1.5 s | 3.0 s |

The second long prompt shares its first ~1,000 tokens (the writer's system prompt) with the first, so it should have been read from the cache in a fraction of that time. It was read in full, because shorter prompts of a different shape ran in between and replaced the single cached prompt. That confirms the cache thrash of section 2, C4. Generation speed was 35–41 tokens/s; reading the prompt was 650–1,300 tokens/s. Two things follow:

* **The cheapest fix is a shorter writer prompt or a server with more cache slots.** `OLLAMA_NUM_PARALLEL` is not set on this machine (`launchctl setenv OLLAMA_NUM_PARALLEL 4`, then quit and reopen Ollama). That is the part of review fix 5 not done, because it changes the Ollama app, not this code.
* **`num_ctx` is 2048 and a prompt already reached 1,838.** If a prompt goes over, Ollama silently drops the oldest tokens, which is the start of the system prompt. Raising it to 4096 in `config/default.yaml` (`ollama.num_ctx`) is cheap; not changed here because it makes Ollama reload the model once.

To try a different server (needs a download, so your OK first). On a Mac with Apple silicon, either:

```
mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8003          (pip package mlx-lm; model about 2.3 GB, unmeasured here)
llama-server -m <Qwen3-4B-Instruct-2507 Q4_K_M .gguf> --port 8003 --parallel 4 -c 8192   (brew install llama.cpp)
```

then `uv run python scripts/run_cluster.py --backend local --workers 1`. Whether either is faster than Ollama on the long-prompt turns is unknown until it is tried.

## 8. Booking-conversation policy fixes (27 Oct, from the RAV4 transcript)

The agent asked "which day?" six times because none of "any day", "what day is free", "and today" (after closing) meant anything to the booking logic. Built in `agent/engine.py` (`_FLEXIBLE`, `_DELEGATES`, `_offer_next_slots`), tests in `tests/unit/test_booking_policy.py` (the transcript replayed at 22:07 after closing):

* "Any day", "whenever", "as soon as possible", "what's free", "when can I come" are answered with the next two free times, not a question.
* A day with nothing free is said ("There's nothing free today.") and the next times are offered.
* "Any day suits me" right after two times were offered takes the earlier one. "What day is free?" offers again.
* The day is asked for once; a second time in a row the agent offers times. Leaving the booking talk resets this.
* The recognizer hint after a day question now includes "any day", "whenever", "as soon as possible", "two in the afternoon".

Not changed: who writes these sentences. Booking questions are spoken by code word for word at every quality level (by design, so nothing is invented about a booking). "All replies from the AI" is a separate decision (section 6 and the hardware question).

## 9. Found while replaying the user's real recording through the live worker (27 Oct, night)

Nine real turns of the Chrome call, fed into the running worker in real time, with the new code (all of sections 7–8 in place):

* **Works:** "what do you have" now lists the Corolla, Camry and RAV4; "No, I want to know something else about the Macan" is heard correctly (no exclamation-mark loop); the first sound came 0.24–0.34 s after the end of speech on 8 of 9 turns (0.61 s on the other).
* **DONE (27 Oct, night): the repeated blurb and the pushing.** Two memories in the session. (1) Per car, which groups of facts were told (`about`, `condition`, `performance`, `features`, `practical`, `economy`, `ownership`, `interior`): "tell me more", "anything else", "something else about the Macan" and "what else can you tell me" now give the next two groups not told yet, a direct question such as "is it reliable?" is answered again each time, and when everything has been told the agent says so instead of repeating one (`agent/answers.py: next_facts`). (2) Invitations to visit: none after the caller says no to one (for the next six exchanges), none in two answers running, and otherwise at most one every third answer; asking about a visit lifts the silence (`agent/wording.py: offer_test_drive`, `Session.no_invite_until`). Also fixed on the way: the list of makes already read out was kept after a car was chosen, so "what else" after a makes overview listed makes again. Replayed with the real recording through the live server: no sentence told twice, no invitation after the No, first sound 0.25–0.35 s on all 9 turns. Tests: `tests/unit/test_not_repeating.py` (16, including ones that fail when each memory is switched off).
  Seen in the same replay and NOT fixed: in prefix mode the model rewords the closing question and sometimes drifts to a topic the sheet cannot answer ("How about its daily driving experience?"). Proposed: give it the code's neutral question and forbid new topics in the continuation prompt, or speak the code's question as it is.
* **DONE (27 Oct, night): the governor no longer tests the slow level on a caller.** It used to try the model-written level on the first turn after a start (nothing measured yet) and every 8th turn after, to see if the model had sped up. On this Mac that turn took 3.3 s to a first sentence and the caller heard "So," and nothing more. Now the model is measured by a probe: a small request with the shape of a model-written turn (the writer's long instructions, a car's facts, a short history), run once before the server accepts a call and then every `dialogue.probe_every_s` = 20 s, only at a moment when no caller's turn is using the model, and never counted as a turn (`AgentEngine.probe_model`, `server/app.py: probe_loop`, `QualityGovernor.record_probe`). While the speed is not yet known the governor plays safe (the code's facts and a short model question); `dialogue.explore_every` is 0 by default and the old live testing remains only as an option for the text chat. Measured on this Mac: 1,538 ms to a first sentence at start-up (budget 700 ms), 1,834–1,904 ms during the replay, so every turn of the replay used the safe level and none the slow one; with the real recording the "Yes, talk about the Macan" turn now comes out whole ("So, the Macan is a 2016 in black. It's done 91,000 kilometres and it's 149,000 dirhams."). On a fast model the probe finds it, and the next turn is model-written. Tests: `tests/unit/test_governor_probe.py` (19, including ones that fail when either old behaviour is put back).
  Caveats. (a) The probe is a real model request every 20 s. On Ollama with one cache slot a probe replaces the cached prompt, so the next turn after it may read its prompt in full; with `OLLAMA_NUM_PARALLEL` above 1 or a server with a prefix cache (vLLM) it does not. `VA__DIALOGUE__PROBE_EVERY_S=0` turns it off. (b) A model-written turn can still be slow when the model is usually fast. Not changed: the first-sentence deadline for the model-written reply is `voice.reply_deadline_ms` = 4,500 ms; a shorter one with a fall back to the code's own sentence would also bound that case.
* **Still open, 3: `voice.model_only` is `true` by default and described in the config as TEST MODE.** With it, a sentence that fails a safety check is dropped with nothing spoken in its place, which is how a reply can end after its opener. The production deployment already sets it to false (`deploy/`).
* **Still open, 4: `ollama.num_ctx` is 2048** and a prompt already reached 1,838 tokens (section 7b).

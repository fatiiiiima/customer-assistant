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

1. **(DONE 28 Oct) A `draining` flag** in the worker's registry report, set on SIGTERM, so the router stops assigning to a pod during its 900 s grace period instead of only after its heartbeat stops (`server/app.py`, `gateway/__main__.py: Assigner.pick`). Small, and it is what makes scale-in clean. INFERRED.
   DONE: on the first SIGTERM/SIGINT the worker sets `draining` (`server/__main__.py: drain_on_stop`), reports `max` 0 and `draining: true` at once (not at the next 2 s beat), turns a late caller away with `busy`, and `WorkerRegistry.pick` and the router's `Assigner.pick` skip it; its calls in progress finish as before. Checked with a real SIGTERM on a real uvicorn server (`drain()` called, cap 0; the process then exits with 143 because uvicorn re-raises the signal after a clean stop). NOT tested: a real Kubernetes rollout. Tests: `tests/unit/test_draining.py` (5; each of the four changes fails a test when removed). The autoscaler's capacity sum (`voice_call_cap`) drops by that worker's share while it drains, which is also true.
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

* **CORRECTED 28 Oct: more Ollama slots do not help.** Earlier text here recommended `OLLAMA_NUM_PARALLEL=4`. Measured with two temporary Ollama servers (1 slot and 4 slots, same replay): the two long fact-sheet turns took 5.3 s and 2.9 s with 1 slot and 5.1 s and 2.8 s with 4, and the second long prompt was read in full in both (about 1.5 s for 1,850 tokens). The Ollama app was not changed and does not need to be. What would help is a shorter writer prompt / fact sheet, or a server with a real prefix cache (vLLM). The first long prompt after a fresh server start was also slow (3.2–3.4 s against 1.3 s on a warmed one), so expect a slow first long turn after any restart.
* **`num_ctx` was 2048 and a prompt already reached 1,838.** If a prompt goes over, Ollama silently drops the oldest tokens, which is the start of the system prompt. **DONE 28 Oct: 4096** (`config/default.yaml`, `ollama.num_ctx`). It made no measurable difference to speed; Ollama reloads the model once when the value changes, and every client of the same Ollama should use the same value.

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
  The drifting closing questions seen in the same replay are fixed in section 10.
* **DONE (27 Oct, night): the governor no longer tests the slow level on a caller.** It used to try the model-written level on the first turn after a start (nothing measured yet) and every 8th turn after, to see if the model had sped up. On this Mac that turn took 3.3 s to a first sentence and the caller heard "So," and nothing more. Now the model is measured by a probe: a small request with the shape of a model-written turn (the writer's long instructions, a car's facts, a short history), run once before the server accepts a call and then every `dialogue.probe_every_s` = 20 s, only at a moment when no caller's turn is using the model, and never counted as a turn (`AgentEngine.probe_model`, `server/app.py: probe_loop`, `QualityGovernor.record_probe`). While the speed is not yet known the governor plays safe (the code's facts and a short model question); `dialogue.explore_every` is 0 by default and the old live testing remains only as an option for the text chat. Measured on this Mac: 1,538 ms to a first sentence at start-up (budget 700 ms), 1,834–1,904 ms during the replay, so every turn of the replay used the safe level and none the slow one; with the real recording the "Yes, talk about the Macan" turn now comes out whole ("So, the Macan is a 2016 in black. It's done 91,000 kilometres and it's 149,000 dirhams."). On a fast model the probe finds it, and the next turn is model-written. Tests: `tests/unit/test_governor_probe.py` (19, including ones that fail when either old behaviour is put back).
  Caveats. (a) The probe is a real model request every 20 s. On Ollama with one cache slot a probe replaces the cached prompt, so the next turn after it may read its prompt in full; with `OLLAMA_NUM_PARALLEL` above 1 or a server with a prefix cache (vLLM) it does not. `VA__DIALOGUE__PROBE_EVERY_S=0` turns it off. (b) A model-written turn can still be slow when the model is usually fast. Not changed: the first-sentence deadline for the model-written reply is `voice.reply_deadline_ms` = 4,500 ms; a shorter one with a fall back to the code's own sentence would also bound that case.
* **DONE 28 Oct, 3: `voice.model_only` now defaults to `false`.** (It was `true`, described in the config as TEST MODE.) A reply the model leaves short, or a sentence the checks drop, is now completed with the code's own facts and question; `VA__VOICE__MODEL_ONLY=true` brings the test mode back. Production files already set it to false. Earlier description, for the record: With it, a sentence that fails a safety check is dropped with nothing spoken in its place, which is how a reply can end after its opener. The production deployment already sets it to false (`deploy/`).
* **DONE 28 Oct, 4: `ollama.num_ctx` is 4096** (section 7b).


## 10. Batch A (28 Oct): five changes, each measured

**1. Closing questions no longer drift (DONE).** In prefix mode the model rewords the closing question the code asks. In the Macan replay five of nine times it asked something else ("Shall we talk about the next available model?" for "hear more, or book a test drive?", "How about its driving experience or features?" for "anything else about it?"). The check on that sentence looked at numbers, money, promises and repeats but never at whether it was still the same question. New rule (`agent/engine.py: _novel_words`, `_check_continuation`): every content word must be grammar, an ordinary way of asking or inviting ("like", "else", "see", "person", "visit"...), or already in the code's question, what was just said, the caller's words, the car's facts or the last lines of the call. Otherwise it is refused, a second try is made, and then the code's own question is spoken. The instruction to the model now says to ask exactly the given question. Tests: `tests/unit/test_closing_question.py` (19; the seven drifting sentences above, six ordinary rewordings that must pass, and the fallback), which fail when the rule is switched off. Cost: a reworded question that uses an unusual word is replaced by the code's question; less variety, never a wrong topic.

**2. Ollama (DONE, with a correction).** `ollama.num_ctx` 4096. The parallel-slots setting was measured and does not help (section 7b was corrected), so nothing in the Ollama app needs changing.

**3. `voice.model_only` false (DONE).** See section 9.

**4. Recognition starts at 0.2 s of quiet inside every turn (BUILT, NOW OFF BY DEFAULT: `voice.early_start_secs` 0; CORRECTED 28 Oct).** The "within noise" hearing claim below was wrong for the live path. Replaying the real recording through the real Silero detector and `SmartEndpointer` (the slice the early start really sends to recognition: the 0.5 s pre-roll plus the turn so far), 9 early starts, all confirmed: 16% of words wrong and 4 of 9 turns heard exactly, against about 4% for the normal path. The slices are clipped at the front ("is anything else…", "both, yes, talk about the macan" lost their first words). Longer pre-rolls scored worse (1.0 s: 28%; 1.5 s: 69%; 2.0 s: 57%) and I do not know why; alignment of the slices to the recording (`ALIGN=1`) found 4 of 9 slices and could not find the other 5 byte-for-byte, so the replay itself may be part of it. UNKNOWN: whether a fixed slice would hear as well as the normal path. Until that is measured it stays off: hearing is the first complaint, and the gain is 0.25 s. The offline numbers below were made with slices cut at the true speech onset, which the live path does not do. Turn on for trials with `VA__VOICE__EARLY_START_SECS=0.2`.

Original text of item 4 (kept for the record, the hearing line is superseded):

**4. Recognition starts at 0.2 s of quiet inside every turn (DONE, on by default).** `voice/endpoint.py` watches loudness against the turn's own peak (quiet = under 6% of the peak and under 0.01). After `voice.early_start_secs` = 0.2 s of quiet it starts recognising and the model's decision in the background (`Speculation`); if the caller speaks again that is dropped, if the turn then ends it is used as it is. At most 3 starts a turn. In tap mode the voice detector waits 0.6 s, so the answer starts up to 0.4 s sooner. MEASURED on the nine real turns through the live worker: the reply began 0.25–0.31 s after the end of the recording before, and 0.00–0.04 s after with it (eight of nine turns; the ninth was 0.6 s in both). HEARING, measured offline with the slice the early start gives: 6% of words wrong against 4% for the normal slice on nine turns; the difference is the word "Macan" being heard as mac and / market / Mazda, and it moves between 3% and 6% with the length of the tail whatever the direction, so within noise on this sample. Cost: wasted recognition runs at pauses that were not the end. `VA__VOICE__EARLY_START_SECS=0` turns it off. Tests: `tests/unit/test_early_start.py` (11, including a boundary bug found by them: ten 20 ms frames add up to 0.19999999999999998 s).

**5. The microphone stays open until the reply starts, and a pause no longer cuts the sentence (DONE, on by default).** Before: when the server decided the turn was over it told the page to close the microphone at once, so the rest of a sentence said after a pause longer than 0.6 s was never heard. Now (`voice.tap_grace_secs` = 2.0): the server tells the page `thinking`, keeps the microphone open until the first reply audio plays (at most 2 s), and if the caller starts speaking in that window (`ContinuationGuard`) the reply that was not yet heard is dropped and the first half is joined to the second (`DealerBrain.take_back`). Three cases: the first half's transcript has not arrived (it is kept and joined), it is being answered (the answer is cancelled and not remembered), or it was answered and is queued for the voice (the exchange is forgotten and the queue interrupted). A half sentence that gets taken back is never shown in the chat. MEASURED on one of the real sentences with a 1 s pause put in at the comma: old behaviour, the microphone closed at the pause, the agent heard "I'm not looking for any." and said "I didn't catch that"; new, one turn with the whole sentence and one reply. Limits: the join is the two transcripts side by side, so the chat line can read "...for any. A specific model..." (each half was written as its own sentence); whatever the aborted first half already changed in the call's state (for example the car in focus) stays; a caller who finishes and then waits is unaffected (the microphone closes when the reply starts, as before). The page change is in the website repo (`voice.js`: a `thinking` message; the old page still works, it just keeps showing "listening" until the reply starts). `VA__VOICE__TAP_GRACE_SECS=0` is the old behaviour. Tests: `tests/unit/test_tap_continuation.py` (17).

**Not changed.** `OLLAMA_NUM_PARALLEL` (no benefit). The `mic=raw` default (waits for the Safari comparison). Anything in the Ollama app.

## 11. A slow model-written turn, bounded; a shorter prompt (28 Oct, MEASURED on Ollama, M3 Pro)

Method: the real engine at the model-written level, the website profile, the same conversations before and after. 8 turns for the first table, then 32 turns on four makes (28 of them model-written) per variant. Temperature is not zero, so identical runs differ a little: three runs of the same variant gave 4, 4 and 5 rejected sentences, so differences of one or two mean nothing.

| | Before | After |
|---|---|---|
| Prompt read per model-written turn (8-turn run) | 1,596 tokens | 1,133 tokens (29% fewer) |
| Time to the first model sentence, 28 turns, median | 2,100 ms (not streamed) | 1,486 ms (streamed) |
| ...90th percentile | 2,679 ms | 2,048 ms |
| ...slowest | 3,830 ms | 2,550 ms |
| Sentences the checks rejected, 28 turns | 9 (old prompt) | 4 (new prompt; 4 and 4 on repeats) |

What each change did, separately:

* **Streaming the reply on Ollama (the biggest effect).** `OllamaBackend` had no streaming of its own, so a model-written reply was generated whole before the first word was spoken. It now streams (`lead_in_stream`), except for a reply begun by code (Ollama's chat cannot continue one). Median first model sentence 2.1 s to 1.5 s and the 3.8 s tail gone. Speaking only whole sentences first was NOT better (median 2.0 s, 90th percentile 4.4 s), so the first clause is still spoken early.
* **The first-sentence deadline.** `voice.first_sentence_deadline_ms` = 1500. If the model has no usable first sentence by then it is given up on (the helper thread stops it), the engine does not ask again, the governor is told at once, and the code's own wording is spoken (`FIRST_SENTENCE_DEADLINE`). Before: a turn that got nothing usable asked the model a second time, so a slow turn could cost two deadlines of 4.5 s. On this Mac 43-46% of model-written turns take longer than 1.5 s to a first sentence (11-14% longer than 2 s), so on this machine the deadline would fire often; the governor keeps this Mac on the prefix level for that reason (section 9), and the deadline is the bound for when it is wrong. On a fast server 1.5 s is far above the typical time. `VA__VOICE__FIRST_SENTENCE_DEADLINE_MS=0` turns it off.
* **The shorter instructions (`SYSTEM_VOICE`, website profile only, used when negotiation is off).** About 2,700 characters against 4,000: the counter-offer and greeting examples and rules are gone, every rule that keeps a sentence true is kept (tests check each one by its words). It did NOT make the first sentence sooner on Ollama (1.52 s against 1.47 s, noise), because Ollama already reuses the unchanged start of the instructions from its cache; it matters for servers without a prefix cache and for the cost per turn. It did not make the model fail the checks more (4 rejected sentences in 28 turns against 9 with the old text). The conversation shown to the model is 6 lines instead of 10 (`voice.history_lines`).
* **Facts given once.** The car in focus was listed twice (as the focus and as one of the cars shown), "prices are fixed" appeared in two wordings, and a line saying what tool ran ("What the customer wanted: get car.") told the model nothing. About 150 tokens a turn.

**Found while measuring, not caused by it. FIXED 28 Oct (see the end of this section).** When the first clause of a reply has been spoken ("Sure, that's the 2016 Porsche Macan, black,") and the next sentence then fails a check, the repair adds the code's facts after a dangling clause: "...black, That is: that's a 2016 Macan, black, 91,000 kilometres...". It appears in 5-7 of 28 model-written turns (18-25%) in every variant that speaks a clause first, including the old prompt (6) and with no streaming (6), and in none when whole sentences are spoken first. It is in the model-written level only, which is the main path on a GPU server and rare on this Mac. Fixing it properly means a repair that continues the spoken clause instead of restarting, or holding the clause until the rest has passed the checks (which costs the 0.6 s that clause-first saves). Proposed as the next change.

**The "That is:" repair, fixed (28 Oct).** What the caller has heard cannot be taken back, so the repair now finishes that thought and gives the code's facts as a sentence of their own: after a spoken comma "and here is the full picture." then the facts; after a half-word "Let me put that properly." then the facts (`_rounding_off` in `agent/engine.py`). Not re-measured on Ollama: I have not counted how often the situation now occurs, and how the new wording sounds spoken is for your ears to judge. Tests: `tests/unit/test_cut_reply_repair.py` (3; two fail on the old wording).

**My own defect, found live and fixed.** With the 1.5 s deadline the start-up probe returned nothing on this Mac ("not possible", `full_first_ms` 0), because the probe also hit the deadline. It now records the time it waited ("at least this slow") when the writer stopped on the deadline; any other empty result is still not a measurement. A second defect showed after the restart: the probe's canned question ("Tell me about the first one.") makes this model start "I'll tell you about...", which a check rejects, so it still said "not possible"; a rejected first sentence now counts as a measurement too (it took that long to write). And the probe flattered the model: the same probe twice took about 480 ms the second time (Ollama's prompt cache) against 1,100 ms for a new one, so the probe now asks a different question about three different cars each time, and with that it reads 1,050-1,330 ms on this Mac (a real turn's median was 1.5 s), so the governor stays on the prefix level here, as it should.

Tests: `tests/unit/test_slow_model.py` (18, each mutation above, including the probe, fails a test when the old behaviour is put back).

## 12. "Yes, I want to come and see the Macan." answered with more facts (28 Oct, Chrome, found by the user)

Three times in one call the agent asked "Would you like to come and see it?" and, to "I want to come and see the Macan", answered with the next facts (seats, boot, owners). Cause (`agent/engine.py: _fast_decide`): the rule that treats a car's name after a list as "tell me about that car" ran before the rule for "come and see", and the words "the Macan" matched it; the model was never asked. Fixed: a booking cue (come, visit, test drive, book, see it, a day) with see/come/visit/drive/try/book/test now starts the booking first, for the car named, picked or in focus. Not covered, on purpose: "I'd like to see the Macan" alone (it can mean the photos). Same call, other findings: with `whisper-small.en` all four "more about the Macan" turns were heard right (base heard "show", "sure", "Nike"), about 1.05 s per turn against 0.24 s; the default is still base. Tests: `tests/unit/test_come_and_see.py` (11; seven fail without the fix).

## 13. Quiet and whispered speech (28 Oct, MEASURED offline on the user's own recordings)

* **The voice detector is not the limit.** Real Silero detector, 14 real turns, speech made quieter with the room's noise kept: a segment starts on 14/14 turns down to 20% of the original level (12/14 at 10%) with the current settings, and on 14/14 even at 10% with confidence 0.5 and min_volume 0.03. Not changed: the recogniser fails first.
* **The recogniser is.** Scaling the clean 27 Oct call down to 15% changed nothing (3-4% words wrong), but that keeps the signal-to-noise ratio. With the room's noise left at its level, speech at 40% of the 28 Oct level already gave wrong words with both models ("What is your name?", "Please feel free to post your comments below"), at 20% and 10% mostly nonsense, and loops ("the second one, the second one...", "please, please, ...", which `clean_transcript` and `is_degenerate` catch). Whispering has no voiced sound at all, so it is worse than a quiet voice. whisper-small.en does not fix this; it needs more signal, not a bigger model.
* **What does not help (tried):** beam search 3 and 5 (no change on the real audio), raising the loudness before recognition (the noise rises with it), a simple spectral-gate noise remover (helped some turns, hurt others: not adopted).
* **Echo of the hint, found on the way, fixed.** The recogniser is given the agent's last sentence as a hint; with weak audio it writes the hint down. Live 28 Oct: "I want to come and see it" came back as "Would you like to come and see it?", the agent's own question. `EchoGuard` (`voice/services.py`) listens again without the agent's sentence in the hint when the words are the agent's own question (same first three words, 70% of its words), and asks the caller to repeat if it is still the question. It also covers the early reading during Smart Turn's hold. Offline, none of the recorded turns triggered it (the recorded slices differ from the live ones), so it is tested with a fake recogniser only (`tests/unit/test_echo_guard.py`, 15; 5 fail without it). Cost: one more recognition (about 0.25 s with base, 1 s with small.en) on the turns where it fires.
* **What is left, in order:** (1) the microphone: a headset or a closer mic is worth more than any model change; (2) ~~a noise remover in front of the recogniser~~ MEASURED 28 Oct, NOT ADOPTED: RNNoise (`pyrnnoise` 0.4.5, installed in `.venv` only, not in `pyproject.toml`; `pipecat.audio.filters.rnnoise_filter`) made hearing worse on 134 words from the user's own recordings (9% wrong raw against 22% with it on base, 4% against 16% on small.en; at 40% level 28% against 43% and 25% against 37%; at 20% level 38% against 49% and 41% against 49%), with the held-back tail fed through and with the best resampler, so it is not the filter's cut-off. Whisper was trained on noisy audio and does worse on the artefacts a denoiser leaves. Cost, had it helped: about 1 ms per 20 ms chunk, 5% of one core per call. Pseudo-references were used for the two 28 Oct calls (the clean-condition readings of small.en); the 27 Oct call has real ones. A different remover could be tried, but the cheap two have both failed; (3) on the GPU recogniser server a larger model (`large-v3-turbo`, 828 MB) hears better but still cannot recover speech buried in the room's noise.

**The "speak up" reply (28 Oct, built).** A turn whose loudest 0.3 s is under `voice.speak_up_below` (0.02), or under `voice.speak_up_suspect_below` (0.05) with words that look invented (a caption phrase such as "please subscribe", "comments below", "thanks for watching", a lone "okay" / "thank you" / "you", or a loop), is not acted on and not remembered; the agent says "Sorry, I can barely hear you. Could you speak up or move closer to the microphone?" (`voice/brain.py`, decision in `voice/services.py: too_quiet_to_trust`). After two such requests in a row the words are used as they are, so a caller who cannot be louder is not stuck. `VA__VOICE__SPEAK_UP_BELOW=0` and `VA__VOICE__SPEAK_UP_SUSPECT_BELOW=0` turn it off.
MEASURED on 252 readings (the owner's 18 turns with a known text, made quieter at 7 levels with the room noise kept, base and small.en): below 0.02, 81% of base readings and 66-81% of small.en readings were badly wrong (more than a quarter of the words); between 0.02 and 0.04 about 27-56% were; at 0.06 and above 0-15%. With the two settings above: small.en, 28 of its 39 badly wrong readings caught and 6 of 87 good ones stopped (those 6 were at level 0.010-0.018 and happened to be heard right); base, 22 of 47 caught and 6 of 79 stopped. What it cannot catch: a wrong but ordinary-looking sentence at a middling level ("Looking for the fish", "I want to show you more about the Macan"), 11 of small.en's 39 and 25 of base's 47. Limits: the quiet versions are scaled copies with added noise, not real whispers (there is no recording of a whisper), and the references for the two 28 Oct calls are the clean-condition readings. The owner's real turns in the 28 Oct Chrome call had levels 0.021-0.25, so ordinary speaking rarely triggers this. Tests: `tests/unit/test_speak_up.py` (27; each of four mutations fails some of them).

## 14. Fixes from the second review, B1-B4 (28 Oct; found by Fable in docs/ARCH_REVIEW_new.md, checked in the code, then fixed)

* **B1, the turn clock (DONE).** It was built with Smart Turn's 0.2 s, but tap mode runs the voice detector at `tap_stop_secs` 0.6 s (0.9 s after a phone-number question), so every "replied in" time, `voice_answer_seconds` and the admission p90 were 0.4 s short, and the call cap was held to about 1.4 s while believing it was 1 s. My own change of 27 Oct (0.35 to 0.6) left the clock behind. `TurnClock` now takes the pipeline's own current value (`brain.vad_stop_secs`, which follows the number-question change) and keeps the one that applied when the turn began. The earlier replay measurements timed from the end of the recording, not from this clock, so they stand. After this, the shown times will read about 0.4 s longer; the admission cap will therefore sit lower until the replies are faster. Tests: `tests/unit/test_turn_clock.py` (4; three fail with the old value).
* **B3, "what day and time are available" and ask, offer, ask, offer (DONE).** `_FLEXIBLE` now allows "and time" between the question word and "is/are available"; `_offer_next_slots` no longer resets `day_asks` to 0 (it keeps it at 1), so a non-answer after the offer is offered times again instead of asked "which day?" again. Tests in `tests/unit/test_booking_policy.py` (two old assertions changed on purpose).
* **B4, "any day would work" after an offer (DONE).** It no longer takes the earlier time silently: the agent says "Then shall we say <time>?" through the existing `time_check` path, and a "yes" books it, a different time is taken instead. The wording is new and has not been heard spoken.
* **B2, the Ollama deadline (DONE).** `OllamaBackend.lead_in` with a deadline reads the answer through a helper that gives up with `BackendTimeout` at the deadline and stops reading (which closes the stream and stops the model); `lead_in_stream` stops at the reply deadline too. Before, only the 20 s client timeout applied. MEASURED 32 real tool decisions on this Mac: median 106 ms, 90th percentile 499 ms, slowest 1,182 ms against the 1,200 ms deadline, 0 over; so on this machine the deadline is the net for a bad turn, not something that fires often. A model still reading a long prompt cannot be interrupted until its first piece arrives; the wait is cut at the deadline but the server finishes reading the prompt. Tests: `tests/unit/test_ollama_deadline.py` (8; three fail with the old code).
* **Not changed, on purpose:** `tap_stop_secs` (0.6) and the model's rewording of the closing question; see the discussion in the conversation. Not done from the second review: C5 (the hint echo in the SLOT state), B5-B8, the rest of the fix list.

## 15. Microphone test, and a bug in the speak-up reply (28 Oct, afternoon)

* **The input device was a generic USB-C adapter** ("AB13X USB Audio", default input, input volume 59/100), with the MacBook's own microphone unused. Test A (MacBook Pro Microphone in, headphones out; two calls, `20261028_125405` and `_125455`, base recogniser): 9 turns, 1 wrong word group ("Any days from" for "Any day is fine", which the booking logic still handled), replies 0.7-1.2 s as shown (0.6 s of that is the detector's wait). Test B (the adapter at volume 100; call `20261028_125720`): 8 turns, 5 with wrong or missing words ("I'm going to hear about the Macan", "I don't want to, do you have any Honda?", "I'm really more about this car", "What else is that can I talk to you about?", and one dropped, below). 1 of 9 against 5 of 8 is suggestive, not proof (Fisher exact p about 0.04 on 17 turns).
* **Level was NOT the difference.** The median level was 0.060 (A) and 0.059 (B); at adapter volume 59 it was 0.05. Chrome's automatic gain evens the level out, so the input-volume slider is not a lever (my earlier advice to raise it to 100 was wrong; measured: no change). The noise floor was 0.0030-0.0032 (A) against 0.0046 (B), about 3-4 dB. The reason the built-in mic hears better is therefore microphone quality and noise, not loudness. UNKNOWN: whether any other USB or wired mic would do as well.
* **Bug found in Test B, mine, fixed:** the speak-up check used the level of the LAST segment only. "I want to hear more about the Macan." was recognised correctly (level 0.041); the microphone, held open until the reply, then picked up a blip (level 0.018, "Yeah, I'm going to go ahead and see what happens."), the two were joined as one sentence, judged by the blip's level, and the caller was told to speak up although they had spoken clearly. Now a too-quiet piece that follows a clear first half is dropped and the first half is answered alone; two clear halves are still joined; two quiet halves are still refused. `voice/brain.py` (`_carry_level`, `_open_level`). Tests in `tests/unit/test_speak_up.py` (5 added; two mutations fail them).

## 16. Fixes 8 and 11 from the second review (28 Oct)

* **Fix 8, leave the agent's question out of the recogniser's hint (TRIED, MEASURED, NOT ADOPTED).** The idea (C5): in the booking states the hint ends with the agent's question and the recogniser echoes it ("day and time suit you" became "do and time otherwise"). I changed `voice/prompts.py` so the hint kept only the agent's statements, and scored old against new on every recorded turn where I know what was said (31 turns, 195 words, calls of 11:12, 11:31, 12:00, 12:40, 12:54), with both recognisers: base, old hint 31% of words wrong, new hint 36%; small.en, 27% against 33%. The new hint was better on 1 turn and worse on 4-5 for each model, among them the very turn the idea came from ("What day and time are available?" heard right with the old hint on base and wrong with the new). Small sample, so not proof either way, but there is no sign of a gain; the question's own words ("day and time") are vocabulary the answer reuses, and the echo case is already caught by `EchoGuard`. Reverted; `voice/prompts.py` is unchanged. Also not done: making `EchoGuard` a plain word-overlap test, because a real answer to "2 in the afternoon or 5 in the evening?" overlaps with it heavily and would be thrown away.
* **Fix 11, slot-offer openings made in advance (DONE).** The first piece of an offer of times is its first four words ("Today I can do", "The soonest is today:") and was made on the spot. `voice/pipeline.py: slot_offer_openings()` lists them from the same wording (`agent/wording.py: offer_slot_variants`, so they cannot drift apart) and `KokoroOV.preload(..., first_only=...)` renders only that first piece of each, at start-up: 25 sentences, 19 distinct pieces, +5.0 s at start-up, 24 more entries in a cache of 300 (112 used). MEASURED with the real voice on this Mac: time to the first sound 318-556 ms before, 0 ms after, for "Today I can do", "Tomorrow I can do", "The soonest is today:", "On Thursday I can", "On Monday I've got". Not covered: "Today I've got 9" and "Tomorrow I've got 9" (a time is inside the first four words, so the piece changes with the hour), that is one of the three wordings on those two days; offers more than a week ahead ("On Saturday, 7 November"). The rest of the sentence is still made while the opening plays. Tests: `tests/unit/test_slot_openings.py` (42; four mutations fail some).

## 17. "No." at the phone-number question was heard as digits; "3" as "See you." (28 Oct, 13:19)

* **The digits hint caused it (FIXED).** At "Can I take your name and mobile number?" the caller said "No." (a 0.2 s word) and the base recogniser, primed with `CONTACT` = "050, 055, 056, 052, double five", wrote "050, 056, 056, 0"; the agent took it for a number and asked for a name. Reproduced offline on the call's own audio: the short "No." with the digits hint gave "050, 056, 056, 056, ..." on base (small.en heard "no"), with no hint "No.". Measured on 12 synthetic short answers in this room's noise (base): "No, thanks.", "Not now." and "My name is Omar." all came back as a run of "056"s with the old hint; with "no, yes, not now" added to it they came back right, and a whole phone number was recognised the same (14 of 14 with both; small.en 14 and 13). Changes: `voice/prompts.py: CONTACT` now ends ", no, yes, not now"; `EchoGuard` listens again with no hint when a CONTACT-primed reading is only digits and either under nine of them or has a group repeated straight after itself (a real number is read in different groups); nothing readable the second time asks the caller to repeat. Limits: the digits test used synthetic speech, not real phone numbers spoken by the owner; room noise alone still produced "050, 057, 057..." on base with the new hint in one of two tries, which the retry catches. Tests: `tests/unit/test_contact_hint.py` (24; two mutations fail some).
* **"3" heard as "See you." (NOT FIXED).** The word is 0.4 s long; base and small.en both wrote "See you." / "See." / "City" with the production hint, with no hint, and with "three, five, 3 PM, 5 PM" added. The agent's fallback worked (it offered the times again and the next "3." was heard), so the cost was one repeated turn. Option, not built: when times were just offered and the words are a near-sound of exactly one offered hour ("see you", "free", "tree" for three; "fire" for five), ask "Just to check, did you say 3?" through the existing `time_check` step instead of offering again. It needs a list of sound-alikes that I cannot validate on one example.

## 18. The 13:30 call: a number that was not a number, and "nothing to move" (28 Oct)

* **A number that is not a whole number (FIXED).** "050, 0790, 054, 056" for 0507932546 is not a valid number; `ex.contact` keeps the digits it heard (`heard_digits`) but the rules sent the turn to the model, which said "none" after 2.3 s and the agent asked "Anything else you'd like to know?". Now, in the contact step, digits that are not a whole number go to `prepare_booking` by the rules (no model wait) and the agent says "I didn't catch the whole number. Could you say it again, a few digits at a time?", keeping the time chosen. The older state machine already did this (`machine.py: _give_contact`); the agent engine had lost it.
* **"Nothing to move" while booking (FIXED).** With nothing booked yet and times on offer, "Do." (for "two") and "I want to change the time" were decided as `reschedule_booking` by the model and answered "I don't see any test drives booked to move." Now, when no booking exists and a booking is being arranged, `reschedule_booking` becomes `prepare_booking` (and a time already chosen is let go if the words say change/different/another/move), and `cancel_booking` becomes `cancel_draft`. With a real booking the tools are used as before. Tests: `tests/unit/test_booking_in_progress.py` (6; three mutations fail some).
* **Digits recognition on a real voice is WORSE than I measured (NOT FIXED, needs a decision).** The 14 synthetic numbers read 14/14 with the digits hint; the owner's real number, re-recognised offline from the recording, was wrong on base with all four hints tried (the digits hint: "050, 0790, 056, 052, 052, 052 ...", a loop; no hint: "0.5.0.790.546"; a hint written as words: "zero five zero seven ninety two five four six", one digit missing), and right on small.en only with no hint ("0 5 0 7 9 3 2 5 4 6"). One real number is too few to conclude, but the direction is clear: base cannot be trusted for digits and small.en without a hint can. Option, not built: for the contact step only, use small.en without a hint (about +0.8 s on that one turn, about 250 MB more per worker, needs measuring on more real numbers). So the earlier statement in section 17 that the digits test showed no loss holds for synthetic speech only.
* **"No." at the contact question** still clears the draft and says "What would you like to change?"; that is the model's `cancel_draft` and was left alone (the caller did not flag it).


## 19. Phase 0: the baseline harness, rehearsed on the laptop (28 Oct; no GPU yet)

**Status: the Phase 0 gate has NOT been passed or failed. No GPU number exists. Every GPU figure is unknown.** What was done, as agreed: rehearse the whole procedure on this Mac against a local OpenAI-compatible server, fix what stops the three scripts from running, and write the runbook for the GPU session (`docs/PHASE0_RUNBOOK.md`). Nothing in `src/` changed.

**Machine for every number below (MEASURED conditions):** Apple M3 Pro, 12 cores, 36 GB, AC power. Model server `mlx_lm.server` 0.32 serving `mlx-community/Qwen3-4B-Instruct-2507-4bit` (4B, 4-bit, the model the project already uses elsewhere; **not** the 7 to 8B the plan names). Recogniser: `python -m dealer_core.sttserver` (Whisper base int8, OpenVINO, CPU), reached through `RemoteWhisper` like a GPU server. One worker, `--agent --prefix --backend gpu_server --max-calls 16`. The load generator, the worker, the recogniser, the model server, Kokoro and the user's own idle worker, Ollama and Postgres all shared the one machine (load average 3.4 to 5.5 before each run, CPU mean 22 to 41%, max 56 to 89%). So these are rehearsal numbers: they show the harness works and give a laptop reference, nothing else.

### Where the plan and the repository disagree (found reading the code; nothing built around them)

1. **The decision backend is not set by the production manifests.** `config.py:173` defaults `dialogue.decision_backend` to `openvino_gpu`; `app.py:113-117` builds the agent's tool-decision chain from it; `--backend gpu_server` selects only the reply model. `deploy/k8s/voice.yaml`, `deploy/Dockerfile` and `deploy/docker-compose.yml` never set `VA__DIALOGUE__DECISION_BACKEND`. INFERRED: in a pod the decisions are tried on `openvino_gpu`, `openvino_cpu`, `ollama` (`llm.fallbacks`), not on vLLM, so a Phase 0 run in that shape would not measure vLLM for decisions. MEASURED on the laptop: with `--backend gpu_server` alone the loaded configuration had `decision_backend` = `ollama` (from `.env`). **DONE after the owner's go-ahead (28 Oct), see "Decision backend" below.**
2. **Model size.** The plan says a 7 to 8B model; `deploy/k8s/voice.yaml` and `config/default.yaml` (`gpu_server.model`) say `Qwen/Qwen3-14B-AWQ`. Proposed: run the plan's two candidates first, the 14B only if the gate fails.
3. **`hearing_check.py` could not score the thing Phase 0 asks it to.** It built only the in-process OpenVINO recogniser (`WhisperOpenVino`); the GPU recogniser is reached through `RemoteWhisper`, which the script never used.
4. **"Time to first sentence" is not what `load_test.py` reports.** It reports the first *sound* (median, p90, worst, share over 2 s). The model's first sentence comes from `capacity_probe.py` at the matching concurrency and from the worker's `/health` `quality.full_first_ms`. The governor level per turn is the change in `/health` `quality.turns` (or `voice_turn_level_total`) across a run; no script records it. The runbook says how to take all three.
5. `capacity_probe.py` read a hard-coded `C:\dev\bmw-portal\site\data.js`, and the synthetic caller in `load_test.py` needs a Piper voice (`amy`) that this machine does not have.

### What changed, file by file

* `scripts/capacity_probe.py`: the website data path is `--portal-data`, default `../bmw-portal/site/data.js` beside the repository (the same file as before on the Windows layout).
* `scripts/hearing_check.py`: `--stt-url` and `--stt-model` score a transcription server (the workers' `RemoteWhisper`) instead of the in-process model; a new `build_engines()` holds the choice; failed requests to the server are reported rather than passing as a very bad score.
* `scripts/portal_call.py`: `caller_audio` accepts `kokoro:<voice>` for the synthetic caller on a machine with only the Kokoro model. `load_test.py` uses it through its existing `--voice`.
* `tests/unit/test_phase0_scripts.py` (4; all four fail with the script changes stashed).
* `docs/PHASE0_RUNBOOK.md`: new.
* Test suite: `pytest tests/unit` 904 passed, 40 skipped.

### Numbers (laptop, the machine above; single runs unless two are shown)

**Model first sentence, `capacity_probe.py --backend gpu_server`, budget 700 ms** (2 x N requests per level, so at 1 and 2 at once the p90 is the slower of two or four samples; two runs, A then B):

| At once | Full prompt, median / p90 ms | Prefix prompt, median / p90 ms |
|---|---|---|
| 1 | A 714 / 859, B 844 / 1,080 | A 629 / 665, B 514 / 536 |
| 2 | A 1,200 / 1,782, B 1,060 / 1,688 | A 952 / 1,435, B 478 / 675 |
| 4 | not run (sweep stops once p90 passes twice the budget) | A 717 / 947, B 600 / 900 |
| 8 | not run | A 975 / 1,654, B 949 / 1,143 |
| 16 | not run | A 1,982 / 2,880, B 2,633 / 4,408 |

For comparison, the 28 Oct Ollama probe on this machine read 1,251 to 1,479 ms for the full prompt (`docs/ARCH_REVIEW_new.md`, from `worker1.log`; different tool, same machine). The start-up probe of the worker read 620 ms on one start and 1,504 ms on another. The full prompt never met the 700 ms budget in the sweep, so `full_n` was 0 and the script's "callers per machine" lines are meaningless here. Its recognition and voice lines (~106 and ~12 callers) come from the laptop's own CPU Whisper and Kokoro, not from any server.

**Load test, one worker, `--mix --stagger 2`, caller voice Kokoro `michael` (the method in `docs/SCALING.md` uses Piper `amy`: not comparable with those tables):**

| Callers | Turns | Failed | First sound median / p90 / worst | Over 2 s | Governor turns full / prefix / code | `full_first_ms` start to end | Worker cap |
|---|---|---|---|---|---|---|---|
| 1 | 7 | 0 | 0.81 / 1.35 / 2.03 s | 14% | 0 / 7 / 0 | 764 to 1,168 | 16 |
| 8 | 56 | 0 | 0.96 / 1.90 / 3.06 s | 9% | 0 / 43 / 10 (3 not counted) | 1,168 to 1,055 | 16 |
| 16 | 115 | 0 | 0.91 / 1.40 / 3.34 s | 4% | 0 / 79 / 29 (7 not counted) | 1,055 to 1,155 | 16 |

(The governor counts for 8 and 16 callers sum to fewer than the turns because the counters are per turn that reached the engine; I did not find out why 3 and 7 are missing, so treat the split as approximate.) **The model-written level ran on 0 of the 168 counted turns (7 + 53 + 108)**, even though the server was faster than Ollama on the probe: its estimate sat at 764 to 1,168 ms against the 700 ms budget. The knee was not found: the three levels show no trend (p90 1.35, 1.90, 1.40 s), which at one run each is noise, and the load generator competed for the same cores. Not run: 4, 12, 20, 24, 32 callers.

**Hearing, `hearing_check.py`, nine recorded turns of 27 Oct, Whisper base int8, hint on / off:** through the recogniser service (`--stt-url`): WER 3% / 10%, 7 of 9 turns exact with the hint, median 214 ms. In-process f32: WER 4% / 16%, 7 of 9 exact, median 214 ms. Nine turns: the difference is noise. This proves the new option works end to end against the OpenAI transcription API; it says nothing about a GPU model.

**Probe options added for the GPU run (28 Oct, same laptop, MLX server, one run):** `--model-only --no-stop --levels 1 4` reached the level the old sweep never did: full prompt 856 ms median / 1,094 ms p90 at 1 at once (1.2 requests/s) and **2,167 ms / 2,682 ms at 4 at once (1.7 requests/s)**; prefix 682 / 749 ms at 1 (1.5 requests/s) and 837 / 932 ms at 4 (4.6 requests/s). So this server finished about 1.2 to 1.7 full-prompt requests a second whatever the concurrency, i.e. it does not batch them. The design rate for 500 callers is 17 to 33 per replica (GUESS, `docs/PHASE0_RUNBOOK.md` section 3). Single run at each level.

### What was learned about running the harness (all MEASURED)

* `mlx_lm.server` loads the model on the first request. That request took 1 min 42 s and `capacity_probe.py`'s first call timed out at 20 s. Send one real request to any new model server before measuring.
* With the model server on the same machine and a free `--backend gpu_server`, the whole pipeline (worker, remote recogniser, MLX server) ran 16 simultaneous calls with 0 failed calls and 0 silent turns.

### Decision backend set in the deployment files (DONE, 28 Oct)

Reasoning is point 1 above: `--backend gpu_server` picks the reply model only, `dialogue.decision_backend` defaults to `openvino_gpu` (`config.py:173`), and none of the deploy files set it, so a pod's tool decisions would have been tried on the Intel-GPU model, then the CPU and Ollama fallbacks, never on vLLM, and a Phase 0 run in the production shape would not have measured vLLM for decisions.

* `deploy/k8s/voice.yaml` (the `voice-worker` env), `deploy/docker-compose.yml` (`worker` environment) and `deploy/Dockerfile` (an `ENV`) now set `VA__DIALOGUE__DECISION_BACKEND=gpu_server`, the same name as the worker's `--backend`.
* `tests/unit/test_deploy_decision_backend.py` (4): loads the configuration the way a worker does (the website overlay plus the environment parsed out of each of the three files, with the laptop's `.env` not read) and asserts the decision backend is the backend named by that file's `--backend`, and is the first tried. All three file cases fail with the deploy changes stashed. A fourth test pins that the overlay alone still gives `openvino_gpu`.
* **`config/portal.yaml` was deliberately NOT changed, which narrows the instruction to set it there too.** That file is not a deployment profile: every laptop run loads it (`server/__main__.py:59`, `sttserver`, and every script), it already carries laptop settings (`voice.stt_model: whisper-small.en-int8-ov`, `stt_device: GPU`), and `serve_portal.cmd` on the Windows laptop passes no environment. Setting `decision_backend: gpu_server` there would send that laptop's decisions to `http://model-server:8000/v1`, which does not exist on it, and drop its iGPU decision model. (This Mac is protected only because its `.env` sets the variable, and environment beats the overlay.) The test above is what keeps the deployment and the overlay in agreement. If a deployment overlay is wanted later, it should be its own file, not `portal.yaml`; `config/default.yaml` is unchanged.

### Not changed, on purpose

All of `src/`; `config/default.yaml` and `config/portal.yaml`; the governor budget, deadlines and thresholds; the worker's request path. No data left the machine except the one model download (2.3 GB, `mlx-community/Qwen3-4B-Instruct-2507-4bit`, approved) and the `mlx-lm` package, installed in a separate environment outside the repository, not added to `pyproject.toml`.

### Still to do for Phase 0 (needs the GPU)

Run `docs/PHASE0_RUNBOOK.md`: the 7 to 8B candidates on vLLM, the real recogniser, the same three scripts, repeats, the knee at 4/12/20/24/32 callers, and the decision gate (first sentence under 300 ms at the design load). Until then Phases 2 and 3 are not unblocked.

## 20. Phase 1: the small open bugs from the second review (28 Oct; B5, B6, B7 built, B8 and C5 not, with the reasons)

No GPU needed, nothing spent. Branch `phase-1`, stacked on `phase-0`. Machine for every number: the same Apple M3 Pro laptop (36 GB, AC); the scale numbers are from the simulation (`scripts/simulate_scale.py`, deterministic given a seed), the AudioTap and database numbers from real runs on this machine. `pytest tests/unit` at the end of the phase: 924 passed, 40 skipped.

**Where the plan was wrong or overstated (found reading the code; each is explained under its item).** (1) B5: "probe at in_flight <= 2" recovered no faster and loads an overloaded model more; the expected 2.5k tokens/s saving belongs to the other option, not this one. (2) B7: the recorder is already off unless `VA_DUMP_AUDIO_DIR` is set, and its cost was about 0.3% of the loop at 16 calls, so it does not explain the 32-caller collapse. (3) B8: `voice_rules` does not check promises, the line is not 562, and "check at MAX" would flag 40% of ordinary replies. (4) C5: the plan's looser echo test throws away real answers, as section 16 had warned. The plan's "Done when" for Phase 1 ("no worker stuck at the prefix level while the model server is idle") is not a clean pass or fail in the simulation; what can be read is the time to recover, below.

### B5, probe starvation (PARTLY DONE)

* **Done:** a probe round is skipped when a caller's model-written turn timed the model within the interval (`QualityGovernor.measured_within`, `server/app.py: probe_loop`). The probe is the only way back UP from prefix; a worker writing at the full level is not below it, and its live turns keep the measure fresh. Idle workers and workers at prefix are probed as before.
* **Not adopted, the plan's other half:** probing when `in_flight <= 2`. It exists as `dialogue.probe_max_in_flight` (and `probe_loop(max_in_flight=...)`), default 0 (the old rule). In the scale simulation, mean of 5 seeds, protection on: after a GPU outage and return, workers were back under their budget in 66 s (skip only), 90 s (in flight <= 2, no skip), 66 s (as the plan: <= 2 plus skip) against 60 s with the old rule; with every turn on the model 126, 108, 132 against 132 s: no gain, differences of one 30 s sample. On a slow GPU, probes sent while the model already had more requests than slots rose from 659 to 872 to 904 (<= 2 and <= 4): +33 to 37%. The mechanism the plan inferred (a busy worker never finds a gap) is real only when a worker nearly always has a turn in flight, and that needs a model slow enough to be above the budget anyway.
* **Numbers (simulation, seed 7, protection on, 500 callers, 10 simulated minutes; "before" = `--old-probe`):**

| Scenario | Probe requests before / after | Answer p90 before / after | Seconds to recover before / after |
|---|---|---|---|
| fast GPUs | 1,257 / 61 | 0.88 / 0.88 s | n/a |
| weak worker CPUs | 1,546 / 118 | 0.96 / 0.95 s | n/a |
| spike 500 to 800 | 1,569 / 107 | 0.88 / 0.88 s | n/a |
| cold start | 1,110 / 48 | 0.88 / 0.88 s | n/a |
| GPU lost at 5 min, back at 7 | 1,372 / 409 | 0.88 / 0.88 s | 60 / 90 (5-seed mean 60 / 66) |
| every turn on the model, GPU lost and back | 1,539 / 577 | 1.37 / 1.37 s | 120 / 120 (5-seed mean 132 / 126) |
| slow GPU (never reaches the budget) | 1,197 / 1,188 | 0.83 / 0.82 s | n/a |

  The saving is the probe load of a healthy cluster: about 95% fewer probe requests (INFERRED to be about 95% of the plan's ~2.5k tokens/s: 45 workers x ~1,100 tokens / 20 s; tokens were not measured). The simulation reproduces the plan's worker-stuck-at-prefix only as the recovery delay after an outage: the column "workers held down while model idle %" is 10 to 36% in the outage scenarios under both rules, which is that delay, and 20 to 60% on the slow GPU, which is a model that is correctly held down. Read the recovery time, not that column.
* **The simulation changed.** `sim/scale.py` now runs the probe, counts the governor's in-flight turns as the engine does (`using_model`), starts every worker with `probed = True` as the server does, and reports probe load, probes sent while the model is overloaded, and seconds to recover; two scenarios (outage with recovery) were added. Its earlier numbers (`docs/PRODUCTION.md`, earlier sections) are not directly comparable.
* Tests: `test_governor_probe.py` (3 added: a live full turn makes the measure fresh and a probe does not; the loop skips a fresh round and probes a stale one; the in-flight limit), `test_sim_scale.py` (2 added: 5x fewer probes with the same answer time; recovery within 120 s). Two mutations each fail a test.

### B6, database connections (DONE, pgbouncer not added)

* A worker opened `PgDb` (pool 8: bookings, stock) and `PgStore`'s own `PgDb` (pool 4: call snapshots, registry): up to 12 connections, a ceiling of ~540 for 45 workers (INFERRED; a pool opens a connection only when a thread needs one, so how many are really open is unknown and was not measured). Now `repos/pg_store.py: shared_db(url, size)` gives a process one pool per database, `app.db_pool_size` (default 8) sets it, and `build_app`, `open_store` and `PgStore` use it: ceiling 8 per worker, ~360 at 45. That is still more than PostgreSQL's default 100.
* The router's registry read was a DELETE, a key list and one query per worker, every 0.5 s per router copy. `StateStore.items()` reads keys and values in one query (Redis: one MGET) and the clean-up DELETE runs at most every 30 s. MEASURED on a local PostgreSQL with 40 workers: 42 transactions per read before, 2 on the first read and 1 after. At 45 workers and two routers that is about 190 transactions a second before and 4 after (INFERRED by arithmetic). The router already cached its view (`Assigner.refresh_s` 0.5 s); the plan's "cache the registry view for a few seconds" was not needed.
* **pgbouncer was NOT added to `deploy/k8s`.** A manifest needs a pinned image tag and its settings, and I cannot check either offline; an unchecked manifest in the deployment folder is worse than none. Needed before 45 workers: pgbouncer in transaction mode (the code uses explicit `BEGIN`/`COMMIT` and transaction-scoped advisory locks, which that mode allows; I did not test it against pgbouncer) or `max_connections` raised. An owner decision.
* Tests: `test_pg_connections.py` (6; three mutations each fail some). `tests/unit/test_shared_state.py` is skipped on this machine (no `fakeredis`), so its memory/Redis/registry cases did not run here; the new tests cover the PostgreSQL store directly.

### B7, AudioTap (DONE, but the reason is smaller than the plan says)

* `voice/audiotap.py`: `record()` and `note()` now queue the chunk with the time it arrived and a thread of the tap's own writes, flushing about once a second and at the end of the call. Before: write + flush of the audio and of the index per 20 ms frame, on the loop. MEASURED (5,000 frames, 3 runs): 4.1 to 4.2 us per `record()` call before, 0.1 us after (p99 9.5 us before, 0.3 to 0.5 us after).
* **The plan's effect claim is not supported.** At 16 calls that is 800 frames a second x 4 us = 0.3% of the loop. The recorder is off unless `VA_DUMP_AUDIO_DIR` is set (`pipeline.py:166`, pinned by an existing test), and the 32-caller collapse in `docs/SCALING.md` (23 Oct) was measured before the recorder existed (27 Oct). B7 only matters in a development run with the recorder on, and then only a little on this SSD; a slower disk would show more. Test: `test_audiotap.py` (1 added; fails with the old tap).

### B8, voice_rules at MAX+2 (NOT DONE: the plan describes it wrongly; proposal)

* The check is `agent/engine.py:571` (not 562): `voice_rules.check(reply, max_sentences=MAX_SENTENCES + 2, max_words=90)`. `voice_rules` checks sentence count, word count, markdown, URLs, more than 3 cars and line breaks. It does **not** check forbidden promises: that is `dialogue/writer.py` (a new first-person promise is rejected sentence by sentence before it is spoken). The result is `dbg.violations`, shown by the CLI trace and the replay tool; nothing acts on it. The plan's Phase 3 line "Forbidden promises: `dialogue/voice_rules.py` as today, checked at the limit" points at the wrong module too.
* MEASURED on the 77 agent replies in the words logs: 1 sentence 1, 2 sentences 45, 3 sentences 13, 4 sentences 16, 5 sentences 2; median 18 words, none over 55. Checking at `MAX_SENTENCES` (2) would flag 31 of 77 (40%): every reply that is an opener, a fact group and a question, which is the design. The present limit (4) flags 2 of 77 and equals the writer's own `MAX_SENTENCES` = 4.
* **Proposed correction, not built:** leave the check where it is (it already matches the writer's limit), and make Phase 3's promise guard a function of `writer.py`'s existing checks. If a stricter diagnostic is wanted, set the limit per reply type (a fact reply is 3 to 4 sentences by design, a booking line 1 to 2).

### C5, the hint echo (NOT ADOPTED: scored, and the plan's rule loses)

* `scripts/score_echo_guard.py` replays the "stt" and "said" lines of `logs/audio_dump/*_in.txt` (79 recognised segments, 16 calls) through an echo test; it recognises nothing again, so it is free and repeatable. The agent's line for a segment is everything it said since the previous one.
* The test in the code (`echoes_agent`, first three words and 70% of the words): flags 1, a real echo ("Would you like to come and see it?"). The plan's rule (60% of the heard words in any one sentence the agent said, 4 or more words, no matching start): flags 4, the same echo and **three real answers**: "I want to know something else about the Macan." and "I want to sure know about the Macan." (to "Would you like to know anything else about the Macan?"), and "Two in the afternoon." (to "2 in the afternoon or 5 in the evening. Which would you prefer?"), the case section 16 predicted. Three wrong flags in 79 (3.8%) against none now; the plan's own exit (a day or time signal) would exempt only the last.
* The plan's second part, a plausibility exit in `_fast_decide`'s SLOT branch: the logs hold 12 answers to a day or time question. Four were garbled ("I would do and time otherwise at the moment.", "Needed.", "What days are we allowed to?", "Any days from."); the agent offered the next free times after each, which is the right recovery, and no garbled answer was routed into a booking. Asking again instead would be a worse reply than offering times. The review's example is not an echo at all: it shares 2 of its 9 words with the question, so no overlap rule at 0.6 could catch it. It is a hint-induced misreading, the thing Fix 8 tried to remove and could not.
* Nothing in `src/` changed. `test_echo_guard.py` gains a test pinning the three real answers as not echoes. Not tried: re-recognising the audio (the 28 Oct recordings have no reference lines, so a word-error score is not possible beyond the one 27 Oct call).

### Not changed, on purpose

Everything on the plan's "What not to change" list; `config/default.yaml` and `config/portal.yaml`; the governor's budgets, deadlines and thresholds; the router's mechanisms; the 500-caller capacity (no GPU number was taken); `deploy/` (no pgbouncer, see B6).

## 21. Scope decision: a laptop demo that can be shown to scale, not a GPU deployment (28 Oct, owner)

**Decision.** There will be no GPU to measure on. The product is a demo that runs on one laptop, built so that growing it to 500 and then 1,000+ callers is a matter of adding machines and a model server, and so that this can be shown on paper and in simulation. Anything that needs hardware nobody has stays unknown and is labelled so.

**Consequences for the plan (`docs/ARCH_PLAN.md`; the plan file itself is unchanged).**

* **Phase 0** is closed as "no GPU available". The baseline is the laptop's (section 19). `docs/PHASE0_RUNBOOK.md` is kept for the day a card exists.
* **Phase 1** is done as built (section 20; B8 and C5 deliberately not).
* **Phases 2 and 3**: their targets (first sentence under 300 ms, 90% of turns at the full level at 16 callers per worker) cannot be tested on a laptop whose model needs 0.6 to 1.2 s. What can be built and judged here is what does not depend on speed: one model call per turn, the MLX server as the local default, and the "model speaks, code guards" path behind a flag, judged on correctness and fact-safety by replays, not on latency. Not started.
* **Phase 4**: hearing and overlap can be scored on the recorded turns. Not started.
* **Phase 5** is replaced by `docs/SCALE_MODEL.md`: the scale policy is shown by the simulation and by arithmetic, with the unknowns listed.

**What was done for this decision.**

* `docs/SCALE_MODEL.md`: what the demo machine measures (section 1), the formulas and the simulation at 500, 1,000 and 1,500 callers (section 2), the unknowns (section 3), what each phase becomes (section 4).
* `sim/scale.py`, `scripts/simulate_scale.py`: three scenarios (1,000 callers with the manifest's 72 workers and with up to 120, 1,500 callers with up to 180) and `--model-capacity`, which reruns the cluster with less and less model capacity to find what each size needs.
* Results (MODELLED, stand-in speeds of 120 ms and 300 ms, which are targets, not measurements): the workers needed follow callers / (16 x 0.7): 45, 89, 132 in the simulation against 45, 90, 134 by formula; answer p90 0.88 s at all three sizes with under 0.3% turned away; the manifest's 72 workers hold 1,000 callers but about 87% full; the model needs about 36, 44 and 55 requests in flight at the target speed; with too little model capacity quality gives way first (e.g. 1,000 callers on 10 slots: p90 0.99 s, 0.0% turned away, 44% of model turns shed) and callers are not turned away.
* Not changed: `deploy/` (`maxReplicas` stays 72; for 1,000 at the 70% target it needs about 90, for 1,500 about 140: see SCALE_MODEL.md, to be set when there is something to test it on). The most likely first failure at scale is the database (720 connections ceiling at 90 workers against PostgreSQL's default 100); pgbouncer is still not added (section 20, B6).
* Tests: `test_sim_scale.py` (3 added: 72 workers hold 1,000 callers under a second; workers and model in-flight follow callers; too little model capacity sheds quality, not callers). `pytest tests/unit`: 927 passed, 40 skipped.

## 22. Phase 4 on the laptop: hearing and overlap scored on the recorded turns (28 Oct)

Scope (section 21): what can be judged without a GPU. Branch `phase-4`, stacked on `demo-scope`. **Nothing in `src/` changed: the scores did not justify a change.** Machine: the same M3 Pro, AC power, other processes running (the owner's worker and Ollama idle); recognition times are therefore laptop times.

### The tool: `scripts/score_overlap.py` (new)

Every recorded turn (the stretch between a tap and the end of the turn, cut as `hearing_check.py` cuts it) is played through the REAL Silero detector (tap-mode silence 0.6 s), the REAL `SmartEndpointer` and the REAL recogniser service, with Whisper replaced by an engine that only records the audio it is given; once with the early start off and once on. So the audio recognised is exactly what the live path would have recognised, not a slice cut by hand (section 10's early-start replay could not say that). Both are then read with the real model and compared. The endpointer keeps its own audio clock, so the result is the same at 20x speed and in real time (checked on three turns). 77 turns with speech in 17 recordings (10 recordings had no speech or no audio; `hearing_check.load_turns` crashed on an empty recording and now returns no turns). Tests: `test_score_overlap.py` (2), `test_phase0_scripts.py` (1 added).

Truth is thin: only the 27 Oct call (9 turns) has reference text. For the rest the script can use a larger recogniser's reading of the normal segment as a PSEUDO-reference: that tells which of two readings is nearer a better listener, not what was said.

### Hearing (MEASURED, the 9 turns with reference text, f32)

| Recogniser | Hint | Words wrong | Turns exactly right | Median time |
|---|---|---|---|---|
| base int8 | on | 4% | 7 of 9 | 263 ms |
| base int8 | off | 16% | 3 of 9 | 190 ms |
| small.en int8 | on | 7% | 7 of 9 | 992 ms |
| small.en int8 | off | 9% | 5 of 9 | 817 ms |

Nine turns: differences of a few points are noise. The hint helps base a lot (4 against 16) and small.en a little; small.en is no better with it and four times slower. Through the repository's own transcription service (`--stt-url`) base scored 3% with the hint. Nothing here supports a bigger model for hearing, as sections 13 and 15 found.

### Overlap: the early start (`voice.early_start_secs`), 77 turns, MEASURED

| Recogniser | Early start after | Turns with a usable start | Extra recognition runs per turn | Same words as the normal reading | Reply could begin sooner (median) | Against the 9 reference turns: normal / early |
|---|---|---|---|---|---|---|
| base | 0.2 s of quiet | 68 of 77 | 0.61 | 53 of 68 (9.9% of words differ) | 0.24 s | 4.5% / 4.5% |
| base | 0.3 s | 66 of 77 | 0.32 | 50 of 66 (9.7%) | 0.24 s | 4.5% / 3.0% |
| small.en | 0.2 s | 68 of 77 | 0.61 | 57 of 68 (8.1%) | 0.52 s | 7.5% / 6.0% |
| small.en | 0.3 s | 66 of 77 | 0.31 | 54 of 66 (9.0%) | 0.42 s | 7.5% / 6.0% |

* "Reply could begin sooner" is `min(time the start gained, the recognition run)`: the early start can hide only the part of the recognition that overlaps the endpoint's wait. The gain is 0.52 s (0.2 s start) or 0.42 s (0.3 s) of audio time, from a 0.6 s detector silence. With base (0.25 s per run) almost every usable start hides the whole run; with small.en (1 s per run) none does and about half of it is hidden.
* **Hearing is not clearly worse.** On the 9 reference turns the early reading is as good or better. Against the pseudo-reference (the other model's reading of the normal segment, 57 to 59 turns) the early reading was nearer on 4, further on 6 (base, 0.2 s), nearer on 2, further on 6 (base, 0.3 s), nearer on 2, further on 3 and nearer on 3, further on 3 (small.en): too few to say. About one turn in four (15 of 68 for base) is read differently, in both directions ("the mic" against "the Macan", "a quick push" against "a quick test", "tell me more about the Macan" against "...the matching").
* **Why the readings differ was looked for and not found.** On 17 turns checked by byte alignment the slice starts at the same place as the normal segment on 10 (0.00 s into the clip), 0.02 to 0.34 s later on 4 and 0.5 s later on 2 (the normal segment carries more lead-in there), and could not be located on 1; it ends 0.5 to 1.2 s (mostly about 1.0 s) before the segment does, because the normal segment includes the quiet up to the detector's stop. Appending 0.5 s of digital silence to the slice changed nothing at all (identical agreement, 53 of 68). Whisper pads its input, so what differs is the real room noise in the tail, which the slice does not have. Neither reading is known to be the right one when they differ.
* **Cost.** Each start the caller then talks over is a wasted recognition run: 0.61 per turn at 0.2 s (47 wasted in 77 turns) and 0.31 at 0.3 s. On this machine recognition shares cores with the calls' detectors, voice and model (section 4 of the first review), so the cost is paid by other callers.

**Left as it is: `voice.early_start_secs` 0 (off).** For one caller on the laptop the numbers favour 0.3 s (the same words as the normal reading on 4 turns in 5, 0.24 s sooner with base and 0.42 s with small.en, 0.3 wasted runs per turn); for many callers they do not, and the choice between them is the owner's. It is `VA__VOICE__EARLY_START_SECS=0.3`.

### Where the plan was off (ARCH_PLAN Phase 4)

* "Effect: 0.7 to 1.0 s per turn, MEASURED as the recognition time in the 28 Oct log": that is small.en's recognition time (0.9 to 1.1 s). An early start can hide only the overlap with the endpoint's wait: 0.42 to 0.52 s of it with small.en on this machine, 0.24 s with base. Hiding all of a 1 s recognition needs recognition that runs WHILE the caller speaks (streaming); no streaming recogniser exists on this machine (Whisper here decodes a finished clip), so that part is untested and stays a GPU-server item.
* "Treat the end of the tap gesture as the endpoint": needs the page to send the tap time; the page (`voice.js`) is in the website repository, and no recording carries a tap timestamp. Not scored. For reference, the endpoint wait in the replay (turn confirmed over, measured from the first early start at 0.2 s) is about 0.6 to 0.7 s after the last loud sound in tap mode, which is the 0.3 s per turn section 14 priced against the 0.35 s setting.
* "`?mic=raw` as an A/B on recorded turns": no recording was made with it; nothing to score. "Document the recommended microphone": the evidence is section 15 (the MacBook's own microphone 1 wrong group in 9 turns, the USB adapter 5 in 8; level was not the difference).
* "Digits": one real phone number exists in the recordings (section 18), so groups of three against a keypad cannot be scored. Unchanged.

### Not changed, on purpose

All of `src/`; the 0.6 s tap silence and the tap-grace join; Smart Turn; `min_rms`, the speak-up thresholds, f32; `early_start_secs` (stays 0).

## 23. Phase 3 behind a flag: the model speaks, the code guards (28 Oct; judged on correctness, not latency)

Scope (section 21): built behind `dialogue.model_speaks` (default **off**), judged on correctness and fact-safety by replays on the laptop, not on the plan's latency targets. Branch `phase-3`, stacked on `phase-4`. Machine for every number: the M3 Pro, model `mlx-community/Qwen3-4B-Instruct-2507-4bit` on `mlx_lm.server` (not the 7 to 8B the plan names; no GPU). Phase 2 (one model call per turn) is not built, so this runs on the existing decide-then-write path, as the plan's "staged" option allows.

### Where the plan and the code disagree (found reading the code)

* **Most of the guard already existed.** `writer.py: verify_sentence` rejects, sentence by sentence and before anything is spoken, invented numbers, names (capitalised), colours, seat counts, claims ("immaculate", "accident-free" when the facts do not say it), negations, "booked/confirmed" and first-person promises; a reply with nothing usable falls back to the code's wording. The exact booking, confirmation and read-back lines are already spoken by code at every level (`EXACT_TOOLS`). Phase 3 adds a stricter check, not the first one.
* **The model's question rewording (`_novel_words`) is on the prefix level only**, not the full one; there was nothing to remove at the full level.
* **"Tell me more about X" already answered X**; it then added the car's blurb (the words "tell me more" were themselves read as the topic "about"), and "its history" got the blurb first. Fixed at every level (below).
* **What really made the full level scripted:** an opener said before the model wrote anything, the code's closing question forced onto every reply (`writer.stream` appends it when the model does not ask it; the engine's cut-reply repair adds "anything else?"), and the model seeing only the next one or two fact groups, so it could not answer "and the boot?" from its facts.

### What changed, file by file

* `agent/answers.py` (all levels, not behind the flag): a named topic replaces the generic "tell me more"; "its history" means accident and service history. Tests: `test_tell_me_more_about.py` (10; 9 fail on the old code).
* `agent/guards.py` (new): `leaks(sentence, allowed, names)`: numbers (digits, spoken numbers over ten, years, prices), colours, makes and models of the stock (lower case too) and commercial terms (return, refund, warranty, finance, delivery, insurance, deposit, trade-in, discount, free, included, reserve...) must appear in **this turn's** facts, the code's wording or the caller's words. `verify_sentence` also accepts the whole call history, so a price of a car mentioned three turns ago passes it; this does not.
* `agent/engine.py`, `config.py` (`dialogue.model_speaks`), `app.py`: with the flag on, at the full level only:
  * no opener before the model;
  * the code's closing question is dropped when it is an invitation or a generic "anything else?" and there are facts to say; it stays in the booking steps (SLOT, CONTACT, CONFIRM, read-back) and when it is the whole reply ("Which car do you mean?"); the cut-reply repair adds no question either;
  * the model is given every fact group of the car in focus, what the caller has already heard, every aspect asked for (spec-sheet topics and colour, mileage, price, year), and where the booking stands (being arranged: do not confirm or suggest times; booked: do not offer another);
  * every sentence passes `guards.leaks`; a leak stops the reply there (fail closed) and the code's wording follows if nothing was said. Counters in `AgentEngine.fidelity`.
* `scripts/replay_model_speaks.py` (new): the five scripted conversations of `load_test.py`, a "tell me more about X" script and the heard lines of every recorded call, through the real engine at the full level, flag on or off.
* Tests: `test_model_speaks.py` (46): what the model is given, nothing added, the questions that stay, the guard failing closed (a figure, a year, a lower-case car name), the plan's fact-fidelity suite (5 faithful and 9 leaking sentences against a fixed fact list), the plan's forbidden-promise suite (17 sentences), and a known limit. `pytest tests/unit`: 986 passed, 40 skipped. Two mutations (guard off; question rule off) fail them.

### Numbers (MEASURED on the laptop; deadlines lifted for the correctness runs, because with the live 1.5 s first-sentence deadline every model-written reply on this machine fell back to the code's wording: 7 of 7 in the first try)

Same 17 conversations, 119 turns, once each:

| | Flag off (full level as before) | Flag on |
|---|---|---|
| Model-written replies | 45 | 44 |
| Fell back to the code's wording | 0 | 1 (the writer's name check) |
| Sentences the new guard stopped | n/a | 0 |
| Spoken sentences with a number, colour, name or commercial term not in that turn's facts | 0 | 0 |
| Replies ending on a question | 30 (67%) | 1 (2%) |
| Replies inviting a test drive | 14 | 0 |
| "Tell me more about X" answered with X | 6 of 6 | 6 of 6 |
| Scripted bookings made | 2 of 3 | 2 of 3 (the same two) |
| Model's first sentence, median / p90 | 1,278 / 1,715 ms | 1,603 / 1,967 ms |

**1,000 model-written replies, flag on** (381 conversations, 2,671 turns, 49 minutes): 0 spoken sentences with a fact not in that turn's facts; 3 sentences stopped by the new guard (0.3%) and 16 replies refused by the writer's name check and spoken in the code's wording instead (1.6%); "tell me more about X" answered with X 138 of 138; 24 replies (2.4%) ended on a question, none invited a test drive; 65 replies opened with the same three words as an earlier reply in the same conversation, and 35 fact sentences were said twice in a conversation; first sentence median 1,613 ms, p90 2,143 ms. The zero is checked by the same family of rules that enforce it (the replay re-checks every spoken sentence with `guards.leaks` against the facts the model was given), so it says the rules were never bypassed, not that every possible leak is a rule.

**What the checks do not catch (MEASURED, rough).** 23 of 1,578 model sentences (1.5%) judge or embellish rather than state: "known for strong performance and responsive acceleration" (most of them, word for word), "enough for groceries, luggage, or a weekend trip", "perfect for both the road and off-road". No number, name or commercial term is in them, so neither check stops them; the prompt forbids praise. And a value true of another attribute passes: 500 is the Macan's boot in litres, so "500 horsepower" would pass the guard (a test pins this as a known limit).

**Bookings.** In the repeated run every scripted booking reads "not made": after the first round the same slots are already taken in the in-memory database, and the agent correctly says "That slot isn't free" and offers others (checked on mix0, round 1 books Wednesday 4 PM, rounds 2 to 4 are offered 5 and 6 PM). The single-pass numbers above are the outcome. The third scripted booking (mix1) fails with the flag on and off alike: the script asks for 10 AM when 9 and 2 are offered, the agent checks "did you say 10 AM?", and the script answers with a name, which this laptop's model was too slow to decide (the code decided).

### Against the plan's "done when"

* "A 20-turn conversation with no two replies with the same opener": not met. 65 of 1,000 replies repeated an earlier reply's first three words in the same conversation, mostly "It has a..." and "The boot has..." in the "tell me more" script. Without the code's openers the repetition is the model's own.
* "'Tell me more about X' answers X": met, 138 of 138.
* "The fact-fidelity suite passes 1,000 replayed turns with zero leaks": met in the sense above (0 spoken; the guard and the writer stopped 19).
* "Latency must not regress against Phase 2": Phase 2 is not built. Against the full level as it was, on this laptop: +325 ms at the median (the whole spec sheet is a longer prompt). On a server with a prefix cache most of that is cached; not measured.

### Left for the owner (not decided here)

* **Questions and invitations.** The plan removes the forced closing question; the model then almost never asks one (2.4%) and never invites a test drive. The pacing rule that already exists (`wording.offer_test_drive`: no invitation after a no, none in two answers running, at most one in three) could keep an invitation when it is due while dropping the generic "anything else?". Not built: it is a sales decision.
* **Embellishment.** A list of praise words as a third check would stop most of the 1.5%; it would also stop some true statements of the site's own blurbs. Not built.
* **The flag stays off.** On this laptop the full level is reached only with the deadlines lifted; turning it on changes nothing a caller hears until the model is fast enough for the full level. Listening tests (ten replies per state, heard by a person) are not done.

### Not changed, on purpose

The prefix and code levels; exact booking and summary lines; `verify_sentence`; the governor; `model_only` false; the opener and the forced question with the flag off.

### 23a. Invitations kept when due (28 Oct, owner's decision), re-measured once

With `model_speaks` on, only the generic "anything else?" closer is now dropped; an invitation to visit stays, and `wording.offer_test_drive` gives one only when it is due (none after a no, never two answers running, at most one in three). Tests: `test_model_speaks.py` (48; the first version's rule fails 3). Same 17 conversations, one pass, same laptop and model, deadlines lifted:

| | Flag off (run above) | Flag on, first version | Flag on, invitations kept |
|---|---|---|---|
| Model-written replies | 45 | 44 | 44 |
| Fell back / guard stops / spoken leaks | 0 / n.a. / 0 | 1 / 0 / 0 | 0 / 0 / 0 |
| Replies ending on a question | 30 | 1 | 15 |
| Replies inviting a test drive | 14 | 0 | 14 |
| Same opener as an earlier reply in the call | 0 | 1 | 3 |
| "Tell me more about X" answered with X | 6 of 6 | 6 of 6 | 6 of 6 |
| Scripted bookings | 2 of 3 | 2 of 3 | 2 of 3 (the same two) |
| First sentence, median / p90 | 1,278 / 1,715 ms | 1,603 / 1,967 ms | 1,668 / 2,214 ms |

The invitations are back at the rate the code had (14), and replies end on a question half as often as with the flag off (15 against 30): the generic closers are gone. **The 1,000-reply figures above are from the first version and were not re-run** (stopped by the owner, to be repeated after a listening test); a single pass of 44 replies is too few to restate them.

## 24. The owner's test call (28 Oct, evening): five faults, fixed

The owner called through the website with Phase 3 on (`model_speaks`, `--quality full`, MLX on the laptop, first-sentence deadline off, the whole-reply deadline left at 4.5 s). The call booked in the end, but only after a second "yes". Branch `fix/test-call-28oct`, stacked on `phase-3`. The model was too slow to decide most turns on this laptop (decision deadline 1.2 s), so the code decided alone, and four of the five faults are in that path, which predates Phase 3.

| # | What the caller heard | Cause (code) | Fix |
|---|---|---|---|
| 1 | After the read-back: "yes, please book it" → "Sure, it's had 2 previous owners... Want to book a test drive?" | `_PLAIN_YES` accepts "yes please" but not "yes please book it"; the model then timed out and `_code_decision` returns "the car in focus" → `get_car` whatever the booking state | "book it", "go ahead", "confirm it" and the like count as a yes to the read-back (`_BOOK_YES`, `_confirms`); during a read-back the code's own decision confirms, fixes or reads back again, never the car's facts |
| 1b | (found replaying the call) "I want to see the car, I want to go on a test drive" while the agent waited for a day → the car's facts; the booking was lost | the same `_code_decision` gap in the SLOT and CONTACT steps | while a booking is being arranged the code's own decision carries it on (`prepare_booking`, which asks for what is missing) |
| 2 | "The 2019 BMW X5 is white, Would you like to come and see it?" | the 4.5 s reply deadline cut the model after its first clause, and `writer.stream` then appended the closing question after the comma | after a reply cut at a comma the writer adds nothing; the engine finishes the sentence ("and here is the full picture." + the facts) and then asks |
| 3 | "...340 horsepower, and here is the full picture. It has a 3.0-litre I6 turbo engine with 340 horsepower." | the section 11 repair added the code's facts even when the clause had just said them | after a cut clause, facts already covered are left out; if nothing is left the clause ends "and that's the main thing." |
| 4 | "I want to see the car." right after "Would you like to come and see it?" → more facts | section 12 left "I'd like to see the Macan" out on purpose (it can mean the photos) | a visit when one of the last two replies invited the caller and the words are "see/look at/view the car, it, the <model>", not photos, pictures, images or videos; without an invitation, unchanged |

Replay of the call's ten caller lines through the engine with a model that never decides in time (as on the laptop), clock at 21:00 on 28 Oct as in the call (`test_test_call_28oct.py`): before, the booking was lost at line 6 and the call ended asking "Which car would you like to test drive?"; after, every line lands as intended and "yes, please book it" books Thursday 29 Oct at 9 AM on the first yes, with the same read-back the owner heard. Tests: 23 in that file; each of the six changes fails some of them when removed. `pytest tests/unit`: 1,011 passed, 40 skipped.

**Why the call was slow (4 to 5 s on the model-written turns, 0.7 to 0.9 s on the others).** At `--quality full` every answer about a car is written whole by the model: on this laptop about 1.6 s to its first sentence (section 23), after a decision call of about 1 s on turns the code cannot route, one request at a time. The code-answered turns (the slot offer, the read-back) were 0.7 to 0.9 s. The default (`--quality auto`) would have put this laptop on the prefix level (the code's facts at once). And the test instructions I gave left the 4.5 s whole-reply deadline on, which cut the longer Phase 3 replies after their first clause and produced faults 2 and 3; `VA__VOICE__REPLY_DEADLINE_MS=10000` avoids the cut on this machine at the cost of a longer wait.

### 24a. The call replayed with the real model (MLX, the laptop; `replay_model_speaks.py --only testcall`, first-sentence deadline off, whole-reply deadline 4.5 s and 10 s)

The replay script's clock is 11:00 on 27 Oct, not the call's 21:00 on 28 Oct, so the times on offer were today's 12 PM and 4 PM instead of tomorrow's 9 AM and 2 PM. That difference exposed a worse fault than the five above, and the replay found two more:

| # | Seen in the replay | Cause | Fix |
|---|---|---|---|
| 6 | With 12 and 4 PM offered, "9 in the morning." → "Perfect, today at 12 PM" and the read-back booked **12 PM, a time the caller never said** | `parse_when` did not read a digit followed by "in the morning / afternoon / evening" (the way the agent itself offers times: "9 in the morning or 2 in the afternoon"), and `_offered_pick` then took "morning" as "the earlier offered time". In the real call 9 AM was on offer, so the hour matched and this was hidden | the parser reads "9 in the morning", "2 in the afternoon", "10 at night"; a named hour that was not offered never picks an offered one; the word "one" in "the morning one" no longer counts as 1 o'clock |
| 7 | "The 2019 BMW X5 is white, and here is the full picture." on two turns in a row | the model copied "360° camera" from the car's own facts and the sentence check refused the "°", cutting the reply after its first clause | the degree sign is allowed (the code's own wording already says "360° camera") |
| 8 | the same cut, again, after the "°" fix | the model wrote "8.9 L/100 km"; the check refuses the "/" (correctly: the voice would say "slash") | `clean_draft` turns "N L/100 km" into "N litres per hundred kilometres" and "N km/h" into "N kilometres an hour" before the check |

**Correction to the paragraph above.** I put faults 2 and 3 down to the 4.5 s whole-reply deadline. The replay does not support that: with all fixes, the 4.5 s run and the 10 s run gave the same whole replies, with no cut. Faults 7 and 8, characters copied from the facts and refused, explain the cuts in this replay. Whether the deadline also cut replies in the live call is unknown (the live worker's notes were not kept).

After the fixes, both runs:

* "I want to hear more about the BMW X5." → "The 2019 BMW X5 is white, with 5 seats, 650-litre boot, 340 hp, and 8.9 litres per hundred kilometres. It's accident-free, with full service history and 146 out of 150 checks passed. Would you like to come and see it?" (the invitation is due, so it is asked)
* "Tell me more about it." → "The 2019 BMW X5 has a 3.0-litre I6 turbo engine with 340 horsepower, Panoramic Sky Lounge, heated seats, 360° camera, and gesture control. It's white, with 71,000 km and 2 previous owners." (whole; no question forced. It says "340 horsepower" and "white" again: the not-repeating memory records the groups the code told, not what the model said. Not fixed.)
* "I want to see the car." → "Sure! What day would you like to come in?"; "9 in the morning." (9 AM already past on this clock) → "That's a bit too soon. How about today at 12:00 PM or today at 1:00 PM?". The scripted caller then gives a name, not a time, so nothing is booked, which is correct here. The booking on the first "yes, please book it" is shown with the call's own clock in `test_test_call_28oct.py`.
* Model-written replies 2, fell back 0, guard stops 0, spoken leaks 0; first sentence median 1,532 ms (4.5 s run) and 1,454 ms (10 s run), two replies each, laptop.

Tests: `test_test_call_28oct.py` now 36; each fix fails some when removed. `pytest tests/unit`: 1,024 passed, 40 skipped. The worker that took the call runs the code before these fixes and must be restarted to have them.

## 25. Speed on the laptop: a quick acknowledgement while a slow model writes (29 Oct)

Owner's request after the test call: replies with `model_speaks` on felt slow. Branch `speed/ack-on-long-wait`, from `fix/hearing-booking-latency-27oct`. Machine and model as in section 23 (M3 Pro, Qwen3-4B 4-bit on MLX). Nothing in this section makes the model faster: on this laptop its first sentence takes about 1.9 s, and only a faster machine changes that (sections 21, 23).

**What changed.** Phase 3 dropped the short acknowledgement said while the model writes ("Sure,", "Right,", "So,": rendered in advance, so it plays at once) because said on every reply it sounds scripted. It is back with `model_speaks` on only when the governor's live measure of the model's first sentence is at least `voice.bridge_after_ms` (450 ms), the same line the "Let me check." bridge uses for slow decisions. On a server fast enough for the full level it is never said; on this laptop it is said on most model-written replies, never the same one twice running (the existing opener picker). The model's own first sentence then carries on after it in lower case (the existing `_after_opener`). `agent/engine.py: _slow_writer`, 3 lines in `handle_stream`. Booking lines are spoken by code and get none. Tests: `test_model_speaks.py` (4 added: slow model gets one, fast model none, none when the bridge is off or nothing is measured, not the same twice running); each of two mutations (never, always) fails some. `pytest tests/unit`: 1,028 passed, 40 skipped. The replay script now times the first thing said in each turn.

**Measured with the replay** (the 17 conversations of section 23, flag on, deadlines lifted; times from the caller's words reaching the engine, so without the detector's 0.6 s and recognition):

| | Before | After |
|---|---|---|
| Something said within 1 s | 0 of 46 model-written replies | 44 of 47 |
| First thing said, median / p90 | 1,908 / 2,524 ms | 1 / 1 ms (the acknowledgement is ready in advance) |
| The model's first sentence (the answer), median | 1,888 ms | 1,884 ms |
| Acknowledgements | 0 | 43 |
| Same opening as an earlier reply in the call | 5 | 0 |
| Spoken leaks / fallbacks | 0 / 1 | 0 / 0 |
| Scripted bookings | 3 of 3 | 3 of 3 |

(Bookings are 3 of 3 in both runs, against 2 of 3 in section 23: the third script says "Ten in the morning", which the parser fix of section 24a now reads.)

**What this does and does not do.** The caller hears something at once instead of 1.9 s of silence; the answer itself arrives as late as before, so on this laptop there is now a short gap after "Sure," (the acknowledgement lasts about half a second, the answer follows about 1.4 s after it ends: INFERRED from the two medians, not heard). Most read naturally ("Sure, the boot is 580 litres. Would you like to come and see it?"); some less so ("So, yes, it comes with a 12-month or 20,000-kilometre AGMC warranty."). Whether the gap after it sounds natural is for a listening test. Not changed: the governor, the bridge, the threshold, anything with `model_speaks` off.

## 26. "Any model and any budget is all you need" answered as if the search had been forgotten (29 Oct, the owner's call)

After "There are 4 Toyotas in stock right now. What budget do you have in mind, or is there a model you like?", the owner said "Any model and any budget is all you need." and "Any model or any budget is alright." and was asked "a make, a type of car, or a budget?" twice; "Any model, Toyota." then worked. Branch `fix/any-model-search-memory`.

* **The memory was not the problem.** Replayed through the engine, the search stays `make: toyota` on every turn; what the caller has said, the cars shown and the last six lines are all kept within a call.
* **Cause 1:** `nlu/expected.py: nothing_in_particular` accepted "any model and any budget" with only a few endings ("is fine", "will do", "works", "for me"); "is all you need" and "is alright" were not recognised, so the turn went to the model. **Fix:** common endings are accepted ("is alright / ok / okay / good", "is all I need / you need", "is fine with me", "either way", "please", "thanks", "at all"); a sentence with something searchable in it is still not "no preference" (tested).
* **Cause 2:** on the laptop the model then missed its 1.2 s decision deadline and `_code_decision` asked the generic "make, type of car or budget?" because the words held no new search term, ignoring the search under way (when the model did answer in the replay it chose "nothing to do": "Is there anything else I can help with?"). **Fix:** when the code decides alone and a search is under way (filters or cars shown), it reads out that search's matches; with nothing under way it still asks what the caller is after.
* Replay of the four lines with a model too slow to decide and with one that answers "none": both "any model" lines now read out the Corolla, Camry and RAV4. Tests: `test_any_model_search.py` (17; removing the endings fails 8, removing the fallback rule fails 1). `pytest tests/unit`: 1,045 passed, 40 skipped.

### 26a. The name step of the same day's Corolla call (29 Oct)

The owner's next call booked the Corolla for Friday 30 Oct at 2 PM, but the name step went wrong twice. Replayed through the engine (`test_name_step_29oct.py`, the call's eleven lines, clock at 21:00 on 29 Oct, with a model too slow to decide and with one that answers "none"):

* **"Thank you." after the number** was answered "You're welcome! Anything else you'd like to know?": the thank-you rule ran before anything looked at the booking in progress, so the question for the name was dropped (the booking itself was kept). Now, during a booking step (SLOT, CONTACT, CONFIRM or the read-back), a thank-you gets "You're welcome." and the question the booking is waiting on, as it was ("You're welcome. Who should I put the booking under?"). After the booking it is still just "You're welcome!".
* **"No but the booking under Fatima."** was not read as a name although the agent had asked "Who should I put the booking under?": the name reader knew "my name is", "it's", "call me" and the like, not "under <name>"; and in the contact step a "no" or "but" stops the code from reading a name at all (so "No, not now" is not taken for one). Now "under Fatima", "put it under Fatima", "under the name Omar Hassan" give the name, but not when the words after "under" look like a figure or a budget ("under eighty thousand", "under 100,000", "under budget"); and a name given with such an explicit cue counts in the contact step whatever comes before it.
* Not a logic fault, left as it is: the number was misheard twice ("5-050-7932546" and "050, 0790, 2546", both 11 digits) and the agent rightly asked again; the third try was right. That is the open digits decision of section 18.

After: "Thank you." → "You're welcome. Who should I put the booking under?"; "No but the booking under Fatima." → the read-back with Fatima; "yes" → booked, Friday 30 Oct 14:00, Fatima. Tests: 15 (each of the three changes fails some when removed). `pytest tests/unit`: 1,060 passed, 40 skipped.

## 27. A phrasing sweep: 131 ways of saying things at each step of a call (29 Oct)

Owner's question after sections 24 to 26: what other bugs like these are waiting, and would a better model avoid them? Branch `test/phrasing-sweep`. `tests/data/phrasing_sweep.yaml` holds 131 caller phrasings across nine points of a call (the search, a list read out, a car described, "what day?", "which time?", name and number, the name alone, the read-back, after the booking), each with what must happen: the action, whether the booking talk goes on, the name, time and phone recorded, the car in focus. `scripts/phrasing_sweep.py` replays each on a fresh copy of the website's stock (clock 21:00 Thu 29 Oct) in two modes: **code fallback** (the model never answers in time, as often on this laptop) and **model decides** (the real model, Qwen3-4B on MLX, with its decision deadline lifted, a stand-in for hardware that always answers in time). Wording is the code's in both: this measures understanding, not wording.

| | Code fallback | Model decides |
|---|---|---|
| Passed / failed of 131 | 114 / 17 | 120 / 11 |

**What the model avoids (8):** ten other ways to say yes to the read-back were not a yes for the code ("that's correct", "sounds good", "please do", "absolutely", "sure, go for it", "lock it in", "that works for me", "all correct, thanks", "yes that's all correct"; the code reads the details back again, which is safe but needless), "the middle one" (not resolved), "sure, why not" to an invitation.

**What the model cannot avoid (6), because a rule decides first or the code reads the detail:**

1. **"hmm let me think" while times are on offer cancels the booking draft** ("No problem, I won't book anything."): the "never mind / let me think" rule treats thinking as giving up. The most serious one.
2. **"it's for my brother Ahmed" records the name "For"** ("Thanks, For."): the name reader takes the words after "it's".
3. **"actually can we make it 9 instead" at the read-back** asks "What should I change?" instead of taking the time given.
4. **"this weekend"** at "what day?" offers tomorrow (Friday), not Saturday or Sunday.
5. **"show me everything"** after "4 Toyotas" lists the other makes instead of the Toyotas.
6. **"yes and does it have a warranty?"** at the read-back is not booked (two things at once: the model answers the warranty).

**What the model makes worse (5):** asked "does it have a warranty?", "how many kilometres has it done?", "is the price negotiable?", "is it automatic?" while choosing a time, the model picks "tell me about the car": the answer is right, but `_t_get_car` sets the call back to "answering" and the times on offer are dropped ("...Would you like to come and see it?"). With the code fallback the booking goes on but the question is not answered. Neither is right: the question should be answered and the booking question asked again. Also with the model: "whatever you have" was answered "Is there anything else I can help with?" (the model chose "nothing").

**Reading.** A model that always answers in time would avoid about half of what fails for the code alone, and would bring its own failures (a question in the middle of a booking ends the booking). Half of the faults are in rules that run before the model is asked, or in the code reading details from the words, and need fixing whatever the model. Nothing was changed in this section; the fixes are proposed in batches below. Run again after each: `python scripts/phrasing_sweep.py` and `--model gpu_server`.

**Proposed fix batches, most serious first:**

* **A, never lose a booking by mistake:** thinking ("let me think", "one sec") is not cancelling; a car question during a booking step is answered and the booking question asked again (both modes).
* **B, read what was said:** "for my brother Ahmed" gives Ahmed; "make it 9 instead" at the read-back changes the time; "this weekend" offers Saturday and Sunday.
* **C, a yes is a yes:** the ordinary ways of saying yes to the read-back and to an invitation; "yes, and does it have a warranty?" books and answers.
* **D, the search:** "show me everything" and "whatever you have" read out the search under way; "the middle one".

### 27a. The four batches fixed (29 Oct)

Same branch, in order, the sweep run in both modes after each (same 131 phrasings; the expectations for nine cases were tightened along the way so that they check the answer, not only that the booking went on, and the five cases that once let a question through as a name now check that no name is recorded):

| After | Code fallback | Model decides |
|---|---|---|
| nothing (section 27) | 114 | 120 |
| A, never lose a booking by mistake | 115 | 125 |
| B, read what was said | 118 | 128 |
| C, a yes is a yes | 129 | 129 |
| D, the search | **131** | **131** |

* **A.** A side remark while a booking is being filled in or read back gets a side answer and the step's own question again (`booking_aside`, a code rule that runs before the model: the times already offered, the missing detail, "Shall I book it?"); thinking out loud ("hmm let me think", "one sec") is not cancelling; "either is fine" proposes one of the offered times; "can we do 3 instead" is checked instead of the day being asked again; a question is never read as a name (found here: "do you need my surname?" was read back as the name "Do You Need").
* **B.** "it's for my brother Ahmed" gives Ahmed ("it's for me" gives nothing; it had given "For", and the first fix gave "For Me", caught by its own test); a new day or time at the read-back changes the time and reads back again; "this weekend" is the next Saturday (or today's Sunday).
* **C.** The ordinary ways to agree count as a yes at the read-back and to an invitation, whole utterances only ("that's not correct" and "sounds good but at 5" are not a yes, tested); "yes, and does it have a warranty?" books and answers.
* **D.** "show me everything", "whatever you have" read out the search under way; "the middle one" of three.
* **A guard.** `test_sweep_fixes.py: test_the_whole_sweep_passes_with_the_code_alone` replays all 131 phrasings with the code alone (2 s); with the side answers switched off it fails, so a regression in any batch is caught. Tests: 56 in `test_sweep_fixes.py`. `pytest tests/unit`: 1,116 passed, 40 skipped.

What this does and does not show. 131 of 131 means the phrasings in the file are handled, in both modes; it does not mean every phrasing is: the file is what I could think of, and the next real call will find others. Add each one to the file when it does. The model run used Qwen3-4B on MLX with its decision deadline lifted; with the live 1.2 s deadline on this laptop most turns the rules do not route are decided by the code, which is the first column.

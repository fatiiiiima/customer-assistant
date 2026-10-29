# Prompts for applying docs/ARCH_PLAN.md

One session per phase, started from `main` after the previous phase has merged. Copy the "Common rules" block into the top of every prompt, then the phase prompt under it. Replace the parts in angle brackets.

Recommended order: 0, 1, 2, 3, then 4 and 5. Phase 1 and Phase 5 can run alongside Phase 2 in separate sessions because they touch different files. Do not start Phase 3 before Phase 0 has numbers: on the laptop the governor never reaches the full level, so the new reply path cannot be seen working at its real speed.

---

## Common rules (paste first, every time)

```
You are working in the repository at <absolute path to voice-agent>, a voice agent for a car dealership built on Pipecat, Whisper, Kokoro and a local or served language model. Start a branch from main named <branch name>.

Before changing anything, read in this order: docs/ARCH_PLAN.md (the plan you are applying), docs/ARCH_REVIEW.md sections 1 to 18 (the history of what was measured and fixed; anything marked DONE is done, do not redo or re-propose it; anything marked TRIED and NOT ADOPTED must not be tried the same way again), docs/ARCH_REVIEW_new.md (the second review), README.md, docs/ARCHITECTURE.md, docs/SCALING.md, docs/VOICE.md, then the code the phase names.

Rules:
- Apply only the phase named below. Do not touch anything in the plan's "What not to change" list.
- Never print .env or any secret value. You may read key names.
- Run nothing that costs money or needs a GPU you do not have without asking me first. Unit tests and offline scripts on recordings are fine.
- Keep `pytest tests/unit` green. Every behaviour change gets a test. Conversation replays use the words logs in logs/audio_dump/*_in.txt (newest first) as material.
- Measure before and after with the same script on the same machine, and say which machine. Never invent a number; write "unknown" where there is none. Label each claim MEASURED, INFERRED or GUESS as the plan does.
- When the phase is done, add a new numbered section to docs/ARCH_REVIEW.md in the same style as the existing ones: what changed, file by file, the numbers before and after, and what was deliberately not changed.
- Commit in small steps with messages that say what changed and why. Do not merge; open a pull request and stop.
- If something in the plan turns out to be wrong when you read the code, say so with file and line and propose the correction before building around it. Do not silently widen or narrow the phase.
```

---

## Phase 0: baseline on representative hardware (Sonnet 5.5, or do it by hand)

```
Phase to apply: Phase 0 of docs/ARCH_PLAN.md, the representative baseline. This phase changes no product code.

Produce a runbook, docs/BASELINE_RUNBOOK.md, that I can follow to stand up the production shape from deploy/k8s/voice.yaml on <the GPU machine or cloud I have: describe it>: vLLM serving <Llama 3.1 8B Instruct or Qwen2.5 7B Instruct>, Whisper on the GPU behind the transcription API that the workers reach with --stt-server, and workers with no recogniser of their own. Use the scripts that exist (scripts/capacity_probe.py, scripts/load_test.py, scripts/hearing_check.py, scripts/run_cluster.py) and say exactly how to invoke each and what to record.

The runbook must end with a table I fill in: time to first sentence at 1, 8 and 16 callers per worker (p50 and p90), the governor level each turn ran at, the per-worker knee, and hearing_check results against base on the recorded turns. State the decision gate from the plan: Phases 2 and 3 proceed as written only if the first sentence is under 300 ms at the design load.

If any script lacks an option the runbook needs (for example reporting the governor level per turn), add that option with a test. Nothing else changes.
```

---

## Phase 1: the small open bugs (Sonnet 5.5)

```
Phase to apply: Phase 1 of docs/ARCH_PLAN.md, the open bugs B5, B6, B7, B8 and C5.

For each, read the file and lines named in docs/ARCH_REVIEW_new.md section 3 first and confirm the bug is still there (several fixes landed after that review; the clock B1, the Ollama deadline B2 and the booking fixes B3 and B4 are done and must not be touched).

B5: server/app.py probe_loop probes only when in_flight == 0. Change it so a busy worker still learns the model's speed: probe at in_flight <= 2, and make the threshold a config value with the current behaviour reachable. Keep the governor's choose/record_full/record_probe mechanism exactly as it is. Add a test that a worker with two calls in flight still probes.

B6: database connections. Count the connections one worker opens today (PgStore in shared/state.py, PgDb in repos/pg_store.py and server/app.py). Make it one PgDb per worker process, put both pool sizes in config, add a pgbouncer Deployment and Service to deploy/k8s/voice.yaml with the workers and router pointed at it, and make shared/state.py workers() a single query without the DELETE per refresh (expire stale rows in one statement on a timer, or in the router view). Document the resulting connections per worker and the total at 45 workers in the review section.

B7: voice/audiotap.py writes and flushes per frame on the event loop. Buffer frames and write from a worker thread, keeping the words log (*_in.txt) format byte-identical so existing tooling still reads it. Add a test that the written audio equals the frames given.

B8: in agent/engine.py the voice rules are checked on a reply longer than MAX+2 sentences; check at MAX. Find the constant, change it, add a test with a MAX+1 reply.

C5: the recogniser echoing the agent's question in the SLOT state. Fix 8 (dropping the question from the hint) was tried and rejected, see docs/ARCH_REVIEW.md section 16; do not repeat it. Instead: in voice/services.py echoes_agent, replace the first-three-words rule with a word-overlap rule (heard has 4 or more words and 60% or more of them appear in one sentence the agent just said); and in agent/engine.py _fast_decide's SLOT branch, when the heard words carry no day, time, flexible or refusal signal and echoes_agent is true, ask again instead of routing. Score the new echoes_agent exactly the way section 16 scored fix 8, on every recorded turn where the right words are known, and report hits, misses and false alarms before and after. Adopt it only if it is not worse on any count; if it is worse, keep the old rule and write down why.

Done when all five have tests, pytest is green, and the review section has the numbers.
```

---

## Phase 2: the model's placement and one call per turn (Opus 5.5)

```
Phase to apply: Phase 2 of docs/ARCH_PLAN.md: vLLM as the production backend, one model call per turn instead of decide-then-write, and prefix caching.

Context you must hold: on the laptop the model runs through nlu/ollama_backend.py and the governor (agent/governor.py) never reaches the full level because the start-up probe takes 1,251 to 1,479 ms against a 700 ms budget. nlu/openai_compat.py already talks to vLLM, honours deadlines and supports structured_outputs.choice. agent/engine.py has two model calls per turn: _decide (around lines 917 to 939) picks a tool, then dialogue/writer.py writes the reply. Both paths must keep working until the new one is proven.

Build, in this order:
1. Make the OpenAI-compatible backend the default for deployment in config/portal.yaml, leaving config/default.yaml on Ollama for the laptop. Move model_budget_ms, decide_deadline_ms and first_sentence_deadline_ms behind one documented place in config with a comment that their values come from docs/BASELINE_RUNBOOK.md measurements (use the Phase 0 numbers if I give you them: <numbers or "not yet measured">).
2. A single-call turn: one request that returns a structured object with the action, its arguments and the reply text ("say"). Reuse _fast_decide's cheap routing where it is confident and only call the model when it is not. Define the schema in one place, validate it, and treat a malformed answer as a timeout (fall back to the existing two-call path). Put it behind a config flag, off by default, so both paths run.
3. Prompt layout for prefix caching: the system prompt and the dealership facts first and unchanged within a deployment; the session state, the focused car and the last turns after. Document the layout in docs/PHRASING.md or a new docs/PROMPTS.md.
4. A replay harness (scripts/replay_turns.py or extend an existing script) that runs the words logs in logs/audio_dump through both paths against the same backend and prints, per turn, the action chosen by each, whether they agree, and the time to the structured answer. Run it on the laptop against Ollama so I can see agreement even though the times there are laptop times.

Do not change what the tools do, the booking lines, the voice rules or the governor's mechanism. Do not start Phase 3 work (the model writing the whole reply with code guards) even where it would be convenient; this phase only changes how many calls there are and where they go.

Done when the replay shows the single-call path choosing the same action as the two-call path on at least 95% of recorded turns, with the disagreements listed and explained, and pytest is green. Report the laptop timings as laptop timings.
```

---

## Phase 3: the model speaks, the code guards (Opus 5.5)

```
Phase to apply: Phase 3 of docs/ARCH_PLAN.md. This is the architecture change: at the governor's full level the model writes the reply and code validates it, instead of code assembling the reply and the model rewording the question. The prefix and code levels stay exactly as they are as the degradation path.

Read docs/ARCH_PLAN.md Phase 3 in full before starting; it lists the inputs, the guards, what is removed at the full level, and the tests. Then read agent/engine.py whole (1,759 lines), dialogue/writer.py, agent/answers.py, agent/wording.py, agent/governor.py, dialogue/voice_rules.py, dialogue/session.py and the current prompts, so you know what exists before designing anything.

Work behind a config flag (for example dialogue.reply_path = "written" or "assembled", default "assembled") so the old path is untouched and both can be replayed.

Build, in this order, with a short design note in docs/REPLY_PATH.md before the code:
1. The turn context: session state, a focus entity (the car under discussion and the aspect last discussed, carried across turns), all fact groups of the focused car as a compact list, the last four to six turns verbatim, and the hard rules. Reuse the session store and the not-repeating memory; add fields, do not replace them.
2. The guards module (agent/guards.py): fact fidelity (every number, price, year, colour and model name in the reply must appear in the facts given; fail closed to the assembled path for that turn), the voice rules at the limit, and verbatim insertion of the code-written lines (slot offers from offer_slot_variants, confirmations, the booking summary, speak-up and didn't-catch-that), which the model may not rewrite.
3. At the full level only: no forced closing question outside SLOT, CONTACT and CONFIRM; no opener rotation; no _novel_words pinning of a reworded question. "Tell me more about <aspect>" answers that aspect from the facts, not the next group in order.
4. Tests: a fact-fidelity suite over generated replies and a fixed fact list; a forbidden-promise suite; conversation replays from logs/audio_dump words logs asserting the booking outcome and that no fact is said twice; and a script that renders ten replies per state with Kokoro to WAV files under a directory I name so I can listen to them.

Keep the time to first sentence: the first sentence must stream to the voice as today (dialogue/writer.py _Pieces and the first-sentence deadline). If a guard needs the whole reply before the first sound, say so and propose where the cut goes rather than deciding alone.

On this laptop the full level is not reached, so for development force the level and relax the deadlines in a test config, and say clearly in the review section that the laptop timings are not the product's timings.

Done when: a 20-turn replayed conversation has no two replies with the same opener, "tell me more about X" answers X, the fact-fidelity suite passes 1,000 replayed turns with zero leaks, the booking outcomes of every replayed words log are unchanged, pytest is green, and the WAV files exist for me to hear. Open the pull request with the flag still defaulting to the old path.
```

---

## Phase 4: hearing and overlap (Sonnet 5.5; needs the GPU recogniser from Phase 0)

```
Phase to apply: Phase 4 of docs/ARCH_PLAN.md: recognition that overlaps the caller's speech, the endpoint in tap mode, capture, and digits.

Facts to hold: recognition is a batch job after the endpoint and took 0.9 to 1.1 s per turn in the 28 Oct call on the laptop; voice/speculate.py and voice.early_start_secs exist, are built, and are off because on the laptop they compete with the recogniser for the same cores (measured, docs/ARCH_REVIEW.md); tap_stop_secs 0.6 was kept on purpose after the second review; docs/ARCH_REVIEW.md section 15 found the level was not the difference between microphones, the noise floor and microphone quality were; small.en on CPU cost 0.8 s a turn and did not help at low level.

Build, in this order:
1. With the recogniser on <the GPU STT server from Phase 0: its address>, turn speculation on in config/portal.yaml only, and measure with scripts/load_test.py the recognition time after the endpoint before and after. Leave config/default.yaml (laptop) off.
2. Evaluate one streaming recogniser behind the same transcription API (faster-whisper chunked streaming, or Parakeet), scored with scripts/hearing_check.py on the recorded turns against base. Adopt only if hearing is not worse and the post-endpoint time falls. Write down the result either way.
3. Endpoint in tap mode: measure exactly one alternative, the end of the tap gesture as the endpoint when the caller gives one, falling back to the silence timer. Do not change tap_stop_secs itself. Report turns where the gesture came before the end of speech.
4. Capture: make the portal page's ?mic=raw an A/B on recorded turns with hearing_check.py and write a one-paragraph microphone recommendation for staff in docs/VOICE.md. Do not change min_rms or the speak-up thresholds.
5. Digits: docs/ARCH_REVIEW.md section 18 leaves real-voice digit recognition open. Prototype one option on the recordings (digits asked in groups of three, each group recognised alone) and report its accuracy against the current way. Build it into the agent only if I say so after seeing the numbers.

Done when the recognition stage is under 0.25 s after the endpoint at p50 on the GPU set-up, hearing_check is not worse than base, pytest is green, and every measurement is in the review section with the machine named.
```

---

## Phase 5: scale policy (Sonnet 5.5)

```
Phase to apply: Phase 5 of docs/ARCH_PLAN.md, the scale policy. The approach of spreading calls across workers at a utilisation target and adding workers on demand stays; this phase sets its numbers from measurement and removes the shared bottlenecks that Phase 1 did not.

Facts to hold: docs/SCALING.md measured the per-process knee on the laptop (flat to 16 callers, 27% of turns over 2 s at 24); deploy/k8s/voice.yaml scales on voice_demand_ratio at 0.7 with minReplicas 8 and maxReplicas 72; the admission controller's p90 uses the corrected turn clock since 28 Oct; the router uses power-of-two placement.

Build:
1. A documented way to derive the autoscaler target from the measured knee: a short section in docs/SCALING.md with the formula (worst-case placement plus scale-up lag under the knee) and the value it gives from <the Phase 0 numbers or "not yet measured">. Change the 0.7 in deploy/k8s/voice.yaml only if the formula gives a different value, and say why.
2. minReplicas as a warm floor: the quietest hour's load plus one spike's worth, with the arithmetic in the comment.
3. Confirm Phase 1's B5 and B6 are merged (one prober per model server or in_flight <= 2 probing; pgbouncer; cached registry view). If not, stop and tell me rather than building on top.
4. Run scripts/load_test.py at the largest simulated load this machine honestly supports and record median, p90, turns over 2 s, callers turned away and the governor level per turn, with the corrected clock. State the machine. Do not claim a 500- or 1000-caller result from a laptop; write what a real run needs.
5. Alerts in deploy/alerts.yml on p90 of the corrected answer-time metric, with the threshold from the plan (1.5 s).

Done when the documentation and manifests are updated, the load run is recorded in the review section, and pytest is green.
```

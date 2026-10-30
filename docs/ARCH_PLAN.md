# Architecture plan: natural replies under 1 s, 500 to 3000 callers

Written 28 Oct 2026 from the second review (`docs/ARCH_REVIEW_new.md`) and the discussion after it. It is a plan for the next model to apply, phase by phase, in its own sessions.

## How to use this plan

* Read `docs/ARCH_REVIEW.md` sections 1 to 18 first. Anything marked DONE there is done: do not propose or redo it. Since the second review these are done: the turn clock (B1), the Ollama deadline (B2), the booking loop and "any day would work" (B3, B4), slot-offer openings rendered in advance (fix 11), the digits-hint retry, and the "nothing to move" fix. Fix 8 (leaving the agent's question out of the recogniser hint) was tried, measured and NOT adopted; do not try it the same way again.
* One phase per branch and pull request, in the order below. Each phase ends with the measurement named under "Done when", recorded in `docs/ARCH_REVIEW.md` as a new numbered section, with the numbers and the machine they were taken on.
* Measure before and after on the same machine with the same script. A number taken on the laptop is a laptop number, never a production number.
* Never print `.env` or any secret. Run nothing that costs money without asking.
* Keep `pytest tests/unit` green. Every behaviour change gets a test; the words logs in `logs/audio_dump/*_in.txt` are the replay material for conversation tests.
* Labels: MEASURED means a number in the docs or logs supports it; INFERRED means it follows from the code; GUESS means it must be measured before it is believed.

## 30 Oct: the conversational agent (the owner's vision; supersedes Phase 0, extends Phase 3, tightens the goal)

The owner, 30 Oct: "It should be flexible, and be able to answer anything the user asks. If it doesn't know, it should clearly state it doesn't know. Basically function like a fully conversational voice bot, but built for the used cars website." The model now runs on Amazon Bedrock (an OpenAI-compatible endpoint, "Mantle"), so the local-GPU limit that made the code-written replies the product (see "Why it sounds scripted") goes away.

**The owner's decisions (30 Oct)**

| Question | Decision |
|---|---|
| Off-topic questions (football, the weather) | Decline politely and steer back to cars |
| A dealership fact that is not in our data | Say plainly that it is not known. No callback offer |
| Arabic | Not yet: English only |
| Speed | **Never more than 1 s** from the end of the caller's speech to the first sound. Up to 1 s is fine |
| Prices | Fixed. No negotiation |

**What the agent may say, by kind of question**

1. *About this dealership* (stock, a car's price, kilometres, colour, history, warranty, finance terms, opening hours, test-drive slots): only what a tool returned this turn. A field we do not hold is given to the model explicitly as "not recorded" (for example accident history), so the honest answer is in front of it: "I don't have that information."
2. *General car knowledge* ("Is a hybrid worth it?", "Camry or Accord for a family?"): from the model's own knowledge, said as general advice, never as a fact about one of our cars.
3. *Anything else*: one polite sentence declining, then back to cars ("I can only help with cars here. Are you looking for anything in particular?").

**Who does what**

* The model holds the conversation. It writes every reply and chooses the tools (function calling), in one call per turn.
* The code keeps the facts and the actions. Today's `engine.py` tools become the model's tools: `search_cars`, `get_car`, `best_of`, `compare_cars`, `colours`, `browse_makes`, `check_slots`, `prepare_booking`, `confirm_booking`, `fix_booking`, `cancel_draft`, `my_bookings`, `cancel_booking`, `reschedule_booking`, `showroom_info`, `end_call`.
* Exact lines stay code-written and are said verbatim: the slot offer, the booking read-back and confirmation, "speak up", the silence prompts (section 49). No booking, cancellation or change without a confirmed yes, as today.
* Guards after the model, before the voice (Phase 3, unchanged): every number, price, year, colour and model name must be in this turn's tool results; no discount, no "I can do", no promise (`voice_rules`); a reply that fails is not said, and the code path answers that turn.

**The 1 s ceiling, honestly**

MEASURED on a test worker with eve, when the first sentence is ready the instant the turn ends (section 42, code level): end of speech to first sound 0.73 to 0.79 s. So turn-end detection, recognition and the voice already use about 0.75 s. That leaves about 0.25 s for a network model to return its first clause. GUESS: no model reached over the network does that on every turn, and many will not on the median. Therefore:

* **A watchdog makes the ceiling a guarantee.** If the model's first clause is not ready 1.0 s after the caller stopped speaking (minus the voice's own ~0.1 s), the code says a short ready-made bridge at once ("Let me check that.", "Good question."), and the model's answer follows it. That guarantees a first *sound* within 1 s on every turn. It cannot guarantee the *answer* within 1 s; no design can, with a model over a network.
* **Earn back the 0.75 s, so the bridge is rare.** Start the model on the transcript while Smart Turn is still deciding (`voice/speculate.py`, built, off on the laptop); recognition streamed during speech (Phase 4); the model told to open with a short clause; prompt caching for the fixed part of the prompt (INFERRED available on Bedrock for some models); the Bedrock region nearest the callers; one kept-open connection per worker.
* **The bridge rate is a number we report**, per model: the share of turns where the model's own first clause made it inside 1 s. A bridge on every turn sounds robotic, so the choice of model depends on it.
* **When Bedrock is slow or down** (no first clause in 3 s, or an error): today's code path answers that turn. That is all that remains of the templates.

**What is retired, once the new path passes its gate**

The regular-expression intent routing in front of the model (`_fast_decide`, `_FLEXIBLE`, `_DELEGATES`), the separate decision call, the opener rotation and two-group templates, the model's rewording of a code-written question (`_check_continuation`, `_novel_words`), and the governor's prefix level as the normal way of speaking. They stay as the fallback above, behind the same switch.

**What stays as it is**: hearing (Smart Turn, the background and first-turn gates, speak-up), barge-in and "what the caller heard", the caller-first start and silence clock, the call store and carry-on, eve and its fallbacks, the booking store, the site messages.

**The cars come from the database, not from `data.js`** (the owner, 30 Oct: "when we go live, the data should be in a database. What if we have thousands of cars?")

Today (INFERRED from the code):
* The 24 demo cars and the showroom details are written in the website's `site/data.js`.
* At start-up a worker copies the cars into an inventory (in-memory SQLite, or PostgreSQL when `DATABASE_URL` is set: `repos/pg_store.py`). The full spec sheets and showroom details are read into memory only (`site/details.py`, `svc.site.cars`).
* The website reads `data.js` directly.
* Several fields are not records at all: data.js works them out by rule for every car. "Accident-free" and "GCC specs" are set on every car; the service history comes from the owner count; the warranty from age and kilometres; the inspection score is computed.

Target:
* **One PostgreSQL database is the only source.** Cars, with every spec as a column (seats, power, body, fuel, transmission, drivetrain, consumption, boot, owners, features, blurb, colour, photos), the status (available, reserved, sold), branches (address, hours, contacts), bookings and reservations (already there), and calls (already there).
* **A fact we do not hold is NULL and is said as "not recorded".** That covers accident history, service history and warranty, until a real source fills them; nothing is derived by rule.
* **`floor_price`** (a negotiation floor) is unused under the fixed-price decision.
* **The website reads an API** (`/api/cars` with filters, `/api/cars/{id}`, `/api/showroom`) instead of `data.js`. `data.js` becomes the demo seed, imported by a script.
* **The voice tools read the database on every search:**
  * the spec sheet comes from the database, not from memory;
  * seats and "sporty" become SQL filters (today up to 200 cars are fetched and filtered in Python);
  * nothing loads the whole stock;
  * the status is read live, and locked again when a car is booked or reserved.
* **Names as callers say them** ("LC300", "the Land Cruiser"): indexes on make, model, price, body, status and branch, plus a fuzzy name match (PostgreSQL's trigram index; INFERRED suitable).
* **Whisper's hint list of makes and models** is built from the database, limited to what fits: the makes in stock first, then the models already named in the call.
* **Who keeps it current** is the owner's question to AGMC: where does the real stock live today (a dealer management system, a spreadsheet)? The importer depends on the answer (UNKNOWN).

Done when:
* the website and the voice server run with no read of `data.js`;
* the demo cars imported into the database give the same replay outcomes as today;
* a load test with 5,000 generated cars records search, count and spec-sheet times, median and p99. A target for p99 is under 20 ms, so the data never shows in the 1 s budget (GUESS; measure);
* the speech hint list stays within Whisper's limit.

**Steps** (one branch and pull request each, no step starts before the one before it has its numbers; step 0 and step 1 are independent and may run side by side)

0. *The database as the only source of cars* (above). It goes first because every tool the model will call reads it, so the tools are built once, on the real data layer. It does not wait for Bedrock: a database search is milliseconds against a model's hundreds (INFERRED), so it does not change which model wins.
1. *Bedrock baseline* (needs a working key; costs money, asked first). List the models on Mantle. For each candidate, from this laptop and with the real prompt and tools: time to first token and to the first clause, median, p90 and worst over at least 100 requests; tool-call accuracy on the replay set; cost per turn. Pick at most three models to carry on.
2. *The scoring set* (no model calls needed to build it): the owner's recorded calls (`logs/audio_dump/*_in.txt`) plus written cases for each kind of question above: "Do you have a Tesla?", "Can you do 60,000?", "Is it accident-free?", "What's your warranty?", "Who won the match?", "Is a hybrid worth it?", a booking, a reschedule, an interruption.
3. *The new path behind a switch* (`dialogue.mode: conversational`): one call per turn with tools, streamed sentence by sentence to the voice, the guards, the watchdog bridge, the fallback.
4. *Replay and compare* both paths on the scoring set and on test calls.

**Done when**, on the chosen model:
* the first sound arrives within 1.0 s on 100% of turns (the watchdog), and the model's own first clause within 1.0 s on a share the owner accepts;
* zero facts outside the tool results over the whole scoring set;
* every unknown dealership fact gets "I don't have that information";
* every off-topic question is declined and steered back;
* no negotiation;
* bookings end the same as today on the replays.

**Changes to the phases below**: Phase 0 (a vLLM GPU server) is replaced by step 1. Phase 2's single call is step 3. Phase 3 stands, with the three kinds of question above added to the model's rules and to its tests. The goal's 1 s median / 1.5 s p90 becomes 1 s on every turn, by the watchdog.

## Goal and the time budget

End of the caller's speech to the first sound of the reply, under 1 s on the median and under 1.5 s at p90, while sounding like a person. What the budget has to look like, against the 28 Oct call on the laptop (tap mode, small.en on CPU, Ollama):

| Stage | 28 Oct call, laptop | Target | How |
|---|---|---|---|
| Silence before the endpoint fires | 0.6 s | 0.2 to 0.3 s | Phase 4 |
| Speech recognition after the endpoint | 0.9 to 1.1 s | 0.1 to 0.2 s | Phase 4 (streaming during speech) |
| Model, first sentence | 0.5 s decide then template, or 1.3 s written | 0.2 to 0.3 s | Phases 2 and 3 (GPU, one call) |
| Speech synthesis, first sound | 0 to 0.35 s | 0.1 to 0.2 s | already there (phrase cache, slot openings) |

The stages must overlap: recognition runs while the caller speaks, the model's prefill starts on the partial transcript, the first sentence goes to the voice while the rest is written.

## Why it sounds scripted (the diagnosis the plan rests on)

1. On the laptop the governor never leaves the prefix level (the probe is 1,251 to 1,479 ms against a 700 ms budget, MEASURED), so every reply is a rotated opener, two fact groups in a fixed order, and a question the model may only reword. The code speaks; the model proofreads.
2. "Tell me more" serves the next fact group, not the aspect the caller asked about (`agent/answers.py: next_facts`, `engine.py: _more_about`). Follow-ups cannot land.
3. Every reply ends in a question by construction (`_prefix_reply`, `writer.py: stream` appends the system question).
4. Intent is decided by regular expressions before the model sees the words (`_fast_decide`, `_FLEXIBLE`, `_DELEGATES`). Fast, but blind to context.
5. The decision and the reply are two sequential model calls.

None of these is a limit of the model, the recogniser or the voice. They are consequences of tuning the product on a CPU laptop where the model cannot meet the budget, so the degradation path became the product.

## Phase 0: a representative baseline (no code changes)

Stand up the production shape described in `deploy/k8s/voice.yaml` on real hardware or a cloud GPU: vLLM with a 7 to 8B instruct model, Whisper on a GPU behind the transcription API (`--stt-server`), workers with no recogniser of their own. Candidate models: Llama 3.1 8B Instruct or Qwen2.5 7B Instruct; the choice is a measurement, not an opinion.

Done when `scripts/capacity_probe.py`, `scripts/load_test.py` and `scripts/hearing_check.py` have been run there and the numbers are in `docs/ARCH_REVIEW.md`: time to first sentence at 1, 8, 16 callers per worker, the governor level each turn actually ran at, and the per-worker knee. Decision gate: if the model's first sentence is under 300 ms at the design load, Phases 2 and 3 go ahead as written. If not, the model or the GPU is wrong and that is fixed first.

Size: small (operations). Risk: none to the code.

## Phase 1: the small open bugs from the second review

All INFERRED unless marked. Each is a one-file change with a test.

* **B5, probe starvation.** `server/app.py: probe_loop` only probes when `in_flight == 0`, so a busy worker never learns the model got faster and stays at the prefix level; and 45 workers each probing a shared server wastes prefill. Probe at `in_flight <= 2`, or run one prober per model server and publish the result through the kv store. Effect at 500 callers: workers recover the full level under load; removes about 2.5k tokens/s of probe prefill (INFERRED from 45 workers x probe size / 20 s).
* **B6, database connections.** Each worker opens PgStore (pool 4) and PgDb (pool 8): about 540 connections at 45 workers against PostgreSQL's default 100. One PgDb per worker process, pool sizes in config, pgbouncer in `deploy/k8s`, and `shared/state.py: workers()` read without the N+1 queries and without the DELETE per refresh (cache the registry view for a few seconds in the router).
* **B7, AudioTap.** `voice/audiotap.py` writes and flushes per frame on the event loop. Buffer and write from a thread, or only when `VA_DUMP_AUDIO_DIR` is set in development. Effect: removes a per-frame syscall from the loop that limits callers per process (the 32-caller collapse in `docs/SCALING.md` is the event loop falling behind, MEASURED).
* **B8, voice_rules at MAX+2.** `engine.py` around line 562 checks the forbidden-promise rules on a reply longer than the limit. Check at MAX.
* **C5, the hint echo, a different way.** Fix 8 (dropping the question from the hint) lost more than it gained, MEASURED. Instead strengthen `voice/services.py: echoes_agent`: a word-overlap test without the first-three-words rule (overlap of 0.6 or more with any sentence the agent just said, heard length 4 or more), plus a plausibility exit in `_fast_decide`'s SLOT branch: when the heard words contain no day, time, flexible or refusal signal and overlap the agent's own question, ask again instead of routing. Score it the way fix 8 was scored, on every recorded turn, before adopting.

Done when the tests pass and the 500-caller simulation (`docs/SCALING.md` method) shows no worker stuck at the prefix level while the model server is idle.

Size: small. Risk: low.

## Phase 2: where the model runs and how many times it is called

* **vLLM is the production backend; Ollama is the laptop.** `nlu/openai_compat.py` already honours deadlines and `structured_outputs.choice`. Make it the default in `config/portal.yaml` for deployment; keep Ollama for `config/default.yaml`. Re-derive `model_budget_ms`, `decide_deadline_ms`, `first_sentence_deadline_ms` from Phase 0's numbers rather than the laptop's.
* **One model call per turn.** Today `_decide` chooses the tool and a second call writes. Replace with a single call that returns the action and the reply together (structured output: `{"action": ..., "args": ..., "say": ...}`), with the facts already retrieved by `_fast_decide`'s cheap routing where it is confident. Effect: removes one time-to-first-token per turn, 0.1 to 0.5 s on the laptop numbers (MEASURED range of `_decide` in the 28 Oct log), and halves the request count on the shared server at 500 callers.
* **Prefix caching.** The system prompt and the dealership facts go first in the prompt and never change within a deployment, so vLLM's prefix cache serves them; the per-turn part is the session state, the focused car's facts and the last turns. Effect at 1000+ callers: prefill cost per turn falls to the delta (INFERRED; measure the cache hit rate on the server).
* **Governor thresholds from measurement.** `agent/governor.py: choose` keeps its mechanism; its budgets come from Phase 0.

Done when, on the Phase 0 hardware, the full level runs on more than 90% of turns at 16 callers per worker with the first sentence under 300 ms at p50 and under 600 ms at p90.

Size: medium (`engine.py: _decide`, `dialogue/writer.py`, config, one new structured schema). Risk: medium; the two-call path stays as the fallback until the single call is proven on replayed conversations.

## Phase 3: the model speaks, the code guards (the major change)

This is the change that removes the scripted feel. It replaces the reply path at the full level; the prefix and code levels stay as the degradation path.

**Inputs to the model, every turn**

* Session state: stage of the booking, what is known (name, number, chosen day and time), what has been offered, what has already been said (the not-repeating memory already in the session).
* A focus entity: the car under discussion, carried across turns so "and the other one", "what about the colour" and "tell me more about the engine" resolve. Today the state knows the car but not which aspect was last discussed.
* The facts of the focused car, all groups, as a compact list. Not the next group in a fixed order.
* The last four to six turns verbatim.
* Hard rules (already in `SYSTEM_VOICE`): no prices or facts not in the list, no promises, two short sentences at most, a question only when the state needs an answer.

**Outputs**: the reply text plus the action, from the Phase 2 single call.

**Guards, in code, after the model**

* Fact fidelity: every number, price, year, colour and model name in the reply must appear in the facts given. Fail closed: fall back to the prefix path for that turn.
* Forbidden promises: `dialogue/voice_rules.py` as today, checked at the limit (B8).
* Exact lines stay code-written and are inserted verbatim: slot offers (`agent/wording.py: offer_slot_variants`), confirmations, the booking summary, the speak-up and "didn't catch that" lines. The model writes around them, never rewrites them.
* No booking, cancellation or change without a confirmed yes, as today.

**Remove at the full level**

* The forced closing question. The code adds the question only in SLOT, CONTACT and CONFIRM, where an answer is needed. Elsewhere the model may end on a statement.
* The opener rotation and the two-group template. They remain the prefix path.
* The model's rewording of a code-written question (`_novel_words` pinning): at the full level the model writes the question or the code inserts the exact one; there is no middle.

**Tests**

* A fact-fidelity suite: replies generated from a fixed fact list must not contain numbers outside it.
* A forbidden-promise suite over the voice rules.
* Conversation replays from the words logs in `logs/audio_dump`, asserting the booking outcome and that no reply repeats a fact already said.
* Listening tests: ten replies per state rendered by Kokoro and heard by a person, because "sounds natural" is not a unit test.

Done when a 20-turn conversation on the Phase 0 hardware has no two replies with the same opener, "tell me more about X" answers X, and the fact-fidelity suite passes 1,000 replayed turns with zero leaks. Latency must not regress against Phase 2.

Size: large. `agent/engine.py` (1,759 lines), `dialogue/writer.py`, `agent/answers.py`, `agent/governor.py` and the prompts all change; expect a new module for the guards and a reduced engine. Risk: high if done as a rewrite, medium if staged: build the new path behind a config flag, replay conversations against both, switch when the replay is clean.

## Phase 4: hearing and overlap

* **Streaming recognition during speech.** With the recogniser on a GPU server, turn speculation on (`voice/speculate.py`, `early_start_secs`, built and off because on the laptop it competes for the same cores, MEASURED). Then evaluate a streaming recogniser (faster-whisper with chunked streaming, or a streaming model such as Parakeet) behind the same transcription API, scored with `hearing_check.py` on the recorded turns before and after. Effect: 0.7 to 1.0 s per turn, MEASURED as the recognition time in the 28 Oct log.
* **The endpoint in tap mode.** The 0.6 s `tap_stop_secs` was kept on purpose after the second review. Measure one alternative only: in tap mode treat the end of the tap gesture, when the caller gives one, as the endpoint, falling back to the silence timer. GUESS on the gain, MEASURED cost of the current setting (0.3 s per turn against 0.35 s).
* **Capture.** Section 15 of the review found the level was not the difference between the two microphones; the noise floor and microphone quality were. So: keep `min_rms` and the speak-up thresholds; make the portal page's `?mic=raw` an A/B on recorded turns with `hearing_check.py`, and document the recommended microphone for staff. Do not buy hearing with a bigger Whisper on CPU: small.en cost 0.8 s per turn and did not help at low level (MEASURED). On a GPU server the model size is free; choose it by `hearing_check.py` there.
* **Digits.** Section 18 leaves real-voice digit recognition as an open decision. Options to measure: ask for digits in groups of three and recognise each group separately; or accept the number by the portal's keypad. Not a latency item.

Done when the recognition stage is under 0.25 s after the endpoint at p50 on the Phase 0 hardware and `hearing_check.py` is not worse than base on the recorded turns.

Size: medium (a new STT service class, pipeline wiring, portal change). Risk: medium.

## Phase 5: scale policy

The current approach, spreading calls across workers at a utilisation target and adding workers on demand, is the right one for long-lived calls with a latency knee (MEASURED in `docs/SCALING.md`: flat to 16 callers per process, 27% of turns over 2 s at 24). Fill-then-add would park new callers on workers at their knee, with no slack while new pods start. Changes:

* **Target from the knee, not a round number.** Run `capacity_probe.py` on the real pods; set the autoscaler target (`deploy/k8s/voice.yaml`, now 0.7) so that worst-case placement plus scale-up lag stays under the knee. The admission controller's p90 now uses the corrected clock, so the caps it sets are real.
* **A warm floor.** `minReplicas` covers the quietest hour plus one spike, because only a warm pod answers in under a second.
* **Thin workers, fat shared servers.** Workers do audio and state; model and recognition are batched on GPUs. One prober per model server (B5), pgbouncer and a cached registry view (B6).
* **Load tests with the corrected clock** at 500, 1000 and 2000 simulated callers, recording median, p90, turns over 2 s, callers turned away, and the level the governor ran at. Alerts in `deploy/alerts.yml` on p90 with the corrected metric.

Done when the 1000-caller run holds the Phase 2 latency and no caller is turned away while the autoscaler is below `maxReplicas`.

Size: small to medium (configuration, two code paths from Phase 1). Risk: low.

## What not to change

* Smart Turn endpointing, the tap-grace join, the 0.9 s silence after a phone question.
* Kokoro, the phrase cache, the slot-offer openings, the first-piece split.
* The governor, the admission controller, load shedding and the power-of-two router as mechanisms. Their thresholds change; the mechanisms do not.
* Exact booking, confirmation and summary lines written by code; `model_only` false; the sentence checks and the voice rules.
* `min_rms` 0.005, the speak-up thresholds, f32 precision, the echo guard's purpose (its test changes in Phase 1).
* The session store and the not-repeating memory; Phase 3 adds to them, it does not replace them.

## Size at a glance

| Phase | Size | Risk | Main files |
|---|---|---|---|
| 0 Baseline | small, operations | none | deploy, scripts |
| 1 Open bugs | small | low | server/app.py, shared/state.py, repos/pg_store.py, voice/audiotap.py, voice/services.py, engine.py |
| 2 Model placement, one call | medium | medium | engine.py `_decide`, dialogue/writer.py, nlu/openai_compat.py, config |
| 3 Model speaks, code guards | large | high unless staged | engine.py, writer.py, answers.py, governor.py, prompts, new guards module, tests |
| 4 Hearing and overlap | medium | medium | voice/stt.py, services.py, speculate.py, pipeline.py, portal voice.js |
| 5 Scale policy | small to medium | low | deploy/k8s, gateway, admission |

Phases 1, 2, 4 and 5 are incremental and each stands alone. Phase 3 is the architecture change: it inverts who writes the reply. Done behind a flag with replays, it is a staged migration rather than a rewrite.

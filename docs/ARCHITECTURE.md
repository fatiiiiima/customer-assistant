# Architecture (as built, text chat + demo)

Fully local. No cloud calls in the production path. Fake data only. English only. The cloud LLM mode exists only as a disabled stub
(`llm/cloud_stub.py`).

## 1. The core idea
The language model **never writes a reply and never decides a price**. It picks one *intent* from a short list (about 6 output tokens).
Everything else is ordinary code:

```
 customer text
      |
      v
 [1 NLU pipeline]   language check -> rules (only for intents whose precision gate is open) -> LLM intent call (grammar-constrained)
      |                                                                                   -> consistency repair
      v   Understanding(intent, via=rules|llm|language, ms)
 [2 Dialogue engine]  state machine + slot extractors (code, not LLM) + pending yes/no questions
      |
      +--> [3 Tools / DealerService] search, car info, price, slots, booking, WhatsApp(mock), escalation(mock)
      |          |--> [4 Policy engine]  negotiation ladder, floor price (never spoken)
      |          '--> [5 Repositories]   SQLite (in memory) behind interfaces, FAKE data
      v
 [6 Templates + voice rules]  the only place reply text is produced; max sentences / length checked
      |
      v
 reply (+ TurnDebug: state, intent, rules-or-LLM, tools, policy decision, ms)
```

## 2. Components (all under `src/dealer_core/`)
| Layer | Files | Job |
|---|---|---|
| Wiring | `app.py`, `config.py` + `config/default.yaml` | Builds config -> repos -> service -> NLU -> engine. `demo_clock` freezes "now" to next Monday 10:00 Dubai. |
| NLU pipeline | `nlu/pipeline.py` | Order: Arabic check (-> out_of_scope, offer a human) -> `RuleEngine` -> LLM classifier -> `consistency.repair`. |
| Rules | `nlu/rules.py`, `config/nlu_gates.yaml` | Regex rules per intent. A rule only fires if its **gate is open**. The loader refuses to open a state-changing gate without measured evidence (Wilson lower bound >= 98%). Today open: greet, search_cars, ask_car_info, ask_policy, send_details, goodbye. Price/booking/cancel/handoff stay closed, so they go to the LLM. |
| LLM backends | `nlu/backend.py`, `openvino_backend.py`, `ollama_backend.py`, `chain.py`, `prompts.py` | Interface `classify(state, car, text) -> IntentResult`. **OpenVINO GenAI** on iGPU (demo default), CPU OpenVINO and **Ollama** (Q4_K_M) as fallbacks. `BackendChain`: warms the active backend at start, timeouts (gpu 6 s, cpu 12 s, ollama 20 s) are non-sticky, hard failures demote to the next backend, all failing -> human handoff (`system_fault`). |
| Constrained output | `dialogue/states.py` (`STATE_SPEC`) | For each state, only the intents valid there are allowed; the JSON-schema grammar is restricted to them (19 intents total). Per-state prompt, KV-cache prefix reuse. |
| Slot extraction | `nlu/extractors.py`, `datetime_parse.py`, `numbers.py`, `grounding.py` | Filters (make/model/body/budget), car references, offer amounts, contact, dates/times by code. `grounding` drops any slot not present in the utterance. |
| Consistency | `nlu/consistency.py` | Deterministic repairs that follow from intent definitions (e.g. `ask_lowest_price` with a named amount -> `make_offer`). |
| Dialogue engine | `dialogue/machine.py`, `session.py`, `templates.py`, `voice_rules.py` | States: greeting, interest, answering, negotiating, slot, contact, confirm, goodbye. Handlers per intent, pending yes/no, read-back before booking, availability re-check, sold-mid-call recovery, reschedule/cancel, handoff reasons (angry, requested_human, out_of_scope, below_floor, price_approval, misunderstood, unsupported_language, system_fault). |
| Tools | `tools/service.py` (+ `slots.py`, `phones.py`, `messaging.py`, `escalation.py`, `results.py`) | `DealerService`: search, car details, `propose_price`, free slots (per-branch hours), `book`, cancel, `similar_to`, WhatsApp (mock), escalation (mock). UAE E.164 phone validation. |
| Policy | `policy/negotiation.py` | Deterministic. Ladder of 3 rungs; counters stepped; never below floor; "lowest price?" returns the next rung; implausible amounts clarified; authority claims ignored; tiers are PLACEHOLDER (2% auto-accept). Below the floor -> escalate. Reply always "subject to confirmation". |
| Repositories | `repos/base.py` (interfaces), `sqlite_store.py`, `demo_data.py` (16 cars), `seed.py` (60 random) | SQLite `:memory:`; DB-level partial unique indexes prevent double booking; booking is re-checked inside the transaction. |
| CLI | `cli/chat.py`, `cli/replay.py`, `scenarios/demo.yaml` | Interactive chat and scripted replayer with per-turn expectations; `--debug` trace; `--runs N --quiet` reliability mode. |
| Eval / labeling | `eval/`, `labeling/` | Wilson bounds, paired bootstrap + McNemar for backend comparison; the local labeling page for your held-out transcripts. |
| Cloud | `llm/cloud_stub.py` | Disabled stub (off by default). |

## 3. One turn, concretely ("Can you do 120,000?" in negotiation)
1. Pipeline: no open rule matches (price gate closed) -> OpenVINO classifies in state `negotiating` -> returns `ask_lowest_price`.
2. Consistency repair sees a named amount -> `make_offer` (shown in the trace as "repaired").
3. Engine extracts the amount with code, calls `svc.propose_price(stock, 120000)`.
4. Policy engine returns `counter 147,100` (round 1 of 3); the floor never leaves the service layer.
5. Template renders "...I can do 147,100, subject to confirmation." Voice-rule check runs; `TurnDebug` is filled.
Booking is similar: slots by code, a full read-back (time, name, phone), availability re-check, then `book` inside a DB transaction.

## 4. How it was built (order and method)
1. **Phase 0 / 0b - measurement before code.** Benchmarked STT/LLM/TTS candidates on this CPU/iGPU/NPU; found OpenVINO prefill with KV-cache reuse is the latency winner (cache miss on iGPU ~90-160 ms vs CPU 1.2-4.7 s); corrected two earlier wrong claims (cache-hit prefill, a sherpa/onnxruntime clash).
2. **Phase 1 plan** (approved with conditions): intent-only LLM, deterministic everything else, gates closed until measured, config-switchable fallbacks, privacy-safe logs.
3. **T0 backend comparison** on Qwen3-4B-Instruct-2507 (own int4 conversion "nocal", AWQ build, Ollama Q4_K_M) with paired statistics; result inconclusive -> OpenVINO for voice path, Ollama kept as fallback.
4. **T1-T3**: config, models, repositories, SQLite mock inventory, tools, policy engine, labeling helper.
5. **T4-T6 (this priority push)**: dialogue engine, NLU pipeline, chat CLI, replayer. Developed against a FakeClassifier in tests (scenarios in `tests/scenarios`), then run live. Live failures drove deterministic fixes only: warm-up + non-sticky timeouts (cold-start cascade), consistency repair (offer misread), capitalisation, contact parsing. Nothing was tuned on your held-out set (`benchmarks/spontaneous`).
6. **Verification:** 254 tests; 5 demo scenarios x 20 live runs on `openvino_gpu` = 100/100 passed; LLM turn p50 731 ms, p95 1,339 ms.

## 5. Not built yet
Latency JSONL logging (T7), full 11-scenario suite, out-of-scope veto, opening-hours and "correction" intents, policy/FAQ RAG (Phase 2),
voice with Pipecat + STT/TTS (Phase 3), gate measurement on your labelled set (opens price/booking rules to skip the LLM).

## 6. Licences / open decisions
Everything is MIT/Apache/BSD except: espeak-ng (GPL-3, internal prototype only, open decision in `docs/LICENSES.md`). Piper voice `joe` is CC0.

## 7. Conversation focus (added after the "it does not understand what I meant" bugs)
`dialogue/focus.py` is the single place that decides what a caller's words mean given what is being searched. It is a pure function
`decide(Ctx, Cues, fits) -> Decision` with named operations: **fresh** (a new request), **refine** (the old request plus this),
**switch** (does not fit the old request: new subject), **others / ask_instead** ("another car"), **list_all** ("any budget", "just show me"),
**cannot_filter**, **ask_which** (pointing at a car not on screen), **replay** (follow-up to "shall I look for a red one?"), **ask_what**.
`about_the_car_in_focus()` separates a question about THE car on screen ("is it red") from a search ("do you have it in red"); the rule
engine and the dialogue engine both use it.
* The decision is stored (`Session.last_focus`) and printed in the debug trace as `reading :`; `Session.focus` is a one-line description of what
  the call is about (`2 cars just read out; looking for: color=red`) and prints as `focus :`.
* `tests/unit/test_focus.py` is a table: (current request, what the caller says) -> (operation, request afterwards), one row per past
  misunderstanding. A new misunderstanding = a new row plus, if needed, a change in `decide()`, not another special case in the engine.
* Not covered yet: pronoun resolution beyond "it/this/that/one", and corrections ("I said red, not white").

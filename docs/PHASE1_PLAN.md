# Phase 1 plan: text-only core (FOR APPROVAL: no code has been written)

Scope per the brief: state machine, policy engine, mock inventory + booking tools, CLI chat, pytest, a scripted-conversation
replayer, and the NLU layer that Phase 3 will plug STT/TTS into. Voice, Pipecat and RAG stay in Phases 2-3.
Inputs: `docs/PHASE0B_REPORT.md` (measurements) and your six conditions on that report.

## 0. What changed from the original plan

| Original | Now | Why (measured) |
|---|---|---|
| Ollama/llama.cpp for all LLM work | **Backend interface**; OpenVINO GenAI for the voice path, Ollama for the text core, website chat and as a fallback | iGPU prefill on a cache miss: 89-158 ms vs 1.2-4.7 s on CPU |
| LLM emits ~20-token JSON | **Intent-only call (~6 tokens)**, slots extracted by code and grounded in the utterance | decode is 15-40 tok/s; small models invent slot values |
| Rules as a nice-to-have | **Rules first, gated per intent** by measured precision | 83% of turns are rule-answerable, but state-changing keyword rules mis-fire on off-topic speech |
| Everything in one process | **Defined process layout** (section 4) | `sherpa-onnx` and `onnxruntime` DLLs clash on Windows |

## 1. Acceptance criteria

1. **Deterministic invariants pass at 100%** (no LLM involved): the price offered/accepted is never below `floor_price`; the
   agent never reveals the floor or discount authority; no double booking (same car+slot, or same customer+car); reschedule/
   cancel update the existing booking and never create a duplicate; every escalation trigger escalates; a car's status is
   re-checked immediately before any booking is confirmed; spoken replies are 1-2 sentences with no markdown/URLs and name at
   most 3 cars.
2. **LLM-dependent scenarios >= 90%** on the local model, per scenario, using the production NLU path. Reported with counts and
   confidence intervals; the cloud run happens only if you enable it (the stub stays disabled).
3. **OpenVINO accuracy gate (task 0)**: if exact+ground drops more than 5 points against Ollama Q4_K_M, I stop and tell you
   before building on it.
4. Each phase ends with a stop-and-show. Nothing is committed to the voice path until you approve this plan.

## 2. Tasks, in order, with checkpoints

**T0. FIRST: NLU backend interface + OpenVINO accuracy comparison (checkpoint A, a hard gate).**
* Define `NluBackend` (section 3) with two real implementations (`OllamaBackend`, `OpenVinoBackend`) and the config switch.
* Quick check, same hour: `import openvino_genai` and `onnxruntime` in one process and run both. If they clash, OpenVINO moves
  into its own process (section 4) before anything else is built.
* Run the three existing sets (dev 31, heldout 40, fresh 29) and, when you supply it, **your recorded + typed set**, through:
  Ollama Q4_K_M 4B and 1.7B vs OpenVINO int4 4B and 1.7B, same prompts, same grammar, temperature 0, full-JSON mode
  (apples to apples), plus the intent-only mode (intent accuracy; slots then come from code).
* Metric: exact+ground, combined and per set, plus intent accuracy, plus McNemar discordant counts (the same utterances go to
  both runtimes, so a paired test is the right one). **With 100 utterances one utterance is a full point and the noise is about
  +/-5 points, so I will not call a result before your set arrives, or I will say plainly that it is inconclusive.**
* **Gate:** exact+ground (combined) drops by more than 5 points for the primary 4B -> stop, report, ask. Same check for the 1.7B
  fallback and for the static-prompt CPU configuration.

**T1. Domain model + repositories.** Typed models (`Car`, `Branch`, `Slot`, `Booking`, `Customer`). `InventoryRepository` and
`BookingRepository` as `Protocol`s; SQLite mock behind them with ~60 fake cars across 2 fake branches (make, model, year, km,
price, `floor_price`, status, branch, body type, `regional_spec`, `service_history`, `image_urls`). Booking writes use
transactions and a uniqueness constraint so a double booking is impossible at the database level, not just in code. Nothing
above this layer imports SQLite. Branch working days/hours come from config (nothing assumes a weekend).

**T2. Negotiation policy engine** (plain code, config in `config/default.yaml`, all marked PLACEHOLDER). Inputs: car, offer,
round number. Output: `Accept | Counter(amount) | NeedsManager | Escalate(below_floor) | Refuse(max_rounds)`, plus the exact
phrasing key. Counters move in `counter_steps` steps and never jump to the floor; max 3 rounds; the engine never returns a
number below the floor and never exposes it. Property tests with a seeded RNG over thousands of random cars/offers (no
Hypothesis: it is MPL-licensed).

**T3. Service layer / tools.** `search_cars`, `get_car`, `check_test_drive_slots`, `book_test_drive`, `reschedule`, `cancel`,
`propose_price`, `send_details` (a `MessageChannel` interface, WhatsApp mocked), `escalate_to_human`. Typed arguments and results
(pydantic), no LLM inside. Idempotency keys so a retried call updates rather than duplicates. Status re-check inside the same
transaction that confirms a booking. This is the package the website chat would also import (one backend, two channels).
**Checkpoint B: show the tool layer + invariant tests.**

**T4. State machine + templates + voice-rule validator.** Explicit transition table (greeting, interest, answering,
negotiating, slot, contact, confirm, goodbye) with per-state allowed intents (the same table the grammar uses). Replies come from
templates; read-back of name, phone, date and time before any confirmation. A validator rejects replies over 2 sentences, with
markdown/URLs, or naming more than 3 cars (unit-tested, used in the replayer).

**T5. NLU pipeline.** Order per turn: (1) language check (Arabic script -> "English only" + handoff offer); (2) rules; (3) LLM
intent-only call if the rules decline or the intent's gate is closed; (4) code slot extractors (numbers, dates, phones, names,
make/model from the inventory-built gazetteer with fuzzy matching); (5) grounding filter; (6) out-of-scope veto before any
state-changing action. The LLM never sees prices, floors or discount authority.

**T6. CLI + scenario replayer + tests.** `chat` CLI; replayer that runs scripted fake-customer conversations N times with
typed variants. Scenarios (original six + your five): happy path; haggling below floor; car sold mid-call; rescheduling;
off-topic; angry customer -> escalation; caller claims to be the manager; asks for the lowest price; misheard number (read-back
catches it); Arabic speaker; changes mind mid-booking. Loader for **your set** (format in section 6).
**Checkpoint C: show pass rates, failures and the latency log.**

**T7. Latency/observability module** (per-stage timings to console + JSONL; no transcripts or phone numbers by default),
exercised in text mode so Phase 3 only adds audio stages.

## 3. NLU backend interface (config-selected; no code yet, signatures only)

```
class NluBackend(Protocol):
    name: str
    def health(self) -> BackendHealth                      # device present? model compiled? used at startup
    def classify(self, state: State, car: str | None, text: str) -> IntentResult   # intent-only, grammar-constrained
    def complete_json(self, state, car, text) -> dict      # full JSON mode: accuracy comparisons only
    def answer(self, question: str, context: str, max_sentences: int = 2) -> Iterator[str]  # open questions, sentence stream
```
Config (`config/default.yaml`):
```
llm:
  backend: openvino_gpu                       # openvino_gpu | openvino_gpu_small | openvino_cpu | ollama
  fallbacks: [openvino_cpu, ollama]           # tried in order if health() fails at start or a call errors/times out
  prompt_mode: auto                           # auto: per-state on iGPU, static on CPU/Ollama
  backends:
    openvino_gpu:       {runtime: openvino, device: GPU, model: Qwen3-4B-int4-ov}
    openvino_gpu_small: {runtime: openvino, device: GPU, model: Qwen3-1.7B-int4-ov}
    openvino_cpu:       {runtime: openvino, device: CPU, model: Qwen3-1.7B-int4-ov}   # do not pin thread counts
    ollama:             {runtime: ollama, model: "qwen3:4b-instruct-2507-q4_K_M"}
```
Every fallback is reachable by changing one line; a startup log line states which backend is active and why. The website chat and
the Phase 1 text core use `ollama` by default; the voice path defaults to `openvino_gpu`. Per-call timeouts trigger fallback
(budget: 1.0 s p95 for the NLU call). The cloud stub stays a disabled fifth entry.

## 4. Process layout (decision for you to approve, before Phase 3)

| Process | Contains | Why separate or together |
|---|---|---|
| **agent** (main) | Pipecat pipeline (Phase 3), Silero VAD + turn model (onnxruntime), state machine, rules, policy, repositories, **OpenVINO STT + LLM in-process** | one process keeps STT->NLU latency lowest; OpenVINO ships no `onnxruntime.dll`. **To verify in T0** that OpenVINO and onnxruntime coexist; if not, OpenVINO moves to a service process |
| **tts** | sherpa-onnx Piper (owns its `onnxruntime.dll`) | the DLL clash; also isolates TTS crashes and lets barge-in cancel by killing a request, not the call |
| **ollama** (existing server) | Ollama models | already a separate process; used by the text core, website chat and the fallback |
| website chat (later) | imports `dealer_core` or calls it over HTTP | out of scope for Phase 1 |

TTS interface: `TtsEngine.synthesize(text) -> AsyncIterator[AudioChunk]` plus `cancel(request_id)` and `health()`. Transport: a
local pipe now (measured round trip < 0.1 ms; time to first audio 112-280 ms for a ~11-word sentence on the primary stack).
Chunk = one sentence from sherpa's callback. A supervisor in the agent process restarts the TTS process on crash and the
current turn falls back to a cached "one moment" clip. No component loads both `sherpa_onnx` and an `onnxruntime` engine
(Silero VAD, Kokoro, Moonshine) in one process; that rule gets a startup self-check.

## 5. Fast path with precision gates (your condition 5)

* **Measured today** (rules unchanged): 98.0% on 49 state-changing firings (lower bound 89.3%); `angry` 80%; 14 of 32
  adversarial off-topic probes fired a state-changing rule. **A 98% claim needs >= 189 correct firings per intent.**
* Design: `config/nlu_gates.yaml` lists, per intent, `enabled: true|false`, its measured precision, n and 95% lower bound,
  and the date of the measurement set. A script (`measure_gates`) recomputes it from any labelled set and only flips an intent
  to enabled when the **lower bound >= 98%** (target you set); a pytest fails if the file claims more than the data supports.
* **Until your set arrives, all state-changing intents are gated closed (defer to the LLM)**; harmless intents (search,
  policy, car info, send details, greeting, goodbye) are enabled but measured too. Cost on the fresh set: LLM share 17% -> 62%,
  mean T0->T2 ~400 -> ~730 ms (primary) / ~630 ms (CPU fallback); the typical turn is then an LLM turn (~1.0 s / 0.8 s).
* **Independent of gates, every state-changing action needs a read-back + explicit yes**, and the policy engine enforces
  prices, so a mis-fired intent costs a clarification turn, not a wrong booking or an under-floor price.
* **Out-of-scope check** (the "tell me a joke" false escalation): escalation rules become anchored patterns; an `out_of_scope`
  veto from the LLM runs before any escalation or state change that did not come from a high-confidence anchored rule; a
  standing probe set (the 32 above, extended with your off-topic speech) must show **zero** state-changing fires.
* I will **not tune on the fresh set**. Fixes land with regression tests drawn from your new set and from the probe set.

## 6. Your new utterance set (what I need from you)

CSV, one row per utterance: `id, state, car_in_focus, utterance, gold_intent, gold_slots_json, source (spoken|typed), notes`.
Spoken ones: WAV 16 kHz mono named `<id>.wav` in `benchmarks/recordings/` (same rules as `docs/RECORDING_GUIDE.md`); I take the
reference text from the CSV. The gate needs volume to be meaningful: **~200 labelled utterances per state-changing intent is
the target; even 50 each gives a usable lower bound for deciding which gates stay closed.** Include off-topic speech, hedged
yes/no, and anything an angry or confused customer would really say. This set is also the T0 comparison set.

## 7. Repository layout

```
src/dealer_core/
  config.py  models.py  errors.py
  repos/      base.py (Protocols)  sqlite_inventory.py  sqlite_booking.py  seed.py
  policy/     negotiation.py
  tools/      search.py booking.py pricing.py messaging.py escalation.py   (the shared service layer)
  dialogue/   states.py  machine.py  templates.py  voice_rules.py
  nlu/        backend.py  openvino_backend.py  ollama_backend.py  rules.py  extractors.py  grounding.py  gates.py
  tts/        base.py  client.py  (server in tts_service/)
  observability/latency.py
tts_service/  sherpa Piper server (own process)
cli/          chat.py  replay.py
tests/        unit/  invariants/  scenarios/  nlu/
```
New dependencies: none beyond Phase 0b's (`openvino-genai`, `rapidfuzz`, `pytest`); no new copyleft. `uv` extras: `core`, `ov`, `ollama`.

## 8. Risks and unknowns

* **OpenVINO accuracy on the int4 builds is unmeasured** (T0 gate). The Q4_K_M and OpenVINO int4 quantizations differ.
* **OpenVINO + onnxruntime coexistence in one process is untested**; handled by the T0 check and the process fallback.
* 100 labelled utterances cannot settle a 5-point accuracy question; I will say so rather than guess.
* Endpointing/turn-detection latency is unmeasured (Phase 3); the 0.6 s / 1.3 s / 1.8 s first-meaningful-word figures assume ~300 ms.
* Real accented speech: all STT numbers are on synthetic audio. Your 37 recordings are still needed.
* Run-to-run noise on this laptop is +/-30-50%; every latency claim is reported as a range from interleaved repeats, on AC only.
* Power: the voice agent needs AC power and a performance power mode on the deployment laptop (battery cost 3x).
* The iGPU is shared by LLM work only; STT must stay on the CPU (measured: sharing slowed NLU from 0.9 s to 1.6 s).
* `espeak-ng` data (GPL-3) is bundled with Piper voices; a legal call for you.

## 9. Decisions I need from you

1. Approve this plan (task order, T0 as a hard gate, process layout in section 4).
2. The utterance set (section 6): when can you provide it, and roughly how many per intent?
3. The 98% gate means state-changing rules stay closed (LLM share ~62%) until your set opens them. Accept that cost, or prefer
   opening some gates earlier at a lower bar? (I recommend not lowering it.)
4. Piper voice: `libritts_r` (CC BY 4.0, needs attribution) or `joe` (CC0)?
5. Is GPL-3 `espeak-ng` data acceptable for an internal, non-distributed deployment?
6. The website's Ollama chat: which model does it use today? I will default to `qwen3:4b-instruct-2507-q4_K_M` unless you say otherwise.

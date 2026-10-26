# Going live: 500 callers at once, on hardware nobody has specified

The requirement: 500 simultaneous callers, replies that sound like a person (not a script), under about a second, on GPUs the company
will provide but has not described. The design does not depend on knowing the hardware: it **measures** whatever it is given and
**adapts**, and it **scales out** by adding machines.

## 1. Quality that adapts to the hardware (built)

Every turn is answered at one of three levels, chosen live by the quality governor (`src/dealer_core/agent/governor.py`):

| Level | Who writes the words | When |
|---|---|---|
| **full** | the model writes the whole reply in its own words, from facts the tools looked up; every sentence is checked before it is spoken | while the model's first sentence arrives within the budget (700 ms by default) |
| **prefix** | code says the facts at once; the model adds the follow-up question | when the model is slower than that (a weak GPU, or peak load) |
| **code** | the code's own wording, no model | when too many turns are waiting on the model, or the model is down |

At every level, booking questions, the read-back and the confirmation are said exactly by code, and nothing is booked without a "yes".
The governor is shared by all calls on a server and keeps trying `full` now and then, so it moves back up when there is room. Its live
state is in the voice server's `/health` (for dashboards and the load balancer).

On a data-centre GPU almost every turn is `full`. On this laptop (one integrated GPU, ~19 tokens/s) `full` takes ~1.2 s to the first
sentence, so turns are mostly `prefix`. The same code, no change.

## 2. Any model server (built)

The model runs in a standard OpenAI-compatible server on the company's machines: **vLLM** (Apache-2.0) is the default choice for NVIDIA
GPUs; OpenVINO Model Server for Intel; llama.cpp server for anything else. The agent connects with one config entry
(`runtime: openai`, `host`, `model`); see `src/dealer_core/nlu/openai_compat.py`. vLLM serves all callers together (continuous
batching) and reuses the shared start of every prompt (prefix caching), which is what makes hundreds of calls per GPU possible.

## 3. Measure the machine you are given (built)

`scripts/capacity_probe.py` measures, on the actual machine and model server:
how many replies can be written at once within the budget (full and prefix), recognitions per second, voice sentences per second; and
says how many callers one machine carries and how many machines the target needs. Then `scripts/load_test.py` confirms it with
simulated callers through the real server. On this laptop: the probe's upper bound is ~40 callers at prefix quality (limited by voice
synthesis); the load test showed ~16 comfortable, ~24 usable. So: probe first, then load-test, then size with a spare.

## 4. The shape of the deployment for 500 callers (to build when the hardware arrives)

```
 callers (browser) --WebRTC--> media server (LiveKit, Apache-2.0)
                                   |
                     voice workers: many small processes, ~15-20 calls each, stateless, scaled by active calls
                     (speech detection, turn-taking, the agent's code, voice synthesis on CPU)
                       |                 |                   |
              model server(s)     speech recognition      Redis (session state)   PostgreSQL (bookings: one per car and slot)
              vLLM, batching      Whisper on GPU,
              + prefix cache      batched
```

* **Model**: at any moment only ~10-20% of callers are waiting on the model (the rest are speaking or listening), so 500 callers means
  ~50-100 replies being written at once. Size it with the probe; keep one GPU spare so a failure does not take the service down.
* **Voice workers**: one process carries ~15-20 calls (measured: beyond that its event loop falls behind). 500 calls = ~30 processes,
  a handful of CPU servers. Stateless: any worker can take any call; a crashed worker loses only its own calls.
* **Media**: LiveKit instead of the current WebSocket for production audio: it handles poor networks, echo and reconnects, and spreads
  calls across workers.
* **Bookings**: PostgreSQL with a unique constraint on (car, slot), so two callers on two servers can never get the same slot. Booking
  requests carry an id so a retry never books twice.
* **Overload**: every server has a call cap and says "busy" (the page falls back to text chat) instead of slowing everyone down; every
  hop has a timeout; the governor drops to `prefix` then `code` before anything breaks.
* **Watching it**: per-turn timings (already logged without transcripts or phone numbers), the governor's level mix, active calls,
  errors; alerts when the 90th-percentile reply time or the share of `code` turns rises.

## 5. Quality is measured, not guessed (started)

`tests/scenarios/test_conversations.py`: realistic multi-turn calls checked for loops, repeated replies, wrong prices and turns that
end without a question, run with no model help (the safety net must hold on its own). Next: grow it to 200+ calls and score the
model-written replies for naturalness, so model, prompt and level choices are made with numbers.

## 6. What is built for scale, and how to run it

| Piece | What it does | Where |
|---|---|---|
| Shared model server | one model for every worker on a machine, OpenAI-compatible like vLLM, batched | `python -m dealer_core.modelserver` |
| Bookings + stock in PostgreSQL | shared by every worker; double booking impossible in the database (unique indexes + car row lock) | `repos/pg_store.py`, `app.database_url` / `DATABASE_URL` |
| Shared call state | each call saved after every turn; a reconnecting call is carried on by any worker (no second greeting) | `shared/state.py` (PostgreSQL, Redis or memory) |
| Worker registry + router | workers report load every 2 s; the router gives each new call the least busy worker, 503 "busy" when all are full | `python -m dealer_core.gateway` |
| Page | asks the router for a worker (`CONFIG.voiceGateway`), sends a call id, reconnects once if its worker drops | `bmw-portal/site/voice.js` |
| Metrics | calls, reply-time histogram, quality-level mix, model load at `/metrics` | `server/metrics.py`, `deploy/prometheus.yml` |
| One-machine cluster | database + model server + N workers + router, restarts a worker that dies | `scripts/run_cluster.py --workers 3` |
| Deployment | container image, one-GPU-machine compose file, Kubernetes for 500 callers (workers autoscaled on active calls, 2 vLLM replicas) | `deploy/` |

Tested here: every booking test on a real PostgreSQL 16 (pgserver), a 16-caller race for one slot across 4 workers (exactly one
booking), workers starting at the same moment, a call handed to another worker mid-booking, the router's choice and its "busy",
the metrics, and the model server's plain, streaming, JSON and continue-a-reply requests.
Not testable on this laptop: vLLM itself, Kubernetes, the ingress, and 500 real callers. Those need the company's hardware; the
capacity probe and the load test (`--gateway`) are the tools to prove them there.

### Proven on this machine (26 Oct, 3 workers + router + shared model + PostgreSQL)

* `scripts/load_test.py --gateway`: 54 calls at 6 / 18 / 30 callers, all assigned by the router, spread evenly (~130 turns per worker),
  0 failed, 0 silent; quality mix 12% full / 88% prefix / 0% code (right for this laptop's GPU). Timings not valid: on battery, and
  RAM at 100% (three workers + model on a 32 GB laptop with other apps open; each worker ~1.4 GB). Production workers get their own nodes.
* `scripts/failover_test.py`: a worker killed mid-booking; the caller reconnected through the router to another worker, which carried the
  booking on (car, day and time from before the crash) and booked it. The killed worker was restarted by the supervisor.

## 7. Under 1 s at 500 callers: the latency budget and what closes it (26 Oct, second pass)

The target is **end of the caller's speech -> first sound they hear, under 1 s**, for every turn, while sounding like a person. The reply is
always "half template, half model": code supplies every fact and the exact booking wording; the model supplies the conversational wording
when it is fast enough (the governor, section 1). What was still in the way, and what was changed:

| Stage | Laptop (measured) | Production target | Notes |
|---|---|---|---|
| End of turn (VAD 0.2 s + Smart Turn) | 0.2-0.35 s | 0.2-0.35 s | unchanged; the floor of the budget |
| Recognition | 0.2 s (Whisper base, CPU) | ~0.05-0.1 s (estimate) | **new:** `voice.stt_url` sends it to a batched GPU server |
| Model's tool decision (turns code cannot route) | 0.25-0.4 s | ~0.1-0.2 s (estimate) | **new:** covered by a pre-rendered "Let me check." when it is slow |
| First sentence | prefix: at once; full: 0.5-1.2 s | full: ~0.2-0.3 s (estimate) | governor picks full / prefix / code per turn |
| First audio | 0.03-0.15 s (Piper); 0 for pre-rendered phrases | same | |

Production figures marked "estimate" are not measured: no GPU here. Run `scripts/capacity_probe.py` and `scripts/load_test.py --gateway` on
the real hardware before sizing.

### The bridge: "Let me check." only when it is needed (built)

Turns the code routes itself (a pick from the list, a booking detail, a price or km question about the car in focus, a plain "yes") already
answer in milliseconds. The slow turns are the ones where the model must first name the tool: nothing could be said until it had.
Now, on those turns, a short lookup line ("Let me check.", "Let me see.", "Let me have a look.", "One moment, let me look.") is spoken at
once, while the model decides and writes. It is pre-rendered by the voice, so it plays immediately. To keep it natural rather than a tic:

* it is said only when the governor's live measurements (decision + first sentence, `expected_wait_ms()`) exceed `voice.bridge_after_ms`
  (450 ms: what is left of 1 s after end of turn, recognition and first audio). On a fast GPU it is simply never said;
* only for requests that sound like a lookup (stock, price, colour, km, times...), never for "thanks", "hello", "bye", "yes";
* never two turns running unless the wait is more than twice the threshold, and never the same words twice in a row;
* the reply that follows drops its own "Sure," / "Of course." so the caller never hears two acknowledgements.

`agent/engine.py: _bridge`, `agent/governor.py: record_decide / expected_wait_ms`, `voice.bridge_after_ms` (0 turns it off).

### Correct with many callers on one worker (fixed)

One engine and one reply writer serve every call of a worker; turns run in parallel threads. Three pieces of per-turn state lived on the
shared objects, so two callers' turns could overwrite each other:
`AgentEngine._last_text` (the decision's spelling-corrected CODE HINT could be built from **another caller's words**), the writer's
temperature (set to 0.2 for a retry and restored, racing other turns), and `last_stream` / `last_level` / `prefix_stats` (debug traces and
the "rejected because" retry text read from another call). The text is now passed explicitly, the retry temperature is a parameter, and
the traces are per thread. Tests: `tests/unit/test_scale.py`.

### Nothing blocks the event loop that carries the audio (fixed)

Each worker's single event loop moves every call's audio frames. Several database calls ran directly on it: the call snapshot after
every turn, the heartbeat to the registry, and the reads when a call starts or resumes. With PostgreSQL a few milliseconds each, at 16
calls that is audible jitter for everyone on the worker, and a slow database stalled all of them. They now run in threads; snapshots are
coalesced per call (only the newest waits while one is being written; the last one is flushed when the call ends).
`shared/state.py: AsyncCallSaver`, `server/app.py`.

### The router under a burst (fixed)

Workers report their load every 2 s. The old `/assign` read every report from the database (one query per worker plus a DELETE, on the
event loop) and gave the call to the least busy worker. In a burst, every call in those 2 s went to the **same** worker (its report did
not change yet), which then told most of them "busy". Now:

* the router keeps the reports in memory, refreshed in the background every 0.5 s: `/assign` makes no database call;
* a call it hands out counts against that worker until the worker's own report can include it (3 s);
* "power of two choices": two workers with room are drawn at random and the emptier one wins, so two router copies spread a burst too;
* workers report again ~0.2 s after a call starts or ends, not only every 2 s.

Tested: 512 callers in one report interval on 32 workers x 16 places fill every worker exactly to 16 and the 513th hears "busy"; 300 callers
through two router copies never put more than 16 on one worker. `gateway/__main__.py: Assigner`.

### Recognition that scales with the number of calls (built)

Each worker had one Whisper behind a lock, so the turns of all its calls waited in line, and Whisper took CPU from every call's speech
detection and voice. Two options now:

* `voice.stt_url` (production): any server with the OpenAI transcription API (vLLM with `openai/whisper-large-v3-turbo`, speaches,
  faster-whisper-server). Turns from every worker are batched on a GPU; the worker runs no recognizer. A failed or slow transcription is
  "not heard" (the caller is asked again); a worker that cannot reach the server at start-up does not start. `deploy/` runs one.
* `voice.stt_copies` (CPU-only machines): several local copies, so a worker's calls transcribe in parallel.

### Work started during Smart Turn's hold (built)

When Smart Turn hears a pause but judges the sentence unfinished, the turn stays open until the caller speaks again or 0.6 s of silence
passes (`voice.smart_fallback_secs`). Usually the caller had in fact finished, and only then did recognition and the model's decision
start. Now the moment a pause is held, the turn so far is recognised and (agent engine) the decision is made in the background
(`voice/speculate.py`). If the silence then ends the turn, both are used as they are: the turn ended on exactly that audio, and the
decision is only used for the same words and if the call has not moved on. If the caller speaks again, both are dropped. This removes
recognition + decision (~0.5 s on the laptop) from the turns Smart Turn hesitated on. Turns Smart Turn ends at once gain nothing (recognition
already starts at once there). Cost: the work done for a caller who carries on is wasted; the decision is skipped when the governor says
the model has no room (`has_room`). The audio from just before speech was detected (0.5 s) is kept, so the first syllable is not lost.
`voice.speculate` (on).

### The decision, constrained to what is in play (built, vLLM)

The model's tool decision was free text, parsed. On an OpenAI-compatible server that supports it, the request now lists every allowed
answer: each tool, and the tools about one car paired only with the cars in play (the code hint for the caller's words, the car in focus,
the cars just listed). vLLM then cannot answer anything else: no chatter, no invented stock id, a few tokens. Field:
`structured_outputs: {"choice": [...]}` (vLLM 0.12+; it replaced `guided_choice`). A server that refuses it (HTTP 400) is asked in free
text from then on; one that ignores it (the local `dealer_core.modelserver`) answers in free text, parsed as before. Not used with the
in-process OpenVINO model: it compiles a grammar per list (seconds) and the list changes every turn.

### Images pinned, production out of test mode (done)

`deploy/` pins vLLM `v0.31.0`, PostgreSQL `16.15`, Prometheus `v3.15.0`; our own image takes a release tag (`VOICE_TAG` in compose,
`RELEASE_TAG` in Kubernetes). Workers run with `VA__VOICE__MODEL_ONLY=false`.

### Not done, and why

* **Prompt order for the prefix cache.** On a closer look it gains little: the conversation in the decision prompt is a sliding window
  (the last 4 lines), so the prompt after the system part changes every turn whatever the order. The long system prompt, the part worth
  caching, is already first and shared by every call.
* **WebRTC media (LiveKit)** instead of the WebSocket (section 4): needs the website's page (`bmw-portal`, another repository) and a LiveKit
  server; the worker side is a transport swap in `server/app.py`.
* The intent call (state-machine engine) still sends `guided_json` next to `response_format`. vLLM 0.12+ no longer reads `guided_json`, but
  its request models accept unknown top-level fields, so `response_format` should carry the schema: confirm on the pinned version with
  the capacity probe, and drop `guided_json` once the local model server reads `structured_outputs`.

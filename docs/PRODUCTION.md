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

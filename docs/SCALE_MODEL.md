# From the laptop demo to 500 and 1,000+ callers: what is measured, what is modelled, what is not known

Decision (28 Oct, owner): there will be no GPU to measure on. The product is a **demo that runs on one laptop**, built so that scaling it out is a matter of adding machines and a model server, and so that this can be shown on paper and in simulation. This file separates three kinds of statement, as the plan does:

* **MEASURED**: a run on the demo machine (Apple M3 Pro, 36 GB, AC power) or a recorded call.
* **MODELLED**: the scale simulation (`scripts/simulate_scale.py`) or arithmetic from the manifest. The simulation runs the real router, quality governor, call cap and probe against **stand-in** model, recogniser and CPU servers whose speeds are typed in (decision 120 ms, first sentence 300 ms: the *targets*, not measurements). It shows that the control mechanisms hold the answer time if the hardware is as fast as that. It does not show that any hardware is.
* **UNKNOWN**: needs hardware nobody has. Listed at the end; nothing in the demo claims them.

## 1. What the demo machine does (MEASURED, one M3 Pro)

| Question | Result | Where |
|---|---|---|
| Calls one worker carries comfortably | about 16 (flat first-sound median to 16, 27% of turns over 2 s at 24; the event loop falls behind at 32) | `docs/SCALING.md`, 23 Oct, Intel laptop |
| 16 simultaneous calls through the real WebSocket, worker + shared recogniser + local model server | 0 failed calls, 0 silent turns, first sound median 0.91 s, p90 1.40 s, worst 3.34 s, 4% over 2 s (synthetic callers, everything on one machine) | `docs/ARCH_REVIEW.md` section 19 |
| Model-written first sentence, Qwen3-4B 4-bit on the MLX server | 620 to 1,168 ms (the governor's estimate); probe 714 to 856 ms median at 1 at once; the 700 ms budget is met only sometimes | section 19 |
| Same on Ollama | 1,251 to 1,479 ms | `docs/ARCH_REVIEW_new.md` |
| Model throughput on the laptop | about 1.2 to 1.7 full-prompt requests a second whatever the concurrency (it does not batch) | section 19 |
| What the governor does here | model-written level on none of 168 counted turns; the demo runs at the code and prefix levels | section 19 |
| Hearing | WER 3 to 4% with the hint on 9 recorded turns; 7 of 9 exactly right | section 19 |

So the demo is honest about what it is: code-written facts at once, the model adding a short question, answers in about 0.9 s. It is not the "model speaks" experience, and cannot be on this machine.

How the numbers above were produced (one machine, nothing else changes): an MLX model server (`mlx_lm.server`, Qwen3-4B-Instruct-2507 4-bit, port 8003), the repository's recogniser service (`python -m dealer_core.sttserver --port 8002`), and a worker:

```bash
export VA__LLM__BACKENDS__GPU_SERVER__HOST=http://127.0.0.1:8003/v1 VA__LLM__BACKENDS__GPU_SERVER__MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit
export VA__DIALOGUE__DECISION_BACKEND=gpu_server VA__VOICE__STT_URL=http://127.0.0.1:8002/v1
python -m dealer_core.server --agent --prefix --backend gpu_server --site ../bmw-portal/site --port 8766 --max-calls 16
```

Send one real request to the model server before the first call: it loads the model on the first request (1 min 42 s).

## 2. How a deployment grows (MODELLED: arithmetic and the simulation)

Every worker is stateless (call state in PostgreSQL), a router hands each caller to the least-loaded of two random workers, and the autoscaler keeps workers about 70% full. The formulas, with the manifest's numbers (`deploy/k8s/voice.yaml`):

| | Formula | 500 callers | 1,000 | 1,500 |
|---|---|---|---|---|
| Voice workers (16 calls each, 70% target) | callers / (16 x 0.7) | 45 | 90 | 134 |
| ...the simulation reached | | 45 | 89 | 132 |
| Worker CPU and RAM (the manifest asks 3 to 4 cores, 6 to 8 GB each; not measured under load) | workers x that | 135 to 180 cores, 270 to 360 GB | 270 to 360 cores, 540 to 720 GB | 400 to 540 cores, 800 to 1,070 GB |
| `maxReplicas` needed | workers at the target | 45 (manifest: 72) | 90 (**raise from 72**) | 134 (**raise to about 140**) |
| Recognitions a second | callers / 15 s between a caller's turns | 33 | 67 | 100 |
| Model requests a second (1 to 2 calls per turn: a decision, a reply; how many turns call the model is unknown) | 1 to 2 x the line above, at most | up to 33 to 67 | up to 67 to 133 | up to 100 to 200 |
| Model requests in flight at target speed (simulation peak) | | 36 | 44 | 55 to 58 |
| Database connections, ceiling (B6) | workers x `app.db_pool_size` (8) | 360 | 720 | 1,070 |
| Registry reads by the router (B6) | 1 query per 0.5 s per router copy, independent of the worker count | 2 per s | 2 per s | 2 per s |
| Registry writes by workers | workers / 2 s, more when load changes | about 23 per s | about 45 per s | about 67 per s |

Three consequences:

* **The manifest as shipped (`maxReplicas` 72) can hold 1,000 callers** in the simulation (72 workers, 0.0% turned away, answer p90 0.88 s) but only by running them about 87% full, with no spare room for a spike. For 1,000 at the 70% target set `maxReplicas` to about 90, for 1,500 to about 140. (Not changed in the manifest: with no hardware to run it on, changing a number nobody can test is just a different guess.)
* **The database is the first thing to fail**, not the model: even with one pool per worker (B6), 90 workers can ask for 720 connections against PostgreSQL's default 100. pgbouncer in transaction mode (or `max_connections` raised, or fewer workers carrying more calls) is needed before about 12 workers. Not added to `deploy/` (an image tag and settings that cannot be checked offline).
* **The model server is sized in requests in flight and requests a second, not callers.** The simulation says about 36, 44 and 55 requests must be served together at the target speed at 500, 1,000 and 1,500 callers, and that with fewer the mechanisms degrade in the order they were built to: wording first, then shedding, with the answer time held (table below). Whether a given server does that is what `scripts/capacity_probe.py --model-only --no-stop` measures; the laptop's 1.2 to 1.7 requests a second at 0.7 to 1.2 s each is roughly one such request in flight, so about 40 laptop-sized model servers per 1,000 callers is the arithmetic (INFERRED, crude: batching on a real server is the whole point and is unmeasured).

### The simulation at 500, 1,000 and 1,500 callers (protection on; 10 simulated minutes, callers ramped in over 90 s; same result every run)

| Scenario | Turned away | Answer p50 | p90 | p99 | Over 2 s | Workers (max) | Calls |
|---|---|---|---|---|---|---|---|
| 500 callers, fast GPUs, up to 72 workers | 0.0% | 0.45 s | 0.88 s | 0.99 s | 0% | 45 | 3,117 |
| 1,000 callers, 72 workers (the manifest) | 0.0% | 0.45 s | 0.88 s | 0.99 s | 0% | 72 | 6,318 |
| 1,000 callers, up to 120 workers | 0.2% | 0.45 s | 0.88 s | 0.99 s | 0% | 89 | 6,321 |
| 1,500 callers, up to 180 workers | 0.1% | 0.45 s | 0.88 s | 0.99 s | 0% | 132 | 9,359 |

Without the protection (fixed cap, no shedding, 5 s deadline, autoscaling on calls in progress only) 1,500 callers had 11.6% turned away and 1,000 callers 1.8 to 1.9%: the autoscaler counting the callers just turned away is what closes that.

### How much model capacity the simulation needs ("slots" = requests the model server(s) serve together at full speed)

| Callers | Slots | Turned away | Answer p90 | Over 2 s | Model full / prefix / code % | Shed % |
|---|---|---|---|---|---|---|
| 500 | 20 or more | 0.0% | 0.88 s | 0% | 100 / 0 / 0 | 0 |
| 500 | 10 | 0.0% | 1.05 s | 0.6% | 56 / 44 / 0 | 0 |
| 1,000 | 30 or more | 0 to 0.2% | 0.88 s | 0% | 100 / 0 / 0 | 0 |
| 1,000 | 20 | 0.0% | 1.13 s | 0.2% | 73 / 27 / 0 | 0 |
| 1,000 | 10 | 0.0% | 0.99 s | 0.3% | 2 / 53 / 46 | 44 |
| 1,500 | 45 or more | 0 to 0.1% | 0.88 s | 0% | 100 / 0 / 0 | 0 |
| 1,500 | 30 | 0.0% | 1.13 s | 0.1% | 77 / 23 / 0 | 0 |
| 1,500 | 10 | 0.0% | 0.78 s | 0.1% | 1 / 31 / 68 | 64 |

With too little model capacity the callers are not turned away and the answers stay near a second; what is given up is quality (the code speaks, the model adds less). That is the property the demo can show without a GPU. Reproduce: `python scripts/simulate_scale.py --model-capacity`, and `--only "callers, fast GPUs"` for the table above.

## 3. What is UNKNOWN (and what to do when hardware exists)

| Unknown | Measured by | Why it matters |
|---|---|---|
| Whether a real GPU server serves the first sentence in under 300 ms at 25 to 100 requests at once | `scripts/capacity_probe.py --model-only --no-stop` (`docs/PHASE0_RUNBOOK.md`) | the stand-in speeds above are targets |
| Calls per worker on production CPUs | `scripts/capacity_probe.py`, `scripts/load_test.py` | 16 is a laptop number; workers scale inversely with it |
| How many connections a worker really opens under load | PostgreSQL's `pg_stat_activity` during a load test | only the ceiling is known |
| How many turns call the model at all, and how many calls each makes | the governor's counters (`voice_turn_level_total`) in a load test | sets the model requests a second within a factor of two |
| Real network, WebRTC, ingress limits on open WebSockets, TLS | a staging deployment | the simulation has none of them |
| Recognition quality of a GPU Whisper on real callers' noise | `scripts/hearing_check.py --stt-url` | only 9 recorded turns exist with reference text |

## 4. What this means for the work plan

* **Phase 0 (GPU baseline)**: closed as "no GPU available". The runbook stays for the day a card turns up; this file is the measured laptop baseline and the modelled scale-out.
* **Phase 1**: done (PR #14).
* **Phases 2, 3**: their latency targets (first sentence under 300 ms) cannot be tested here. What can be built and judged on the laptop is the part that does not depend on speed: one model call per turn instead of two, and the MLX server as the local default (faster than Ollama here). The "model speaks, code guards" change can be built behind a flag and judged on correctness and fact-safety (replays and the fact-fidelity suite), not latency.
* **Phase 4**: hearing and overlap can be scored on the recorded turns.
* **Phase 5**: the policy (spread, autoscale on demand at 70%, shed, hold the answer time) is shown by the simulation and the arithmetic above. Its real-cluster items stay unverified and are listed in section 3: the knee on production CPUs, a warm floor sized from a real quiet hour, and alerts checked against a live metric.

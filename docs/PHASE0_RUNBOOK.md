# Phase 0 runbook: the baseline on GPU hardware

For the session that has the GPU. It is the procedure `docs/ARCH_PLAN.md` Phase 0 asks for, written down after rehearsing every step that can be rehearsed on the M3 Pro laptop (`docs/ARCH_REVIEW.md` section 19). Nothing here has been run on a GPU: where a step needs one, it says what to check instead of what to expect.

Nothing in this runbook changes the code. The one configuration setting that differs from `deploy/k8s/voice.yaml` is flagged in step 3.

## What Phase 0 must produce

All of these, each with the machine it was taken on, written into `docs/ARCH_REVIEW.md` as a new numbered section:

1. `scripts/capacity_probe.py`: the model's time to a first sentence at 1, 2, 4, 8, 16, 32 at once, for the full and the prefix prompt.
2. `scripts/load_test.py`: first sound median, p90, worst and share over 2 s at **1, 8 and 16 callers on one worker**.
3. The governor level each turn actually ran at, at each of those loads.
4. The per-worker knee (the caller count where p90 first passes 1 s, and where the share of turns over 2 s passes 10%).
5. `scripts/hearing_check.py` against the GPU recogniser on the recorded turns.
6. The decision gate: **is the model's first sentence under 300 ms at the design load?** If yes, Phases 2 and 3 go ahead as written. If no, the model or the GPU is wrong and that is fixed first.

## 1. What to stand up

| Piece | What | Where it is set |
|---|---|---|
| Model server | vLLM, a 7 to 8B instruct model. Candidates in the plan: Llama 3.1 8B Instruct, Qwen2.5 7B Instruct. Run both and compare: the choice is a measurement. | `deploy/k8s/voice.yaml` `model-server` |
| Recogniser | Whisper on a GPU behind the OpenAI transcription API (`POST /v1/audio/transcriptions`) | `deploy/k8s/voice.yaml` `stt-server` |
| Workers | `python -m dealer_core.server --agent --prefix --backend gpu_server`, no recogniser of their own | `deploy/k8s/voice.yaml` `voice-worker` |
| Load generator | `scripts/load_test.py`, **on a different machine from the worker** (on one machine the generator's own speech synthesis and audio handling compete with the worker for the same cores; the rehearsal could not avoid it) | |

Two things in the repository do not match the plan and need a decision before the run:

* The plan says a 7 to 8B model; `deploy/k8s/voice.yaml` and `config/default.yaml` (`llm.backends.gpu_server.model`) say `Qwen/Qwen3-14B-AWQ`. A 14B model has a slower first token than the plan's candidates. Run the plan's two candidates; run the 14B only if the gate fails.
* The manifest pins `vllm/vllm-openai:v0.31.0` and its own comment says "confirm the exact serving flags for the vLLM version you pin" for the Whisper server. Nothing in this repository has been run against that image. Confirm the flags first.

## 2. Check the servers before any worker

```bash
curl -s $MODEL/v1/models          # the model's name, exactly as a worker must send it
curl -s $STT/health
```

**Send one real request to each and wait for the answer before starting a test.** In the rehearsal the first chat request to a freshly started `mlx_lm.server` took 1 min 42 s (it downloads and loads the model on the first request, not at start-up), and `capacity_probe.py`'s first call timed out at its 20 s limit. A GPU server that loads lazily would do the same.

## 3. Start one worker, in the production shape

Environment, from the manifest plus two things the manifest does not set:

```bash
export VA__LLM__BACKENDS__GPU_SERVER__HOST=http://<model-server>:8000/v1
export VA__LLM__BACKENDS__GPU_SERVER__MODEL=<the name from /v1/models>
export VA__VOICE__STT_URL=http://<stt-server>:8000/v1
export VA__VOICE__STT_REMOTE_MODEL=<the STT server's model name>
export VA__VOICE__MODEL_ONLY=false
export VA__DIALOGUE__DECISION_BACKEND=gpu_server      # NOT in deploy/k8s/voice.yaml, see below
unset VA_DUMP_AUDIO_DIR                               # the audio recorder writes a file per frame on the event loop (review B7)
python -m dealer_core.server --agent --prefix --backend gpu_server --site <bmw-portal/site> --port 8766 --max-calls 16
```

**Why the decision backend is set here.** `config/default.yaml` and `config.py:173` default `dialogue.decision_backend` to `openvino_gpu` (the Intel iGPU model), and the agent engine builds its tool-decision chain from that (`app.py:113-117`). `deploy/k8s/voice.yaml`, `deploy/Dockerfile` and `deploy/docker-compose.yml` start the worker with `--backend gpu_server` and never set it, and `--backend` only selects the reply model. So as written, a pod's tool decisions are tried on `openvino_gpu`, then `openvino_cpu`, then `ollama` (`llm.fallbacks`), none of which a CPU pod is meant to have. INFERRED from the code; not run in a pod. The rehearsal showed the same thing from the other side (MEASURED by loading the config, not by tracing a decision): the laptop's `.env` sets `VA__DIALOGUE__DECISION_BACKEND`, and the configuration a worker started with `--backend gpu_server` loads has `decision_backend` = `ollama`, so `--backend` does not move it. For the baseline, set it as above. Whether the manifests should set it is a change for a later phase, not Phase 0.

On the worker's log you want to see:

```
NLU backend active: gpu_server (openai-compatible <model> at http://<model-server>:8000/v1)
Model speed measured before the first call: <N> ms to a first sentence
```

That second line is the governor's start-up probe. Write it down: it is the first number of the baseline and the one the governor acts on.

## 4. The measurements, in this order

Run each twice and keep both; the rehearsal's single runs are not repeats.

**4a. Capacity probe** (the model alone, no workers needed):

```bash
python scripts/capacity_probe.py --backend gpu_server --levels 1 2 4 8 16 32 --portal-data <bmw-portal/site/data.js>
```

Read the `full` and `prefix` rows. The `full` row is the model-written first sentence: the Phase 0 gate is its median at the design load. The recognition and voice lines at the end of the output are measured on the machine running the probe, with its own Whisper and Kokoro, **not** on the GPU recogniser: ignore them here. The per-level medians come from `2 x N` requests, so at 1 or 2 at once the p90 is the slower of two samples.

**4b. Load test, one worker** (the worker from step 3, `/health` snapshots around each level):

```bash
for n in 1 8 16; do
  curl -s http://<worker>:8766/health > health_before_$n.json
  python scripts/load_test.py --url ws://<worker>:8766/voice --origin http://localhost:8080 --levels $n --mix --stagger 2 --voice <caller voice>
  curl -s http://<worker>:8766/health > health_after_$n.json
  cp logs/load_test.json load_test_$n.json
done
```

`--voice` is a Piper voice name (default `amy`, the method in `docs/SCALING.md`). On a machine without the Piper models use `kokoro:<name>` with a voice other than the agent's (`tts.kokoro_voice`, default `sarah`).

The governor level per turn is the difference of `quality.turns` (`full`, `prefix`, `code`) between the two `/health` snapshots. The same counters are on `/metrics` as `voice_turn_level_total{level}`. Share at full = `full / (full + prefix + code)`.

`load_test.py` reports time to the first **sound**, not to the model's first **sentence**. The plan's "time to first sentence" is the model's part of that: take it from 4a at the matching concurrency, and from the worker's `voice_model_first_ms` (the governor's estimate) in `/health` `quality.full_first_ms`. Say which of the three a number is.

To find the knee, repeat 4b at 4, 12, 20, 24 and 32 (the worker's `--max-calls` must be at least the level; admission may hold the cap lower, which `/health` `cap` shows).

**4c. Hearing:**

```bash
python scripts/hearing_check.py 20261027_215011_9456_in --stt-url http://<stt-server>:8000/v1 --stt-model <name>
python scripts/hearing_check.py 20261027_215011_9456_in                     # the in-process base recogniser, same machine, for comparison
```

`--stt-url` is new (section 19). The recorded turns and their reference lines are on the laptop only (`logs/audio_dump`, not committed); copy the three files with that stem to the machine that runs the script. Only one recorded call has reference lines, so the sample is nine turns: read differences of a few points as noise.

## 5. What to write down

For every number: the machine (GPU model and count, CPU, RAM, the load generator's machine), the model and quantisation, the vLLM version, the date, whether it was a repeat, and one of MEASURED / INFERRED / GUESS. Where a number was not taken, write "unknown".

| | 1 caller | 8 callers | 16 callers |
|---|---|---|---|
| Model first sentence, median / p90 (probe 4a) | | | |
| First sound, median / p90 / worst / over 2 s (4b) | | | |
| Governor: turns at full / prefix / code (4b) | | | |
| `quality.full_first_ms` at the end of the level | | | |
| Worker cap (`/health cap`) at the end | | | |

Then the knee, the hearing table, and the gate decision in one sentence.

## 6. Do not

* Do not treat a laptop number as a GPU number or the reverse. The rehearsal numbers are in `docs/ARCH_REVIEW.md` section 19 under the laptop's name.
* Do not change a threshold (`model_budget_ms`, the deadlines, the governor) during Phase 0. They are re-derived from these numbers in Phase 2.
* Do not leave `VA_DUMP_AUDIO_DIR` set.

# Phase 0 runbook: the baseline on GPU hardware

For the session that has the GPU. It is the procedure `docs/ARCH_PLAN.md` Phase 0 asks for, written after rehearsing every step that can be rehearsed on the M3 Pro laptop (`docs/ARCH_REVIEW.md` section 19). **Nothing here has been run on a GPU.** The card is not known yet, so the settings are given per memory class and every one of them is a GUESS until a run confirms it. No step in this file starts by itself, and none should be started by an assistant: renting or starting hardware is the owner's decision.

Labels, as in the plan: MEASURED = a number from a run supports it; INFERRED = follows from the code or arithmetic; GUESS = must be measured before it is believed.

## 1. What Phase 0 must produce

Each with the machine it was taken on, written into `docs/ARCH_REVIEW.md` as a new numbered section (or added to section 19):

1. `scripts/capacity_probe.py`: the model's time to a first sentence, **at 1, 2, 4, 8, 16, 25, 32, 50, 64 and 100 at once**, full and prefix prompt, with requests per second at each level (section 3).
2. `scripts/load_test.py`: first sound median, p90, worst and share over 2 s at 1, 8 and 16 callers on one worker (section 6; optional on a rented card).
3. The governor level each turn actually ran at, at each of those loads (section 6).
4. The per-worker knee.
5. `scripts/hearing_check.py` against the GPU recogniser (section 6).
6. The decision gate (plan, Phase 0): **is the model's first sentence under 300 ms at the design load?** Defined in section 3.

## 2. Choosing the settings by memory (all GUESS)

The three candidates: **Llama 3.1 8B Instruct** (plan), **Qwen2.5 7B Instruct** (plan), **Qwen3-14B-AWQ** (what `deploy/k8s/voice.yaml` and `config/default.yaml` name). Licence note, from `docs/LICENSES.md`: Llama is under the Llama Community License, which the project treats as benchmark-only; a Llama result can inform the choice, picking it needs a licence decision. Qwen2.5-7B and Qwen3 are Apache-2.0 on their model cards (Qwen2.5-7B from memory: check the card).

Quantised repositories (names from memory: check they exist before the rental starts): `Qwen/Qwen2.5-7B-Instruct-AWQ`, `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`, `Qwen/Qwen3-14B-AWQ`. Llama 3.1 is a gated repository: the licence must be accepted on the Hugging Face account and `HF_TOKEN` set in the shell on the rented machine (never in the repository, never in a file that is copied back).

How the table was worked out, so a row can be redone for the real card (INFERRED; architecture numbers from memory, check each model's `config.json`):

* Weights: fp16 about 15.2 GB (Qwen2.5-7B) and 16 GB (Llama 8B); AWQ 4-bit about 5.5 to 6 GB; Qwen3-14B-AWQ about 9.5 GB; Qwen3-14B fp16 about 29.5 GB.
* KV cache per token, fp16: Llama 3.1 8B 32 layers x 8 KV heads x 128 x 2 x 2 bytes = 131 KB; Qwen2.5-7B 28 x 4 x 128 x 2 x 2 = 57 KB; Qwen3-14B 40 x 8 x 128 x 2 x 2 = 164 KB.
* Tokens held: prompts reached 1,838 tokens on the 27 Oct replay (section 7b, MEASURED) and the first ~1,000 are shared by every request (the writer's instructions). With prefix caching, 50 sequences hold about 50 x (2,000 - 1,000) + 1,000 = 51,000 tokens; 100 sequences about 101,000. That is, for 50 / 100 sequences: Llama 8B 6.7 / 13.2 GB, Qwen2.5-7B 2.9 / 5.8 GB, Qwen3-14B 8.4 / 16.5 GB of KV cache.
* Usable memory: card memory x `--gpu-memory-utilization`, minus about 1.5 GB for the CUDA context and graphs (GUESS).
* `--max-model-len 4096` everywhere: the Ollama setting is also 4096 and the longest prompt seen is 1,838 plus a reply of at most 110 tokens.
* Memory bandwidth (spec sheets, from memory): T4 320 GB/s, L4 300, A10G 600, A5000 768, RTX 4090 1,008, A100 1,555 (40 GB) / 2,039 (80 GB), H100 3,350. At one request, decoding speed is at most bandwidth / weight size, so a 5.5 GB model on an L4 cannot decode faster than about 55 tokens/s: a 12-token first sentence is then at least 220 ms of decoding before any prompt reading. **Expect an L4 not to meet the 300 ms gate at any load (GUESS)**; a failed gate on a 24 GB card says the card is too small or too slow, not that the model is wrong.

| Memory class | Cards | vLLM flags | Which candidates fit | Status |
|---|---|---|---|---|
| 16 GB | T4 | `--quantization awq --dtype half --max-model-len 4096 --gpu-memory-utilization 0.90 --max-num-seqs 32 --enable-prefix-caching` (a T4 has no bf16: `--dtype half`) | **Qwen2.5-7B-AWQ: fits** (about 5.5 + 1.8 GB KV at 32 seqs). **Llama-3.1-8B-AWQ: tight** (5.7 + 4.3 GB; ok at 32 seqs, not 50). **Qwen3-14B-AWQ: does not fit** (9.5 + 5.4 GB against about 13 GB usable). fp16 7B/8B: no. | GUESS |
| 24 GB | L4, A10G, RTX 4090, A5000 | `--quantization awq --dtype half --max-model-len 4096 --gpu-memory-utilization 0.90 --max-num-seqs 64 --enable-prefix-caching` | **All three in AWQ fit at 50 sequences** (Qwen2.5-7B 5.5 + 2.9; Llama 5.7 + 6.7; Qwen3-14B 9.5 + 8.4 = 17.9 of about 20 GB). At 100 sequences Qwen3-14B-AWQ does not (9.5 + 16.5). **fp16 fits only Qwen2.5-7B** (15.2 + 2.9); Llama 8B fp16 does not at 50 sequences. | GUESS |
| 40 to 80 GB | A100 40/80, H100 80 | `--dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.85 --max-num-seqs 256 --enable-prefix-caching` (the manifest's own numbers) | **40 GB:** Qwen2.5-7B and Llama 8B in bf16 fit at 100 sequences (16 + 13.2 for Llama); Qwen3-14B only as AWQ (`--quantization awq --dtype half`, 9.5 + 16.5). **80 GB:** all three in bf16 fit, Qwen3-14B included (29.5 + 16.5). | GUESS |

When a run confirms a row (the server starts, holds the sequences without evictions, and the probe completes), change its status to MEASURED and write the machine, the vLLM version and the date next to it. When a row is wrong, correct the row, do not just note it.

Run the same quantisation across the candidates when comparing them (AWQ for all three on a 24 GB card), otherwise the comparison measures the quantisation. Say which one each number used.

## 3. The gate, and why it is measured at 25 to 100 at once

The manifest (`deploy/k8s/voice.yaml`) says 10 to 20% of callers wait on the model at any moment. At 500 callers that is **50 to 100 requests in flight in total**. The manifest runs 2 model replicas, so one replica's share is **25 to 50 at once**, and if one replica fails the other holds 50 to 100. The probe therefore runs 1, 2, 4, 8, 16, 25, 32, 50, 64, 100 at once.

Two cautions, both INFERRED:

* **In flight is not a fixed number; it depends on how fast the server answers (Little's law: in flight = request rate x time each takes).** The manifest's 10 to 20% is only true at about 1.5 to 3 s per request. At the 300 ms target the same traffic is about 10 in flight, not 50 to 100. The probe is a closed loop (each of the N threads sends the next request the moment its last one is answered), so at 100 at once it asks for a far higher request rate than 500 callers produce. It is the right way to find where the server breaks and the wrong way to read a latency for the design load by itself. So every level now also reports **requests per second**. The design rate is 500 / 15 s = 33 turns a second (15 s between a caller's turns: `capacity_probe.py --turn-every`, from recorded calls) times 1 to 2 model calls per turn (a decision and a reply; how many turns call the model at all is unknown, code answers many): **at most 33 to 67 requests/s for 500 callers in total, so at most 17 to 33 per replica of two** (GUESS on the calls per turn, an upper bound on the turns).
* The probe's requests share the same facts and instructions and differ only in the caller's line, so the prefix cache hits more than real turns will. The latency it reports is a best case for the cache. Record the server's prefix-cache hit rate from `/metrics` (`grep prefix_cache`; the metric names change between vLLM versions).

**The gate, stated so it can pass or fail:** for the card and model under test, the `full` first-sentence **median is under 300 ms and the p90 under 600 ms (the Phase 2 targets) at the lowest level whose requests per second reaches the per-replica design rate**, checked at 17 and at 33. Report the whole table, and mark the levels where each of these happens: median under 300 ms; p90 under 600 ms; requests/s reaches 17; reaches 33. If none of the levels reaches 17 requests/s, the card cannot carry the design load whatever its latency at 1, and the gate fails on throughput.

## 4. Check the servers before any measurement

```bash
curl -s $MODEL/v1/models                      # the model's name, exactly as a worker must send it
curl -s $MODEL/health
```

**Send one real request and wait for the answer before starting a test.** In the rehearsal the first chat request to a freshly started `mlx_lm.server` took 1 min 42 s (it loads the model on the first request), and `capacity_probe.py`'s first call timed out at its 20 s limit. Then check the constrained-choice request the agent uses for tool decisions (`nlu/openai_compat.py: choose`):

```bash
curl -s $MODEL/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"'$NAME'","messages":[{"role":"user","content":"Pick one."}],"max_tokens":8,"structured_outputs":{"choice":["search_cars","get_car","none"]}}'
```

Expect one of the three words. HTTP 400 is not fatal: the code then asks for free text and stops asking this server (`can_choose`), which costs tokens and time on every decision, so write it down.

## 5. The rental: step by step with expected times

A short rental of a 24 GB card, to run this before the real hardware arrives. **Times are GUESSES** (nothing here has been run on such a machine); they are there to plan the length of the rental, and the first run of a step is expected to take longer than the second. The order puts the gate first and the optional end-to-end test last, so stopping early still leaves the most valuable numbers.

Start the meter only when everything in "before you start" is ready. Stop the instance at the end of step R4 (or when the budget is spent) after copying the results back.

**Before you start (on this laptop, free):**
1. Accept the Llama 3.1 licence on the Hugging Face account if Llama is to be run; have `HF_TOKEN` ready.
2. Choose the card by what the rental offers using section 2. Prefer an RTX 4090 or A10G over an L4 (bandwidth).
3. Have the repository ready to copy **without** secrets or large folders:
   ```bash
   rsync -a --exclude .venv --exclude models --exclude logs --exclude data --exclude .env --exclude .git /Users/fatima/dev/voice-agent/ rental:voice-agent/
   rsync -a /Users/fatima/dev/bmw-portal/site/data.js rental:bmw-portal/site/data.js
   ```
   `.env` is excluded on purpose: it holds this laptop's settings.

| Step | What | Commands | Expected time (GUESS) |
|---|---|---|---|
| R0 | Start the instance, log in, check the card | `nvidia-smi`; `docker --version`; `df -h ~` (60 GB free is enough for three AWQ models and the image) | 5 to 15 min |
| R1 | Copy the repository (above) and install | `cd voice-agent && sudo apt-get install -y portaudio19-dev && curl -LsSf https://astral.sh/uv/install.sh \| sh && uv sync --python 3.12 --extra voice --extra openvino`. If PyAudio will not build, use the package list of `deploy/Dockerfile` instead (checked on the laptop: with `uv pip install . pipecat-ai openvino openvino-genai sherpa-onnx soxr numpy fastapi uvicorn websockets httpx pg8000 redis prometheus-client rapidfuzz psutil`, the probe's imports work, but the versions are not the lock file's). Then `.venv/bin/python scripts/capacity_probe.py --help`. | 8 to 15 min |
| R2 | First candidate: pull the image, start vLLM, warm it, run section 4 | see "Start vLLM" below | 8 to 15 min (image 5 to 10 GB; weights 5 to 10 GB at 100 to 500 MB/s; loading and graph capture 2 to 5 min) |
| R3 | The probe on that candidate, twice | see "The probe" below | 10 to 20 min (about 600 requests per sweep, 1 to 3 requests/s on a small card at the high levels; full and prefix sweeps; two repeats) |
| R2-R3 again | Second candidate, then the third | stop the container, start the next | 20 to 35 min each |
| R4 | Copy results back, stop the instance | see "Copy back" below | 3 to 5 min |
| R5 (optional) | Worker + recogniser on the same card, load test at 1, 8, 16, hearing check | section 6 | 35 to 50 min; not the production shape (the review measured a language model and Whisper on one GPU: 0.9 s became 1.6 s), so use it to prove the chain works, not for numbers |

Expected total: **about 1.5 to 2 hours for two candidates (R0 to R4), 2.5 hours for three**, plus the optional R5. If the budget is short, run only Qwen2.5-7B-AWQ and Llama-3.1-8B-AWQ, and the 14B only if both fail the gate.

**Start vLLM** (the image the manifest pins; `--model` form as in the manifest; confirm the flags against `--help` of that image first, the manifest's own comment says to do so):

```bash
export NAME=Qwen/Qwen2.5-7B-Instruct-AWQ
docker run -d --name vllm --gpus all --ipc=host -p 8000:8000 -v $HOME/hf:/root/.cache/huggingface -e HF_TOKEN \
  vllm/vllm-openai:v0.31.0 --model $NAME --quantization awq --dtype half --max-model-len 4096 \
  --gpu-memory-utilization 0.90 --max-num-seqs 64 --enable-prefix-caching
docker logs -f vllm            # wait for "Application startup complete"; Ctrl-C leaves it running
export MODEL=http://127.0.0.1:8000
```

(No Docker on the rental: `uv pip install vllm` at the version in the manifest tag and `vllm serve $NAME` with the same flags. Different packaging, same server.) Then section 4.

**The probe** (model only; its own recognition and voice timing are skipped; the long sweep does not stop at the first slow level):

```bash
export VA__LLM__BACKENDS__GPU_SERVER__HOST=$MODEL/v1 VA__LLM__BACKENDS__GPU_SERVER__MODEL=$NAME VA__DIALOGUE__DECISION_BACKEND=gpu_server
for run in A B; do
  .venv/bin/python scripts/capacity_probe.py --backend gpu_server --levels 1 2 4 8 16 25 32 50 64 100 --no-stop --model-only \
    --portal-data ../bmw-portal/site/data.js | tee probe_${NAME##*/}_$run.txt
  cp logs/capacity_probe.json probe_${NAME##*/}_$run.json
done
curl -s $MODEL/metrics | grep -i prefix_cache > prefix_cache_${NAME##*/}.txt
```

Each level sends `2 x N` requests, so at 1 and 2 at once the p90 is the slower of two or four samples: read the low levels from the medians and from the repeats. The probe's first call is a warm-up.

**Copy back, then stop the instance:**

```bash
mkdir results && cp probe_* prefix_cache_* results/ && nvidia-smi > results/nvidia-smi.txt && docker inspect vllm --format '{{.Config.Image}}' > results/vllm-image.txt
# from the laptop:  rsync -a rental:voice-agent/results/ ./phase0_results_<card>/
```

Stopping the instance is the last step: until it is stopped, it is billed.

## 6. The worker and the end-to-end numbers (R5, or on the real hardware)

Environment, from the manifest. `VA__DIALOGUE__DECISION_BACKEND` is now set in `deploy/k8s/voice.yaml`, `deploy/docker-compose.yml` and `deploy/Dockerfile` (section 19); it is exported here too because a worker started by hand does not read those files.

```bash
export VA__LLM__BACKENDS__GPU_SERVER__HOST=http://<model-server>:8000/v1
export VA__LLM__BACKENDS__GPU_SERVER__MODEL=<the name from /v1/models>
export VA__VOICE__STT_URL=http://<stt-server>:8000/v1
export VA__VOICE__STT_REMOTE_MODEL=<the STT server's model name>
export VA__VOICE__MODEL_ONLY=false
export VA__DIALOGUE__DECISION_BACKEND=gpu_server
unset VA_DUMP_AUDIO_DIR                               # the audio recorder writes a file per frame on the event loop (review B7)
python -m dealer_core.server --agent --prefix --backend gpu_server --site <bmw-portal/site> --port 8766 --max-calls 16
```

The speech models (Kokoro at least) are not in the repository: copy `models/kokoro-v1.0` to the worker's `models/` (about 350 MB; the other contents were not checked). The recogniser on the GPU: the vLLM Whisper server of the manifest (`openai/whisper-large-v3-turbo`, `--gpu-memory-utilization 0.1` when it shares the card, as `deploy/docker-compose.yml` does).

On the worker's log you want to see `NLU backend active: gpu_server (...)` and `Model speed measured before the first call: <N> ms to a first sentence`: write that number down, it is what the governor acts on.

```bash
for n in 1 8 16; do
  curl -s http://<worker>:8766/health > health_before_$n.json
  .venv/bin/python scripts/load_test.py --url ws://<worker>:8766/voice --origin http://localhost:8080 --levels $n --mix --stagger 2 --voice kokoro:michael
  curl -s http://<worker>:8766/health > health_after_$n.json
  cp logs/load_test.json load_test_$n.json
done
```

`--voice` is a Piper voice name (default `amy`, the method in `docs/SCALING.md`); on a machine without the Piper models use `kokoro:<name>` with a voice other than the agent's (`tts.kokoro_voice`, default `sarah`). On the real hardware run the load generator on a different machine from the worker.

* **Governor level per turn** = the change in `quality.turns` (`full`, `prefix`, `code`) between the two `/health` snapshots (also `voice_turn_level_total{level}` on `/metrics`). Share at full = `full / (full + prefix + code)`. In the laptop rehearsal the counts summed to fewer than the turns (3 and 7 missing at 8 and 16 callers); the cause was not found, so cross-check against the turn count.
* `load_test.py` reports time to the first **sound**, not to the model's first **sentence**. The first sentence comes from the probe at the matching concurrency and from `quality.full_first_ms` in `/health`. Say which of the three a number is.
* The knee: repeat at 4, 12, 20, 24 and 32 callers (the worker's `--max-calls` at least the level; admission may hold `cap` lower, which `/health` shows). The knee is the count where p90 first passes 1 s, and where the share over 2 s passes 10%.

Hearing (the three files of the recorded call, `logs/audio_dump/20261027_215011_9456_in.{pcm,csv,ref.txt,agent.txt}`, are on the laptop only; copy them to the machine that runs the script):

```bash
.venv/bin/python scripts/hearing_check.py 20261027_215011_9456_in --stt-url http://<stt-server>:8000/v1 --stt-model <name>
```

Nine turns: differences of a few points are noise. Rehearsal on the laptop (base, CPU, through the repository's own transcription server): WER 3% with the hint, 10% without; 7 of 9 turns exact.

## 7. What to write down

For every number: the machine (GPU model and count, CPU, RAM, where the load generator ran), the model and quantisation, the vLLM image tag, the date, whether it was a repeat, and MEASURED / INFERRED / GUESS. Where a number was not taken, write "unknown".

| Level (at once) | Full: median / p90 ms | Full: requests/s | Prefix: median / p90 ms | Prefix: requests/s |
|---|---|---|---|---|
| 1, 2, 4, 8, 16, 25, 32, 50, 64, 100 | | | | |

Then: the prefix-cache hit rate; the gate result in one sentence (section 3); for the end-to-end run the first-sound table at 1 / 8 / 16 callers, the governor split, the cap at the end, and the knee.

## 8. Do not

* Do not treat a laptop number as a GPU number or the reverse. The rehearsal numbers are in section 19 under the laptop's name.
* Do not change a threshold (`model_budget_ms`, the deadlines, the governor) during Phase 0. They are re-derived from these numbers in Phase 2.
* Do not leave `VA_DUMP_AUDIO_DIR` set.
* Do not start, resize or leave running any rented hardware without the owner's say-so. Do not put `HF_TOKEN` or any key in a file that is committed or copied back.

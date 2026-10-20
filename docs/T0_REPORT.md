# T0 report: OpenVINO vs Ollama accuracy on the existing labelled sets (Checkpoint A)

Date: 2026-10-20. Data: dev (31) + heldout (40) + fresh (29) = **100 utterances**, all written by me, none from your set yet.
Your utterance set has not arrived, so **T0 will be re-run when it does** (`dealer_core.eval.compare --csv your_set.csv`).

## 0. Verdict

* **Our own OpenVINO int4 conversion of Qwen3-4B-Instruct-2507 matches Ollama's Q4_K_M of the same checkpoint: 0.0 points, 95% CI
  [-5.0, +5.0], intent accuracy 0.95 vs 0.96.** By the agreed rule the interval touches the -5 threshold, so the formal label is
  **INCONCLUSIVE**: proceed with OpenVINO for the voice path, keep Ollama as the configured fallback, re-run when your set grows.
  There is no sign of a drop.
* The OpenVINO org's prebuilt `Qwen3-4B-int4-ov` is **a different checkpoint** (`Qwen/Qwen3-4B`, the original hybrid-thinking model).
  Against Ollama's same-checkpoint build it trails by 5.0 points (CI [-12, +2], inconclusive), and against Ollama's 2507 by 8.0 points
  (CI [-16, 0]). **Do not use the prebuilt build for the product; use our 2507 conversion** (the default config now points at it).
* **No per-call error, timeout or malformed output occurred in any accuracy run** (7 files, 700 calls x 2): see section 3.
* The **no-iGPU fallback is weak**: Qwen3 1.7B with the static prompt scores intent 0.60-0.66 (vs 0.82 with the per-state prompt on the iGPU). That is a
  prompt-style effect, not a runtime effect (section 4).
* **AWQ build: not finished** (running since 14:54). Its row is in the tables as PENDING; nothing is claimed about it.

## 1. What was compared (the same checkpoint on both sides, per your condition 2)

| Label | Runtime and weights | Checkpoint | Prompt | Notes |
|---|---|---|---|---|
| `ollama_2507` | Ollama `qwen3:4b-instruct-2507-q4_K_M` | **Qwen3-4B-Instruct-2507** | per-state | the product baseline |
| `ov_2507_nocal` | OpenVINO int4, **our conversion**, iGPU | **Qwen3-4B-Instruct-2507** | per-state | INT4_SYM, group 128, ratio 1.0, no calibration data |
| `ov_2507_awq` | OpenVINO int4, our conversion, AWQ + scale estimation | Qwen3-4B-Instruct-2507 | per-state | **PENDING** (export running) |
| `ollama_orig` | Ollama `qwen3:4b-q4_K_M` | Qwen3-4B (original, thinking off) | per-state | same checkpoint as the org build |
| `ov_orig` | OpenVINO org `Qwen3-4B-int4-ov`, iGPU | Qwen3-4B (original) | per-state | the org's own AWQ recipe |
| `ollama_1p7`, `ov_1p7_cpu`, `ov_1p7_gpu` | Qwen3 1.7B Q4_K_M / OpenVINO int4 CPU / OpenVINO int4 iGPU | Qwen3-1.7B | static / static / per-state | the fallback tier |

Identical for every row: the production prompt and grammar code (`dealer_core.nlu.prompts`), the full-JSON call (intent + slots), temperature 0 / greedy,
thinking off, the grounding filter, the scorer. "Exact+ground" = intent and every slot right after the deterministic grounding filter.
Because this harness uses the production prompt code, numbers differ slightly from the Phase 0b tables (Ollama 2507: 0.85 here vs 0.82 there).

## 2. Results

Cells are **exact+ground / intent accuracy / intent-only accuracy** (intent-only = the production call that emits just the intent).

| Config | dev (31) | heldout (40) | fresh (29) | ALL 100 |
|---|---|---|---|---|
| `ollama_2507` | .84 / 1.00 / .97 | .85 / .93 / .93 | .86 / .97 / .93 | **.85 / .96 / .94** |
| `ov_2507_nocal` | .84 / 1.00 / 1.00 | .88 / .93 / .93 | .83 / .93 / .93 | **.85 / .95 / .95** |
| `ov_2507_awq` | PENDING | PENDING | PENDING | PENDING |
| `ollama_orig` | .84 / .97 / .97 | .82 / .95 / .93 | .79 / .90 / .90 | .82 / .94 / .93 |
| `ov_orig` (org build) | .77 / .94 / .90 | .75 / .85 / .85 | .79 / .93 / .93 | .77 / .90 / .89 |
| `ollama_1p7` (static) | .55 / .65 / .65 | .55 / .70 / .70 | .48 / .62 / .59 | .53 / .66 / .65 |
| `ov_1p7_cpu` (static) | .58 / .65 / .65 | .40 / .57 / .57 | .45 / .59 / .59 | .47 / .60 / .60 |
| `ov_1p7_gpu` (per-state) | .77 / .94 / .94 | .55 / .75 / .75 | .59 / .79 / .79 | .63 / .82 / .82 |

The production intent-only call agrees with the full-JSON intent 98-100% of the time on every config, so the intent accuracy above is what the
live path gets.

### Paired comparisons (same 100 utterances; 95% paired-bootstrap interval; exact McNemar)

| Candidate vs baseline | exact+ground diff | 95% CI | McNemar (cand-only / base-only), p | Rule verdict |
|---|---|---|---|---|
| **`ov_2507_nocal` vs `ollama_2507`** (same checkpoint: the product pair) | **+0.0** | [-5.0, +5.0] | 3 / 3, p=1.00 | INCONCLUSIVE (touches -5): proceed, keep fallback, re-run |
| `ov_2507_awq` vs `ollama_2507` | PENDING | | | |
| `ov_orig` vs `ollama_orig` (same checkpoint: the org build) | -5.0 | [-12.0, +2.0] | 4 / 9, p=0.27 | INCONCLUSIVE |
| `ov_2507_nocal` vs `ov_orig` (checkpoint effect on OpenVINO) | +8.0 | [0.0, +16.0] | 13 / 5, p=0.10 | no drop |
| `ov_orig` vs `ollama_2507` (cross-checkpoint, information only) | -8.0 | [-16.0, 0.0] | 4 / 12, p=0.08 | INCONCLUSIVE |
| `ov_1p7_cpu` vs `ollama_1p7` (fallback, both static) | -6.0 | [-13.0, +1.0] | 4 / 10, p=0.18 | INCONCLUSIVE |
| `ov_1p7_gpu` (per-state) vs `ollama_1p7` (static) | +10.0 | [-1.0, +21.0] | 21 / 11, p=0.11 | no drop (confounded: different prompt style) |

**Statistical power (estimate).** About 10% of utterances are discordant between two systems here, so the paired interval is roughly +/-1.96 x sqrt(0.10/n)
points: about +/-7 at n=100 and about **+/-4.4 at n=200** (existing sets + your first ~100). A 5-point question needs ~150-300 utterances to resolve.
Nothing above is "significant"; the McNemar p-values are all above 0.05.

## 3. No timeouts, errors or fallbacks fired (your question 1)

All seven cached prediction files (`benchmarks/results/t0/*.json`), counted from the stored outputs:

| File | utterances | full-JSON errors | intent-only errors | malformed JSON | calls over 25 s |
|---|---|---|---|---|---|
| ollama_2507, ollama_orig, ollama_1p7 | 100 each | 0 | 0 | 0 | 0 |
| ov_2507_nocal, ov_orig, ov_1p7_cpu, ov_1p7_gpu | 100 each | 0 | 0 | 0 | 0 |

The comparison harness calls each backend directly with a 30 s per-call timeout and records any failure as an error row, so a timeout would have been
counted above. It does not use the fallback chain, so a silent fallback to another backend cannot have happened. Nothing needs re-running.
(The older Phase 0b `bench_llm` runs are a different harness; they recorded 100% valid JSON.)

## 4. The fallback tier and the prompt style (new finding)

On the **same OpenVINO int4 weights**, the per-state prompt on the iGPU scores intent **0.82** and the static prompt on CPU scores **0.60**: a 22-point
intent gap and a 16-point exact+ground gap. Ollama with the static prompt behaves like OpenVINO with the static prompt (0.66), so the cause is the
static prompt (all 19 intents listed, state only enforced by the grammar), which small models handle poorly. The 4B model does not suffer from it
(Phase 0b: static vs per-state within a few points). Consequences for the plan:

* the `openvino_cpu` fallback (1.7B, static prompt) is **usable only behind the rule fast path**, and its LLM-handled turns will often be misread;
  the state-changing intents then go through the read-back confirmation anyway;
* options (not built, not measured): a hybrid prompt with a small static prefix and the state's intent list appended in the user message (more
  prefill, less accuracy loss), or Ollama 4B as the no-iGPU fallback (accurate but 2.6 s on a cache hit, 6 s on a miss). I recommend deciding this
  when your utterance set can measure it, not before.

## 5. The conversion (your questions 2, 3 and 4)

**Recipe** (`scripts/convert_qwen3_2507_ov.ps1`, reproducible): `optimum-cli export openvino -m Qwen/Qwen3-4B-Instruct-2507 --weight-format int4 --sym
--group-size 128 --ratio 1.0` (data-free = `nocal`); the AWQ variant adds `--awq --scale-estimation --dataset wikitext2`, the recipe the OpenVINO org documents
for its own builds.

**Calibration data (question 2).** Only the AWQ build uses any. `--dataset wikitext2` resolves in the installed `optimum-intel` to
`Salesforce/wikitext`, config `wikitext-2-raw-v1`, split `train`: it reads the first 1,024 rows (about 299,000 characters of English Wikipedia articles: the first
is "Valkyria Chronicles III") and draws **128 windows of 32 tokens = 4,096 tokens** from that text. It contains no dealership, car, price or booking text and none of
your data. I checked it against every set: **0 exact utterances and 0 five-word matches in dev, heldout, fresh, the 37-clip script, and (read-only, for this
check only, no tuning) the held-out spontaneous transcripts**; 3-4-word overlaps are common phrases only. The check is `scratchpad/check_calib.py` logic and should be
re-run against your CSV when it exists, since the AWQ build is already using fixed text. The data-free build uses no calibration data at all.

**Environment isolation (question 3).** The tooling is in an isolated environment, **`C:\dev\ov-convert\.venv`, not the project environment**:
`optimum 2.3.0`, `optimum-intel 2.2.0`, `nncf 3.4.0`, `transformers 5.5.4`, `torch 2.14.1`, `datasets 5.0.1`. The project environment contains none of
`torch / transformers / optimum / nncf / datasets` (checked with `uv pip list`); it has only the runtime pieces (`openvino`, `openvino-genai`,
`openvino-tokenizers`, `onnxruntime`, `sherpa-onnx(-core)`). Nothing needed moving. **Coexistence re-check: OpenVINO, onnxruntime and sherpa-onnx load and
run together in one process in all 12 import orderings** (real smoke tests: OpenVINO compiles a model, onnxruntime opens a session, sherpa synthesises a word;
`scripts/check_runtime_coexistence.py`). That check also showed that my earlier "never mix sherpa and onnxruntime in one process" claim was wrong; it is corrected
in the Phase 0b report, README, plan and code comments.

**Cost (measured).** Install of the isolated environment 4.5 min (1.2 GB); data-free export 7.9 min including the ~8 GB checkpoint download (kept in the
global Hugging Face cache: **7.9 GB, safe to delete after conversion**); output 2.2 GB (same as the org build). AWQ export: still running after ~15 min,
11.9 GB of RAM in use (31 GB machine); its duration will be recorded in `logs/convert_timings.txt`.

**Licenses (question 4).** Converted weights: Apache-2.0 (derivative of an Apache-2.0 checkpoint, produced and kept locally). All conversion tools: Apache-2.0
(torch: an SPDX expression of Apache-2.0 and BSD-2-Clause parts). The calibration text, WikiText, is **CC-BY-SA-3.0 / GFDL**; it only fed quantization statistics, the
model was not trained on it, and I have not sought a legal view. Details in `docs/LICENSES.md`.

## 6. Decision and next steps

* Default `openvino_gpu` now points at `Qwen3-4B-Instruct-2507-int4-ov-nocal`. Ollama stays the configured fallback and the text-core/website backend.
* When the AWQ build finishes, compare it with:
  `uv run python -m dealer_core.eval.compare --spec ov_2507_awq=openvino:Qwen3-4B-Instruct-2507-int4-ov:GPU:state --spec ov_2507_nocal=openvino:Qwen3-4B-Instruct-2507-int4-ov-nocal:GPU:state --spec ollama_2507=ollama:qwen3:4b-instruct-2507-q4_K_M::state --pair ov_2507_awq:ollama_2507 --pair ov_2507_awq:ov_2507_nocal`
  (cached predictions are reused; only the AWQ build is run). Keep AWQ only if it is better by more than the noise; otherwise the data-free build is simpler.
* Re-run all pairs when your utterance set arrives (~+4.4 points of precision at n=200).
* Latency of the converted model has **not** been measured separately (same architecture as the org 4B build, so I expect the same speed); the Phase 0b
  numbers for the org build are the reference until it is.

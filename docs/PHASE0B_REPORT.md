# Phase 0b report: latency + accuracy spike (2026-10-20)

Everything here was measured on this laptop. Nothing is extrapolated unless it says "estimate".
Raw data: `benchmarks/results/*.json` (gitignored); tables: `benchmarks/results/llm_tables*.md`.
Reproduce: `scripts/download_models.ps1`, `scripts/pull_models.ps1`, `scripts/run_llm_suite.ps1`,
`scripts/run_remaining.ps1` (these refuse to run on battery).

## 0. Headline

1. **Sub-second first audio is achievable by design; a sub-second first *meaningful* word is achievable
   for ~83% of turns** (rules answer them in <1 ms). LLM turns land at about **1.3-1.9 s**. Not yet measured
   end to end: VAD/turn-detection latency (Phase 3), so the 300 ms endpoint figure below is an assumption.
2. **The iGPU matters more than the runtime.** CORRECTION: my first version of this report said OpenVINO reads a
   293-token prompt in 94 ms against 1.8 s for Ollama. That was wrong: OpenVINO GenAI silently reuses its KV cache across
   `generate()` calls, so my repeated-prompt test measured cache *hits*. With a guaranteed-unique prefix
   (`benchmarks/diag_ov_cache.py`): OpenVINO **CPU** misses cost 1172 ms (1.7B) and 4720 ms (4B), about the same as
   Ollama; the **iGPU** misses cost only **89 ms (1.7B) and 158 ms (4B)**. The win is the iGPU's prefill speed (plus
   prompt-prefix reuse on CPU), not the runtime. Token *generation* (15-40 tok/s) is the hardware limit, so the LLM must
   emit very few tokens: an **intent-only constrained call (~6 tokens)**, slots filled by code.
3. **Qwen3 1.7B is not accurate enough on its own** (intent 0.72-0.76 on unseen phrasing); **Qwen3 4B is**
   (0.90-0.93). With rules answering ~83% of turns, the LLM only sees the hard 17%.
4. **Licensing found real blockers**: the popular Piper "lessac" voice is research-only; Piper's PyPI package and
   Kokoro's phonemizer are GPL-3. See `docs/LICENSES.md`.
5. **Power state is a hard requirement**: on battery, sustained speed fell to **29%** of burst. All numbers below
   are AC power.

## 1. Test conditions (item 5)

| | |
|---|---|
| CPU | Intel Core Ultra 7 255H, 16 cores / 16 threads (hybrid P/E/LP-E), base 2.0 GHz |
| RAM | 30.9 GB |
| iGPU / NPU | Intel Arc 140T (15 GB shared) / Intel AI Boost NPU (OpenVINO arch 3720) |
| OS | Windows 11 Enterprise 10.0.26200 |
| Power | Scheme reported by `powercfg`: **Balanced**. You set the Windows power *mode* to Best performance; I could not read that setting back (unverified). **AC connected** for every number in this report (BatteryStatus recorded in each result file; the runs taken after the charger came off were discarded and redone). |
| Throttling | **Observed on battery**: sustained 147-176 GFLOP/s vs 270-490 burst, 25-104 of 150 s below half of burst speed (ratio 0.29-0.30). **On AC (clean test)**: ratio 0.78 at 11:06, then **1.2 at 13:19, nothing below half of burst**. Temperatures 56-83 C. The first AC test I ran was badly designed (BLAS x Python thread oversubscription) and is not used. |
| Run-to-run noise | Same configuration, different minutes: Kokoro 487 vs 844 ms; the LLM "solo" baseline in the concurrency harness 740 vs 1405 ms. **Treat every figure as +/-30-50%**, and the slowdown *ratios* in section 6 as less reliable than the absolute numbers. |
| Background load | 5-15% from the Claude desktop app while idle. |
| Ollama | 0.40.5, defaults (iGPU deliberately disabled by Ollama; `OLLAMA_IGPU_ENABLE=1` **not tried**). The server died once overnight and was restarted. |

## 2. STT (item: speed AND accuracy)

37 SYNTHETIC clips (Piper voice reading `script.tsv`): clean and unaccented, so **optimistic**. Real WER needs your
37 recordings (`docs/RECORDING_GUIDE.md`). Latency is for a 3.2 s clip: streaming = time after the last chunk;
offline = decode after endpoint. "terms" = car makes/models/places/names recognised (of 37 expected).

| Engine | Latency | WER | Numbers exact | Terms |
|---|---|---|---|---|
| sherpa streaming zipformer 20M | 12 ms | 0.577 | 10/17 | 5/37 |
| sherpa streaming zipformer 2023-06-26 | 79 ms | 0.230 | 13/17 | 6/37 |
| + hotwords (modified beam search) | 56 ms | 0.212 | 14/17 | 10/37 |
| Moonshine tiny | 598 ms | 0.137 | 16/17 | 21/37 |
| Moonshine base | 891 ms | 0.121 | 17/17 | 24/37 |
| faster-whisper tiny.en int8 | 360 ms | 0.176 | 14/17 | 20/37 |
| + hotwords | 307 ms | 0.144 | 14/17 | 30/37 |
| faster-whisper base.en int8 + hotwords | 771 ms | 0.119 | 13/17 | 32/37 |
| **OpenVINO Whisper tiny int8 (multilingual), iGPU, + prompt** | **58 ms** | **0.130** | 13/17 | **34/37** |
| OpenVINO Whisper tiny, CPU, + prompt | 92 ms | 0.139 | 13/17 | 33/37 |
| OpenVINO Whisper base, iGPU, + prompt | 91 ms | 0.134 | 14/17 | 33/37 |
| OpenVINO Whisper base, NPU (no prompt: prompt fails to compile) | 138 ms | 0.223 | 13/17 | 18/37 |

Findings: biasing the decoder with car terms is the single biggest accuracy lever (terms 20 -> 30 for
faster-whisper, 18 -> 33 for OpenVINO). The streaming model is fastest but mishears brands ("TOYODE ALLIANCE
CRUISER"); it is **not** usable unless you accept fuzzy-matching against inventory. OpenVINO Whisper is 4-8x faster than
faster-whisper at similar accuracy. The NPU works for Whisper but is no faster than the iGPU and cannot take a prompt.
Only multilingual OpenVINO Whisper builds exist (no `.en`), which is fine for English-only but worth re-checking on
real, accented audio.

## 3. TTS (time to first audio, one sentence at a time)

| Engine / voice | Short phrase (7 words) | 15 words | License of training data |
|---|---|---|---|
| Piper libritts_r (sherpa-onnx, fp32) | **56 ms** | 143 ms | CC BY 4.0 (attribution) |
| Piper joe | 60 ms | 157 ms | CC0 |
| Piper ljspeech | 89 ms | 183 ms | public domain |
| Piper alba (en-GB) | 86 ms | 185 ms | CC BY 4.0 |
| Piper lessac | 71 ms | 183 ms | **research-only: do not ship** |
| Kokoro fp32 (kokoro-onnx) | 487-844 ms | 1.3-2.6 s | Apache-2.0, but GPL phonemizer |
| Piper / Kokoro **int8** | 4-14x **slower** than fp32 (this CPU has no fast int8 path in ORT) | | |

**Pick Piper fp32 through sherpa-onnx; drop Kokoro** (cannot meet first-audio on this CPU: hardware limit
for Kokoro, not a bug). Piper TTFA scales with the first sentence, so the first clause must be short. Pre-rendered
fixed phrases play with no synthesis at all.

## 4. LLM (item 1)

### 4a. Speed: Ollama Q4, per-state prompt (293 prompt tokens), median of 2 passes, AC

"miss" = prefix not cached (state changed); "hit" = prefix cached. TTFT = server load + prefill + first token
(Ollama delivers constrained JSON in one burst, so client-side TTFT would just equal total time). Output is the full
JSON (~22 tokens). **Total NLU time per turn is shown for both cases.**

| Model | Prefill miss | Prefill hit | TTFT miss / hit | Decode tok/s | Total miss / hit |
|---|---|---|---|---|---|
| qwen3:0.6b | 776 ms | 92 ms | 822 / 133 ms | 45.6 | 1578 / 840 ms |
| qwen3:1.7b | 1764 | 186 | 1808 / 229 | 33.5 | 2414 / 821 |
| qwen3:4b-instruct-2507 | 4861 | 546 | 4954 / 671 | 11.4 | 6445 / 2618 |
| qwen3.5:0.8b | 1749 | **1543** | 1795 / 1585 | 39.0 | 3001 / 2722 |
| qwen3.5:2b | 4194 | **4617** | 4265 / 4695 | 21.3 | 6538 / 6094 |
| ministral-3:3b | 3557 | 283 | 3625 / 353 | 17.6 | 4852 / 1931 |
| llama3.2:1b (benchmark-only licence) | 2911 | 230 | 2965 / 296 | 25.5 | 3518 / 1042 |
| llama3.2:3b (benchmark-only) | 2870 | 218 | 2931 / 282 | 20.5 | 3775 / 1206 |
| gemma3:4b (benchmark-only) | 4578 | **4330** | 4681 / 4412 | 17.5 | 6656 / 5653 |

One static prompt (always a cache hit) with the state enforced only by the JSON-schema grammar ("cached" mode),
Ollama, 391-token prompt: 4B 2055 ms, 1.7B 1313 ms, 0.6B 1145 ms per turn on a hit. Thread count: 10-12 was best
for the 1.7B (34 tok/s) and 6-12 for the 4B (16 tok/s); 8 and 16 were worse. The curve is noisy, not monotonic.

### 4b. Accuracy (Ollama Q4, state-restricted grammar, temperature 0)

JSON validity is 100% for every model because the grammar enforces it. "Exact" = intent AND every slot right.
"+ground" = after the deterministic filter that drops slot values not present in the utterance (small models invent
slots such as `"offer_amount": 0`). **dev (31) is the set I developed against and is flattering. heldout (40) and
fresh (29) are the honest columns.** Per-state prompt, corrected menu (`ask_car_info` only when a car is in focus):

| Model | Intent dev / held / fresh | Exact+ground dev / held / fresh | All 100 | Turn p50 / p95 |
|---|---|---|---|---|
| qwen3:0.6b | 0.58 / 0.45 / 0.41 | 0.32 / 0.30 / 0.21 | 0.28 | 537 / 881 ms |
| qwen3:1.7b | 0.87 / 0.72 / 0.76 | 0.81 / 0.60 / 0.59 | 0.66 | 1077 / 2826 |
| **qwen3:4b-instruct-2507** | **0.97 / 0.93 / 0.93** | **0.81 / 0.82 / 0.83** | **0.82** | 1833 / 4391 |
| ministral-3:3b | 0.94 / 0.93 / 0.86 | 0.71 / 0.78 / 0.62 | 0.71 | 1765 / 3921 |
| qwen3.5:2b* | 0.97 / 0.85 / 0.83 | 0.71 / 0.72 / 0.66 | 0.70 | 5158 / 8410 |
| llama3.2:3b* | 0.90 / 0.85 / 0.83 | 0.65 / 0.70 / 0.66 | 0.67 | 1459 / 3354 |
| gemma3:4b* | 0.90 / 0.95 / 0.86 | 0.74 / 0.85 / 0.79 | 0.80 | 5134 / 8055 |

\* run before I fixed the menu, so slightly pessimistic. qwen3.5:0.8b and llama3.2:1b were only run on dev
(intent 0.77 and 0.58) and dropped. "Turn p50/p95" is the whole call with the state changing between turns (cache
misses included).

Static prompt + grammar-only state, intent / exact+ground, held / fresh: 4B 0.90 / 0.86 and 0.88 / 0.79;
1.7B 0.80 / 0.72 and 0.75 / 0.59; ministral 0.78 / 0.86 and 0.75 / 0.69.

**Prompt design finding.** The 1.7B swung from 0.97 to 0.77 (dev) just by changing the intent menu. A narrower menu
restores 0.97 on dev but only **0.75-0.76 on unseen phrasing**, so the dev score was mostly fit to my own wording. The
4B barely moves. Small models are brittle to menu changes: another reason to keep the LLM's job tiny.

## 5. Why Qwen3.5 0.8B "misses the cache" (item 2): the model, not your setup

Controlled test (`benchmarks/diag_cache.py`): identical code and options for every model; "prefill later" is the
median prompt-evaluation time of repeat calls after the first.

| Variant (what changed) | qwen3:1.7b control | qwen3.5:0.8b | qwen3.5:2b |
|---|---|---|---|
| A. identical request repeated | 27 ms | **55 ms** | **147 ms** |
| B. same system prompt, different user text (my benchmark) | 118 ms | **1707 ms** | **5155 ms** |
| C. B without JSON-schema grammar | 122 | 2283 | 3570 |
| D. B without the `think` parameter | 95 | 2596 | 4080 |
| E. B with a ~900-token shared prefix | 333 | 3044 | 6555 |
| F. B with a different `num_predict` each call | 127 | 1360 | 4263 |

How I know it is not my setup: (1) the control model hits the cache in every variant under the same code;
(2) removing the grammar, removing `think`, and varying `num_predict` change nothing for Qwen3.5 (C/D/F);
(3) Qwen3.5 *does* hit the cache when the whole request is identical (A), so slots and keep-alive are fine.
Mechanism, from the server log: Qwen3.5 is a hybrid with recurrent (`qwen35.ssm.*`) layers in 3 of every 4 blocks
(`full_attention_interval = 4`). Its state cannot be rolled back to an arbitrary token, so llama.cpp saves one
"context checkpoint" at the END of each prompt (`n_tokens = 280`). The next request diverges at token 262 (where
the customer text starts), and the log shows `checking checkpoint with [279,279] against 262... erased invalidated
context checkpoint`, then re-evaluates from token 0. Gemma 3 behaves the same way (prefill hit ~ miss), probably
because of sliding-window attention (not diagnosed). A possible workaround (a raw-prefix priming request) is
**untested**; Qwen3.5 was not more accurate or faster than Qwen3, so I did not pursue it.

## 6. Per-turn NLU budget and what fits (item 3)

Proposed budget, measured from the moment the caller stops talking:

| Stage | Target p50 | Target p95 | Note |
|---|---|---|---|
| Endpointing (VAD + turn model) | 300 ms | 500 ms | **assumption, not measured** (Phase 3) |
| STT final | 150 | 250 | |
| NLU: rules | 5 | 10 | |
| NLU: LLM fallback | **600** | **1000** | only ~17% of turns |
| Policy + SQLite tools | 20 | 50 | not built yet |
| TTS first audio | 200 | 300 | cached clips: ~0 |
| **First audio (ack clip)** | **<=0.4 s** | **<=0.6 s** | independent of STT/LLM |
| **First meaningful word, rule turn** | ~0.7 s | ~1.1 s | |
| **First meaningful word, LLM turn** | ~1.3 s | ~2.0 s | inside your 1.5-3 s allowance |

**All three stages running flat out at once** (a deliberate worst case; real turns are mostly sequential),
p50 / p95 in ms per stage, all AC:

| STT | LLM | TTS | STT | LLM (NLU call) | TTS 1st audio | Fits LLM budget? |
|---|---|---|---|---|---|---|
| sherpa streaming + hotwords | Ollama qwen3:1.7b, full JSON | Piper | 85 / 191 | **3035 / 5285** | 745 / 1912 | no |
| faster-whisper tiny.en + hot | Ollama qwen3:1.7b, full JSON | Piper | 703 / 864 | 1233 / 1708 | 557 / 647 | no |
| faster-whisper tiny.en + hot | Ollama qwen3:4b, full JSON | Piper | 834 / 1280 | 2174 / 2801 | 638 / 773 | no |
| OV Whisper tiny CPU + prompt | OV Qwen3 1.7B **CPU**, intent-only | Piper | 270 / 326 | 948 / 1214 | 364 / 451 | marginal |
| OV Whisper tiny CPU + prompt | OV Qwen3 **4B iGPU**, intent-only | Piper | 274 / 297 | **891 / 1072** | 430 / 502 | **marginal, most accurate** |
| OV Whisper base iGPU + prompt | OV Qwen3 1.7B CPU, intent-only | Piper | 187 / 237 | 821 / 969 | 523 / 616 | marginal |
| OV Whisper tiny iGPU + prompt | OV Qwen3 4B iGPU, intent-only | Piper | 175 / 215 | 1584 / 2111 | 379 / 431 | no (STT and LLM fight over the iGPU) |

Reading it: Ollama-based stacks do not fit. The OpenVINO stacks degrade only 1.2-1.9x when everything runs at once,
and put the NLU call at ~0.8-1.1 s: slightly over my 600 ms p50 target when everything is saturated, but the LLM
normally runs alone during its turn (standalone intent-only: 1.7B CPU 325 ms, 4B iGPU 762 ms; harness "solo" 460-740 ms).
Pinning `INFERENCE_NUM_THREADS=8` on the CPU made OpenVINO ~1.5x slower (hybrid cores); leave OpenVINO's defaults.
Do not put STT and the LLM on the same iGPU.

### Where a target can't be met, and why
* Sub-second *meaningful word on LLM turns*: **a hardware limit** on decoding (15-40 tok/s on this CPU/iGPU), only
  reachable by emitting ~6 tokens. A free-form 25-token reply costs 0.7-1.7 s by itself, so such replies belong in
  the 1.5-3 s allowance, not the 1 s one.
* Cache-miss prefill on the CPU (1.2-4.9 s for a 285-token prompt, in Ollama and OpenVINO alike): **a hardware limit of
  CPU prefill**, avoided by (a) running the LLM on the iGPU, where a miss costs 89-158 ms, or (b) on CPU keeping ONE static
  system prompt so the prefix is always cached.
* Kokoro under 1 s: hardware limit.

## 7. Deterministic fast path (item 4)

Prototype: `benchmarks/fastpath.py` (rules + make/model gazetteer with fuzzy matching + number/date/phone parsers),
state-gated, returns `None` when unsure. Evaluated three ways:

| Set | n | Rules answer | Right when they answer | Median / p95 |
|---|---|---|---|---|
| dev (rules written against it) | 31 | 97% | 29/30 | 0.4 / 1.6 ms |
| heldout (I tuned on its misses, so contaminated) | 40 | 95% | 38/38 | 0.3 / 2.1 ms |
| **fresh (written after tuning, run once, not tuned)** | 29 | **83%** | **22/24 = 92%** | 0.2 / 0.8 ms |

Fresh-set failures, left unfixed on purpose: "tell me a joke" tripped the angry keyword `joke` (false escalation);
a bare "Yousef Hassan 0551112233" got the phone but not the name (safe; the agent just asks). Not covered by rules:
open off-topic chatter, rare phrasings of yes ("ok that's acceptable"), spelled-out dates. Same author wrote rules
and test sets, so this is optimistic against real callers.

### 7a. Precision of the STATE-CHANGING intents (your condition 5; `benchmarks/bench_precision.py`, rules unchanged)

| Group | Intents | Rule firings (all 100 labelled) | Correct | 95% lower bound |
|---|---|---|---|---|
| booking | book_test_drive, give_slot, give_contact, reschedule, confirm_yes/no | 25 | 25 | 34-61% per intent |
| price | make_offer, ask_lowest_price | 11 | 11 | 44-68% |
| cancel | cancel_booking | 2 | 2 | 34% |
| escalate | request_human, angry, claim_authority | 11 | 10 (**angry 4/5 = 80%**; fresh set 1/2) | 38-51% |
| **all state-changing** | | **49** | **48 = 98.0%** | **89.3%** |

* **A 98% claim cannot be demonstrated on this data.** Even with zero errors you need n >= 189 firings *per intent* for a
  95% lower bound of 98%. The 100 labelled examples give n of 2-8 per intent. Only your recorded + typed set can open the gate.
* **Out-of-scope probes (32 adversarial utterances, written before any result, each placed in a state where the trigger
  intent is valid): 20 triggered a rule, 14 of them a state-changing intent.** Examples: "cancel my Netflix subscription" ->
  cancel_booking; "move the sofa to the left" -> reschedule; "today is my birthday" -> give_slot; "tell me a joke about
  cars" and "I hate waiting in traffic" -> angry; "how low can the temperature go" -> ask_lowest_price. The probes are
  deliberately keyword-hostile, so real callers will trigger far fewer, but the rules are plainly keyword-driven.
* **Decision implied by your gate:** state-changing intents defer to the LLM until each passes the lower-bound test. Cost,
  measured on the fresh set: the share of turns needing the LLM rises from **17% to 62%**; mean T0->T2 goes from ~400 ms to
  ~730 ms (primary) or ~630 ms (CPU fallback), and the typical turn becomes an LLM turn (~1.0 s / 0.8 s). Harmless intents
  (search, policy, car info, send details, greeting, goodbye) keep the fast path. Mitigation planned, not built: anchored
  whole-utterance patterns per intent so each can pass the gate on your data.
* The "tell me a joke" false escalation is a design finding, not a one-line bug: escalation rules need (a) anchoring,
  (b) a domain / out-of-scope check before any escalation, (c) the LLM's `out_of_scope` as a veto. Left unfixed on purpose
  so the fresh set stays clean.

Hybrid (rules first, LLM for the rest), exact intent+slots, fresh set only (the other two are contaminated):

| Setup | LLM-only | Hybrid | Share to LLM |
|---|---|---|---|
| Ollama qwen3:1.7b, per-state prompt | 0.59 | 0.79 | 17% |
| Ollama qwen3:1.7b, static prompt | 0.59 | 0.93 | 17% |
| Ollama qwen3:4b, static prompt | 0.79 | 0.93 | 17% |

n=29, so the 0.79 vs 0.93 gap is ~4 utterances: do not over-read it. The ceiling here is ~0.93 because of the two
rule errors. What should be rules: yes/no/confirm, goodbye/greeting, policy-topic keywords, numbers, money, km,
years, dates and times, phone, names after "my name is", make/model/body-type (built from the inventory table),
human-handoff and anger keywords. What stays LLM: `search_cars` vs `ask_car_info` in ambiguous phrasing, hedged
yes/no, off-topic, authority claims with unusual wording.

## 8. OpenVINO CPU / iGPU / NPU (the brief said "don't assume it helps": the iGPU does, the NPU does not)

**Read the TTFT rows below as cache-HIT numbers** (same system prompt repeated). Cache MISS (unique prefix), measured
separately, 285-token prompt, intent-only call:

| Model | CPU hit -> miss | iGPU hit -> miss |
|---|---|---|
| Qwen3 1.7B int4 | TTFT 102 -> **1172 ms** (total 362 -> 1453) | 60 -> **89 ms** (total 299 -> 322) |
| Qwen3 4B int4 | TTFT 608 -> **4720 ms** (total 1633 -> 5527) | 94 -> **158 ms** (total 695 -> 752) |


| Workload | CPU | iGPU (Arc 140T) | NPU |
|---|---|---|---|
| Qwen3 0.6B int4, 285-token prompt, TTFT / tok/s | 61 ms / 61 | 45 ms / 86 | **fails to compile** |
| Qwen3 1.7B int4, TTFT / tok/s | 94 ms / 41 | 57 ms / 45 | fails |
| Qwen3 4B int4, TTFT / tok/s | 225 ms / 21 | 124 ms / 15-19 | not tried |
| Qwen3 intent-only call, total (0.6B / 1.7B / 4B) | 180 / 325 / 1093 ms | 203 / 388 / **762** ms | n/a |
| JSON-schema constrained decoding | works; adds 0-240 ms to TTFT (grammar setup), decode speed unchanged | works; adds ~40-100 ms TTFT | n/a |
| Whisper tiny int8, 3.2 s clip | 78 ms | 68 ms | 79 ms |

NPU: the pre-converted int4 LLMs fail with "duplicated names" at compile (it needs channel-wise `-cw` builds, which
that org publishes only for 8B). The LLM numbers are *speed only*: the OpenVINO int4 builds have **not** yet been run
through the accuracy sets (expected similar to Ollama Q4_K_M; verify in Phase 1).
OpenVINO adds a runtime outside "Ollama or llama.cpp". It is Apache-2.0 and CPU-only compatible, but it is a decision for you.

## 9. Recommended stack (CORRECTED per your condition 1: STT and the LLM never share the iGPU)

| Role | Config name | STT | NLU LLM (intent-only, grammar-constrained) | TTS | Prompt style |
|---|---|---|---|---|---|
| **Primary** | `openvino_gpu` | OpenVINO Whisper tiny int8 on **CPU**, car-term prompt | **Qwen3 4B-Instruct-2507 int4 on the iGPU** | Piper fp32, **separate process** | per-state (a miss costs only ~160 ms on the iGPU) |
| iGPU, faster | `openvino_gpu_small` | same | Qwen3 1.7B int4, iGPU | same | per-state |
| **No-iGPU fallback** | `openvino_cpu` | OpenVINO Whisper tiny, CPU | **Qwen3 1.7B int4, CPU** | same | **one static prompt** (state enforced by the grammar only): a CPU cache miss costs 1.2 s |
| Ollama fallback | `ollama` | faster-whisper tiny.en + hotwords (CPU) | Qwen3 4B or 1.7B Q4_K_M via Ollama | same | static prompt |

Voice: Piper `libritts_r` (CC BY 4.0) or `joe` (CC0). Not used: Kokoro, Qwen3.5, Gemma 3, Llama 3.2, qwen3:0.6b, int8 TTS,
the sherpa streaming STT. Rules answer first; the LLM is called only when they decline; every reply is templated except
open questions.

### 9a. Realistic SEQUENTIAL turns (STT finishes, then NLU, then TTS)
`benchmarks/bench_turns.py`: 69 scripted utterances (heldout + fresh, spoken by a different Piper voice, transcribed by the
real STT) + 7 open questions. T0 = caller audio complete; **endpointing/VAD is not included** (unmeasured; assume ~300 ms
more). T2 = first TTS audio chunk **of the reply, i.e. the first meaningful word (not the ack clip)**; the ack clip is
pre-rendered, so its first audio is about "endpointing" alone (audio device start-up not measured). TTS runs in its own process behind a pipe (round trip under 0.1 ms). Three interleaved
repetitions per config, on AC. Each cell is the **median of the three per-run p50s, with the min-max across runs**.

| Config | Rule turn T0->T2 p50 (run p95s) | LLM turn T0->T2 p50 (run p95s) | NLU alone, LLM turn | Open answer, first sentence, T0->T2 p50 (run p95s) |
|---|---|---|---|---|
| **Primary** (4B iGPU) | **277 ms** (269-416); p95 401-551 | **1012 ms** (986-1146); p95 1112-1522 | 685-752 ms | 1463 ms (1461-1774); p95 1597-2338 |
| iGPU, faster (1.7B iGPU) | 320 (260-326); p95 318-545 | 716 (572-716); p95 630-881 | 318-379 | 953 (943-1400); p95 1060-1521 |
| CPU fallback (1.7B CPU, static prompt) | 310 (278-412); p95 320-697 | 825 (766-1032); **p95 1343-2154** | 480-616 | 1139 (1058-1484); p95 1215-1609 |

Rules answered 83% of the 69 scripted turns from real STT transcripts (91% from the gold text, so transcription errors cost
about 8 points of coverage). Adding ~300 ms endpointing gives **rule turn ~0.6 s, LLM turn ~1.3 s, open answer ~1.8 s
to the first meaningful word**; the cached ack clip needs only the endpoint. Run-to-run spread is large (the primary
rule-turn median ranged 269-416 ms), so quote ranges. Two mistakes kept visible: a CPU run with a per-state prompt
gave LLM turns of 2438 ms (p95 3433) because every state change was a cache miss; and a single non-interleaved run ranked
configurations wrongly because the machine drifted between runs.

## 10. Things that went wrong or are unverified

* **CORRECTION (found in Phase 1): the "Windows DLL clash" I reported here was wrong.** The failure I saw happened because `uv` did not
  install `sherpa-onnx-core`, so sherpa loaded the old `System32` onnxruntime.dll (v1.17.1). With `sherpa-onnx-core` installed, OpenVINO, onnxruntime and sherpa-onnx load and run together in ONE process in every import order (`scripts/check_runtime_coexistence.py`: 12 of 12 orderings, with real smoke tests). Earlier notes saying otherwise were wrong, and the Silero-VAD-plus-sherpa risk I listed for Phase 3 does not exist on this machine. Keeping TTS in its own process is still
  recommended, but for other reasons: crash isolation, cancelling a request for barge-in, and keeping GPL-3 espeak-ng in a separate process.
* **My OpenVINO "94 ms prefill" claim in the first version of this report was wrong** (it measured KV-cache hits). I found
  it because the sequential test disagreed with the standalone one. Corrected in the headline and sections 8-9.
* The first LLM timings (battery, with other scripts running) were contaminated and were all redone.
* My `ask_car_info`-in-every-state menu mistake cost the 1.7B ~10 intent points before I caught it.
* Ollama's server vanished once and gemma3:4b crashed a connection in the first run (fine in the AC rerun).
* Not tested: Ollama on the iGPU (`OLLAMA_IGPU_ENABLE=1`), llama.cpp directly, prefix cache with raw-prompt
  priming, VAD/turn-detection latency, real recordings, the cloud column (skipped, per your instruction), OpenVINO
  accuracy, Arabic names in TTS.
* Sample sizes are small (29-40 utterances per set); the models would need the Phase 1 scenario suite for a real pass rate.

# Dependency and model licenses

Rule: production uses only MIT / Apache-2.0 / BSD. Status as of 2026-10-20, from PyPI metadata,
Hugging Face model-card tags and the vendor pages named. Anything marked **BLOCKER** or
**OPEN** needs your decision before it ships.

## Libraries

| Component | Version | License | Status |
|---|---|---|---|
| pipecat-ai | 1.13.0 | BSD-2-Clause | OK |
| faster-whisper | 1.2.1 | MIT | OK |
| sherpa-onnx / sherpa-onnx-core | 1.13.8 | Apache-2.0 (own code) | **Embeds GPL-3 espeak-ng statically: see the OPEN DECISION below** |
| useful-moonshine-onnx | 20251121 | MIT | OK |
| silero-vad | 6.2.3 | MIT | OK |
| kokoro-onnx | 0.6.1 | MIT | OK, but pulls GPL deps (below) |
| onnxruntime / onnxruntime-openvino | 1.24.4 / 1.24.1 | MIT | OK |
| openvino / openvino-genai / optimum-intel | 2026.4.x | Apache-2.0 | OK |
| ollama (client) / Ollama server | 0.6.3 / 0.40.5 | MIT | OK |
| sentencepiece, sentence-transformers, huggingface-hub, tokenizers | various | Apache-2.0 | OK |
| rapidfuzz (fast path) | 3.14 | MIT | OK |
| pydantic, pyyaml, python-dotenv, numpy, psutil, soundfile, jiwer, pytest | various | MIT / BSD / Apache | OK |
| **soxr** (benchmark resampling only) | 1.0.0 | **LGPL-2.1+** | Dev-only helper; replace with scipy/numpy resampling before shipping |
| **phonemizer** (pulled in by kokoro-onnx) | 3.4.0 | **GPL-3.0+** | **BLOCKER if Kokoro is used.** Kokoro is not needed (Piper wins on speed) |
| **piper-tts** (PyPI package, OHF-Voice/piper1-gpl) | 1.8.0 | **GPL-3.0+** | **Not used.** Piper voices run through sherpa-onnx instead |
| espeakng-loader | n/a | MIT (loader); bundled espeak-ng is GPL-3 | See "espeak-ng" below |
| librosa (pulled in by moonshine/kokoro) | 1.0.0 | ISC | OK |

### espeak-ng (GPL-3): OPEN DECISION for legal/IT review before ANY production use

**Status:** accepted by you for the **internal prototype** (decision recorded 2026-10-20). **Not accepted for production.**

**Exactly what pulls it in** (verified by scanning the installed binaries, not assumed):

| Component | How espeak-ng gets in | In the production path? |
|---|---|---|
| **sherpa-onnx native libraries** (`_sherpa_onnx.pyd`, `sherpa-onnx-c-api.dll`, wheel `sherpa-onnx-core`) | espeak-ng is **statically linked into the library** (37 references in each file). This is true for every TTS model run through sherpa, including the lexicon-based ones | **Yes** (Piper TTS) |
| **Every Piper voice archive** (libritts_r, joe, ...) | ships an `espeak-ng-data/` directory, used for text-to-phoneme conversion at runtime | **Yes** |
| `espeakng-loader` 0.2.4 (MIT loader) | bundles `espeak-ng.dll` + data (GPL-3) | No (Kokoro only) |
| `phonemizer` 3.4.0 | GPL-3 Python package that calls espeak-ng | No (Kokoro only) |
| `piper-tts` 1.8.0 | GPL-3 (embeds espeak-ng) | No (not installed) |

espeak-ng's own license is GPL-3.0-or-later (its repo also carries other licences for some files, e.g. UCD data and a BSD
getopt shim). The GPL's obligations attach mainly on **distribution**; running it inside one company on one laptop without
handing the program to anyone else is the usual "internal use" reading. Handing the agent to a customer, a franchise, or
another legal entity is distribution. **This is a legal question, not an engineering one.**

**Non-GPL alternatives and what each would cost (my estimates; none built or listened to yet):**

| Alternative | Removes GPL from the process? | Estimated cost | Risks |
|---|---|---|---|
| Lexicon voices in sherpa-onnx (`vits-ljs`, `vits-vctk`; no espeak data, lexicon + tokens only) | **No**: the sherpa binary still contains espeak-ng. Removes only the data and the runtime *use* | ~half a day to benchmark + listen | out-of-lexicon words (car names, Arabic names) may be mispronounced or fail; older VITS quality |
| Build sherpa-onnx from source with espeak-ng disabled, use lexicon voices | Probably yes (not verified that a build switch exists) | 1-2 days (MSVC + CMake toolchain, plus the lexicon-voice evaluation above) | Piper voices unusable; build maintenance on every upgrade |
| Run Piper ONNX voices ourselves (plain onnxruntime) with a different phonemizer | Yes if the phonemizer is permissive | 3-5 days | Piper voices expect espeak's IPA phoneme set; a replacement G2P risks mispronunciation, especially for car and Arabic names |
| Pre-render all fixed phrases offline, keep runtime TTS for dynamic text only | No (runtime TTS still needed for numbers, names, models) | 0.5 day | does not remove the dependency; only reduces how often it runs |
| A TTS whose full chain is permissive (not yet identified or benchmarked) | Yes | unknown: survey + benchmark, probably 1-2 days | may not reach the first-audio target on this CPU |

**Decision needed before production:** (1) legal/IT confirms whether shipping/hosting the agent counts as distribution for
your use, and if so (2) which row above to fund. Until then: keep the TTS in its own process (planned anyway), keep the TTS
engine behind the `TtsEngine` interface so any row above can be swapped in, and do not distribute the build.

## Models

| Model | License | Status |
|---|---|---|
| Qwen3 0.6B / 1.7B / 4B-Instruct-2507 | Apache-2.0 (HF cards) | OK. Production family |
| Qwen3.5 0.8B / 2B | Apache-2.0 (HF cards: Qwen/Qwen3.5-0.8B, -2B) | OK. Verified now; **but not recommended** (no prompt-cache reuse) |
| Ministral 3 (3B) | Apache-2.0 (HF + Ollama page) | OK |
| Llama 3.2 1B / 3B | Llama Community License | **Benchmark only** (your decision) |
| Gemma 3 4B | Gemma Terms of Use | **Benchmark only** (your decision) |
| qwen2.5:3b (already on laptop) | Believed Qwen Research License | Unverified; not used |
| faster-whisper tiny.en / base.en (Systran) | MIT (HF tag) | OK |
| Moonshine tiny / base | MIT (HF tag) | OK |
| sherpa-onnx streaming zipformer en (2023-06-26, 20M) | Apache-2.0 (HF tag / README) | OK. Trained on LibriSpeech |
| OpenVINO/Qwen3-*-int4-ov, whisper-*-int8-ov | Apache-2.0 (HF tags) | OK |
| Kokoro-82M | Apache-2.0 | OK, but not used |
| Silero VAD / Pipecat smart-turn model | MIT / BSD | Verify in Phase 3 |

### Our OpenVINO conversion of Qwen3-4B-Instruct-2507 (Phase 1, T0)

| Item | License | Notes |
|---|---|---|
| Converted weights `models/ov/Qwen3-4B-Instruct-2507-int4-ov-nocal` (and `...-int4-ov` when the AWQ run finishes) | **Apache-2.0** (derivative of `Qwen/Qwen3-4B-Instruct-2507`, whose card says Apache-2.0) | produced locally, not redistributed; keep the Qwen licence/notice with the files if they are ever shared |
| `optimum` 2.3.0, `optimum-intel` 2.2.0, `nncf` 3.4.0, `transformers` 5.5.4, `datasets` 5.0.1, `huggingface-hub`, `openvino`, `openvino-tokenizers`, `safetensors`, `accelerate` | Apache-2.0 | conversion-time only, in an isolated environment (`C:\dev\ov-convert\.venv`), never part of the runtime |
| `torch` 2.14.1 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause (PyPI SPDX expression) | conversion-time only |
| AWQ calibration text `Salesforce/wikitext` (`wikitext-2-raw-v1`, train split) | **CC-BY-SA-3.0 and GFDL (share-alike)** | used only to compute quantization statistics (128 windows of 32 tokens from the first 1,024 rows); the weights are not trained on it. Not a legal opinion: flag for the same legal/IT review as espeak-ng if the converted weights are ever distributed |

### Piper voices: license is set by the TRAINING DATA, and several popular voices are NOT usable
Checked from each voice's MODEL_CARD (rhasspy/piper-voices):

| Voice | Training data license | Commercial use |
|---|---|---|
| en_US **lessac** | Blizzard 2013 licence: research only; excludes building speech products/services | **NO (BLOCKER)** |
| en_US **ryan** | CC BY-NC-SA 4.0 | **NO** |
| en_US **hfc_male / hfc_female** | CC BY-NC-SA 4.0 | **NO** |
| en_US **amy (medium)**, `amy_medium` | "See URL" (Mimic3); model card: **"Finetuned from U.S. English lessac voice"** (lessac = research-only, see above) | **OPEN DECISION, worse than the low build: internal prototype ONLY.** Do not ship. In use because it is the owner's chosen voice and 3x faster than the low build |
| en_US amy (low, int8), `amy` | "See URL" (Mimic3); model card: finetuned from the Ryan voice | **OPEN DECISION.** In use for the INTERNAL PROTOTYPE on the owner's request (best pronunciation). Licence still UNCONFIRMED: the model card says "See URL" (github.com/MycroftAI/mimic3-voices), and that repo has no amy entry and no per-voice licence column. Not shippable until the dataset licence is confirmed; switch `tts.voice` (libritts_r speaker 1 is CC BY 4.0, joe is CC0) |
| en_US **libritts_r** | CC BY 4.0 | Yes, with attribution |
| en_US **joe**, **kathleen** | CC0 | Yes |
| en_US **ljspeech**, **kristin**, **norman**, **john**, **bryce** | public domain | Yes |
| en_GB **alba** | CC BY 4.0 | Yes, with attribution |

The benchmark used lessac early on; every later number uses libritts_r / joe / ljspeech / alba. The
voice *model* files themselves also have their own distribution terms; confirm before shipping.

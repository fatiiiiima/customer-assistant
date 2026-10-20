# Dependency and model licenses

Rule: production uses only MIT / Apache-2.0 / BSD. Status as of 2026-10-20, from PyPI metadata,
Hugging Face model-card tags and the vendor pages named. Anything marked **BLOCKER** or
**OPEN** needs your decision before it ships.

## Libraries

| Component | Version | License | Status |
|---|---|---|---|
| pipecat-ai | 1.13.0 | BSD-2-Clause | OK |
| faster-whisper | 1.2.1 | MIT | OK |
| sherpa-onnx / sherpa-onnx-core | 1.13.8 | Apache-2.0 | OK |
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

### espeak-ng (OPEN)
Piper voices need a phonemizer. sherpa-onnx ships `espeak-ng-data` inside each Piper voice folder.
espeak-ng is GPL-3. Running it as data/library inside an internal, non-distributed deployment is a
legal call for you (GPL obligations mainly attach on distribution). If it must be avoided, the
alternative is a TTS whose chain has no espeak-ng; not benchmarked yet.

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

### Piper voices: license is set by the TRAINING DATA, and several popular voices are NOT usable
Checked from each voice's MODEL_CARD (rhasspy/piper-voices):

| Voice | Training data license | Commercial use |
|---|---|---|
| en_US **lessac** | Blizzard 2013 licence: research only; excludes building speech products/services | **NO (BLOCKER)** |
| en_US **ryan** | CC BY-NC-SA 4.0 | **NO** |
| en_US **hfc_male / hfc_female** | CC BY-NC-SA 4.0 | **NO** |
| en_US amy | "See URL" (Mimic3) | Unclear: do not use until read |
| en_US **libritts_r** | CC BY 4.0 | Yes, with attribution |
| en_US **joe**, **kathleen** | CC0 | Yes |
| en_US **ljspeech**, **kristin**, **norman**, **john**, **bryce** | public domain | Yes |
| en_GB **alba** | CC BY 4.0 | Yes, with attribution |

The benchmark used lessac early on; every later number uses libritts_r / joe / ljspeech / alba. The
voice *model* files themselves also have their own distribution terms; confirm before shipping.

# Dependency and model licenses

Rule: production uses only MIT / Apache-2.0 / BSD. Status as of 2026-10-20, from PyPI
metadata and the vendor pages named below. "Unverified" means a source did not state the
license; it must be read from the primary model card or repo before use.

## Libraries

| Component | Version | License | Status |
|---|---|---|---|
| pipecat-ai | 1.13.0 | BSD-2-Clause | OK |
| faster-whisper | 1.2.1 | MIT | OK |
| sherpa-onnx | 1.13.8 | Apache-2.0 | OK |
| useful-moonshine-onnx | 20251121 | MIT | OK (English models; check each model card) |
| silero-vad | 6.2.3 | MIT | OK |
| kokoro-onnx | 0.6.1 | MIT (README) | OK; see espeak/G2P caveat |
| onnxruntime / onnxruntime-openvino | 1.31.0 / 1.24.1 | MIT | OK |
| openvino / openvino-genai / optimum-intel | 2026.4.x | Apache-2.0 | OK |
| ollama (client) | 0.6.3 | MIT | OK |
| llama-cpp-python | 0.3.36 | MIT | OK |
| sentence-transformers | 6.1.0 | Apache-2.0 | OK |
| sqlite-vec | 0.1.9 | MIT / Apache-2.0 | OK |
| pydantic, pyyaml, python-dotenv, numpy, psutil, soundfile, jiwer, pytest | various | MIT / BSD / Apache | OK |
| **piper-tts** (current, OHF-Voice/piper1-gpl) | 1.8.0 | **GPL-3.0-or-later** | **NOT permissive** |
| **phonemizer / phonemizer-fork** | 3.x | **GPL-3.0-or-later** | **NOT permissive** |
| espeak-ng (used for phonemization by Piper and Kokoro paths) | n/a | GPL-3 (not verified here) | **Open question** |
| misaki (Kokoro G2P) | n/a | Unverified | Check repo |

## Models

| Model | License | Status |
|---|---|---|
| Qwen3 0.6B / 1.7B / 4B (and 4B-Instruct-2507) | Apache-2.0 (HF card for 1.7B) | OK. Production default family |
| Kokoro-82M | Apache-2.0 (HF card) | OK |
| Ministral 3 (3B) | Apache-2.0 (Ollama page) | OK; benchmark candidate |
| Qwen3.5 (0.8B/2B/4B) | Unverified (Ollama page names none) | Benchmark only until verified |
| Gemma 4 (E2B/E4B) | Unverified | Not planned (4.6 GB+); verify if considered |
| Llama 3.2 1B/3B | Llama Community License | **Benchmark only** (your decision) |
| Gemma 3 4B | Gemma Terms of Use | **Benchmark only** (your decision) |
| qwen2.5:3b (already on this laptop) | Believed Qwen Research License (non-commercial) | **Unverified; not used** |
| Piper voices | Per-voice (some CC-BY-NC or restricted) | Check each voice's MODEL_CARD before use |
| Moonshine English models | MIT (verify each card) | Check card |
| Silero VAD / Pipecat smart-turn model | MIT / BSD (verify in Phase 3) | Check card |

## Open issue for your decision: Piper and espeak-ng are GPL

Piper's maintained runtime is GPL-3 and embeds espeak-ng. Kokoro's phonemization may also
fall back to espeak-ng. Options, in my order of preference:

1. Benchmark Piper **voices** through sherpa-onnx (Apache-2.0) and Kokoro through
   kokoro-onnx, then check whether the espeak-ng-data they bundle is acceptable. This
   keeps GPL Python code out of the process, but the espeak-ng data question remains.
2. Treat GPL as acceptable for an internal, non-distributed deployment on one laptop.
   GPL obligations mainly apply when you distribute. This is a legal question for you.
3. Pick a TTS whose whole chain is permissive, if the spike shows one is fast enough.

Benchmarking GPL tools locally is fine; the question is only what ships.

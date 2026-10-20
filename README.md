# Voice agent: used-car dealership (UAE)

Fully local, CPU-only voice agent. An LLM only extracts intent and slots as constrained
JSON; a state machine, policy engine and templates produce the spoken text.
Production path needs no API keys and sends no data off the machine.

Status: **Phase 0b done** (latency/accuracy spike). Results and the recommended stack: [docs/PHASE0B_REPORT.md](docs/PHASE0B_REPORT.md).
Benchmarks must be run on **AC power** (battery cuts sustained speed to ~30%); the suite scripts refuse to run on battery.

## Requirements
- Windows laptop, Python 3.11 or 3.12 (3.12 recommended), [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) installed and running
- Git

## Setup
```powershell
# one-time: install uv, then let it fetch Python 3.12
python -m pip install --user uv
uv python install 3.12

cd C:\dev\voice-agent
uv sync --python 3.12            # core + dev deps
uv run pytest                    # config tests
```
Extras are installed per phase: `uv sync --extra spike` (Phase 0b), `--extra openvino`,
`--extra rag` (Phase 2), `--extra voice` (Phase 3).

## Models (Ollama)
```powershell
.\scripts\pull_models.ps1                          # Apache-2.0 candidates
.\scripts\pull_models.ps1 -IncludeBenchOnly        # + Llama 3.2, Gemma 3 (benchmark only)
.\scripts\pull_models.ps1 -IncludeUnverified       # + Qwen3.5 (license unverified)
```
Qwen3 thinking mode is disabled per request (`think: false`), set in `config/default.yaml`.

### Manual model download (if a pull fails)
If a download fails because of network restrictions, the script stops and names the model.
Don't work around it. Download the file on an allowed network and tell me where it is.
Ollama can import a GGUF with a `Modelfile` (`FROM C:\path\model.gguf`) and `ollama create`.

## Configuration
`config/default.yaml` holds everything; override any key with `VA__SECTION__KEY` env vars
or a gitignored `.env` (see `.env.example`). Items marked **PLACEHOLDER** or
**TO CONFIRM WITH SALES** are invented for development: negotiation limits, test-drive
rules, branches and working hours.

## Cloud LLM mode (dev only, off by default)
Local is the default. Cloud needs `llm.provider: cloud` **and** `VOICE_AGENT_ALLOW_CLOUD=1`,
prints a warning at startup, and is currently a disabled stub
(`src/dealer_core/llm/cloud_stub.py`). Synthetic data only; STT and TTS stay local.

## Privacy
Latency logs (`logs/latency.jsonl`) contain timings only. Transcripts and phone numbers are
excluded unless `latency_log.include_transcripts` is set to true.

## Windows gotchas found in Phase 0b
- `sherpa-onnx` needs its `sherpa-onnx-core` package (which carries its own `onnxruntime.dll`); `uv` does not install it
  transitively, so it is pinned. With it present, OpenVINO, onnxruntime and sherpa-onnx load and run together in ONE process in every import order (`scripts/check_runtime_coexistence.py`: 12 of 12 orderings, with real smoke tests). Earlier notes saying otherwise were wrong: they came from the period when `sherpa-onnx-core` was missing and sherpa picked up the old `System32` DLL. Re-run the script after any runtime upgrade.
- `uv` does not install `sherpa-onnx-core` transitively; it is pinned in the `spike` extra.
- Piper's `lessac`, `ryan` and `hfc_*` voices are not licensed for commercial use. See docs/LICENSES.md.

## Layout
```
config/            default.yaml, hotwords.txt
src/dealer_core/   backend service layer (config now; inventory, policy, tools later)
benchmarks/        Phase 0b spike + your recordings (see docs/RECORDING_GUIDE.md)
docs/              LICENSES.md, RECORDING_GUIDE.md
scripts/           pull_models.ps1
tests/
```
See [docs/LICENSES.md](docs/LICENSES.md) for dependency licenses and open issues.

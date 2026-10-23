# Voice agent: used-car dealership (UAE)

A fully local voice agent for a used-car dealership (AED, km). It answers questions about cars, books test drives and can hand over to a person. Speech
recognition, the language model and the voice all run on this laptop; the production path needs no API keys and sends no data off the machine.

## What actually runs (current defaults)

| Piece | Default | Where it is set |
|---|---|---|
| Conversation engine | **state machine**: code decides what the caller meant and what happens next (search, price, slots, booking) | `dialogue.engine` in `config/default.yaml`; `--agent` switches the website server / voice demo to the **agent** engine |
| Who writes the spoken words | **the language model, always** (Qwen3-4B-Instruct-2507 int4 on the iGPU). `voice.model_only: true` means no canned fallback sentence is ever spoken in its place; a sentence that fails a safety check is dropped | `voice.model_only` |
| Safety checks on every sentence | no invented number, name, colour or claim; no "booked/confirmed" unless the system said so; no first-person promises | `src/dealer_core/dialogue/writer.py` |
| Opener ("Got it!") | none: the reply starts with the answer. `voice.opener` exists but is off | `voice.opener` |
| Greeting | fixed text with the advisor's name (`app.agent_name`: Wazir) | `config/default.yaml` |
| End of your turn | Smart Turn v3 (a model) after 0.2 s of silence | `voice.smart_turn`, `voice.smart_stop_secs` |
| Talking over the agent | on for the website and for the demo with `--headset` | `voice.barge_in` |
| Speech recognition | Whisper base int8 (OpenVINO, CPU), hint list follows the question the agent just asked | `voice.stt_model`, `voice/prompts.py` |
| Voice | Piper "Amy" medium (licence unconfirmed: internal prototype only, see `docs/LICENSES.md`) | `tts.voice` |

The **website** (`serve_portal.cmd`) uses the same defaults plus `config/portal.yaml`: the site's own 24 cars (read from its `site/data.js`), one showroom open
9 AM to 8 PM, fixed prices (no negotiation) and no message sending. So **the website default is the state machine, not the agent**; the agent is opt-in with `--agent`.

The **agent engine** (experimental): each turn the model only *names* a tool (`search_cars`, `get_car`, `check_slots`, `prepare_booking`, `confirm_booking`,
`cancel_draft`, `handoff`, `end_call`, `none`); code reads the details out of the caller's words, runs the tool against the real inventory/slots/bookings, and the
model speaks the result through the same safety checks. A booking needs a read-back and a "yes".

## Run it

```powershell
cd C:\dev\voice-agent
serve_portal.cmd                      # voice server for the website (port 8765); then start the site (bmw-portal\start.bat) and press the mic button
serve_portal.cmd --agent              # same, with the agent engine
demo_voice.cmd --headset --input-device 2 --output-device 4      # talk to it in the terminal (demo_voice.cmd --list-devices shows the numbers)
.venv\Scripts\python.exe -m pytest -q # tests
```

More on the voice side (models, measured latency, Smart Turn, barge-in, model comparison, the agent engine): [docs/VOICE.md](docs/VOICE.md).
Benchmarks must be run on **AC power** (battery roughly halves speed).

## Requirements
- Windows laptop, Python 3.11 or 3.12 (3.12 recommended), [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) installed and running (fallback model server)
- Git

## Setup
```powershell
# one-time: install uv, then let it fetch Python 3.12
python -m pip install --user uv
uv python install 3.12

cd C:\dev\voice-agent
uv sync --python 3.12            # core + dev deps
uv run pytest
```
Extras are installed per phase: `uv sync --extra spike` (Phase 0b), `--extra openvino`,
`--extra rag` (Phase 2), `--extra voice` (voice: Pipecat, websocket, uvicorn).

## Models
OpenVINO models live in `models/ov/` (the language models, Whisper); Piper voices in `models/`. Ollama models:
```powershell
.\scripts\pull_models.ps1                          # Apache-2.0 candidates
.\scripts\pull_models.ps1 -IncludeBenchOnly        # + Llama 3.2, Gemma 3 (benchmark only)
.\scripts\pull_models.ps1 -IncludeUnverified       # + Qwen3.5 (license unverified)
```
Qwen3 thinking mode is disabled (`think: false`), set in `config/default.yaml`.

### Manual model download (if a pull fails)
If a download fails because of network restrictions, the script stops and names the model.
Don't work around it. Download the file on an allowed network and tell me where it is.
Ollama can import a GGUF with a `Modelfile` (`FROM C:\path\model.gguf`) and `ollama create`.

## Configuration
`config/default.yaml` holds everything; `config/portal.yaml` is an overlay for the website. Override any key with `VA__SECTION__KEY` env vars
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

## Windows gotchas
- `sherpa-onnx` needs its `sherpa-onnx-core` package (which carries its own `onnxruntime.dll`); `uv` does not install it
  transitively, so it is pinned. With it present, OpenVINO, onnxruntime and sherpa-onnx load and run together in ONE process in every import order (`scripts/check_runtime_coexistence.py`). Re-run the script after any runtime upgrade.
- Piper's `lessac`, `ryan` and `hfc_*` voices are not licensed for commercial use. See docs/LICENSES.md.
- Hugging Face repos marked "gated" (for example the OpenVINO build of Gemma 3) need a logged-in account that has accepted the licence.

## Layout
```
config/            default.yaml, portal.yaml (website overlay), hotwords.txt, nlu_gates.yaml
src/dealer_core/   dialogue/ (state machine, reply writer + checks), agent/ (agent engine), nlu/, voice/ (Pipecat pipeline, Smart Turn, barge-in),
                   server/ (website WebSocket server), site/ (reads the website's cars), tools/, policy/, repos/
scripts/           measuring and comparison tools (portal_call.py synthetic caller, compare_models.py, replay_utterances.py, ...)
tests/             unit + scenario tests; tests/data/real_utterances.yaml holds lines that went wrong in real calls
docs/              VOICE.md (voice side), ARCHITECTURE.md, LICENSES.md, PHASE0B_REPORT.md, ...
```
See [docs/LICENSES.md](docs/LICENSES.md) for dependency licenses and open issues.

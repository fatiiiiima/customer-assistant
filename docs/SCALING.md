# Latency, quality, hearing and scale: what was changed, what was measured, what is next

All numbers: this laptop (Core Ultra 7 255H, Arc iGPU), AC power, agent + prefix engine, Qwen3-4B int4, Whisper base int8, Piper amy.

## What changed (23 October 2026)

| Area | Change | Measured effect |
|---|---|---|
| Scale | The model runs with **continuous batching + prefix cache** (`llm.backends.openvino_gpu.batching: true`, `nlu/ov_batching.py`) | 8 requests at once: first token 5.1 s -> 0.23 s; one request alone: 0.20 s -> 0.08 s |
| Scale | Batching runs in **its own process** | its scheduler held Python's lock for up to 0.83 s and stalled every call's audio; now 8 ms |
| Scale | **One model instance** per backend (the decision chain and the reply writer each loaded it) | 2 x 4.5 GB -> 4.5 GB; RAM 89% -> 68% at 24 callers |
| Scale | **Two shared synthesizers** (`tts.copies: 2`) | 8 sentences at once: worst 2.35 s -> 1.35 s |
| Scale | Thread pool sized to the call cap | the default pool (20 threads) ran out at ~20 callers |
| Latency | **Code decides certain turns** without the model (a pick from the list, the next booking detail, a plain yes, km/colour/price/year/haggling about the car just discussed) | those turns skip the 0.25-0.4 s decision call |
| Quality | Short answers to follow-ups ("It has 71,000 kilometres.") and "Our prices are fixed..." once, instead of the whole line again | no more 3x repeated fact sentence; no more "Which day would suit you?" for a km question |
| Hearing | Name question: no hint (a list of names made Whisper write a list of names; "my name is" in the hint made it loop) | name clip went from "my name is, my name is, ..." to a name |
| Hearing | Looped-phrase filter on transcripts | "my name is" x 20 -> "my name is" |
| Hearing | Quiet-speech gate measures the loudest 0.3 s, threshold 0.012 -> 0.005 | at 10% volume: 0/24 clips heard -> 20/24 (Whisper hears quiet speech; the old gate threw it away) |
| Hearing | Browser: proper low-pass before 48 -> 16 kHz (63-tap filter) and automatic gain on | noise at 9-20 kHz: -5..-12 dB (folded into speech) -> -65..-110 dB |

## Results

One caller, 23 turns of 5 scripted conversations (`bench_end_to_end.py --engine agent --prefix`, synthetic caller voice):

| | Before | After |
|---|---|---|
| Median to first sound | 0.88 s | **0.84 s** |
| 90th percentile | 2.55 s | **1.53 s** |
| Worst | 3.47 s | **1.88 s** |
| Turns over 2 s | 4 of 23 | **0 of 23** |
| Silent turns | 0 | 0 |

Many callers at once (`load_test.py --mix --stagger 2`, 4 different conversations, tap-to-talk):

| Callers | Before: median / p90 / worst / over 2 s | After: median / p90 / worst / over 2 s |
|---|---|---|
| 1 | 1.16 / 2.07 / 2.09 s / 29% | 0.92 / 2.00 / 2.23 s / 14% |
| 8 | 1.22 / 2.47 / 3.34 s / 17% | 1.02 / 2.03 / 3.61 s / 11% |
| 16 | 1.62 / 2.78 / 11.9 s / 31% | **1.12 / 1.87 / 3.77 s / 6%** |
| 24 | 2.62 / 7.34 / 25.8 s / 71% | 1.70 / 3.17 / 5.41 s / 36% |
| 32 | 7.69 / 13.4 / 30.7 s / 76% | most calls dropped (see below) |

Comfortable capacity of this laptop: **about 16 callers** (was about 8). At 32 the voice server's single event loop falls behind (every
call's per-frame speech detection runs in it) and connections time out. That limit is per process, not per machine.

Hearing, 24 of your own recordings, key words right (`study_hearing.py`):

| Recognizer | Hint | Right | Time per clip |
|---|---|---|---|
| Whisper base (in use) | what the agent just asked | 20/24 | 0.23 s |
| Whisper small.en | same | 21/24 | 0.73 s |
| Whisper small | same | 18/24 | 0.77 s |

A bigger Whisper on this CPU buys one clip for half a second per turn: not worth it. The misses are "the grey one" heard as "Degre one",
"Land Cruiser" as "new car", and the name "Ahmed Khan". The read-back before booking catches a misheard name or number.

## Why "under 1 s for every response" is not reachable on this laptop

The median is under 1 s. What remains above it:
- **Waiting for the caller to finish** (0.2-0.35 s; 0.9 s right after asking for a phone number, so digits said in groups are not cut).
- **Turns the model writes** (a question the code cannot answer from the data): 0.5-1.5 s of iGPU time at ~19 tokens/s.
- **Speech recognition** 0.2 s on the CPU.
On a GPU server these become ~0.05 s (recognition) and ~0.2-0.4 s (model), which puts every turn under 1 s.

## To serve 100+ callers without losing quality or speed

1. **Split the heavy parts into services** (the code already talks to the model through one backend interface):
   - Model server with continuous batching: vLLM (Apache-2.0) or OpenVINO Model Server (Apache-2.0) on one GPU. A single 24-48 GB GPU
     (L4 / L40S / A10 class) serves 100+ concurrent short conversations with a 4-8B model at 0.1-0.3 s to first token.
   - Speech recognition on a GPU with batching: faster-whisper (MIT) or Whisper via OVMS; large-v3-turbo there is faster than base is here.
   - Voice synthesis stays CPU (Piper is cheap); 2-4 copies per voice server.
2. **Several voice-server processes** (each ~15-20 calls, one per CPU core group) behind a load balancer with sticky WebSockets. The
   server already refuses calls over its cap with "busy" and reports `/health` (active calls, cap) for the balancer to read.
3. **Shared booking store**: bookings live in one SQLite file today. With more than one server, move them to PostgreSQL with a unique
   constraint on (car, slot) so two callers can never book the same slot.
4. **Secure transport and abuse limits**: HTTPS / wss, per-IP call limits, a maximum call length (exists: `max_call_secs`).
5. **Measure before buying**: run `load_test.py` against the GPU setup with 50, 100 and 150 callers, plus real recordings with noise, and
   a one-hour soak for memory growth.

Licences: everything named above is MIT or Apache-2.0. Piper voices and espeak-ng (GPL-3) remain the open licence question (docs/LICENSES.md).

## Re-measured with the new features (23 Oct 2026, evening, AC power)

One caller, 32 turns (the 23 before plus a 9-turn "details" conversation: hours, compare, warranty, accident, boot, repeat, thanks),
synthetic caller voice Amy (the agent is now Joe):

| Run | Median | Mean | p90 | Worst | Over 2 s | Silent |
|---|---|---|---|---|---|---|
| 1 | 0.76 s | 1.01 s | 1.69 s | 3.25 s | 1 of 32 | 0 |
| 2 | 0.88 s | 1.09 s | 1.70 s | 3.16 s | 1 of 32 | 0 |

The new-feature questions (hours, compare, warranty, accident, boot, repeat) took 0.42-0.75 s each: code answers them without a model call.
The one slow turn is a mishearing ("any budget" heard as "AE Budget"), which the model then answered.

Many callers, five different conversations including the new features, starts 2 s apart:

| Callers | Failed | Median | p90 | Worst | Over 2 s |
|---|---|---|---|---|---|
| 1 | 0 | 1.17 s | 1.73 s | 1.75 s | 0% |
| 8 | 0 | 1.05 s | 1.69 s | 2.53 s | 5% |
| 16 | 0 | 1.16 s | 1.79 s | 2.55 s | 5% |
| 24 | 0 | 1.52 s | 2.56 s | 3.39 s | 27% |

(RAM peaked at 83% because the owner's own voice server was running at the same time, holding a second copy of the model.)

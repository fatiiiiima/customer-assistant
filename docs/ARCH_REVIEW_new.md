# Architecture review, second pass (28 Oct 2026, read-only)

Sources: `docs/ARCH_REVIEW.md` (DONE items not proposed again), README, docs/, the code, `logs/cluster/worker1.log` (one-worker run from 11:55) and the words log of the latest real call, `logs/audio_dump/20261028_115958_0112_in.txt` (12:00, Chrome, 10 turns). Nothing was run. A number not in those sources is **unknown**.

## 1. What the system is

1. One WebSocket per call, 16 kHz PCM both ways (`bmw-portal/site/voice.js`); tap-to-talk, mic held open until the reply starts (`tap_grace_secs`).
2. Per call: MicGate → Silero (0.6 s stop in tap mode) → Smart Turn → Whisper → DealerBrain → Kokoro (`voice/pipeline.py`).
3. `AgentEngine` routes most turns in code (`_fast_decide`); otherwise the model names a tool, code runs it and writes a "core" wording plus facts.
4. A governor picks full / prefix / code from a background probe of the model; booking lines are exact code at every level.
5. At `prefix` (this Mac, always): code speaks the facts, the model rewords only the closing question, which is checked back to the code's.
6. Hearing guards: level gate, "speak up", echo-of-hint, loop caps, quiet-goodbye (`voice/services.py`, `voice/brain.py`).
7. Per-worker cap from measured p90; router with power-of-two choice; draining on SIGTERM.
8. Snapshots, registry and bookings in PostgreSQL; production: vLLM, GPU Whisper, ~45 CPU workers (`deploy/`).
9. Budget: end of speech → first sound under 1 s.
10. Today: one worker, Ollama, probe 1,251 ms against a 700 ms budget (so every turn is `prefix`); `.env` points the recogniser at small.en (STT 904–1,109 ms per turn in the log; base is 0.24 s per `docs/VOICE.md`); the audio recorder is on.

## 2. Top 5 causes

**C1. The model never writes a sentence here; you are hearing the template level** (problem 1). Probe 1,251–1,479 ms at every start since 11:00 (`worker1.log:5282, 5134, 5002, 4839`) → `choose()` returns `prefix` (`governor.py:71-76`). A fact turn is then: rotated opener (`engine.py:613-624`, `wording.py:82`) + the next two spec-sheet groups in a fixed order (`answers.py:172, 199-209`; `engine.py:1268-1277`) + a question. The model's reworded question is refused if it has any new content word (`engine.py:640-649`), so it comes back as the code's question in other words. That is the transcript word for word. The prompt even forbids the model from opening with "Sure"/"Okay" (`writer.py:368-369`) while the code does it on 6 of 7 replies. MEASURED (probe), INFERRED (that this is what reads as stilted).

**C2. The booking exchange is three logic faults in a row** (problem 1 and your finding).
(a) "What day and time are available?" misses `_FLEXIBLE` (`engine.py:84-89` wants `day|time` directly before `is|are`), so `_t_check_slots` asked "Which day would suit you?" right after offering two times (`engine.py:1350-1354`; log 468.6).
(b) `_offer_next_slots` resets `day_asks` to 0 (`engine.py:1330`), so the next non-answer asks the day again: ask 449, offer 455, ask 468, offer 472.
(c) "Any day would work" after an offer silently takes the earlier time (`_DELEGATES`, `engine.py:1397-1400`). "Any day" is about the day, not a pick between 2 and 5. The `Pending("time_check")` path that asks first exists at `engine.py:1424-1429` and is not used here. MEASURED from the log.
(d) "Let me see." then a one-line question (log 468.15): two fillers because the decision went to the model.

**C3. Every reply ends with a question by construction** (problem 1): `engine.py:721-722`, `writer.py:696-698`. A person answering "tell me more" sometimes just stops. INFERRED.

**C4. Input level, not the model, is behind most of the 30%** (problem 2). This call ran at 0.050–0.088 (`worker1.log:5341-5421`) against 0.19 in the controlled test. Section 13 of the old review measured 27–56% badly wrong between 0.02 and 0.04 and 0–15% at 0.06 and above; this call sat on that edge and got 3 of 10 wrong. small.en added ~0.8 s per turn (MEASURED) and the same section found it does not help at low level. Browser automatic gain is on by default (`voice.js:80`); whether `?mic=raw` was used: unknown.

**C5. The hint is echoed back and nothing checks plausibility** (problem 2). In SLOT state the hint ends with the agent's sentence (`voice/prompts.py:44-45, 52-60`). After "What day and time suit you?" came "I would do and time otherwise at the moment" (level 0.053). `EchoGuard` needs the first three words to match (`services.py:107`), and the SLOT branch of `_fast_decide` has no "not an answer to a time question, re-offer" exit (`engine.py:903-915`), so it went to the model (481 ms) and on into booking. INFERRED.

## 3. Bugs, ranked

| # | Sev. | Where | Evidence |
|---|---|---|---|
| B1 | HIGH | `server/app.py:307` vs `:329`; `server/__main__.py:75` | `TurnClock` gets `state.stop_secs` (0.2) but the tap pipeline runs the VAD at 0.6 (0.9 after a phone question). Every reported time, the page's "replied in", `voice_answer_seconds` and the admission p90 are 0.4 s short. Turn at 12:01:11.472 → sound 13.138: **1.87 s shown**, 2.27 s real. The cap is held to ~1.4 s, not 1 s. |
| B2 | HIGH (laptop) | `nlu/ollama_backend.py:79-80, 82-92`; `engine.py:926` | `lead_in` / `lead_in_stream` drop `deadline_s`: the 1.2 s decision and 4.5 s reply deadlines are not enforced on Ollama, only the 20 s client timeout. The vLLM path honours them (`openai_compat.py:137-147`). |
| B3 | MEDIUM | `engine.py:84-89, 1330, 1350-1354, 1451-1454` | C2 (a) and (b). |
| B4 | MEDIUM | `engine.py:1397-1400` | C2 (c). |
| B5 | MEDIUM at 500+ | `server/app.py:96-103`; `governor.py:81-91`; `engine.py:519-521` | The only way back from `prefix` to `full` is the probe, which runs only at `in_flight == 0`. At 16 calls with 10–20% waiting on the model that is rare: a worker that stepped down in a spike stays down. Each probe is ~1,100 tokens; 45 workers every 20 s ≈ 2.5k tokens/s of prefill. INFERRED. |
| B6 | MEDIUM at 500+ | `repos/pg_store.py:69` (pool 8), `app.py:69`, `shared/state.py:84` (pool 4), `:105, :285-294` | Up to 12 connections per worker: 45 workers ≈ 540 against PostgreSQL's default 100 (`deploy/` does not raise it). Router refresh is N+1 reads plus a DELETE every 0.5 s per copy. INFERRED. |
| B7 | LOW | `voice/audiotap.py:37-40` | With `VA_DUMP_AUDIO_DIR` set (it is, in these runs) every 20 ms frame is written and flushed on the event loop while you measure. |
| B8 | LOW | `engine.py:562`; `voice_rules.py:9` | Violations are checked at MAX_SENTENCES+2, so a three-sentence reply plus opener never registers. |

## 4. Fixes, smallest first

| # | Fix | Label | Latency at 500+ |
|---|---|---|---|
| 1 | B1: give `TurnClock` the pipeline's stop time, following `number_stop_secs`. | MEASURED 0.4 s gap | None for the caller; the admission target becomes a real 1 s. Do first. |
| 2 | B3: add "day and time", "what('s) available", "when can I" to `_FLEXIBLE`; in SLOT with `offered_slots` a non-answer re-offers; stop resetting `day_asks` in `_offer_next_slots`. | INFERRED | One model decision fewer on those turns (0.3–0.5 s here, ~0.1–0.2 s on vLLM). |
| 3 | B4: route `_DELEGATES` through `Pending("time_check")`: "Then shall we say today at 2?" | INFERRED | None. |
| 4 | `tap_stop_secs` 0.6 → 0.35: the reason for 0.6 is now covered by the tap-grace join (old review §10.5). | MEASURED −0.3 s per turn (`docs/VOICE.md`, 547 vs 867 ms) | The largest fixed cost on every turn. Risk: a cut sentence costs a take-back; watch `continued`. |
| 5 | Drop the opener and the model's question rewording on `prefix` turns; speak the code's question. | MEASURED cost 0.8–1.0 s of Ollama per fact turn (`worker1.log:5356-5364`); GUESS on sound | About halves model requests from prefix turns. |
| 6 | B2: run Ollama's `lead_in` through the `_Pieces` reader with the deadline. | INFERRED | Laptop only: bounds a bad decision at 1.2 s, not 20 s. |
| 7 | Hearing: base again, or small.en on `--stt-server`; then `hearing_check.py` at this call's level with `?mic=raw` on and off. | MEASURED 0.8 s per turn | Laptop and mic choice only; production already uses the GPU recogniser. |
| 8 | C5: in SLOT/CONTACT keep only content words of the agent's sentence in the hint; make `EchoGuard` a word-overlap test. | INFERRED | None. |
| 9 | "Tell me more": one group plus a choice question ("performance, or the practical side?"), using `said_about` as is. | GUESS | None. |
| 10 | B5: one prober per model server, or probe at `in_flight ≤ 2`. | INFERRED | Removes ~2.5k tokens/s of wasted prefill; lets workers return to `full` after a spike. |
| 11 | Pre-render "Today I can do", "Tomorrow I can do", "The soonest is" (`kokoro_tts.py:89-108`). | MEASURED 0.35 s (log 12:00:55.571→55.917) | −0.35 s on every slot offer. |
| 12 | B6: one `PgDb` per worker, pool size in config, pgbouncer in `deploy/`; cache the registry read. | INFERRED | Avoids a connection ceiling near 10 workers on default PostgreSQL. |

## 5. What NOT to change

* Governor, admission, shedding, router: fix their clock (B1), not their thresholds.
* Tap-to-talk, the tap-grace join, Smart Turn, the 0.9 s phone-number silence.
* The model's role, the sentence checks, exact booking lines, `model_only` false.
* `min_rms` 0.005, speak-up thresholds, `stt_precision` f32, early start off until measured.
* No bigger Whisper and no denoiser for hearing: section 13 measured both. The lever is signal level.
* No laptop number as a production number; after fix 1, re-measure before sizing.

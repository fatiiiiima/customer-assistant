# Prompt layout for prefix caching

Written 28 Oct 2026 for Phase 2 of `docs/ARCH_PLAN.md`. Labels as in the plan: MEASURED = a number in the docs or logs supports it; INFERRED = it follows from the code; GUESS = must be measured before it is believed.

## Why the order of a prompt matters

A model server with a prefix cache (vLLM's automatic prefix caching) keeps the attention state of prompts it has already read. A new request that starts with the same tokens as an earlier one reuses that state up to the **first token that differs**, and only reads the rest. So the cost of a request is the cost of the part after the first difference, and one changing token near the start throws away everything behind it (INFERRED from how prefix caches work; whether a cache hit really happens on the production server is not measured: see "How to check" below).

Ollama on the laptop is a poor place to judge layout. On 27 Oct a short prompt of another shape in between made the next long prompt be read in full (MEASURED, `docs/ARCH_REVIEW.md` section 7b: one cache slot, the last prompt only). On 28 Oct, Phase 2, the same decision prompt took 115 ms the second time and 124 ms after an unrelated request in between (MEASURED, 4 pairs), so on the Ollama now running the cache is not simply "the last prompt"; why is unknown. Either way the single call's time on the laptop is decoding, not reading the prompt (about 1.55 s whether the prompt was cached or not), so the laptop says nothing about layout; section 21 of the review has what it does say.

## The layout of the single-call prompt (`agent/turn_prompt.py`, built in `agent/engine.py: _single_system, _single_user`)

From what never changes to what changes every turn. Nothing that changes may come before something that does not.

| Part | Where | Changes | Content |
|---|---|---|---|
| Instructions | system message | never in a deployment | the answer format, the actions (the two-call decision prompt's list, word for word), what `say` may contain |
| Dealership facts | system message, after the instructions | never in a deployment (built from the configuration: advisor, currency, branches and hours, test-drive length, "prices are fixed") | `AgentEngine._dealership_facts()` |
| SESSION STATE | user message | when the call's state does | car in focus, cars just shown, search so far, booking draft |
| CARS | user message | with the cars in play | the true facts of the focus car, the cars the caller's words match, and the cars shown (at most five), each with its stock id |
| NOW | user message | every minute | the date and time |
| CONVERSATION | user message | every turn | the last `reply_writer.history_lines` lines (6 in the voice profile) |
| CODE HINT | user message | every turn (left out when the words match no make) | the make and model the caller's words match, and their stock ids |
| CUSTOMER (now) | user message | every turn | the words just heard, always last |

The order of the six user sections is pinned in `tests/unit/test_turn_prompt.py` (`SECTIONS`); the system prompt being the same for every caller, turn and minute is pinned there too.

What is NOT in the static part, on purpose: the inventory. Cars are sold and added during a deployment, so a stock list would change the prefix; the model is given the cars in play instead, found by code (`_code_hint`, `_ids_in_play`).

## What the two-call prompts do (unchanged in Phase 2: they are the reference)

* The decision (`engine.py: _context`, `DECIDE_SYSTEM`): the system message is static, which a cache reuses. The user message starts with `NOW: <date, minute>`, so from the first line of the user message on, every request in a new minute differs from the one before; the car in focus, the cars shown and the booking draft come after the clock.
* The reply writer (`dialogue/writer.py: stream`): the system message (`SYSTEM_VOICE`) is static. The user message starts with CONVERSATION SO FAR, the last six lines, then the caller's words, the facts and the standard wording. The six-line window moves one exchange every turn once a call has more than six lines, so from that turn on the user message differs from the previous one at its very first line.
* Both are cheap to fix (move the clock behind the state; put the state and facts before the conversation) and neither fix is made here: moving text around in the prompts the two-call path uses changes what the model does with them, and that path is the reference the replay compares against. Proposed for after the single call is proven.

## A limit that no layout removes

The conversation is a sliding window of the last N lines. Once a call is past N lines, each turn's CONVERSATION section starts one exchange later than the previous turn's, so the cache can reuse the prompt only up to the start of that section (the instructions, the dealership facts, the state and the cars, and the clock within the same minute). The alternative, showing the whole call so far, makes each prompt the previous one plus two lines and lets the cache reuse all of it, at the price of a prompt that grows with the call (about 25 tokens a line, MEASURED nowhere: GUESS). Not chosen; revisit with the cache hit rate in hand.

## How much of a prompt is shared

`scripts/replay_turns.py` prints, for consecutive model-decided turns of the same recorded call, the share of each prompt (in characters) that starts exactly like the previous turn's prompt: an upper bound on what a prefix cache could reuse, at the granularity of characters. The numbers are in `docs/ARCH_REVIEW.md` section 21. vLLM caches whole blocks of tokens (16 by default), so the real share is slightly lower.

What it shows (MEASURED, characters, one laptop, 28 Oct):

| | decision prompt (two-call) | single-call prompt |
|---|---|---|
| system message, characters | 2,175 | 3,660 |
| one hand-built state (the M3 in focus, one exchange), the minute changes | 83% of the next prompt shared | 94% |
| the same state, one more exchange in the same minute | 92% | 97% |
| the 8 consecutive pairs of model-decided turns in the 16 recorded calls, with each call's own clock | 86% | 85% |

The first two rows are the layout doing what it is meant to (the clock no longer sits in front of the state). The recorded calls do NOT show a gain: the single call's user message carries more that changes (up to five cars' facts, the state), and with only 8 pairs the replay cannot separate the two. Whether the production server's hit rate is better is the open measurement below.

## How to check on the production server (not done: it needs the server)

1. Send the same conversation twice through `scripts/replay_turns.py --backend gpu_server --first-sentence` and compare the time to the first sentence of the first and the second pass.
2. Read the server's prefix-cache counters before and after (hits and queries; the metric names differ between vLLM versions, so take them from `/metrics` of the version in use), and the cached-token count in the usage block of a response where the server reports one.
3. Record the hit rate per prompt (decision, writer, single call) in `docs/ARCH_REVIEW.md`. Until then every statement about the hit rate is INFERRED.

## Rules for changing a prompt

* Put anything that varies per call or per turn after everything that does not. A timestamp, a request id, a caller's name, a counter: after the dealership facts, in a user-message section, never in the system message.
* Do not reorder or reword the system message between releases without a reason: a deployment's cache is warm for the old text.
* Static text belongs in the system message and is built once; if it is built from configuration, build it from configuration only.

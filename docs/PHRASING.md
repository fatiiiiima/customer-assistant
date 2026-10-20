# Making replies feel natural, safely (plan and status)

Principle: the model never writes a price, date, booking or confirmation. Naturalness is added in three stages, safest first.

## Stage 1 - approved wording variants (BUILT; currently ON with the owner-approved set)
* `src/dealer_core/dialogue/phrasebook.py`: 13 low-risk situations (opening, goodbye, hello again, "what car", "say that again", four
  "too many matches" questions, "can't search on that", "no match", two off-topic lines). Each has a base wording (what the demo was
  verified with) and placeholders (`{total}`, `{desc}`).
* `config/phrase_variants.yaml`: variants written **offline** by the local model (`scripts/gen_phrase_variants.py`), generated as `proposed`; the owner then approved all of them (see "Review log").
* **Nothing is used until a person sets `status: approved` AND `dialogue.variants: true`** in `config/default.yaml` (default `false`; currently `true`).
* At run time: no model call. A seeded random generator (`dialogue.variant_seed`, default 7) picks among [base + approved variants],
  never the same one twice in a row in a call; the same seed reproduces the same call. Values are filled in by code.
* Validator (also re-run on every approved entry at start-up; an invalid one stops start-up and names the entry):
  same placeholders; required phrases verbatim (`just show me`); no new commitment or commercial words (guarantee, promise, make sure,
  discount, offer, deal, free, warranty, financing, lowest...); negation parity with the base (a "don't have" cannot become "have");
  not much longer than the base; voice rules (1-2 short sentences).
* Not covered on purpose: prices, dates, bookings, read-backs, confirmations, hand-offs. They stay exact templates.

### How to review
1. Open `config/phrase_variants.yaml`. For each line set `status:` to `approved`, `rejected`, or leave `proposed`. You may edit the text
   (keep `{placeholders}`).
2. Generate more any time (existing entries are never changed): `.venv\Scripts\python.exe scripts\gen_phrase_variants.py --per 8 --only no_match`
3. Turn it on: `dialogue.variants: true` (or env var `VA__DIALOGUE__VARIANTS=true`). Run `pytest`, then the demo replay.
   **Scripted-demo note:** the demo expectations check key words (for example "6 SUVs under 150,000 dirhams"). If you approve variants,
   re-run `demo_replay.cmd` once and fix any expectation that was tied to the exact base wording.

### Review log
* Owner decision: every opening names the **AGMC dealership** (the base opening wording was changed to match: "Hello, welcome to the
  AGMC dealership. ..."); every other proposed variant approved.
* Two garbled entries had a typo fixed, not reworded ("I's read out" and "I we'll read" became "I'll read"); they then equalled the base
  wording and were removed, as were duplicates. The validator checks safety (facts, negation, length), not grammar or taste.
* To go back to the single verified wording: set `dialogue.variants: false` (one line in `config/default.yaml`).

## Stage 2 - per-state style flag (NOT BUILT; agreed design)
`style: template | rephrase | llm_write`, set per state/situation in config.
* **rephrase:** the model rewrites the template with the placeholders intact and never sees or outputs the real values; code fills
  them in afterwards. Verifier before anything is spoken: all placeholders survive; required phrases verbatim (for example "subject to
  confirmation"); no new commitment words; sentence/length limits; no flipped negation (a test set of tricky rewrites). Any failure ->
  the template. Before enabling: measure the added latency and the fallback rate on the local model.
* The stage-1 validator (`check_variant`) is the starting point for this verifier.

## Stage 3 - `llm_write` (NOT BUILT; restricted)
Disabled for prices, dates, bookings and confirmations. Only to be considered for small talk and general car information, text mode first.

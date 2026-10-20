# Making replies feel natural, safely (plan and status)

Principle: the model never writes a price, date, booking or confirmation. Naturalness is added in three stages, safest first.

## Stage 1 - approved wording variants (BUILT, off by default)
* `src/dealer_core/dialogue/phrasebook.py`: 13 low-risk situations (opening, goodbye, hello again, "what car", "say that again", four
  "too many matches" questions, "can't search on that", "no match", two off-topic lines). Each has a base wording (what the demo was
  verified with) and placeholders (`{total}`, `{desc}`).
* `config/phrase_variants.yaml`: variants written **offline** by the local model (`scripts/gen_phrase_variants.py`), all `proposed`.
* **Nothing is used until a person sets `status: approved` AND `dialogue.variants: on`** in `config/default.yaml` (default `off`).
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
3. Turn it on: `dialogue.variants: on` (or env var `VA__DIALOGUE__VARIANTS=on`). Run `pytest`, then the demo replay.
   **Scripted-demo note:** the demo expectations check key words (for example "6 SUVs under 150,000 dirhams"). If you approve variants,
   re-run `demo_replay.cmd` once and fix any expectation that was tied to the exact base wording.
4. Known weak spots in the first batch (review will catch them): two `narrow_again` entries are garbled ("I's read out", "I we'll read");
   some `goodbye` / `hello_again` lines are slightly off ("You're welcome. Have a good day."; "Again, how can I help you?").
   The validator checks safety (facts, negation, length), not grammar or taste: that is the human review.

## Stage 2 - per-state style flag (NOT BUILT; agreed design)
`style: template | rephrase | llm_write`, set per state/situation in config.
* **rephrase:** the model rewrites the template with the placeholders intact and never sees or outputs the real values; code fills
  them in afterwards. Verifier before anything is spoken: all placeholders survive; required phrases verbatim (for example "subject to
  confirmation"); no new commitment words; sentence/length limits; no flipped negation (a test set of tricky rewrites). Any failure ->
  the template. Before enabling: measure the added latency and the fallback rate on the local model.
* The stage-1 validator (`check_variant`) is the starting point for this verifier.

## Stage 3 - `llm_write` (NOT BUILT; restricted)
Disabled for prices, dates, bookings and confirmations. Only to be considered for small talk and general car information, text mode first.

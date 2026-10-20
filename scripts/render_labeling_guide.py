"""Render docs/LABELING_GUIDE.md from the same tables the labeling UI uses, so the guide cannot drift from the tool.

  uv run python scripts/render_labeling_guide.py
"""
from pathlib import Path

from dealer_core.dialogue.states import STATE_CHANGING, STATE_SPEC, Intent
from dealer_core.labeling import core

ROOT = Path(__file__).resolve().parents[1]

HEAD = """# Labeling guide (batch 1 and beyond)

## Run it

From the project folder (PowerShell). This loads your **38 spontaneous transcripts as batch 1** (one utterance per file,
text used verbatim, the matching `.wav` is playable in the page):

```powershell
cd C:\\dev\\voice-agent
.\\.venv\\Scripts\\python.exe -m dealer_core.labeling --input-dir benchmarks\\spontaneous --audio-dir benchmarks\\spontaneous --out data\\nlu_sets\\batch1_spontaneous.csv --open
```

(`--open` opens http://127.0.0.1:8765/ in your browser; drop it to open the page yourself. Ctrl+C in the terminal stops the server.
If you prefer `uv`: `& "$env:APPDATA\\Python\\Python314\\Scripts\\uv.exe" run python -m dealer_core.labeling ...` with the same arguments.)

**Where things are saved** (nothing is written until you save your first label):

| File | What it is |
|---|---|
| `data\\nlu_sets\\batch1_spontaneous.csv` | **the labelled CSV**, rewritten after every save; this is the file the evaluation loads (`--csv`) |
| `data\\nlu_sets\\batch1_spontaneous.labels.jsonl` | the autosave log (one line per save, last label per utterance wins). Close the page and come back any time: it resumes at the first unlabeled utterance |

**Your utterance text is never edited.** The transcripts are shown read-only and copied into the CSV exactly as stored (only the
trailing line break is dropped); the helper never writes to `benchmarks\\spontaneous`. **Nothing is pre-filled or suggested**: every
menu starts blank and Save stays disabled until you have chosen a state and an intent yourself.

**Caution about the held-out status.** `benchmarks\\spontaneous\\README.txt` says these files are never to be used for tuning. Labelled, they
become a measurement set for the NLU (T0 comparison, precision gates). If any rule, prompt, grammar or threshold is later changed *because of* what these
utterances show, they stop being held-out and we should record a replacement set. I will not tune on them.

## How to label one utterance

1. **State**: the state the agent was in when the caller said it. Spontaneous clips have no conversation around them, so pick the state where
   the utterance would most naturally occur; if two are plausible, choose the likelier one and say so in Notes.
2. **Car in focus**: only if the utterance depends on a car already being discussed ("it", "this one", "the car"). Otherwise leave it blank.
3. **Gold intent**: what the caller meant, from the menu. Intents valid in the chosen state come first; the group *"NOT valid in this
   state"* is for the case where the caller really meant something the agent cannot represent there. **Choose it anyway** if that is the truth:
   it is a design finding, not an error.
4. **Slots**: only values the caller actually said (numbers are parsed; phone digits are normalised).
5. **Notes**: anything the next reader needs.

## Ambiguous utterances: what to do

* **Pick one primary intent**: the one the agent must respond to *first*.
* **Tick "Ambiguous / more than one intent"** and, in the multi-select that appears, pick every other intent that would also be an acceptable
  reading. If you cannot name alternatives, write a note explaining the ambiguity instead. **The helper refuses a bare ambiguous flag**:
  it needs an "also acceptable" intent or a note.
* **Slots** are for the primary intent only; mention other slots in Notes.
* **How ambiguous items are used:** they stay in the CSV (columns `ambiguous` and `also_ok_intents`, with your note in `notes`). Evaluation scores the strict label
  and also a lenient intent accuracy that accepts the alternates, and reports ambiguous items separately; **they will not count towards a precision gate**.
* **Discard** is for utterances that are not a caller request at all, not understandable, or not English. They are kept in the autosave log with your note and left out of the CSV.

## States (what the agent was doing)

| State | Meaning | Example (invented) |
|---|---|---|
"""

TAIL = """

## Slots

| Slot | Type | Used with |
|---|---|---|
| make, model | text (lower-case) | search_cars |
| year_min, max_price, max_km | whole number | search_cars |
| body_type | suv / sedan / hatchback / coupe / pickup | search_cars |
| offer_amount | whole number (AED) | make_offer |
| topic | warranty / financing / inspection / trade_in / documents | ask_policy |
| date, time | text as the caller said it | give_slot, reschedule |
| name | text | give_contact |
| phone | digits only | give_contact |

Intents marked * change something in the world or reach a human; rules for them are held to the precision gate in `docs/PHASE1_PLAN.md` section 5.
The examples above are invented; none is an utterance from any test set (a unit test enforces that).
"""


def main() -> None:
    lines = [HEAD.rstrip("\n")]
    for state, (meaning, example) in core.STATE_GUIDE.items():
        lines.append(f"| `{state}` | {meaning} | \"{example}\" |")
    lines.append("\n(`goodbye` is a terminal state and is not offered: label the state the caller was in.)\n")
    lines.append("## Intents (what the caller meant)\n")
    lines.append("| Intent | Meaning | Example (invented) | Valid in these states |")
    lines.append("|---|---|---|---|")
    for intent in Intent:
        meaning, example = core.INTENT_GUIDE[intent.value]
        valid = ", ".join(s.value for s, (ints, _) in STATE_SPEC.items() if intent in ints)
        star = " *" if intent in STATE_CHANGING else ""
        lines.append(f"| `{intent.value}`{star} | {meaning} | \"{example}\" | {valid} |")
    out = ROOT / "docs" / "LABELING_GUIDE.md"
    out.write_text("\n".join(lines) + TAIL, encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()

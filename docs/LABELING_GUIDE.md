# Labeling guide (batch 1 and beyond)

## Run it

From the project folder (PowerShell). This loads your **38 spontaneous transcripts as batch 1** (one utterance per file,
text used verbatim, the matching `.wav` is playable in the page):

```powershell
cd C:\dev\voice-agent
.\.venv\Scripts\python.exe -m dealer_core.labeling --input-dir benchmarks\spontaneous --audio-dir benchmarks\spontaneous --out data\nlu_sets\batch1_spontaneous.csv --open
```

(`--open` opens http://127.0.0.1:8765/ in your browser; drop it to open the page yourself. Ctrl+C in the terminal stops the server.
If you prefer `uv`: `& "$env:APPDATA\Python\Python314\Scripts\uv.exe" run python -m dealer_core.labeling ...` with the same arguments.)

**Where things are saved** (nothing is written until you save your first label):

| File | What it is |
|---|---|
| `data\nlu_sets\batch1_spontaneous.csv` | **the labelled CSV**, rewritten after every save; this is the file the evaluation loads (`--csv`) |
| `data\nlu_sets\batch1_spontaneous.labels.jsonl` | the autosave log (one line per save, last label per utterance wins). Close the page and come back any time: it resumes at the first unlabeled utterance |

**Your utterance text is never edited.** The transcripts are shown read-only and copied into the CSV exactly as stored (only the
trailing line break is dropped); the helper never writes to `benchmarks\spontaneous`. **Nothing is pre-filled or suggested**: every
menu starts blank and Save stays disabled until you have chosen a state and an intent yourself.

**Caution about the held-out status.** `benchmarks\spontaneous\README.txt` says these files are never to be used for tuning. Labelled, they
become a measurement set for the NLU (T0 comparison, precision gates). If any rule, prompt, grammar or threshold is later changed *because of* what these
utterances show, they stop being held-out and we should record a replacement set. I will not tune on them.

## How to label one utterance

1. **State**: for these spontaneous clips **nobody can know it**: there was no agent and no conversation. The state you pick is a *test
   condition* ("assume the agent was here"), not a fact. Ask "what was the agent most plausibly just doing?":

   | The caller is... | Pick |
   |---|---|
   | greeting | `greeting` |
   | describing a car or budget they want | `interest` |
   | asking about / reacting to one specific car ("it", "this one") | `answering` (put a stock id such as S1042 in the car box as a stand-in) |
   | talking about price, discounts, offers | `negotiating` |
   | giving a day or time | `slot` |
   | giving a name or phone number | `contact` |
   | answering yes or no to something the agent just asked | `confirm` (or `negotiating` for a price) and **tick "state is a guess"**: a bare yes/no has no meaning without context |
   | off-topic, angry, asking for a person, saying goodbye, asking about warranty/financing | the state barely matters (these intents are valid in every state): pick `interest`, or `answering` if a car is implied, and **tick "state is a guess"** |

   **Tick "The state is a guess"** whenever the utterance would be equally natural in more than one state, and optionally pick those other states.
   It is stored in the CSV (`state_guess`, `other_plausible_states`); the evaluation will report guessed-state items separately so they cannot distort a
   state-specific number.
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
| `greeting` | the call has just opened and the agent greeted the caller | "hello, good morning" |
| `interest` | the caller is describing what they want; no specific car is chosen yet | "I'm after a family SUV under ninety thousand" |
| `answering` | a specific car is in focus and the agent is answering questions about it | "does it come with a spare key?" |
| `negotiating` | the conversation is about the price of the car in focus | "could you do fifty five thousand?" |
| `slot` | the agent asked which day/time the caller wants a test drive | "Saturday afternoon would suit me" |
| `contact` | the agent asked for the caller's name and phone number | "it's Aisha, zero five zero, one two three, four five six seven" |
| `confirm` | the agent read the booking back and is waiting for yes or no | "yeah, that all sounds right" |

(`goodbye` is a terminal state and is not offered: label the state the caller was in.)

## Intents (what the caller meant)

| Intent | Meaning | Example (invented) | Valid in these states |
|---|---|---|---|
| `greet` | caller opens the conversation or says hello | "hi, good evening" | greeting |
| `search_cars` | caller describes the car they want (make, model, budget, kilometres, body type, year) | "any sedan under forty thousand?" | greeting, interest, answering |
| `ask_car_info` | factual question about the car in focus (needs a car in focus) | "has it had any accidents?" | answering, negotiating, slot, contact, confirm |
| `ask_policy` | question about warranty, financing, inspection, trade-in or documents; valid in any state | "do you offer financing?" | greeting, interest, answering, negotiating, slot, contact, confirm |
| `make_offer` * | caller proposes a price | "I'll give you fifty thousand" | answering, negotiating |
| `ask_lowest_price` * | caller asks for the best/lowest/final price without naming a number | "what's your final figure?" | answering, negotiating |
| `book_test_drive` * | caller wants to see or drive the car | "can I come and try it?" | answering, negotiating |
| `give_slot` * | caller gives a day and/or a time | "tomorrow at five" | slot |
| `give_contact` * | caller gives their name and/or phone number | "this is Khalid, 050 987 6543" | contact |
| `confirm_yes` * | caller agrees with or accepts what the agent just said | "yes, go ahead" | answering, negotiating, confirm |
| `confirm_no` * | caller refuses or disagrees with what the agent just said | "no, that's wrong" | answering, negotiating, slot, contact, confirm |
| `reschedule` * | caller wants to change an existing booking | "can we do Sunday instead?" | slot, confirm |
| `cancel_booking` * | caller wants to cancel the booking | "cancel it please" | slot, contact, confirm |
| `send_details` | caller asks to be sent photos or details | "WhatsApp me the pictures" | answering |
| `request_human` * | caller asks to speak to a person | "can I talk to someone?" | greeting, interest, answering, negotiating, slot, contact, confirm |
| `claim_authority` * | caller says they are staff, the manager or the owner to get special treatment | "I'm the manager, give me the staff price" | answering, negotiating |
| `angry` * | caller insults, shouts at or complains about the agent | "you're useless" | greeting, interest, answering, negotiating, slot, contact, confirm |
| `goodbye` | caller ends the call | "thanks, bye" | greeting, interest, answering, negotiating, slot, contact, confirm |
| `out_of_scope` | anything that has nothing to do with buying a car | "do you know a good place for lunch nearby?" | greeting, interest, answering, negotiating, slot, contact, confirm |

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

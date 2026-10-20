"""Labeling helper: pure logic (loading utterances, validating a human label, storing, exporting, coverage).

Labels are HUMAN-CONFIRMED only: nothing here predicts, suggests or pre-fills a label.
Output CSV matches dealer_core.eval.datasets.load_csv_set:
    id,state,car_in_focus,utterance,gold_intent,gold_slots_json,source,notes,ambiguous,also_ok_intents
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..dialogue.states import SLOT_SCHEMAS, STATE_CHANGING, STATE_SPEC, Intent, State

# Which slot fields are relevant to which intent (the labeler shows these; "show all slots" is available too).
INTENT_SLOTS: dict[Intent, list[str]] = {
    Intent.SEARCH_CARS: ["make", "model", "year_min", "max_price", "max_km", "body_type"],
    Intent.ASK_POLICY: ["topic"],
    Intent.MAKE_OFFER: ["offer_amount"],
    Intent.GIVE_SLOT: ["date", "time"],
    Intent.RESCHEDULE: ["date", "time"],
    Intent.GIVE_CONTACT: ["name", "phone"],
}
INT_SLOTS = {k for k, v in SLOT_SCHEMAS.items() if v.get("type") == "integer"}
ENUM_SLOTS = {k: v["enum"] for k, v in SLOT_SCHEMAS.items() if "enum" in v}

# One-line definitions and a generic example per state / intent. These are shown to the human as a reference; they are NOT used to
# suggest or pre-fill a label, and the examples are invented (none is taken from any test set).
STATE_GUIDE: dict[str, tuple[str, str]] = {
    "greeting": ("the call has just opened and the agent greeted the caller", "hello, good morning"),
    "interest": ("the caller is describing what they want; no specific car is chosen yet", "I'm after a family SUV under ninety thousand"),
    "answering": ("a specific car is in focus and the agent is answering questions about it", "does it come with a spare key?"),
    "negotiating": ("the conversation is about the price of the car in focus", "could you do fifty five thousand?"),
    "slot": ("the agent asked which day/time the caller wants a test drive", "Saturday afternoon would suit me"),
    "contact": ("the agent asked for the caller's name and phone number", "it's Aisha, zero five zero, one two three, four five six seven"),
    "confirm": ("the agent read the booking back and is waiting for yes or no", "yeah, that all sounds right"),
}
INTENT_GUIDE: dict[str, tuple[str, str]] = {
    "greet": ("caller opens the conversation or says hello", "hi, good evening"),
    "search_cars": ("caller describes the car they want (make, model, budget, kilometres, body type, year)", "any sedan under forty thousand?"),
    "ask_car_info": ("factual question about the car in focus (needs a car in focus)", "has it had any accidents?"),
    "ask_policy": ("question about warranty, financing, inspection, trade-in or documents; valid in any state", "do you offer financing?"),
    "make_offer": ("caller proposes a price", "I'll give you fifty thousand"),
    "ask_lowest_price": ("caller asks for the best/lowest/final price without naming a number", "what's your final figure?"),
    "book_test_drive": ("caller wants to see or drive the car", "can I come and try it?"),
    "give_slot": ("caller gives a day and/or a time", "tomorrow at five"),
    "give_contact": ("caller gives their name and/or phone number", "this is Khalid, 050 987 6543"),
    "confirm_yes": ("caller agrees with or accepts what the agent just said", "yes, go ahead"),
    "confirm_no": ("caller refuses or disagrees with what the agent just said", "no, that's wrong"),
    "reschedule": ("caller wants to change an existing booking", "can we do Sunday instead?"),
    "cancel_booking": ("caller wants to cancel the booking", "cancel it please"),
    "send_details": ("caller asks to be sent photos or details", "WhatsApp me the pictures"),
    "request_human": ("caller asks to speak to a person", "can I talk to someone?"),
    "claim_authority": ("caller says they are staff, the manager or the owner to get special treatment", "I'm the manager, give me the staff price"),
    "angry": ("caller insults, shouts at or complains about the agent", "you're useless"),
    "goodbye": ("caller ends the call", "thanks, bye"),
    "out_of_scope": ("anything that has nothing to do with buying a car", "do you know a good place for lunch nearby?"),
}

# Coverage targets per gold intent. "batch1" = the first ~100-utterance batch (breadth), "v1" = a usable first gate decision,
# "gate" = enough zero-error firings to show a 98% lower bound for ONE intent (n >= 189; see docs/PHASE0B_REPORT.md 7a).
TARGETS = {"batch1": 8, "v1": 50, "gate": 189}


@dataclass(frozen=True)
class Item:
    id: str
    utterance: str
    source: str = "typed"          # typed | spoken
    audio: Path | None = None


@dataclass
class Label:
    id: str
    state: str = ""
    car: str = ""
    intent: str = ""
    slots: dict = field(default_factory=dict)
    notes: str = ""
    discard: bool = False
    labeled_at: str = ""
    ambiguous: bool = False                       # the utterance holds more than one intent, or the primary is a judgment call
    also_ok: list = field(default_factory=list)   # other intents that would also be an acceptable reading

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def load_items(path: Path, audio_dir: Path | None = None) -> list[Item]:
    """.txt: one utterance per line (blank and '#' lines skipped; optional TAB-separated `id<TAB>utterance[<TAB>source]`).
    .csv: columns `utterance` (required), `id`, `source`."""
    rows: list[tuple[str, str, str]] = []
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            for i, r in enumerate(csv.DictReader(f), 1):
                if (r.get("utterance") or "").strip():
                    rows.append(((r.get("id") or f"u{i:03d}").strip(), r["utterance"].strip(), (r.get("source") or "typed").strip()))
    else:
        n = 0
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            n += 1
            parts = line.split("\t")
            if len(parts) >= 2:
                rows.append((parts[0].strip(), parts[1].strip(), (parts[2].strip() if len(parts) > 2 else "typed")))
            else:
                rows.append((f"u{n:03d}", line.strip(), "typed"))
    seen: set[str] = set()
    items = []
    for id_, text, source in rows:
        if id_ in seen:
            raise ValueError(f"duplicate id {id_!r} in {path}")
        seen.add(id_)
        wav = (audio_dir / f"{id_}.wav") if audio_dir else None
        items.append(Item(id_, text, "spoken" if (wav and wav.exists()) else source, wav if wav and wav.exists() else None))
    return items


def load_items_from_dir(directory: Path, pattern: str = "spont*.txt", audio_dir: Path | None = None) -> list[Item]:
    """One utterance per file, id = file stem, text VERBATIM (only the trailing line break is removed; the helper never
    rewrites the source files). Audio `<id>.wav` is looked up in `audio_dir` (default: the same folder)."""
    audio_dir = audio_dir or directory
    items = []
    for f in sorted(directory.glob(pattern)):
        text = f.read_text(encoding="utf-8-sig").rstrip("\r\n")
        if not text.strip():
            continue
        wav = audio_dir / f"{f.stem}.wav"
        items.append(Item(f.stem, text, "spoken" if wav.exists() else "typed", wav if wav.exists() else None))
    if not items:
        raise ValueError(f"no utterances matched {pattern!r} in {directory}")
    return items


def slot_fields(intent: Intent | None, show_all: bool = False) -> list[str]:
    return list(SLOT_SCHEMAS) if (show_all or intent is None) else INTENT_SLOTS.get(intent, [])


def valid_intents(state: State) -> list[Intent]:
    return list(STATE_SPEC[state][0])


def clean_slots(raw: dict) -> dict:
    """Turn the form's slot fields into gold_slots: drop empties, parse integers, digits-only phone, enums checked."""
    out: dict = {}
    for k, v in raw.items():
        if k not in SLOT_SCHEMAS:
            raise ValueError(f"unknown slot {k!r}")
        v = "" if v is None else str(v).strip()
        if not v:
            continue
        if k in INT_SLOTS:
            digits = re.sub(r"[,\s]", "", v)
            if not digits.isdigit():
                raise ValueError(f"{k} must be a whole number, got {v!r}")
            out[k] = int(digits)
        elif k == "phone":
            digits = re.sub(r"\D", "", v)
            if len(digits) < 8:
                raise ValueError(f"phone needs digits (at least 8), got {v!r}")
            out[k] = digits
        elif k in ENUM_SLOTS:
            if v.lower() not in ENUM_SLOTS[k]:
                raise ValueError(f"{k} must be one of {ENUM_SLOTS[k]}, got {v!r}")
            out[k] = v.lower()
        else:
            out[k] = v.lower()
    return out


def validate_label(raw: dict, known_ids: set[str]) -> Label:
    """Validate a label submitted by the human. Raises ValueError with a message the UI shows verbatim."""
    id_ = str(raw.get("id", ""))
    if id_ not in known_ids:
        raise ValueError(f"unknown utterance id {id_!r}")
    if raw.get("discard"):
        return Label(id=id_, discard=True, notes=str(raw.get("notes", "")).strip(), labeled_at=_now())
    try:
        state = State(str(raw.get("state", "")))
        intent = Intent(str(raw.get("intent", "")))
    except ValueError as e:
        raise ValueError("choose a state and an intent (nothing is pre-filled)") from e
    if state is State.GOODBYE:
        raise ValueError("goodbye is a terminal state; label the state the caller was in")
    notes = str(raw.get("notes", "")).strip()
    ambiguous = bool(raw.get("ambiguous"))
    also_ok: list[str] = []
    if ambiguous:
        for v in raw.get("also_ok") or []:
            try:
                other = Intent(str(v))
            except ValueError as e:
                raise ValueError(f"unknown intent in 'also acceptable': {v!r}") from e
            if other is intent:
                raise ValueError("'also acceptable' must be different from the primary intent")
            if other.value not in also_ok:
                also_ok.append(other.value)
        if not also_ok and not notes:
            raise ValueError("ambiguous needs at least one 'also acceptable' intent or a note explaining why")
    return Label(id=id_, state=state.value, car=str(raw.get("car", "")).strip(), intent=intent.value,
                 slots=clean_slots(raw.get("slots") or {}), notes=notes, labeled_at=_now(), ambiguous=ambiguous, also_ok=also_ok)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LabelStore:
    """Append-only JSONL (resumable, last label per id wins) + a CSV regenerated after every save."""

    def __init__(self, items: list[Item], jsonl: Path, csv_path: Path) -> None:
        self.items, self.jsonl, self.csv_path = items, jsonl, csv_path
        self.by_id = {i.id: i for i in items}
        self.labels: dict[str, Label] = {}
        if jsonl.exists():
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    if d["id"] in self.by_id:
                        self.labels[d["id"]] = Label(**d)

    def save(self, label: Label) -> None:
        self.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(label.to_json() + "\n")
        self.labels[label.id] = label
        self.export_csv()

    def export_csv(self) -> int:
        n = 0
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "state", "car_in_focus", "utterance", "gold_intent", "gold_slots_json", "source", "notes",
                        "ambiguous", "also_ok_intents"])
            for it in self.items:
                lb = self.labels.get(it.id)
                if lb and not lb.discard:
                    w.writerow([it.id, lb.state, lb.car, it.utterance, lb.intent, json.dumps(lb.slots, ensure_ascii=False),
                                it.source, lb.notes, int(lb.ambiguous), ";".join(lb.also_ok)])
                    n += 1
        return n

    def coverage(self) -> dict[str, int]:
        counts: dict[str, int] = {i.value: 0 for i in Intent}
        for lb in self.labels.values():
            if not lb.discard:
                counts[lb.intent] += 1
        return counts

    def first_unlabeled(self) -> int:
        for n, it in enumerate(self.items):
            if it.id not in self.labels:
                return n
        return len(self.items)


def existing_utterances(data_dir: Path) -> dict[str, str]:
    """normalised utterance -> set name, for the built-in sets: so the labeler can warn about leakage / duplicates."""
    out: dict[str, str] = {}
    for p in sorted(data_dir.glob("*.json")):
        for r in json.loads(p.read_text(encoding="utf-8")):
            out[re.sub(r"\W+", " ", r["utterance"].lower()).strip()] = p.stem
    return out


def intent_is_state_changing(intent: str) -> bool:
    return Intent(intent) in STATE_CHANGING

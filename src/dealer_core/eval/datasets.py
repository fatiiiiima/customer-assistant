"""Labelled utterance sets for NLU evaluation: JSON (the built-in sets) and CSV (your labelled set).

CSV columns: id,state,car_in_focus,utterance,gold_intent,gold_slots_json,source,notes[,ambiguous,also_ok_intents]
(`source` is spoken|typed; `car_in_focus` empty = none). Only human-confirmed labels belong here.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..dialogue.states import Intent, State


@dataclass(frozen=True)
class Example:
    id: str
    set: str
    state: State
    car: str | None
    utterance: str
    gold_intent: Intent
    gold_slots: dict = field(default_factory=dict)
    source: str = "typed"
    ambiguous: bool = False
    also_ok: tuple[str, ...] = ()          # other intents that are also an acceptable reading (only when ambiguous)


def load_json_set(path: Path) -> list[Example]:
    name = path.stem
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Example(id=f"{name}{i:03d}", set=name, state=State(r["state"]), car=r.get("car"), utterance=r["utterance"],
                    gold_intent=Intent(r["intent"]), gold_slots=r.get("slots", {}), source=r.get("source", "typed"))
            for i, r in enumerate(rows)]


def load_csv_set(path: Path, set_name: str | None = None) -> list[Example]:
    name = set_name or path.stem
    out: list[Example] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out.append(Example(id=r["id"], set=name, state=State(r["state"].strip()),
                               car=(r.get("car_in_focus") or "").strip() or None, utterance=r["utterance"].strip(),
                               gold_intent=Intent(r["gold_intent"].strip()),
                               gold_slots=json.loads(r.get("gold_slots_json") or "{}"),
                               source=(r.get("source") or "typed").strip(),
                               ambiguous=(r.get("ambiguous") or "0").strip() in ("1", "true", "True"),
                               also_ok=tuple(x for x in (r.get("also_ok_intents") or "").split(";") if x)))
    return out


def load_sets(data_dir: Path, names: list[str], csv_paths: list[Path] | None = None) -> dict[str, list[Example]]:
    sets = {n: load_json_set(data_dir / f"{n}.json") for n in names}
    for p in csv_paths or []:
        sets[p.stem] = load_csv_set(p)
    return sets

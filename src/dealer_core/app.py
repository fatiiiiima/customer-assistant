"""Wiring: config -> repositories -> DealerService -> NLU (rules + LLM chain) -> DialogueEngine.

Used by the chat CLI and the scenario replayer. Tests build the same thing with a fake classifier and a frozen clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from .config import Config
from .dialogue.machine import DialogueEngine
from .dialogue.phrasebook import Phrasebook
from .nlu.chain import BackendChain
from .nlu.pipeline import IntentClassifier, NluPipeline
from .nlu.rules import RuleEngine, load_enabled_intents
from .repos.demo_data import seed_demo_inventory
from .repos.seed import seed_inventory
from .repos.sqlite_store import SqliteBookingRepo, SqliteDb, SqliteInventoryRepo
from .tools.escalation import MockEscalationSink
from .tools.messaging import MockChannel
from .tools.service import DealerService, utc_now

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class App:
    engine: DialogueEngine
    service: DealerService
    inventory: SqliteInventoryRepo
    bookings: SqliteBookingRepo
    whatsapp: MockChannel
    escalations: MockEscalationSink
    chain: BackendChain | None
    classifier_name: str


def demo_clock(cfg: Config) -> Callable[[], datetime]:
    """A frozen 'now': the next Monday at 10:00 in the first branch's timezone. Keeps scripted demos identical on any day."""
    tz = ZoneInfo(cfg.branches[0].timezone)
    now = datetime.now(tz)
    monday = (now + timedelta(days=(7 - now.weekday()) % 7 or 7)).replace(hour=10, minute=0, second=0, microsecond=0)
    return lambda: monday.astimezone(timezone.utc)


def build_app(cfg: Config, *, inventory: str = "demo", backend: str | None = None, classifier: IntentClassifier | None = None,
              clock: Callable[[], datetime] | None = None, chain: BackendChain | None = None) -> App:
    """inventory: 'demo' (16 hand-written fake cars) or 'seed' (60 random fake cars).
    backend: a name from config llm.backends (default: llm.text_backend). Pass `classifier` to inject a fake (tests),
    or `chain` to reuse one started BackendChain across many calls."""
    db = SqliteDb(":memory:")
    inv = SqliteInventoryRepo(db)
    if inventory == "demo":
        seed_demo_inventory(inv)
    else:
        seed_inventory(inv, [b.id for b in cfg.branches])
    bookings = SqliteBookingRepo(db)
    whatsapp, sink = MockChannel(), MockEscalationSink()
    svc = DealerService(cfg, inv, bookings, {"whatsapp": whatsapp}, sink, clock=clock or utc_now)
    if classifier is None:                      # a ready-made chain can be shared across calls (loading a model is slow)
        chain = chain or BackendChain.from_config(cfg, primary=backend or cfg.llm.text_backend)
        classifier = chain
    rules = RuleEngine(inv.vocabulary, load_enabled_intents(ROOT / "config" / "nlu_gates.yaml"))
    phrasebook = Phrasebook.from_file(ROOT / "config" / "phrase_variants.yaml") if cfg.dialogue.variants == "on" else Phrasebook()
    engine = DialogueEngine(svc, NluPipeline(classifier, rules), phrasebook, cfg.dialogue.variant_seed)
    return App(engine, svc, inv, bookings, whatsapp, sink, chain, getattr(classifier, "name", type(classifier).__name__))

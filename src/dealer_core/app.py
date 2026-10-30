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
from .dialogue.draft import DraftWriter
from .dialogue.writer import ReplyWriter
from .dialogue.leadin import LeadInWriter
from .dialogue.machine import DialogueEngine
from .dialogue.phrasebook import Phrasebook
from .nlu.chain import BackendChain
from .nlu.pipeline import IntentClassifier, NluPipeline
from .nlu.rules import RuleEngine, load_enabled_intents
from .repos.demo_data import seed_demo_inventory
from .repos.seed import seed_inventory
from .repos.sqlite_store import SqliteBookingRepo, SqliteInventoryRepo
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


def engine_obj_writer(e):
    return getattr(e, "reply_writer", None)


def build_app(cfg: Config, *, inventory: str = "demo", backend: str | None = None, classifier: IntentClassifier | None = None,
              clock: Callable[[], datetime] | None = None, chain: BackendChain | None = None,
              lead_in: bool | None = None, lead_backend=None, draft: bool | None = None,
              reply_mode: str | None = None, portal_data: Path | None = None, engine: str | None = None, agent_prefix: bool | None = None, quality: str | None = None) -> App:
    """inventory: 'db' (the cars already in the database: the stock imported with `python -m dealer_core.inventory import`),
    'portal' (the website's data.js imported into the database first: the demo stock), 'demo' (16 hand-written fake cars) or 'seed' (60 random fake cars).
    The database: config app.database_url or DATABASE_URL (postgresql://..., sqlite:///file.db); none is an in-memory SQLite.
    backend: a name from config llm.backends (default: llm.text_backend). Pass `classifier` to inject a fake (tests),
    or `chain` to reuse one started BackendChain across many calls."""
    import os

    from .inventory import branches_of, open_db

    inv, bookings = open_db(cfg.app.database_url or os.environ.get("DATABASE_URL", ""), cfg.app.db_pool_size)
    if inventory == "portal":                    # the website's demo cars, imported from its data.js like any other stock list
        from .inventory.sync import import_file
        if portal_data is None:
            raise ValueError("inventory='portal' needs portal_data=<path to the site's data.js>")
        report = import_file(inv, Path(portal_data), branches_of(cfg), fmt="datajs")
        if report.refused:
            raise ValueError(report.summary())
    elif inventory == "demo":
        seed_demo_inventory(inv)
    elif inventory == "seed":
        seed_inventory(inv, [b.id for b in cfg.branches])
    elif inventory != "db":
        raise ValueError(f"unknown inventory {inventory!r} (db, portal, demo or seed)")
    whatsapp, sink = MockChannel(), MockEscalationSink()
    svc = DealerService(cfg, inv, bookings, {"whatsapp": whatsapp}, sink, clock=clock or utc_now)
    if classifier is None:                      # a ready-made chain can be shared across calls (loading a model is slow)
        chain = chain or BackendChain.from_config(cfg, primary=backend or cfg.llm.text_backend)
        classifier = chain
    rules = RuleEngine(inv.vocabulary, load_enabled_intents(ROOT / "config" / "nlu_gates.yaml"))
    phrasebook = Phrasebook.from_file(ROOT / "config" / "phrase_variants.yaml") if cfg.dialogue.variants else Phrasebook()
    mode = reply_mode or cfg.dialogue.reply_mode
    if lead_in is not None or draft is not None:
        mode = "hybrid"                              # the explicit lead-in / draft switches belong to the hybrid mode
    use_lead = lead_in if lead_in is not None else (mode == "hybrid" and cfg.dialogue.lead_in)
    writer_backend = lead_backend or chain
    writer = LeadInWriter(writer_backend, set(cfg.dialogue.lead_in_intents)) if use_lead and writer_backend is not None else None
    use_draft = draft if draft is not None else (mode == "hybrid" and cfg.dialogue.draft)
    drafter = DraftWriter(writer_backend, set(cfg.dialogue.lead_in_intents)) if use_draft and writer_backend is not None else None
    dialogue = DialogueEngine(svc, NluPipeline(classifier, rules), phrasebook, cfg.dialogue.variant_seed, writer, drafter,
                            ReplyWriter(writer_backend, cfg.dialogue.high_risk_drafts, judge=cfg.dialogue.judge,
                                                                                  retries=cfg.dialogue.retries, max_tokens=cfg.dialogue.max_reply_tokens,
                                                                                  deadline_s=(cfg.dialogue.reply_deadline_ms / 1000.0) or None,
                                                                                  max_words=cfg.dialogue.reply_max_words) if mode == "conversational" and writer_backend is not None else None)
    if (engine or cfg.dialogue.engine) == "agent":
        from .agent.engine import AgentEngine

        if engine_obj_writer(dialogue) is None:
            raise ValueError("the agent engine needs a model-written reply (reply_mode conversational and an LLM backend)")
        decide = None
        if cfg.dialogue.decision_backend and chain is not None:
            from .nlu.chain import BackendChain as _Chain

            decide = _Chain.from_config(cfg, primary=cfg.dialogue.decision_backend)
            decide.start()                                                       # loads the small model
        dialogue = AgentEngine(svc, decide or writer_backend, engine_obj_writer(dialogue))
        dialogue.prefix_mode = cfg.dialogue.agent_prefix if agent_prefix is None else agent_prefix
        dialogue.openers = cfg.dialogue.openers
        dialogue.model_speaks = cfg.dialogue.model_speaks
        dialogue.prefetch_on = cfg.dialogue.prefetch
        if quality or cfg.dialogue.quality != "off":
            from .agent.governor import QualityGovernor

            dialogue.governor = QualityGovernor(quality or cfg.dialogue.quality, cfg.dialogue.model_budget_ms, max_in_flight_model=cfg.dialogue.max_model_turns,
                                                shed_after_ms=cfg.dialogue.model_shed_after_ms, explore_every=cfg.dialogue.explore_every)
    return App(dialogue, svc, inv, bookings, whatsapp, sink, chain, getattr(classifier, "name", type(classifier).__name__))

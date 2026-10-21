"""Scenario harness: the real dialogue engine, tools, policy and demo inventory; only the LLM is faked (a text -> intent table)
so flow logic is tested deterministically. Utterances the RULES answer never reach the fake."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from dealer_core.app import build_app
from dealer_core.config import load_config
from dealer_core.dialogue.states import Intent
from dealer_core.nlu.backend import IntentResult

NOW = datetime(2030, 3, 4, 10, 0, tzinfo=ZoneInfo("Asia/Dubai"))      # a Monday, 10:00 in Dubai


class FakeClassifier:
    name = "fake-llm"

    def __init__(self, table):
        self.table, self.calls = {k.lower(): Intent(v) for k, v in table.items()}, []

    def classify(self, state, car, text, last_agent=""):
        self.calls.append((state.value, text))
        if text.lower() not in self.table:
            raise AssertionError(f"the rules declined {text!r} in state {state.value} and the fake LLM has no answer for it")
        return IntentResult(self.table[text.lower()], 1.0, self.name, "{}")


class Call:
    """One scripted call. `say()` returns (reply, debug) and checks the voice rules on every reply."""

    def __init__(self, table, variants=False):
        # the scenario tests pin the exact base wording; wording variants have their own tests (test_phrasebook.py)
        self.cfg = load_config(environ={"VA__DIALOGUE__VARIANTS": str(variants).lower()})
        self.clf = FakeClassifier(table)
        self.app = build_app(self.cfg, inventory="demo", classifier=self.clf, clock=lambda: NOW.astimezone(timezone.utc))
        self.s = self.app.engine.new_session()
        self.log = []

    def say(self, text):
        r = self.app.engine.handle(self.s, text)
        assert not r.debug.violations, f"voice-rule violations in {r.reply!r}: {r.debug.violations}"
        self.log.append((text, r.reply, r.debug))
        return r


@pytest.fixture
def call():
    return Call

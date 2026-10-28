"""28 Oct, measured on Ollama: in 18-25% of model-written turns the first clause was already spoken ("Sure, that's the 2016 Macan, black,") and the next sentence failed a
check, and the repair then said "That is: that's a 2016 Macan, black, ..." straight after the comma, a restart in the middle of a sentence. The repair now finishes the thought
and gives the facts as a sentence of their own."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dealer_core.agent.engine import AgentEngine, _rounding_off
from dealer_core.agent.governor import QualityGovernor
from dealer_core.app import build_app
from dealer_core.config import load_config
from dealer_core.dialogue.states import Intent
from dealer_core.nlu.backend import IntentResult

NOW = datetime(2026, 10, 27, 11, 0, tzinfo=ZoneInfo("Asia/Dubai"))


class Brain:
    name = "scripted"

    def classify(self, *a, **k):
        return IntentResult(Intent.SEARCH_CARS, 1.0, self.name, "{}")

    def lead_in(self, *a, **k):
        return "none"

    def lead_in_stream(self, *a, **k):
        yield "ok."


class CutWriter:
    """Says one clause and then has a sentence rejected by the checks (what the real writer reports as `stopped`)."""

    def __init__(self, clause):
        self.clause, self.last_stream, self.backend = clause, {}, None

    def stream(self, w, temperature=None):
        self.last_stream = {"sentences": 1, "first_ms": 5.0, "stopped": "a claim the facts do not support: 'reliable'", "appended": False}
        yield self.clause


def reply_to(clause):
    app = build_app(load_config(environ={}), inventory="demo", classifier=Brain(), reply_mode="conversational", clock=lambda: NOW.astimezone(timezone.utc), lead_backend=Brain())
    e = AgentEngine(app.service, Brain(), CutWriter(clause))
    e.governor = QualityGovernor("full", budget_ms=700)
    s = e.new_session()
    list(e.handle_stream(s, "I'm looking for a Toyota"))
    s.car = s.shown[0].stock_id
    out = list(e.handle_stream(s, "tell me about the first one"))
    return [x for x in out if isinstance(x, str)], out[-1]


def test_after_a_spoken_clause_the_facts_come_as_a_sentence_of_their_own():
    spoken, result = reply_to("Sure, that's the Camry,")
    assert spoken[0] == "Sure, that's the Camry,"
    assert spoken[1] == "and here is the full picture."                      # finishes the clause the caller has heard
    assert "That is:" not in " ".join(spoken)
    assert spoken[2][:1].isupper() and "kilometres" in spoken[2]            # the facts, not cut or lower-cased into the clause
    assert spoken[-1].endswith("?") or spoken[-1].endswith(".")


def test_a_reply_cut_in_the_middle_of_a_word_starts_over_politely():
    spoken, _ = reply_to("Sure, it has 91,000")
    assert spoken[1] == "Let me put that properly." and "That is:" not in " ".join(spoken)


def test_the_rounding_off_sentence_depends_only_on_how_the_spoken_text_ended():
    assert _rounding_off("Sure, black,") == "and here is the full picture."
    assert _rounding_off("Sure, black, ") == "and here is the full picture."
    assert _rounding_off("It has") == "Let me put that properly."

"""Realistic multi-turn calls, run with NO help from the model (every model decision returns nothing, every model-written sentence is
rejected), so they show exactly what the code-written answers do on their own. Each call is checked against rules that hold for any good call:

  * never the same question twice in a row (a loop), never the same reply twice in a row
  * every price said is the price of a real car in stock
  * every turn ends with a question, or is a goodbye / a hand-off
  * nothing claims "nothing matches" right after cars were found for the same thing

The lines are the follow-ups people actually say (several came from the owner's test calls).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from dealer_core.agent.engine import AgentEngine
from dealer_core.app import ROOT, build_app
from dealer_core.config import load_config
from dealer_core.dialogue.states import Intent
from dealer_core.nlu.backend import IntentResult

DATA_JS = Path("C:/dev/bmw-portal/site/data.js")
NOW = datetime(2030, 3, 4, 10, 0, tzinfo=ZoneInfo("Asia/Dubai"))
pytestmark = pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")


class NoModel:
    """A model that never helps: every decision is 'none', every sentence it would write is empty."""
    name = "no-model"

    def classify(self, *a, **k):
        return IntentResult(Intent.SEARCH_CARS, 1.0, self.name, "{}")

    def lead_in(self, system, user, max_tokens=36, temperature=None, deadline_s=None):
        return "none"

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
        return iter(())


def agent():
    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=DATA_JS, classifier=NoModel(), reply_mode="conversational",
                    clock=lambda: NOW.astimezone(timezone.utc), lead_backend=NoModel())
    e = AgentEngine(app.service, NoModel(), app.engine.reply_writer)
    e.prefix_mode = True
    return e, app


CALLS = {
    "toyota then booking with a bare hour": ["Toyota.", "Any budget?", "RAV4.", "Yes.", "Tomorrow at", "9.", "050, 793, 2546 and my name is Fatima", "yes"],
    "audi booking then other makes": ["Audi.", "Q7.", "Yes.", "Tomorrow.", "9 in the morning.", "Fatima 050 793 2546", "yes",
                                      "Do you have any other car models?", "What are the other makes that you have?", "BMW", "the X5"],
    "cheaper and family": ["I'm looking for a Toyota", "the second one", "something cheaper", "is it a good family car", "how big is the boot"],
    "seven seats": ["do you have anything with seven seats", "any budget", "which one is the cheapest", "does it have a warranty"],
    "sporty and fastest": ["I want something sporty", "any budget", "which one is the fastest", "the BMW M3", "can I see it this weekend"],
    "details of one car": ["tell me about the Land Cruiser", "is it reliable", "how many owners has it had", "does it have a sunroof",
                           "is it four wheel drive", "what's the fuel economy like", "has it been in an accident"],
    "haggling": ["BMW X5", "how much is it", "can you do 180,000", "what is your best price", "is that a good price"],
    "showroom questions": ["what time do you open", "where are you located", "do you have a branch in sharjah", "what's your whatsapp number"],
    "compare": ["Toyota", "compare the Camry and the RAV4", "which one is better for a family", "the RAV4"],
    "time not offered": ["Toyota", "the Corolla", "can I book a test drive", "tomorrow", "5 PM", "no, 2 PM", "Sara", "055 123 4567", "yes"],
    "never mind": ["Mazda", "the CX-5", "I'd like to test drive it", "actually never mind, maybe later", "no thanks, just looking", "bye"],
    "repeat and thanks": ["Toyota", "any budget", "sorry can you repeat that", "the Camry", "thank you"],
    "greeting and browse": ["hi there", "what cars do you have", "what other brands do you have", "Kia"],
    "misheard car after a list": ["I'm looking for a Toyota.", "any budget?", "the Audi A4.", "the RAV4", "yes", "Saturday", "the afternoon one"],
    "budget said different ways": ["an SUV under 100k", "the second one", "anything under ninety thousand", "what about a white one"],
    "colour and year": ["do you have a white SUV", "anything newer than 2021", "the first one", "what colour is it"],
    "change car mid booking": ["Toyota", "the Camry", "book a test drive", "tomorrow", "2 PM", "actually can I see the RAV4 instead", "yes"],
    "correction of name": ["the BMW M3", "yes", "Friday", "the morning one", "my name is Omar", "055 987 6543", "no, the name is Omer"],
    "two cars one call": ["Toyota", "the Corolla", "yes", "tomorrow", "9", "Sara 050 111 2222", "yes", "and the Camry too", "yes", "tomorrow", "2"],
    "cancel and move": ["Toyota", "the Corolla", "yes", "tomorrow", "9", "Sara 050 111 2222", "yes", "what have I booked", "can I move it to Thursday at 4 pm", "yes",
                        "cancel my test drive", "yes"],
    "vague opener": ["I need a car", "not sure, something reliable", "for my family", "any budget"],
    "electric": ["do you have any electric cars", "what's the range", "how fast is it"],
    "kilometres": ["something with low mileage", "under 40,000 kilometres", "the cheapest one"],
    "location and booking": ["where are you", "I'd like to come and see a car", "the Kia", "tomorrow at 2 pm"],
    "price questions": ["how much is the Land Cruiser", "is it negotiable", "do you do finance", "ok what about the Patrol"],
    "person and complaint": ["I want to talk to a real person"],
}


def prices_in_stock(app) -> set[int]:
    return {c.price for c in app.inventory.search(__import__("dealer_core.models", fromlist=["SearchFilters"]).SearchFilters(), 500)}


def spoken_prices(reply: str) -> set[int]:
    found = re.findall(r"\b(\d{2,3},\d{3}) dirhams\b(?! cheaper)|\bat (\d{2,3},\d{3})\b|\bis (\d{2,3},\d{3})\b(?! dirhams cheaper)", reply)
    return {int((a or b or c).replace(",", "")) for a, b, c in found}


@pytest.mark.parametrize("name", list(CALLS))
def test_a_call_holds_up_without_the_model(name):
    e, app = agent()
    stock = prices_in_stock(app)
    s = e.new_session()
    e.opening(s)
    replies = []
    for line in CALLS[name]:
        r = e.handle(s, line)
        replies.append((line, r.reply, r.debug.intent))
    problems = []
    for i, (line, reply, intent) in enumerate(replies):
        if not reply.strip():
            problems.append(f"silent after {line!r}")
            continue
        if i and reply == replies[i - 1][1] and intent != "repeat":
            problems.append(f"the same reply twice in a row after {line!r}: {reply!r}")
        q = re.findall(r"[^.!?]*\?", reply)
        prev_q = re.findall(r"[^.!?]*\?", replies[i - 1][1]) if i else []
        if i and intent != "repeat" and q and prev_q and q[-1].strip() == prev_q[-1].strip() and len(q[-1].split()) > 3:
            problems.append(f"the same question twice in a row after {line!r}: {q[-1].strip()!r}")
        bad = spoken_prices(reply) - stock
        if bad:
            problems.append(f"price(s) not of any car {sorted(bad)} after {line!r}")
        ends_ok = reply.rstrip().endswith("?") or intent in ("end_call", "handoff") or re.search(r"\b(?:goodbye|bye|you're booked|you're all set)\b", reply, re.I)
        if not ends_ok:
            problems.append(f"no question to answer after {line!r}: {reply!r}")
        if i and "anything that matches" in reply and spoken_prices(replies[i - 1][1]):
            problems.append(f"'nothing matches' right after a list, after {line!r}")
    transcript = "\n".join(f"  YOU: {l}\n  WAZIR: {r}" for l, r, _ in replies)
    assert not problems, "\n".join(problems) + "\n\n" + transcript

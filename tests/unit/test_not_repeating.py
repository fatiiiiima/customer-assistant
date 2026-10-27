"""27 Oct, Chrome, the Macan: after "Yes, talk about the Macan" four replies in a row opened with the same sentence ("it offers sports-car handling in a practical SUV
body") and every one ended with a different invitation to visit, including after the caller said "No, I want to know something else about the Macan".
Now: a new group of facts each time, no invitation after a no, and an invitation at most every third answer."""
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from dealer_core.agent import answers as A
from dealer_core.agent import wording as W
from dealer_core.agent.engine import AgentEngine
from dealer_core.app import ROOT, build_app
from dealer_core.config import load_config
from dealer_core.dialogue.session import Session
from dealer_core.dialogue.states import Intent
from dealer_core.nlu.backend import IntentResult
from dealer_core.shared.state import restore, snapshot
from dealer_core.site.details import CarDetails

SITE = next((p for p in (ROOT.parent / "bmw-portal" / "site" / "data.js", Path("C:/dev/bmw-portal/site/data.js")) if p.exists()), None)
NOW = datetime(2026, 10, 27, 11, 0, tzinfo=ZoneInfo("Asia/Dubai"))
INVITING = ("test drive", "come and see", "visit", "in person", "take a look", "firsthand", "behind the wheel", "book")


def macan() -> CarDetails:
    return CarDetails("AG-130", "Macan S", "Black", 5, 340, 460, "3.0L V6 Twin-Turbo", "AWD", 5.2, 254, "9.8 L/100km", 500, "Black leather", 2,
                      ("Sport Chrono", "PASM suspension", "Bose surround", "Panoramic roof"), "Sports-car handling in a practical SUV body.", "8-speed automatic", 5,
                      "GCC spec", "accident-free", "full dealer service history", "6 months or 10,000 km", 149)


def invites(text: str) -> bool:
    return any(w in text.lower() for w in INVITING)


# ------------------------------------------------------------------------------------------------ facts are told once
def test_each_request_for_more_gives_new_facts_until_none_are_left():
    d, told, spoken = macan(), [], []
    for _ in range(6):
        spoken.append(A.next_facts(d, told))
    assert spoken[0][0].startswith("It offers sports-car handling")                      # the blurb first
    flat = [s for group in spoken for s in group]
    assert len(flat) == len(set(flat)) and sum(1 for s in flat if "sports-car handling" in s) == 1
    assert set(told) == set(A.FACT_GROUPS) and spoken[-1] == []                         # everything told, then nothing: never a repeat


def test_a_specific_question_marks_its_group_as_told():
    told = []
    A.mark_said(told, ["engine", "power", "boot"])
    assert told == ["performance", "practical"]
    assert all("horsepower" not in s and "litre boot" not in s for s in A.next_facts(macan(), told, limit=8))


def test_the_preferred_group_comes_first():
    assert "accident-free" in A.next_facts(macan(), [], prefer=("condition",), limit=1)[0]


def test_no_details_means_nothing_to_say():
    assert A.next_facts(None, []) == []


# ------------------------------------------------------------------------------------------------ invitations
def test_an_invitation_is_made_at_most_every_third_answer():
    s, said = Session(), []
    for _ in range(9):
        said.append(W.offer_test_drive(s))
        s.history += [("customer", "tell me more"), ("agent", said[-1])]                # one exchange
    assert [invites(x) for x in said] == [True, False, False, True, False, False, True, False, False]


def test_nothing_invites_while_a_no_is_remembered_and_it_expires():
    s = Session()
    s.no_invite_until = len(s.history) + 12
    for _ in range(6):
        assert not invites(W.offer_test_drive(s))
        s.history += [("customer", "x"), ("agent", "y")]
    assert invites(W.offer_test_drive(s))                                                # six exchanges later


def test_one_car_search_asks_more_or_book_only_when_allowed():
    s = Session()
    assert "book a test drive" in W.hear_more_or_book(s)
    assert "book" not in W.hear_more_or_book(s)                                          # straight after: no


def test_the_memory_survives_a_worker_change():
    s = Session()
    s.no_invite_until, s.said_about = 14, {"AG-130": ["about", "condition"]}
    fresh = Session()
    eng = type("E", (), {})()
    restore(snapshot(s), fresh, eng)
    assert fresh.no_invite_until == 14 and fresh.said_about == {"AG-130": ["about", "condition"]}


def test_the_phrases_for_more_are_recognised():
    for text in ("tell me more about it", "what else can you tell me", "what else should I know", "tell me something else", "anything else you can tell me about it"):
        assert "about" in A.topics_in(text), text
    for text in ("what colours do you have", "how much is it", "book a test drive"):
        assert "about" not in A.topics_in(text), text


# ------------------------------------------------------------------------------------------------ the transcript
class Scripted:
    name = "scripted"

    def classify(self, *a, **k):
        return IntentResult(Intent.SEARCH_CARS, 1.0, self.name, "{}")

    def lead_in(self, *a, **k):
        return "none"

    def lead_in_stream(self, *a, **k):
        yield "ok."


def talk(e, s, text):
    out = list(e.handle_stream(s, text))
    return out[-1].debug.intent, " ".join(x for x in out if isinstance(x, str))


@pytest.fixture(scope="module")
def site_engine():
    if SITE is None:
        pytest.skip("the website is not on this machine")
    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=SITE, classifier=Scripted(), reply_mode="conversational", clock=lambda: NOW.astimezone(timezone.utc), lead_backend=Scripted())
    return AgentEngine(app.service, Scripted(), app.engine.reply_writer)


def test_the_27_oct_macan_conversation_has_no_repeats_and_no_pushing(site_engine):
    e = site_engine
    s = e.new_session()
    talk(e, s, "Porsche.")
    replies = [talk(e, s, t)[1] for t in ("Yes, talk about the Macan.", "I do want to know, tell me more about it.", "Is there anything else you can tell me about it?",
                                           "No, I want to know something else about the Macan.", "No, I want to know something else about the Macan.")]
    sentences = [x.strip() for r in replies for x in r.replace("?", ".").split(".") if x.strip()]
    facts = [x for x in sentences if not x.lower().startswith(("would you", "want to", "shall i", "what else", "anything else", "is there anything"))]
    assert len(facts) == len(set(facts)), facts                                          # not one sentence told twice
    assert sum("sports-car handling" in r for r in replies) == 1
    after_no = replies[3:]
    assert not any(invites(r) for r in after_no)                                          # the caller said No: no invitation in the next answers
    assert sum(invites(r) for r in replies) <= 2


def test_what_else_after_a_makes_overview_and_a_car_is_about_the_car(site_engine):
    e = site_engine
    s = e.new_session()
    talk(e, s, "Porsche.")
    talk(e, s, "What are the nine other makes?")
    talk(e, s, "Porsche.")
    talk(e, s, "tell me more")
    tool, said = talk(e, s, "what else")
    assert tool == "get_car" and "other makes" not in said and "Besides" not in said


def test_a_direct_reliability_question_is_answered_each_time(site_engine):
    e = site_engine
    s = e.new_session()
    talk(e, s, "Porsche.")
    talk(e, s, "Yes, talk about the Macan.")
    first = talk(e, s, "is it reliable?")[1]
    second = talk(e, s, "is it reliable?")[1]
    assert "accident-free" in first and "accident-free" in second


def test_a_yes_to_a_visit_invitation_still_starts_a_booking(site_engine):
    e = site_engine
    s = e.new_session()
    talk(e, s, "Porsche.")
    talk(e, s, "Yes, talk about the Macan.")
    s.no_invite_until = 0
    _, said = talk(e, s, "tell me more")
    s.history.append(("agent", "Would you like to come and see it?"))                    # an invitation was made
    tool, said = talk(e, s, "yes please")
    assert tool == "prepare_booking"


def test_a_no_to_a_visit_silences_invitations_for_the_next_exchanges_on_its_own(site_engine):
    """The every-third-answer rule alone would invite again within three answers; only remembering the No keeps it quiet for six."""
    e = site_engine
    s = e.new_session()
    talk(e, s, "Porsche.")
    talk(e, s, "Yes, talk about the Macan.")
    s.no_invite_until = 0                                                              # the cadence allows an invitation right now
    s.history.append(("agent", "Would you like to come and see it?"))                  # ...and one has just been made
    talk(e, s, "no thanks, what colour is it")
    assert s.no_invite_until >= len(s.history) + 6
    later = [talk(e, s, "what colour is it")[1] for _ in range(5)]
    assert not any(invites(r) for r in later), later


def test_a_no_to_something_that_was_not_an_invitation_is_not_a_declined_visit(site_engine):
    e = site_engine
    s = e.new_session()
    talk(e, s, "Porsche.")
    talk(e, s, "Yes, talk about the Macan.")
    s.no_invite_until = 0
    s.history.append(("agent", "Which one would you like to hear more about?"))
    talk(e, s, "no, what colour is it")
    assert s.no_invite_until < len(s.history) + 6                                       # no decline recorded; only the normal spacing


def test_asking_to_visit_lifts_the_silence(site_engine):
    e = site_engine
    s = e.new_session()
    talk(e, s, "Porsche.")
    talk(e, s, "Yes, talk about the Macan.")
    s.no_invite_until = len(s.history) + 12
    talk(e, s, "when can I come and see it")
    assert s.no_invite_until == 0 or s.no_invite_until <= len(s.history) + 6

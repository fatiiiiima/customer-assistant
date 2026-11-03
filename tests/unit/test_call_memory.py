"""A call is remembered until the visitor closes the browser (owner, 3 Nov: "when the bot ends the call, it remembers the entire [conversation] until they close the browser, and not
only 15 mins"). The server keeps it a day; the page forgets it with the tab; and a call that ENDED is open again when the visitor returns, with its conversation."""
from dealer_core.config import load_config
from dealer_core.dialogue.session import Session
from dealer_core.dialogue.states import State
from dealer_core.shared.state import CallStore, MemoryStore, reopen, restore, snapshot

from .test_live_call_fixes import said
from .test_scale import agent


def test_the_server_keeps_a_call_a_day_not_fifteen_minutes():
    assert load_config(environ={}).voice.calls_kept_secs == 86400.0
    assert load_config(environ={"VA__VOICE__CALLS_KEPT_SECS": "900"}).voice.calls_kept_secs == 900.0       # still settable


def test_the_store_is_given_that_time_when_a_call_is_saved():
    seen = {}

    class Spy(MemoryStore):
        def set(self, key, value, ttl_s=None):
            seen[key] = ttl_s
            super().set(key, value, ttl_s)

    calls = CallStore(Spy(), ttl_s=load_config(environ={}).voice.calls_kept_secs)
    e, _ = agent(None)
    calls.save("abc", e.new_session())
    assert seen["call:abc"] == 86400.0


def test_a_call_the_agent_ended_is_open_again_with_the_whole_conversation_when_the_visitor_comes_back():
    e, _ = agent(None)
    s = e.new_session()
    said(e, s, "do you have any Toyotas")
    said(e, s, "tell me about the first one")
    r = said(e, s, "that's all, thanks, bye")
    assert s.ended and s.state is State.GOODBYE and r.ended
    said_before = list(s.history)
    car, shown = s.car, [c.stock_id for c in s.shown]
    calls = CallStore(MemoryStore(), ttl_s=86400.0)
    calls.save("abc", s)                                              # the call is saved as it ended
    back = e.new_session()                                            # the visitor returns on a new connection
    assert calls.load("abc", back, e.svc) and back.ended              # restored as it was: ended ...
    assert reopen(back) is True
    assert not back.ended and back.handoff is None and back.state is not State.GOODBYE      # ... and open again
    assert back.history == said_before and back.car == car and [c.stock_id for c in back.shown] == shown      # nothing forgotten
    r = said(e, back, "what is the mileage")                          # and it answers, instead of saying "Goodbye." to everything
    assert r.reply != "Goodbye." and not r.ended


def test_a_call_that_ended_without_a_car_comes_back_to_interest_and_one_that_did_not_end_is_left_alone():
    s = Session()
    s.ended, s.state = True, State.GOODBYE
    assert reopen(s) is True and s.state is State.INTEREST and not s.ended
    s2 = Session()
    s2.state = State.SLOT
    assert reopen(s2) is False and s2.state is State.SLOT


def test_a_booking_in_progress_survives_the_end_and_the_return():
    e, _ = agent(None)
    s = e.new_session()
    s.draft_name, s.draft_phone, s.state, s.ended = "Fatima", "0501234567", State.GOODBYE, True
    back = Session()
    restore(snapshot(s), back, e.svc)
    reopen(back)
    assert back.draft_name == "Fatima" and back.draft_phone == "0501234567"


def test_the_page_forgets_the_call_with_the_tab_and_has_no_time_limit():
    from pathlib import Path

    js = Path(__file__).resolve().parents[2].parent / "bmw-portal" / "site" / "voice.js"
    if not js.exists():
        import pytest
        pytest.skip("the website repository is not next to this one")
    src = js.read_text()
    assert 'sessionStorage.getItem("wzLastCall")' in src and 'sessionStorage.setItem("wzLastCall"' in src
    assert "localStorage.getItem(\"wzLastCall\")" not in src and "KEEP_MS" not in src
    assert "within 15 minutes" not in src and "until you close this tab" in src                  # what the visitor is told after a quiet spell

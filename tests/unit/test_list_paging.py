"""'Go through the list' goes on through the stock instead of reading the same first cars again (3 Nov live call: 74 cars, "go through the list" and "keep listing the
cars, please" both got the same three cars). ARCH_REVIEW section 93."""
import pytest

from .test_live_call_fixes import said
from .test_model_only import Model
from .test_scale import agent

OFFER = "Which one would you like to hear more about, or shall I go through the rest?"


class Lister(Model):
    """Decides search_cars every time and writes a reply that ends on the offer to go on, as the live model did."""

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
        self.writes.append(user)
        yield "Here are some cars. "
        yield OFFER


def lister(rules=False):
    b = Lister(["search_cars"] * 12)
    e, _ = agent(None, backend=b)
    e.model_only = e.reply_writer.model_only = True
    e.silent_fallback = False
    e.rules_decide = rules
    s = e.new_session()
    said(e, s, "all the cars")                       # too many to read out: asked to narrow
    said(e, s, "you have")                            # the first three
    return e, s, b


def ids(s):
    return [c.stock_id for c in s.shown]


@pytest.mark.parametrize("rules", [False, True])
def test_go_through_the_list_moves_on_each_time_and_never_repeats_a_car(rules):
    e, s, b = lister(rules)
    seen = ids(s)
    assert len(seen) == 3
    for line in ("go through the list", "keep listing the cars please", "go through the list"):
        said(e, s, line)
        assert ids(s) and not set(ids(s)) & set(seen), (line, ids(s), seen)           # new cars only
        seen += ids(s)
    assert len(seen) == len(set(seen)) and len(seen) >= 9


def test_the_code_picks_this_without_the_model_when_rules_decide_is_on():
    e, s, b = lister(rules=True)
    asked = len(b.prompts)
    r = said(e, s, "keep listing the cars, please")
    assert r.debug.via == "agent:rules" and r.debug.intent == "search_cars" and len(b.prompts) == asked      # no decision call


def test_the_facts_say_how_many_were_read_out_and_how_many_are_left():
    e, s, b = lister()
    said(e, s, "go through the list")
    assert "already read out earlier in this list" in b.writes[-1] and "more are still left" in b.writes[-1]


def test_at_the_end_of_the_list_it_says_so_instead_of_repeating():
    e, s, b = lister()
    for _ in range(6):
        said(e, s, "go through the list")
    r = said(e, s, "go through the list")
    assert "none are left" in b.writes[-1] and "whole list" in b.writes[-1]
    assert r.debug.intent == "search_cars"


def test_a_new_question_starts_the_list_again():
    e, s, b = lister()
    said(e, s, "go through the list")
    said(e, s, "do you have any Toyotas")             # names something new: not "more of the same"
    assert s.excluded <= set(ids(s))                   # nothing kept from the earlier list: at most what this answer read out


def test_it_only_means_the_list_when_the_agent_offered_to_go_on():
    b = Lister(["search_cars"] * 4)
    e, _ = agent(None, backend=b)
    e.model_only = e.reply_writer.model_only = True
    e.silent_fallback = False
    s = e.new_session()
    s.shown = []                                       # nothing has been read out
    assert e._continues_list(s, "go through the list") is False
    said(e, s, "you have")
    assert e._continues_list(s, "go through the list") is True and e._continues_list(s, "keep listing the cars please") is True
    assert e._continues_list(s, "go through the Toyotas") is False               # a make: a new search
    assert e._continues_list(s, "I'm looking for a red car under 50000") is False
    s.remember("agent", "The Camry is white. Want to book a test drive?")
    assert e._continues_list(s, "go through the list") is False                  # the agent's last line did not offer a list


def test_tell_me_more_and_the_next_one_are_not_a_request_to_go_on_with_the_list():
    b = Lister(["search_cars"] * 4)
    e, _ = agent(None, backend=b)
    e.model_only = e.reply_writer.model_only = True
    e.silent_fallback = False
    s = e.new_session()
    said(e, s, "you have")
    for line in ("tell me more", "tell me more about it", "the next one", "what else"):
        assert e._continues_list(s, line) is False, line

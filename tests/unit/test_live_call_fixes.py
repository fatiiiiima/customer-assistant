"""Lines from the live test call of 27 Oct 2026 (agent engine, website voice) that went wrong, and the fixes:
  1. colour questions about the cars went to the model, which answered with the makes, a count, and the showroom address;
  2. a model reply stopped mid-list left a clause hanging ("We've got the Honda Civic, Toyota Corolla, Is there anything else ...");
  3. a turn cut off mid-phrase ("I'm looking for a...", heard once as "I'm looking forward.") got a 3 s model guess instead of a quick "say again"."""
from dealer_core.agent.engine import AgentEngine
from dealer_core.dialogue.states import State
from dealer_core.dialogue.writer import ReplyWriter, WriterInput, split_point

from .test_scale import NEEDS_THE_MODEL, agent


def said(e: AgentEngine, s, text: str):
    out = list(e.handle_stream(s, text))
    return out[-1]


# ------------------------------------------------------------------------------------------------ 1. colours
def test_what_colours_are_these_cars_names_each_listed_car_with_its_colour():
    e, b = agent(None)
    s = e.new_session()
    said(e, s, "Actually, I want a sedan.")
    cars = list(s.shown)
    assert len(cars) >= 2
    for line in ("What colors are they?", "Yeah, but what colors are these cars?", "I want to know what colors these cars are."):
        asked = len(b.prompts)
        r = said(e, s, line)
        assert r.debug.intent == "colours" and len(b.prompts) == asked, line      # decided by code, no model call
        for c in cars[:4]:
            assert f"the {c.model.split()[0].lower()}" in r.reply.lower() or c.model in r.reply
            assert c.color in r.reply
        assert "showroom" not in r.reply.lower() and r.reply.endswith("?")


def test_what_colours_do_you_have_gives_the_colours_in_stock_most_common_first():
    e, _ = agent(None)
    s = e.new_session()
    r = said(e, s, "What colours do you have of different cars?")
    stock = e.svc.inventory.search(__import__("dealer_core.models", fromlist=["SearchFilters"]).SearchFilters(), 500)
    counts = {}
    for c in stock:
        counts[c.color] = counts.get(c.color, 0) + 1
    top = max(counts, key=counts.get)
    assert r.debug.intent == "colours" and top in r.reply and "Is there a colour you'd like?" in r.reply
    assert not any(make in r.reply for make in ("Toyota", "Nissan", "Honda"))      # colours, not the makes


def test_asking_what_colours_you_have_right_after_a_list_is_about_the_stock_not_the_list():
    e, _ = agent(None)
    s = e.new_session()
    said(e, s, "Actually, I want a sedan.")
    assert len(s.shown) >= 2 and not s.car
    assert said(e, s, "What colours do you have of different cars?").reply.startswith("Our cars come in")
    assert said(e, s, "what colours are they").reply.startswith("The ")                  # pointing at the list: the list


def test_the_colours_of_one_make_and_the_colour_of_the_car_in_focus():
    e, _ = agent(None)
    s = e.new_session()
    r = said(e, s, "What colours do the Toyotas come in?")
    toyota = {c.color for c in e.svc.inventory.search(__import__("dealer_core.models", fromlist=["SearchFilters"]).SearchFilters(make="toyota"), 50)}
    others = {c.color for c in e.svc.inventory.search(__import__("dealer_core.models", fromlist=["SearchFilters"]).SearchFilters(), 500)} - toyota
    assert r.debug.intent == "colours" and all(c in r.reply for c in toyota) and not any(f" {c}" in r.reply for c in others)
    said(e, s, "tell me about the Land Cruiser")
    assert s.car == "S1042"
    assert said(e, s, "What colour is it?").debug.intent == "get_car"                # one car: its own answer, as before


def test_a_colour_named_is_still_a_search_not_a_colour_question():
    e, _ = agent(None)
    s = e.new_session()
    assert said(e, s, "do you have it in red").debug.intent != "colours"


# ------------------------------------------------------------------------------------------------ 3. cut off mid-phrase
def test_a_turn_cut_off_mid_phrase_is_asked_again_at_once_by_code():
    e, b = agent(None)
    s = e.new_session()
    for line in ("I'm looking for a", "I'm looking for a...", "I'm looking forward.", "show me the", "something under"):
        asked = len(b.prompts)
        r = said(e, s, line)
        assert len(b.prompts) == asked and r.debug.note == "cut off mid-phrase: asked again", line
        assert r.reply.startswith("Sorry") and r.reply.endswith("?")


def test_complete_lines_and_booking_details_are_not_taken_for_cut_offs():
    e, _ = agent(None)
    s = e.new_session()
    for line in ("I'm looking for a Toyota sedan.", "no I don't need any", "I'm looking forward to it"):
        assert said(e, s, line).debug.note != "cut off mid-phrase: asked again", line
    s.state = State.CONTACT                                                     # a name or number can end anywhere
    assert not AgentEngine._unfinished(s, ["my", "name", "is", "fatima", "and"])


# ------------------------------------------------------------------------------------------------ 2. nothing left hanging
def test_only_the_first_piece_of_a_reply_may_be_a_clause():
    text = "We've got the Honda Civic, Toyota Corolla, and the Camry."
    assert split_point("I see you're after a sedan for the desert, so here goes")          # first piece: a 6+ word clause may go early
    streamed = text + " Which"
    assert streamed[:split_point(streamed, clauses=False)[0]].strip() == text            # later: only at the end of the sentence
    assert split_point("We've got the Honda Civic, Toyota Corolla, and", clauses=False) is None   # an unfinished sentence: nothing yet


def test_a_reply_stopped_mid_list_leaves_no_clause_hanging():
    class StopsMidList:
        def lead_in_stream(self, system, user, max_tokens, temperature, deadline):
            yield "I see you're after a sedan for the desert. "
            yield "We've got the Honda Civic, Toyota Corolla, "                # the model stops here (deadline, or the next words fail a check)

    w = ReplyWriter(StopsMidList())
    out = list(w.stream(WriterInput("I'm looking for a", "Which one would you like to hear more about?", "low", [],
                                    "Cars: Honda Civic, Toyota Corolla", True)))
    assert not any(x.rstrip().endswith(",") for x in out)
    assert not any("Corolla" in x for x in out)                                 # the unfinished list is not spoken at all
    assert out[-1].endswith("?")


# ------------------------------------------------------------------------------------------------ 4. "Yeah" taken as the caller's name
def test_yes_to_come_and_see_it_is_not_a_name_and_the_name_is_then_asked_for():
    from dealer_core.nlu import extractors as ex

    assert ex.contact("Yeah, I want to come and see it.").name is None
    for opener in ("Yeah", "Totally"):                                       # "Totally" is not a stop word: the booking step itself must refuse it
        e, _ = agent(None)
        s = e.new_session()
        said(e, s, "tell me about the Land Cruiser")
        said(e, s, f"{opener}, I want to come and see it.")
        said(e, s, "Tomorrow?")
        r = said(e, s, "2 in the afternoon.")
        assert s.draft_name is None, opener
        assert "name" in r.reply.lower(), r.reply                            # so the agent asks for it


def test_a_name_without_a_cue_is_still_taken_when_it_was_asked_for_or_comes_with_the_number():
    e, _ = agent(None)
    s = e.new_session()
    said(e, s, "tell me about the Land Cruiser")
    said(e, s, "I want to come and see it tomorrow at 2 PM")
    said(e, s, "Fatima, 050 793 2546")
    assert s.draft_name == "Fatima" and s.read_back


# ------------------------------------------------------------------------------------------------ 5. "Sure, The 2018 Honda Civic ..."
def test_after_an_opener_the_next_word_is_lower_case_unless_it_is_a_name():
    from dealer_core.agent.engine import _after_opener, _lower_first

    proper = {"toyota", "land", "cruiser", "honda", "civic"}
    assert _after_opener("The 2018 Honda Civic is red.", proper) == "the 2018 Honda Civic is red."      # the model did not start with "Sure,"
    assert _lower_first("Besides Toyota, we've got 20 more cars.", proper) == "besides Toyota, we've got 20 more cars."
    assert _lower_first("Sharp styling and a punchy turbo engine.", proper) == "sharp styling and a punchy turbo engine."
    for name in ("Toyota has two in stock.", "Land Cruisers hold their value.", "I can check that.", "AGMC cars are inspected.",
                 "SUVs are popular here.", "BMWs hold up well.", "Monday works too.", "Dubai traffic is light then."):
        assert _lower_first(name, proper) == name
    assert _after_opener("Sure, we've got 4 SUVs.", proper, new_sentence=True) == "We've got 4 SUVs."   # after "Let me check." a new sentence


def test_an_opener_then_the_models_sentence_reads_as_one_sentence():
    from dealer_core.agent.governor import QualityGovernor

    from .test_scale import Scripted

    class Describes(Scripted):
        def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
            yield "The 2021 Toyota Land Cruiser is white with 62,000 kilometres. Want to book a test drive?"

    e, _ = agent(None, backend=Describes())
    e.governor, e.openers = QualityGovernor("full"), True
    s = e.new_session()
    said(e, s, "do you have any SUVs")
    out = list(e.handle_stream(s, "the Land Cruiser"))[:-1]
    assert out[0].endswith(",") and out[1].startswith("the 2021 Toyota Land Cruiser"), out


# ------------------------------------------------------------------------------------------------ 6. "anything else about the car?" taken as "book it"
def site_details(stock_id: str, blurb: str):
    from dealer_core.site.details import CarDetails

    return CarDetails(stock_id=stock_id, full_model="Land Cruiser GXR", colour_name="Pearl White", seats=7, power_hp=275, torque_nm=439,
                      engine="4.0L V6", drive="4WD", accel_0_100=8.5, top_speed=180, consumption="12.5 L/100km", boot_litres=909, interior="Beige leather",
                      owners=1, features=("Reverse camera", "Sunroof"), blurb=blurb, transmission="automatic", doors=5, spec="GCC specs",
                      accident="accident-free", service="full dealer service history", warranty="12 months", inspection=150)


def test_more_about_the_car_in_focus_is_answered_by_code_from_its_details():
    for line in ("Is there anything else about the car that you'll know?", "Is there anything else about the car that I need to know?",
                 "what else should I know about it", "tell me more about it", "anything else I should know", "I'd like to know more"):
        e, b = agent(None)
        e.svc.site.cars["S1042"] = site_details("S1042", "A desert-proof legend with proper go-anywhere presence")
        s = e.new_session()
        said(e, s, "tell me about the Land Cruiser")
        asked = len(b.prompts)
        r = said(e, s, line)
        assert r.debug.intent == "get_car" and len(b.prompts) == asked and s.car == "S1042", line
        assert "It's a desert-proof legend" in r.reply and "day" not in r.reply.lower(), (line, r.reply)


def test_what_else_without_a_car_in_focus_is_still_the_rest_of_the_stock():
    e, _ = agent(None)
    s = e.new_session()
    assert said(e, s, "what else do you have").debug.intent == "browse_makes"
    said(e, s, "tell me about the Land Cruiser")
    assert said(e, s, "tell me more about the Nissan Patrol").debug.intent != "get_car" or s.car != "S1042"   # another car: not this one


def test_a_website_description_is_said_as_a_sentence():
    from dealer_core.agent.answers import blurb_sentence

    assert blurb_sentence("A thunderous V8 super-SUV with proper Range Rover presence.") == "It's a thunderous V8 super-SUV with proper Range Rover presence."
    assert blurb_sentence("Sharp styling and a punchy turbo engine in an affordable, reliable package") == \
        "It offers sharp styling and a punchy turbo engine in an affordable, reliable package."
    assert blurb_sentence("V8 power and seven seats") == "It offers V8 power and seven seats."
    assert blurb_sentence("Sharp looks and a big boot") == "It offers sharp looks and a big boot."
    assert blurb_sentence("This Patrol is built for the dunes.") == "This Patrol is built for the dunes."          # already a sentence

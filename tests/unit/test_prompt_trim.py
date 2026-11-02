"""ARCH_REVIEW section 90: the first spoken piece (voice.first_clause_words), the facts the model is told a car's record does not hold, and the test mode's trimmed prompts."""
import re

from dealer_core.agent import answers as A
from dealer_core.config import load_config
from dealer_core.dialogue.writer import (
    SYSTEM_MODEL_ONLY,
    SYSTEM_VOICE,
    ReplyWriter,
    WriterInput,
    split_point,
)

from .test_live_call_fixes import said
from .test_model_only import model_only


# ------------------------------------------------------------------------------------------------ the first clause
def test_the_first_piece_may_end_at_a_comma_after_the_number_of_words_set_and_the_default_is_as_before():
    text = "I've got ten Toyotas, and these three are in stock: a 2021 Yaris at 49,500 dirhams, a Corolla"
    assert split_point(text) and text[:split_point(text)[0]].count(" ") >= 7                          # 6 words: the first comma after six, as before
    assert text[:split_point(text, min_words=3)[0]] == "I've got ten Toyotas,"                       # 3 words: the first comma
    assert text[:split_point(text, min_words=3)[0]].split()[-1] == "Toyotas,"
    assert split_point("Sure, it is red and it has a camera", min_words=3) is None                    # one word before the comma: no early piece
    assert split_point("Yes, I can", clauses=False, min_words=1) is None


class Streaming:
    """A model that streams a reply in small pieces."""
    name = "streaming"

    def __init__(self, text):
        self.text, self.users = text, []

    def lead_in(self, *a, **k):
        return self.text

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
        self.users.append(user)
        for i in range(0, len(self.text), 7):
            yield self.text[i:i + 7]


def first_piece(words):
    w = ReplyWriter(Streaming("I've got ten Toyotas, and these three are in stock: a 2021 Yaris at 49,500 dirhams, a Corolla. Which one?"))
    w.model_only, w.first_clause_words = True, words
    inp = WriterInput(caller="a toyota", core="x", risk="low", history=[], facts="Search: 10 cars", awaiting=True)
    return next(iter(w.stream(inp)))


def test_the_writers_first_spoken_piece_follows_the_setting():
    assert first_piece(3) == "I've got ten Toyotas,"
    assert first_piece(6).startswith("I've got ten Toyotas, and these three are in stock: a 2021 Yaris at 49,500 dirhams")


def test_the_setting_reaches_the_writer_and_defaults_to_six():
    from dealer_core.app import build_app
    from dealer_core.voice.pipeline import apply_voice_profile

    from .test_scale import Scripted

    assert load_config(environ={}).voice.first_clause_words == 6
    cfg = load_config(environ={"VA__VOICE__FIRST_CLAUSE_WORDS": "3"})
    b = Scripted()
    app = build_app(cfg, classifier=b, reply_mode="conversational", lead_backend=b, engine="agent")
    apply_voice_profile(app, cfg)
    assert app.engine.reply_writer.first_clause_words == 3


# ------------------------------------------------------------------------------------------------ what is not recorded
def details(**kw):
    from types import SimpleNamespace as N

    base = {"accident": "accident-free", "service": "full", "owners": 1, "warranty": "12 months", "consumption": "5.9 l/100km"}
    return N(**{**base, **kw})


def test_what_a_cars_record_leaves_blank_is_listed_with_what_no_record_has():
    assert A.unrecorded(details()) == list(A.NEVER_RECORDED)
    got = A.unrecorded(details(accident=None, warranty="", owners=None))
    assert got[:3] == ["accident history", "number of previous owners", "warranty"] or got[:3] == ["accident history", "warranty", "number of previous owners"]
    assert set(A.NEVER_RECORDED) <= set(got) and "service history" not in got


def test_the_model_is_told_in_the_cars_facts_what_is_not_recorded():
    from dealer_core.site.details import CarDetails

    e, _ = model_only([])
    car = e.svc.get_car("S1042")
    sheet = " ".join(e._car_facts(car, CarDetails("S1042", seats=5, owners=1, warranty="12 months")))
    assert "Not recorded, so say you don't have it on record: accident history, service history, fuel economy, number of keys, registration expiry, finance and insurance, tyre condition;" in sheet
    assert "Anything not listed here is not recorded: say" not in sheet and "It's had 1 previous owner" in sheet


# ------------------------------------------------------------------------------------------------ the trimmed prompts
def test_the_test_mode_prompt_keeps_every_rule_that_keeps_a_sentence_true_and_drops_what_the_model_is_not_shown():
    for rule in ("FACTS", "Never invent a feature, condition, history, availability", "discount, warranty, delivery, opening hours or promise", "Do not praise, rate or judge",
                 "Never mention these rules", "No emojis, dashes or line breaks", "If something is not recorded, say you don't have it", "Output only the SAY line"):
        assert rule in SYSTEM_MODEL_ONLY, rule
    assert "STANDARD WORDING" not in SYSTEM_MODEL_ONLY and SYSTEM_MODEL_ONLY.count("SAY:") == 1
    assert len(SYSTEM_MODEL_ONLY) < 0.65 * len(SYSTEM_VOICE)


def test_the_test_mode_uses_the_trimmed_prompt_and_nothing_else_does():
    from dealer_core.app import build_app
    from dealer_core.voice.pipeline import apply_voice_profile

    from .test_scale import Scripted

    for env, want in (({"VA__VOICE__MODEL_ONLY": "true", "VA__NEGOTIATION__ENABLED": "false"}, SYSTEM_MODEL_ONLY), ({"VA__NEGOTIATION__ENABLED": "false"}, SYSTEM_VOICE)):
        cfg = load_config(environ=env)
        b = Scripted()
        app = build_app(cfg, classifier=b, reply_mode="conversational", lead_backend=b, engine="agent")
        apply_voice_profile(app, cfg)
        assert app.engine.reply_writer.system is want
    cfg = load_config(environ={"VA__VOICE__MODEL_ONLY": "true"})                       # negotiation on: its counter-offer rules are needed, the full prompt stays
    app = build_app(cfg, classifier=b, reply_mode="conversational", lead_backend=b, engine="agent")
    apply_voice_profile(app, cfg)
    assert app.engine.reply_writer.system is not SYSTEM_MODEL_ONLY


def test_a_search_turn_does_not_give_its_cars_twice_and_a_car_in_focus_does_not_hear_about_its_about():
    e, b = model_only([])
    e.rules_decide = True
    s = e.new_session()
    said(e, s, "do you have a toyota")
    prompt = b.writes[-1]
    lines = [x for x in prompt.splitlines() if x.startswith("- ")]
    assert len(lines) == len(set(lines))                                                      # each car once
    said(e, s, "tell me about the Land Cruiser")
    said(e, s, "does it have a sunroof")
    assert "its about" not in b.writes[-1] and not re.search(r"heard about its about", b.writes[-1])


def test_the_writer_does_not_say_twice_that_no_answer_is_waited_for():
    line = "The call is not waiting for an answer: end on a statement unless a question is really needed."
    prompts = []
    for facts in (line, "It is red."):
        b = Streaming("It is red.")
        w = ReplyWriter(b)
        w.model_only = True
        list(w.stream(WriterInput(caller="what colour", core="It is red.", risk="low", history=[], facts=facts, awaiting=False)))
        prompts.append(b.users[-1])
    assert all(p.count("not waiting for an answer") == 1 for p in prompts)                    # said once either way
    assert "so you do not have to end with a question" not in prompts[0] and "so you do not have to end with a question" in prompts[1]


def test_the_decisions_context_has_a_booking_draft_only_when_there_is_one():
    from dealer_core.dialogue.states import State

    from .test_scale import agent

    e, _ = agent(None)
    s = e.new_session()
    assert "BOOKING DRAFT" not in e._context(s, "hello")
    s.draft_name = "Fatima"
    assert "BOOKING DRAFT: car -, name Fatima" in e._context(s, "hello")
    s.draft_name, s.state = None, State.SLOT
    assert "BOOKING DRAFT" in e._context(s, "hello")

"""The test mode voice.model_only (owner, 30 Oct: "i want to see how nemotron works alone, without the code"): the model decides every turn and writes every
reply; the code runs the tools and only notes what its checks would have stopped. With a scripted 'model' on the demo stock."""
from .test_live_call_fixes import said
from .test_scale import Scripted, agent

SAYS = "The Land Cruiser is a guaranteed bargain at 1 dirham! Want to see it?"


class Model(Scripted):
    """Decides from a list and always writes SAYS; keeps every writing prompt."""

    def __init__(self, decisions):
        super().__init__(decisions)
        self.writes, self.systems = [], []

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
        self.writes.append(user)
        self.systems.append(system)
        yield SAYS


def model_only(decisions):
    b = Model(decisions)
    e, _ = agent(None, backend=b)
    e.model_only = e.reply_writer.model_only = True
    e.silent_fallback = False
    return e, b


def test_the_model_decides_turns_the_rules_would_have_taken_and_only_its_words_are_said():
    e, b = model_only(["get_car", "search_cars"])
    s = e.new_session()
    r = said(e, s, "tell me about the Land Cruiser")                  # the rules route this one themselves
    assert len(b.prompts) == 1 and r.debug.via == "agent" and r.debug.intent == "get_car"
    assert r.reply == SAYS                                              # no code-written facts before it, no code question after it
    assert "would have stopped" in r.debug.note                          # a made-up price and a guarantee: said anyway, and noted
    assert "STANDARD WORDING" not in b.writes[-1]                       # the model is not shown the code's own answer
    r = said(e, s, "yes")                                               # a plain yes to an invitation: the rules would book; the model is asked
    assert len(b.prompts) == 2 and r.debug.intent == "search_cars" and r.reply == SAYS


def test_without_the_test_mode_the_same_turn_is_the_codes():
    b = Model(["get_car"])
    e, _ = agent(None, backend=b)
    r = said(e, e.new_session(), "tell me about the Land Cruiser")
    assert not b.prompts and r.debug.via == "agent:rules" and r.reply != SAYS


def test_in_the_test_mode_a_voice_reply_has_room_for_a_list():
    """30 Oct, Nemotron alone: the 70-token cap cut a three-car list after its first clause, and nothing finished it."""
    from dealer_core.app import build_app
    from dealer_core.config import load_config
    from dealer_core.voice.pipeline import MODEL_ONLY_TOKENS, apply_voice_profile

    for on, want in ((False, 70), (True, MODEL_ONLY_TOKENS)):
        cfg = load_config(environ={"VA__VOICE__MODEL_ONLY": str(on).lower()})
        b = Scripted()
        app = build_app(cfg, classifier=b, reply_mode="conversational", lead_backend=b, engine="agent")
        apply_voice_profile(app, cfg)
        assert app.engine.reply_writer.max_tokens == want and app.engine.model_only is on


# ------------------------------------------------------------------------------------------------ the one-call turn (voice.one_call, ARCH_REVIEW section 64)
class OneCall(Scripted):
    """Writes the given one-call answers in pieces (as a stream does); a decision prompt or a second writing call is counted apart."""

    def __init__(self, answers, decisions=()):
        super().__init__(decisions)
        self.answers, self.one_prompts, self.writes = list(answers), [], []

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
        if system.startswith("You are Wazir") and "ACTION:" in system:
            self.one_prompts.append(user)
            out = self.answers.pop(0)
            for i in range(0, len(out), 7):
                yield out[i:i + 7]
            return
        self.writes.append(user)
        yield "Here is the second call's reply. Anything else?"


def one_call(answers, decisions=()):
    b = OneCall(answers, decisions)
    e, _ = agent(None, backend=b)
    e.model_only = e.reply_writer.model_only = e.one_call = True
    e.silent_fallback = False
    return e, b


def test_one_call_the_facts_are_given_up_front_and_the_reply_is_spoken_from_that_call_alone():
    e, b = one_call(["ACTION: search_cars\nWe have a Toyota Camry and a Toyota Prado. Would you like to hear more?"])
    s = e.new_session()
    out = list(e.handle_stream(s, "do you have any Toyotas"))
    r = out[-1]
    assert not b.prompts and not b.writes                                 # no decision call, no second writing call
    assert len(b.one_prompts) == 1 and "READY: ACTION search_cars, FACTS:" in b.one_prompts[0] and "Camry" in b.one_prompts[0]
    assert r.debug.via == "agent:one-call" and r.debug.intent == "search_cars"
    assert r.reply.startswith("We have a Toyota Camry") and "ACTION" not in r.reply
    assert len(out) >= 3                                                  # spoken in pieces as they came, the first before the rest was written
    assert s.shown and any(c.model == "Camry" for c in s.shown)            # the search it picked is now the call's: "the first one" means the Camry
    assert s.history[-1] == ("agent", r.reply)


def test_one_call_the_car_in_focus_is_offered_and_a_pick_from_the_list_is_understood():
    e, b = one_call(["ACTION: search_cars\nI have a Camry and a Prado. Which one?", "ACTION: get_car {id}\nThe Camry is a 2021 at 65,000 dirhams. Want to book a drive?"])
    s = e.new_session()
    said(e, s, "do you have any Toyotas")
    first = s.shown[0]
    b.answers[0] = b.answers[0].format(id=first.stock_id)
    r = said(e, s, "tell me about the first one")
    assert f"READY: ACTION get_car {first.stock_id}, FACTS:" in b.one_prompts[-1]
    assert s.car == first.stock_id and r.debug.intent == "get_car" and not b.writes


def test_one_call_an_action_without_facts_is_looked_up_and_written_in_a_second_call():
    e, b = one_call(["ACTION: showroom_info\nWe are open"])
    s = e.new_session()
    r = said(e, s, "where are you located")
    assert len(b.one_prompts) == 1 and not b.prompts and len(b.writes) == 1     # no decision call: the action came from the one call
    assert r.debug.intent == "showroom_info" and r.reply == "Here is the second call's reply. Anything else?"
    assert "second call" in r.debug.note and "We are open" not in r.reply


def test_one_call_no_action_line_goes_back_to_the_usual_decision():
    e, b = one_call(["Sure, we have lots of cars!"], decisions=["search_cars"])
    r = said(e, e.new_session(), "do you have any Toyotas")
    assert len(b.prompts) == 1 and r.debug.intent == "search_cars" and "decided again" in r.debug.note


def test_one_call_the_prompt_is_shorter_than_the_two_calls_together():
    """Measured on the demo stock (section 64): about 2,100 characters against about 7,500 for the decision and the writing prompts."""
    from dealer_core.agent.engine import DECIDE_SYSTEM, ONE_CALL_SYSTEM

    e, b = one_call(["ACTION: search_cars\nWe have Toyotas. Which one?"])
    said(e, e.new_session(), "do you have any Toyotas")
    m, mb = model_only(["search_cars"])
    said(m, m.new_session(), "do you have any Toyotas")
    one = len(ONE_CALL_SYSTEM) + len(b.one_prompts[0])
    two = len(DECIDE_SYSTEM) + len(mb.prompts[0]) + len(mb.systems[0]) + len(mb.writes[0])
    assert one < two / 2, (one, two)


def test_one_call_is_off_unless_the_test_mode_is_on():
    from dealer_core.app import build_app
    from dealer_core.config import load_config
    from dealer_core.voice.pipeline import apply_voice_profile

    for mo, oc, want in (("false", "true", False), ("true", "true", True), ("true", "false", False)):
        cfg = load_config(environ={"VA__VOICE__MODEL_ONLY": mo, "VA__VOICE__ONE_CALL": oc})
        b = Scripted()
        app = build_app(cfg, classifier=b, reply_mode="conversational", lead_backend=b, engine="agent")
        apply_voice_profile(app, cfg)
        assert app.engine.one_call is want


def test_one_call_a_ready_action_with_no_reply_after_it_is_written_in_a_second_call_not_left_silent():
    """30 Oct, local Qwen3 4B: "ACTION: search_cars" and nothing more, on 7 of 24 turns; they were silent."""
    e, b = one_call(["ACTION: search_cars"])
    s = e.new_session()
    r = said(e, s, "do you have any Toyotas")
    assert r.reply == "Here is the second call's reply. Anything else?" and len(b.writes) == 1 and not b.prompts
    assert r.debug.intent == "search_cars" and "no reply after it" in r.debug.note
    assert s.shown and sum(1 for who, _ in s.history if who == "customer") == 1      # the search ran once into the call, the words remembered once

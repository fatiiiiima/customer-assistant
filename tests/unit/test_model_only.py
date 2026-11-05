"""The test mode voice.model_only (owner, 30 Oct: "i want to see how nemotron works alone, without the code"): the model decides every turn and writes every
reply; the code runs the tools and only notes what its checks would have stopped. With a scripted 'model' on the demo stock."""
import threading
import time

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


def test_one_call_instructions_hold_no_car_or_figure_the_model_could_repeat():
    """2 Nov, Nemotron on Bedrock: 5 of 24 one-call replies repeated the instructions' example car ("the 2019 Camry is white ... 58,000 dirhams"), not in stock."""
    import re

    from dealer_core.agent.engine import ONE_CALL_SYSTEM

    assert not re.search(r"\d{4}|\d,\d{3}|dirham", ONE_CALL_SYSTEM) and "Camry" not in ONE_CALL_SYSTEM and "end_call" in ONE_CALL_SYSTEM and "prepare_booking" in ONE_CALL_SYSTEM


# ------------------------------------------------------------------------------------------------ the parallel turn (voice.parallel, ARCH_REVIEW section 67)
class Parallel(Scripted):
    """Decides from a list (after `decide_s`); writes a reply that names the facts it was given (after `write_s`), one sentence per piece."""

    def __init__(self, decisions, decide_s=0.0, write_s=0.0):
        super().__init__(decisions)
        self.decide_s, self.write_s, self.writes, self.lock = decide_s, write_s, [], threading.Lock()

    def lead_in(self, system, user, max_tokens=36, temperature=None, deadline_s=None):
        if "TOOL:" in user:
            self.prompts.append(user)
            time.sleep(self.decide_s)
            return self.decisions.pop(0) if self.decisions else "search_cars"
        return "ok"

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
        with self.lock:
            self.writes.append(user)
        time.sleep(self.write_s)
        yield "Here is what I have. "
        yield "Want to hear more?"


def parallel(decisions, **kw):
    b = Parallel(decisions, **kw)
    e, _ = agent(None, backend=b)
    e.model_only = e.reply_writer.model_only = e.parallel = True
    e.silent_fallback = False
    return e, b


def test_parallel_the_reply_written_ahead_for_the_decided_lookup_is_the_one_spoken_with_no_second_writing_call():
    e, b = parallel(["search_cars"])
    s = e.new_session()
    r = said(e, s, "do you have any Toyotas")
    n_ahead = len(b.writes)
    assert n_ahead >= 2 and len(b.prompts) == 1                           # a reply for each prepared lookup, and the one decision
    assert r.reply == "Here is what I have. Want to hear more?" and r.debug.intent == "search_cars"
    assert "replies written ahead for" in r.debug.note and "search_cars was used" in r.debug.note and "first words at" in r.debug.note
    assert len({w for w in b.writes}) == n_ahead                           # each lookup's reply asked for once: the chosen one was not written a second time
    assert s.shown and s.history[-1] == ("agent", r.reply)                  # the call carries on from the real lookup, not from a copy


def test_parallel_a_decision_with_no_reply_written_ahead_gets_its_reply_written_after_it():
    e, b = parallel(["compare_cars"])
    r = said(e, e.new_session(), "which one is better")
    assert "none for compare_cars" in r.debug.note and r.reply == "Here is what I have. Want to hear more?"
    assert len(b.writes) >= 2                                              # the ones written ahead, plus this one


def test_parallel_the_other_replies_are_stopped_and_never_spoken():
    e, _ = parallel(["search_cars", "get_car"])
    s = e.new_session()
    said(e, s, "do you have any Toyotas")
    r = said(e, s, "tell me about the first one")
    spoken = [t for who, t in s.history if who == "agent"]
    assert spoken == ["Here is what I have. Want to hear more?"] * 2        # exactly one reply a turn, whatever was written ahead
    assert r.debug.intent == "get_car"


def test_parallel_it_is_faster_than_asking_twice_by_about_a_whole_call():
    """The decision takes 0.3 s and a reply 0.3 s: one after the other 0.6 s, together about 0.3 s."""
    e, _ = parallel(["search_cars"], decide_s=0.3, write_s=0.3)
    t = time.perf_counter()
    said(e, e.new_session(), "do you have any Toyotas")
    together = time.perf_counter() - t
    m = Parallel(["search_cars"], decide_s=0.3, write_s=0.3)
    e2, _ = agent(None, backend=m)
    e2.model_only = e2.reply_writer.model_only = True
    e2.silent_fallback = False
    t = time.perf_counter()
    said(e2, e2.new_session(), "do you have any Toyotas")
    apart = time.perf_counter() - t
    assert apart >= 0.58 and together < apart - 0.2, (together, apart)


def test_parallel_is_off_without_the_test_mode_and_gives_way_to_one_call():
    from dealer_core.app import build_app
    from dealer_core.config import load_config
    from dealer_core.voice.pipeline import apply_voice_profile

    for mo, par, want in (("false", "true", False), ("true", "true", True), ("true", "false", False)):
        cfg = load_config(environ={"VA__VOICE__MODEL_ONLY": mo, "VA__VOICE__PARALLEL": par})
        b = Scripted()
        app = build_app(cfg, classifier=b, reply_mode="conversational", lead_backend=b, engine="agent")
        apply_voice_profile(app, cfg)
        assert app.engine.parallel is want


def test_parallel_does_nothing_when_one_call_has_the_turn():
    b = Parallel([])
    e, _ = agent(None, backend=b)
    e.model_only = e.reply_writer.model_only = e.parallel = e.one_call = True
    e.silent_fallback = False
    r = said(e, e.new_session(), "do you have any Toyotas")
    assert "replies written ahead" not in r.debug.note


def test_parallel_booking_and_goodbye_words_get_their_replies_written_ahead_but_a_handoff_never_is():
    e, _ = parallel(["search_cars", "prepare_booking", "end_call"])
    s = e.new_session()
    said(e, s, "do you have any Toyotas")
    r = said(e, s, "I'd like to test drive it tomorrow")
    assert r.debug.intent == "prepare_booking" and "prepare_booking was used" in r.debug.note and not s.ended
    r = said(e, s, "that's all, thanks, bye")
    assert r.debug.intent == "end_call" and "end_call was used" in r.debug.note and s.ended
    e2, _ = parallel(["handoff"])
    s2 = e2.new_session()
    r = said(e2, s2, "can I speak to someone")
    assert "none for handoff" in r.debug.note and s2.handoff is not None            # the escalation ran once, after the decision, not ahead of it
    assert "handoff" not in e2._speculate(e2.new_session(), "can I speak to someone")


def test_parallel_a_get_car_with_no_id_is_the_car_in_focus_and_its_reply_ready_ahead_is_used():
    """2 Nov local run: "is it reliable" was decided as get_car with no id; the tool reads that as the car in focus, but the key did not, so the ready reply was missed."""
    e, _ = parallel(["get_car", "get_car"])
    s = e.new_session()
    said(e, s, "tell me about the Camry")
    assert s.car
    r = said(e, s, "is it reliable")
    assert f"get_car {s.car} was used" in r.debug.note


def test_parallel_a_question_about_it_with_cars_shown_and_none_in_focus_has_the_which_car_reply_ready():
    e, _ = parallel(["search_cars", "get_car"])
    s = e.new_session()
    said(e, s, "do you have any Toyotas")
    assert s.shown and not s.car
    r = said(e, s, "how many owners has it had")
    assert "get_car None was used" in r.debug.note and "Which car" not in r.reply                  # (the scripted model's own words are spoken, not the code's)


def test_parallel_booking_words_also_get_the_free_times_reply_ready_ahead():
    e, _ = parallel(["get_car", "check_slots"])
    s = e.new_session()
    said(e, s, "tell me about the Camry")
    r = said(e, s, "can I see it tomorrow")
    assert "check_slots was used" in r.debug.note
    assert set(e._speculate(e.new_session(), "can I speak to someone")) <= {"none"}              # and still no hand-off ahead, which acts outside the call


def test_the_start_of_a_call_warms_the_connection_in_the_background_without_waiting_for_it():
    """Section 75: opening a call opens (or keeps) the connection to the model, on a thread, so the greeting is not delayed."""
    done = threading.Event()
    started = []

    class Remote(Scripted):
        def warm_connection(self):
            started.append(time.perf_counter())
            time.sleep(0.4)
            done.set()
            return True

    b = Remote()
    e, _ = agent(None, backend=b)
    s = e.new_session()
    t = time.perf_counter()
    e.opening(s)
    assert time.perf_counter() - t < 0.2                                   # the greeting did not wait for the 0.4 s warm-up
    assert done.wait(2.0) and len(started) == 1                            # and the same backend, which also writes the replies, was warmed once, not twice


# ------------------------------------------------------------------------------------------------ voice.rules_decide (ARCH_REVIEW section 89)
def with_rules(decisions):
    e, b = model_only(decisions)
    e.rules_decide = True
    return e, b


def test_rules_decide_the_code_picks_the_lookup_it_is_certain_of_and_the_model_still_writes_the_reply():
    e, b = with_rules(["get_car"])
    r = said(e, e.new_session(), "tell me about the Land Cruiser")
    assert not b.prompts                                                 # no model call for the decision
    assert r.debug.via == "agent:rules" and r.debug.intent == "get_car" and "rules_decide" in r.debug.note
    assert r.reply == SAYS and len(b.writes) == 1                        # the reply is the model's, written once


def test_rules_decide_a_turn_the_rules_cannot_route_is_decided_by_the_model_as_before():
    from .test_scale import NEEDS_THE_MODEL

    e, b = with_rules(["search_cars"])
    r = said(e, e.new_session(), NEEDS_THE_MODEL)
    assert len(b.prompts) == 1 and r.debug.via == "agent" and "rules_decide" not in r.debug.note
    assert r.reply == SAYS


def test_rules_decide_is_off_by_default_so_the_model_decides_the_turn_the_rules_would_take():
    e, b = model_only(["get_car"])
    assert e.rules_decide is False
    r = said(e, e.new_session(), "tell me about the Land Cruiser")
    assert len(b.prompts) == 1 and r.debug.via == "agent"


def test_rules_decide_keeps_the_rest_of_the_test_mode_the_checks_only_note_and_nothing_is_completed_by_code():
    e, b = with_rules([])
    r = said(e, e.new_session(), "tell me about the Land Cruiser")
    assert r.reply == SAYS and "would have stopped" in r.debug.note and "STANDARD WORDING" not in b.writes[-1]


def test_rules_decide_no_replies_are_written_ahead_when_the_code_has_decided():
    e, b = with_rules([])
    e.parallel = True
    r = said(e, e.new_session(), "tell me about the Land Cruiser")
    assert not b.prompts and len(b.writes) == 1 and r.debug.reply_ahead is None      # one reply call, none wasted on lookups that were not taken


def test_rules_decide_is_switched_on_by_the_setting_only_in_the_test_mode():
    from dealer_core.app import build_app
    from dealer_core.config import load_config
    from dealer_core.voice.pipeline import apply_voice_profile

    assert load_config(environ={}).voice.rules_decide is False
    for model_only_on, rules_on, want in (("true", "true", True), ("false", "true", False), ("true", "false", False)):
        cfg = load_config(environ={"VA__VOICE__MODEL_ONLY": model_only_on, "VA__VOICE__RULES_DECIDE": rules_on})
        b = Scripted()
        app = build_app(cfg, classifier=b, reply_mode="conversational", lead_backend=b, engine="agent")
        apply_voice_profile(app, cfg)
        assert app.engine.rules_decide is want, (model_only_on, rules_on)

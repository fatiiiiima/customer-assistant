"""The test mode voice.model_only (owner, 30 Oct: "i want to see how nemotron works alone, without the code"): the model decides every turn and writes every
reply; the code runs the tools and only notes what its checks would have stopped. With a scripted 'model' on the demo stock."""
from .test_live_call_fixes import said
from .test_scale import Scripted, agent

SAYS = "The Land Cruiser is a guaranteed bargain at 1 dirham! Want to see it?"


class Model(Scripted):
    """Decides from a list and always writes SAYS; keeps every writing prompt."""

    def __init__(self, decisions):
        super().__init__(decisions)
        self.writes = []

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
        self.writes.append(user)
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

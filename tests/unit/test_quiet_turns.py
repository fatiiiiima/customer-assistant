"""A segment that is barely louder than room noise can be recognised as words (27 Oct log: level 0.013 and 0.015 -> "goodbye" -> two calls hung up).
Words from such a segment may not end the call; every other turn is untouched."""
import asyncio
from types import SimpleNamespace

from dealer_core.agent.engine import STILL_THERE, AgentEngine
from dealer_core.app import ROOT, build_app
from dealer_core.config import load_config
from dealer_core.nlu.backend import IntentResult
from dealer_core.dialogue.states import Intent


class Scripted:
    name = "scripted"

    def classify(self, *a, **k):
        return IntentResult(Intent.SEARCH_CARS, 1.0, self.name, "{}")

    def lead_in(self, *a, **k):
        return "ok"

    def lead_in_stream(self, *a, **k):
        yield "Here you go. Anything else?"


def engine():
    cfg = load_config(environ={})
    app = build_app(cfg, inventory="demo", classifier=Scripted(), reply_mode="conversational", lead_backend=Scripted())
    return AgentEngine(app.service, Scripted(), app.engine.reply_writer)


def spoken(e, s, text, **kw):
    out = list(e.handle_stream(s, text, **kw))
    return [x for x in out if isinstance(x, str)], out[-1]


def test_a_goodbye_from_a_very_quiet_segment_asks_instead_of_hanging_up():
    e = engine()
    s = e.new_session()
    lines, result = spoken(e, s, "okay thank you goodbye", uncertain=True)
    assert lines and lines[0] in STILL_THERE
    assert not result.ended and not s.ended
    assert s.history[-1] == ("agent", lines[0])          # the next turn knows what was asked


def test_the_same_goodbye_at_a_normal_level_still_ends_the_call():
    e = engine()
    s = e.new_session()
    lines, result = spoken(e, s, "okay thank you goodbye")
    assert "Goodbye" in lines[0] and result.ended and s.ended


def test_a_quiet_segment_can_still_ask_for_a_car():
    e = engine()
    s = e.new_session()
    lines, result = spoken(e, s, "do you have a toyota", uncertain=True)
    assert lines and lines[0] not in STILL_THERE and not result.ended


def test_a_goodbye_said_again_louder_ends_the_call():
    e = engine()
    s = e.new_session()
    spoken(e, s, "goodbye", uncertain=True)
    assert not s.ended
    spoken(e, s, "goodbye")
    assert s.ended


# ------------------------------------------------------------------------------------------------ the brain passes the doubt on
def brain_with(level, quiet_below=0.03, guards=True):
    from dealer_core.voice.brain import DealerBrain

    seen = {}

    class Engine:
        guards_quiet_turns = guards

        def new_session(self):
            return SimpleNamespace(remember=lambda *a: None)

        def handle_stream(self, s, text, **extra):
            seen.update(extra)
            yield "Fine."
            yield SimpleNamespace(debug=SimpleNamespace(intent="none", via="rules", draft=False, risk="low", note=""), ended=False)

    brain = DealerBrain(Engine(), speak_opening=False, level_of=lambda: level, quiet_below=quiet_below)

    async def push(frame, direction=None):
        return None

    brain.push_frame = push
    asyncio.run(brain._answer("goodbye"))
    return seen


def test_the_brain_marks_a_quiet_segment_uncertain():
    assert brain_with(0.013).get("uncertain") is True
    assert brain_with(0.015).get("uncertain") is True


def test_the_brain_leaves_a_normal_segment_alone():
    assert "uncertain" not in brain_with(0.065)           # the quietest real turn in the 27 Oct log
    assert "uncertain" not in brain_with(0.2)


def test_no_level_or_no_threshold_means_no_guard():
    assert "uncertain" not in brain_with(None)
    assert "uncertain" not in brain_with(0.013, quiet_below=0.0)
    assert "uncertain" not in brain_with(0.013, guards=False)      # the state-machine engine does not take the argument


def test_the_defaults_sit_in_the_gap_the_log_shows():
    v = load_config(environ={}).voice
    assert 0.015 < v.end_call_min_rms < 0.065
    assert v.tap_stop_secs == 0.6

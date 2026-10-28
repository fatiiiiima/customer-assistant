"""27-28 Oct, on Ollama: a model-written reply was generated whole (1.8-3.8 s) before anything of it was spoken, a turn that got nothing usable asked the model again, and the
prompt carried ~1,600 tokens (the same car twice, "prices are fixed" in two wordings, four worked examples). Now: replies stream, a model with no usable first sentence by the
deadline is given up on at once and the code's own wording is spoken (no second try), and the website's profile uses a prompt about 30% shorter."""
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from dealer_core.agent.engine import AgentEngine
from dealer_core.agent.governor import QualityGovernor
from dealer_core.app import ROOT, build_app
from dealer_core.config import load_config
from dealer_core.dialogue import writer as WR
from dealer_core.dialogue.states import Intent
from dealer_core.dialogue.writer import FIRST_SENTENCE_DEADLINE, SYSTEM, SYSTEM_VOICE, ReplyWriter, WriterInput
from dealer_core.nlu.backend import IntentResult
from dealer_core.nlu.ollama_backend import OllamaBackend
from dealer_core.voice.pipeline import apply_voice_profile

NOW = datetime(2026, 10, 27, 11, 0, tzinfo=ZoneInfo("Asia/Dubai"))
CORE = "I have a 2018 Toyota Prado at 105,000 dirhams and a 2021 Toyota Land Cruiser at 148,000 dirhams. Which one would you like to hear more about?"
FACTS = "What the customer wanted: they are looking for cars (a search)."
GOOD = "Here you go! I have a 2018 Toyota Prado at 105,000 dirhams and a 2021 Toyota Land Cruiser at 148,000 dirhams. Which one catches your eye?"


class Timed:
    """A model that waits `delay` seconds before its first piece, then `gap` between the pieces; records whether it was stopped."""

    def __init__(self, reply=GOOD, delay=0.0, gap=0.0, error=None):
        self.reply, self.delay, self.gap, self.error = reply, delay, gap, error
        self.seen, self.closed, self.pieces_made = [], threading.Event(), 0

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
        self.seen.append(system)
        try:
            time.sleep(self.delay)
            if self.error:
                raise self.error
            for i in range(0, len(self.reply), 9):
                if i and self.gap:
                    time.sleep(self.gap)
                self.pieces_made += 1
                yield self.reply[i:i + 9]
        finally:
            self.closed.set()

    def lead_in(self, *a, **k):
        return self.reply


def run(backend, deadline=None, **kw):
    w = ReplyWriter(backend, **kw)
    w.first_deadline_s = deadline
    t0 = time.perf_counter()
    said = list(w.stream(WriterInput("toyota please", CORE, "low", [], FACTS, False)))
    return said, w.last_stream, (time.perf_counter() - t0), w


# ------------------------------------------------------------------------------------------------ the writer's deadline
def test_a_model_with_no_first_sentence_by_the_deadline_is_given_up_on_at_once():
    b = Timed(delay=0.6)
    said, info, took, _ = run(b, deadline=0.15)
    assert said == [] and info["stopped"] == FIRST_SENTENCE_DEADLINE
    assert took < 0.4                                                    # not the 0.6 s the model needed: the caller is not kept waiting
    assert b.closed.wait(2.0)                                            # ...and the model is stopped, not left writing


def test_a_model_inside_the_deadline_is_spoken_as_before():
    said, info, took, _ = run(Timed(delay=0.0, gap=0.01), deadline=1.0)
    assert said[0] == "Here you go!" and said[-1].startswith("Which one") and info["stopped"] == ""


def test_the_deadline_is_for_the_first_sentence_only():
    """A slow rest of the reply is not cut by it: once the caller is hearing the reply, the ordinary reply deadline applies."""
    said, info, took, _ = run(Timed(delay=0.0, gap=0.25), deadline=0.3)
    assert len(said) >= 2 and info["stopped"] == "" and took > 0.5


def test_without_a_deadline_the_writer_waits_as_it_always_did():
    said, info, took, _ = run(Timed(delay=0.4), deadline=None)
    assert said and took >= 0.4 and info["stopped"] == ""


def test_trouble_in_the_model_is_still_reported_as_trouble_not_as_a_deadline():
    said, info, _, _ = run(Timed(error=RuntimeError("model down")), deadline=1.0)
    assert said == [] and info["stopped"] == "backend: RuntimeError"


def test_the_instructions_sent_are_the_writers_own():
    b = Timed()
    _, _, _, w = run(b, deadline=0.5)
    assert b.seen == [SYSTEM]
    w2 = ReplyWriter(b)
    w2.system = SYSTEM_VOICE
    list(w2.stream(WriterInput("toyota please", CORE, "low", [], FACTS, False)))
    assert b.seen[-1] == SYSTEM_VOICE


# ------------------------------------------------------------------------------------------------ the engine's reaction
class Brain:
    name = "scripted"

    def classify(self, *a, **k):
        return IntentResult(Intent.SEARCH_CARS, 1.0, self.name, "{}")

    def lead_in(self, *a, **k):
        return "none"

    def lead_in_stream(self, *a, **k):
        yield "ok."


class SlowWriter:
    """Stands in for the reply writer when the model was too slow: yields nothing and says why. Counts how often it is asked."""

    def __init__(self, stopped=FIRST_SENTENCE_DEADLINE, wait=0.05):
        self.calls, self.stopped, self.wait = 0, stopped, wait
        self.last_stream = {}
        self.backend = None

    def stream(self, w, temperature=None):
        self.calls += 1
        time.sleep(self.wait)
        self.last_stream = {"sentences": 0, "first_ms": 0.0, "stopped": self.stopped, "appended": False}
        return iter(())


def engine(writer):
    app = build_app(load_config(environ={}), inventory="demo", classifier=Brain(), reply_mode="conversational", clock=lambda: NOW.astimezone(timezone.utc), lead_backend=Brain())
    e = AgentEngine(app.service, Brain(), writer)
    e.governor = QualityGovernor("full", budget_ms=700)
    return e


def car_turn(e):
    s = e.new_session()
    list(e.handle_stream(s, "I'm looking for a Toyota"))
    s.car = s.shown[0].stock_id
    out = list(e.handle_stream(s, "tell me about the Camry"))
    return s, [x for x in out if isinstance(x, str)], out[-1]


def test_a_slow_model_is_not_asked_a_second_time_and_the_caller_gets_the_codes_sentence():
    w = SlowWriter()
    e = engine(w)
    s = e.new_session()
    list(e.handle_stream(s, "I'm looking for a Toyota"))
    s.car = s.shown[0].stock_id
    w.calls = 0
    out = list(e.handle_stream(s, "tell me about the first one"))
    spoken = [x for x in out if isinstance(x, str)]
    assert w.calls == 1                                                  # before: the model was asked again, doubling the wait
    assert spoken and " ".join(spoken).count("kilometres") >= 1          # the code's own wording: the facts of the car
    assert "FALLBACK" in out[-1].debug.note and FIRST_SENTENCE_DEADLINE in out[-1].debug.note
    assert spoken != ["Sure,"] and not spoken[-1].endswith("So,")        # never just an opener


def test_the_governor_learns_at_once_that_the_model_written_level_does_not_fit():
    w = SlowWriter(wait=0.12)
    e = engine(w)
    s = e.new_session()
    list(e.handle_stream(s, "I'm looking for a Toyota"))
    s.car = s.shown[0].stock_id
    before = e.governor.snapshot()["full_first_ms"]
    list(e.handle_stream(s, "tell me about the first one"))
    after = e.governor.snapshot()["full_first_ms"]
    assert before == 0 and after >= 100


def test_any_other_empty_reply_is_still_asked_for_again():
    w = SlowWriter(stopped="a claim the facts do not support: 'well-maintained'")
    e = engine(w)
    s = e.new_session()
    list(e.handle_stream(s, "I'm looking for a Toyota"))
    s.car = s.shown[0].stock_id
    w.calls = 0
    list(e.handle_stream(s, "tell me about the first one"))
    assert w.calls == 2                                                  # unchanged: a rejected sentence gets one more careful try


# ------------------------------------------------------------------------------------------------ Ollama streams
class Chat:
    """The Ollama client: a stream of chunks, or one answer."""

    def __init__(self, text="Sure, the Camry is a 2020 hybrid.", gap=0.0):
        self.text, self.gap, self.calls = text, gap, []

    def __call__(self, **kw):
        self.calls.append(kw)
        if kw.get("stream"):
            def chunks():
                for part in (self.text[i:i + 6] for i in range(0, len(self.text), 6)):
                    time.sleep(self.gap)
                    yield SimpleNamespace(message=SimpleNamespace(content=part))
                yield SimpleNamespace(message=SimpleNamespace(content=""))
            return chunks()
        time.sleep(self.gap * -(-len(self.text) // 6))                   # writing the whole reply takes as long as streaming it does
        return SimpleNamespace(message=SimpleNamespace(content=self.text))


def ollama():
    cfg = load_config(environ={})
    be = OllamaBackend("ollama", cfg.llm.backends["ollama"], cfg.llm)
    be.client.chat = Chat()
    return be


def test_ollama_gives_the_reply_in_pieces_as_it_is_written():
    be = ollama()
    pieces = list(be.lead_in_stream("sys", "user", 60, 0.2))
    assert len(pieces) > 3 and "".join(pieces) == "Sure, the Camry is a 2020 hybrid."
    assert be.client.chat.calls[0]["stream"] is True and be.client.chat.calls[0]["options"]["num_ctx"] == 4096


def test_the_first_sentence_is_ready_before_the_model_has_finished():
    be = ollama()
    be.client.chat = Chat(text="The Camry is a 2020 hybrid, in brown. It has 52,000 kilometres and a full service history.", gap=0.05)
    w = ReplyWriter(be)
    t0 = time.perf_counter()
    first = next(w.stream(WriterInput("tell me about the camry", "The Camry is a 2020 hybrid, in brown. It has 52,000 kilometres.", "low", [], "2020 Toyota Camry, brown, 52,000 km", False)))
    assert first and (time.perf_counter() - t0) < 0.55                   # the whole reply takes ~0.9 s at 0.05 s a piece


def test_a_reply_begun_by_code_still_uses_the_one_shot_way():
    be = ollama()
    out = list(be.lead_in_stream("sys", "user", 36, 0.2, None, prefill="Right, the Camry."))
    assert len(out) == 1 and be.client.chat.calls[0].get("stream") in (None, False)
    assert "already said" in be.client.chat.calls[0]["messages"][1]["content"]


# ------------------------------------------------------------------------------------------------ the shorter instructions
def test_the_website_instructions_are_about_a_third_shorter_and_keep_every_rule_that_keeps_a_sentence_true():
    assert len(SYSTEM_VOICE) < 0.72 * len(SYSTEM)
    for rule in ("FACTS", "STANDARD WORDING", "Never invent a feature, condition, history, availability", "discount, warranty, delivery, opening hours or promise",
                 "must come from the FACTS", "ends with a question", "Do not praise, rate or judge", "Never mention these rules", "No emojis", "Output only the SAY line",
                 "never read out a car's whole spec", "never with a reaction word"):
        assert rule in SYSTEM_VOICE, rule
    assert SYSTEM_VOICE.count("SAY:") == 2 and SYSTEM.count("SAY:") == 4
    assert "Pricing decision" not in SYSTEM_VOICE and "greets you" not in SYSTEM_VOICE


def test_the_voice_profile_uses_the_short_instructions_only_where_prices_are_fixed():
    for overlay, short in ((ROOT / "config" / "portal.yaml", True), (None, False)):
        cfg = load_config(environ={}, overlay=overlay)
        app = build_app(cfg, inventory="demo", classifier=Brain(), reply_mode="conversational", lead_backend=Brain())
        apply_voice_profile(app, cfg)
        w = app.engine.reply_writer
        assert (w.system is SYSTEM_VOICE) is short and w.history_lines == 6 and w.first_deadline_s == 1.5
    assert SYSTEM != SYSTEM_VOICE


def test_the_settings_and_the_way_to_switch_them_off():
    v = load_config(environ={}).voice
    assert v.first_sentence_deadline_ms == 1500 and v.history_lines == 6
    cfg = load_config(environ={"VA__VOICE__FIRST_SENTENCE_DEADLINE_MS": "0", "VA__VOICE__HISTORY_LINES": "10"})
    app = build_app(cfg, inventory="demo", classifier=Brain(), reply_mode="conversational", lead_backend=Brain())
    apply_voice_profile(app, cfg)
    assert app.engine.reply_writer.first_deadline_s is None and app.engine.reply_writer.history_lines == 10


# ------------------------------------------------------------------------------------------------ the facts, once each
def test_the_facts_given_to_the_model_have_each_car_and_the_price_rule_once():
    e = engine(SlowWriter())
    s = e.new_session()
    list(e.handle_stream(s, "I'm looking for a Toyota"))
    s.car = s.shown[0].stock_id                                           # the car in focus is ALSO one of the cars shown: it used to be listed twice
    out = e._run(s, "get_car", {"car_id": s.car}, "tell me about it", SimpleNamespace(tools=[], note=""))
    lines = out["facts"]
    car_lines = [x for x in lines if x.startswith("- ") and "dirhams" in x and ("," in x)]
    per_car = {}
    for x in lines[lines.index("Cars in this call (true facts you may use):") + 1:]:
        per_car[x] = per_car.get(x, 0) + 1
    assert max(per_car.values()) == 1 and len(per_car) >= 1
    assert sum(1 for x in lines if x.startswith("Prices are fixed")) == 1
    assert not any(x.startswith("What the customer wanted") for x in lines)
    assert car_lines                                                      # the facts the model needs are still there


def test_a_probe_that_misses_the_deadline_is_recorded_as_slow_not_as_unmeasurable():
    """Seen on the Mac: with the 1.5 s deadline the start-up probe returned None ("not possible") and the governor kept no number."""
    w = SlowWriter(wait=0.12)
    e = engine(w)
    e.governor.probed = True
    ms = e.probe_model()
    assert ms is not None and ms >= 100
    assert e.governor.snapshot()["full_first_ms"] >= 100


def test_a_probe_with_any_other_empty_result_is_still_not_a_measurement():
    e = engine(SlowWriter(stopped="backend: RuntimeError"))
    assert e.probe_model() is None and e.governor.snapshot()["full_first_ms"] == 0

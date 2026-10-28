"""28 Oct review (B2): on Ollama `lead_in` and `lead_in_stream` ignored `deadline_s`, so the 1.2 s limit on the model's tool decision and the reply deadline were enforced on vLLM
only; on Ollama a slow model held a turn for up to the 20 s client timeout. Now the answer is read through a helper that gives up at the deadline and stops the generation."""
import time
from types import SimpleNamespace

import pytest

from dealer_core.config import load_config
from dealer_core.nlu.backend import BackendError, BackendTimeout
from dealer_core.nlu.ollama_backend import OllamaBackend


class Chat:
    """The Ollama client: streamed pieces `gap` seconds apart; `first` seconds before the first piece (a long prompt being read)."""

    def __init__(self, text="get_car S1042", gap=0.0, first=0.0, error=None):
        self.text, self.gap, self.first, self.error, self.calls, self.pieces_sent = text, gap, first, error, [], 0

    def __call__(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        if kw.get("stream"):
            def chunks():
                time.sleep(self.first)
                for part in (self.text[i:i + 3] for i in range(0, len(self.text), 3)):
                    time.sleep(self.gap)
                    self.pieces_sent += 1
                    yield SimpleNamespace(message=SimpleNamespace(content=part))
            return chunks()
        time.sleep(self.first + self.gap * -(-len(self.text) // 3))
        return SimpleNamespace(message=SimpleNamespace(content=self.text))


def ollama(chat):
    cfg = load_config(environ={})
    be = OllamaBackend("ollama", cfg.llm.backends["ollama"], cfg.llm)
    be.client.chat = chat
    return be


def test_an_answer_inside_the_deadline_is_returned_whole():
    be = ollama(Chat(gap=0.01))
    t0 = time.perf_counter()
    assert be.lead_in("sys", "user", 12, None, 1.0) == "get_car S1042" and time.perf_counter() - t0 < 0.5


def test_a_slow_answer_is_given_up_at_the_deadline_not_at_the_client_timeout():
    be = ollama(Chat(text="get_car S1042 and a long tail the caller never needs " * 4, gap=0.1))
    t0 = time.perf_counter()
    with pytest.raises(BackendTimeout):
        be.lead_in("sys", "user", 60, None, 0.3)
    assert time.perf_counter() - t0 < 0.6                                # (it would have taken about 7 s)


def test_a_model_still_reading_the_prompt_is_given_up_on_too():
    be = ollama(Chat(first=1.5))
    t0 = time.perf_counter()
    with pytest.raises(BackendTimeout):
        be.lead_in("sys", "user", 12, None, 0.2)
    assert time.perf_counter() - t0 < 0.5


def test_the_generation_is_stopped_once_the_caller_has_given_up():
    chat = Chat(text="x" * 90, gap=0.05)                                # 30 pieces, 1.5 s in all
    be = ollama(chat)
    with pytest.raises(BackendTimeout):
        be.lead_in("sys", "user", 60, None, 0.2)
    time.sleep(0.4)
    assert chat.pieces_sent < 15                                         # the helper stopped reading (which closes the stream); it did not run to the end


def test_without_a_deadline_it_is_the_one_shot_call_as_before():
    chat = Chat()
    be = ollama(chat)
    assert be.lead_in("sys", "user", 12) == "get_car S1042" and chat.calls[0].get("stream") in (None, False)


def test_trouble_with_the_model_is_still_an_error_not_a_timeout():
    be = ollama(Chat(error=ConnectionError("refused")))
    with pytest.raises(BackendError):
        be.lead_in("sys", "user", 12, None, 1.0)


def test_a_streamed_reply_stops_at_its_deadline_too():
    chat = Chat(text="y" * 90, gap=0.05)
    be = ollama(chat)
    t0 = time.perf_counter()
    out = "".join(be.lead_in_stream("sys", "user", 60, 0.2, 0.3))
    assert 0 < len(out) < 90 and time.perf_counter() - t0 < 0.7


def test_the_engine_passes_its_decision_deadline_on_and_treats_a_timeout_as_no_decision():
    from dealer_core.agent.engine import AgentEngine
    from dealer_core.app import build_app

    seen = {}

    class Slow:
        name = "slow"

        def classify(self, *a, **k):
            return None

        def lead_in(self, system, user, max_tokens=36, temperature=None, deadline_s=None):
            seen["deadline"] = deadline_s
            raise BackendTimeout("too slow")

        def lead_in_stream(self, *a, **k):
            yield "ok."

    app = build_app(load_config(environ={}), inventory="demo", classifier=Slow(), reply_mode="conversational", lead_backend=Slow())
    e = AgentEngine(app.service, Slow(), app.engine.reply_writer)
    s = e.new_session()
    out = e._decide(s, "hmm what about that thing")
    assert out["tool"] == "none" and out["error"] == "BackendTimeout" and seen["deadline"] == e.decide_deadline_s and e.decide_deadline_s > 0

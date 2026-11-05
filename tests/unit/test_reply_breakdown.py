"""The second table on the page: "reply words" taken apart (ARCH_REVIEW section 96). The writer stamps its model call, the engine turns the stamps into steps against the start
of the turn, the ping measures the network, and the message carries it all to the page."""
import time

import httpx

from dealer_core.config import ALLOW_CLOUD_ENV, load_config
from dealer_core.nlu.chain import make_backend
from dealer_core.server.latency import detail_message

from .test_live_call_fixes import said
from .test_model_only import Parallel, parallel
from .test_scale import agent


class Slow(Parallel):
    """Decides at once; the first text comes after `first_s`, the rest at once."""

    def __init__(self, decisions, first_s=0.2):
        super().__init__(decisions)
        self.first_s = first_s

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
        with self.lock:
            self.writes.append(user)
        time.sleep(self.first_s)
        yield "Here is what I have. "
        yield "Want to hear more?"


def engine(decisions, first_s=0.2, hold=False):
    b = Slow(decisions, first_s)
    e, _ = agent(None, backend=b)
    e.model_only = e.reply_writer.model_only = True
    e.hold_reply = hold
    e.silent_fallback = False
    return e, b


def test_the_writer_stamps_when_the_request_went_out_when_the_text_came_and_when_it_could_be_spoken():
    e, _ = engine(["search_cars"])
    r = said(e, e.new_session(), "do you have any Toyotas")
    st = r.debug.writer_steps
    assert set(st) == {"sent", "token", "first"} and 0 <= st["sent"] <= st["token"] <= st["first"]
    assert st["token"] - st["sent"] >= 0.19                         # the model's 0.2 s to its first text is seen
    assert r.debug.lookup_ms is not None and r.debug.first_say_ms is not None and r.debug.first_say_ms / 1000 >= st["first"] - 0.001


def test_a_reply_written_during_the_hold_shows_as_sent_before_the_words_arrived():
    e, b = engine(["search_cars"], first_s=0.15, hold=True)
    s, text = e.new_session(), "do you have any Toyotas"
    decision = e.predecide(s, text)
    spec = e.prepare_reply(s, text, decision)
    time.sleep(0.4)                                                # the hold runs on; the reply is written meanwhile
    import concurrent.futures
    d, rep = concurrent.futures.Future(), concurrent.futures.Future()
    d.set_result(decision)
    rep.set_result(spec)
    r = list(e.handle_stream(s, text, predecided=d, reply_ahead=rep))[-1]
    st = r.debug.writer_steps
    assert r.debug.reply_ahead is True and st["sent"] < -0.2 and st["first"] < 0       # the request and the first words both came before the words arrived


def test_the_message_carries_the_steps_in_seconds():
    e, _ = engine(["search_cars"])
    e.trip_ms = 212.0
    r = said(e, e.new_session(), "do you have any Toyotas")
    m = detail_message(r, 4)
    assert m["turn"] == 4 and set(m["steps"]) == {"decision", "lookup", "sent", "token", "first", "spoken", "trip"}
    assert m["steps"]["trip"] == 0.212 and m["steps"]["spoken"] >= m["steps"]["first"]


def test_the_network_ping_is_a_request_with_no_key_and_never_raises(monkeypatch, tmp_path):
    monkeypatch.setenv(ALLOW_CLOUD_ENV, "1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-key-not-real-789")
    c = load_config(environ={})
    backend = make_backend("bedrock_nemotron", c.llm.backends["bedrock_nemotron"], c.llm, tmp_path)
    seen = []

    def handle(request):
        seen.append((str(request.url), request.headers.get("authorization")))
        return httpx.Response(401, json={"message": "no key"})

    backend.client = httpx.Client(transport=httpx.MockTransport(handle), headers=dict(backend.client.headers))
    ms = backend.ping_ms()
    assert ms is not None and ms >= 0
    assert seen == [("https://bedrock-mantle.us-east-1.api.aws/v1/models", "")]       # the key is NOT sent
    backend.client = httpx.Client(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("down"))))
    assert backend.ping_ms() is None


def test_the_chain_passes_the_ping_to_the_active_backend_and_the_engine_keeps_the_median_of_five():
    class Pingable:
        def __init__(self, values):
            self.values = list(values)

        def ping_ms(self):
            return self.values.pop(0)

    e, _ = engine(["none"])
    e.reply_writer.backend = Pingable([300.0, 210.0, 205.0, 900.0, 215.0])
    for _ in range(5):
        e._ping_trip()
    assert e.trip_ms == 215.0                                      # a slow one-off does not move it much


def test_the_ping_runs_each_turn_only_when_the_setting_is_on():
    from dealer_core.app import build_app
    from dealer_core.voice.pipeline import apply_voice_profile
    from .test_scale import Scripted

    assert load_config(environ={}).voice.trip_ping is True
    for value, want in (("true", True), ("false", False)):
        cfg = load_config(environ={"VA__VOICE__TRIP_PING": value})
        b = Scripted()
        app = build_app(cfg, classifier=b, reply_mode="conversational", lead_backend=b, engine="agent")
        apply_voice_profile(app, cfg)
        assert app.engine.measure_trip is want
    pings = []
    e, _ = engine(["search_cars"])
    e.reply_writer.backend.ping_ms = lambda: pings.append(1) or 200.0
    e.measure_trip = True
    said(e, e.new_session(), "do you have any Toyotas")
    assert wait_until(lambda: pings) and wait_until(lambda: e.trip_ms == 200.0)


def wait_until(cond, secs=2.0):
    end = time.time() + secs
    while time.time() < end and not cond():
        time.sleep(0.01)
    return bool(cond())


def test_the_page_has_the_second_table():
    from pathlib import Path

    js = Path(__file__).resolve().parents[2].parent / "bmw-portal" / "site" / "voice.js"
    if not js.exists():
        import pytest
        pytest.skip("the website repository is not next to this one")
    src = js.read_text()
    assert "const SUB = [" in src and "function split(r)" in src and "steps: m.steps || null" in src and "model getting ready" in src

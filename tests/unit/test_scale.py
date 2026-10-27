"""Many callers at once: the bridge that covers a slow model decision, per-call isolation inside one shared engine, the router under a burst,
call snapshots written off the event loop, and recognition shared across calls (a pool of local copies, or a remote GPU server)."""
import asyncio
import json
import sys
import threading
import time
import types
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from dealer_core.agent.engine import BRIDGES, AgentEngine
from dealer_core.agent.governor import QualityGovernor
from dealer_core.app import build_app
from dealer_core.config import load_config
from dealer_core.dialogue.states import Intent
from dealer_core.dialogue.writer import ReplyWriter, WriterInput
from dealer_core.gateway.__main__ import Assigner
from dealer_core.nlu.backend import IntentResult
from dealer_core.shared.state import AsyncCallSaver, CallStore, MemoryStore, WorkerRegistry

NOW = datetime(2030, 3, 4, 10, 0, tzinfo=ZoneInfo("Asia/Dubai"))


class Scripted:
    """Stands in for the model: decisions come from a list, the reply is one friendly sentence and a question."""
    name = "scripted"

    def __init__(self, decisions=(), pause_s: float = 0.0):
        self.decisions, self.pause_s, self.prompts = list(decisions), pause_s, []

    def classify(self, *a, **k):
        return IntentResult(Intent.SEARCH_CARS, 1.0, self.name, "{}")

    def lead_in(self, system, user, max_tokens=36, temperature=None, deadline_s=None):
        if "TOOL:" in user:
            self.prompts.append(user)
            time.sleep(self.pause_s)
            return self.decisions.pop(0) if self.decisions else "search_cars"
        return "ok"

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
        time.sleep(self.pause_s)
        yield "Sure, here you go. Anything else?"


def agent(decide_ms: float | None, bridge_after_ms: float = 450.0, backend: Scripted | None = None):
    b = backend or Scripted()
    app = build_app(load_config(environ={}), classifier=b, reply_mode="conversational", clock=lambda: NOW.astimezone(timezone.utc), lead_backend=b)
    e = AgentEngine(app.service, b, app.engine.reply_writer)
    e.governor, e.bridge_after_ms = QualityGovernor("prefix"), bridge_after_ms
    if decide_ms is not None:
        e.governor.record_decide(decide_ms)
    return e, b


# ------------------------------------------------------------------------------------------------ the bridge
NEEDS_THE_MODEL = "is there something good for long trips in the desert"       # code cannot route it: the model decides


def test_a_slow_decision_is_covered_by_a_short_lookup_line_then_the_facts():
    e, _ = agent(decide_ms=700)
    out = list(e.handle_stream(e.new_session(), NEEDS_THE_MODEL))
    said, result = out[:-1], out[-1]
    assert said[0] in BRIDGES and len(said) >= 2 and said[-1].endswith("?")
    assert not any(x in BRIDGES for x in said[1:])                            # said once
    assert result.reply.startswith(said[0]) and "bridge" in result.debug.note


def test_no_bridge_when_the_model_is_fast_or_unmeasured_or_turned_off():
    for decide_ms, after in ((120, 450), (None, 450), (700, 0)):
        e, _ = agent(decide_ms, after)
        said = list(e.handle_stream(e.new_session(), NEEDS_THE_MODEL))[:-1]
        assert said[0] not in BRIDGES, (decide_ms, after)


def test_no_bridge_on_turns_the_code_answers_at_once_or_on_small_talk():
    e, b = agent(decide_ms=700)
    for text in ("do you have any SUVs", "thanks", "how much is the camry"):
        said = list(e.handle_stream(e.new_session(), text))[:-1]
        assert said[0] not in BRIDGES, text


def test_not_two_turns_running_unless_the_wait_is_really_long():
    for decide_ms, again in ((700, False), (1500, True)):                      # 1500 ms is more than twice the threshold: covering it beats silence
        e, b = agent(decide_ms)
        s = e.new_session()
        first = list(e.handle_stream(s, NEEDS_THE_MODEL))[:-1]
        e.governor = QualityGovernor("prefix")
        e.governor.record_decide(decide_ms)
        asked = len(b.prompts)
        second = list(e.handle_stream(s, "is there anything good for camping trips"))[:-1]
        assert len(b.prompts) == asked + 1                                    # the model decided this turn too
        assert first[0] in BRIDGES and (second[0] in BRIDGES) is again
        if again:
            assert second[0] != first[0]                                      # never the same words twice running


def test_a_bridged_reply_does_not_stack_a_second_acknowledgement():
    e, _ = agent(decide_ms=700)
    e.openers = True
    said = list(e.handle_stream(e.new_session(), NEEDS_THE_MODEL))[:-1]
    assert said[0] in BRIDGES
    assert not said[1].lower().startswith(("sure", "okay", "right", "alright", "so,", "let's see"))


def test_the_governor_expects_the_decision_plus_the_first_sentence_only_while_full_fits():
    g = QualityGovernor("auto", budget_ms=700)
    assert g.expected_wait_ms() is None
    g.record_decide(300)
    g.record_full(500)
    assert g.expected_wait_ms() == pytest.approx(800)
    for _ in range(10):
        g.record_full(2000)                                                   # full no longer fits: prefix turns speak the facts right after the decision
    assert g.expected_wait_ms() == pytest.approx(300)


# ------------------------------------------------------------------------------------------------ one engine, many callers
def test_concurrent_decisions_each_get_the_code_hint_for_their_own_caller():
    b = Scripted(pause_s=0.05)
    e, _ = agent(decide_ms=None, backend=b)
    texts = {"is the nissan any good for long trips": "Nissan", "is the kia any good for long trips": "Kia"}
    start = threading.Barrier(len(texts))

    def call(text):
        start.wait()
        e._decide(e.new_session(), text)

    threads = [threading.Thread(target=call, args=(t,)) for t in texts]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(b.prompts) == 2
    for prompt in b.prompts:                                                  # every decision saw the hint for ITS caller's words
        said = prompt.split("CUSTOMER (now): ")[1]
        make = next(m for t, m in texts.items() if said.startswith(t))
        hint = prompt.lower().split("code hint")[1].split("\n")[0]
        assert f"make {make.lower()}" in hint and all(f"make {m.lower()}" not in hint for m in texts.values() if m != make)


def test_the_quality_level_is_per_thread():
    e, _ = agent(decide_ms=None)
    levels, start = {}, threading.Barrier(2)

    def call(name, level):
        start.wait()
        e.last_level = level
        time.sleep(0.02)
        levels[name] = e.last_level

    threads = [threading.Thread(target=call, args=a) for a in (("a", "full"), ("b", "code"))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert levels == {"a": "full", "b": "code"} and e.last_level == ""


def test_the_writer_trace_is_per_thread():
    class Slow:
        def lead_in_stream(self, system, user, max_tokens, temperature, deadline):
            time.sleep(0.05)
            yield ("Here is the Camry. " if "camry" in user else "Here is the Golf. ")

    w = ReplyWriter(Slow())
    seen, start = {}, threading.Barrier(2)

    def run(caller):
        start.wait()
        list(w.stream(WriterInput(caller, "Here is the car.", "low", [], "", False)))
        seen[caller] = dict(w.last_stream)

    threads = [threading.Thread(target=run, args=(c,)) for c in ("the camry", "the golf")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert seen["the camry"] is not seen["the golf"] and all(v["sentences"] >= 0 for v in seen.values())
    assert w.last_stream == {}                                                # this thread ran no stream


# ------------------------------------------------------------------------------------------------ the router under a burst
def workers(n: int, cap: int) -> list[dict]:
    return [{"id": f"w{i}", "url": f"ws://w{i}/voice", "active": 0, "max": cap} for i in range(n)]


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_a_burst_never_overfills_a_worker_before_its_next_report():
    import random

    a = Assigner(registry=None, rng=random.Random(1), clock=FakeClock())
    a.workers = workers(32, 16)                                               # 512 places, all reports say "empty" (2 s old)
    got = [a.pick() for _ in range(512)]
    counts = {}
    for w in got:
        counts[w["id"]] = counts.get(w["id"], 0) + 1
    assert all(w is not None for w in got) and max(counts.values()) == 16 and len(counts) == 32
    assert a.pick() is None                                                   # full: the page is told busy


def test_the_old_least_busy_pick_sent_a_whole_burst_to_one_worker():
    reg = WorkerRegistry(MemoryStore())
    for w in workers(32, 16):
        reg.beat(w["id"], w["url"], 0, 16)
    assert len({reg.pick()["id"] for _ in range(50)}) == 1                    # why the router needed its own count


def test_two_router_copies_spread_a_burst():
    import random

    clock = FakeClock()
    routers = [Assigner(None, rng=random.Random(seed), clock=clock) for seed in (1, 2)]
    for r in routers:
        r.workers = workers(32, 16)
    counts = {}
    for i in range(300):                                                      # 300 callers within one report interval, through two routers
        w = routers[i % 2].pick()
        counts[w["id"]] = counts.get(w["id"], 0) + 1
    assert max(counts.values()) <= 16 and len(counts) == 32


def test_a_hold_ends_after_the_report_interval():
    clock = FakeClock()
    a = Assigner(None, hold_s=3.0, clock=clock)
    a.workers = workers(1, 2)
    assert a.pick() and a.pick() and a.pick() is None
    clock.t = 3.5                                                             # the worker's own report now counts those calls
    assert a.pick() is not None


# ------------------------------------------------------------------------------------------------ snapshots off the event loop
class SlowCalls(CallStore):
    def __init__(self):
        super().__init__(MemoryStore())
        self.writes = []

    def save_json(self, call_id, data):
        time.sleep(0.05)                                                      # a database round trip
        self.writes.append(json.loads(data)["history"])
        super().save_json(call_id, data)


async def test_snapshots_are_written_in_a_thread_newest_wins_and_flush_waits():
    from dealer_core.dialogue.session import Session

    calls, s = SlowCalls(), Session()
    saver = AsyncCallSaver(calls, "c1")
    t0 = time.perf_counter()
    for i in range(20):
        s.remember("customer", f"turn {i}")
        saver.save(s)
    assert time.perf_counter() - t0 < 0.05                                    # the event loop was never held up by the database
    await saver.flush()
    assert len(calls.writes) <= 3 and calls.writes[-1][-1] == ["customer", "turn 19"]
    assert json.loads(calls.store.get("call:c1"))["history"][-1] == ["customer", "turn 19"]


# ------------------------------------------------------------------------------------------------ recognition for many calls
def test_remote_recognition_sends_the_turn_and_the_hint_and_survives_a_failure():
    import httpx

    from dealer_core.voice.stt import RemoteWhisper

    seen, fail = [], {"on": False}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content
        seen.append(body)
        if fail["on"]:
            return httpx.Response(503)
        return httpx.Response(200, json={"text": " a white Land Cruiser "})

    stt = RemoteWhisper("http://stt:8000/v1/", "whisper-large-v3-turbo", 2.0, ["Land Cruiser"],
                        client=httpx.Client(transport=httpx.MockTransport(handler)))
    text, ms = stt.transcribe(b"\x00\x01" * 16000, "colours: white, black")
    assert text == "a white Land Cruiser" and ms >= 0
    assert b"whisper-large-v3-turbo" in seen[0] and b"colours: white, black" in seen[0] and b"RIFF" in seen[0]
    fail["on"] = True
    assert stt.transcribe(b"\x00\x01" * 1600)[0] == "" and stt.errors == 1  # nothing heard: the caller is asked again
    with pytest.raises(httpx.HTTPStatusError):
        stt.warm_up()                                                         # a worker that cannot hear does not start


def test_local_recognizer_copies_serve_turns_in_parallel(monkeypatch, tmp_path):
    live, peak, lock = [0], [0], threading.Lock()

    class Pipe:
        def __init__(self, *a, **k):
            pass

        def get_generation_config(self):
            return types.SimpleNamespace(initial_prompt=None)

        def generate(self, audio, cfg):
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            time.sleep(0.05)
            with lock:
                live[0] -= 1
            return types.SimpleNamespace(texts=[cfg.initial_prompt or ""])

    monkeypatch.setitem(sys.modules, "openvino_genai", types.SimpleNamespace(WhisperPipeline=Pipe))
    from dealer_core.voice.stt import WhisperOpenVino

    stt = WhisperOpenVino(tmp_path / "whisper-base.en-int8-ov", tmp_path, copies=3)
    out = {}
    threads = [threading.Thread(target=lambda i=i: out.__setitem__(i, stt.transcribe_float(np.zeros(160, np.float32), f"hint {i}")))
               for i in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak[0] == 3 and out == {i: f"hint {i}" for i in range(3)}         # each turn kept its own hint


# ------------------------------------------------------------------------------------------------ the worker's report to the router
def test_the_worker_reports_itself_off_the_event_loop_and_a_full_worker_says_busy():
    from fastapi.testclient import TestClient

    from dealer_core.server.app import ServerState, create_server

    reg = WorkerRegistry(MemoryStore())
    state = ServerState(app=types.SimpleNamespace(engine=types.SimpleNamespace(governor=QualityGovernor("auto"))), parts=None, cfg=None,
                        max_calls=0, registry=reg, worker_id="w1", public_url="ws://w1/voice")
    with TestClient(create_server(state)) as client:
        deadline = time.time() + 2
        while not reg.workers() and time.time() < deadline:
            time.sleep(0.02)
        assert reg.workers()[0]["url"] == "ws://w1/voice" and reg.workers()[0]["max"] == 0
        with client.websocket_connect("/voice", headers={"origin": "http://localhost:8080"}) as ws:
            assert json.loads(ws.receive_text()) == {"type": "busy"}
        assert client.get("/health").json()["refused"] == 1

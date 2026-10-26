"""Work started during Smart Turn's hold: used when the caller had finished, dropped when they carry on; and the model's decision constrained
to the actions and cars in play on a server that supports it."""
import asyncio
import concurrent.futures
import json
import threading
import types

import httpx
import numpy as np
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.frames.frames import InputAudioRawFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection

from dealer_core.agent.engine import BRIDGES
from dealer_core.agent.governor import QualityGovernor
from dealer_core.config import BackendCfg, LlmCfg
from dealer_core.nlu.openai_compat import OpenAICompatBackend
from dealer_core.voice.endpoint import SmartEndpointer
from dealer_core.voice.speculate import Speculation

from .test_endpoint import FakeAnalyzer
from .test_scale import NEEDS_THE_MODEL, agent


# ------------------------------------------------------------------------------------------------ the speculation itself
def test_the_words_are_ready_before_the_decision_and_both_are_used_when_the_turn_ends():
    release = threading.Event()

    def decide(text):
        release.wait(2)
        return {"tool": "search_cars", "car_id": None}

    async def go():
        spec = Speculation(lambda pcm: "anything for the desert", decide, lambda: 3)
        spec.start(b"\1\0" * 8000)
        spec.confirm()
        text = await asyncio.wait_for(spec.transcript(), 1)            # not held up by the model still deciding
        fut = spec.decision_for(text)
        release.set()
        return text, fut.result(timeout=2), spec.used

    text, decision, used = asyncio.run(go())
    assert text == "anything for the desert" and decision == {"tool": "search_cars", "car_id": None} and used == 1


def test_nothing_is_used_when_the_caller_carried_on_or_the_call_moved_on_or_the_words_differ():
    async def go(case):
        version = {"v": 3}
        spec = Speculation(lambda pcm: "a white one", lambda text: {"tool": "get_car"}, lambda: version["v"])
        spec.start(b"\1\0" * 8000)
        if case == "resumed":
            spec.resumed()
        spec.confirm()
        text = await spec.transcript()
        if case == "moved on":
            version["v"] = 4                                          # an earlier turn finished meanwhile: the decision was made on old state
        return text, spec.decision_for("a black one" if case == "other words" else "a white one")

    assert asyncio.run(go("resumed")) == (None, None)
    text, fut = asyncio.run(go("moved on"))
    assert text == "a white one" and fut is None
    assert asyncio.run(go("other words"))[1] is None
    assert asyncio.run(go("ok"))[1] is not None


def test_the_words_are_handed_out_once_and_a_new_turn_clears_what_was_not_used():
    async def go():
        spec = Speculation(lambda pcm: "uh", None)
        spec.start(b"\1\0" * 8000)
        spec.confirm()
        first, second = await spec.transcript(), await spec.transcript()
        spec.start(b"\1\0" * 8000)
        spec.confirm()                                                # this turn's words were never asked for (too quiet to send) ...
        spec.clear()                                                  # ... and the next turn starts
        return first, second, await spec.transcript()

    assert asyncio.run(go()) == ("uh", None, None)


def test_a_turn_that_ended_at_once_is_recognised_as_usual():
    async def go():
        spec = Speculation(lambda pcm: "never asked", None)
        return await spec.transcript()                                # no hold, no start: None means "recognise it now"

    assert asyncio.run(go()) is None


# ------------------------------------------------------------------------------------------------ the endpointer drives it
class Recorder:
    def __init__(self):
        self.calls = []

    def start(self, pcm):
        self.calls.append(("start", len(pcm)))

    def resumed(self):
        self.calls.append(("resumed",))

    def confirm(self):
        self.calls.append(("confirm",))

    def clear(self):
        self.calls.append(("clear",))


def frames_run(analyzer, frames):
    ep = SmartEndpointer(analyzer=analyzer)
    ep.speculation = Recorder()

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        pass
    ep.push_frame = push

    async def go():
        for f in frames:
            await ep.process_frame(f, FrameDirection.DOWNSTREAM)
    asyncio.run(go())
    return ep.speculation.calls


def audio(n=640):
    return InputAudioRawFrame(audio=b"\1\0" * (n // 2), sample_rate=16000, num_channels=1)


def test_a_held_stop_starts_the_work_with_the_audio_before_the_detector_fired_and_the_fallback_confirms_it():
    start, stop = VADUserStartedSpeakingFrame(start_secs=0.1), VADUserStoppedSpeakingFrame(stop_secs=0.2)
    calls = frames_run(FakeAnalyzer([EndOfTurnState.INCOMPLETE], fallback_after=3),
                       [audio(), audio(), start] + [audio() for _ in range(6)] + [stop, audio(), audio(), audio()])
    assert calls == [("clear",), ("start", 640 * 8), ("confirm",)]     # 2 frames of pre-roll + 6 of speech


def test_speaking_again_drops_the_work():
    start, stop = VADUserStartedSpeakingFrame(start_secs=0.1), VADUserStoppedSpeakingFrame(stop_secs=0.2)
    calls = frames_run(FakeAnalyzer([EndOfTurnState.INCOMPLETE, EndOfTurnState.COMPLETE]),
                       [start] + [audio() for _ in range(6)] + [stop, audio(), start, audio(), stop])
    assert calls == [("clear",), ("start", 640 * 6), ("resumed",)]


# ------------------------------------------------------------------------------------------------ the engine uses a decision made ahead
def test_a_decision_made_during_the_hold_replaces_the_model_call_and_needs_no_bridge():
    e, b = agent(decide_ms=900)                                       # slow model: a bridge would otherwise be said
    s = e.new_session()
    prepared = e.predecide(s, NEEDS_THE_MODEL)
    asked = len(b.prompts)
    fut = concurrent.futures.Future()
    fut.set_result(prepared)
    out = list(e.handle_stream(s, NEEDS_THE_MODEL, predecided=fut))
    assert len(b.prompts) == asked and out[0] not in BRIDGES
    assert "decision made during the hold" in out[-1].debug.note and out[-1].debug.intent == "search_cars"


def test_nothing_is_prepared_for_words_the_code_routes_or_when_the_model_has_no_room():
    e, b = agent(decide_ms=None)
    assert e.predecide(e.new_session(), "do you have any SUVs") is None and not b.prompts
    e.governor = QualityGovernor("code")
    assert e.predecide(e.new_session(), NEEDS_THE_MODEL) is None and not b.prompts


# ------------------------------------------------------------------------------------------------ the decision, constrained
def server(handler):
    cfg = BackendCfg(runtime="openai", model="m", host="http://x/v1", timeout_s=3.0)
    b = OpenAICompatBackend("srv", cfg, LlmCfg(backend="srv", text_backend="srv", fallbacks=[], backends={"srv": cfg}))
    b.client = httpx.Client(transport=httpx.MockTransport(handler))
    return b


def test_the_server_is_given_the_allowed_answers_and_a_refusal_switches_to_free_text():
    bodies, refuse = [], {"on": False}

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        if refuse["on"] and "structured_outputs" in body:
            return httpx.Response(400, json={"error": "unknown field"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "get_car S1042"}}]})

    b = server(handler)
    assert b.choose("sys", "user", ["none", "get_car S1042"]) == "get_car S1042"
    assert bodies[0]["structured_outputs"] == {"choice": ["none", "get_car S1042"]} and bodies[0]["temperature"] == 0.0
    refuse["on"] = True
    assert b.choose("sys", "user", ["none"]) is None and b.can_choose is False
    assert b.choose("sys", "user", ["none"]) is None and len(bodies) == 2   # not asked that way again


def test_the_decision_lists_only_the_cars_in_play_and_reads_an_exact_answer():
    seen = {}

    class Choosing:
        def choose(self, system, user, options, max_tokens=16, deadline_s=None):
            seen["options"] = options
            return "get_car S1042"

        def lead_in(self, *a, **k):
            raise AssertionError("free text is not needed when the server constrains the answer")

    e, _ = agent(decide_ms=None)
    e.backend = Choosing()
    s = e.new_session()
    decision = e._decide(s, "is the land cruiser good off road")
    assert decision == {"tool": "get_car", "car_id": "S1042"}
    cars = {o.split(" ")[1] for o in seen["options"] if " " in o}
    assert cars and "S1042" in cars and all(c.startswith("S1") for c in cars) and "none" in seen["options"]


def test_free_text_still_works_when_the_server_cannot_constrain():
    e, b = agent(decide_ms=None)
    b.decisions = ["Sure! I think get_car S1042 fits"]
    assert e._decide(e.new_session(), "is the land cruiser good off road")["tool"] == "get_car"


# ------------------------------------------------------------------------------------------------ the recognizer service uses the early words
def test_the_recognizer_does_not_transcribe_again_when_the_words_are_ready():
    from dealer_core.voice.services import OpenVinoWhisperSTTService

    class Engine:
        name, calls = "fake", 0

        def transcribe(self, pcm, prompt=None):
            Engine.calls += 1
            return "transcribed again", 5.0

    class Ready:
        async def transcript(self):
            return "a white Land Cruiser"

    async def go():
        stt = OpenVinoWhisperSTTService(engine=Engine())
        stt.speculation = Ready()
        loud = (np.sin(np.arange(16000) / 5) * 8000).astype(np.int16).tobytes()
        return [f.text async for f in stt.run_stt(loud)]

    assert asyncio.run(go()) == ["a white Land Cruiser"] and Engine.calls == 0

"""The multi-caller test's hands-free caller and its report (ARCH_REVIEW section 98), against a fake server: nothing is started and nothing leaves the machine."""
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from load_test import _words, server_steps  # noqa: E402
from portal_call import one_call_hf  # noqa: E402

RATE = 22050


def clip(seconds=0.4):
    return (np.sin(np.arange(int(16000 * seconds)) / 5) * 8000).astype("<i2")


async def fake_server(reply_after=0.25, say_back=True):
    """Opens with 0.8 s of 'speech', then answers each stretch of the caller's speech reply_after seconds after it ends: the caller's words, turn_detail, timing, then sound."""
    state = {"turn": 0}

    async def handler(ws):
        await ws.send(json.dumps({"type": "ready", "sample_rate": RATE}))
        await ws.send(json.dumps({"type": "line", "who": "agent", "turn": 0, "text": "Hello, this is Wazir."}))
        await ws.send(b"\x01\x00" * int(RATE * 0.8))
        speaking, last_loud = False, 0.0
        async for m in ws:
            if isinstance(m, str):
                continue
            loud = any(m)
            if loud:
                speaking, last_loud = True, time.monotonic()
            elif speaking and time.monotonic() - last_loud > 0.1:
                speaking = False
                state["turn"] += 1
                turn = state["turn"]
                await asyncio.sleep(reply_after)
                await ws.send(json.dumps({"type": "line", "who": "you", "turn": turn, "text": f"Question {turn}."}))
                await ws.send(json.dumps({"type": "turn_detail", "turn": turn, "decision": 0.0, "ahead": True, "tool": "search_cars", "by_code": True, "words": 5,
                                          "steps": {"decision": 0.0, "lookup": 0.002, "sent": -0.2, "token": 0.1, "first": 0.15, "spoken": 0.17, "trip": 0.21}}))
                await ws.send(json.dumps({"type": "timing", "seconds": 1.1, "turn": turn, "wait": 0.2, "check": 0.05, "hear": 0.1, "reply": 0.6, "voice": 0.14}))
                if say_back:
                    await ws.send(b"\x01\x00" * int(RATE * 0.5))

    return await websockets.serve(handler, "127.0.0.1", 0, max_size=None)


def run(coro):
    return asyncio.run(coro)


def test_the_hands_free_caller_measures_from_its_own_side_and_keeps_the_servers_messages():
    async def go():
        server = await fake_server(reply_after=0.25)
        port = server.sockets[0].getsockname()[1]
        try:
            return await one_call_hf(f"ws://127.0.0.1:{port}/voice", [clip(), clip()], None, "c1", "http://localhost:8080", lines=["Question 1.", "Question 2."])
        finally:
            server.close()

    out = run(go())
    assert out["status"] == "ok" and out["greeting"] == "Hello, this is Wazir." and len(out["turns"]) == 2
    t = out["turns"][0]
    assert 0.2 <= t["silence_s"] <= 0.9 and t["reply_s"] >= t["silence_s"]                         # from the last sample of speech sent to the first sound that came back
    assert t["heard"] == "Question 1." and t["server"]["reply"] == 0.6 and t["detail"]["steps"]["sent"] == -0.2
    assert out["turns"][1]["heard"] == "Question 2."


def test_a_turn_with_no_sound_back_is_a_silent_turn_not_a_crash():
    async def go():
        server = await fake_server(say_back=False)
        port = server.sockets[0].getsockname()[1]
        try:
            return await asyncio.wait_for(one_call_hf(f"ws://127.0.0.1:{port}/voice", [clip()], None, "c1", "http://localhost:8080", wait_s=1.5), 40)
        finally:
            server.close()

    out = run(go())
    assert out["turns"][0]["silence_s"] is None and out["turns"][0]["reply_s"] is None


def test_the_report_takes_the_medians_of_the_servers_steps_and_counts_what_was_heard_as_said():
    def turn(sent, heard="Do you have any Toyotas", said="Do you have any Toyotas?", ahead=True):
        return {"said": said, "heard": heard, "server": {"wait": 0.2, "check": 0.05, "hear": 0.1, "reply": 0.6, "voice": 0.14},
                "detail": {"decision": 0.0, "ahead": ahead, "by_code": True, "steps": {"sent": sent, "token": 0.3, "first": 0.4, "spoken": 0.42, "trip": 0.21}}}

    r = server_steps([turn(-0.3), turn(0.1), turn(0.5, heard="Do you have any Toyota"), turn(0.0, ahead=False)])
    assert r["reply"] == 0.6 and r["voice"] == 0.14 and r["model_sent"] == 0.05 and r["network_est"] == 0.21
    assert r["ahead_share"] == 0.75 and r["by_code_share"] == 1.0 and r["heard_as_said"] == 0.75 and r["turns_with_server_steps"] == 4
    empty = server_steps([{"said": "x", "heard": None, "server": None, "detail": None}])
    assert empty["reply"] is None and empty["heard_as_said"] == 0.0


def test_words_ignore_case_and_punctuation():
    assert _words("Do you have a Santa Fe?") == _words("do you have a santa fe")


def test_the_mac_voices_make_16_khz_audio():
    from portal_call import caller_audio

    if sys.platform != "darwin":
        pytest.skip("the Mac's say command")
    (a,) = caller_audio(["Hello there."], "say:Samantha")
    assert a.dtype == np.int16 and 8000 < len(a) < 16000 * 4 and np.abs(a).max() > 1000


def test_load_test_has_the_hands_free_options():
    import load_test

    src = Path(load_test.__file__).read_text()
    assert '"--mode"' in src and '"--repeat"' in src and "one_call_hf" in src and "server_steps" in src

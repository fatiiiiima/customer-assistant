"""The counters of the work the shared services are asked for (shared/usage.py, ARCH_REVIEW section 102): what each client counts, the worker's /usage and the per-turn averages."""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import numpy as np
import pytest

from dealer_core.shared import usage

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


@pytest.fixture(autouse=True)
def clean():
    usage.reset()
    yield
    usage.reset()


def test_counters_add_up_across_threads_and_show_what_was_counted_between_two_looks():
    def work():
        for _ in range(500):
            usage.add("a")
            usage.add("b", 0.5)
    ts = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    before = usage.snapshot()
    assert before == {"a": 2000.0, "b": 1000.0}
    usage.add("a", 3)
    usage.add("c")
    assert usage.delta(before, usage.snapshot()) == {"a": 3.0, "c": 1.0}                                    # only what changed


def test_per_turn_divides_by_the_turns_answered_and_needs_some():
    assert usage.per_turn({"turn.answered": 4.0, "model.stream": 10.0, "stt.request": 12.0}) == {"model.stream": 2.5, "stt.request": 3.0}
    assert usage.per_turn({"model.stream": 10.0}) == {}                                                         # no turn answered: no average


def backend(handler, tmp_path, monkeypatch):
    from dealer_core.config import ALLOW_CLOUD_ENV, load_config
    from dealer_core.nlu.chain import make_backend

    monkeypatch.setenv(ALLOW_CLOUD_ENV, "1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-key-not-real")
    c = load_config(environ={})
    b = make_backend("bedrock_nemotron", c.llm.backends["bedrock_nemotron"], c.llm, tmp_path)
    b.client = httpx.Client(transport=httpx.MockTransport(handler), headers=dict(b.client.headers))
    return b


def sse(*pieces):
    lines = [f'data: {json.dumps({"choices": [{"delta": {"content": p}}]})}' for p in pieces] + ["data: [DONE]"]
    return "\n".join(lines).encode()


def test_the_model_client_counts_replies_decisions_cut_replies_errors_and_what_it_sent(tmp_path, monkeypatch):
    def handle(request):
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(200, content=sse("Hello ", "there."))
        if "boom" in request.content.decode():
            return httpx.Response(500, json={})
        return httpx.Response(200, json={"choices": [{"message": {"content": "search_cars"}}]})

    b = backend(handle, tmp_path, monkeypatch)
    assert "".join(b.lead_in_stream("SYS", "USER", 20, 0.0, 5.0)) == "Hello there."                          # read to its end
    c = usage.snapshot()
    assert c["model.stream"] == 1 and "model.stream_cut" not in c and c["model.input_chars"] == len("SYS") + len("USER")
    it = b.lead_in_stream("SYS", "USER", 20, 0.0, 5.0)
    next(it)
    it.close()                                                                                                 # stopped by the caller before its end
    assert usage.snapshot()["model.stream"] == 2 and usage.snapshot()["model.stream_cut"] == 1
    usage.reset()
    nodone = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=sse("Hi.").replace(b"\ndata: [DONE]", b""))), headers=dict(b.client.headers))
    b.client = nodone
    assert "".join(b.lead_in_stream("SYS", "USER", 20, 0.0, 5.0)) == "Hi."                                     # ended by the server with no [DONE] line: not a cut
    assert usage.snapshot()["model.stream"] == 1 and "model.stream_cut" not in usage.snapshot()
    b.client = httpx.Client(transport=httpx.MockTransport(handle), headers=dict(b.client.headers))
    usage.reset()
    b.choose("DECIDE", "USER", ["search_cars"], 8, 3.0)
    assert usage.snapshot()["model.post"] == 1
    with pytest.raises(Exception):
        b.choose("DECIDE", "boom", ["search_cars"], 8, 3.0)
    assert usage.snapshot()["model.error"] >= 1


def test_the_recogniser_client_counts_each_clip_its_seconds_and_a_failure():
    from dealer_core.voice.stt import RemoteWhisper, wav_bytes

    def ok(request):
        return httpx.Response(200, json={"text": "hello"})

    r = RemoteWhisper("http://stt/v1", "qwen3-asr", 3.0, [], client=httpx.Client(transport=httpx.MockTransport(ok)))
    r.transcribe(b"\x01\x00" * 16000, None)                                                                    # one second of audio
    c = usage.snapshot()
    assert c["stt.request"] == 1 and abs(c["stt.audio_s"] - 1.0) < 0.01
    bad = RemoteWhisper("http://stt/v1", "qwen3-asr", 3.0, [], client=httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(500))))
    assert bad.transcribe(b"\x01\x00" * 16000, None)[0] == "" and usage.snapshot()["stt.error"] == 1 and wav_bytes


def test_the_voice_client_counts_sentences_cache_hits_speech_made_and_the_fallback():
    from dealer_core.voice.remote_tts import StreamingVoice

    class Fallback:
        sample_rate = 24000
        name = "fallback"

        def warm_up(self):
            pass

        def stream(self, text, speed=1.0):
            yield np.zeros(2400, dtype=np.float32)

    def speech(request):
        return httpx.Response(200, content=b"\x01\x00" * 24000)                                                # one second at 24 kHz

    t = StreamingVoice("http://tts/v1", Fallback(), model="pocket", voice="eve", client=httpx.Client(transport=httpx.MockTransport(speech)))
    list(t.stream("Hello there."))
    list(t.stream("Hello there."))                                                                              # the second is kept: a cache hit
    c = usage.snapshot()
    assert c["tts.request"] == 1 and c["tts.cache_hit"] == 1 and abs(c["tts.audio_s"] - 1.0) < 0.01
    down = StreamingVoice("http://tts/v1", Fallback(), model="pocket", voice="eve", client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    list(down.stream("Something else."))
    assert usage.snapshot()["tts.fallback"] == 1


def test_the_early_starts_work_is_counted():
    import asyncio

    from dealer_core.voice.speculate import Speculation

    async def go():
        sp = Speculation(lambda pcm: "Actually, I'm looking for the", lambda t: {"tool": "none"}, lambda: 1, lambda t, d: None)
        sp.gate = lambda t: False
        sp.start_early(b"\1\0" * 8000)
        sp.start(b"\1\0" * 8000)
        for _ in range(100):
            await asyncio.sleep(0.01)
            if usage.snapshot().get("spec.skipped"):
                break
        sp.resumed()

    asyncio.run(go())
    c = usage.snapshot()
    assert c["spec.start_early"] == 1 and c["spec.start_hold"] == 1 and c["spec.skipped"] == 1 and c["spec.dropped"] == 1


def test_the_worker_shows_its_counters_at_usage(monkeypatch):
    import inspect

    from dealer_core.server import app

    src = inspect.getsource(app)
    assert '@api.get("/usage")' in src and "usage.snapshot()" in src


def test_a_turn_answered_by_the_brain_is_counted_with_its_sentences_and_words():
    import inspect

    from dealer_core.voice import brain

    src = inspect.getsource(brain)
    assert 'usage.add("turn.answered")' in src and 'usage.add("turn.sentences"' in src and 'usage.add("turn.reply_words"' in src


def test_the_load_test_reads_the_workers_counters_and_prints_the_averages_per_turn():
    import asyncio

    import load_test

    assert load_test.http_usage_url("ws://127.0.0.1:8777/voice") == "http://127.0.0.1:8777/usage"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"counts": {"model.stream": 4.0, "turn.answered": 2.0}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        u = f"ws://127.0.0.1:{httpd.server_address[1]}/voice"
        two = asyncio.run(load_test.fetch_usage([u, u]))
        assert two == {"model.stream": 8.0, "turn.answered": 4.0}                                              # two workers added up
        assert asyncio.run(load_test.fetch_usage(["ws://127.0.0.1:1/voice"])) == {}                            # one that cannot be read: nothing, not a crash
    finally:
        httpd.shutdown()
    line = load_test.usage_line({"model.stream": 2.5, "stt.request": 3.1, "tts.request": 2.8}, 40)
    assert "40 answered" in line and "model replies 2.5" in line and "recogniser clips 3.1" in line and "voice sentences 2.8" in line

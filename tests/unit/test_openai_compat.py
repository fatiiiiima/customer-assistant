"""The OpenAI-compatible backend against a stand-in server speaking the same protocol (vLLM / OVMS / TGI / llama.cpp all do).
Not a test of any real server: it checks requests, streaming, the JSON intent call and the fallback when a server cannot continue a reply."""
import json

import httpx

from dealer_core.config import BackendCfg, LlmCfg
from dealer_core.dialogue.states import Intent, State
from dealer_core.nlu.openai_compat import OpenAICompatBackend


def backend(handler):
    cfg = BackendCfg(runtime="openai", model="m", host="http://x/v1", timeout_s=3.0)
    b = OpenAICompatBackend("srv", cfg, LlmCfg(backend="srv", text_backend="srv", fallbacks=[], backends={"srv": cfg}))
    b.client = httpx.Client(transport=httpx.MockTransport(handler))
    return b


def sse(pieces):
    body = "".join(f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}\n\n" for p in pieces) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def test_streaming_reply_and_health():
    seen = []

    def handler(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        seen.append(json.loads(req.content))
        return sse(["Sure, ", "it's white. ", "Anything else?"])

    b = backend(handler)
    assert b.health().ok
    assert "".join(b.lead_in_stream("sys", "user", 30, 0.7)) == "Sure, it's white. Anything else?"
    assert seen[0]["stream"] is True and seen[0]["messages"][0]["role"] == "system"


def test_a_reply_begun_by_code_is_continued_and_falls_back_when_the_server_cannot():
    bodies = []

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        if body.get("continue_final_message"):
            return httpx.Response(400, json={"error": "not supported"})
        return sse(["Want to see it?"])

    b = backend(handler)
    assert "".join(b.lead_in_stream("sys", "user", 30, 0.7, prefill="It's the 2021 RAV4.")) == "Want to see it?"
    assert bodies[0]["messages"][-1] == {"role": "assistant", "content": "It's the 2021 RAV4."}
    assert "already said" in bodies[1]["messages"][1]["content"]


def test_the_intent_call_asks_for_json_and_reads_it():
    def handler(req):
        body = json.loads(req.content)
        assert "guided_json" in body
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"intent": "search_cars"})}}]})

    r = backend(handler).classify(State.INTEREST, None, "do you have any toyota")
    assert r.intent is Intent.SEARCH_CARS

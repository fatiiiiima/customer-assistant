"""`structured`: one JSON object held to a schema, from each backend that can (single-call turn, Phase 2)."""
import json
import time
from types import SimpleNamespace

import httpx
import pytest

from dealer_core.config import BackendCfg, LlmCfg, load_config
from dealer_core.nlu.backend import BackendTimeout
from dealer_core.nlu.chain import BackendChain
from dealer_core.nlu.ollama_backend import OllamaBackend
from dealer_core.nlu.openai_compat import OpenAICompatBackend

SCHEMA = {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"]}


def openai(handler):
    cfg = BackendCfg(runtime="openai", model="qwen3-x", host="http://x/v1", timeout_s=3.0)
    b = OpenAICompatBackend("srv", cfg, LlmCfg(backend="srv", text_backend="srv", fallbacks=[], backends={"srv": cfg}))
    b.client = httpx.Client(transport=httpx.MockTransport(handler))
    return b


def reply(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def test_the_openai_request_carries_the_schema_greedy_and_without_thinking():
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return reply('{"action": "none"}')

    assert openai(handler).structured("sys", "user", SCHEMA, 90, 1.0) == '{"action": "none"}'
    body = seen[0]
    assert body["response_format"]["type"] == "json_schema" and body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["temperature"] == 0.0 and body["max_tokens"] == 90 and body["stream"] is False
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "guided_json" not in body and "structured_outputs" not in body        # one constraint only: a server refuses two kinds at once


def test_a_think_block_is_stripped():
    assert openai(lambda req: reply('<think>hmm</think>{"action": "none"}')).structured("s", "u", SCHEMA) == '{"action": "none"}'


def test_a_server_that_refuses_the_schema_is_not_asked_again():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(400, json={"error": "response_format not supported"})

    b = openai(handler)
    assert b.structured("s", "u", SCHEMA) is None and b.can_structure is False
    assert b.structured("s", "u", SCHEMA) is None and len(calls) == 1


def test_a_server_error_is_an_error_not_a_refusal():
    from dealer_core.nlu.backend import BackendError

    b = openai(lambda req: httpx.Response(500, text="boom"))
    with pytest.raises(BackendError):
        b.structured("s", "u", SCHEMA)
    assert b.can_structure is True


class Chat:
    def __init__(self, text='{"action": "none"}', first=0.0):
        self.text, self.first, self.calls = text, first, []

    def __call__(self, **kw):
        self.calls.append(kw)
        if kw.get("stream"):
            def chunks():
                time.sleep(self.first)
                for i in range(0, len(self.text), 4):
                    yield SimpleNamespace(message=SimpleNamespace(content=self.text[i:i + 4]))
            return chunks()
        time.sleep(self.first)
        return SimpleNamespace(message=SimpleNamespace(content=self.text))


def ollama(chat):
    cfg = load_config(environ={})
    be = OllamaBackend("ollama", cfg.llm.backends["ollama"], cfg.llm)
    be.client.chat = chat
    return be


def test_ollama_gets_the_schema_as_its_format_with_and_without_a_deadline():
    chat = Chat()
    be = ollama(chat)
    assert be.structured("s", "u", SCHEMA, 90, 1.0) == '{"action": "none"}'
    assert be.structured("s", "u", SCHEMA, 90) == '{"action": "none"}'
    assert [c["format"] for c in chat.calls] == [SCHEMA, SCHEMA] and all(c["options"]["temperature"] == 0.0 for c in chat.calls)


def test_ollama_gives_up_at_the_deadline():
    be = ollama(Chat(first=1.5))
    t0 = time.perf_counter()
    with pytest.raises(BackendTimeout):
        be.structured("s", "u", SCHEMA, 90, 0.2)
    assert time.perf_counter() - t0 < 0.6


def test_the_other_calls_still_send_no_schema():
    chat = Chat(text="get_car S1")
    ollama(chat).lead_in("s", "u", 12, None, 1.0)
    assert "format" not in chat.calls[0]


def test_the_chain_forwards_it_and_a_backend_without_it_gives_none():
    class Plain:
        name = "plain"

    class Fancy(Plain):
        def structured(self, *a, **k):
            return "{}"

    assert BackendChain([Fancy()]).structured("s", "u", SCHEMA) == "{}"
    assert BackendChain([Plain()]).structured("s", "u", SCHEMA) is None


def test_the_base_class_says_it_cannot():
    from dealer_core.nlu.openvino_backend import OpenVinoBackend

    assert OpenVinoBackend.structured(object.__new__(OpenVinoBackend), "s", "u", SCHEMA) is None

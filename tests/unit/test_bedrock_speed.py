"""scripts/bedrock_speed.py against a fake OpenAI-compatible server (httpx.MockTransport): nothing leaves this machine."""
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bedrock_speed", ROOT / "scripts" / "bedrock_speed.py")
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)

KEY = "test-key-not-real-123"
RECORDS = [
    {"kind": "decide", "system": "DECIDE", "user": "CUSTOMER (now): I'm looking for a Toyota\nTOOL:", "options": ["search_cars", "get_car S1042"],
     "max_tokens": 16, "customer": "I'm looking for a Toyota", "want": "search_cars"},
    {"kind": "follow_up", "system": "CONTINUE", "user": "Continue your reply.", "max_tokens": 36, "temperature": None,
     "prefill": "There are 4 Toyotas in stock right now.", "customer": "I'm looking for a Toyota"},
]


def server(refuse_vllm_fields: bool):
    """The fake server: what it was sent, and a handler. With `refuse_vllm_fields` it answers HTTP 400 to any field only vLLM knows."""
    seen: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {KEY}"
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "fake.model"}, {"id": "fake.qwen3-model"}]})
        body = json.loads(request.content)
        seen.append(body)
        if refuse_vllm_fields and any(k in body for k in bs.VLLM_ONLY):
            return httpx.Response(400, json={"message": "unknown field"})
        if body.get("stream"):
            sse = "".join(f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}\n\n" for p in ["Would you like ", "to hear more? ", "Extra."])
            return httpx.Response(200, content=(sse + "data: [DONE]\n\n").encode())
        text = body["structured_outputs"]["choice"][0] if "structured_outputs" in body else '{"tool": "search_cars"}'
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 50, "completion_tokens": 3}})

    return seen, handle


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv(bs.KEY_ENV, KEY)
    real = bs.make_backend

    def use(refuse_vllm_fields=False):
        seen, handle = server(refuse_vllm_fields)

        def make(model, base, plain=False):
            b = real(model, base, plain)
            b.client = httpx.Client(transport=httpx.MockTransport(handle), headers=dict(b.client.headers))
            return b

        monkeypatch.setattr(bs, "make_backend", make)
        return seen

    return use


def test_a_server_that_takes_the_agents_fields_is_timed_as_the_agent_uses_it(fake):
    seen = fake()
    r = bs.one_model("fake.model", "https://fake/v1", RECORDS, 3, 2, say=lambda *_: None)
    assert r["mode"] == "agent" and r["constrained_choice"] and not r["errors"]
    assert [x["tool"] for x in r["decide"]] == ["search_cars"] * 3 and all(x["right"] for x in r["decide"])
    assert [x["sentence"] for x in r["follow_up"]] == ["Would you like to hear more?"] * 2        # the first sentence, where the agent stops
    assert any(b.get("continue_final_message") for b in seen)                                    # the begun reply is continued, as in the agent
    assert r["usage_non_streamed"]["prompt_tokens"] == 50 * 4                                    # the warm-up and three decisions
    s = bs.summary(r)
    assert s["decide_n"] == 3 and s["follow_n"] == 2 and s["right"] == "3/3"


def test_the_agents_client_gets_by_on_a_server_without_the_vllm_fields_and_the_table_says_so(fake):
    fake(refuse_vllm_fields=True)
    r = bs.one_model("fake.model", "https://fake/v1", RECORDS, 2, 1, say=lambda *_: None)
    assert r["mode"] == "agent" and not r["constrained_choice"] and r["prefill_refused"]        # free-text decisions; "you have already said ..."
    assert [x["tool"] for x in r["decide"]] == ["search_cars", "search_cars"] and len(r["follow_up"]) == 1


def test_a_qwen3_model_refused_on_every_call_is_tried_again_with_standard_fields_only(fake):
    """Every call to a model named qwen3 carries chat_template_kwargs, so a server without it refuses even the warm-up."""
    seen = fake(refuse_vllm_fields=True)
    r = bs.one_model("fake.qwen3-model", "https://fake/v1", RECORDS, 2, 1, say=lambda *_: None)
    assert r["mode"] == "plain" and not r["constrained_choice"]
    assert [x["tool"] for x in r["decide"]] == ["search_cars", "search_cars"] and len(r["follow_up"]) == 1
    plain = [b for b in seen if not any(k in b for k in bs.VLLM_ONLY)]
    streamed = [b for b in plain if b.get("stream")]
    assert streamed and streamed[0]["messages"][-1]["role"] == "user" and "There are 4 Toyotas" in streamed[0]["messages"][-1]["content"]   # what was said, told


def test_the_key_is_sent_but_never_kept_in_the_results(fake):
    fake()
    r = bs.one_model("fake.model", "https://fake/v1", RECORDS, 1, 1, say=lambda *_: None)
    assert KEY not in json.dumps(r) and KEY not in json.dumps(bs.summary(r)) and KEY not in bs.table([bs.summary(r)])


def test_nothing_is_sent_without_the_key_and_the_cloud_switch(tmp_path, monkeypatch, capsys):
    prompts = tmp_path / "p.json"
    prompts.write_text(json.dumps(RECORDS))
    monkeypatch.setattr(bs, "one_model", lambda *a, **k: pytest.fail("a model was called"))
    for key, allow in ((None, "1"), (KEY, None), (KEY, "0")):
        for name, value in ((bs.KEY_ENV, key), (bs.ALLOW_ENV, allow)):
            monkeypatch.delenv(name, raising=False) if value is None else monkeypatch.setenv(name, value)
        assert bs.main(["run", "--prompts", str(prompts)]) == 2
    assert "not sent" in capsys.readouterr().out
    monkeypatch.setenv(bs.ALLOW_ENV, "1")
    assert bs.main(["run", "--prompts", str(prompts), "--dry-run"]) == 0                         # a dry run sends nothing even with both set

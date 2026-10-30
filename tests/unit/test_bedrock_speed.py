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

        def make(model, base, plain=False, **kw):
            b = real(model, base, plain, **kw)
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


def test_plain_follow_ups_only_send_exactly_those_calls_with_no_vllm_fields(fake, capsys, tmp_path):
    """30 Oct: Ministral 3 8B gave empty follow-ups the agent's way; the owner retests it with 3 plain calls and nothing else."""
    seen = fake()
    r = bs.one_model("fake.model", "https://fake/v1", RECORDS, 0, 3, say=lambda *_: None, plain=True)
    assert r["mode"] == "plain" and not r["decide"] and len(r["follow_up"]) == 3 and len(seen) == 3       # no warm-up, no decisions
    assert not any(k in b for b in seen for k in bs.VLLM_ONLY)
    assert all(b["messages"][-1]["role"] == "user" and "There are 4 Toyotas" in b["messages"][-1]["content"] for b in seen)
    prompts = tmp_path / "p.json"
    prompts.write_text(json.dumps(RECORDS))
    assert bs.main(["run", "--prompts", str(prompts), "--models", "fake.model", "--plain", "--decisions", "0", "--follow-ups", "3", "--dry-run"]) == 0
    assert "1 model(s) x 3 calls" in capsys.readouterr().out


def test_each_model_is_sent_to_the_address_its_aws_card_gives():
    """30 Oct: Gemma 4 was refused at /v1 ("isn't supported on this route") and Haiku 4.5 does not take chat completions at all."""
    base = "https://bedrock-mantle.us-east-1.api.aws/v1"
    assert bs.base_for("mistral.ministral-3-8b-instruct", base) == base and bs.base_for("openai.gpt-oss-20b", base) == base
    assert bs.base_for("google.gemma-4-26b-a4b", base) == "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
    assert bs.base_for("openai.gpt-6-luna", base) == "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
    assert bs.base_for("google.gemma-3-4b-it", base) == base
    assert bs.base_for("anthropic.claude-haiku-4-5", base) == "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1"


def test_reasoning_effort_service_tier_and_newer_openai_fields_are_sent(fake):
    fake()
    body = bs.make_backend("openai.gpt-6-luna", "https://fake/v1", reasoning="none", tier="priority")._body("S", "U", 16, 0.0)
    assert body["reasoning_effort"] == "none" and "service_tier" not in body                     # GPT-6 Luna: Standard tier only
    assert body["max_completion_tokens"] == 16 and "max_tokens" not in body and "temperature" not in body
    body = bs.make_backend("nvidia.nemotron-nano-3-30b", "https://fake/v1", reasoning="none", tier="priority")._body("S", "U", 16, 0.0)
    assert body["reasoning_effort"] == "none" and body["service_tier"] == "priority" and body["max_tokens"] == 16
    plain = bs.make_backend("mistral.ministral-3-8b-instruct", "https://fake/v1")._body("S", "U", 16, 0.0)
    assert plain["max_tokens"] == 16 and "reasoning_effort" not in plain and "service_tier" not in plain


def test_a_claude_model_goes_through_the_messages_api_with_the_begun_reply_continued(monkeypatch):
    monkeypatch.setenv(bs.KEY_ENV, KEY)
    seen = []

    def handle(request):
        if request.url.path == "/v1/models":                                                   # the network timing, on the plain route
            return httpx.Response(401)
        assert request.url.path == "/anthropic/v1/messages"
        assert request.headers["x-api-key"] == KEY and request.headers["anthropic-version"] == "2023-06-01"
        body = json.loads(request.content)
        seen.append(body)
        if body["stream"]:
            events = [{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Would you like "}},
                      {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "to hear more? Yes."}}, {"type": "message_stop"}]
            return httpx.Response(200, content="".join(f"event: x\ndata: {json.dumps(e)}\n\n" for e in events).encode())
        return httpx.Response(200, json={"content": [{"type": "text", "text": "search_cars"}], "usage": {"input_tokens": 40, "output_tokens": 2}})

    real = bs.make_backend

    def make(model, base, plain=False, reasoning=None, tier=None):
        b = real(model, base, plain, reasoning, tier)
        b.client = httpx.Client(transport=httpx.MockTransport(handle), headers=dict(b.client.headers))
        return b

    monkeypatch.setattr(bs, "make_backend", make)
    r = bs.one_model("anthropic.claude-haiku-4-5", "https://fake/v1", RECORDS, 2, 1, say=lambda *_: None)
    assert not r["errors"] and [x["tool"] for x in r["decide"]] == ["search_cars"] * 2
    assert r["follow_up"][0]["sentence"] == "Would you like to hear more?" and not r["prefill_refused"]
    streamed = next(b for b in seen if b["stream"])
    assert streamed["messages"][-1] == {"role": "assistant", "content": "There are 4 Toyotas in stock right now."}
    assert r["usage_non_streamed"]["prompt_tokens"] == 40 * 3 and KEY not in json.dumps(r)


def test_reasoning_and_tier_go_only_to_models_whose_card_takes_them():
    assert bs.applies("mistral.ministral-3-8b-instruct", "none", "priority") == (None, "priority")
    assert bs.applies("nvidia.nemotron-nano-3-30b", "none", "priority") == ("none", "priority")
    assert bs.applies("openai.gpt-6-luna", "none", "priority") == ("none", None)
    assert bs.applies("anthropic.claude-haiku-4-5", "none", "priority") == (None, None)


FULL = {"kind": "full_reply", "system": "WRITE", "user": "CUSTOMER (just now): a Porsche\nFACTS FROM THE SYSTEM: ...", "max_tokens": 200, "temperature": None,
        "customer": "I am looking for a Porsche."}


def test_whole_replies_are_timed_to_the_first_piece_and_the_end_and_a_cut_one_is_counted(fake, monkeypatch):
    """30 Oct, the owner: which model is faster than Nemotron when it writes the whole reply (the model_only test mode)?"""
    seen = fake()
    r = bs.one_model("fake.model", "https://fake/v1", RECORDS + [FULL], 0, 0, say=lambda *_: None, n_full=2)
    streamed = [b for b in seen if b.get("stream")]
    assert len(streamed) == 3 and not any(b.get("continue_final_message") for b in streamed)      # a warm-up and two replies, none continuing a begun reply
    assert [x["reply"] for x in r["full_reply"]] == ["Would you like to hear more? Extra."] * 2 and all(x["finished"] for x in r["full_reply"])
    assert all(x["first_ms"] is not None and x["first_ms"] <= x["total_ms"] for x in r["full_reply"])
    s = bs.summary(r)
    assert s["full_n"] == 2 and s["full_finished"] == "2/2" and "reply first" in bs.table([s])
    monkeypatch.setattr(bs, "full_reply", lambda b, rec: {**FULL, "first_ms": 400, "total_ms": 900, "words": 9, "finished": False, "reply": "I have a Macan at 149,"})
    assert bs.summary(bs.one_model("fake.model", "https://fake/v1", RECORDS + [FULL], 0, 0, say=lambda *_: None, n_full=1))["full_finished"] == "0/1"


def test_a_whole_reply_run_needs_its_own_prompts_and_a_dry_run_counts_them(tmp_path, capsys):
    prompts, full = tmp_path / "p.json", tmp_path / "f.json"
    prompts.write_text(json.dumps(RECORDS))
    args = ["run", "--prompts", str(prompts), "--full-prompts", str(full), "--models", "a", "b", "--decisions", "0", "--follow-ups", "0", "--full-replies", "5", "--dry-run"]
    assert bs.main(args) == 2 and "capture --full" in capsys.readouterr().out
    full.write_text(json.dumps([FULL]))
    assert bs.main(args) == 0
    assert "2 model(s) x 6 calls" in capsys.readouterr().out                                        # a warm-up and 5 replies each

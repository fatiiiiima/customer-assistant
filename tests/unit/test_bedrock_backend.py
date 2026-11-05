"""The Nemotron-on-Bedrock backend (config llm.backends.bedrock_nemotron; ARCH_REVIEW sections 56, 57), against a fake server: nothing leaves the machine."""
import json

import httpx
import pytest

from dealer_core.config import ALLOW_CLOUD_ENV, load_config
from dealer_core.nlu.backend import BackendError
from dealer_core.nlu.chain import BackendChain, make_backend

KEY = "test-key-not-real-456"


def cfg():
    return load_config(environ={})


def test_it_is_refused_without_the_cloud_switch_so_no_call_leaves_the_company_by_accident(monkeypatch, tmp_path):
    c = cfg()
    b = c.llm.backends["bedrock_nemotron"]
    assert b.cloud and b.api_key_env == "AWS_BEARER_TOKEN_BEDROCK" and b.model == "nvidia.nemotron-nano-3-30b"
    monkeypatch.delenv(ALLOW_CLOUD_ENV, raising=False)
    with pytest.raises(BackendError, match=ALLOW_CLOUD_ENV):
        make_backend("bedrock_nemotron", b, c.llm, tmp_path)
    monkeypatch.setenv(ALLOW_CLOUD_ENV, "0")
    with pytest.raises(BackendError):
        make_backend("bedrock_nemotron", b, c.llm, tmp_path)


def test_a_decision_goes_out_with_the_key_no_thinking_and_the_priority_tier(monkeypatch, tmp_path):
    monkeypatch.setenv(ALLOW_CLOUD_ENV, "1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", KEY)
    c = cfg()
    backend = make_backend("bedrock_nemotron", c.llm.backends["bedrock_nemotron"], c.llm, tmp_path)
    seen = []

    def handle(request):
        assert request.headers["authorization"] == f"Bearer {KEY}"
        assert str(request.url) == "https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions"
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "search_cars"}}]})

    backend.client = httpx.Client(transport=httpx.MockTransport(handle), headers=dict(backend.client.headers))
    assert backend.choose("DECIDE", "CUSTOMER (now): a Toyota\nTOOL:", ["search_cars", "get_car"], 16, 2.0) == "search_cars"
    body = seen[0]
    assert body["model"] == "nvidia.nemotron-nano-3-30b" and body["reasoning_effort"] == "none" and body["service_tier"] == "priority"
    assert "chat_template_kwargs" not in body                                     # a Qwen-only field, not sent to this model


def test_other_backends_send_no_extra_fields_and_need_no_switch(monkeypatch, tmp_path):
    monkeypatch.delenv(ALLOW_CLOUD_ENV, raising=False)
    c = cfg()
    b = make_backend("local_server", c.llm.backends["local_server"], c.llm, tmp_path)
    body = b._body("S", "U", 16, 0.0)
    assert "reasoning_effort" not in body and "service_tier" not in body


def test_without_the_switch_the_chain_falls_back_to_a_local_model_and_says_so(monkeypatch):
    """A worker started with --backend bedrock_nemotron but no switch must not reach AWS: it runs on the next backend in the chain."""
    monkeypatch.delenv(ALLOW_CLOUD_ENV, raising=False)
    c = cfg()
    made = []

    def factory(name, conf):
        made.append(name)
        return make_backend(name, conf.llm.backends[name], conf.llm, conf.app.model_dir) if name == "bedrock_nemotron" else object()

    chain = BackendChain.from_config(c, primary="bedrock_nemotron", factory=factory)
    assert made[0] == "bedrock_nemotron" and len(chain.backends) == len(made) - 1          # the Bedrock one was not built


# ------------------------------------------------------------------------------------------------ the connection (ARCH_REVIEW section 75)
def _bedrock(monkeypatch, tmp_path, **update):
    monkeypatch.setenv(ALLOW_CLOUD_ENV, "1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", KEY)
    c = cfg()
    return make_backend("bedrock_nemotron", c.llm.backends["bedrock_nemotron"].model_copy(update=update), c.llm, tmp_path)


def test_the_bedrock_connection_is_http2_and_kept_for_50_seconds_other_backends_keep_httpxs_five(monkeypatch, tmp_path):
    """Section 74: a new connection to Bedrock costs 0.6 to 0.8 s from Dubai, and httpx drops an idle one after 5 s."""
    pool = _bedrock(monkeypatch, tmp_path).client._transport._pool
    assert pool._http2 is True and pool._keepalive_expiry == 50.0
    c = cfg()
    local = make_backend("local_server", c.llm.backends["local_server"], c.llm, tmp_path)
    assert local.client._transport._pool._http2 is False and local.client._transport._pool._keepalive_expiry == 5.0      # nothing changes for the others


def test_without_the_h2_package_it_falls_back_to_http11_and_says_so(monkeypatch, tmp_path, caplog):
    from dealer_core.nlu import openai_compat

    monkeypatch.setattr(openai_compat, "_http2_available", lambda: False)
    with caplog.at_level("WARNING"):
        pool = _bedrock(monkeypatch, tmp_path).client._transport._pool
    assert pool._http2 is False and "h2 package is missing" in caplog.text


def test_warming_a_connection_is_one_cheap_request_and_never_raises(monkeypatch, tmp_path):
    backend = _bedrock(monkeypatch, tmp_path)
    seen = []

    def ok(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"data": []})

    backend.client = httpx.Client(transport=httpx.MockTransport(ok), headers=dict(backend.client.headers))
    assert backend.warm_connection() is True and seen == [("GET", "/v1/models")]

    def boom(request):
        raise httpx.ConnectError("no route")

    backend.client = httpx.Client(transport=httpx.MockTransport(boom))
    assert backend.warm_connection() is False                              # the first real call just pays, as before


def test_the_chain_warms_the_active_backend_and_a_backend_with_nothing_to_warm_says_false():
    class Remote:
        name = "remote"
        warmed = 0

        def warm_connection(self):
            Remote.warmed += 1
            return True

    class Local:
        name = "local"

    chain = BackendChain([Remote(), Local()])
    assert chain.warm_connection() is True and Remote.warmed == 1
    chain.active = 1
    assert chain.warm_connection() is False                                 # (the base class does the same for a backend that has no method)

"""The defaults changed in batch A (27 Oct), and the Ollama context actually reaches Ollama."""
import re
from pathlib import Path

from dealer_core.app import ROOT
from dealer_core.config import load_config


def test_the_defaults_are_the_production_behaviour():
    v = load_config(environ={}).voice
    assert v.model_only is False and v.early_start_secs == 0.2 and v.tap_grace_secs == 2.0
    assert load_config(environ={"VA__VOICE__MODEL_ONLY": "true"}).voice.model_only is True          # the test mode is still one setting away


def test_every_change_can_be_switched_off_with_one_setting():
    v = load_config(environ={"VA__VOICE__EARLY_START_SECS": "0", "VA__VOICE__TAP_GRACE_SECS": "0"}).voice
    assert v.early_start_secs == 0 and v.tap_grace_secs == 0


def test_the_ollama_context_is_4096_and_the_other_backends_are_untouched():
    b = load_config(environ={}).llm.backends
    assert b["ollama"].num_ctx == 4096
    assert b["openvino_gpu"].runtime == "openvino" and b["local_server"].runtime == "openai"       # untouched


def test_the_context_size_is_what_ollama_is_asked_for():
    from dealer_core.nlu.ollama_backend import OllamaBackend

    cfg = load_config(environ={})
    be = OllamaBackend("ollama", cfg.llm.backends["ollama"], cfg.llm)
    seen = {}

    def chat(**kw):
        seen.update(kw)
        return type("R", (), {"message": type("M", (), {"content": "ok"})()})()
    be.client.chat = chat
    be.lead_in("system", "user")
    assert seen["options"]["num_ctx"] == 4096


def test_production_files_still_say_what_the_default_now_says():
    for rel in ("deploy/k8s/voice.yaml", "deploy/docker-compose.yml"):
        text = (ROOT / rel).read_text()
        assert re.search(r"MODEL_ONLY[^\n]{0,20}false", text), rel

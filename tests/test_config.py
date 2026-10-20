import pytest

from dealer_core.config import ConfigError, load_config


def test_defaults_are_local_and_cloud_off():
    cfg = load_config(environ={})
    assert not cfg.cloud_active
    assert cfg.llm.think is False
    assert cfg.latency_log.include_transcripts is False
    assert cfg.llm.backends[cfg.llm.backend].runtime == "openvino"
    assert cfg.llm.backends[cfg.llm.text_backend].runtime == "ollama"


def test_two_branches_with_per_branch_hours():
    cfg = load_config(environ={})
    assert len(cfg.branches) == 2
    assert cfg.branches[0].working_hours != cfg.branches[1].working_hours


def test_env_override_coerces_types():
    cfg = load_config(environ={"VA__LLM__BACKENDS__OLLAMA__NUM_THREAD": "8", "VA__LLM__BACKEND": "openvino_cpu"})
    assert cfg.llm.backends["ollama"].num_thread == 8
    assert cfg.llm.backend == "openvino_cpu"


def test_fallback_chain_has_no_duplicates_and_starts_with_primary():
    cfg = load_config(environ={})
    assert cfg.llm.chain() == ["openvino_gpu", "openvino_cpu", "ollama"]
    assert cfg.llm.chain("ollama") == ["ollama", "openvino_cpu"]


def test_undefined_backend_name_rejected():
    with pytest.raises(Exception, match="undefined backends"):
        load_config(environ={"VA__LLM__BACKEND": "nope"})


def test_cloud_requires_explicit_env_var():
    with pytest.raises(ConfigError):
        load_config(environ={"VA__LLM__BACKEND": "cloud_stub"})
    with pytest.raises(ConfigError):  # also when it is only a fallback
        load_config(environ={"VA__LLM__FALLBACKS": "[cloud_stub]"})


def test_cloud_allowed_with_flag_and_warns(capsys):
    cfg = load_config(environ={"VA__LLM__BACKEND": "cloud_stub", "VOICE_AGENT_ALLOW_CLOUD": "1"})
    assert cfg.cloud_active
    assert "CLOUD LLM MODE ENABLED" in capsys.readouterr().err


def test_unknown_key_rejected():
    with pytest.raises(Exception):
        load_config(environ={"VA__LLM__BACKENDS__OLLAMA__MODLE": "typo"})

import pytest

from dealer_core.config import ConfigError, load_config


def test_defaults_are_local_and_cloud_off():
    cfg = load_config(environ={})
    assert cfg.llm.provider == "local"
    assert not cfg.cloud_active
    assert cfg.llm.local.think is False
    assert cfg.latency_log.include_transcripts is False


def test_two_branches_with_per_branch_hours():
    cfg = load_config(environ={})
    assert len(cfg.branches) == 2
    assert cfg.branches[0].working_hours != cfg.branches[1].working_hours


def test_env_override_coerces_types():
    cfg = load_config(environ={"VA__LLM__LOCAL__NUM_THREAD": "8", "VA__LLM__LOCAL__MODEL": "qwen3:4b"})
    assert cfg.llm.local.num_thread == 8
    assert cfg.llm.local.model == "qwen3:4b"


def test_cloud_requires_explicit_env_var():
    with pytest.raises(ConfigError):
        load_config(environ={"VA__LLM__PROVIDER": "cloud"})


def test_cloud_allowed_with_flag_and_warns(capsys):
    cfg = load_config(environ={"VA__LLM__PROVIDER": "cloud", "VOICE_AGENT_ALLOW_CLOUD": "1"})
    assert cfg.cloud_active
    assert "CLOUD LLM MODE ENABLED" in capsys.readouterr().err


def test_unknown_key_rejected():
    with pytest.raises(Exception):
        load_config(environ={"VA__LLM__LOCAL__MODLE": "typo"})

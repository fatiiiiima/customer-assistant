"""Typed configuration: YAML defaults + VA__SECTION__KEY env overrides + .env.

Cloud LLM mode is refused unless explicitly enabled by env var, and prints a warning.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"
ALLOW_CLOUD_ENV = "VOICE_AGENT_ALLOW_CLOUD"
ENV_PREFIX = "VA__"


class ConfigError(RuntimeError):
    pass


class _Strict(BaseModel):
    # Unknown keys are typos; fail loudly instead of silently ignoring them.
    model_config = ConfigDict(extra="forbid")


class AppCfg(_Strict):
    log_dir: str = "logs"
    model_dir: str = "models"
    currency: str = "AED"
    distance_unit: str = "km"


class LatencyLogCfg(_Strict):
    console: bool = True
    jsonl_path: str = "logs/latency.jsonl"
    include_transcripts: bool = False  # privacy: off by default


class SttCfg(_Strict):
    backend: Literal["sherpa_onnx", "moonshine", "faster_whisper"] = "sherpa_onnx"
    model: str | None = None
    language: str = "en"
    hotwords_file: str | None = None


class LocalLlmCfg(_Strict):
    backend: Literal["ollama", "llama_cpp"] = "ollama"
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen3:1.7b"
    think: bool = False  # Qwen3 thinking must stay off
    temperature: float = 0.0
    num_ctx: int = 2048
    num_thread: int | None = None
    keep_alive: str = "30m"
    max_nlu_tokens: int = 64
    max_freeform_tokens: int = 60


class CloudLlmCfg(_Strict):
    model_env: str = "GEMINI_MODEL"
    api_key_env: str = "GEMINI_API_KEY"


class LlmCfg(_Strict):
    provider: Literal["local", "cloud"] = "local"
    local: LocalLlmCfg = Field(default_factory=LocalLlmCfg)
    cloud: CloudLlmCfg = Field(default_factory=CloudLlmCfg)


class TtsCfg(_Strict):
    backend: Literal["piper", "kokoro"] = "piper"
    voice: str | None = None
    cache_dir: str = "models/tts_cache"


class TurnCfg(_Strict):
    vad: str = "silero"
    turn_detector: str = "pipecat_smart_turn"
    fallback_silence_ms: int = 700


class BranchCfg(_Strict):
    id: str
    name: str
    timezone: str = "Asia/Dubai"
    # weekday (mon..sun) -> [open, close] "HH:MM". Missing weekday = closed.
    working_hours: dict[Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"], tuple[str, str]]


class TestDriveCfg(_Strict):
    # TO CONFIRM WITH SALES: placeholder rules.
    slot_minutes: int = 30
    licence_required: bool = True
    collect_fields: list[str] = ["name", "phone"]


class TierCfg(_Strict):
    up_to: float | None  # list price upper bound in AED; None = no upper bound
    max_discount_pct: float
    auto_accept_pct: float


class NegotiationCfg(_Strict):
    # PLACEHOLDERS: to be set by the business.
    max_rounds: int = 3
    counter_steps: int = 3
    tiers: list[TierCfg]

    @model_validator(mode="after")
    def _check(self) -> "NegotiationCfg":
        for t in self.tiers:
            if t.auto_accept_pct > t.max_discount_pct:
                raise ValueError("auto_accept_pct cannot exceed max_discount_pct")
        if not self.tiers or self.tiers[-1].up_to is not None:
            raise ValueError("last tier must have up_to: null")
        return self


class SendDetailsCfg(_Strict):
    default_channel: str = "whatsapp"
    mock: bool = True


class Config(_Strict):
    app: AppCfg = Field(default_factory=AppCfg)
    latency_log: LatencyLogCfg = Field(default_factory=LatencyLogCfg)
    stt: SttCfg = Field(default_factory=SttCfg)
    llm: LlmCfg = Field(default_factory=LlmCfg)
    tts: TtsCfg = Field(default_factory=TtsCfg)
    turn: TurnCfg = Field(default_factory=TurnCfg)
    branches: list[BranchCfg]
    test_drive: TestDriveCfg = Field(default_factory=TestDriveCfg)
    negotiation: NegotiationCfg
    send_details: SendDetailsCfg = Field(default_factory=SendDetailsCfg)

    @property
    def cloud_active(self) -> bool:
        return self.llm.provider == "cloud"


def _coerce(value: str) -> Any:
    """Env values are strings; let YAML decide whether they are bool/int/float/null."""
    return yaml.safe_load(value)


def _apply_env_overrides(data: dict[str, Any], environ: dict[str, str]) -> None:
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = [p.lower() for p in key[len(ENV_PREFIX):].split("__")]
        node = data
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = _coerce(raw)


def load_config(path: Path | None = None, environ: dict[str, str] | None = None) -> Config:
    """Load config. `environ` is injectable for tests; defaults to os.environ (after .env)."""
    if environ is None:
        load_dotenv(REPO_ROOT / ".env")
        environ = dict(os.environ)
    data = yaml.safe_load((path or DEFAULT_CONFIG).read_text(encoding="utf-8")) or {}
    _apply_env_overrides(data, environ)
    cfg = Config.model_validate(data)

    if cfg.cloud_active:
        if environ.get(ALLOW_CLOUD_ENV) != "1":
            raise ConfigError(
                f"llm.provider=cloud requires {ALLOW_CLOUD_ENV}=1. Cloud mode is dev-only "
                "and off by default."
            )
        print(
            "WARNING: CLOUD LLM MODE ENABLED. Prompts are sent to a third-party API. "
            "Use SYNTHETIC data only. STT and TTS remain local.",
            file=sys.stderr,
        )
    return cfg

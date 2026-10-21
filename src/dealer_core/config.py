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


class BackendCfg(_Strict):
    """One named NLU backend. Which fields apply depends on `runtime`."""

    runtime: Literal["openvino", "ollama", "cloud_stub"]
    model: str                                   # openvino: folder under <model_dir>/ov ; ollama: model tag
    device: Literal["CPU", "GPU", "NPU"] | None = None   # openvino only
    host: str = "http://127.0.0.1:11434"         # ollama only
    # auto: per-state prompt on GPU (a cache miss costs ~160 ms), static prompt on CPU/ollama (a miss costs 1.2 s+).
    prompt_mode: Literal["auto", "state", "static"] = "auto"
    timeout_s: float = 3.0                       # per-call budget; exceeding it triggers the next fallback
    num_ctx: int = 2048                          # ollama only
    keep_alive: str = "30m"                      # ollama only
    num_thread: int | None = None                # ollama only. Never pin threads for OpenVINO (hybrid cores)
    max_intent_tokens: int = 16
    max_json_tokens: int = 96
    max_answer_tokens: int = 60
    model_env: str | None = None                 # cloud_stub only: env var holding the model name (never hard-coded)
    api_key_env: str | None = None               # cloud_stub only


class LlmCfg(_Strict):
    backend: str = "openvino_gpu"                # voice path
    text_backend: str = "ollama"                 # Phase 1 text core and the website chat
    fallbacks: list[str] = Field(default_factory=lambda: ["openvino_cpu", "ollama"])
    think: bool = False                          # Qwen3 thinking must stay off
    temperature: float = 0.0
    backends: dict[str, BackendCfg]

    @model_validator(mode="after")
    def _names_exist(self) -> "LlmCfg":
        wanted = [self.backend, self.text_backend, *self.fallbacks]
        missing = [n for n in wanted if n not in self.backends]
        if missing:
            raise ValueError(f"llm refers to undefined backends: {missing}")
        return self

    def chain(self, primary: str | None = None) -> list[str]:
        """Backend names to try in order: the primary, then the fallbacks (no duplicates)."""
        order = [primary or self.backend, *self.fallbacks]
        return [n for i, n in enumerate(order) if n not in order[:i]]


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
    min_lead_minutes: int = 60       # earliest bookable slot is this far from now
    horizon_days: int = 14           # how far ahead slots are offered
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
    # Offers outside [low, high] percent of list price are treated as misheard numbers and re-confirmed, not negotiated.
    implausible_low_pct: float = 30.0
    implausible_high_pct: float = 150.0
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


class DialogueCfg(_Strict):
    variants: bool = False                       # true: use human-approved wording variants (config/phrase_variants.yaml)
    reply_mode: Literal["conversational", "hybrid", "template"] = "conversational"   # who writes the spoken reply (see dialogue/writer.py)
    judge: bool = True                           # conversational mode: a second strict yes/no check by the model for made-up statements
    high_risk_drafts: bool = True                # conversational mode: let the model word price/booking/hand-off replies too (exact figures verified)
    lead_in: bool = False                        # hybrid replies: a verified model-written sentence before low-risk templated answers
    draft: bool = False                          # EXPERIMENT: the model rewrites the whole low-risk reply (verified); overrides lead_in
    lead_in_intents: list[str] = Field(default_factory=lambda: ["search_cars", "ask_car_info", "out_of_scope"])
    variant_seed: int = 7                        # seeds the per-call pick, so a call can be reproduced


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
    dialogue: DialogueCfg = Field(default_factory=DialogueCfg)

    @property
    def cloud_active(self) -> bool:
        """True if any backend this config can ever select is the (disabled) cloud stub."""
        names = {self.llm.backend, self.llm.text_backend, *self.llm.fallbacks}
        return any(self.llm.backends[n].runtime == "cloud_stub" for n in names)


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
                f"a cloud_stub backend is selected, which requires {ALLOW_CLOUD_ENV}=1. Cloud mode is dev-only "
                "and off by default."
            )
        print(
            "WARNING: CLOUD LLM MODE ENABLED. Prompts are sent to a third-party API. "
            "Use SYNTHETIC data only. STT and TTS remain local.",
            file=sys.stderr,
        )
    return cfg

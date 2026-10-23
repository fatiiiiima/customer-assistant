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
    agent_name: str | None = None      # the advisor's name, said in the greeting (the website: Wazir)


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
    voice: str | None = "amy_medium"    # Piper voice folder name (see voice/tts.py VOICES). Owner choice. LICENCE UNCONFIRMED: internal prototype only (docs/LICENSES.md)
    speaker: int = 0                    # speaker id inside a multi-speaker voice (libritts_r has many; amy has one)
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
    enabled: bool = True              # False = fixed prices (the website): the agent never counters, never offers a discount
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
    enabled: bool = True              # False (the website): nothing is sent by message; the full specs are on the car's page
    default_channel: str = "whatsapp"
    mock: bool = True


class DialogueCfg(_Strict):
    variants: bool = False                       # true: use human-approved wording variants (config/phrase_variants.yaml)
    reply_mode: Literal["conversational", "hybrid", "template"] = "conversational"   # who writes the spoken reply (see dialogue/writer.py)
    judge: bool = False                          # conversational mode: a second strict yes/no check by the model for made-up statements
    retries: int = 0                             # extra tries when a model reply is rejected (each costs a full generation); speed profile: 0
    reply_deadline_ms: int = 0                   # stop the model-written reply after this long and speak the standard wording (0 = no limit; voice sets one)
    reply_max_words: int = 0                     # ask the model for at most this many words (0 = no limit; voice sets one)
    max_reply_tokens: int = 70                   # cap on the model-written reply (the main driver of reply time)
    decision_backend: str | None = "openvino_gpu"   # agent engine: the (smaller, faster) model that picks the tool each turn; None = the reply model
    agent_prefix: bool = False                   # agent engine, experimental: code speaks every fact at once; the model only continues with a short, strictly checked follow-up
    engine: str = "state_machine"                # "state_machine" (code decides, model words) | "agent" (the model picks tools, code runs them; experimental)
    high_risk_drafts: bool = True                # conversational mode: let the model word price/booking/hand-off replies too (exact figures verified)
    lead_in: bool = False                        # hybrid replies: a verified model-written sentence before low-risk templated answers
    draft: bool = False                          # EXPERIMENT: the model rewrites the whole low-risk reply (verified); overrides lead_in
    lead_in_intents: list[str] = Field(default_factory=lambda: ["search_cars", "ask_car_info", "out_of_scope"])
    variant_seed: int = 7                        # seeds the per-call pick, so a call can be reproduced


class VoiceCfg(_Strict):
    reply_deadline_ms: int = 4500                # model-written reply must be ready by then, else the standard wording is spoken
    reply_max_words: int = 30
    reply_max_tokens: int = 110
    opener: bool = False                          # the reply starts with a very short reaction, so the first audio does not wait for the facts
    barge_in: bool = True                         # the caller can talk over the agent (needs a headset or the browser's echo cancellation; off on open laptop speakers)
    model_only: bool = True                       # TEST MODE: only the model's own words are spoken: no standard-wording fallback, no appended facts or questions (the safety checks still drop unsafe sentences)
    high_risk_drafts: bool = True                 # the model also words price / booking / hand-off turns in voice (exact figures, dates and numbers are verified; the standard wording is the fallback)
    smart_turn: bool = True                       # Smart Turn v3 decides when you have finished (short VAD silence + a model), instead of a fixed silence
    smart_stop_secs: float = 0.2                  # the VAD silence used with smart_turn
    smart_fallback_secs: float = 0.6              # with smart_turn: silence after which the turn ends even if the model says "still going"
    smart_short_answer_secs: float = 0.0          # OFF. The length rule (an answer this short after a question counts as finished) cut 20 of 36 sentences in half instead of 4: "Toyota" and "Can I book..." are both short
    smart_peek: bool = False                      # OFF until real recordings show a gain (synthetic voices: no benefit, +3 cut sentences in 36). When Smart Turn hesitates, check the WORDS: a complete answer to the agent's last question ends the turn at once
    vad_start_secs: float = 0.1                   # speech must last this long to start a turn (0.2 lost one-word answers such as "Toyota")
    vad_min_volume: float = 0.1                   # volume gate of the detector (Pipecat's default 0.6 rejected short words)
    vad_confidence: float = 0.7
    number_stop_secs: float = 0.9                 # silence that ends the caller's turn right after the agent asked for a phone number: people dictate digits in groups with pauses ("050" ... "123" ... "4567")
    tap_stop_secs: float = 0.35                   # tap mode: a cut turn closes the mic mid-sentence, so it waits longer (24 real recordings: 0.35 s cut 2 sentences vs 4-5 at 0.2 s, for ~0.2 s more wait)
    silent_turn_fallback: bool = True             # when the model or a safety check leaves a turn with nothing to say, speak the code-written reference wording instead of silence
    stt_model: str = "whisper-base-int8-ov"      # OpenVINO Whisper folder under models/ov. base: ~0.1 s slower than tiny, noticeably fewer mishearings
    stop_secs: float = 0.3                       # silence that ends the caller's turn (VAD)


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
    voice: VoiceCfg = Field(default_factory=VoiceCfg)

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


def _merge(base: dict, over: dict) -> dict:
    """Deep merge of `over` onto `base`; lists and scalars in `over` replace (a branch list is replaced wholesale)."""
    for k, v in over.items():
        base[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def load_config(path: Path | None = None, environ: dict[str, str] | None = None, overlay: Path | None = None) -> Config:
    """Load config. `environ` is injectable for tests; defaults to os.environ (after .env). `overlay` is a second YAML merged on top
    (config/portal.yaml: the website's showroom hours and fixed prices)."""
    if environ is None:
        load_dotenv(REPO_ROOT / ".env")
        environ = dict(os.environ)
    data = yaml.safe_load((path or DEFAULT_CONFIG).read_text(encoding="utf-8")) or {}
    if overlay is not None:
        data = _merge(data, yaml.safe_load(overlay.read_text(encoding="utf-8")) or {})
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

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
    database_url: str = ""             # postgresql://user:pass@host:port/db: bookings shared by every worker (repos/pg_store.py). Empty: in-memory SQLite (one process)
    db_pool_size: int = 8              # the most connections ONE worker process opens to PostgreSQL, shared by bookings, stock, call snapshots and the registry (was 8 + 4: ARCH_REVIEW B6). Opened only when needed
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

    runtime: Literal["openvino", "ollama", "openai", "cloud_stub"]       # openai = any OpenAI-compatible server on the company network (vLLM, OVMS, TGI, llama.cpp)
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
    batching: bool = False                       # openvino only: continuous batching + prefix cache, so many callers are served together (nlu/ov_batching.py)
    kv_cache_gb: float = 2.0                     # openvino batching only: memory for the shared KV cache
    max_parallel: int = 32                       # openvino batching only: requests in flight at once
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
    backend: Literal["piper", "kokoro"] = "kokoro"
    kokoro_runtime: Literal["auto", "openvino", "onnxruntime"] = "auto"   # auto: OpenVINO on Intel / AMD, ONNX Runtime on Apple chips (OpenVINO distorts the voice there)
    kokoro_voice: str = "sarah"         # kokoro backend: sarah / heart / bella / emma (female), michael / adam / puck / eric / liam / onyx (US male), fenrir, george / lewis (UK male)
    voice: str | None = "joe"           # Piper voice folder name (see voice/tts.py VOICES). Owner choice: Joe (CC0)
    speaker: int = 0                    # speaker id inside a multi-speaker voice (libritts_r has many; amy has one)
    cache_dir: str = "models/tts_cache"
    copies: int = 2                     # synthesizers shared by all calls (8 sentences at once: worst 2.35 s with one copy, 1.35 s with two)
    threads: int | None = None          # CPU threads per synthesizer (None = the engine's own default: Kokoro all cores, Piper 4). MEASURED 27 Oct on an M3 Pro: capping only slows it (one sentence 629 ms uncapped,
                                        # 798 at 4 threads, 975 at 3, 1350 at 2). Left unset; a setting for hardware where several workers share a processor and the measurement says otherwise


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


class LatencyCfg(_Strict):
    """How long a caller may wait on the model: the three numbers that bound it, in ONE place (they used to be spread over `dialogue` and `voice`).

    WHERE THE VALUES COME FROM. Not from the production hardware. They were set on the owner's laptop (Ollama / OpenVINO on an Intel iGPU or an M3 Pro, 4B model) in the 27-28 Oct
    reviews (docs/ARCH_REVIEW.md sections 9, 11 and 14). The values that belong here are the ones measured on the model server the deployment really uses, by the procedure in
    docs/PHASE0_RUNBOOK.md (`scripts/capacity_probe.py`, `scripts/load_test.py`: the first-sentence time at 1 to 100 callers at once, and the governor level each turn ran at).
    Phase 0 numbers: NOT YET MEASURED (no GPU run has been made). Until they exist these stay as they were and must not be read as production figures. When they exist, set them
    here, in the deployment's overlay or with VA__LATENCY__<KEY>, and write the run into docs/ARCH_REVIEW.md.

    The old names (`dialogue.model_budget_ms`, `voice.decide_deadline_ms`, `voice.first_sentence_deadline_ms`) are still read, so an existing overlay or environment keeps working;
    the new name wins when both are given (`Config._move_latency_keys`)."""

    model_budget_ms: float = 700.0               # governor: time to the model's first sentence that still keeps the reply under ~1 s; a model slower than this is stepped down to the prefix level
    decide_deadline_ms: float = 1200.0           # agent: the model's tool decision must come within this, else the code decides (was 5 s). With dialogue.single_call it also bounds the
                                                 # fall-back decision after a malformed one-call answer
    first_sentence_deadline_ms: int = 1500       # a model-written reply must have its FIRST usable sentence by this time (about twice the budget), else the model is given up on and the code's
                                                 # own wording is spoken at once, with no second try. With dialogue.single_call it is also the deadline of the one call, which carries the decision
                                                 # and the first sentence together. 0 = no limit


class DialogueCfg(_Strict):
    variants: bool = False                      # true: use human-approved wording variants (config/phrase_variants.yaml)
    reply_mode: Literal["conversational", "hybrid", "template"] = "conversational"   # who writes the spoken reply (see dialogue/writer.py)
    judge: bool = False                          # conversational mode: a second strict yes/no check by the model for made-up statements
    retries: int = 0                             # extra tries when a model reply is rejected (each costs a full generation); speed profile: 0
    reply_deadline_ms: int = 0                   # stop the model-written reply after this long and speak the standard wording (0 = no limit; voice sets one)
    reply_max_words: int = 0                     # ask the model for at most this many words (0 = no limit; voice sets one)
    max_reply_tokens: int = 70                   # cap on the model-written reply (the main driver of reply time)
    decision_backend: str | None = "openvino_gpu"   # agent engine: the (smaller, faster) model that picks the tool each turn; None = the reply model
    quality: str = "auto"                        # agent engine: auto (per turn, from the model's live speed) | full | prefix | code (agent/governor.py)
    model_shed_after_ms: float = 450.0           # auto: past this measured decision time a growing share of turns skip the model (load shedding; 0 = never)
    explore_every: int = 0                       # auto: test the model-written level on every Nth caller's turn while stepped down (0 = never; the voice server measures with a background probe instead)
    probe_every_s: float = 20.0                  # voice server: measure the model with a small request of the same shape as a model-written turn this often, only when no turn is using it (0 = off)
    probe_max_in_flight: int = 0                 # ...or when at most this many turns are using the model. 0 = only when none is. The plan (B5) asked for 2; in the scale simulation 2 and 4 recovered no faster after a GPU outage and sent 33-37% more probes to an overloaded model (ARCH_REVIEW section 20), so the default stays 0
    max_model_turns: int = 0                     # auto: turns using the model at once above which replies are spoken by code (0 = no cap)
    openers: bool = True                         # agent engine: fact replies start with a short acknowledgement ('Sure,') that the voice has ready, so sound starts at once
    agent_prefix: bool = False                   # agent engine, experimental: code speaks every fact at once; the model only continues with a short, strictly checked follow-up
    engine: str = "state_machine"                # "state_machine" (code decides, model words) | "agent" (the model picks tools, code runs them; experimental)
    high_risk_drafts: bool = True                # conversational mode: let the model word price/booking/hand-off replies too (exact figures verified)
    lead_in: bool = False                        # hybrid replies: a verified model-written sentence before low-risk templated answers
    draft: bool = False                          # EXPERIMENT: the model rewrites the whole low-risk reply (verified); overrides lead_in
    lead_in_intents: list[str] = Field(default_factory=lambda: ["search_cars", "ask_car_info", "out_of_scope"])
    variant_seed: int = 7                        # seeds the per-call pick, so a call can be reproduced


class VoiceCfg(_Strict):
    reply_deadline_ms: int = 4500                # model-written reply must be ready by then, else the standard wording is spoken
    history_lines: int = 6                       # lines of the conversation the model is shown for a reply (the text chat keeps 10)
    reply_max_words: int = 30
    reply_max_tokens: int = 110
    opener: bool = False                          # the reply starts with a very short reaction, so the first audio does not wait for the facts
    barge_in: bool = True                         # the caller can talk over the agent (needs a headset or the browser's echo cancellation; off on open laptop speakers)
    model_only: bool = False                      # False (the production behaviour): a reply the model leaves short, or a sentence the checks drop, is completed with the code's own facts and question.
                                                  # True = TEST MODE: only the model's own words are spoken, nothing in their place (27 Oct: a reply could end after its opener). Was True by default
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
    tap_grace_secs: float = 2.0                   # tap mode: after the turn is over the microphone stays open until the reply starts to play, at most this long. A caller who was only pausing carries on and
                                                  # the two halves are ONE sentence (before: the page closed the microphone at once and the rest of the sentence was lost). 0 = the old behaviour
    tap_stop_secs: float = 0.6                    # tap mode: a cut turn closes the mic mid-sentence and the rest is never heard, so it waits longer. 24 real recordings: 0.2 s cut 4-5 sentences, 0.35 s cut 2, 0.6 s cut 0.
                                                  # Cost: about +0.3 s on every turn vs 0.35 s (synthetic callers, median 547 -> 867 ms; docs/VOICE.md). Put it back with VA__VOICE__TAP_STOP_SECS=0.35
    silent_turn_fallback: bool = True             # when the model or a safety check leaves a turn with nothing to say, speak the code-written reference wording instead of silence
    stt_model: str = "whisper-base-int8-ov"      # OpenVINO Whisper folder under models/ov. base: ~0.1 s slower than tiny, noticeably fewer mishearings
    stt_device: str = "CPU"                      # CPU | GPU (NPU cannot take the recognizer hint). small.en on the GPU: 21/24 of the owner's recordings in 0.17 s
    stt_threads: int | None = None               # CPU threads for the recognizer (None = OpenVINO's default, all cores). MEASURED 27 Oct on an M3 Pro: one turn 242 ms uncapped, 296 at 4, 382 at 3, 547 at 2;
                                                 # with THREE recognizers running at once: 595 ms uncapped vs 617 ms capped to 3 each, so capping gains nothing here (the CPU is the limit). Unset by default
    stt_precision: str | None = "f32"            # CPU recognizer math. OpenVINO's default on Apple chips is float16, where Whisper base int8 loops ("No!!!!...") on many turns; f32 fixed all 9 of 9 (~50-100 ms slower). Empty (VA__VOICE__STT_PRECISION=) = OpenVINO's default
    stop_secs: float = 0.3                       # silence that ends the caller's turn (VAD)
    min_rms: float = 0.005                        # a segment whose loudest 0.3 s is below this is room noise and is not recognised. NOT raised: at 0.012 the owner's 10%-volume clips went 20/24 -> 0/24 heard (docs/SCALING.md)
    end_call_min_rms: float = 0.03                # a goodbye heard at a level below this is not believed: the agent asks "are you still there?" instead of hanging up. Real turns in the 27 Oct log were 0.065-0.26; the two false goodbyes 0.013 and 0.015
    speak_up_below: float = 0.02                 # a segment whose loudest 0.3 s is below this is not acted on: the agent asks the caller to speak up. Measured 28 Oct on 252 readings of the owner's own turns made quieter
                                                 # (room noise kept): below 0.02 about 80% of the turns were badly wrong with the base model and 65-80% with small.en, and only 6 of 79-87 turns that were
                                                 # heard right were stopped. 0 = off
    speak_up_suspect_below: float = 0.05         # between this and the one above, only when the words look like a recogniser's invention ("please subscribe", "comments below", a lone "okay"). 0 = off
    admission: bool = True                       # voice server: the call cap follows the measured reply times (server/admission.py); --max-calls is the ceiling
    admission_target_ms: float = 1000.0          # the 90th percentile of replies the cap is held to
    early_start_secs: float = 0.0                 # speculate: also start recognising (and the model's decision) after this much real quiet INSIDE a turn, before the turn is confirmed over. In tap mode the
                                                  # voice detector waits 0.6 s, so this answers up to 0.4 s sooner; wasted work if the caller was only pausing (at most 3 times a turn). 0 = only when Smart Turn holds a pause
    speculate: bool = True                       # smart_turn: while it holds a pause as "still going", recognise the turn and make the model's decision in the
                                                 # background; used as they are if the caller has in fact finished (no new speech before the fallback silence)
    bridge_after_ms: float = 450.0               # agent engine: when the model's decision (+ first sentence) is measured to take longer than this, a pre-rendered
                                                 # "Let me check." plays at once while it is made. ~1 s target minus end-of-turn (0.2-0.35 s), recognition and first audio. 0 = off
    stt_copies: int = 1                          # local recognizer copies per worker: calls in one worker transcribe in parallel instead of queueing on one lock
    stt_url: str = ""                            # recognition on a shared GPU server instead of in the worker (OpenAI-compatible /v1/audio/transcriptions:
                                                 # vLLM with a Whisper model, speaches / faster-whisper-server). Empty = local OpenVINO Whisper
    stt_remote_model: str = "openai/whisper-large-v3-turbo"   # the name that server knows the model by
    stt_timeout_s: float = 3.0                   # a remote transcription slower than this counts as "not heard" (the caller is asked again)


_OLD_LATENCY_KEYS = (("dialogue", "model_budget_ms"), ("voice", "decide_deadline_ms"), ("voice", "first_sentence_deadline_ms"))


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
    latency: LatencyCfg = Field(default_factory=LatencyCfg)

    @model_validator(mode="before")
    @classmethod
    def _move_latency_keys(cls, data: Any) -> Any:
        """The three latency numbers used to live in `dialogue` and `voice`: an overlay or environment variable that still uses the old name is moved to `latency`
        (unless the new name is also given, which wins), so nothing that worked stops working."""
        if not isinstance(data, dict):
            return data
        for section, key in _OLD_LATENCY_KEYS:
            old = data.get(section)
            if isinstance(old, dict) and key in old:
                data = {**data, section: {k: v for k, v in old.items() if k != key}}
                latency = dict(data.get("latency") or {})
                latency.setdefault(key, old[key])
                data["latency"] = latency
        return data

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

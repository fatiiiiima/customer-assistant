"""Builds the voice pipeline:  audio in -> VAD -> STT -> DealerBrain (dialogue engine) -> TTS -> audio out.

All local. The language model runs on the iGPU, Whisper on the CPU, Piper on the CPU (see docs/PHASE0B_REPORT.md section 9).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameProcessor

from ..app import ROOT, App
from ..config import Config
from .brain import DealerBrain
from .gate import HalfDuplexGate
from .services import OpenVinoWhisperSTTService, SherpaPiperTTSService
from .prompts import hotword_groups, stt_prompt
from .stt import WhisperOpenVino, hotword_terms
from .tts import PiperSherpa

STT_RATE = 16000


@dataclass
class VoiceParts:
    stt_engine: WhisperOpenVino
    tts_engine: PiperSherpa
    hotword_groups: dict = field(default_factory=dict)


def load_engines(cfg: Config, stt_model: str | None = None, voice: str | None = None) -> VoiceParts:
    """Load and warm the local STT and TTS engines (about 5 s, once per process)."""
    models = ROOT / cfg.app.model_dir
    stt = WhisperOpenVino(models / "ov" / (stt_model or cfg.voice.stt_model), models / "ov_cache", "CPU", hotword_terms(ROOT / cfg.stt.hotwords_file))
    tts = PiperSherpa(models, voice or cfg.tts.voice or "joe", speaker=cfg.tts.speaker if voice is None else 0)
    stt.warm_up()
    tts.warm_up()
    return VoiceParts(stt, tts, hotword_groups(ROOT / cfg.stt.hotwords_file) if cfg.stt.hotwords_file else {})


def apply_voice_profile(app: App, cfg: Config) -> None:
    """Voice replies are short and time-boxed: ask for ~20 words, cap the tokens, and STOP the model at the deadline (the standard wording is then
    spoken). Text chat keeps its longer, richer replies."""
    w = app.engine.reply_writer
    if w is None:
        return
    w.max_words = cfg.voice.reply_max_words
    w.deadline_s = cfg.voice.reply_deadline_ms / 1000.0 or None
    w.max_tokens = min(w.max_tokens, cfg.voice.reply_max_tokens)
    w.retries = 0
    w.opener = cfg.voice.opener
    app.engine.stream_high_risk = cfg.voice.high_risk_drafts
    app.engine.model_only = w.model_only = cfg.voice.model_only
    app.engine.silent_fallback = cfg.voice.silent_turn_fallback


def build_pipeline(app: App, parts: VoiceParts, source: FrameProcessor, sink: FrameProcessor, *, stop_secs: float = 0.35,
                   speak_opening: bool = True, probes: dict[str, FrameProcessor] | None = None, on_turn=None, half_duplex: bool = False, on_heard=None, on_say=None,
                   smart_turn: bool = False, smart_fallback_secs: float = 1.0, gate_tail_secs: float = 0.5, barge_in: bool = False,
                   smart_short_answer_secs: float | None = None, smart_peek: bool | None = None, mic_gate=None, on_user_stop=None):
    """`source` produces InputAudioRawFrame (microphone or a file); `sink` consumes the speech (speaker or a recorder).
    `probes` optionally inserts timing probes after: vad, stt, brain, tts."""
    probes = probes or {}
    vcfg = app.service.cfg.voice
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(sample_rate=STT_RATE, params=VADParams(
        confidence=vcfg.vad_confidence, start_secs=vcfg.vad_start_secs, stop_secs=stop_secs, min_volume=vcfg.vad_min_volume)))
    stt = OpenVinoWhisperSTTService(engine=parts.stt_engine)
    brain = DealerBrain(app.engine, speak_opening=speak_opening, on_turn=on_turn, on_heard=on_heard, on_say=on_say)
    stt.prompt_for = lambda: stt_prompt(brain.session.state, brain.session.last_agent(), parts.hotword_groups, app.inventory.vocabulary())
    tts = SherpaPiperTTSService(engine=parts.tts_engine)
    stages: list[FrameProcessor] = [source] + ([mic_gate] if mic_gate is not None else []) + ([HalfDuplexGate(gate_tail_secs)] if half_duplex else []) + [vad]
    if smart_turn:                                           # `stop_secs` is then the short VAD silence; Smart Turn decides whether the turn is really over
        from .endpoint import SmartEndpointer

        endpointer = SmartEndpointer(STT_RATE, smart_fallback_secs, short_answer_secs=vcfg.smart_short_answer_secs if smart_short_answer_secs is None else smart_short_answer_secs)
        endpointer.asked_question = lambda: brain.session.last_agent().rstrip().endswith("?")        # the agent's last turn ended on a question
        if vcfg.smart_peek if smart_peek is None else smart_peek:
            from zoneinfo import ZoneInfo

            from .answers import is_complete_answer

            tz = ZoneInfo(app.service.cfg.branches[0].timezone)
            endpointer.peek = lambda pcm: asyncio.to_thread(lambda: parts.stt_engine.transcribe(pcm, stt.prompt_for())[0])
            endpointer.is_answer = lambda text: is_complete_answer(text, brain.session.last_agent(), brain.session.state, brain.session.shown,
                                                                   app.inventory.vocabulary(), app.service.clock(), tz)
        stages.append(endpointer)
    if on_user_stop is not None:                              # tap-to-talk: told when the turn is over, so the microphone can be closed
        from .micgate import StopNotifier

        stages.append(StopNotifier(on_user_stop))
    if barge_in and not half_duplex:                          # the caller may talk over the agent (needs a headset or echo cancellation)
        from .bargein import BargeIn

        stages.append(BargeIn())
    for name, proc in (("vad", None), ("stt", stt), ("brain", brain), ("tts", tts)):
        if proc is not None:
            stages.append(proc)
        if name in probes:
            stages.append(probes[name])
    stages.append(sink)
    return Pipeline(stages), brain

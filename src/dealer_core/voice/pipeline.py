"""Builds the voice pipeline:  audio in -> VAD -> STT -> DealerBrain (dialogue engine) -> TTS -> audio out.

All local. The language model runs on the iGPU, Whisper on the CPU, Piper on the CPU (see docs/PHASE0B_REPORT.md section 9).
"""
from __future__ import annotations

from dataclasses import dataclass
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
from .stt import WhisperOpenVino, hotword_terms
from .tts import PiperSherpa

STT_RATE = 16000


@dataclass
class VoiceParts:
    stt_engine: WhisperOpenVino
    tts_engine: PiperSherpa


def load_engines(cfg: Config, stt_model: str | None = None, voice: str | None = None) -> VoiceParts:
    """Load and warm the local STT and TTS engines (about 5 s, once per process)."""
    models = ROOT / cfg.app.model_dir
    stt = WhisperOpenVino(models / "ov" / (stt_model or cfg.voice.stt_model), models / "ov_cache", "CPU", hotword_terms(ROOT / cfg.stt.hotwords_file))
    tts = PiperSherpa(models, voice or cfg.tts.voice or "joe", speaker=cfg.tts.speaker if voice is None else 0)
    stt.warm_up()
    tts.warm_up()
    return VoiceParts(stt, tts)


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


def build_pipeline(app: App, parts: VoiceParts, source: FrameProcessor, sink: FrameProcessor, *, stop_secs: float = 0.35,
                   speak_opening: bool = True, probes: dict[str, FrameProcessor] | None = None, on_turn=None, half_duplex: bool = False, on_heard=None, on_say=None,
                   smart_turn: bool = False, smart_fallback_secs: float = 1.0):
    """`source` produces InputAudioRawFrame (microphone or a file); `sink` consumes the speech (speaker or a recorder).
    `probes` optionally inserts timing probes after: vad, stt, brain, tts."""
    probes = probes or {}
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(sample_rate=STT_RATE, params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=stop_secs)))
    stt = OpenVinoWhisperSTTService(engine=parts.stt_engine)
    brain = DealerBrain(app.engine, speak_opening=speak_opening, on_turn=on_turn, on_heard=on_heard, on_say=on_say)
    tts = SherpaPiperTTSService(engine=parts.tts_engine)
    stages: list[FrameProcessor] = [source] + ([HalfDuplexGate()] if half_duplex else []) + [vad]
    if smart_turn:                                           # `stop_secs` is then the short VAD silence; Smart Turn decides whether the turn is really over
        from .endpoint import SmartEndpointer

        stages.append(SmartEndpointer(STT_RATE, smart_fallback_secs))
    for name, proc in (("vad", None), ("stt", stt), ("brain", brain), ("tts", tts)):
        if proc is not None:
            stages.append(proc)
        if name in probes:
            stages.append(probes[name])
    stages.append(sink)
    return Pipeline(stages), brain

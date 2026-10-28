"""Builds the voice pipeline:  audio in -> VAD -> STT -> DealerBrain (dialogue engine) -> TTS -> audio out.

All local. The language model runs on the iGPU, Whisper on the CPU, Piper on the CPU (see docs/PHASE0B_REPORT.md section 9).
"""
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameProcessor

from ..app import ROOT, App
from ..config import Config
from .audiotap import AudioTap
from .brain import DealerBrain
from .mic_health import MicHealth
from .gate import HalfDuplexGate
from .services import OpenVinoWhisperSTTService, SherpaPiperTTSService
from .prompts import hotword_groups, stt_prompt
from .stt import RemoteWhisper, WhisperOpenVino, hotword_terms
from .tts import PiperSherpa
from ..agent.engine import BRIDGES, CUT_OFF, STILL_THERE
from ..agent.wording import OPENERS

STT_RATE = 16000


@dataclass
class VoiceParts:
    stt_engine: WhisperOpenVino | RemoteWhisper
    tts_engine: PiperSherpa
    hotword_groups: dict = field(default_factory=dict)


def load_engines(cfg: Config, stt_model: str | None = None, voice: str | None = None) -> VoiceParts:
    """Load and warm the local STT and TTS engines (about 5 s, once per process)."""
    models = ROOT / cfg.app.model_dir
    hotwords = hotword_terms(ROOT / cfg.stt.hotwords_file)
    if cfg.voice.stt_url:                               # a shared, batched recognizer on a GPU server (production): no Whisper in the worker
        stt = RemoteWhisper(cfg.voice.stt_url, cfg.voice.stt_remote_model, cfg.voice.stt_timeout_s, hotwords)
    else:
        stt = WhisperOpenVino(models / "ov" / (stt_model or cfg.voice.stt_model), models / "ov_cache", cfg.voice.stt_device, hotwords, copies=cfg.voice.stt_copies,
                              precision=cfg.voice.stt_precision, threads=cfg.voice.stt_threads)
    if cfg.tts.backend == "kokoro":                     # Kokoro-82M on the CPU with OpenVINO (voice/kokoro_tts.py)
        from .kokoro_tts import KokoroOV

        tts = KokoroOV(models, voice or cfg.tts.kokoro_voice, runtime=cfg.tts.kokoro_runtime, threads=cfg.tts.threads)
    else:
        tts = PiperSherpa(models, voice or cfg.tts.voice or "joe", speaker=cfg.tts.speaker if voice is None else 0, copies=cfg.tts.copies,
                         threads=cfg.tts.threads or 4)
    stt.warm_up()
    tts.warm_up()
    if hasattr(tts, "preload"):                         # the phrases said all the time play with no wait
        tts.preload(common_phrases(cfg))
    return VoiceParts(stt, tts, hotword_groups(ROOT / cfg.stt.hotwords_file) if cfg.stt.hotwords_file else {})


def common_phrases(cfg: Config) -> list[str]:
    """What Wazir says word for word again and again: rendered once at start-up by voices that are slow to make speech."""
    name = cfg.app.agent_name
    who = f"I'm {name}, and I'm here to help you find your perfect car." if name else "How can I help you find a car?"
    return [f"Hi, welcome to the AGMC dealership! {who} What are you looking for today?",
            "Is there anything else I can help with?", "What else can I do for you?", "Anything else you'd like to know?",
            "Would you like to come and see it?", "Want to book a test drive?", "Shall I set up a test drive for you?",
            "Sure! What day would you like to come in?", "Of course. What day and time suit you?", "What day and time work for you?",
            "When would you like to come in?", "Perfect. Can I take your name and mobile number?", "Great. What's your name and mobile number?",
            "Great. And your name, please?", "Lovely. Who should I put the booking under?", "Which one would you like to hear more about?",
            "Could you tell me your budget or a make?", "What budget do you have in mind, or is there a model you like?",
            "Sorry, I didn't catch that. Could you say it again?", "Thank you for calling AGMC. Goodbye!", "You're welcome!",
            "Which suits you better?", "Shall I book it?", "Is that all correct now?", "Just bring your driving licence along.",
            "No problem at all, take your time.", "Our prices are fixed,"] + OPENERS + [
            "Sure.", "Sure!", "Okay.", "Great,", "Great.", "Great!", "Perfect,", "Perfect.", "Lovely,", "Lovely.", "Thanks,", "Thanks.", "Done,",
            "All set,", "Of course,", "Of course.", "No problem,", "No problem.", "Great choice.", "Good choice,", "Absolutely,", "Absolutely!",
            "Yes,", "Sorry,", "Got it,", "Got it.", "Hmm,"] + BRIDGES + CUT_OFF + STILL_THERE + ["Is there a colour you'd like?"]


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
    if hasattr(app.engine, "decide_deadline_s"):              # agent engine: a slow decision is replaced by the code's own, not waited for
        app.engine.decide_deadline_s = cfg.voice.decide_deadline_ms / 1000
    if hasattr(app.engine, "bridge_after_ms"):                # agent engine: "Let me check." while a slow model decides (voice only; text chat does not wait on audio)
        app.engine.bridge_after_ms = cfg.voice.bridge_after_ms


_NUMBER_ASK = re.compile(r"\b(?:phone|mobile|number|digits)\b", re.I)


def asks_for_number(session) -> bool:
    """Did the agent's last turn ask for the caller's phone number?"""
    from ..dialogue.states import State
    from ..nlu.expected import last_question

    q = last_question(session.last_agent())
    return bool(q and _NUMBER_ASK.search(q)) or (session.state is State.CONTACT and bool(_NUMBER_ASK.search(session.last_agent())))


def build_pipeline(app: App, parts: VoiceParts, source: FrameProcessor, sink: FrameProcessor, *, stop_secs: float = 0.35,
                   speak_opening: bool = True, probes: dict[str, FrameProcessor] | None = None, on_turn=None, half_duplex: bool = False, on_heard=None, on_say=None,
                   smart_turn: bool = False, smart_fallback_secs: float = 1.0, gate_tail_secs: float = 0.5, barge_in: bool = False,
                   smart_short_answer_secs: float | None = None, smart_peek: bool | None = None, mic_gate=None, on_user_stop=None,
                   seed_agent_line: str | None = None, seed_state=None):
    """`source` produces InputAudioRawFrame (microphone or a file); `sink` consumes the speech (speaker or a recorder).
    `probes` optionally inserts timing probes after: vad, stt, brain, tts."""
    probes = probes or {}
    vcfg = app.service.cfg.voice
    vad_analyzer = SileroVADAnalyzer(sample_rate=STT_RATE, params=VADParams(
        confidence=vcfg.vad_confidence, start_secs=vcfg.vad_start_secs, stop_secs=stop_secs, min_volume=vcfg.vad_min_volume))
    vad = VADProcessor(vad_analyzer=vad_analyzer)
    current_stop = [stop_secs]

    def follow_the_question() -> None:
        """Right after the agent asked for a phone number the silence that ends the turn is longer (digits are dictated in groups with pauses)."""
        want = max(stop_secs, vcfg.number_stop_secs) if asks_for_number(brain.session) else stop_secs
        if want != current_stop[0]:
            current_stop[0] = want
            params = VADParams(confidence=vcfg.vad_confidence, start_secs=vcfg.vad_start_secs, stop_secs=want, min_volume=vcfg.vad_min_volume)
            if vad_analyzer.sample_rate:
                vad_analyzer.set_params(params)
            else:                                                 # not started yet: the analyzer applies these params when the stream starts
                vad_analyzer._params = params

    def after_turn(result) -> None:
        follow_the_question()
        if on_turn:
            on_turn(result)

    stt = OpenVinoWhisperSTTService(engine=parts.stt_engine, min_rms=vcfg.min_rms)
    brain = DealerBrain(app.engine, speak_opening=speak_opening, on_turn=after_turn, on_heard=on_heard, on_say=on_say,
                        level_of=lambda: stt.last_level, quiet_below=vcfg.end_call_min_rms)
    if seed_agent_line:                                       # tests and studies: start as if the agent had just said this
        brain.session.remember("agent", seed_agent_line)
        brain.session.state = seed_state if seed_state is not None else brain.session.state
    follow_the_question()
    stt.prompt_for = lambda: stt_prompt(brain.session.state, brain.session.last_agent(), parts.hotword_groups, app.inventory.vocabulary())
    tts = SherpaPiperTTSService(engine=parts.tts_engine)
    dump = os.environ.get("VA_DUMP_AUDIO_DIR")                  # diagnostic only (voice/audiotap.py): copy what goes in to the detector and what goes out to the page
    stages: list[FrameProcessor] = [source] + ([mic_gate] if mic_gate is not None else []) + ([HalfDuplexGate(gate_tail_secs)] if half_duplex else []) + (
        [AudioTap(dump, "in", InputAudioRawFrame)] if dump else []) + [vad]
    if smart_turn:                                           # `stop_secs` is then the short VAD silence; Smart Turn decides whether the turn is really over
        from .endpoint import SmartEndpointer

        endpointer = SmartEndpointer(STT_RATE, smart_fallback_secs, short_answer_secs=vcfg.smart_short_answer_secs if smart_short_answer_secs is None else smart_short_answer_secs,
                                     early_start_secs=vcfg.early_start_secs if vcfg.speculate else 0.0)
        endpointer.asked_question = lambda: brain.session.last_agent().rstrip().endswith("?")        # the agent's last turn ended on a question
        if vcfg.smart_peek if smart_peek is None else smart_peek:
            from zoneinfo import ZoneInfo

            from .answers import is_complete_answer

            tz = ZoneInfo(app.service.cfg.branches[0].timezone)
            endpointer.peek = lambda pcm: asyncio.to_thread(lambda: parts.stt_engine.transcribe(pcm, stt.prompt_for())[0])
            endpointer.is_answer = lambda text: is_complete_answer(text, brain.session.last_agent(), brain.session.state, brain.session.shown,
                                                                   app.inventory.vocabulary(), app.service.clock(), tz)
        if vcfg.speculate:                                    # recognise (and decide) during Smart Turn's hold; used if the caller has finished
            from .services import clean_transcript
            from .speculate import Speculation

            spec = Speculation(lambda pcm: clean_transcript(parts.stt_engine.transcribe(pcm, stt.prompt_for())[0]),
                               (lambda text: app.engine.predecide(brain.session, text)) if hasattr(app.engine, "predecide") else None,
                               lambda: len(brain.session.history))
            endpointer.speculation = stt.speculation = brain.speculation = spec
        stages.append(endpointer)
    stages.append(MicHealth())                                # one log line per turn, only when the microphone audio arrived damaged (voice/mic_health.py)
    if mic_gate is not None and on_user_stop is not None:     # tap-to-talk: a caller who carries on before the reply is heard continues the same sentence
        from .micgate import ContinuationGuard

        stages.append(ContinuationGuard(mic_gate, brain))
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
    if dump:
        stages.append(AudioTap(dump, "out", OutputAudioRawFrame))
    stages.append(sink)
    return Pipeline(stages), brain

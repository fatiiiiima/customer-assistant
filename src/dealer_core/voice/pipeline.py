"""Builds the voice pipeline:  audio in -> VAD -> STT -> DealerBrain (dialogue engine) -> TTS -> audio out.

All local. The language model runs on the iGPU, Whisper on the CPU, Piper on the CPU (see docs/PHASE0B_REPORT.md section 9).
"""
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameProcessor

from ..app import ROOT, App
from ..config import Config
from .audiotap import AudioTap
from .brain import SILENT_GOODBYE, WELCOME_BACK, DealerBrain
from .mic_health import MicHealth
from .gate import HalfDuplexGate
from .services import OpenVinoWhisperSTTService, SherpaPiperTTSService
from .prompts import hotword_groups, stt_prompt
from .stt import RemoteWhisper, WhisperOpenVino, hotword_terms
from .tts import PiperSherpa
from ..agent.wording import NARROW_QUESTIONS
from ..agent.engine import BRIDGES, CUT_OFF, STILL_THERE
from ..agent.wording import OPENERS
from ..dialogue.templates import greeting

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
    if cfg.tts.backend in ("kokoro", "chatterbox", "pocket"):   # Kokoro-82M on the CPU with OpenVINO (voice/kokoro_tts.py)
        from .kokoro_tts import KokoroOV

        tts = KokoroOV(models, voice or cfg.tts.kokoro_voice, runtime=cfg.tts.kokoro_runtime, threads=cfg.tts.threads, opening_first=cfg.voice.opening_first,
                       join_silence_ms=cfg.voice.join_silence_ms)
        if cfg.tts.backend == "chatterbox":             # Chatterbox Turbo from its streaming server; Kokoro says a sentence when the server cannot (voice/remote_tts.py)
            from .remote_tts import StreamingVoice

            kokoro = tts
            tts = StreamingVoice(cfg.tts.url, kokoro, first_timeout_s=cfg.tts.first_timeout_s)
            stt.warm_up()
            tts.warm_up()
            kokoro_ready = kokoro.preload(common_phrases(cfg), first_only=slot_offer_openings(kokoro.pieces))     # the fallback, as Kokoro would be
            tts.preload(common_phrases(cfg), first_only=())          # (Kokoro's part already done above; the server's made in the background)
            logger.debug(f"voice: chatterbox at {cfg.tts.url}, Kokoro ready as fallback in {kokoro_ready:.1f} s")
            return VoiceParts(stt, tts, hotword_groups(ROOT / cfg.stt.hotwords_file) if cfg.stt.hotwords_file else {})
        if cfg.tts.backend == "pocket":                 # Pocket-TTS's eve; its alba from another server when that fails; Kokoro, the last resort, only when both do
            from .remote_tts import StreamingVoice

            alba = StreamingVoice(cfg.tts.pocket_fallback_url, tts, model="pocket", voice=cfg.tts.pocket_fallback_voice, first_timeout_s=cfg.tts.pocket_first_timeout_s,
                                  spell_letters=False)
            tts = StreamingVoice(cfg.tts.pocket_url, alba, model="pocket", voice=cfg.tts.pocket_voice, first_timeout_s=cfg.tts.pocket_first_timeout_s,
                                 spell_letters=False)        # "AGMC" as written: Pocket says it quickly (section 43); Kokoro, the last resort, keeps the spelling
            stt.warm_up()
            tts.warm_up()                               # warms alba and Kokoro too (each warms its fallback)
            tts.preload(common_phrases(cfg), first_only=())          # eve's versions, made in the background (Kokoro's are not made: it speaks only if both servers are down)
            logger.debug(f"voice: {tts.name}")
            return VoiceParts(stt, tts, hotword_groups(ROOT / cfg.stt.hotwords_file) if cfg.stt.hotwords_file else {})
    else:
        tts = PiperSherpa(models, voice or cfg.tts.voice or "joe", speaker=cfg.tts.speaker if voice is None else 0, copies=cfg.tts.copies,
                         threads=cfg.tts.threads or 4)
    stt.warm_up()
    tts.warm_up()
    if hasattr(tts, "preload"):                         # the phrases said all the time play with no wait
        tts.preload(common_phrases(cfg), first_only=slot_offer_openings(tts.pieces))
    return VoiceParts(stt, tts, hotword_groups(ROOT / cfg.stt.hotwords_file) if cfg.stt.hotwords_file else {})


def slot_offer_openings(split=None) -> list[str]:
    """The sentences that offer test-drive times, as agent/wording.py writes them, with sample times: only their first piece is rendered in advance (voice/kokoro_tts.py first_only).
    It can be kept ahead of time where the day is the only thing that varies ("Today I can do", "The soonest is today:"), and not where a time comes in it ("Today I've got 9",
    with A's four-word first piece; with B's three words it is "Today I've got", which is kept). `split`: how the voice cuts a sentence (default A). Measured 28 Oct: about 0.35 s
    to the first sound of every slot offer."""
    from ..agent.wording import offer_slot_variants
    from .kokoro_tts import first_piece_split

    split = split or first_piece_split
    days = ["today", "tomorrow"] + [f"on {d}" for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")]
    out: list[str] = []
    for d in days:
        out += [v for v in offer_slot_variants(d, "9 in the morning or 2 in the afternoon") if "9" not in split(v)[0]]
    return out


def common_phrases(cfg: Config) -> list[str]:
    """What Wazir says word for word again and again: rendered once at start-up by voices that are slow to make speech."""
    name = cfg.app.agent_name
    return [greeting(name) if name else "Hello, welcome to AGMC. What kind of car are you after?",
            "Is there anything else I can help with?", "What else can I do for you?", "Anything else you'd like to know?",
            "Would you like to come and see it?", "Want to book a test drive?", "Shall I set up a test drive for you?",
            "Sure! What day would you like to come in?", "Of course. What day and time suit you?", "What day and time work for you?",
            "When would you like to come in?", "Perfect. Can I take your name and mobile number?", "Great. What's your name and mobile number?",
            "Great. And your name, please?", "Lovely. Who should I put the booking under?", "Which one would you like to hear more about?",
            "Sorry, I didn't catch that. Could you say it again?", "Thank you for calling AGMC. Goodbye!", "You're welcome!",
            *NARROW_QUESTIONS, "Which suits you better?", "Shall I book it?", "Is that all correct now?", "Just bring your driving licence along.",
            "No problem at all, take your time.", "Our prices are fixed,"] + OPENERS + [
            "Sure.", "Sure!", "Okay.", "Great,", "Great.", "Great!", "Perfect,", "Perfect.", "Lovely,", "Lovely.", "Thanks,", "Thanks.", "Done,",
            "All set,", "Of course,", "Of course.", "No problem,", "No problem.", "Great choice.", "Good choice,", "Absolutely,", "Absolutely!",
            "Yes,", "Sorry,", "Got it,", "Got it.", "Hmm,"] + BRIDGES + CUT_OFF + STILL_THERE + ["Is there a colour you'd like?", SILENT_GOODBYE, WELCOME_BACK]


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
    w.first_deadline_s = (cfg.voice.first_sentence_deadline_ms / 1000.0) or None   # a model that has not given a usable first sentence by then is not waited for
    w.history_lines = cfg.voice.history_lines
    if not cfg.negotiation.enabled:                         # the website: fixed prices, so the instructions need no counter-offer example (about half the tokens)
        from ..dialogue.writer import SYSTEM_VOICE

        w.system = SYSTEM_VOICE
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
                   seed_agent_line: str | None = None, seed_state=None, greet_after: float = 0.0, still_there_after: float = 0.0, hang_up_after: float = 0.0,
                   welcome_back: bool = False, on_before_greeting=None, on_silent_end=None):
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

    dump = os.environ.get("VA_DUMP_AUDIO_DIR")                  # diagnostic only (voice/audiotap.py): copy what goes in to the detector and what goes out to the page
    in_tap = AudioTap(dump, "in", InputAudioRawFrame) if dump else None
    if in_tap is not None:                                      # ...and, beside the audio, the words (what was recognised, given to the agent, said back)
        heard_hook, say_hook = on_heard, on_say
        on_heard = lambda text: (in_tap.note("heard", text), heard_hook(text) if heard_hook else None)[1]
        on_say = lambda text: (in_tap.note("said", text), say_hook(text) if say_hook else None)[1]
    from .services import EchoGuard

    hear = EchoGuard(parts.stt_engine, lambda: brain.session.last_agent(),            # (brain is made just below; the guard only asks for the line when a turn is heard)
                     lambda: stt_prompt(brain.session.state, "", parts.hotword_groups, app.inventory.vocabulary()))
    stt = OpenVinoWhisperSTTService(engine=hear, min_rms=vcfg.min_rms)
    stop_now = lambda: current_stop[0]                           # the silence the detector waits right now: the turn clock subtracts it to find the real end of speech
    if in_tap is not None:
        stt.on_text = lambda text: in_tap.note("stt", text)
    brain = DealerBrain(app.engine, speak_opening=speak_opening, on_turn=after_turn, on_heard=on_heard, on_say=on_say,
                        level_of=lambda: stt.last_level, quiet_below=vcfg.end_call_min_rms,
                        speak_up=(vcfg.speak_up_below, vcfg.speak_up_suspect_below),
                        background_below=vcfg.background_below if mic_gate is None else 0.0,     # hands-free only: in tap mode the caller chose to speak
                        first_turn_below=vcfg.first_turn_below if mic_gate is None else 0.0,
                        greet_after=greet_after, still_there_after=still_there_after, hang_up_after=hang_up_after, welcome_back=welcome_back,
                        on_before_greeting=on_before_greeting, on_silent_end=on_silent_end)
    brain.vad_stop_secs = stop_now
    if seed_agent_line:                                       # tests and studies: start as if the agent had just said this
        brain.session.remember("agent", seed_agent_line)
        brain.session.state = seed_state if seed_state is not None else brain.session.state
    follow_the_question()
    stt.prompt_for = lambda: stt_prompt(brain.session.state, brain.session.last_agent(), parts.hotword_groups, app.inventory.vocabulary())
    tts = SherpaPiperTTSService(engine=parts.tts_engine)
    stages: list[FrameProcessor] = [source] + ([mic_gate] if mic_gate is not None else []) + ([HalfDuplexGate(gate_tail_secs)] if half_duplex else []) + (
        [in_tap] if in_tap is not None else []) + [vad]
    if smart_turn:                                           # `stop_secs` is then the short VAD silence; Smart Turn decides whether the turn is really over
        from .endpoint import SmartEndpointer

        endpointer = SmartEndpointer(STT_RATE, smart_fallback_secs, short_answer_secs=vcfg.smart_short_answer_secs if smart_short_answer_secs is None else smart_short_answer_secs,
                                     early_start_secs=vcfg.early_start_secs if vcfg.speculate else 0.0)
        endpointer.asked_question = lambda: brain.session.last_agent().rstrip().endswith("?")        # the agent's last turn ended on a question
        if vcfg.smart_peek if smart_peek is None else smart_peek:
            from zoneinfo import ZoneInfo

            from .answers import is_complete_answer

            tz = ZoneInfo(app.service.cfg.branches[0].timezone)
            endpointer.peek = lambda pcm: asyncio.to_thread(lambda: hear.transcribe(pcm, stt.prompt_for())[0])
            endpointer.is_answer = lambda text: is_complete_answer(text, brain.session.last_agent(), brain.session.state, brain.session.shown,
                                                                   app.inventory.vocabulary(), app.service.clock(), tz)
        if vcfg.speculate:                                    # recognise (and decide) during Smart Turn's hold; used if the caller has finished
            from .services import clean_transcript
            from .speculate import Speculation

            spec = Speculation(lambda pcm: clean_transcript(hear.transcribe(pcm, stt.prompt_for())[0]),
                               (lambda text: app.engine.predecide(brain.session, text)) if hasattr(app.engine, "predecide") else None,
                               lambda: len(brain.session.history))
            endpointer.speculation = stt.speculation = brain.speculation = spec
        stages.append(endpointer)
    stages.append(MicHealth())                                # one log line per turn, only when the microphone audio arrived damaged (voice/mic_health.py)
    if in_tap is not None:                                    # the recorder notes when the turn was decided over, so scripts/call_latency.py can tell the detector's wait from Whisper's
        from .micgate import StopNotifier

        async def turn_end() -> None:
            in_tap.note("turn_end", "")

        stages.append(StopNotifier(turn_end))
    if mic_gate is not None and on_user_stop is not None:     # tap-to-talk: a caller who carries on before the reply is heard continues the same sentence
        from .micgate import ContinuationGuard

        stages.append(ContinuationGuard(mic_gate, brain))
    if on_user_stop is not None:                              # tap-to-talk: told when the turn is over, so the microphone can be closed
        from .micgate import StopNotifier

        stages.append(StopNotifier(on_user_stop))
    from .heard import Playback, PlaybackTracker

    playback = Playback()                                     # what the page is playing, worked out here (voice/heard.py)
    brain.playback = playback
    if mic_gate is not None:
        mic_gate.playback = playback                          # tap mode: a tap while the agent is audible interrupts it (not only while the server is still sending)
    late = None
    if barge_in and not half_duplex:                          # the caller may talk over the agent (needs a headset or echo cancellation)
        from .bargein import BargeIn, LateBargeIn

        stages.append(BargeIn(playback, min_speech_secs=vcfg.barge_in_min_speech_secs, level=vcfg.barge_in_level))
        late = LateBargeIn(brain, playback)                   # words said over the agent that were too short to stop it: they stop it when recognised
    for name, proc in (("vad", None), ("stt", stt), ("late", late), ("brain", brain), ("tts", tts)):
        if proc is not None:
            stages.append(proc)
        if name in probes:
            stages.append(probes[name])
    stages.append(PlaybackTracker(playback, on_cut=brain.reply_cut))     # last before the page: the audio's timing as it goes out
    if dump:
        stages.append(AudioTap(dump, "out", OutputAudioRawFrame))
    stages.append(sink)
    return Pipeline(stages), brain

r"""Talk to the agent: microphone in, speaker out. Everything runs on this machine.

  .venv\Scripts\python.exe -m dealer_core.cli.voice [--list-devices] [--input-device N] [--output-device N] [--headset] [--debug]

Use a headset if you can (--headset): with the laptop speaker the microphone hears the agent, so the agent's own voice is ignored while it
speaks (half duplex) and you cannot interrupt it. Press Ctrl+C to hang up.
Privacy: by default the console shows timings and the detected intent only. What was said is printed with --transcript and written to a file only with --save-transcript.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

from loguru import logger


def list_devices() -> None:
    import pyaudio

    p = pyaudio.PyAudio()
    default_in, default_out = p.get_default_input_device_info()["index"], p.get_default_output_device_info()["index"]
    print("index  in  out  name")
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        print(f"{i:5d} {d['maxInputChannels']:3d} {d['maxOutputChannels']:4d}  {d['name']}" + ("   <- default input" if i == default_in else "")
              + ("   <- default output" if i == default_out else ""))


def _write_transcript(who: str, text: str, to_console: bool, file) -> None:
    stamp = time.strftime("%H:%M:%S")
    if to_console:
        print(f"  {who}: {text}", flush=True)
    if file:
        file.write(f"[{stamp}] {who}: {text}\n")
        file.flush()


def _device_names(inp: int | None, out: int | None) -> tuple[str, str]:
    import pyaudio

    p = pyaudio.PyAudio()
    i = inp if inp is not None else p.get_default_input_device_info()["index"]
    o = out if out is not None else p.get_default_output_device_info()["index"]
    return f"{i}: {p.get_device_info_by_index(i)['name']}", f"{o}: {p.get_device_info_by_index(o)['name']}"


async def run(a: argparse.Namespace) -> None:
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
    from pipecat.workers.runner import WorkerRunner

    from ..app import build_app, demo_clock
    from ..config import load_config
    from ..voice.pipeline import STT_RATE, apply_voice_profile, build_pipeline, load_engines
    from .chat import start

    cfg = load_config()
    app = build_app(cfg, backend=a.backend, clock=demo_clock(cfg) if a.clock == "demo" else None,
                    reply_mode="template" if a.template_only else a.mode, engine="agent" if a.agent else None)
    start(app)                                                     # loads and warms the language model
    apply_voice_profile(app, cfg)
    print("Loading speech models...", flush=True)
    parts = load_engines(cfg, a.stt_model)
    transport = LocalAudioTransport(LocalAudioTransportParams(
        audio_in_enabled=True, audio_out_enabled=True, audio_in_sample_rate=STT_RATE, audio_out_sample_rate=parts.tts_engine.sample_rate,
        input_device_index=a.input_device, output_device_index=a.output_device))

    def on_turn(result) -> None:
        d = result.debug
        line = f"  [{d.intent} | {d.via.split(':')[0]} | {'model-written' if d.draft else 'standard wording'} | risk {d.risk or '-'}]"
        if a.show_text:
            line += f"\n  agent: {result.reply}"
        if a.debug:
            print(line, flush=True)

    from ..voice.status import StatusLine

    from ..voice.status import TurnClock

    def timing(total: float, parts: dict[str, float]) -> None:
        detail = ", ".join(f"{name} {secs:.2f} s" for name, secs in parts.items())
        if a.transcript or a.save_transcript:
            _write_transcript("TIME ", f"{total:.1f} s from the end of your speech to the first sound ({detail})", a.transcript, transcript)
        else:
            print(f"  [first sound {total:.1f} s after you finished: {detail}]", flush=True)

    smart = cfg.voice.smart_turn and not a.fixed_silence
    stop_secs = a.stop_secs or (cfg.voice.smart_stop_secs if smart else cfg.voice.stop_secs)
    clock = TurnClock(stop_secs, timing)
    probes = {"vad": StatusLine({"vad", "bot"}, clock=clock, stage="vad"), "stt": StatusLine({"stt"}, clock=clock, stage="stt"),
              "brain": StatusLine(set(), clock=clock, stage="brain")}
    transcript = open(a.save_transcript, "a", encoding="utf-8") if a.save_transcript else None     # only if explicitly asked: the call is otherwise never written to disk

    def heard(text: str) -> None:
        _write_transcript("YOU  ", text, a.transcript, transcript)

    def said(text: str) -> None:
        _write_transcript("AGENT", text, a.transcript, transcript)
    pipeline, brain = build_pipeline(app, parts, transport.input(), transport.output(), stop_secs=stop_secs, smart_turn=smart, smart_fallback_secs=cfg.voice.smart_fallback_secs,
                                     half_duplex=not a.headset, barge_in=a.headset and cfg.voice.barge_in and not a.no_barge_in, on_turn=on_turn, probes=probes,
                                     on_heard=heard if (a.transcript or transcript) else None, on_say=said if (a.transcript or transcript) else None)
    worker = PipelineWorker(pipeline, params=PipelineParams(audio_in_sample_rate=STT_RATE, audio_out_sample_rate=parts.tts_engine.sample_rate),
                            idle_timeout_secs=None)
    runner = WorkerRunner(handle_sigint=True)
    await runner.add_workers(worker)
    names = _device_names(a.input_device, a.output_device)
    print(f"\nMicrophone: {names[0]}\nSpeaker   : {names[1]}"
          f"\n\nReady. You should hear the greeting now ('[agent speaking]' below). Then speak when it says '[your turn]'. Ctrl+C to hang up."
          f"\nIf you see no '[agent speaking]' within 3 s the speaker is the problem; if '[hearing you]' never appears when you talk the microphone is.\n",
          flush=True)
    t0 = time.time()
    await runner.run()
    if transcript:
        transcript.close()
    print(f"\nCall ended after {time.time() - t0:.0f} s, {len(brain.turn_log)} turns.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-devices", action="store_true", help="show the audio devices and exit")
    ap.add_argument("--input-device", type=int, help="microphone device index (see --list-devices); default: the system default")
    ap.add_argument("--output-device", type=int, help="speaker device index; default: the system default")
    ap.add_argument("--headset", action="store_true", help="a headset has no echo, so the microphone stays open while the agent speaks")
    ap.add_argument("--backend", help="llm backend from config; default llm.text_backend")
    ap.add_argument("--mode", choices=["conversational", "hybrid", "template"], help="who writes the reply; default dialogue.reply_mode")
    ap.add_argument("--template-only", action="store_true", help="fixed wording only (fastest)")
    ap.add_argument("--no-barge-in", action="store_true", help="the caller cannot talk over the agent")
    ap.add_argument("--agent", action="store_true", help="tool-calling agent engine (experimental): the model picks the next action, code runs it")
    ap.add_argument("--fixed-silence", action="store_true", help="end your turn after a fixed silence instead of Smart Turn (voice.smart_turn)")
    ap.add_argument("--stop-secs", type=float, help="silence that ends your turn; default voice.stop_secs (0.3)")
    ap.add_argument("--stt-model", help="OpenVINO Whisper folder under models/ov, e.g. whisper-base-int8-ov (default voice.stt_model)")
    ap.add_argument("--clock", choices=["demo", "real"], default="demo")
    ap.add_argument("--debug", action="store_true", help="print the intent and who wrote the reply after every turn (no transcripts)")
    ap.add_argument("--show-text", action="store_true", help="(older) print what the agent said after each turn; --transcript shows both sides")
    ap.add_argument("--transcript", action="store_true", help="print what you said (as the speech recognizer heard it) and what the agent says, live")
    ap.add_argument("--save-transcript", metavar="FILE", help="also append the transcript to this file (it holds exactly what was said, so treat it as personal data)")
    a = ap.parse_args()
    if a.list_devices:
        list_devices()
        return 0
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    try:
        asyncio.run(run(a))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

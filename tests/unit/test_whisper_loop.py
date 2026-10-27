"""27 Oct, Chrome: "No, I want to know something else about the Macan" came back as "No" + ~400 exclamation marks after 2.8 s and the caller heard
"I didn't catch that" 6.1 s later. Cause: OpenVINO's CPU plugin runs in float16 on Apple chips and Whisper base int8 then loops on token 0 ("!").
Fix: float32 for the CPU recognizer (checked on the real recording: 9 of 9 turns right), plus a safety net for any other loop."""
import asyncio
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dealer_core.config import load_config
from dealer_core.voice.services import UNCLEAR, OpenVinoWhisperSTTService, clean_transcript, squeeze_punctuation
from dealer_core.voice.stt import MIN_TOKENS, WhisperOpenVino, is_degenerate, token_cap

LOOP = "No," + "!" * 400


# ------------------------------------------------------------------------------------------------ what a loop looks like
def test_a_loop_is_told_from_speech():
    assert is_degenerate(LOOP, 2.9) and is_degenerate("No" + "!" * 6, 2.9) and is_degenerate("a" * 12, 2.0)
    assert is_degenerate("word " * 120, 1.0)                                  # far more text than one second can hold
    assert not is_degenerate("No, I want to know something else about the Macan.", 2.9)
    assert not is_degenerate("zero five zero one two three four five six seven eight", 4.0)     # digits said one by one
    assert not is_degenerate("Really?!?!", 1.0) and not is_degenerate("Wait... what?", 1.0)


def test_the_output_cap_grows_with_the_audio_and_stays_inside_whispers_limit():
    assert token_cap(0.5) == MIN_TOKENS
    assert token_cap(2.9) == 50 and token_cap(10) == 136
    assert token_cap(120) == 440                                              # Whisper's own ceiling is 448


def test_a_loop_is_never_passed_on_as_words():
    assert clean_transcript(LOOP) == UNCLEAR
    assert squeeze_punctuation("Yes!!!") == "Yes!" and clean_transcript("Yes!!!") == "Yes!"
    assert clean_transcript("Really?!?!") == "Really?!?!"


def test_the_recognizer_service_turns_a_loop_into_a_request_to_repeat():
    class Engine:
        name = "fake"

        def transcribe(self, pcm, prompt=None):
            return LOOP, 2800.0

    async def go():
        stt = OpenVinoWhisperSTTService(engine=Engine())
        loud = (np.sin(np.arange(48000) / 5) * 8000).astype(np.int16).tobytes()
        return [f.text async for f in stt.run_stt(loud)]

    assert asyncio.run(go()) == [UNCLEAR]


def test_the_agent_answers_the_unclear_marker_with_sorry_i_didnt_catch_that():
    from dealer_core.agent.engine import AgentEngine
    from dealer_core.app import build_app
    from tests.unit.test_stock_routing import Scripted

    app = build_app(load_config(environ={}), inventory="demo", classifier=Scripted(), reply_mode="conversational", lead_backend=Scripted())
    e = AgentEngine(app.service, Scripted(), app.engine.reply_writer)
    out = list(e.handle_stream(e.new_session(), UNCLEAR))
    assert "didn't catch that" in out[0] and not out[-1].ended


# ------------------------------------------------------------------------------------------------ the recognizer's settings
class FakeGenAI:
    """Stands in for openvino_genai: records how each pipeline was made and what each call was given."""
    made, calls = [], []

    class WhisperPipeline:
        def __init__(self, model, device, **props):
            FakeGenAI.made.append((device, props))

        def get_generation_config(self):
            return SimpleNamespace(initial_prompt="", max_new_tokens=0, language="", task="")

        def generate(self, samples, cfg):
            FakeGenAI.calls.append((len(samples), cfg.max_new_tokens, cfg.initial_prompt))
            return SimpleNamespace(texts=["hello there"])


@pytest.fixture
def fake_genai(monkeypatch):
    FakeGenAI.made, FakeGenAI.calls = [], []
    monkeypatch.setitem(sys.modules, "openvino_genai", FakeGenAI)
    return FakeGenAI


def test_the_cpu_recognizer_runs_in_float32_by_default(fake_genai, tmp_path):
    WhisperOpenVino(tmp_path / "whisper-base-int8-ov", tmp_path)
    device, props = fake_genai.made[0]
    assert device == "CPU" and props["INFERENCE_PRECISION_HINT"] == "f32" and "CACHE_DIR" in props


def test_the_precision_is_only_forced_on_the_cpu_and_can_be_left_to_openvino(fake_genai, tmp_path):
    WhisperOpenVino(tmp_path / "whisper-small.en-int8-ov", tmp_path, device="GPU")
    WhisperOpenVino(tmp_path / "whisper-base-int8-ov", tmp_path, precision="")
    assert all("INFERENCE_PRECISION_HINT" not in props for _, props in fake_genai.made)


def test_every_call_is_capped_by_its_audio_and_carries_its_hint(fake_genai, tmp_path):
    eng = WhisperOpenVino(tmp_path / "whisper-base-int8-ov", tmp_path)
    eng.transcribe(bytes(2 * 16000 * 3), "Toyota, Porsche")                  # 3 s of audio
    n_samples, cap, hint = fake_genai.calls[0]
    assert n_samples == 48000 and cap == token_cap(3.0) == 52 and hint == "Toyota, Porsche"


def test_the_defaults_in_the_configuration():
    v = load_config(environ={}).voice
    assert v.stt_precision == "f32" and v.stt_device == "CPU"
    assert not load_config(environ={"VA__VOICE__STT_PRECISION": ""}).voice.stt_precision          # an empty value leaves it to OpenVINO


def test_the_recognizer_thread_count_is_a_setting_and_off_by_default(fake_genai, tmp_path):
    WhisperOpenVino(tmp_path / "whisper-base-int8-ov", tmp_path)
    WhisperOpenVino(tmp_path / "whisper-base-int8-ov", tmp_path, threads=3)
    WhisperOpenVino(tmp_path / "whisper-small.en-int8-ov", tmp_path, device="GPU", threads=3)
    (_, plain), (_, capped), (_, gpu) = fake_genai.made
    assert "INFERENCE_NUM_THREADS" not in plain and capped["INFERENCE_NUM_THREADS"] == 3 and "INFERENCE_NUM_THREADS" not in gpu
    v, t = load_config(environ={}).voice, load_config(environ={}).tts
    assert v.stt_threads is None and t.threads is None
    assert load_config(environ={"VA__VOICE__STT_THREADS": "4", "VA__TTS__THREADS": "2"}).voice.stt_threads == 4


def test_the_speech_engines_take_a_thread_count():
    from dealer_core.voice.kokoro_tts import ort_options

    assert ort_options(3).intra_op_num_threads == 3
    assert ort_options(None).intra_op_num_threads == ort_options().intra_op_num_threads

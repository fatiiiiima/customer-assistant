"""The stand-in recogniser and voice (scripts/stub_services.py) answer in the shapes the worker's clients read."""
import importlib.util
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

spec = importlib.util.spec_from_file_location("stub_services", Path(__file__).resolve().parents[2] / "scripts" / "stub_services.py")
ss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ss)


def client(**kw):
    return TestClient(ss.make_app(**kw))


def test_the_recogniser_answers_a_phrase_to_any_upload():
    c = client(stt_delay_s=0.0)
    r = c.post("/v1/audio/transcriptions", files={"file": ("t.wav", b"RIFF....", "audio/wav")}, data={"language": "en"})
    assert r.status_code == 200 and r.json()["text"] in ss.PHRASES
    assert c.post("/v1/audio/transcriptions", files={"file": ("t.wav", b"x", "audio/wav")}).json()["text"] != r.json()["text"]           # it goes through the list


def test_the_voice_streams_audio_as_long_as_the_text_would_take_and_not_silence():
    c = client(speed=1000.0)
    text = "a" * 45                                                                      # 45 characters: 3 s
    r = c.post("/v1/audio/speech", json={"input": text, "stream": True, "response_format": "pcm"})
    pcm = np.frombuffer(r.content, dtype="<i2")
    assert r.status_code == 200
    assert abs(len(pcm) / ss.RATE - 3.0) < 0.01 and np.abs(pcm).max() > 100              # 3 s of audible audio
    assert len(c.post("/v1/audio/speech", json={"input": "Hi"}).content) / 2 / ss.RATE >= 0.29        # a short word still takes some time


def test_the_voice_is_paced_faster_than_real_time_but_not_instant():
    c = client(speed=ss.REAL_TIME_FACTOR)
    t0 = time.perf_counter()
    c.post("/v1/audio/speech", json={"input": "b" * 30})                                # 2 s of speech: about 0.26 s at 7.8x
    took = time.perf_counter() - t0
    assert 0.15 < took < 1.0


def test_health_and_models_answer_for_start_up_checks():
    c = client()
    assert c.get("/health").json()["ok"] and c.get("/v1/models").json()["data"][0]["id"] == "stub"

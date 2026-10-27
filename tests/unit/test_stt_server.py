"""The shared local speech recognizer (dealer_core.sttserver): the same API a GPU Whisper server offers, so the workers' RemoteWhisper talks to it unchanged."""
import io
import wave

import httpx
import numpy as np
import pytest
from starlette.testclient import TestClient

from dealer_core.sttserver.__main__ import RATE, create_app, parse_multipart, read_wav
from dealer_core.voice.stt import RemoteWhisper, wav_bytes


def wav(samples: np.ndarray, rate: int = 16000, channels: int = 1, width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels), w.setsampwidth(width), w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes() if width == 2 else bytes(len(samples) * width))
    return buf.getvalue()


class FakeEngine:
    name = "fake"

    def __init__(self, text="No, I want to know something else about the Macan."):
        self.calls, self.text = [], text

    def transcribe_float(self, audio, prompt=None):
        self.calls.append((len(audio), prompt))
        return self.text


def tone(seconds=1.0, rate=16000):
    return (np.sin(np.arange(int(seconds * rate)) / 9) * 0.3).astype(np.float32)


# ------------------------------------------------------------------------------------------------ the upload
def test_a_multipart_upload_keeps_the_audio_bytes_exactly():
    data = wav(tone(2.0))
    req = httpx.Request("POST", "http://x/v1/audio/transcriptions", data={"model": "m", "prompt": "Toyota, Macan", "language": "en"}, files={"file": ("turn.wav", data, "audio/wav")})
    fields, files = parse_multipart(req.read(), req.headers["content-type"])
    assert files["file"] == data and fields["prompt"] == "Toyota, Macan" and fields["model"] == "m"


def test_a_body_without_multipart_gives_nothing():
    assert parse_multipart(b"{}", "application/json") == ({}, {})


def test_wav_is_read_as_16k_mono_float():
    assert read_wav(wav(tone(1.0))).shape == (16000,)
    up = read_wav(wav(tone(1.0, 8000), rate=8000))
    assert len(up) == 16000 and up.dtype == np.float32                        # resampled to 16 kHz
    stereo = np.stack([tone(0.5), tone(0.5)], axis=1).reshape(-1)
    assert read_wav(wav(stereo, channels=2)).shape == (8000,)
    with pytest.raises(ValueError):
        read_wav(wav(tone(0.2), width=3))


# ------------------------------------------------------------------------------------------------ the service
def test_the_workers_client_gets_the_text_and_the_hint_arrives():
    eng = FakeEngine()
    client = TestClient(create_app(eng, "whisper-test"))
    remote = RemoteWhisper("http://testserver/v1", "whisper-test", 3.0, ["Toyota"], client=client)
    text, ms = remote.transcribe(bytes(2 * 16000 * 3), "Toyota, Porsche, Macan")           # 3 s of audio
    assert text == "No, I want to know something else about the Macan." and remote.errors == 0
    assert eng.calls == [(48000, "Toyota, Porsche, Macan")]


def test_no_hint_means_the_engines_own_default():
    eng = FakeEngine()
    client = TestClient(create_app(eng))
    RemoteWhisper("http://testserver/v1", "m", 3.0, [], client=client).transcribe(bytes(32000), "")
    assert eng.calls[0][1] is None


def test_a_bad_request_is_a_400_not_a_crash():
    client = TestClient(create_app(FakeEngine()))
    assert client.post("/v1/audio/transcriptions", content=b"nothing").status_code == 400
    assert client.post("/v1/audio/transcriptions", data={"model": "m"}, files={"other": ("a.txt", b"x")}).status_code == 400
    assert client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav(tone(0.2), width=3), "audio/wav")}).status_code == 400
    assert client.get("/health").json()["errors"] == 3


def test_health_and_models():
    client = TestClient(create_app(FakeEngine(), "whisper-base-int8-ov"))
    assert client.get("/health").json()["ok"] is True
    assert client.get("/v1/models").json()["data"][0]["id"] == "whisper-base-int8-ov"
    assert RATE == 16000


# ------------------------------------------------------------------------------------------------ the cluster launcher
def test_workers_are_pointed_at_the_shared_recognizer_only_when_asked():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("run_cluster", Path(__file__).resolve().parents[2] / "scripts" / "run_cluster.py")
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    assert "VA__VOICE__STT_URL" not in rc.worker_env("ollama", "postgresql://db")
    assert rc.worker_env("ollama", "postgresql://db", "http://127.0.0.1:8002/v1")["VA__VOICE__STT_URL"] == "http://127.0.0.1:8002/v1"

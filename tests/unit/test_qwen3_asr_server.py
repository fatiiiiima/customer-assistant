"""scripts/qwen3_asr_server.py (ARCH_REVIEW sections 79 to 81): Qwen3-ASR behind the OpenAI transcription API, so the worker's RemoteWhisper talks to it unchanged.
A fake model stands in for MLX: nothing here needs the private environment or a GPU."""
import importlib.util
import io
import sys
import threading
import time
import types
import wave
from pathlib import Path

import httpx
import numpy as np
import pytest

from dealer_core.voice.stt import RemoteWhisper

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("qwen3_asr_server", ROOT / "scripts" / "qwen3_asr_server.py")
qs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qs)


def wav(samples: np.ndarray, rate: int = 16000, channels: int = 1, width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels), w.setsampwidth(width), w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes() if width == 2 else bytes(len(samples) * width))
    return buf.getvalue()


def tone(seconds=1.0, rate=16000):
    return (np.sin(np.arange(int(seconds * rate)) / 9) * 0.3).astype(np.float32)


class FakeEngine:
    name = "fake"

    def __init__(self, text="Yes, talk about the Macan.", delay=0.0):
        self.calls, self.text, self.delay, self.inside, self.overlap = [], text, delay, 0, 0

    def transcribe_float(self, audio, prompt=None, language="en", hotwords=None):
        self.inside += 1
        self.overlap = max(self.overlap, self.inside)
        time.sleep(self.delay)
        self.calls.append({"samples": len(audio), "prompt": prompt, "language": language, "hotwords": hotwords})
        self.inside -= 1
        if self.text is None:
            raise RuntimeError("model fell over")
        return self.text


@pytest.fixture
def server():
    started = []

    def start(engine):
        httpd = qs.serve(engine, "127.0.0.1", 0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        started.append(httpd)
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    yield start
    for h in started:
        h.shutdown()
        h.server_close()


# ------------------------------------------------------------------------------------------------ the upload and the audio
def test_a_multipart_upload_keeps_the_audio_bytes_and_the_fields():
    data = wav(tone(2.0))
    req = httpx.Request("POST", "http://x/v1/audio/transcriptions", data={"model": "m", "prompt": "Toyota, Macan", "language": "en"}, files={"file": ("turn.wav", data, "audio/wav")})
    fields, files = qs.parse_multipart(req.read(), req.headers["content-type"])
    assert files["file"] == data and fields["prompt"] == "Toyota, Macan" and fields["language"] == "en"
    assert qs.parse_multipart(b"{}", "application/json") == ({}, {})


def test_wav_becomes_16k_mono_float_whatever_it_was_sent_as():
    assert qs.read_wav(wav(tone(1.0))).shape == (16000,)
    up = qs.read_wav(wav(tone(1.0, 8000), rate=8000))
    assert len(up) == 16000 and up.dtype == np.float32
    stereo = np.stack([tone(0.5), tone(0.5)], axis=1).reshape(-1)
    assert qs.read_wav(wav(stereo, channels=2)).shape == (8000,)
    with pytest.raises(ValueError):
        qs.read_wav(wav(tone(0.2), width=3))


def test_the_workers_language_code_becomes_a_language_name_and_never_auto_detect():
    """2 Nov: left to itself the model answered "Any budget, any model." in Russian."""
    assert qs.language_name("en") == "English" and qs.language_name("EN") == "English" and qs.language_name("ar") == "Arabic"
    assert qs.language_name(None) == "English" and qs.language_name("") == "English" and qs.language_name("xx") == "English"


# ------------------------------------------------------------------------------------------------ the service
def test_the_workers_own_client_gets_the_text_and_the_hint_reaches_the_model_as_its_prompt(server):
    eng = FakeEngine()
    url = server(eng)
    remote = RemoteWhisper(f"{url}/v1", "qwen3-asr", 3.0, ["Toyota"])
    text, _ = remote.transcribe(bytes(2 * 16000 * 3), "Toyota, Porsche, Macan. Which one?")          # 3 s of audio
    assert text == "Yes, talk about the Macan." and remote.errors == 0
    assert eng.calls == [{"samples": 48000, "prompt": "Toyota, Porsche, Macan. Which one?", "language": "en", "hotwords": None}]


def test_no_hint_and_hot_words_and_the_text_format(server):
    eng = FakeEngine()
    url = server(eng)
    files = {"file": ("t.wav", wav(tone(1.0)), "audio/wav")}
    r = httpx.post(f"{url}/v1/audio/transcriptions", data={"model": "m"}, files=files)
    assert r.status_code == 200 and r.json()["text"] == "Yes, talk about the Macan." and eng.calls[-1]["prompt"] is None
    r = httpx.post(f"{url}/v1/audio/transcriptions", data={"hotwords": "Porsche, Macan ,, Land Cruiser", "response_format": "text"}, files=files)
    assert r.text == "Yes, talk about the Macan." and eng.calls[-1]["hotwords"] == ["Porsche", "Macan", "Land Cruiser"]


def test_health_models_unknown_paths_and_bad_requests_never_take_it_down(server):
    eng = FakeEngine()
    url = server(eng)
    assert httpx.get(f"{url}/health").json()["ok"] is True and httpx.get(f"{url}/v1/models").json()["data"][0]["id"] == "qwen3-asr"
    assert httpx.get(f"{url}/nope").status_code == 404 and httpx.post(f"{url}/v1/other", data={"a": "b"}).status_code == 404
    assert httpx.post(f"{url}/v1/audio/transcriptions", data={"model": "m"}).status_code == 400                       # no file
    bad = httpx.post(f"{url}/v1/audio/transcriptions", files={"file": ("t.wav", b"not a wav", "audio/wav")})
    assert bad.status_code == 400 and bad.json() == {"error": "Error"}                                                 # only the error's type, nothing of the upload
    assert eng.calls == []
    assert httpx.post(f"{url}/v1/audio/transcriptions", files={"file": ("t.wav", wav(tone(1.0)), "audio/wav")}).status_code == 200      # and it still serves


def test_a_model_failure_is_an_error_answer_with_only_its_type_and_the_server_goes_on(server):
    eng = FakeEngine(text=None)
    url = server(eng)
    files = {"file": ("t.wav", wav(tone(1.0)), "audio/wav")}
    r = httpx.post(f"{url}/v1/audio/transcriptions", files=files)
    assert r.status_code == 500 and r.json() == {"error": "RuntimeError"}
    eng.text = "ok"
    assert httpx.post(f"{url}/v1/audio/transcriptions", files=files).json()["text"] == "ok"


# ------------------------------------------------------------------------------------------------ the model adapter
def fake_mlx(monkeypatch, seen):
    class Model:
        def generate(self, audio, **kw):
            seen.append((len(audio), audio.dtype, kw))
            return types.SimpleNamespace(text=" Porsche. ")

    stt = types.ModuleType("mlx_audio.stt")
    stt.load = lambda repo: Model()
    pkg = types.ModuleType("mlx_audio")
    pkg.stt = stt
    monkeypatch.setitem(sys.modules, "mlx_audio", pkg)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", stt)


def test_the_adapter_passes_the_hint_as_the_system_prompt_and_forces_the_language(monkeypatch):
    seen = []
    fake_mlx(monkeypatch, seen)
    asr = qs.Qwen3ASR()
    assert asr.transcribe_float(tone(1.0), "Toyota, Porsche, Macan") == "Porsche."                                   # stripped
    assert seen[-1] == (16000, np.dtype("float32"), {"language": "English", "system_prompt": "Toyota, Porsche, Macan"})
    asr.transcribe_float(tone(1.0), None, "en", ["Macan"])
    assert seen[-1][2] == {"language": "English", "hotwords": ["Macan"]}                                              # no prompt: no system_prompt at all
    asr.transcribe_float(tone(1.0), "", "ar")
    assert seen[-1][2] == {"language": "Arabic"}


def test_recognition_is_one_at_a_time_and_two_callers_both_get_answers(server):
    eng = FakeEngine(delay=0.15)
    lock = threading.Lock()                                                                             # the lock lives in the adapter: wrap the fake the way it does
    inner = eng.transcribe_float

    def locked(*a, **k):
        with lock:
            return inner(*a, **k)

    eng.transcribe_float = locked
    url = server(eng)
    out = []

    def call():
        out.append(httpx.post(f"{url}/v1/audio/transcriptions", files={"file": ("t.wav", wav(tone(1.0)), "audio/wav")}, timeout=5).json()["text"])

    ts = [threading.Thread(target=call) for _ in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert out == ["Yes, talk about the Macan."] * 3 and eng.overlap == 1


# ------------------------------------------------------------------------------------------------ the kept prompt beginning (4 Nov, ARCH_REVIEW section 104)
def test_when_the_fast_path_cannot_run_the_model_does_it_the_old_way_and_the_fast_path_stays_off(monkeypatch):
    seen = []
    fake_mlx(monkeypatch, seen)                                                      # no mlx.core here and a model with only `generate`: the fast path fails
    asr = qs.Qwen3ASR()
    assert asr.prefix_cache is True
    assert asr.transcribe_float(tone(1.0), None, "en", ["Macan"]) == "Porsche."      # still answered, by generate
    assert asr.prefix_cache is False and asr.cache_error                             # switched off, with the reason kept
    assert seen[-1][2] == {"language": "English", "hotwords": ["Macan"]}
    asr.transcribe_float(tone(1.0))
    assert len(seen) == 2                                                            # later clips go straight to generate


def test_the_fast_path_can_be_turned_off_and_long_audio_never_uses_it(monkeypatch):
    seen = []
    fake_mlx(monkeypatch, seen)
    off = qs.Qwen3ASR(prefix_cache=False)
    off.transcribe_float(tone(1.0))
    assert off.cache_error == "" and len(seen) == 1                                  # never tried
    long = qs.Qwen3ASR()
    long.transcribe_float(tone(qs.MAX_CACHED_S + 1))
    assert long.prefix_cache is True and long.cache_error == ""                      # too long for it: generate, and the fast path is left on for short clips

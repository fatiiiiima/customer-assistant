"""scripts/parakeet_asr_server.py (ARCH_REVIEW section 86): Parakeet-TDT behind the OpenAI transcription API, so the worker's RemoteWhisper talks to it unchanged.
A fake model stands in for MLX: nothing here needs the private environment or a GPU. The service itself is tested in test_qwen3_asr_server.py."""
import importlib.util
import sys
import threading
import types
from pathlib import Path

import httpx
import numpy as np
import pytest

from dealer_core.voice.services import EchoGuard
from dealer_core.voice.stt import RemoteWhisper

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("parakeet_asr_server", ROOT / "scripts" / "parakeet_asr_server.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)


def fake_mlx(monkeypatch, seen, text=" Go through the rest. "):
    class Model:
        def generate(self, audio, **kw):
            seen.append((audio, kw))
            return types.SimpleNamespace(text=text)

    stt = types.ModuleType("mlx_audio.stt")
    stt.load = lambda repo: Model()
    pkg = types.ModuleType("mlx_audio")
    pkg.stt = stt
    core = types.ModuleType("mlx.core")
    core.array = lambda a: ("mx", a)                                         # what the model receives: the audio wrapped
    mlx = types.ModuleType("mlx")
    mlx.core = core
    for name, mod in (("mlx_audio", pkg), ("mlx_audio.stt", stt), ("mlx", mlx), ("mlx.core", core)):
        monkeypatch.setitem(sys.modules, name, mod)


@pytest.fixture
def server():
    started = []

    def start(engine):
        httpd = ps.service.serve(engine, "127.0.0.1", 0, "parakeet")
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        started.append(httpd)
        return f"http://127.0.0.1:{httpd.server_address[1]}/v1"

    yield start
    for h in started:
        h.shutdown()
        h.server_close()


def test_the_adapter_gives_the_model_only_the_audio_and_strips_the_text(monkeypatch):
    seen = []
    fake_mlx(monkeypatch, seen)
    asr = ps.ParakeetASR()
    audio = np.zeros(16000, dtype=np.float32)
    assert asr.transcribe_float(audio, "Toyota, Macan. Which one?", "ar", ["Macan"]) == "Go through the rest."          # the hint, the list and the language are ignored
    (wrapped, kw), = seen
    assert wrapped[0] == "mx" and wrapped[1].dtype == np.float32 and len(wrapped[1]) == 16000 and kw == {}


def test_the_worker_s_client_gets_the_text_and_the_server_reports_its_name(monkeypatch, server):
    seen = []
    fake_mlx(monkeypatch, seen)
    url = server(ps.ParakeetASR())
    assert httpx.get(f"{url}/models").json()["data"][0]["id"] == "parakeet"
    remote = RemoteWhisper(url, "parakeet", 3.0, ["Toyota"], hotwords_field=True)
    remote.hotword_source = lambda: ["Toyota", "Macan"]
    assert remote.transcribe(bytes(2 * 16000 * 2), "hint")[0] == "Go through the rest." and remote.errors == 0
    assert len(seen) == 1 and seen[0][0][1].shape == (32000,)


def test_with_the_setting_on_the_guard_never_asks_a_second_time(monkeypatch, server):
    seen = []
    fake_mlx(monkeypatch, seen, text="050, 056")
    remote = RemoteWhisper(server(ps.ParakeetASR()), "parakeet", 3.0, [], hotwords_field=True)
    guard = EchoGuard(remote, lambda: "Which day and time would you like?", lambda: "plain")
    assert guard.transcribe(bytes(2 * 16000), "Phone number:")[0] == "050, 056" and len(seen) == 1


def test_the_default_port_and_the_one_at_a_time_lock(monkeypatch):
    seen = []
    fake_mlx(monkeypatch, seen)
    asr = ps.ParakeetASR()
    assert isinstance(asr.lock, type(threading.Lock())) and asr.name.startswith("parakeet")

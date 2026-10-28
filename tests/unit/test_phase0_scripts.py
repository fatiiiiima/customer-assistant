"""The Phase 0 measurement scripts must run on a machine that is not the Windows laptop, and score the server the workers really use (docs/ARCH_REVIEW.md section 19)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def script(name: str):
    spec = importlib.util.spec_from_file_location(f"phase0_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_capacity_probe_does_not_hardcode_a_windows_path():
    mod = script("capacity_probe")
    assert mod.DEFAULT_PORTAL_DATA == ROOT.parent / "bmw-portal" / "site" / "data.js"
    assert "C:" not in (ROOT / "scripts" / "capacity_probe.py").read_text(encoding="utf-8").replace("was C:", "")


def test_hearing_check_scores_the_server_when_given_a_url():
    mod = script("hearing_check")
    (name, math, engine), = mod.build_engines(["whisper-base-int8-ov"], "default,f32", "http://stt-server:8000/v1", "openai/whisper-large-v3-turbo", 3.0, ["Macan"])
    assert (name, math) == ("openai/whisper-large-v3-turbo", "server")        # once, whatever the models and precisions say
    assert type(engine).__name__ == "RemoteWhisper"
    assert engine.base == "http://stt-server:8000/v1" and engine.model == "openai/whisper-large-v3-turbo"


def test_hearing_check_without_a_url_scores_each_local_model_at_each_precision(monkeypatch):
    mod = script("hearing_check")
    built = []

    class Fake:
        def __init__(self, folder, cache, device, hotwords, precision=""):
            built.append((folder.name, precision))

    monkeypatch.setattr("dealer_core.voice.stt.WhisperOpenVino", Fake)
    rows = mod.build_engines(["a", "b"], "default,f32", None, "unused", 3.0, [])
    assert [(n, p) for n, p, _ in rows] == [("a", "default"), ("a", "f32"), ("b", "default"), ("b", "f32")]
    assert built == [("a", ""), ("a", "f32"), ("b", ""), ("b", "f32")]       # "default" means OpenVINO's own math


def test_caller_voice_can_be_a_kokoro_voice(monkeypatch):
    mod = script("portal_call")
    asked = []

    class FakeKokoro:
        sample_rate = 24000

        def __init__(self, models, voice):
            asked.append(voice)

        def synthesize(self, text):
            return np.zeros(24000, dtype=np.float32)

    monkeypatch.setattr("dealer_core.voice.kokoro_tts.KokoroOV", FakeKokoro)
    clips = mod.caller_audio(["Hello"], "kokoro:michael")
    assert asked == ["michael"]
    assert clips[0].dtype == np.int16 and len(clips[0]) == 16000              # one second, resampled to 16 kHz


def test_probe_sweep_stops_after_a_level_over_twice_the_budget_unless_told_not_to():
    mod = script("capacity_probe")
    slow = lambda i: 5000.0                                   # every request takes 5 s: over twice a 700 ms budget at once
    _, stopped = mod.sweep("full", slow, [1, 2, 4], 700.0)
    assert [r["at_once"] for r in stopped] == [1]
    _, every = mod.sweep("full", slow, [1, 2, 4], 700.0, stop_early=False)
    assert [r["at_once"] for r in every] == [1, 2, 4]          # the 50-100 at once the gate needs, however slow the card


def test_probe_options_exist_and_the_log_folder_is_made():
    text = (ROOT / "scripts" / "capacity_probe.py").read_text(encoding="utf-8")
    assert '"--no-stop"' in text and '"--model-only"' in text
    assert text.count('(ROOT / "logs").mkdir(exist_ok=True)') == 2        # both ways of finishing write logs/capacity_probe.json


def test_probe_rows_say_how_many_requests_a_second_the_server_finished():
    mod = script("capacity_probe")
    import time as t

    def fn(i):
        t.sleep(0.01)
        return 10.0

    _, rows = mod.sweep("full", fn, [2], 700.0)
    assert rows[0]["requests_per_s"] > 0 and set(rows[0]) == {"at_once", "median_ms", "p90_ms", "requests_per_s"}

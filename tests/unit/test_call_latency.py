"""scripts/call_latency.py: the caller's wait on each turn of a recorded call, split into the turn end, hearing, the agent and the voice (owner, 29 Oct: "1.3 s to 5 s")."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _script():
    spec = importlib.util.spec_from_file_location("call_latency", ROOT / "scripts" / "call_latency.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_turn_is_split_into_its_waits(tmp_path):
    t0 = 1000.0
    loud = (np.sin(np.arange(320) / 3) * 8000).astype(np.int16)
    quiet = np.zeros(320, dtype=np.int16)
    frames = [loud] * 50 + [quiet] * 100                                     # 1 s of speech, then 2 s of silence
    (tmp_path / "20261029_120000_1111_in.pcm").write_bytes(b"".join(f.tobytes() for f in frames))
    (tmp_path / "20261029_120000_1111_in.csv").write_text("unix_time,bytes,sample_rate\n" + "".join(f"{t0 + i * 0.02:.3f},640,16000\n" for i in range(150)))
    (tmp_path / "20261029_120000_1111_in.txt").write_text(f"{t0 + 1.6}\tturn_end\t\n{t0 + 1.8}\tstt\tTell me about the Macan.\n{t0 + 2.9}\tsaid\tIt's black.\n")
    (tmp_path / "20261029_120000_2222_out.csv").write_text(f"unix_time,bytes,sample_rate\n{t0 + 3.3},4800,24000\n")
    m = _script()
    [row] = m.turns(*m.load(tmp_path / "20261029_120000_1111_in.txt"))
    assert abs(row["turn_end"] - 0.6) < 0.03 and abs(row["hearing"] - 0.2) < 0.01
    assert abs(row["agent"] - 1.1) < 0.01 and abs(row["voice"] - 0.4) < 0.01 and abs(row["total"] - 2.3) < 0.03


def test_a_call_with_no_caller_audio_is_skipped_not_a_crash(tmp_path, monkeypatch, capsys):
    """29 Oct: the latest recording was a call that ended at the greeting (0 bytes of caller audio) and the script crashed on it."""
    m = _script()
    folder = tmp_path / "logs" / "audio_dump"
    folder.mkdir(parents=True)
    (folder / "20261028_210725_5312_in.txt").write_text("1793207245.846\tsaid\tHi, welcome.\n")
    (folder / "20261028_210725_5312_in.pcm").write_bytes(b"")
    (folder / "20261028_210725_5312_in.csv").write_text("unix_time,bytes,sample_rate\n")
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["call_latency.py"])
    assert m.main() == 1 and "VA_DUMP_AUDIO_DIR" in capsys.readouterr().out

"""scripts/score_overlap.py replays recorded turns through the real detector and endpointer (docs/ARCH_REVIEW.md section 22). The pure parts, and one replay when the Silero and
Smart Turn models are available."""
import asyncio
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def script():
    spec = importlib.util.spec_from_file_location("score_overlap_script", ROOT / "scripts" / "score_overlap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_turn_is_cut_into_20_ms_frames_and_followed_by_silence():
    mod = script()
    frames = mod.frames_of(np.full(1000, 0.5, dtype=np.float32))
    assert all(len(f) == 640 for f in frames)                                      # 320 samples of 16-bit
    assert len(frames) * 0.02 >= 1000 / 16000 + mod.TAIL_S
    assert frames[-1] == bytes(640)                                                # ends in digital silence so the detector can finish the turn


def test_a_nonsense_clip_has_no_usable_early_start_and_nothing_is_recognised():
    """No speech: the detector never opens a turn, the endpointer never starts anything, and the recogniser service is never asked."""
    pytest.importorskip("pipecat")
    mod = script()
    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

    from dealer_core.config import load_config

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    analyzer = LocalSmartTurnAnalyzerV3(sample_rate=16000, params=SmartTurnParams(stop_secs=cfg.voice.smart_fallback_secs))
    analyzer.set_sample_rate(16000)
    out = asyncio.run(mod.replay(np.zeros(16000, dtype=np.float32), 0.2, cfg, analyzer, 0.6, 0.0))
    assert out["segments"] == [] and out["starts"] == [] and out["started"] == 0

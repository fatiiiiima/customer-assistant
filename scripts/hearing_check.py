r"""How well does the recognizer hear real turns? Replays a recorded call (logs/audio_dump, written when VA_DUMP_AUDIO_DIR is set) through the recognizer
settings you choose and scores the words against what was really said.

  uv run python scripts/hearing_check.py 20261027_215011_9456_in
  uv run python scripts/hearing_check.py 20261027_215011_9456_in --model whisper-small.en-int8-ov     (a second model, once its folder is under models/ov)
  uv run python scripts/hearing_check.py 20261027_215011_9456_in --stt-url http://stt-server:8000/v1 --stt-model openai/whisper-large-v3-turbo     (the GPU server the workers use)

Next to the recording (logs/audio_dump/<stem>.pcm) put two small text files, one line per turn, in order:
  <stem>.ref.txt     what the caller really said
  <stem>.agent.txt   what the agent had just said before that turn (it shapes the recognizer's hint); a missing line means no hint context

Scores: word error rate (WER = word edits / words said, lower is better), turns heard exactly right, turns where Whisper looped, median time.
The recordings hold the caller's voice: they stay in logs/ (not committed). This is the "measure hearing without earphones" step of docs/ARCH_REVIEW.md.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def wer(ref: str, hyp: str) -> tuple[int, int]:
    """(word edits, words in the reference)."""
    from rapidfuzz.distance import Levenshtein

    r = words_of(ref)
    return Levenshtein.distance(r, words_of(hyp)), len(r)


def load_turns(stem: str, folder: Path, limit: int | None) -> list[np.ndarray]:
    """One clip per tap-to-end-of-turn stretch of the recording (a gap of more than 1 s between chunks starts a new one)."""
    rows = list(csv.DictReader(open(folder / f"{stem}.csv")))
    if not rows:                                           # a call whose microphone never delivered audio (the index has only its header)
        return []
    t = np.array([float(r["unix_time"]) for r in rows])
    n = np.array([int(r["bytes"]) for r in rows])
    pcm = np.frombuffer(open(folder / f"{stem}.pcm", "rb").read(), dtype="<i2").astype(np.float32) / 32768
    off = np.concatenate([[0], np.cumsum(n // 2)])
    cuts = [0] + [i + 1 for i in np.where(np.diff(t) > 1.0)[0]] + [len(t)]
    clips = [pcm[off[a]:off[b]] for a, b in zip(cuts[:-1], cuts[1:]) if t[b - 1] - t[a] > 1.0]
    return clips[:limit] if limit else clips


def build_engines(models: list[str], precisions: str, stt_url: str | None, stt_model: str, timeout_s: float, hotwords: list[str]) -> list[tuple[str, str, object]]:
    """(name, math, recognizer) to score. With `stt_url`: the transcription server the workers use (voice.stt_url), once; its math is the server's own.
    Without: each in-process OpenVINO model at each CPU math setting."""
    if stt_url:
        from dealer_core.voice.stt import RemoteWhisper

        return [(stt_model, "server", RemoteWhisper(stt_url, stt_model, timeout_s, hotwords))]
    from dealer_core.voice.stt import WhisperOpenVino

    return [(m, p, WhisperOpenVino(ROOT / "models" / "ov" / m, ROOT / "models" / "ov_cache", "CPU", hotwords, precision="" if p == "default" else p))
            for m in models for p in precisions.split(",")]


def lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem", help="recording name in logs/audio_dump, without .pcm")
    ap.add_argument("--folder", type=Path, default=ROOT / "logs" / "audio_dump")
    ap.add_argument("--model", action="append", help="a recognizer folder under models/ov (repeatable; default voice.stt_model)")
    ap.add_argument("--precisions", default="default,f32", help="comma list of CPU math settings to compare (default = OpenVINO's own)")
    ap.add_argument("--stt-url", help="score a transcription server (OpenAI API: vLLM Whisper, speaches, dealer_core.sttserver) instead of the in-process recognizer, e.g. http://stt-server:8000/v1")
    ap.add_argument("--stt-model", help="the model name that server expects (default voice.stt_remote_model)")
    a = ap.parse_args()

    from dealer_core.app import build_app
    from dealer_core.config import load_config
    from dealer_core.dialogue.states import State
    from dealer_core.voice.prompts import hotword_groups, stt_prompt
    from dealer_core.voice.stt import hotword_terms, is_degenerate

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    ref, agent = lines(a.folder / f"{a.stem}.ref.txt"), lines(a.folder / f"{a.stem}.agent.txt")
    if not ref:
        print(f"no {a.stem}.ref.txt next to the recording: write what was said, one line per turn")
        return 1
    clips = load_turns(a.stem, a.folder, len(ref))
    site = ROOT.parent / "bmw-portal" / "site" / "data.js"
    app = build_app(cfg, inventory="portal" if site.exists() else "demo", portal_data=site if site.exists() else None, reply_mode="template",
                    classifier=type("Stub", (), {"name": "stub", "classify": lambda *x, **k: None})())
    groups, vocab = hotword_groups(ROOT / cfg.stt.hotwords_file), app.inventory.vocabulary()
    hotwords = hotword_terms(ROOT / cfg.stt.hotwords_file)
    print(f"{a.stem}: {len(clips)} turns of audio, {len(ref)} reference lines, {len(agent)} agent lines\n")
    print(f"{'model':<26}{'math':<9}{'hint':<6}{'WER':>6}{'exact':>8}{'loops':>7}{'median ms':>11}")
    for model, prec, engine in build_engines(a.model or [cfg.voice.stt_model], a.precisions, a.stt_url, a.stt_model or cfg.voice.stt_remote_model, cfg.voice.stt_timeout_s, hotwords):
        engine.warm_up()
        for hint in (True, False):
            edits = total = exact = loops = 0
            ms = []
            for i, (clip, said) in enumerate(zip(clips, ref)):
                prompt = stt_prompt(State.ANSWERING, agent[i] if i < len(agent) else "", groups, vocab) if hint else ""
                t0 = time.perf_counter()
                heard = engine.transcribe_float(clip, prompt)
                ms.append((time.perf_counter() - t0) * 1000)
                e, n = wer(said, heard)
                edits, total, exact = edits + e, total + n, exact + (e == 0)
                loops += is_degenerate(heard, len(clip) / 16000)
            print(f"{model[:25]:<26}{prec:<9}{'on' if hint else 'off':<6}{edits / total * 100:5.0f}%{f'{exact}/{len(clips)}':>8}{loops:>7}{statistics.median(ms):>11.0f}")
            if getattr(engine, "errors", 0):             # a server error is "nothing heard" in the worker; here it would pass as a very bad score
                print(f"  {engine.errors} requests to the server failed: read the scores above as the server's failures, not its hearing")
    print("\nSmall sample: read differences of a few points as noise. Numbers said as digits vs words ('9' / 'nine') count as errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

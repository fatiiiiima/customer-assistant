"""Benchmark-only engine variants used to TEST proposed decoding guards. Nothing here changes src/.

  fwt0:<size>[+hot]                 faster-whisper with temperature fixed at 0 (no random sampling fallback)
  ovg:<model>@<DEV>[+hot][+cap][+rep][+chk]
        OpenVINO Whisper with proposed guards:
          cap  max_new_tokens tied to clip duration:  int(12 + 8 * seconds)  (the default is a fixed 448)
          rep  repetition blocking: no_repeat_ngram_size=6 and repetition_penalty=1.1
          chk  post-hoc no-speech/loop check (like faster-whisper's compression-ratio test): reject the text if
               zlib compression ratio > 2.4 or one word repeats >= 8 times in a row
All other specs go to bench_stt.build.
"""
from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_stt import Engine, FasterWhisper, OvWhisper, build  # noqa: E402
from common import MODELS, hotword_terms, now  # noqa: E402


def looks_like_loop(text: str) -> bool:
    t = text.strip()
    if len(t) > 20 and len(t.encode()) / max(1, len(zlib.compress(t.encode()))) > 2.4:
        return True
    return bool(re.search(r"\b(\w+)(?:[\s,.]+\1\b){7,}", t, flags=re.I))


class FWFixedTemp(FasterWhisper):
    def __init__(self, size: str, threads: int, hotwords: bool) -> None:
        super().__init__(size, threads, hotwords)
        self.name = f"faster-whisper:{size}-int8 temp=0" + ("+hotwords" if hotwords else "")

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        t0 = now()
        segs, _ = self.m.transcribe(audio, language="en", beam_size=1, condition_on_previous_text=False,
                                    without_timestamps=True, hotwords=self.hot, temperature=0.0)
        return " ".join(s.text for s in segs).strip(), (now() - t0) * 1000


class OvGuarded(OvWhisper):
    def __init__(self, model: str, device: str, hotwords: bool, threads: int, cap: bool, rep: bool, chk: bool) -> None:
        super().__init__(model, device, hotwords, threads)
        self.cap, self.rep, self.chk = cap, rep, chk
        self.name = (f"openvino-whisper:{model}@{device}" + ("+prompt" if hotwords else "") + "".join(
            f" [{n}]" for n, on in (("cap", cap), ("rep", rep), ("chk", chk)) if on))
        if rep:
            self.cfg.no_repeat_ngram_size = 6
            self.cfg.repetition_penalty = 1.1

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        if self.cap:
            self.cfg.max_new_tokens = int(12 + 8 * len(audio) / 16000)
        text, ms = super().transcribe(audio)
        if self.chk and looks_like_loop(text):
            text = ""
        return text, ms


def build_ext(spec: str, threads: int) -> Engine:
    kind, _, arg = spec.partition(":")
    if kind == "fwt0":
        size, _, hw = arg.partition("+")
        return FWFixedTemp(size, threads, hw == "hot")
    if kind == "ovg":
        model, _, rest = arg.partition("@")
        parts = rest.split("+")
        flags = set(parts[1:])
        return OvGuarded(model, parts[0], "hot" in flags, threads, "cap" in flags, "rep" in flags, "chk" in flags)
    eng = build(spec, threads)
    if kind == "ovw" and ".en-" in arg:  # English-only OpenVINO builds reject a language token
        eng.cfg.language = None
        eng.cfg.task = None
    return eng

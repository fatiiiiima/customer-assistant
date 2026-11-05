"""What the caller has actually heard of the agent's reply, worked out on the server (barge-in, owner 29 Oct: "while the model is speaking, if I interrupt it, it is
able to understand and changes the response").

The server cannot see the page's speaker. It sends audio about twice as fast as it plays (pipecat's websocket output paces at half the audio's length), so its own
"bot started / stopped speaking" ends around half way through what the caller hears. The page plays each chunk back to back, starting no earlier than CUSHION_S after
it arrives (site/voice.js `play`). Replaying that here gives, at any moment, whether the agent is audible and how far into each sentence the caller has got.

`Playback` is the model; `PlaybackTracker` sits right before the output transport, feeds it the sentences and their audio, and on an interruption tells the brain
what was heard (`on_cut(text)`), so the call history keeps only that.
"""
from __future__ import annotations

import time

from pipecat.frames.frames import AggregatedTextFrame, Frame, InterruptionFrame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

CUSHION_S = 0.12          # the page starts a chunk at least this long after it arrives (network jitter cushion in site/voice.js)
NETWORK_S = 0.03          # the chunk's way to the page on a local network (a guess; a few hundredths either way changes a word at most)


def _weight(word: str) -> int:
    """How long a written word takes to say, roughly: the length of its spoken form ("62,500" is "sixty two thousand five hundred")."""
    from .tts import pronounce

    return max(1, len(pronounce(word).replace(" ", "")))


class Playback:
    def __init__(self, clock=time.monotonic, cushion_s: float = CUSHION_S) -> None:
        self.clock, self.cushion_s = clock, cushion_s
        self.new_reply()

    def new_reply(self) -> None:
        """A new reply starts (or the old one was thrown away): forget the old one's sentences and audio."""
        self._sentences: list[str] = []
        self._chunks: list[tuple[float, float, int]] = []        # (play start, seconds, sentence index)
        self._send_free = 0.0                                     # when the transport can send the next chunk (it paces at half the audio's length)
        self._play_free = 0.0                                     # when the page's speaker is free

    def sentence(self, text: str) -> None:
        self._sentences.append(text)

    def text(self) -> str:
        """The reply being played, as written."""
        return " ".join(self._sentences)

    def audio(self, seconds: float, at: float | None = None) -> None:
        """A chunk of the latest sentence reached the output transport at `at`."""
        if not self._sentences or seconds <= 0:
            return
        now = self.clock() if at is None else at
        sent = max(now, self._send_free)
        self._send_free = sent + seconds / 2
        start = max(self._play_free, sent + NETWORK_S + self.cushion_s)
        self._play_free = start + seconds
        self._chunks.append((start, seconds, len(self._sentences) - 1))

    def playing(self, at: float | None = None) -> bool:
        """Is the agent audible to the caller right now?"""
        now = self.clock() if at is None else at
        return bool(self._chunks) and self._chunks[0][0] <= now < self._play_free

    def started(self, at: float | None = None) -> bool:
        now = self.clock() if at is None else at
        return bool(self._chunks) and self._chunks[0][0] <= now

    def heard(self, at: float | None = None) -> str:
        """The words of this reply the caller has heard by now: whole sentences, then the start of the one being spoken (weighted by how long each word
        takes to say, rounded down: never a word that was not heard), ending in "..." when it was cut."""
        now = self.clock() if at is None else at
        played = [0.0] * len(self._sentences)
        total = [0.0] * len(self._sentences)
        for start, secs, i in self._chunks:
            total[i] += secs
            played[i] += min(secs, max(0.0, now - start))
        out = []
        for text, p, t in zip(self._sentences, played, total):
            if t <= 0 or p <= 0:
                break
            if p >= t - 1e-6:
                out.append(text)
                continue
            words = text.split()
            weights = [_weight(w) for w in words]
            budget, kept, acc = p / t * sum(weights), [], 0
            for w, wt in zip(words, weights):
                if acc + wt > budget:
                    break
                acc += wt
                kept.append(w)
            if kept:
                out.append(" ".join(kept).rstrip(",;:") + " ...")
            break
        return " ".join(out)


class PlaybackTracker(FrameProcessor):
    """Before the output transport: tells `Playback` each sentence and its audio, and on an interruption hands what the caller heard to `on_cut`."""

    def __init__(self, playback: Playback, on_cut=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.playback, self.on_cut = playback, on_cut
        self.cuts: list[str] = []

    def observe(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        if direction is not FrameDirection.DOWNSTREAM:
            return
        if isinstance(frame, AggregatedTextFrame):
            self.playback.sentence(frame.text)
        elif isinstance(frame, TTSAudioRawFrame):
            self.playback.audio(len(frame.audio) / (2 * frame.sample_rate * max(1, frame.num_channels)))
        elif isinstance(frame, InterruptionFrame):
            if self.playback.started():
                heard = self.playback.heard()
                self.cuts.append(heard)
                if self.on_cut is not None:
                    self.on_cut(heard)
            self.playback.new_reply()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.observe(frame, direction)
        await self.push_frame(frame, direction)

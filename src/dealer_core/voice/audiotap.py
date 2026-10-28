"""Diagnostic recorder, OFF unless the environment variable VA_DUMP_AUDIO_DIR names a folder.

It copies the audio that passes through one point of the pipeline to disk, so a "it did not hear me" or "the voice breaks up" report can be
checked against what the server really received and really sent:

  <stamp>_in.pcm / .csv    what the voice detector is given: 16 kHz mono signed 16-bit PCM, only while the microphone is open
  <stamp>_out.pcm / .csv   what is sent to the page: the voice's own sample rate (in the csv)
  .csv: unix_time of each chunk, its size in bytes, its sample rate (to line the audio up with the worker log's clock, and to see gaps)
  <stamp>_in.txt          what was recognised and said, with the time: "stt" the recognizer's words for a segment, "heard" what the agent was given (after any
                          joining of a sentence cut by a pause), "said" each sentence the agent spoke. So a bad call can be scored against its audio without anyone
                          retyping the chat.

The files hold the caller's voice. They stay on this machine (logs/ is not committed); delete the folder when done. Never on by default.

The files are written by a thread of their own: `record()` and `note()` run on the event loop that carries every call's audio and only put the chunk on a queue (with
the time it arrived, so the csv's clock is the frame's, not the write's). Before B5-B8 (docs/ARCH_REVIEW.md section 20) each 20 ms frame was written and flushed on the loop:
four system calls per frame per call, for as long as the recorder was on. Flushed about once a second, and at the end of the call.
"""
from __future__ import annotations

import asyncio
import pathlib
import queue
import threading
import time

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


_STOP = object()
FLUSH_EVERY_S = 1.0


class AudioTap(FrameProcessor):
    def __init__(self, directory: str | pathlib.Path, name: str, frame_type: type, **kwargs) -> None:
        super().__init__(**kwargs)
        d = pathlib.Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{id(self) % 10000:04d}_{name}"
        self._frame_type = frame_type
        self._pcm = open(d / f"{stem}.pcm", "wb")
        self._idx = open(d / f"{stem}.csv", "w")
        self._idx.write("unix_time,bytes,sample_rate\n")
        self._notes = open(d / f"{stem}.txt", "w", encoding="utf-8") if name == "in" else None       # the words are kept with the input audio only
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._write_loop, name=f"audiotap-{name}", daemon=True)
        self._thread.start()

    def record(self, audio: bytes, sample_rate: int) -> None:
        self._queue.put(("audio", time.time(), audio, sample_rate))

    def note(self, kind: str, text: str) -> None:
        if self._notes is not None:
            self._queue.put(("note", time.time(), kind, text))

    def _files(self) -> list:
        return [f for f in (self._pcm, self._idx, self._notes) if f is not None]

    def _write_loop(self) -> None:
        """The only place the files are touched while the call runs. A write that fails (disk full) is dropped: a diagnostic must never take a call down."""
        flushed = time.monotonic()
        while True:
            try:
                item = self._queue.get(timeout=FLUSH_EVERY_S)
            except queue.Empty:
                item = None
            if item is _STOP:
                break
            try:
                if item is not None and item[0] == "audio":
                    _, at, audio, rate = item
                    self._pcm.write(audio)
                    self._idx.write(f"{at:.3f},{len(audio)},{rate}\n")
                elif item is not None:
                    _, at, kind, text = item
                    self._notes.write(f"{at:.3f}\t{kind}\t{text}\n")
                if time.monotonic() - flushed >= FLUSH_EVERY_S:
                    for f in self._files():
                        f.flush()
                    flushed = time.monotonic()
            except Exception:
                pass
        for f in self._files():
            try:
                f.close()
            except Exception:
                pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM and isinstance(frame, self._frame_type):
            self.record(frame.audio, frame.sample_rate)
        await self.push_frame(frame, direction)

    async def cleanup(self):
        await super().cleanup()
        self._queue.put(_STOP)                          # everything queued before this is written, then the files are closed
        await asyncio.to_thread(self._thread.join, 10.0)

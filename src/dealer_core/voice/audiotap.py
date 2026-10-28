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
"""
from __future__ import annotations

import pathlib
import time

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


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

    def record(self, audio: bytes, sample_rate: int) -> None:
        self._pcm.write(audio)
        self._pcm.flush()
        self._idx.write(f"{time.time():.3f},{len(audio)},{sample_rate}\n")
        self._idx.flush()

    def note(self, kind: str, text: str) -> None:
        if self._notes is not None:
            self._notes.write(f"{time.time():.3f}\t{kind}\t{text}\n")
            self._notes.flush()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM and isinstance(frame, self._frame_type):
            self.record(frame.audio, frame.sample_rate)
        await self.push_frame(frame, direction)

    async def cleanup(self):
        await super().cleanup()
        for f in (self._pcm, self._idx, self._notes):
            try:
                if f is None:
                    continue
                f.close()
            except Exception:
                pass

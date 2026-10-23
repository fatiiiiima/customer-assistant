"""Continuous batching for the OpenVINO model: every caller's request is served together instead of one after another.

Measured on this laptop (Qwen3-4B int4, iGPU, 40 new tokens, scripts/bench_llm_concurrency.py):
  one pipeline, one request at a time:  8 requests at once -> first token 5.1 s median (10.6 s worst), done 6.6 s median
  continuous batching + prefix cache:   8 requests at once -> first token 0.23 s,                     done 2.4 s median
  one request alone, prompt start cached: first token 0.08 s, done 1.07 s (the single pipeline: 0.20 s / 1.27 s)

`BatchedPipeline` looks like `openvino_genai.LLMPipeline` to the backend (generate(prompt, config, streamer) and get_tokenizer()), so the backend
code is the same either way. One thread steps the scheduler; requests are added and stopped from any thread.
"""
from __future__ import annotations

import itertools
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _Request:
    prompt: str
    config: object
    streamer: object | None
    handle: object = None
    ids: list[int] = field(default_factory=list)
    text: str = ""
    done: threading.Event = field(default_factory=threading.Event)
    stop: bool = False
    error: BaseException | None = None


class _Result:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    def __str__(self) -> str:
        return self.texts[0] if self.texts else ""


class BatchedPipeline:
    def __init__(self, model_path: Path, device: str, cache_dir: Path, kv_cache_gb: float = 2.0, max_batched_tokens: int = 4096) -> None:
        import openvino_genai as ov

        self._ov = ov
        sc = ov.SchedulerConfig()
        sc.enable_prefix_caching = True                 # the system prompt and the call so far are shared: only the new part is computed
        sc.cache_size = max(1, int(round(kv_cache_gb)))
        sc.max_num_batched_tokens = max_batched_tokens
        self._cb = ov.ContinuousBatchingPipeline(str(model_path), sc, device, {"CACHE_DIR": str(cache_dir)})
        self._tok = self._cb.get_tokenizer()
        self._new: queue.Queue[_Request] = queue.Queue()
        self._ids = itertools.count(1)
        self._wake = threading.Event()
        threading.Thread(target=self._loop, name="ov-batching", daemon=True).start()

    def get_tokenizer(self):
        return self._tok

    def generate(self, prompt, config, streamer=None):
        """Same contract as LLMPipeline.generate: a list of prompts returns an object with .texts; one prompt returns the text."""
        if isinstance(prompt, list):
            reqs = [self._submit(p, config, None) for p in prompt]
            return _Result([self._wait(r) for r in reqs])
        return self._wait(self._submit(prompt, config, streamer))

    # -- internals -------------------------------------------------------------------------------
    def _submit(self, prompt: str, config, streamer) -> _Request:
        r = _Request(prompt, config, streamer)
        self._new.put(r)
        self._wake.set()
        return r

    @staticmethod
    def _wait(r: _Request) -> str:
        r.done.wait()
        if r.error is not None:
            raise r.error
        return r.text

    def _loop(self) -> None:
        ov = self._ov
        active: list[_Request] = []
        while True:
            if not active and self._new.empty():
                self._wake.wait()
            self._wake.clear()
            while not self._new.empty():
                r = self._new.get()
                try:
                    r.handle = self._cb.add_request(next(self._ids), r.prompt, r.config)
                    active.append(r)
                except Exception as e:                                       # bad config / prompt: fail this request only
                    r.error = e
                    r.done.set()
            if not active:
                continue
            try:
                self._cb.step()
            except Exception as e:                                           # the device failed: fail everything in flight
                for r in active:
                    r.error = e
                    r.done.set()
                active = []
                continue
            still: list[_Request] = []
            for r in active:
                self._drain(r)
                running = r.handle.get_status() == ov.GenerationStatus.RUNNING
                if r.stop and running:
                    r.handle.stop()
                    running = False
                if running:
                    still.append(r)
                else:
                    self._drain(r)
                    r.done.set()
            active = still

    def _drain(self, r: _Request) -> None:
        """New tokens since the last step -> new text for the streamer (decoded as a whole, so multi-token characters come out right)."""
        if not r.handle.can_read():
            return
        for out in r.handle.read().values():
            r.ids.extend(out.generated_ids)
        text = self._tok.decode(r.ids, skip_special_tokens=True) if r.ids else ""
        if text.endswith("�"):                                          # half a character: wait for the rest
            return
        piece, r.text = text[len(r.text):], text
        if piece and r.streamer is not None and not r.stop:
            try:
                status = r.streamer(piece)
            except Exception as e:
                r.error, r.stop = e, True
                return
            if status == self._ov.StreamingStatus.STOP or status is True:
                r.stop = True

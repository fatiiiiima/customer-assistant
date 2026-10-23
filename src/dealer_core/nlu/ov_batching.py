"""Continuous batching for the OpenVINO model: every caller's request is served together instead of one after another.

Measured on this laptop (Qwen3-4B int4, iGPU, 40 new tokens, scripts/bench_llm_concurrency.py):
  one pipeline, one request at a time:  8 requests at once -> first token 5.1 s median (10.6 s worst), done 6.6 s median
  continuous batching + prefix cache:   8 requests at once -> first token 0.23 s,                     done 2.4 s median
  one request alone, prompt start cached: first token 0.08 s, done 1.07 s (the single pipeline: 0.20 s / 1.27 s)

The batching scheduler runs in its OWN PROCESS. Its step() holds Python's interpreter lock while the iGPU works (measured: other threads stalled
for up to 0.83 s, so the voice of every call stuttered and started late); in a separate process it cannot stall the calls.

`BatchedPipeline` looks like `openvino_genai.LLMPipeline` to the backend (generate(prompt, config, streamer) and get_tokenizer()), so the backend
code is the same either way.
"""
from __future__ import annotations

import itertools
import multiprocessing as mp
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path


def _config_to_dict(cfg) -> dict:
    """The GenerationConfig fields the backend uses, as plain data (the config object does not cross a process boundary)."""
    so = getattr(cfg, "structured_output_config", None)
    return {"max_new_tokens": int(cfg.max_new_tokens), "do_sample": bool(cfg.do_sample), "temperature": float(cfg.temperature),
            "top_p": float(cfg.top_p), "top_k": int(cfg.top_k), "json_schema": (so.json_schema if so is not None else None)}


def _config_from_dict(ov, d: dict):
    cfg = ov.GenerationConfig()
    cfg.max_new_tokens, cfg.do_sample = d["max_new_tokens"], d["do_sample"]
    if d["do_sample"]:
        cfg.temperature, cfg.top_p, cfg.top_k = d["temperature"], d["top_p"], d["top_k"]
    if d.get("json_schema"):
        cfg.structured_output_config = ov.StructuredOutputConfig(json_schema=d["json_schema"])
    return cfg


# ------------------------------------------------------------------------------------------------ the model process
@dataclass
class _Job:
    rid: int
    handle: object
    ids: list[int] = field(default_factory=list)
    text: str = ""
    stop: bool = False


def _serve(model_path: str, device: str, cache_dir: str, kv_cache_gb: float, max_batched_tokens: int, conn) -> None:
    """Child process: owns the pipeline; requests arrive on `conn`, text pieces go back on it."""
    import openvino_genai as ov

    sc = ov.SchedulerConfig()
    sc.enable_prefix_caching = True                 # the system prompt and the call so far are shared: only the new part is computed
    sc.cache_size = max(1, int(round(kv_cache_gb)))
    sc.max_num_batched_tokens = max_batched_tokens
    cb = ov.ContinuousBatchingPipeline(model_path, sc, device, {"CACHE_DIR": cache_dir})
    tok = cb.get_tokenizer()
    inbox: queue.Queue = queue.Queue()
    wake = threading.Event()
    send_lock = threading.Lock()

    def send(msg) -> None:
        with send_lock:
            conn.send(msg)

    def reader() -> None:
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                msg = ("quit",)
            inbox.put(msg)
            wake.set()
            if msg[0] == "quit":
                return

    threading.Thread(target=reader, daemon=True).start()
    send(("ready",))
    jobs: dict[int, _Job] = {}
    while True:
        if not jobs and inbox.empty():
            wake.wait()
        wake.clear()
        while not inbox.empty():
            msg = inbox.get()
            if msg[0] == "quit":
                return
            if msg[0] == "add":
                _, rid, prompt, cfg = msg
                try:
                    jobs[rid] = _Job(rid, cb.add_request(rid, prompt, _config_from_dict(ov, cfg)))
                except Exception as e:                                   # bad config / prompt: fail this request only
                    send(("error", rid, f"{type(e).__name__}: {e}"))
            elif msg[0] == "stop" and msg[1] in jobs:
                jobs[msg[1]].stop = True
        if not jobs:
            continue
        try:
            cb.step()
        except Exception as e:                                           # the device failed: fail everything in flight
            for rid in jobs:
                send(("error", rid, f"{type(e).__name__}: {e}"))
            jobs.clear()
            continue
        for rid, j in list(jobs.items()):
            if j.handle.can_read():
                for out in j.handle.read().values():
                    j.ids.extend(out.generated_ids)
                text = tok.decode(j.ids, skip_special_tokens=True) if j.ids else ""
                if not text.endswith("�") and len(text) > len(j.text):   # a half character waits for the rest
                    send(("piece", rid, text[len(j.text):]))
                    j.text = text
            running = j.handle.get_status() == ov.GenerationStatus.RUNNING
            if j.stop and running:
                j.handle.stop()
                running = False
            if not running:
                send(("done", rid))
                del jobs[rid]


# ------------------------------------------------------------------------------------------------ the proxy the backend uses
class _Result:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    def __str__(self) -> str:
        return self.texts[0] if self.texts else ""


class BatchedPipeline:
    def __init__(self, model_path: Path, device: str, cache_dir: Path, kv_cache_gb: float = 2.0, max_batched_tokens: int = 4096,
                 start_timeout_s: float = 300.0) -> None:
        import openvino_genai as ov

        self._ov = ov
        self._tok = ov.Tokenizer(str(model_path))                       # for the chat template, in this process
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_serve, args=(str(model_path), device, str(cache_dir), kv_cache_gb, max_batched_tokens, child),
                                 name="ov-batching", daemon=True)
        self._proc.start()
        if not self._conn.poll(start_timeout_s) or self._conn.recv()[0] != "ready":
            raise RuntimeError("the model process did not start")
        self._ids = itertools.count(1)
        self._waiting: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._receive, name="ov-batching-rx", daemon=True).start()

    def get_tokenizer(self):
        return self._tok

    def generate(self, prompt, config, streamer=None):
        """Same contract as LLMPipeline.generate: a list of prompts returns an object with .texts; one prompt returns the text."""
        if isinstance(prompt, list):
            rids = [self._submit(p, config) for p in prompt]
            return _Result([self._collect(r, None) for r in rids])
        return self._collect(self._submit(prompt, config), streamer)

    # -- internals -------------------------------------------------------------------------------
    def _submit(self, prompt: str, config) -> int:
        rid = next(self._ids)
        with self._lock:
            self._waiting[rid] = queue.Queue()
            self._conn.send(("add", rid, prompt, _config_to_dict(config)))
        return rid

    def _collect(self, rid: int, streamer) -> str:
        q, text, stopped = self._waiting[rid], "", False
        try:
            while True:
                kind, payload = q.get()
                if kind == "done":
                    return text
                if kind == "error":
                    raise RuntimeError(payload)
                text += payload
                if streamer is not None and not stopped:
                    status = streamer(payload)
                    if status == self._ov.StreamingStatus.STOP or status is True:
                        stopped = True
                        with self._lock:
                            self._conn.send(("stop", rid))
        finally:
            with self._lock:
                self._waiting.pop(rid, None)

    def _receive(self) -> None:
        while True:
            try:
                msg = self._conn.recv()
            except (EOFError, OSError):
                with self._lock:
                    waiting = list(self._waiting.values())
                for q in waiting:
                    q.put(("error", "the model process stopped"))
                return
            q = self._waiting.get(msg[1])
            if q is not None:
                q.put((msg[0], msg[2] if len(msg) > 2 else None))

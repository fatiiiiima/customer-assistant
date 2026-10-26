r"""One model for every voice worker on the machine, behind the same OpenAI-compatible API vLLM offers.

  .venv\Scripts\python.exe -m dealer_core.modelserver --model Qwen3-4B-Instruct-2507-int4-ov-nocal --device GPU --port 8001

Workers then use a backend with `runtime: openai` and `host: http://127.0.0.1:8001/v1` (config `openvino_server`). In production, the same
workers point at vLLM on the company's GPUs instead; nothing else changes.

Serves POST /v1/chat/completions (streaming and not), GET /v1/models, GET /health. Requests from all workers are batched together
(continuous batching + prefix cache, see nlu/ov_batching.py). Supports the two extensions the agent uses: `guided_json` (a JSON-schema
constrained answer) and `continue_final_message` (continue an assistant turn code has begun).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[3]


def create_app(model_dir: Path, device: str, kv_cache_gb: float = 2.0, served_name: str | None = None):
    import openvino_genai as ov

    from ..nlu.ov_batching import BatchedPipeline

    pipe = BatchedPipeline(model_dir, device, ROOT / "models" / "ov_cache", kv_cache_gb=kv_cache_gb)
    tok = pipe.get_tokenizer()
    name = served_name or model_dir.name
    api = FastAPI(title="local model server")
    stats = {"requests": 0, "in_flight": 0, "started": time.time()}

    def prompt_of(body: dict) -> str:
        msgs = [{"role": m["role"], "content": m.get("content") or ""} for m in body.get("messages", [])]
        prefill = ""
        if body.get("continue_final_message") and msgs and msgs[-1]["role"] == "assistant":
            prefill = msgs.pop()["content"]
        if model_dir.name.lower().startswith("mistral") and msgs and msgs[0]["role"] == "system":   # no system role in Mistral's template
            first = msgs.pop(0)["content"]
            if msgs and msgs[0]["role"] == "user":
                msgs[0] = {"role": "user", "content": first + "\n\n" + msgs[0]["content"]}
        thinking = bool((body.get("chat_template_kwargs") or {}).get("enable_thinking", False))
        return tok.apply_chat_template(msgs, add_generation_prompt=True, extra_context={"enable_thinking": thinking}) + prefill

    def config_of(body: dict):
        cfg = ov.GenerationConfig()
        cfg.max_new_tokens = int(body.get("max_tokens") or 128)
        t = float(body.get("temperature") or 0.0)
        cfg.do_sample = t > 0
        if t > 0:
            cfg.temperature, cfg.top_p, cfg.top_k = t, float(body.get("top_p") or 0.9), 40
        schema = body.get("guided_json") or ((body.get("response_format") or {}).get("json_schema") or {}).get("schema")
        if schema:
            cfg.structured_output_config = ov.StructuredOutputConfig(json_schema=json.dumps(schema))
        return cfg

    @api.get("/health")
    async def health() -> dict:
        return {"ok": True, "model": name, **stats}

    @api.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": name, "object": "model"}]}

    @api.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        try:
            prompt, cfg = prompt_of(body), config_of(body)
        except Exception as e:
            return JSONResponse({"error": {"message": f"{type(e).__name__}: {e}"}}, status_code=400)
        rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        stats["requests"] += 1
        if not body.get("stream"):
            stats["in_flight"] += 1
            try:
                text = await asyncio.to_thread(pipe.generate, prompt, cfg)
            finally:
                stats["in_flight"] -= 1
            return {"id": rid, "object": "chat.completion", "created": created, "model": name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}

        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def streamer(piece: str):
            loop.call_soon_threadsafe(q.put_nowait, piece)
            return ov.StreamingStatus.STOP if stop.is_set() else ov.StreamingStatus.RUNNING

        def work():
            try:
                pipe.generate(prompt, cfg, streamer)
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        async def events():
            stats["in_flight"] += 1
            threading.Thread(target=work, daemon=True).start()
            try:
                while True:
                    item = await q.get()
                    if item is None or isinstance(item, Exception):
                        break
                    chunk = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": name,
                             "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                stop.set()                                   # the worker hung up (or finished): stop generating
                stats["in_flight"] -= 1

        return StreamingResponse(events(), media_type="text/event-stream")

    return api


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen3-4B-Instruct-2507-int4-ov-nocal", help="folder under models/ov")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--kv-cache-gb", type=float, default=2.0)
    a = ap.parse_args()
    import uvicorn

    app = create_app(ROOT / "models" / "ov" / a.model, a.device, a.kv_cache_gb)
    print(f"Model server ready on http://{a.host}:{a.port}/v1  ({a.model} on {a.device})", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

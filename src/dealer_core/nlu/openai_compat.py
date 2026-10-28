"""A model behind any OpenAI-compatible server (vLLM, OpenVINO Model Server, TGI, llama.cpp server, SGLang) on the company's own machines.

This is how the agent runs on hardware it knows nothing about: whatever GPU is provided, the model is served by one of those servers (all
Apache-2.0 / MIT), and this backend talks to it over HTTP inside the company network. Continuous batching and prefix caching are the server's
job; vLLM does both. Nothing here leaves the network: `base_url` points at the company's server.

  backends:
    gpu_server:
      runtime: openai
      host: http://10.0.0.5:8000/v1          # the model server
      model: Qwen/Qwen3-14B-AWQ              # the name the server knows it by
      timeout_s: 6.0

Structured output for the intent call uses the server's JSON-schema support (vLLM: `guided_json`, others: `response_format`); the agent's
tool decision uses vLLM's `structured_outputs: {"choice": [...]}` (vLLM 0.12+, where it replaced `guided_choice`): the model can only answer
one of the listed actions; see `choose`. A reply begun
by code ("prefill") is continued with `continue_final_message` (vLLM) and, for servers without it, by telling the model what was said.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Iterator

from ..config import BackendCfg, LlmCfg
from ..dialogue.states import Intent, State
from . import prompts
from .backend import BackendError, BackendHealth, BackendTimeout, IntentResult, NluBackend, resolve_prompt_mode


class OpenAICompatBackend(NluBackend):
    def __init__(self, name: str, cfg: BackendCfg, llm: LlmCfg) -> None:
        import httpx

        self.name, self.cfg, self.llm = name, cfg, llm
        self.mode = resolve_prompt_mode(cfg)
        self.base = cfg.host.rstrip("/")
        key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
        self.client = httpx.Client(timeout=cfg.timeout_s, headers={"Authorization": f"Bearer {key}"} if key else {},
                                   limits=httpx.Limits(max_connections=cfg.max_parallel, max_keepalive_connections=cfg.max_parallel))
        self.can_choose = True                     # False once the server refused a constrained choice (then the decision is asked as free text)
        self.can_structure = True                  # False once the server refused a JSON schema (then the single-call turn is not used on it)

    def health(self) -> BackendHealth:
        try:
            r = self.client.get(f"{self.base}/models", timeout=5.0)
            r.raise_for_status()
            names = [m.get("id") for m in r.json().get("data", [])]
        except Exception as e:
            return BackendHealth(False, f"model server unreachable at {self.base}: {type(e).__name__}")
        if names and self.cfg.model not in names:
            return BackendHealth(False, f"model {self.cfg.model!r} not served at {self.base} (it serves {names[:3]})")
        return BackendHealth(True, f"openai-compatible {self.cfg.model} at {self.base}")

    # -- internals ---------------------------------------------------------------------------
    def _body(self, system: str, user: str, max_tokens: int, temperature: float | None, prefill: str = "", schema: dict | None = None,
              stream: bool = False) -> dict:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        body: dict = {"model": self.cfg.model, "messages": messages, "max_tokens": max_tokens, "stream": stream,
                      "temperature": self.llm.temperature if temperature is None else temperature}
        if temperature:
            body["top_p"] = 0.9
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
            body.update({"continue_final_message": True, "add_generation_prompt": False})   # vLLM: continue the reply code began
        if schema is not None:
            body["guided_json"] = schema                                                      # vLLM
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
        if "qwen3" in self.cfg.model.lower() and not self.llm.think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    def _post(self, body: dict, timeout: float) -> dict:
        try:
            r = self.client.post(f"{self.base}/chat/completions", json=body, timeout=timeout)
        except Exception as e:
            if "Timeout" in type(e).__name__:
                raise BackendTimeout(f"{self.name} exceeded {timeout:.1f}s") from e
            raise BackendError(f"{type(e).__name__}: {e}") from e
        if r.status_code == 400 and body.get("continue_final_message"):
            raise _NoPrefill()
        if r.status_code >= 400:
            raise BackendError(f"{self.name} HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def _json_call(self, state: State, car: str | None, text: str, last_agent: str, kind: str) -> tuple[dict, str, float]:
        schema = prompts.intent_only_schema(state) if kind == "intent" else prompts.full_schema(state)
        t0 = time.perf_counter()
        out = self._post(self._body(prompts.system_prompt(state, self.mode), prompts.user_message(state, car, text, last_agent),
                                    self.cfg.max_intent_tokens if kind == "intent" else self.cfg.max_json_tokens, 0.0, schema=schema), self.cfg.timeout_s)
        raw = out["choices"][0]["message"]["content"] or ""
        try:
            return json.loads(raw), raw, (time.perf_counter() - t0) * 1000
        except json.JSONDecodeError as e:
            raise BackendError(f"malformed JSON from {self.name}: {raw!r}") from e

    # -- interface ---------------------------------------------------------------------------
    def classify(self, state: State, car: str | None, text: str, last_agent: str = "") -> IntentResult:
        obj, raw, ms = self._json_call(state, car, text, last_agent, "intent")
        return IntentResult(Intent(obj["intent"]), ms, self.name, raw)

    def complete_json(self, state: State, car: str | None, text: str) -> dict:
        return self._json_call(state, car, text, "", "json")[0]

    def lead_in(self, system: str, user: str, max_tokens: int = 36, temperature: float | None = None, deadline_s: float | None = None) -> str:
        out = self._post(self._body(system, user, max_tokens, temperature), deadline_s or self.cfg.timeout_s)
        return re.sub(r"^\s*<think>.*?</think>\s*", "", out["choices"][0]["message"]["content"] or "", flags=re.S)

    def choose(self, system: str, user: str, options: list[str], max_tokens: int = 16, deadline_s: float | None = None) -> str | None:
        """One of `options` word for word (vLLM `structured_outputs.choice`): a few tokens, never chatter, never an action or car id not listed.
        A server that ignores the field answers in free text, which the caller parses as before. None if the server refuses it (HTTP 400):
        the caller then asks for free text, and this server is not asked this way again."""
        if not self.can_choose:
            return None
        body = self._body(system, user, max_tokens, 0.0)
        body["structured_outputs"] = {"choice": list(options)}
        try:
            out = self._post(body, deadline_s or self.cfg.timeout_s)
        except BackendError as e:
            if "HTTP 400" in str(e):
                self.can_choose = False
                return None
            raise
        return re.sub(r"^\s*<think>.*?</think>\s*", "", out["choices"][0]["message"]["content"] or "", flags=re.S).strip()

    def structured(self, system: str, user: str, schema: dict, max_tokens: int = 120, deadline_s: float | None = None) -> str | None:
        """One JSON object that follows `schema`, asked the portable way (`response_format` json_schema: vLLM, llama.cpp server and TGI honour it, and vLLM turns it into its
        own structured-output constraint, so the model cannot leave the schema). Greedy. None if the server refuses it (HTTP 400): this server is not asked this way again.
        NOT checked against a real vLLM server yet (none is available without a GPU): see docs/ARCH_REVIEW.md section 21."""
        if not self.can_structure:
            return None
        body = self._body(system, user, max_tokens, 0.0)
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "turn", "schema": schema, "strict": True}}
        try:
            out = self._post(body, deadline_s or self.cfg.timeout_s)
        except BackendError as e:
            if "HTTP 400" in str(e):
                self.can_structure = False
                return None
            raise
        return re.sub(r"^\s*<think>.*?</think>\s*", "", out["choices"][0]["message"]["content"] or "", flags=re.S).strip()

    def lead_in_stream(self, system: str, user: str, max_tokens: int = 36, temperature: float | None = None,
                       deadline_s: float | None = None, prefill: str = "") -> Iterator[str]:
        """Pieces of text as the server streams them. Stopping iteration closes the connection, which stops the generation on the server."""
        try:
            yield from self._stream(self._body(system, user, max_tokens, temperature, prefill=prefill, stream=True), deadline_s)
        except _NoPrefill:                                     # a server that cannot continue a begun reply: tell the model what was said
            told = f"{user}\n(You have already said: \"{prefill.strip()}\". Write only what comes next.)"
            yield from self._stream(self._body(system, told, max_tokens, temperature, stream=True), deadline_s)

    def _stream(self, body: dict, deadline_s: float | None) -> Iterator[str]:
        limit = deadline_s or self.cfg.timeout_s * 3
        started = time.perf_counter()
        try:
            with self.client.stream("POST", f"{self.base}/chat/completions", json=body, timeout=limit + 2) as r:
                if r.status_code == 400 and body.get("continue_final_message"):
                    raise _NoPrefill()
                if r.status_code >= 400:
                    raise BackendError(f"{self.name} HTTP {r.status_code}")
                for line in r.iter_lines():
                    if time.perf_counter() - started > limit:
                        return
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    delta = (json.loads(data).get("choices") or [{}])[0].get("delta", {}).get("content")
                    if delta:
                        yield delta
        except _NoPrefill:
            raise
        except BackendError:
            raise
        except Exception as e:
            if "Timeout" in type(e).__name__:
                raise BackendTimeout(f"{self.name} produced nothing within {limit:.0f}s") from e
            raise BackendError(f"{type(e).__name__}: {e}") from e

    def answer(self, question: str, context: str, max_sentences: int = 2) -> Iterator[str]:
        yield from self.lead_in_stream(prompts.OPEN_ANSWER_SYSTEM.format(n=max_sentences), f"CONTEXT: {context}\nCALLER: {question}",
                                       self.cfg.max_answer_tokens)


class _NoPrefill(Exception):
    pass

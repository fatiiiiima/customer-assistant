r"""NVIDIA Canary-Qwen-2.5B on the Mac's GPU (MLX), as a speech-recognition server with the OpenAI transcription API, for the worker's `voice.stt_url`.

  ~/mlxenv312/bin/python scripts/canary_server.py                    # http://127.0.0.1:8010/v1/audio/transcriptions
  VA__VOICE__STT_URL=http://127.0.0.1:8010/v1 VA__VOICE__STT_REMOTE_MODEL=canary <the worker's usual command>

Runs in the MLX environment (mlx, mlx-lm, transformers, librosa), not the project's. The weights are a community MLX port (Hugging Face
`Santosh-bold/speechlm-canary`: the FastConformer encoder with its projection, and the Qwen3-1.7B decoder with NVIDIA's fine-tuning merged in, both bf16),
with Qwen3-1.7B's own config and tokenizer. Its model code (model.py, modules.py, attention.py, subsampling.py) is used as published; its generate.py is not
(it reads the author's own folders and NVIDIA's original checkpoint): loading and the prompt are done here, as that script does them.

English only, and not a streaming model: each turn is recognised whole, as Whisper's is. The worker's hint ("prompt") is not used: Canary-Qwen's recognition
prompt is fixed ("Transcribe the following:"). docs/ARCH_REVIEW.md section 36.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
import wave
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

PORT_REPO = "Santosh-bold/speechlm-canary"
QWEN_REPO = "Qwen/Qwen3-1.7B"
SAMPLE_RATE = 16000
# The prompt Canary-Qwen was trained with, as token ids (the port's generate.py, from NeMo's QwenPromptFormatter):
# "<|im_start|>user\nTranscribe the following: " ... audio ... "<|im_end|>\n<|im_start|>assistant\n"
BEFORE = [151644, 872, 198, 3167, 3114, 279, 2701, 25, 220]
AFTER = [151645, 198, 151644, 77091, 198]


class Canary:
    def __init__(self) -> None:
        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from mlx_lm.models import qwen3
        from transformers import AutoTokenizer

        port = Path(snapshot_download(PORT_REPO, local_files_only=True,
                                      allow_patterns=["*.safetensors", "model.py", "modules.py", "attention.py", "subsampling.py"]))   # not its generate.py / scripts
        qwen = Path(snapshot_download(QWEN_REPO, local_files_only=True, allow_patterns=["config.json", "generation_config.json", "tokenizer*.json", "vocab.json", "merges.txt"]))
        sys.path.insert(0, str(port))
        from model import CanaryModel, FastConformerEncoder                    # the port's model code (read before use: MLX layers only)

        self.mx = mx
        encoder = FastConformerEncoder(feat_in=128, n_layers=32, d_model=1024, ff_expansion_factor=4, n_heads=8, conv_kernel_size=9, dropout=0.0)
        self.audio = CanaryModel(encoder=encoder, llm_dim=2048)
        self.audio.load_weights(str(port / "mlx_canary_encoder.safetensors"), strict=False)
        self.audio.eval()
        config = json.loads((qwen / "config.json").read_text())
        self.llm = qwen3.Model(qwen3.ModelArgs.from_dict(config))
        self.llm.load_weights(list(mx.load(str(port / "mlx_canary_llm.safetensors")).items()), strict=False)
        self.llm.eval()
        mx.eval(self.audio.parameters(), self.llm.parameters())
        self.tokenizer = AutoTokenizer.from_pretrained(str(qwen))
        self.stop = {self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|im_end|>")}
        self.lock = threading.Lock()                                           # one turn at a time on the GPU

    def features(self, audio: np.ndarray):
        """Log-mel features as NeMo makes them (the port's extract_mel): pre-emphasis, 512-point STFT every 10 ms, 128 Slaney mels, per-feature normalised."""
        import librosa

        y = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        stft = librosa.stft(y, n_fft=512, hop_length=160, win_length=400, window="hann", center=True, pad_mode="constant")
        mel = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=512, n_mels=128, fmin=0.0, fmax=8000.0, norm="slaney", htk=False) @ (np.abs(stft) ** 2)
        log_mel = np.log(np.maximum(mel, 1e-5))
        log_mel = (log_mel - log_mel.mean(axis=1, keepdims=True)) / (log_mel.std(axis=1, keepdims=True) + 1e-5)
        return self.mx.array(log_mel.T[None].astype(np.float32))

    def transcribe(self, audio: np.ndarray, max_tokens: int = 160) -> str:
        from mlx_lm.models.cache import make_prompt_cache

        mx = self.mx
        if len(audio) < SAMPLE_RATE // 10:
            return ""
        with self.lock:
            embeds, _ = self.audio.encode_audio(self.features(audio))
            table = self.llm.model.embed_tokens
            prompt = mx.concatenate([table(mx.array([BEFORE])), embeds.astype(table.weight.dtype), table(mx.array([AFTER]))], axis=1)
            cache = make_prompt_cache(self.llm)
            logits = self.llm(mx.zeros((1, prompt.shape[1]), dtype=mx.int32), cache=cache, input_embeddings=prompt)
            token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            out = []
            while token not in self.stop and len(out) < max_tokens:
                out.append(token)
                logits = self.llm(mx.array([[token]]), cache=cache)
                token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            return self.tokenizer.decode(out).strip()


def read_wav(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError("expects 16-bit mono WAV")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        rate = w.getframerate()
    if rate != SAMPLE_RATE:
        import librosa

        x = librosa.resample(x, orig_sr=rate, target_sr=SAMPLE_RATE)
    return x


def serve(model: Canary, host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):                                             # no transcripts or caller details in the log
            pass

        def _reply(self, code: int, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self._reply(200, {"ok": True, "model": "canary-qwen-2.5b"} if self.path.rstrip("/") in ("/health", "/v1/models") else {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/audio/transcriptions":
                return self._reply(404, {"error": "not found"})
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            msg = BytesParser(policy=email_policy).parsebytes(f"Content-Type: {self.headers.get('Content-Type')}\r\n\r\n".encode() + body)
            wav = next((p.get_payload(decode=True) for p in msg.iter_parts() if p.get_param("name", header="content-disposition") == "file"), None)
            if wav is None:
                return self._reply(400, {"error": "no file"})
            try:
                t0 = time.perf_counter()
                text = model.transcribe(read_wav(wav))
                self._reply(200, {"text": text, "ms": round((time.perf_counter() - t0) * 1000)})
            except Exception as e:
                self._reply(400, {"error": type(e).__name__})

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Canary-Qwen ready on http://{host}:{port}/v1", flush=True)
    httpd.serve_forever()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--wav", type=Path, help="recognise this file and exit (a check, no server)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    model = Canary()
    model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))               # warm up
    print(f"loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    if a.wav:
        t1 = time.perf_counter()
        print(model.transcribe(read_wav(a.wav.read_bytes())), f"({(time.perf_counter() - t1) * 1000:.0f} ms)")
        return 0
    serve(model, a.host, a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

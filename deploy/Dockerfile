# One image for the voice worker and the router (the model runs in vLLM's own image).
#   docker build -f deploy/Dockerfile -t agmc/voice:latest .
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY config ./config
COPY scripts ./scripts
# Speech models (Whisper, Piper voice, Silero, Smart Turn) are mounted at /app/models, not baked into the image.
RUN pip install --no-cache-dir uv && uv pip install --system --no-cache . \
    pipecat-ai openvino openvino-genai sherpa-onnx soxr numpy fastapi uvicorn websockets httpx pg8000 redis prometheus-client
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
# The tool decisions go to the same server as the replies (the CMD's --backend). Without this they would be tried on dialogue.decision_backend's default, the Intel-GPU model.
ENV VA__DIALOGUE__DECISION_BACKEND=gpu_server
EXPOSE 8765 8760
# Worker by default; the router runs the same image with: python -m dealer_core.gateway
CMD ["python", "-m", "dealer_core.server", "--agent", "--prefix", "--backend", "gpu_server", "--host", "0.0.0.0", "--port", "8765", "--max-calls", "16"]

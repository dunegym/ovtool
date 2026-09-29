# ovtool runtime image — inference/serve/webui/tts/embed/rerank, no model weights.
# Convert/quantization extras (torch, optimum-intel) are intentionally NOT
# included; pull pre-converted models with `ovtool download` (see README) or
# convert on the host and mount the directory.
FROM python:3.11-slim

LABEL org.opencontainers.image.title="ovtool" \
      org.opencontainers.image.description="OpenVINO GenAI CLI: LLM/VLM inference, TTS, embeddings & rerank, image generation, OpenAI-compatible server, web UI" \
      org.opencontainers.image.source="https://github.com/dunegym/ovtool" \
      org.opencontainers.image.licenses="Apache-2.0"

# libgomp1 is the only shared lib the OpenVINO CPU plugin needs on slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY ovtool ./ovtool
RUN pip install --no-cache-dir .

# models and the Hugging Face download cache live in volumes, never in the image
ENV OVTOOL_MODELS_PATH=/models \
    HF_HOME=/cache \
    PYTHONUNBUFFERED=1
VOLUME [/models, /cache]
EXPOSE 7860 8000

ENTRYPOINT ["ovtool"]
CMD ["--help"]

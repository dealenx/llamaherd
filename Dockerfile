FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY llamaherd/ llamaherd/
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.11-slim AS runtime

RUN groupadd --system llamaherd \
    && useradd --system --gid llamaherd --create-home llamaherd

WORKDIR /app
ENV LLAMAHERD_HOST=0.0.0.0
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

COPY --chown=llamaherd:llamaherd config.example.yaml config.yaml
COPY --chown=llamaherd:llamaherd openrouter_pricing.yaml openrouter_pricing.yaml

USER llamaherd
EXPOSE 8399

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8399/healthz', timeout=3)"]

CMD ["llamaherd", "--config", "/app/config.yaml", "serve"]

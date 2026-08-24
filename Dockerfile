FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY llamaherd/ llamaherd/
RUN pip wheel --no-cache-dir --wheel-dir /wheels '.[scraping]'

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

# Create directories that docker-compose bind-mounts files into.
# Without these, the runtime image (built from a wheel, not a local pip install)
# has no /app/llamaherd/ or /app/data/ and the proxy.db / usage.db bind mounts
# fail with "unable to open database file". The pre-hardening Dockerfile used
# `pip install .` which incidentally created /app/llamaherd/ via the package
# metadata; the multi-stage wheel build does not.
RUN mkdir -p /app/llamaherd /app/data \
    && chown -R llamaherd:llamaherd /app/llamaherd /app/data

USER llamaherd
EXPOSE 8399

# Explicit config path. This is set as an env var (not via the --config CLI
# flag) so that proxy.py sees it at module import time, before CONFIG_PATH is
# resolved. The file is provided either by the COPY below or by a bind-mount.
ENV LLAMAHERD_CONFIG=/app/config.yaml

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8399/healthz', timeout=3)"]

CMD ["llamaherd", "serve"]

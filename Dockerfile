FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY llamaherd/ llamaherd/
COPY config.example.yaml config.yaml
COPY .env.example .env.example

RUN pip install --no-cache-dir .

# Point LlamaHerd at /app/config.yaml (the example config copied above).
# For production: mount your real config.yaml at /app/config.yaml or set
# LLAMAHERD_CONFIG to a different path.
# Override via env vars: LLAMAHERD_ADMIN_TOKEN, LLAMAHERD_DB,
# LLAMAHERD_DB_AUTH_TOKEN, LLAMAHERD_DB_AUTH_USER, etc.
# Mount a .env file at /app/.env or set env vars in Dokploy/Docker for secrets.
ENV LLAMAHERD_CONFIG=/app/config.yaml
ENV LLAMAHERD_HOST=0.0.0.0
ENV LLAMAHERD_PORT=8399

EXPOSE 8399

CMD ["llamaherd", "serve"]
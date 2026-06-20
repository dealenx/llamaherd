FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY llamaherd/ llamaherd/
COPY config.example.yaml config.yaml
COPY .env.example .env.example

RUN pip install --no-cache-dir .

# Default to 0.0.0.0 so the proxy is reachable from outside the container.
# Override via env vars: LLAMAHERD_HOST, LLAMAHERD_PORT, LLAMAHERD_ADMIN_TOKEN,
# LLAMAHERD_DB, LLAMAHERD_DB_AUTH_TOKEN, LLAMAHERD_DB_AUTH_USER, etc.
# Mount a .env file at /app/.env or set env vars in Dokploy/Docker for secrets.
ENV LLAMAHERD_HOST=0.0.0.0
ENV LLAMAHERD_PORT=8399

EXPOSE 8399

CMD ["llamaherd", "serve"]
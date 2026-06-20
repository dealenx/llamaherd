FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY llamaherd/ llamaherd/
COPY .env.example .env.example

RUN pip install --no-cache-dir .

# config.yaml is OPTIONAL — LlamaHerd loads keys/clients from the database.
# All configuration can be done via environment variables and the dashboard.
# To use a custom config.yaml, mount it at /app/config.yaml or set LLAMAHERD_CONFIG.
ENV LLAMAHERD_HOST=0.0.0.0
ENV LLAMAHERD_PORT=8399

EXPOSE 8399

CMD ["llamaherd", "serve"]
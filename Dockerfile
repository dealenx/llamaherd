FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY llamaherd/ llamaherd/
COPY config.example.yaml config.yaml

RUN pip install --no-cache-dir .

EXPOSE 8399

CMD ["llamaherd"]
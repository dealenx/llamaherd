# LlamaHerd project context

LlamaHerd is a FastAPI Ollama Cloud multi-key proxy in `llamaherd/proxy.py`, with CLI in `llamaherd/cli.py` and README docs.

## Important constraints
- Do not print or expose admin tokens, client tokens, Ollama API keys, cookies, or config secrets.
- Do not overwrite production `proxy.db` or `config.yaml` with test fixtures.
- Production runs as user systemd service `ollama-cloud-proxy.service` on `0.0.0.0:8399`.
- Staging/sidecar runs as user systemd service `llamaherd-sidecar.service` on `127.0.0.1:8499`.
- Prefer tests that use temp files/databases and mocked upstreams; avoid real Ollama Cloud calls in automated tests.

## Current feature work
- Dashboard uses SSE `/admin/events` for live calls/status/models.
- Dashboard has date period filtering.
- Model registry uses native `/api/tags` and `/api/show` metadata; `/v1/models`, `/api/tags`, and `/admin/models` expose model metadata.
- OpenClaw should use native Ollama API with no `/v1`; Hermes/OpenAI clients use `/v1`.

## Verification commands
- Syntax: `python3 -m compileall llamaherd`
- Smoke production: `python3 ~/.hermes/skills/devops/ollama-cloud-proxy/scripts/production_smoke_test.py`
- Unit/e2e tests: add pytest tests under `tests/` and run `python3 -m pytest`.

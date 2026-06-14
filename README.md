# 🦙 LlamaHerd

**One endpoint. Many llamas. Smarter routing.**

LlamaHerd is an agent-first proxy for Ollama Cloud: multi-key routing, usage tracking, client API keys, rate limits, and a live dashboard. It pools your subscriptions behind one OpenAI-compatible and native Ollama-compatible endpoint.

```text
clients --> [ LlamaHerd ] --> { Sub 1 | Sub 2 | Sub N }
              route by slots, session %, weekly %, health
```

```text
    __    __                      __  __              __
   / /   / /___ _____ ___  ____ _/ / / /__  _________/ /
  / /   / / __ `/ __ `__ \/ __ `/ /_/ / _ \/ ___/ __  / 
 / /___/ / /_/ / / / / / / /_/ / __  /  __/ /  / /_/ /  
/_____/_/\__,_/_/ /_/ /_/\__,_/_/ /_/\___/_/   \__,_/   
```

## ✨ Features

- **Sticky session routing** — keeps the same conversation on the same upstream subscription to maximize KV-cache reuse. Falls back to hashing the start of the conversation context when clients don't send a session ID.
- **Multi-key load balancing** — routes across N Ollama Cloud keys, preferring freshest billing cycle and least usage
- **Auto model discovery** — polls `/v1/models` from each key, merges into one list
- **Live dashboard** — SSE-powered real-time updates, time-period filtering, per-model usage, session/weekly progress tracking
- **Usage attribution** — per-client API keys track which service made each request
- **Dynamic key management** — add/remove subscriptions and client keys via API, no restart needed
- **Cookie-based usage scraping** — tracks session & weekly usage % from ollama.com/settings
- **429 auto-retry** — upstream rate limits trigger automatic retry on the next best key
- **Overflow queuing** — requests queue (up to 60s) instead of failing fast
- **OpenAI & native Ollama protocol** — both `/v1/*` and `/api/*` routes supported
- **Context length metadata** — injects correct context windows into `/v1/models` so clients don't fall back to 128K defaults

## Sticky sessions and cache affinity

LlamaHerd implements sticky session routing with the goal of maximizing the chance that repeated turns of the same conversation hit a warm KV/context cache on the upstream Ollama Cloud subscription. This is particularly valuable for long-context and 1M-context models, where re-transmitting the full prompt to a different subscriber on every turn would be expensive.

How sessions are identified (in order):

1. `X-LlamaHerd-Session` request header.
2. `llamaherd-session` cookie returned by LlamaHerd.
3. `X-Conversation-ID` request header.
4. Hash of the first system + user messages from the request body.

If a client (e.g. OpenClaw using the OpenAI JS client) does not send an explicit session identifier, LlamaHerd falls back to option 4. The TTL is configurable via `sticky_ttl_seconds` (default 3600s).

On upstream 429/402 or other errors, the sticky mapping for that session is cleared so the next turn can rebalance to a healthier subscription.

## 🚀 Quick Start

```bash
# Install
pip install llamaherd

# Or from source
git clone https://github.com/bennybuoy/llamaherd.git
cd llamaherd
pip install -e .

# Configure
cp config.example.yaml config.yaml
# Edit config.yaml with your Ollama Cloud API keys

# Run
llamaherd --config config.yaml

# Or with env var overrides
LLAMAHERD_ADMIN_TOKEN=my-secret python -m llamaherd.proxy
```

Then point your OpenAI-compatible client at `http://127.0.0.1:8399/v1`.

## 📊 Dashboard

Open `http://127.0.0.1:8399/dashboard?token=YOUR_ADMIN_TOKEN` in a browser.

The dashboard features:
- **Overview tab** — key status cards with session/weekly usage progress bars, totals, per-client/model/daily breakdowns, live call feed
- **Models tab** — all discovered models with context lengths, key availability, and 7-day usage stats. Search and sort.
- **Subscriptions tab** — add/remove Ollama Cloud keys, edit cookies for usage tracking, see plan and billing info
- **Time period filtering** — Today, Yesterday, Last 7 days, This Week, This Month, Last Month, or Custom dates
- **SSE live updates** — new calls and status changes stream in real-time, no polling

## 🔑 Client Keys

LlamaHerd uses internal client keys for usage attribution (not real Ollama keys). Manage them via API:

```bash
# Create a client
curl -X POST http://127.0.0.1:8399/admin/clients \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "my-app", "label": "My Application"}'

# List clients
curl http://127.0.0.1:8399/admin/clients \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

Then point your client at `http://127.0.0.1:8399/v1` with `Authorization: Bearer ocp-my-app-...`.

## 🍪 Usage Tracking Cookies

To see session and weekly usage percentages, you need browser cookies from each Ollama account:

1. Log into ollama.com/settings in your browser
2. Open DevTools → Application → Cookies → ollama.com
3. Copy `__Secure-session` (required, per-account), `aid`, `cf_clearance`, and `__stripe_mid`
4. Add them to `config.yaml` under each key's `cookies` section, or via the Subscriptions tab in the dashboard

## 🐳 Docker

```bash
docker build -t llamaherd .
docker run -p 8399:8399 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/usage.db:/app/usage.db \
  llamaherd
```

## 🤖 CLI (Agent-Friendly)

LlamaHerd has a full CLI designed for agents and scripts. All commands output **JSON by default** (pipe to `jq` for filtering). Use `--format table` for human-readable output.

```bash
# List clients with rate limits
llamaherd clients list -f table

# Create a client with rate limits
llamaherd clients create my-app --label "My App" --daily-token-limit 100000 --rpm-limit 30

# Update limits (null = unlimited)
llamaherd clients update my-app --daily-request-limit 500

# Clear all limits (set to unlimited)
llamaherd clients update my-app --clear-limits

# Regenerate a compromised token
llamaherd clients regenerate-token my-app

# Delete a client
llamaherd clients delete my-app

# Show proxy status
llamaherd status -f table

# List upstream keys
llamaherd keys -f table

# List discovered models
llamaherd models -f table

# Usage stats
llamaherd usage --days 7

# Connection options (defaults to 127.0.0.1:8399)
llamaherd --host 10.0.0.1 --port 8399 -t YOUR_ADMIN_TOKEN clients list
```

### Rate Limiting

Each client key can have optional rate limits:

| Limit | Field | Unit | Example |
|-------|-------|------|---------|
| Daily tokens | `daily_token_limit` | tokens (in+out) per day | `100000` |
| Daily requests | `daily_request_limit` | requests per day | `500` |
| RPM | `rpm_limit` | requests per minute | `30` |

When a limit is exceeded, the proxy returns `429` with a JSON body:
```json
{
  "error": "rate_limit_exceeded",
  "detail": "Daily token limit of 100000 exceeded for client 'my-app'",
  "limit_type": "daily_tokens",
  "limit": 100000,
  "used": 104312,
  "reset_at": 1719792000
}
```

Set limits via API or CLI. `null` means unlimited (default).

## 🔌 Connecting Agents

### Hermes Agent

Hermes currently uses LlamaHerd through its OpenAI-compatible route. Include `/v1` in the base URL:

```yaml
model:
  default: deepseek-v3.2
  provider: custom
  base_url: http://127.0.0.1:8399/v1
  api_key: ocp-hermes-gateway-XXXXXXXX
```

Or via CLI:
```bash
hermes config set model.provider custom
hermes config set model.default deepseek-v3.2
hermes config set model.base_url http://127.0.0.1:8399/v1
hermes config set model.api_key ocp-hermes-gateway-XXXXXXXX
```

Create the client key first:
```bash
llamaherd clients create hermes-gateway --label "Hermes Gateway" --rpm-limit 60
```

### OpenClaw

OpenClaw should use LlamaHerd's **native Ollama API**, not the OpenAI-compatible `/v1` route. Do **not** add `/v1`; using `/v1` can break tool calling.

```yaml
# In your OpenClaw config
baseUrl: http://127.0.0.1:8399
api: ollama
apiKey: ocp-openclaw-gateway-XXXXXXXX
```

Create the client key first:
```bash
llamaherd clients create openclaw-gateway --label "OpenClaw Gateway" --rpm-limit 60
```

### Copy-paste agent setup prompt

If you are asking an agent to configure itself, give it this prompt and the client token privately:

```text
Configure this agent to use LlamaHerd, the local Ollama Cloud proxy.

Endpoint rules:
- If this agent is OpenClaw, use native Ollama protocol: baseUrl http://127.0.0.1:8399, api ollama, no /v1 suffix.
- If this agent is Hermes Agent or another OpenAI-compatible client, use base_url http://127.0.0.1:8399/v1.
- Use the provided client API key as the Authorization bearer/api_key.
- Pick a model from GET /api/tags for native Ollama clients or GET /v1/models for OpenAI-compatible clients.
- Verify with a small chat request and do not print the API key.
```

### Any OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8399/v1",
    api_key="ocp-my-app-XXXXXXXX",
)

response = client.chat.completions.create(
    model="deepseek-v3.2",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## 📡 API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/admin/status` | GET | Key health, slots, usage %, billing data |
| `/admin/events?token=X` | GET (SSE) | Real-time event stream |
| `/admin/models` | GET | Discovered models with context length, capabilities, size, family, parameters |
| `/admin/totals` | GET | All-time or date-filtered totals |
| `/admin/keys` | GET/POST/PUT/DELETE | Manage subscription keys |
| `/admin/clients` | GET/POST/PUT/DELETE | Manage client attribution keys |
| `/admin/usage/*` | GET | Usage breakdowns (by-client, by-model, daily) |
| `/admin/recent-calls` | GET | Individual call log |
| `/admin/refresh` | POST | Trigger model discovery |
| `/admin/poll-subscriptions` | POST | Poll /api/me for all keys |
| `/admin/scrape-usage` | POST | Scrape usage from ollama.com/settings |
| `/v1/*` | ANY | OpenAI-compatible proxy routes |
| `/api/*` | ANY | Native Ollama protocol routes |

All `/admin/*` endpoints require the admin token, either in the Authorization bearer header or as a `?token=...` query param.

### Date Range Filtering

Usage endpoints support `start_date` and `end_date` params (ISO date: `2025-01-15`):

```
GET /admin/usage/by-client?start_date=2025-05-01&end_date=2025-05-03
GET /admin/totals?start_date=2025-05-01&end_date=2025-05-03
```

### SSE Events

Connect to `/admin/events?token=YOUR_TOKEN` for real-time updates:

| Event | When | Data |
|---|---|---|
| `status` | After subscription poll or usage scrape | Key health array |
| `call` | Every proxied request completes | Call record (model, tokens, latency) |
| `models` | Model registry refreshes | Count, new models list |
| `heartbeat` | Every 15 seconds | Keepalive |

## ⚙️ Configuration

See `config.example.yaml` for all options. Key settings:

- **keys** — List of Ollama Cloud subscriptions with token, max_concurrent, cycle_day, label, and optional cookies
- **clients** — Internal attribution keys (can also be managed via API)
- **admin_token** — Secret for dashboard and admin API access
- **usage_db** — SQLite database path for usage tracking
- **health_check_interval** — Seconds between model discovery (default 300)
- **usage_scrape_interval** — Seconds between usage scrapes (default 1800)

## 🏗️ Architecture

```
┌──────────┐     ┌──────────────┐     ┌─────────────┐
│  Hermes  │────▶│  LlamaHerd   │────▶│ ollama.com  │
│  Cron    │     │  :8399       │     │  /v1/*       │
│  Eval    │────▶│              │────▶│  /api/*      │
└──────────┘     │  ┌────────┐  │     └─────────────┘
                 │  │Key Mgr │  │     ┌─────────────┐
                 │  │(rotate)│  │────▶│ Key 1 (pro) │
                 │  └────────┘  │     │ Key 2 (pro) │
                 │  ┌────────┐  │     │ Key 3 (pro) │
                 │  │UsageDB │  │     └─────────────┘
                 │  │(SQLite)│  │
                 │  └────────┘  │
                 │  ┌────────┐  │
                 │  │Dashboard│  │
                 │  │(SSE)    │  │
                 │  └────────┘  │
                 └──────────────┘
```

## 📝 License

MIT — see [LICENSE](LICENSE).
# 🦙 LlamaHerd

**Herd your Ollama Cloud subscriptions.**

LlamaHerd is an OpenAI-compatible proxy that routes requests across multiple Ollama Cloud API keys — load balancing, usage tracking, and a live dashboard included. It pools your $20/mo subscriptions into one high-concurrency endpoint.

```
Client (Hermes/Cron/Eval) ──[client key]──▶ LlamaHerd :8399 ──[real key]──▶ ollama.com/v1
```

## ✨ Features

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

## 🚀 Quick Start

```bash
# Install
pip install llamaherd

# Or from source
git clone https://github.com/llamaherd/llamaherd.git
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

## 📡 API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/admin/status` | GET | Key health, slots, usage %, billing data |
| `/admin/events?token=X` | GET (SSE) | Real-time event stream |
| `/admin/models` | GET | Discovered models with context length |
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

All `/admin/*` endpoints require authentication via `Authorization: Bearer TOKEN` header or `?token=TOKEN` query param.

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
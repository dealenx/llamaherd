# LlamaHerd Branding Pack

Working direction for the project brand, CLI/gateway copy, ASCII art, and image-generation prompts.

## Core Positioning

**LlamaHerd** is a friendly, agent-first control plane for routing Ollama Cloud usage across multiple subscriptions.

Short tagline options:

1. **Pool the herd. Route the load.**
2. **Multi-key Ollama Cloud routing for agents.**
3. **One endpoint. Many llamas. Smarter routing.**
4. **Keep your agents fed without melting one subscription.**
5. **A tiny traffic shepherd for Ollama Cloud.**

Preferred default:

> **One endpoint. Many llamas. Smarter routing.**

Why: it explains the product quickly, works in GitHub, CLI, README hero, and dashboard header.

## Voice

LlamaHerd should feel:

- Competent, not corporate
- Slightly playful, not goofy
- Agent-first, but human-readable
- Infrastructure-grade, but small and hackable

Good words:

- herd, route, graze, pasture, shepherd, stable, trail, feed, pool, balance, slots, budget

Avoid overdoing llama puns in errors. One tasteful mascot is good; errors should stay clear and structured.

## Visual Identity

### Concept A — The Traffic Shepherd

A calm llama wearing a tiny sysadmin headset, standing beside a network switch / router, with multiple glowing request paths flowing through it.

- Mood: reliable, friendly infra
- Colors: dark slate, warm cream, cyan/green request lines
- Best for: GitHub social image, README hero, dashboard empty states

### Concept B — The Herd Router

A stylized herd of llamas moving through a geometric load-balancing gateway, each llama representing an upstream subscription/key.

- Mood: technical, distributed, scalable
- Colors: dark terminal background, amber llama shapes, blue/green routes
- Best for: architecture diagrams, project docs, landing page

### Concept C — The Terminal Mascot

A minimalist pixel/ASCII llama face embedded in a terminal prompt, with request counters and model names scrolling behind it.

- Mood: CLI-native, developer-first
- Colors: monochrome or green-on-black
- Best for: CLI splash, README, PyPI logo

Recommended combined direction:

> **Use Concept A for the logo/hero, Concept C for CLI identity, and Concept B for diagrams.**

## Color Palette

Dark terminal-native palette:

```text
Background        #0D1117  GitHub dark
Panel             #161B22
Border            #30363D
Llama Cream       #F2D6A2
Warm Wool         #C9964A
Pasture Green     #3FB950
Route Cyan        #58A6FF
Alert Amber       #D29922
Error Red         #F85149
Text Primary      #E6EDF3
Text Muted        #8B949E
```

Dashboard should keep GitHub-dark compatibility but add Llama Cream / Route Cyan accents.

## ASCII Art Candidates

### CLI Banner — Slant

```text
    __    __                      __  __              __
   / /   / /___ _____ ___  ____ _/ / / /__  _________/ /
  / /   / / __ `/ __ `__ \/ __ `/ /_/ / _ \/ ___/ __  / 
 / /___/ / /_/ / / / / / / /_/ / __  /  __/ /  / /_/ /  
/_____/_/\__,_/_/ /_/ /_/\__,_/_/ /_/\___/_/   \__,_/   
```

Good for `llamaherd serve` startup and README.

### CLI Banner — Compact

```text
  _    _                 _  _            _ 
 | |  | |__ _ _ __  __ _| || |___ _ _ __| |
 | |__| / _` | '  \/ _` | __ / -_) '_/ _` |
 |____|_\__,_|_|_|_\__,_|_||_\___|_| \__,_|
```

Good for narrow terminals and `--help` epilogues.

### Tiny Mascot

```text
       __
   .-"`  `"-.
  /  .--.    \
 |  /    \    |   LlamaHerd
 |  \__/ /\   |   one endpoint, many llamas
  \      /  _/
   `-.___.-`   
      || ||
```

### Request Herd Mini

```text
clients ──▶ [ LlamaHerd ] ──▶ 🦙 Sub 1
           load / usage      ├─▶ 🦙 Sub 2
           aware routing     └─▶ 🦙 Sub N
```

Plain ASCII fallback:

```text
clients --> [ LlamaHerd ] --> { Sub 1 | Sub 2 | Sub N }
              route by slots, session %, weekly %, health
```

## CLI / Gateway Copy

### `llamaherd --help` description

```text
LlamaHerd — one endpoint, many llamas, smarter Ollama Cloud routing.
```

### Startup banner

```text
LlamaHerd is grazing on 0.0.0.0:8399
Dashboard: http://127.0.0.1:8399/dashboard
Upstream keys: 2 | clients: 6 | models: 39
```

Alternative less cute:

```text
LlamaHerd listening on 0.0.0.0:8399
Routing Ollama Cloud traffic across 2 upstream keys for 6 clients.
```

Recommended: use the less-cute version in production logs; optional cute banner only in CLI interactive output.

### Dashboard hero

```text
LlamaHerd
One endpoint. Many llamas. Smarter routing.
Live Ollama Cloud load balancing, usage tracking, and client limits.
```

### GitHub description

```text
Agent-first Ollama Cloud proxy: multi-key routing, usage tracking, live dashboard, client API keys, and rate limits.
```

### PyPI summary

```text
Agent-first multi-key proxy and dashboard for Ollama Cloud.
```

## Image Generation Prompts

### Logo / GitHub Avatar Prompt

> Create a clean vector-style logo for an open-source developer tool called “LlamaHerd”. The logo should feature a friendly but competent llama mascot acting as a network traffic shepherd. The llama has subtle sysadmin vibes — maybe a tiny headset or terminal cursor badge — but keep it simple and iconic. Include visual hints of load balancing: three small glowing route lines or nodes around the llama. Style: modern open-source infrastructure branding, flat vector, dark GitHub-compatible background, warm cream llama, cyan and green routing accents, crisp silhouette, readable at small sizes. No text in the image. Square 1:1 composition.

### README Hero Prompt

> Design a wide README hero image for “LlamaHerd”, an agent-first Ollama Cloud proxy that routes requests across multiple subscriptions. Show a friendly llama herd passing through a central glowing gateway/router, with request lines entering from AI agents on the left and distributing to multiple cloud keys on the right. The mood should be reliable infrastructure with a playful llama theme. Style: polished vector / semi-flat illustration, dark terminal-inspired background (#0D1117), warm cream and amber llamas, cyan/green network paths, subtle dashboard panels in the background, clean open-source developer-tool aesthetic. Include title text “LlamaHerd” and subtitle “One endpoint. Many llamas. Smarter routing.” Landscape 16:9.

### CLI Mascot Prompt

> Create a pixel-art / terminal-style mascot for “LlamaHerd”: a small llama head made of blocky shapes, wearing a tiny router badge or terminal prompt symbol. It should feel like an old-school command-line companion, monochrome-friendly but with optional cream/cyan accents. Transparent background, centered, simple silhouette, suitable for converting to ASCII art. No text.

### Dashboard Header Prompt

> Create a sleek dashboard header illustration for “LlamaHerd”, an Ollama Cloud load-balancing proxy. Visualize live request streams as glowing lines moving through a central llama-shaped gateway into multiple subscription nodes. Dark GitHub-style UI background, subtle cards and metrics, cyan/green route lines, warm llama accent, modern observability dashboard feel, not childish, not cartoonish, open-source infrastructure aesthetic. Wide banner, 3:1 aspect ratio, no text.

### Sticker / Fun Prompt

> Create a playful open-source sticker design for “LlamaHerd”: a confident llama standing in front of a small server rack, herding tiny glowing API requests like sheep. Use bold vector shapes, thick outline, limited colors, transparent background, high contrast, sticker-friendly. Include optional small text “route the herd” if legible.

## Implementation Targets

### Git / repo

- Add `BRANDING.md`
- Update README hero copy and tagline
- Update GitHub repo description/topics
- Add `assets/` directory:
  - `assets/logo.svg` or `assets/logo.png`
  - `assets/social-card.png`
  - `assets/ascii-banner.txt`

### CLI

- Add `llamaherd --version` if missing
- Add `llamaherd banner` command or `--banner` flag for humans
- Keep JSON default untouched for agents
- Do not print banners in JSON commands by default
- In `serve`, print concise startup identity to logs

### Gateway / dashboard

- Page title: `LlamaHerd — Ollama Cloud Router`
- Header tagline: `One endpoint. Many llamas. Smarter routing.`
- Add subtle llama/route mark to header
- Keep admin/API behavior unchanged

## Non-goals

- No noisy puns in machine-readable errors
- No banner by default in commands expected to return JSON
- No secrets in screenshots or generated images

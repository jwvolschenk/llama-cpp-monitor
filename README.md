# llama-cpp-monitor

Lightweight monitoring dashboard for the local llama.cpp inference server.
No external Python dependencies — stdlib only. Chart.js loaded from CDN.

## Quick Start

```bash
cp .env.example .env
# Edit .env — set LLAMA_API_KEY if your llama-server requires auth

python3 monitor.py
# Dashboard at http://localhost:8100 (or MONITOR_PORT from .env)
```

## As a systemd user service

```bash
# Copy service file
mkdir -p ~/.config/systemd/user
cp llama-monitor.service ~/.config/systemd/user/

# Enable and start
systemctl --user daemon-reload
systemctl --user enable --now llama-monitor

# Check status
systemctl --user status llama-monitor
```

## Environment Variables

| Variable        | Default                | Description                       |
|-----------------|------------------------|-----------------------------------|
| `LLAMA_HOST`    | `localhost`            | llama.cpp server host             |
| `LLAMA_PORT`    | `8080`                 | llama.cpp server port             |
| `LLAMA_API_KEY` | (none)                 | API key for metrics endpoint (set in `.env`) |
| `MONITOR_PORT`  | `8100`                 | Port for this dashboard           |
| `GPU_INDEX`     | `1`                    | nvidia-smi GPU index              |
| `POLL_INTERVAL` | `1`                    | Seconds between metric collection |
| `HISTORY_MAX`   | `300`                  | Chart data points (5min @ 1s)     |
| `HISTORY_HOUR_MAX` | `3600`             | Hourly avg window (1h @ 1s)       |

## What It Monitors

- **Health**: llama.cpp /health endpoint (ok / degraded / error)
- **Throughput**: prefill tokens/s and decode tokens/s (instant + 1h rolling average)
- **Token counters**: total prompt tokens, total generated tokens
- **Requests**: active and deferred request counts
- **GPU (nvidia-smi)**: temperature, VRAM usage, utilization %, power draw, clocks
- **Server info**: model alias, context size, slot count, batch config

## API Endpoints

| Path           | Description                           |
|----------------|---------------------------------------|
| `GET /`        | Dashboard (HTML)                      |
| `GET /api/status`   | Current metrics snapshot (JSON)  |
| `GET /api/history`  | Time-series data for charts (JSON) |
| `GET /api/props`    | Server properties (JSON)          |

## Architecture

```
 monitor.py (MONITOR_PORT, default 8100)
   ├── Background thread (every POLL_INTERVAL s, default 1s)
   │     ├── GET localhost:8080/health
   │     ├── GET localhost:8080/metrics  (Prometheus format)
   │     └── nvidia-smi --id=1
   ├── /api/status   → latest snapshot
   ├── /api/history  → deque of last 360 snapshots
   └── /             → dashboard.html (Chart.js)
```

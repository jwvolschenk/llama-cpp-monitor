# llama-cpp-monitor

Lightweight monitoring dashboard for a local **llama.cpp** inference server.
No external Python dependencies — stdlib only. Chart.js loaded from CDN.

The monitor polls llama.cpp health and Prometheus metrics, tracks GPU stats via `nvidia-smi`, and serves a live dashboard with throughput charts and rolling averages.

## Prerequisites

- **Python 3.10+** (stdlib only — no `pip install`)
- A running **llama-server** with the `--metrics` flag enabled (Prometheus endpoint at `/metrics`)
- **`nvidia-smi`** on the host if you want GPU temperature, VRAM, and utilization stats
- Optional: **`LLAMA_API_KEY`** if your llama-server is started with `--api-key`

## Configuration

Copy the example env file and set your values:

```bash
cp .env.example .env
```

Edit `.env` — at minimum set `LLAMA_API_KEY` if your llama-server requires authentication. All other variables have sensible defaults.

| Variable           | Default     | Description                                      |
|--------------------|-------------|--------------------------------------------------|
| `LLAMA_HOST`       | `localhost` | llama.cpp server host                            |
| `LLAMA_PORT`       | `8080`      | llama.cpp server port                            |
| `LLAMA_API_KEY`    | (none)      | Bearer token for `/health`, `/metrics`, `/props` |
| `MONITOR_PORT`     | `8100`      | Port for this dashboard                          |
| `GPU_INDEX`        | `1`         | `nvidia-smi` GPU index                           |
| `POLL_INTERVAL`    | `1`         | Seconds between metric collection                |
| `HISTORY_MAX`      | `300`       | Chart data points (5 min @ 1s)                   |
| `HISTORY_HOUR_MAX` | `3600`      | Hourly avg window (1h @ 1s)                      |
| `HISTORY_24H_MAX`  | `86400`     | 24h generated-token window (24h @ 1s)            |

`monitor.py` loads `.env` from the project directory on startup (stdlib dotenv loader — no `python-dotenv` package).

## Running manually

From the project directory:

```bash
python3 monitor.py
```

Dashboard: **http://localhost:8100** (or whatever `MONITOR_PORT` is set to).

Override env vars inline:

```bash
LLAMA_HOST=localhost LLAMA_PORT=8080 MONITOR_PORT=8100 GPU_INDEX=1 python3 monitor.py
```

The HTTP server binds to `0.0.0.0`, so the dashboard is reachable from other machines on the LAN.

## Running as a systemd user service (recommended)

This project ships `llama-monitor.service`, a **systemd user unit** that starts the monitor automatically and restarts it on failure.

### How it fits together

```
llama-server.service          llama-monitor.service
  (inference server)    →       (this dashboard)
  port 8080                     port 8100
  --metrics enabled             polls /health, /metrics, nvidia-smi
```

The monitor unit declares `After=llama-server.service` so systemd starts the dashboard after the inference server. The monitor does not start llama.cpp itself — only watches it.

On this machine, llama.cpp is managed separately as `llama-server.service` (user unit), which runs `start-server.sh` from the llama-tq install with `--metrics` and `--api-key`.

### First-time setup

1. **Edit paths in `llama-monitor.service`** — update `ExecStart`, `WorkingDirectory`, and `EnvironmentFile` to match where you cloned this repo:

   ```ini
   ExecStart=/usr/bin/python3 /path/to/llama-cpp-monitor/monitor.py
   WorkingDirectory=/path/to/llama-cpp-monitor
   EnvironmentFile=-/path/to/llama-cpp-monitor/.env
   ```

   The leading `-` on `EnvironmentFile` means systemd will not fail if `.env` is missing.

2. **Create `.env`** with your API key (see [Configuration](#configuration) above).

3. **Install the unit file:**

   ```bash
   mkdir -p ~/.config/systemd/user
   cp llama-monitor.service ~/.config/systemd/user/
   systemctl --user daemon-reload
   ```

4. **Enable and start:**

   ```bash
   systemctl --user enable --now llama-monitor
   ```

5. **Verify it is running:**

   ```bash
   systemctl --user status llama-monitor
   ```

   Open **http://localhost:8100** in a browser.

### Start at boot (without logging in)

User services only run while a login session is active unless **linger** is enabled:

```bash
sudo loginctl enable-linger "$USER"
```

With linger enabled, `systemctl --user enable llama-monitor` keeps the dashboard running across reboots.

### Service management

| Action              | Command                                      |
|---------------------|----------------------------------------------|
| Start               | `systemctl --user start llama-monitor`       |
| Stop                | `systemctl --user stop llama-monitor`        |
| Restart             | `systemctl --user restart llama-monitor`     |
| Status + recent log | `systemctl --user status llama-monitor`      |
| Follow logs         | `journalctl --user -u llama-monitor -f`      |
| Disable on boot     | `systemctl --user disable llama-monitor`     |

After editing the unit file in `~/.config/systemd/user/`, run `systemctl --user daemon-reload` before restarting.

### Unit file reference

The repo copy (`llama-monitor.service`) contains:

```ini
[Unit]
Description=Llama.cpp Server Monitor Dashboard
After=llama-server.service

[Service]
Type=simple
ExecStart=/usr/bin/python3 /path/to/llama-cpp-monitor/monitor.py
WorkingDirectory=/path/to/llama-cpp-monitor
EnvironmentFile=-/path/to/llama-cpp-monitor/.env
Environment=LLAMA_HOST=localhost
Environment=LLAMA_PORT=8080
Environment=MONITOR_PORT=8100
Environment=GPU_INDEX=1
Environment=HISTORY_HOUR_MAX=3600
Environment=HISTORY_24H_MAX=86400
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

`Environment=` lines override defaults; values in `.env` are loaded first and are not overwritten unless also set in the unit file.

## What it monitors

- **Health**: llama.cpp `/health` endpoint (ok / degraded / error)
- **Throughput**: prefill tokens/s and decode tokens/s (instant + 1h rolling average)
- **Token counters**: total prompt tokens, total generated tokens, rolling 24h generated
- **Requests**: active and deferred request counts; completed requests since monitor start
- **GPU (`nvidia-smi`)**: temperature, VRAM usage, utilization %, power draw, clocks
- **Server info**: model alias, context size, slot count, batch config (from `/props`, cached)

Throughput is **delta-based**: llama.cpp exposes cumulative counters; the monitor computes per-interval deltas so charts spike under load and flatline when idle.

## API endpoints

| Path              | Description                         |
|-------------------|-------------------------------------|
| `GET /`           | Dashboard (HTML)                    |
| `GET /api/status` | Current metrics snapshot (JSON)     |
| `GET /api/history`| Time-series data for charts (JSON)  |
| `GET /api/props`  | Server properties (JSON)            |

## Architecture

```
monitor.py (MONITOR_PORT, default 8100)
  ├── Background thread (every POLL_INTERVAL s, default 1s)
  │     ├── GET {LLAMA_URL}/health
  │     ├── GET {LLAMA_URL}/metrics   (Prometheus text format)
  │     ├── GET {LLAMA_URL}/props     (once, cached)
  │     └── nvidia-smi --id={GPU_INDEX}
  ├── compute_deltas()              → instant throughput
  ├── compute_hourly_averages()     → 1h rolling prefill/decode rates
  ├── compute_24h_generated()       → rolling 24h token total
  ├── GET /api/status   → latest snapshot
  ├── GET /api/history  → deque of recent chart points
  ├── GET /api/props    → cached server properties
  └── GET /             → dashboard.html (Chart.js from CDN)
```

## Project layout

| File                  | Purpose                                              |
|-----------------------|------------------------------------------------------|
| `monitor.py`          | HTTP server, collector thread, metrics API           |
| `dashboard.html`      | Single-page dashboard UI                             |
| `llama-monitor.service` | systemd user unit template                         |
| `.env.example`        | Environment variable template                        |

#!/usr/bin/env python3
"""
Llama.cpp Server Monitor
========================
Self-contained monitoring dashboard for llama.cpp inference server.
No external dependencies — uses only Python stdlib.

Computes DELTA-based metrics (instant throughput per poll interval) so
the dashboard shows real-time activity: spikes when busy, flat when idle.

Exposes:
  GET /           — Dashboard HTML (Chart.js from CDN)
  GET /api/status — Current snapshot (health + metrics + GPU + deltas)
  GET /api/history — Time-series data for charts (last 5 min)

Collects metrics every 1s in a background thread.
Tracks 1-hour rolling averages for prefill and decode throughput.
"""

import http.server
import json
import os
import subprocess
import threading
import time
import urllib.request
import urllib.error
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file (stdlib only; no python-dotenv)."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(Path(__file__).parent / ".env")

LLAMA_HOST = os.environ.get("LLAMA_HOST", "localhost")
LLAMA_PORT = os.environ.get("LLAMA_PORT", "8080")
LLAMA_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}"
LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY", "")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "1"))       # seconds
HISTORY_MAX = int(os.environ.get("HISTORY_MAX", "300"))          # 5 min @ 1s
HISTORY_HOUR_MAX = int(os.environ.get("HISTORY_HOUR_MAX", "3600"))  # 1 hour @ 1s
HISTORY_24H_MAX = int(os.environ.get("HISTORY_24H_MAX", "86400"))  # 24 hours @ 1s
MONITOR_PORT = int(os.environ.get("MONITOR_PORT", "8100"))
GPU_INDEX = int(os.environ.get("GPU_INDEX", "1"))                # RTX 5060 Ti

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
history: deque = deque(maxlen=HISTORY_MAX)
history_hour: deque = deque(maxlen=HISTORY_HOUR_MAX)
history_24h: deque = deque(maxlen=HISTORY_24H_MAX)
current_status: dict = {}
props_cache: dict = {}
props_fetched: bool = False
prev_snapshot: dict = {}   # previous raw snapshot for delta computation
prev_req_processing: int = 0  # for detecting request completions
requests_processed_total: int = 0  # running count of completed requests
lock = threading.Lock()

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def fetch_json(url: str, timeout: float = 3.0) -> dict | None:
    req = urllib.request.Request(url)
    if LLAMA_API_KEY:
        req.add_header("Authorization", f"Bearer {LLAMA_API_KEY}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def fetch_metrics_raw(url: str, timeout: float = 3.0) -> str | None:
    req = urllib.request.Request(url)
    if LLAMA_API_KEY:
        req.add_header("Authorization", f"Bearer {LLAMA_API_KEY}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode()
    except Exception:
        return None


def parse_prometheus_metrics(text: str) -> dict:
    metrics = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            name, value = parts
            try:
                metrics[name] = float(value)
            except ValueError:
                metrics[name] = value
    return metrics


def get_gpu_stats() -> dict:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={GPU_INDEX}",
                "--query-gpu=name,memory.used,memory.total,temperature.gpu,"
                "utilization.gpu,utilization.memory,power.draw,"
                "clocks.current.graphics,clocks.current.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return {"error": "nvidia-smi failed"}
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) >= 9:
            return {
                "name": parts[0],
                "vram_used_mb": int(parts[1]),
                "vram_total_mb": int(parts[2]),
                "temperature_c": int(parts[3]),
                "gpu_util_pct": int(parts[4]),
                "mem_util_pct": int(parts[5]),
                "power_w": float(parts[6]),
                "clock_graphics_mhz": int(parts[7]),
                "clock_memory_mhz": int(parts[8]),
            }
        return {"error": "unexpected nvidia-smi output"}
    except Exception as e:
        return {"error": str(e)}


def collect_once() -> dict:
    """Collect one raw snapshot of all metrics (cumulative counters)."""
    ts = time.time()
    health = fetch_json(f"{LLAMA_URL}/health")
    health_ok = health is not None and health.get("status") == "ok"
    raw_metrics = fetch_metrics_raw(f"{LLAMA_URL}/metrics")
    metrics = parse_prometheus_metrics(raw_metrics) if raw_metrics else {}
    gpu = get_gpu_stats()

    return {
        "ts": ts,
        "ts_iso": datetime.now(timezone.utc).isoformat(),
        "health_ok": health_ok,
        "health": health or {"status": "unreachable"},
        "metrics": metrics,
        "gpu": gpu,
    }


def compute_deltas(prev: dict, curr: dict) -> dict:
    """
    Compute instant (delta-based) metrics between two snapshots.
    Returns a dict of derived values that reflect what happened in THIS interval.
    """
    dt = curr["ts"] - prev["ts"]
    if dt <= 0:
        dt = POLL_INTERVAL

    cm = curr.get("metrics", {})
    pm = prev.get("metrics", {})

    # Delta counters
    d_prompt_tokens = max(0, cm.get("llamacpp:prompt_tokens_total", 0)
                          - pm.get("llamacpp:prompt_tokens_total", 0))
    d_gen_tokens = max(0, cm.get("llamacpp:tokens_predicted_total", 0)
                       - pm.get("llamacpp:tokens_predicted_total", 0))
    d_prompt_sec = max(0, cm.get("llamacpp:prompt_seconds_total", 0)
                       - pm.get("llamacpp:prompt_seconds_total", 0))
    d_gen_sec = max(0, cm.get("llamacpp:tokens_predicted_seconds_total", 0)
                    - pm.get("llamacpp:tokens_predicted_seconds_total", 0))
    d_decode_calls = max(0, cm.get("llamacpp:n_decode_total", 0)
                         - pm.get("llamacpp:n_decode_total", 0))

    # Instant throughput: tokens processed in this interval / wall time
    # These go to 0 when idle — that's the point
    instant_prefill_tps = d_prompt_tokens / dt if dt > 0 else 0
    instant_decode_tps = d_gen_tokens / dt if dt > 0 else 0

    # True per-request throughput (only counts actual processing time)
    prefill_per_sec = d_prompt_tokens / d_prompt_sec if d_prompt_sec > 0.01 else 0
    decode_per_sec = d_gen_tokens / d_gen_sec if d_gen_sec > 0.01 else 0

    # Busy state
    req_processing = cm.get("llamacpp:requests_processing", 0)
    req_deferred = cm.get("llamacpp:requests_deferred", 0)
    busy = req_processing > 0 or req_deferred > 0 or d_gen_tokens > 0 or d_prompt_tokens > 0

    return {
        # Instant (wall-clock) throughput — what you see on the chart
        "instant_prefill_tps": round(instant_prefill_tps, 1),
        "instant_decode_tps": round(instant_decode_tps, 1),
        # True per-request throughput (only during processing)
        "prefill_per_sec": round(prefill_per_sec, 1),
        "decode_per_sec": round(decode_per_sec, 1),
        # Tokens this interval
        "delta_prompt_tokens": int(d_prompt_tokens),
        "delta_gen_tokens": int(d_gen_tokens),
        # Activity
        "busy": busy,
        "requests_processing": int(req_processing),
        "requests_deferred": int(req_deferred),
        "decode_calls_delta": int(d_decode_calls),
    }


def compute_hourly_averages() -> dict:
    """Compute 1-hour rolling averages of prefill and decode throughput.
    Only counts intervals where the model was actually processing."""
    prefill_rates = []
    decode_rates = []
    for snap in history_hour:
        d = snap.get("deltas", {})
        pp = d.get("prefill_per_sec", 0)
        dp = d.get("decode_per_sec", 0)
        if pp > 0:
            prefill_rates.append(pp)
        if dp > 0:
            decode_rates.append(dp)
    return {
        "avg_prefill_1h": round(sum(prefill_rates) / len(prefill_rates), 1) if prefill_rates else 0,
        "avg_decode_1h": round(sum(decode_rates) / len(decode_rates), 1) if decode_rates else 0,
        "prefill_samples": len(prefill_rates),
        "decode_samples": len(decode_rates),
        "window_seconds": len(history_hour),
    }


def compute_24h_generated() -> int:
    """Compute tokens generated in the last 24 hours by diffing the
    cumulative counter between the oldest and newest snapshot in the window."""
    if len(history_24h) < 2:
        return 0
    oldest = history_24h[0].get("metrics", {}).get("llamacpp:tokens_predicted_total", 0)
    newest = history_24h[-1].get("metrics", {}).get("llamacpp:tokens_predicted_total", 0)
    return max(0, int(newest - oldest))


def collector_loop():
    global current_status, props_cache, props_fetched, prev_snapshot
    global prev_req_processing, requests_processed_total
    while True:
        try:
            snap = collect_once()

            # Compute deltas from previous snapshot
            if prev_snapshot:
                deltas = compute_deltas(prev_snapshot, snap)
            else:
                deltas = {
                    "instant_prefill_tps": 0, "instant_decode_tps": 0,
                    "prefill_per_sec": 0, "decode_per_sec": 0,
                    "delta_prompt_tokens": 0, "delta_gen_tokens": 0,
                    "busy": False, "requests_processing": 0,
                    "requests_deferred": 0, "decode_calls_delta": 0,
                }

            # Enrich snapshot with deltas
            snap["deltas"] = deltas

            # Detect request completions: when processing count drops, the
            # difference is requests that just finished.
            curr_processing = snap.get("metrics", {}).get("llamacpp:requests_processing", 0)
            delta_req = max(0, prev_req_processing - curr_processing)
            requests_processed_total += delta_req
            prev_req_processing = int(curr_processing)

            with lock:
                current_status = snap
                history.append(snap)
                history_hour.append(snap)
                history_24h.append(snap)
                current_status["averages"] = compute_hourly_averages()
                current_status["generated_24h"] = compute_24h_generated()
                current_status["requests_processed_total"] = requests_processed_total
                current_status["delta_requests_processed"] = delta_req

            prev_snapshot = snap

            # Fetch props once
            if not props_fetched:
                p = fetch_json(f"{LLAMA_URL}/props")
                if p:
                    with lock:
                        props_cache = p
                        props_fetched = True

        except Exception as e:
            with lock:
                current_status = {
                    "ts": time.time(),
                    "ts_iso": datetime.now(timezone.utc).isoformat(),
                    "health_ok": False,
                    "health": {"status": "error", "detail": str(e)},
                    "metrics": {},
                    "gpu": {"error": str(e)},
                    "deltas": {
                        "instant_prefill_tps": 0, "instant_decode_tps": 0,
                        "prefill_per_sec": 0, "decode_per_sec": 0,
                        "delta_prompt_tokens": 0, "delta_gen_tokens": 0,
                        "busy": False, "requests_processing": 0,
                        "requests_deferred": 0, "decode_calls_delta": 0,
                    },
                }
            with lock:
                current_status["averages"] = compute_hourly_averages()
                current_status["generated_24h"] = compute_24h_generated()
                current_status["requests_processed_total"] = requests_processed_total
                current_status["delta_requests_processed"] = 0
        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

DASHBOARD_HTML_PATH = Path(__file__).parent / "dashboard.html"


class MonitorHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html: bytes, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            try:
                html = DASHBOARD_HTML_PATH.read_bytes()
                self._html(html)
            except FileNotFoundError:
                self._html(b"<h1>dashboard.html not found</h1>", 404)
        elif self.path == "/api/status":
            with lock:
                self._json(current_status)
        elif self.path == "/api/history":
            with lock:
                self._json({"points": list(history)})
        elif self.path == "/api/props":
            with lock:
                self._json(props_cache if props_fetched else {})
        else:
            self.send_error(404)


def main():
    t = threading.Thread(target=collector_loop, daemon=True)
    t.start()
    print(f"[monitor] collecting from {LLAMA_URL} every {POLL_INTERVAL}s")
    print(f"[monitor] dashboard at http://0.0.0.0:{MONITOR_PORT}")

    server = http.server.HTTPServer(("0.0.0.0", MONITOR_PORT), MonitorHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[monitor] shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()

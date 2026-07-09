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
USD_TO_ZAR = float(os.environ.get("USD_TO_ZAR", "17"))           # Rand per USD
ZAR_PER_KWH = float(os.environ.get("ZAR_PER_KWH", "2"))          # local GPU electricity
STATE_FILE = Path(os.environ.get(
    "STATE_FILE",
    str(Path(__file__).parent / "data" / "monitor_state.json"),
))
STATE_SAVE_INTERVAL = int(os.environ.get("STATE_SAVE_INTERVAL", "30"))  # seconds

# Frontier model API pricing (USD per 1M tokens: input, output).
# Standard list rates as of mid-2026; no cache/batch discounts applied.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "haiku_4_5": {"label": "Haiku 4.5", "input": 1.0, "output": 5.0},
    "sonnet_5": {"label": "Sonnet 5", "input": 2.0, "output": 10.0},
    "gpt_5_5": {"label": "GPT 5.5", "input": 5.0, "output": 30.0},
    "fable_5": {"label": "Fable 5", "input": 10.0, "output": 50.0},
    "grok_4_5": {"label": "Grok 4.5", "input": 2.0, "output": 6.0},
}

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
prompt_tokens_total: int = 0  # cumulative prompt tokens since monitor start
gen_tokens_total: int = 0  # cumulative generated tokens since monitor start
energy_wh_total: float = 0.0  # cumulative GPU energy (watt-hours) since monitor start
last_state_save: float = 0.0
lock = threading.Lock()

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _auth_request(url: str, timeout: float = 3.0) -> tuple[bytes | None, str | None]:
    """HTTP GET with optional API key. Returns (body, error)."""
    req = urllib.request.Request(url)
    if LLAMA_API_KEY:
        req.add_header("Authorization", f"Bearer {LLAMA_API_KEY}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), None
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:120]
        return None, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return None, str(e)


def fetch_json(url: str, timeout: float = 3.0) -> dict | None:
    body, err = _auth_request(url, timeout)
    if err or body is None:
        return None
    try:
        return json.loads(body.decode())
    except Exception:
        return None


def fetch_metrics_raw(url: str, timeout: float = 3.0) -> tuple[str | None, str | None]:
    body, err = _auth_request(url, timeout)
    if err:
        return None, err
    return body.decode() if body is not None else None, None


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
    raw_metrics, metrics_error = fetch_metrics_raw(f"{LLAMA_URL}/metrics")
    metrics = parse_prometheus_metrics(raw_metrics) if raw_metrics else {}
    metrics_ok = bool(metrics)
    if health_ok and not metrics_ok and not metrics_error:
        metrics_error = "empty /metrics response"
    elif health_ok and not metrics_ok and not LLAMA_API_KEY:
        metrics_error = "LLAMA_API_KEY not set (required for /metrics)"
    gpu = get_gpu_stats()

    return {
        "ts": ts,
        "ts_iso": datetime.now(timezone.utc).isoformat(),
        "health_ok": health_ok,
        "metrics_ok": metrics_ok,
        "metrics_error": metrics_error,
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
        # Tokens and processing time this interval
        "delta_prompt_tokens": int(d_prompt_tokens),
        "delta_gen_tokens": int(d_gen_tokens),
        "delta_prompt_seconds": round(d_prompt_sec, 4),
        "delta_gen_seconds": round(d_gen_sec, 4),
        # Activity
        "busy": busy,
        "requests_processing": int(req_processing),
        "requests_deferred": int(req_deferred),
        "decode_calls_delta": int(d_decode_calls),
    }


def compute_hourly_averages() -> dict:
    """Compute 1-hour rolling averages of prefill and decode throughput.

    Uses total tokens / total active processing seconds over the window,
    so idle/downtime intervals (zero deltas) do not dilute the average."""
    total_prompt_tokens = 0
    total_prompt_sec = 0.0
    total_gen_tokens = 0
    total_gen_sec = 0.0
    for snap in history_hour:
        d = snap.get("deltas", {})
        total_prompt_tokens += d.get("delta_prompt_tokens", 0)
        total_prompt_sec += d.get("delta_prompt_seconds", 0)
        total_gen_tokens += d.get("delta_gen_tokens", 0)
        total_gen_sec += d.get("delta_gen_seconds", 0)
    return {
        "avg_prefill_1h": round(total_prompt_tokens / total_prompt_sec, 1)
        if total_prompt_sec > 0.01 else 0,
        "avg_decode_1h": round(total_gen_tokens / total_gen_sec, 1)
        if total_gen_sec > 0.01 else 0,
        "prefill_active_seconds": round(total_prompt_sec, 1),
        "decode_active_seconds": round(total_gen_sec, 1),
        "window_seconds": len(history_hour) * POLL_INTERVAL,
    }


def compute_24h_generated() -> int:
    """Compute tokens generated in the last 24 hours by diffing the
    cumulative counter between the oldest and newest snapshot in the window."""
    tokens = compute_24h_token_totals()
    return tokens["generated"]


def load_persisted_state() -> None:
    """Restore session accumulators from disk (survives monitor restarts)."""
    global prompt_tokens_total, gen_tokens_total, requests_processed_total
    global energy_wh_total
    if not STATE_FILE.is_file():
        return
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        prompt_tokens_total = int(data.get("prompt_tokens_total", 0))
        gen_tokens_total = int(data.get("gen_tokens_total", 0))
        requests_processed_total = int(data.get("requests_processed_total", 0))
        energy_wh_total = float(data.get("energy_wh_total", 0))
        print(
            f"[monitor] loaded state: {prompt_tokens_total} prompt / "
            f"{gen_tokens_total} generated tokens, "
            f"{energy_wh_total:.1f} Wh energy",
        )
    except Exception as e:
        print(f"[monitor] warning: could not load {STATE_FILE}: {e}")


def save_persisted_state() -> None:
    """Persist session accumulators to disk."""
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "prompt_tokens_total": prompt_tokens_total,
            "gen_tokens_total": gen_tokens_total,
            "requests_processed_total": requests_processed_total,
            "energy_wh_total": round(energy_wh_total, 4),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(STATE_FILE)
    except Exception as e:
        print(f"[monitor] warning: could not save {STATE_FILE}: {e}")


def maybe_save_persisted_state() -> None:
    """Throttle state writes to avoid excessive disk I/O."""
    global last_state_save
    now = time.time()
    if now - last_state_save >= STATE_SAVE_INTERVAL:
        save_persisted_state()
        last_state_save = now


def compute_24h_energy_wh() -> float:
    """GPU energy (watt-hours) integrated over the rolling 24h window."""
    if len(history_24h) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(history_24h)):
        prev = history_24h[i - 1]
        curr = history_24h[i]
        dt = curr["ts"] - prev["ts"]
        if dt <= 0:
            continue
        power = curr.get("gpu", {}).get("power_w")
        if power is not None:
            total += float(power) * (dt / 3600.0)
    return total


def electricity_cost_zar(wh: float) -> float:
    """Local electricity cost in ZAR for the given watt-hours."""
    return round((wh / 1000.0) * ZAR_PER_KWH, 2)


def compute_local_electricity() -> dict:
    """Estimated local GPU electricity cost (not frontier API pricing)."""
    wh_24h = compute_24h_energy_wh()
    return {
        "label": "Local GPU (electricity)",
        "zar_per_kwh": ZAR_PER_KWH,
        "cumulative": {
            "wh": round(energy_wh_total, 2),
            "kwh": round(energy_wh_total / 1000.0, 4),
            "zar": electricity_cost_zar(energy_wh_total),
        },
        "last_24h": {
            "wh": round(wh_24h, 2),
            "kwh": round(wh_24h / 1000.0, 4),
            "zar": electricity_cost_zar(wh_24h),
        },
    }


def compute_24h_token_totals() -> dict:
    """Prompt and generated token totals over the rolling 24h window."""
    if len(history_24h) < 2:
        return {"prompt": 0, "generated": 0}
    oldest_m = history_24h[0].get("metrics", {})
    newest_m = history_24h[-1].get("metrics", {})
    prompt = max(
        0,
        int(newest_m.get("llamacpp:prompt_tokens_total", 0)
            - oldest_m.get("llamacpp:prompt_tokens_total", 0)),
    )
    generated = max(
        0,
        int(newest_m.get("llamacpp:tokens_predicted_total", 0)
            - oldest_m.get("llamacpp:tokens_predicted_total", 0)),
    )
    return {"prompt": prompt, "generated": generated}


def model_cost_zar(tokens_in: int, tokens_out: int, input_per_m: float,
                   output_per_m: float) -> float:
    """Hypothetical API cost in ZAR for the given token counts."""
    usd = (tokens_in * input_per_m + tokens_out * output_per_m) / 1_000_000
    return round(usd * USD_TO_ZAR, 2)


def compute_frontier_costs(tokens_in: int, tokens_out: int) -> dict:
    """Hypothetical frontier-model API costs for prompt + output tokens."""
    models = {}
    for key, pricing in MODEL_PRICING.items():
        zar = model_cost_zar(
            tokens_in, tokens_out, pricing["input"], pricing["output"],
        )
        models[key] = {"label": pricing["label"], "zar": zar}
    return models


def compute_cost_comparison() -> dict:
    """Cumulative and 24h costs: local electricity + hypothetical frontier APIs."""
    tokens_24h = compute_24h_token_totals()
    cumulative = compute_frontier_costs(prompt_tokens_total, gen_tokens_total)
    last_24h = compute_frontier_costs(
        tokens_24h["prompt"], tokens_24h["generated"],
    )
    cumulative_order = sorted(
        cumulative.keys(),
        key=lambda k: cumulative[k]["zar"],
        reverse=True,
    )
    return {
        "usd_to_zar": USD_TO_ZAR,
        "zar_per_kwh": ZAR_PER_KWH,
        "tokens_total": {
            "prompt": prompt_tokens_total,
            "generated": gen_tokens_total,
        },
        "tokens_24h": tokens_24h,
        "cumulative": cumulative,
        "last_24h": last_24h,
        "cumulative_order": cumulative_order,
        "local_electricity": compute_local_electricity(),
    }


def collector_loop():
    global current_status, props_cache, props_fetched, prev_snapshot
    global prev_req_processing, requests_processed_total
    global prompt_tokens_total, gen_tokens_total, energy_wh_total
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
                    "delta_prompt_seconds": 0, "delta_gen_seconds": 0,
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
            prompt_tokens_total += deltas.get("delta_prompt_tokens", 0)
            gen_tokens_total += deltas.get("delta_gen_tokens", 0)

            dt_energy = snap["ts"] - prev_snapshot["ts"] if prev_snapshot else POLL_INTERVAL
            if dt_energy <= 0:
                dt_energy = POLL_INTERVAL
            power_w = snap.get("gpu", {}).get("power_w")
            if power_w is not None:
                energy_wh_total += float(power_w) * (dt_energy / 3600.0)

            with lock:
                current_status = snap
                history.append(snap)
                history_hour.append(snap)
                history_24h.append(snap)
                current_status["averages"] = compute_hourly_averages()
                current_status["generated_24h"] = compute_24h_generated()
                current_status["cost_comparison"] = compute_cost_comparison()
                current_status["requests_processed_total"] = requests_processed_total
                current_status["delta_requests_processed"] = delta_req

            maybe_save_persisted_state()

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
                    "metrics_ok": False,
                    "metrics_error": str(e),
                    "health": {"status": "error", "detail": str(e)},
                    "metrics": {},
                    "gpu": {"error": str(e)},
                    "deltas": {
                        "instant_prefill_tps": 0, "instant_decode_tps": 0,
                        "prefill_per_sec": 0, "decode_per_sec": 0,
                        "delta_prompt_tokens": 0, "delta_gen_tokens": 0,
                        "delta_prompt_seconds": 0, "delta_gen_seconds": 0,
                        "busy": False, "requests_processing": 0,
                        "requests_deferred": 0, "decode_calls_delta": 0,
                    },
                }
            with lock:
                current_status["averages"] = compute_hourly_averages()
                current_status["generated_24h"] = compute_24h_generated()
                current_status["cost_comparison"] = compute_cost_comparison()
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
    load_persisted_state()
    t = threading.Thread(target=collector_loop, daemon=True)
    t.start()
    print(f"[monitor] collecting from {LLAMA_URL} every {POLL_INTERVAL}s")
    print(f"[monitor] dashboard at http://0.0.0.0:{MONITOR_PORT}")

    server = http.server.HTTPServer(("0.0.0.0", MONITOR_PORT), MonitorHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[monitor] shutting down")
        save_persisted_state()
        server.shutdown()


if __name__ == "__main__":
    main()

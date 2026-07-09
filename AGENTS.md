# llama-cpp-monitor — Agent Instructions

Lightweight monitoring dashboard for a local **llama.cpp** inference server. Python stdlib only (no pip dependencies). Chart.js loaded from CDN in `dashboard.html`.

Read `README.md` for environment variables, API endpoints, and systemd setup.

## Project layout

| Path | Role |
|------|------|
| `monitor.py` | HTTP server, background collector thread, delta-based metrics, JSON API |
| `dashboard.html` | Single-page dashboard (Chart.js, polls `/api/*`) |
| `llama-monitor.service` | systemd user unit example |
| `README.md` | User-facing docs |
| `.codedbignore` | CodeDB skip rules (runtime artifacts, caches, large binaries) |
| `.codedbrc` | CodeDB config (`require_git_repo = false` — this directory is not a git repo) |

## Architecture (high level)

```
monitor.py (MONITOR_PORT, default 8100)
  ├── Background thread (every POLL_INTERVAL s, default 1s)
  │     ├── GET {LLAMA_URL}/health
  │     ├── GET {LLAMA_URL}/metrics  (Prometheus text → parse_prometheus_metrics)
  │     ├── GET {LLAMA_URL}/props    (once, cached)
  │     └── nvidia-smi --id={GPU_INDEX}
  ├── compute_deltas(prev, curr)     → instant throughput (idle → 0)
  ├── compute_hourly_averages()      → 1h rolling prefill/decode rates
  └── MonitorHandler (GET /, /api/status, /api/history, /api/props)
```

**Key idea:** counters from llama.cpp are cumulative; the dashboard shows **delta-based** instant throughput so charts spike when busy and flatline when idle.

## Agent operating guidance

1. **Keep changes minimal** — this is a small, self-contained tool. Prefer editing `monitor.py` and `dashboard.html` only unless the task touches deployment docs or the service file.
2. **No external Python dependencies** — use only the stdlib (`http.server`, `urllib`, `threading`, `subprocess`, etc.). Do not add `requirements.txt` or pip packages unless explicitly requested.
3. **Preserve delta semantics** — new metrics should follow the existing pattern: raw cumulative values in `collect_once()`, derived instant values in `compute_deltas()`, and chart-friendly fields on each history point.
4. **Thread safety** — shared state (`current_status`, `history*`, `props_cache`) is guarded by `lock`. Any new shared state must use the same lock.
5. **Config via env vars** — new tunables belong as `os.environ.get(...)` at the top of `monitor.py` and should be documented in `README.md`.
6. **Dashboard contract** — `dashboard.html` expects specific JSON shapes from `/api/status` and `/api/history`. If you change field names or types, update both sides.

## High-value symbols (start here)

| Area | Symbols / locations |
|------|---------------------|
| Collection | `collect_once`, `fetch_json`, `fetch_metrics_raw`, `parse_prometheus_metrics`, `get_gpu_stats` |
| Throughput | `compute_deltas`, `compute_hourly_averages`, `compute_24h_generated` |
| Loop | `collector_loop` |
| HTTP API | `MonitorHandler.do_GET`, routes `/api/status`, `/api/history`, `/api/props` |
| Frontend | `refresh()`, `loadProps()`, `holdValue()` in `dashboard.html` |
| Entry | `main()` |

Prometheus metric names used (llama.cpp): `llamacpp:prompt_tokens_total`, `llamacpp:tokens_predicted_total`, `llamacpp:requests_processing`, `llamacpp:requests_deferred`, etc.

## Running

```bash
python3 monitor.py
# Dashboard at http://localhost:8100 (or MONITOR_PORT)
```

Useful env overrides:

```bash
LLAMA_HOST=localhost LLAMA_PORT=8080 MONITOR_PORT=8100 GPU_INDEX=1 python3 monitor.py
```

## Conventions

- Python 3.10+ type hints (`dict | None`, etc.) — match existing style in `monitor.py`
- No request logging in `MonitorHandler.log_message` (intentionally silenced)
- `LLAMA_API_KEY` sent as `Authorization: Bearer …` when set
- Dark-theme dashboard; keep UI changes consistent with existing CSS variables in `dashboard.html`

## Code intelligence: using CodeDB

CodeDB is an indexed, symbol-aware navigation layer (MCP tools). Use it **before** blind `grep` or reading whole files — this repo is small, but CodeDB still gives outlines, dependency context, and targeted reads.

### First-time setup for this project

This directory is **not a git repository**. CodeDB refuses non-git roots by default. Use one of:

1. **Project `.codedbrc`** (included here) — `require_git_repo = false`. CodeDB loads config from the MCP server’s startup CWD, so ensure that setting is active (e.g. start MCP from this directory, or mirror the line in your global `~/.codedb/.codedbrc` / binary-dir config).
2. **Environment override** — `CODEDB_REQUIRE_GIT_REPO=0` when starting the MCP server.
3. **`git init`** — a fresh repo with no commits still satisfies the work-tree check if you prefer to keep the default.

Index once per session (lazy MCP mode):

```
codedb_index path=/mnt/storage/repos/llama-cpp-monitor
```

Confirm health:

```
codedb_status
codedb_tree
```

Re-index after large structural changes or if results look stale:

```
codedb_index path=/mnt/storage/repos/llama-cpp-monitor force=true
```

### Recommended tool flow

1. **`codedb_status`** — confirm snapshot and scan state.
2. **`codedb_tree`** or **`codedb_ls`** — orient (only a handful of source files).
3. **`codedb_symbol`** / **`codedb_word`** — locate a function or identifier.
4. **`codedb_outline`** — see structure before reading (especially `dashboard.html`).
5. **`codedb_read`** — read only the line range you need (`line_start` / `line_end`).
6. **`codedb_callers`** / **`codedb_deps`** / **`codedb_relations`** — impact analysis before edits.

Chain steps in one call with **`codedb_query`** when narrowing then reading:

```json
[
  {"op": "symbol", "name": "compute_deltas"},
  {"op": "read", "context_lines": 5}
]
```

### Project-specific examples

| Task | CodeDB approach |
|------|-----------------|
| How is throughput calculated? | `codedb_symbol(name="compute_deltas")` → `codedb_read` around the definition |
| What calls the collector? | `codedb_callers(name="collect_once")` |
| Where are API routes defined? | `codedb_symbol(name="MonitorHandler")` or `codedb_search(query="do_GET")` |
| Find Prometheus parsing | `codedb_symbol(name="parse_prometheus_metrics")` |
| Dashboard polling logic | `codedb_outline(path="dashboard.html")` → `codedb_read` at `refresh` / `loadProps` |
| Search for a metric name | `codedb_search(query="llamacpp:tokens_predicted_total")` |
| Who uses `history_hour`? | `codedb_word(word="history_hour")` |

### Tool quick reference

| Tool | Use when |
|------|----------|
| `codedb_word` | Exact identifier occurrences (fastest) |
| `codedb_symbol` | Find where a function/class is defined |
| `codedb_search` | Substring or regex search across files |
| `codedb_outline` | File structure before `codedb_read` |
| `codedb_read` | Read lines; pass `if_hash` to skip unchanged files |
| `codedb_callers` | Who invokes a function |
| `codedb_deps` | Import/usage blast radius for a file |
| `codedb_relations` | One-shot map: defs, callers, deps for a symbol |
| `codedb_query` | Pipeline multiple ops (discover → read) |
| `codedb_hot` | Recently modified files |

### Investigation patterns

**Pattern A — Add a new metric to the dashboard**

1. `codedb_symbol("collect_once")` + `codedb_symbol("compute_deltas")` — see where raw and derived metrics live.
2. `codedb_search(query="deltas")` in `dashboard.html` — find chart bindings.
3. Extend collector → deltas → `refresh()` in HTML; keep field names consistent.

**Pattern B — Change an API response shape**

1. `codedb_symbol("MonitorHandler")` + `codedb_read` on `do_GET`.
2. `codedb_word("api/status")` or search `fetch('/api/status')` in `dashboard.html`.
3. Update handler and frontend together.

**Pattern C — Debug GPU stats**

1. `codedb_symbol("get_gpu_stats")` — nvidia-smi invocation and field mapping.
2. `codedb_search(query="GPU_INDEX")` — env wiring.

## Guardrails

- Do not commit or hardcode real API keys; `LLAMA_API_KEY` defaults in `monitor.py` should be treated as local-dev only.
- Do not add heavyweight frameworks (Flask, FastAPI, etc.) without explicit request — stdlib HTTP server is intentional.
- Do not change poll/history defaults without updating `README.md` and any chart time-window logic in `dashboard.html`.
- GPU stats require `nvidia-smi` on the host; handle missing GPU gracefully (existing `{"error": ...}` pattern).

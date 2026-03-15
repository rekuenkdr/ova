# Tools / Function Calling

OVA supports LLM tool calling — the chat model can invoke real Python functions during a conversation and incorporate the results into its spoken response. For example, the user can ask "What time is it?" and the LLM will call `get_time`, receive the result, and speak the answer naturally.

Tools are **plugin-style**: drop a `.py` file in `ova/tools/` and the registry auto-discovers it. No manual registration required.

**Latency impact:** ~200ms added only when the LLM actually invokes a tool. Regular conversations with no tool calls are unaffected.

---

## Quick Start

1. Enable tools in `.env`:
   ```
   OVA_ENABLE_TOOLS=true
   ```

2. Restart OVA.

3. Verify via the info endpoint:
   ```bash
   curl http://localhost:5173/v1/info | jq '{tools_enabled, tools_available}'
   ```
   ```json
   {
     "tools_enabled": true,
     "tools_available": ["check_timers", "get_date", "get_time", "set_timer"]
   }
   ```

---

## Architecture

```
User audio/text
      │
      ▼
  API endpoint (/v1/chat/audio, /v1/chat)
      │
      ├── _tools_active()? ──── No ──▶ pipeline.chat() (streaming LLM)
      │
      Yes
      │
      ▼
  pipeline.chat_with_tools()          ◀─── Non-streaming LLM + tool loop
      │
      │  ┌─────────────────────────────────────────────┐
      │  │  for iteration in range(MAX_TOOL_ITERATIONS):│
      │  │    ollama.chat(messages, tools=...)           │
      │  │    if no tool_calls → break                  │
      │  │    for each tool_call:                       │
      │  │      ToolRegistry.execute(name, args)        │
      │  │      append {"role": "tool", "content": ...} │
      │  │    re-prompt LLM with results                │
      │  └─────────────────────────────────────────────┘
      │
      ▼
  Final text response
      │
      ▼
  TTS streaming → audio to client
```

### Discovery Flow

1. `OVAPipeline.__init__()` imports and calls `tool_registry.discover()` (`pipeline.py:204-207`)
2. `discover()` scans all `.py` files in `ova/tools/` (skipping `_`-prefixed modules like `_base.py`)
3. Each public function with a Google-style docstring (containing `Args:`) is registered
4. Enable/disable rules are evaluated per function (see [Configuration](#configuration))
5. The registry is stored on the pipeline as `self.tool_registry`

### API Integration

- `_tools_active()` in `api.py:335` checks `pipeline.tool_registry.enabled`
- When active, all chat endpoints route through `chat_with_tools()` instead of `chat()`
- `GET /v1/info` exposes `tools_enabled` and `tools_available` (`api.py:977-978`)

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_ENABLE_TOOLS` | `false` | Master toggle — must be `true` for any tools to work |
| `OVA_MAX_TOOL_ITERATIONS` | `5` | Max LLM↔tool round-trips per request (prevents runaway loops) |
| `OVA_DISABLED_TOOLS` | *(empty)* | Comma-separated function names to disable (e.g. `web_search,set_timer`) |
| `OVA_TOOL_<NAME>_ENABLED` | *(unset)* | Per-tool override: `true` or `false` (name is UPPER_CASE, e.g. `OVA_TOOL_WEB_SEARCH_ENABLED`) |
| `OVA_SEARCH_API_KEY` | *(empty)* | API key for the web search tool |

### Enable/Disable Priority

The registry evaluates each function in this order (first match wins):

1. **Per-tool env override** — `OVA_TOOL_<NAME>_ENABLED=true/false`
2. **Disabled list** — `OVA_DISABLED_TOOLS=name1,name2`
3. **Module default** — `TOOL_ENABLED_DEFAULT` variable in the tool's module

Example: `web_search` has `TOOL_ENABLED_DEFAULT = False`, so it's excluded by default. Setting `OVA_TOOL_WEB_SEARCH_ENABLED=true` overrides this and enables it.

---

## Built-in Tools

### `get_current_datetime`

| | |
|---|---|
| **File** | `ova/tools/get_datetime.py` |
| **Enabled by default** | Yes |
| **Signature** | `get_current_datetime(query: str = "") -> str` |

Returns the current date and time, or resolves a natural-language date expression using `dateparser` (e.g. "yesterday", "next Friday", "3 days ago", "ayer", "la semana pasada").

**Arguments:**
- `query` — Natural language date expression (e.g. `"yesterday"`, `"next Tuesday"`). Leave empty for current date and time.

**Returns:**
- No query: `"Friday, February 13, 2026 at 14:32 CET"`
- With query: `"Wednesday, February 12, 2026"`

Returns an error message if the expression cannot be parsed.

---

### `set_timer`

| | |
|---|---|
| **File** | `ova/tools/timer.py` |
| **Enabled by default** | Yes |
| **Signature** | `set_timer(label: str, seconds: int) -> str` |

Sets an in-memory countdown timer. When the timer expires, it fires a `timer_expired` event via the [EventBus](#real-time-events), which triggers a chime and toast notification on connected frontends.

**Arguments:**
- `label` — A short name for the timer (e.g. `"pasta"`, `"break"`)
- `seconds` — Duration in seconds, clamped to 1–3600

**Returns:** Confirmation like `"Timer 'pasta' set for 300 seconds."`

Thread-safe — uses a module-level lock for shared state. Uses `threading.Timer` for active expiration callbacks.

---

### `check_timers`

| | |
|---|---|
| **File** | `ova/tools/timer.py` |
| **Enabled by default** | Yes |
| **Signature** | `check_timers() -> str` |

Checks the status of all active timers. Expired timers are reported as `DONE` and automatically cleaned up.

**Arguments:** None (docstring uses `(none)` placeholder)

**Returns:** A multi-line summary:
```
- pasta: 4m 12s remaining
- break: DONE (finished 3s ago)
```
Or `"No active timers."` if empty.

---

## Real-Time Events

Tools can push real-time notifications to connected frontend clients via the **EventBus**. Events flow from server to browser over Server-Sent Events (SSE).

### How It Works

```
Tool function                    Backend                          Frontend
─────────────                    ───────                          ────────
publish_event("my_event", data)
       │
       ▼
  EventBus.publish()             ──▶  SSE endpoint (GET /v1/events)
  (thread-safe via                         │
   call_soon_threadsafe)                   ▼
                                    EventSource auto-reconnect
                                           │
                                           ▼
                                    onEvent("my_event", handler)
                                           │
                                           ▼
                                    handler(event)  →  toast / chime / etc.
```

### Event Payload Format

Every event published through the bus has this shape:

```json
{"type": "timer_expired", "data": {"label": "pasta", "seconds": 300}, "ts": 1737456789.123}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | Event name (matches the SSE `event:` field) |
| `data` | `object` | Arbitrary payload — tool-specific |
| `ts` | `float` | Unix timestamp when the event was published |

### SSE Endpoint

**`GET /v1/events`** — Long-lived SSE stream. The browser's `EventSource` API connects here and auto-reconnects on drop.

- **Named events:** Each event uses the SSE `event:` field (e.g. `event: timer_expired`), so the frontend can register listeners per type.
- **Keepalive:** A `: keepalive` comment is sent every 30 seconds to prevent proxy timeouts.
- **Headers:** `Cache-Control: no-cache`, `X-Accel-Buffering: no` (nginx compatibility).

### Publishing Events from a Tool

Import `publish_event` from `_base.py` and call it — one line:

```python
from ._base import publish_event

def my_tool(arg: str) -> str:
    """Do something and notify the frontend.

    Args:
        arg: Some argument.
    """
    # ... do work ...
    publish_event("my_event", {"result": "success", "detail": arg})
    return "Done."
```

`publish_event` is thread-safe — it uses `call_soon_threadsafe` to push events onto the async event loop from sync tool code running in the threadpool.

### Handling Events in the Frontend

**1. Register a handler** in `app.js` (or your module) before calling `connectEventStream()`:

```javascript
import { onEvent } from './events.js';
import { showNotification, playAlarmTone } from './notifications.js';

onEvent("my_event", (event) => {
    playAlarmTone();
    showNotification({
        title: "My Tool",
        message: event.data.detail,
    });
});
```

**2. The `event` object** passed to your handler is the full JSON payload (`{type, data, ts}`).

**3. Available notification helpers** (from `notifications.js`):
- `requestPermission()` — Requests OS notification permission via the Web Notifications API. Called once at startup; safe to call multiple times (browser no-ops after first grant/deny).
- `showNotification({ title, message })` — **Hybrid:** fires an OS-level `new Notification()` when the tab is hidden and permission is granted; otherwise falls back to an in-page toast (top-center, persists until dismissed). Permission is requested at startup via `requestPermission()`.
- `playAlarmTone()` — Two-tone chime via Web Audio API (no external files, doesn't trigger barge-in). Independent of `showNotification()` — callers decide whether to play it.

### Timer as a Complete Example

The `set_timer` tool demonstrates the full event flow:

**Backend** (`ova/tools/timer.py`):
```python
from ._base import publish_event

def _on_expire(label: str, seconds: int):
    with _lock:
        _timers.pop(label, None)
    publish_event("timer_expired", {"label": label, "seconds": seconds})
```

**Frontend** (`static/js/app.js`):
```javascript
onEvent("timer_expired", (event) => {
    playAlarmTone();
    showNotification({                   // OS notification if tab hidden,
        title: event.data.label,            // in-page toast otherwise
        message: "Timer finished",
    });
});
connectEventStream();
requestPermission();                     // ask for OS notification permission once
```

### Creating a New Tool with Events — Checklist

1. Add `from ._base import publish_event` in your tool module
2. Call `publish_event("your_event_type", {…})` at the appropriate point
3. In `static/js/app.js`, register a handler with `onEvent("your_event_type", fn)` before `connectEventStream()`
4. Use `showNotification()` and/or `playAlarmTone()` from `notifications.js` for UI feedback

---

## MCP (External Tool Servers)

In addition to native Python tools, OVA can connect to external **MCP (Model Context Protocol)** servers. MCP tools appear alongside native tools in the LLM's tool list — the LLM doesn't know or care whether a tool is native or MCP.

This lets you add capabilities (filesystem access, databases, APIs, etc.) without writing Python code — just add a server entry to `mcp_servers.json` and restart.

See [`MCP.md`](MCP.md) for the full guide — server configuration, transport types, examples, architecture, and security considerations.

### Quick Summary

- Enable with `OVA_ENABLE_MCP=true` (requires `OVA_ENABLE_TOOLS=true` as well)
- Configure servers in `mcp_servers.json` (same format as Claude Desktop / VS Code)
- Supports stdio (local subprocess) and SSE/HTTP (remote) transports
- Native tools always take priority on name collisions
- Failed MCP servers don't block startup — graceful degradation

---

### `web_search`

| | |
|---|---|
| **File** | `ova/tools/web_search.py` |
| **Enabled by default** | No (`TOOL_ENABLED_DEFAULT = False`) |
| **Signature** | `web_search(query: str) -> str` |

Stub for web search. Currently returns a not-configured or not-implemented message.

**Arguments:**
- `query` — The search query string

**To enable:** Set both `OVA_TOOL_WEB_SEARCH_ENABLED=true` and `OVA_SEARCH_API_KEY=<key>` in `.env`.

---

## Creating a New Tool

This walkthrough creates a hypothetical `get_weather` tool.

### Step 1: Create the module

Create `ova/tools/get_weather.py`:

```python
"""Weather lookup tool."""
import os

TOOL_ENABLED_DEFAULT = True


def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: City name (e.g. "Madrid", "New York").
        units: Temperature units, "celsius" or "fahrenheit".

    Returns:
        str: Weather summary or error message.
    """
    api_key = os.getenv("OVA_WEATHER_API_KEY", "")
    if not api_key:
        return "Weather is not configured. Set OVA_WEATHER_API_KEY."

    # Your API call here...
    return f"Weather in {city}: 22°C, partly cloudy"
```

### Step 2: Understand the requirements

Your function **must**:

- Be a **public function** (no `_` prefix)
- Have a **Google-style docstring** with an `Args:` section — this is how the registry discovers it
- Be **defined in the module** (not imported/re-exported from elsewhere)
- Return a `str` (or something that `str()` can convert)
- Use **simple argument types** that the LLM can produce: `str`, `int`, `float`, `bool`

### Step 3: Module-level settings

- `TOOL_ENABLED_DEFAULT = True` — tool is on by default
- `TOOL_ENABLED_DEFAULT = False` — tool is off unless explicitly enabled via `OVA_TOOL_<NAME>_ENABLED=true`

If the variable is absent, it defaults to `True`.

### Step 4: Use shared helpers (optional)

`ova/tools/_base.py` provides utilities:

```python
from ._base import get_pipeline_language, get_pipeline_timezone
```

- `get_pipeline_language()` — returns `OVA_LANGUAGE` (default `"es"`)
- `get_pipeline_timezone()` — returns `OVA_TIMEZONE` (default `"UTC"`)

### Step 5: Thread safety

If your tool uses module-level mutable state (like the timer tool does), protect it with a lock:

```python
import threading

_state = {}
_lock = threading.Lock()

def my_tool(arg: str) -> str:
    """...

    Args:
        arg: ...
    """
    with _lock:
        _state[arg] = "value"
    return "done"
```

### Step 6: Error handling

You don't need to catch your own exceptions — the registry wraps every call in a try/except and returns `"Error calling <name>: <exception>"` on failure. But you may want to return user-friendly messages for expected failures (missing API keys, invalid input, etc.).

### Step 7: Restart and verify

Restart OVA. Your tool is auto-discovered:

```bash
curl http://localhost:5173/v1/info | jq .tools_available
# ["check_timers", "get_date", "get_time", "get_weather", "set_timer"]
```

---

## Tool Function Contract

| Requirement | Detail |
|---|---|
| **Name** | Public (no `_` prefix), must not collide with other tool names |
| **Docstring** | Google-style with `Args:` section (even if no args — use `(none)`) |
| **Module** | Defined in its own `ova/tools/<name>.py` (not re-exported) |
| **Return type** | `str` (or `str()`-able) |
| **Argument types** | Simple: `str`, `int`, `float`, `bool` |
| **Thread safety** | Required if using shared mutable state |
| **Module file** | Must not start with `_` (those are skipped during discovery) |

---

## Registry API Reference

The `ToolRegistry` class lives in `ova/tools/__init__.py`. A module-level singleton `registry` is created at import time.

### `discover()`

Scans `ova/tools/*.py`, imports each non-`_`-prefixed module, and registers qualifying public functions. Called once during `OVAPipeline.__init__()`.

### `enabled` (property)

Returns `True` if the master toggle (`OVA_ENABLE_TOOLS`) is on **and** at least one tool is registered.

### `get_enabled_functions() -> list[Callable]`

Returns the list of callable function objects for passing to `ollama.chat(tools=...)`. Returns an empty list if the master toggle is off.

### `get_tool_names() -> list[str]`

Returns a sorted list of registered tool function names (regardless of master toggle).

### `execute(tool_name: str, arguments: dict) -> str`

Calls the named tool with the given keyword arguments. Returns the string result. If the tool is unknown or raises an exception, returns an error string instead (never raises).

---

## Testing Tools

Tests live in `tests/test_tools.py`. Run with:

```bash
pytest tests/test_tools.py -v
```

### Test patterns

**Fresh registry per test** — avoids shared state:
```python
def _make_registry():
    from ova.tools import ToolRegistry
    reg = ToolRegistry()
    reg.discover()
    return reg
```

**Testing env var toggles** — use `importlib.reload()` since `TOOLS_ENABLED` and `DISABLED_TOOLS` are evaluated at import time:
```python
def test_disabled_tools_env():
    with mock.patch.dict(os.environ, {
        "OVA_ENABLE_TOOLS": "true",
        "OVA_DISABLED_TOOLS": "get_time",
    }):
        import importlib
        import ova.tools as tools_mod
        importlib.reload(tools_mod)

        reg = tools_mod.ToolRegistry()
        reg.discover()
        assert "get_time" not in reg.get_tool_names()
```

**Testing execution** — call `registry.execute()` and check the string result:
```python
def test_execute_get_time():
    reg = _make_registry()
    result = reg.execute("get_time", {})
    assert ":" in result  # HH:MM format
```

**Testing individual tools** — import the function directly:
```python
def test_timer_lifecycle():
    from ova.tools.timer import set_timer, check_timers, _timers, _lock
    with _lock:
        _timers.clear()

    result = set_timer(label="pasta", seconds=60)
    assert "pasta" in result

    status = check_timers()
    assert "remaining" in status
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Tool not discovered | Missing `Args:` in docstring | Add a Google-style docstring with `Args:` section |
| Tool not discovered | Function name starts with `_` | Rename to a public name |
| Tool not discovered | Module file starts with `_` | Rename file (e.g. `_helper.py` → `helper.py`) |
| Tool not discovered | Function imported from another module | Define the function directly in its `ova/tools/*.py` file |
| Tool disabled unexpectedly | `OVA_TOOL_<NAME>_ENABLED=false` overriding | Check per-tool env vars, then `OVA_DISABLED_TOOLS`, then `TOOL_ENABLED_DEFAULT` |
| Tool errors in logs | Exception during execution | Check logs — the registry catches exceptions and returns error strings |
| LLM not calling tools | Model doesn't support function calling | Ensure the Ollama model supports tool/function calling |
| `tools_enabled: false` | Master toggle off or no tools registered | Set `OVA_ENABLE_TOOLS=true` and verify at least one tool module exists |

---

## Security

### Execution model

Tools run as server-side Python in the OVA process — there is no sandbox. The LLM decides when to call them based on user input. Any tool you register has full access to the host environment.

### Principle of least privilege

Only enable tools you actually need. `web_search` is disabled by default (`TOOL_ENABLED_DEFAULT = False`) for this reason. Use `OVA_DISABLED_TOOLS` or per-tool overrides (`OVA_TOOL_<NAME>_ENABLED=false`) to restrict the active set.

### Prompt injection

A crafted user input can trick the LLM into calling tools in unintended ways or sequences. Mitigations:

- Keep `OVA_MAX_TOOL_ITERATIONS` low (default 5) to bound runaway loops.
- Validate all arguments inside tool functions (see below).
- Avoid creating tools that perform destructive or irreversible actions.

### Input validation

The LLM controls what arguments are passed to tools — treat them like untrusted user input. Always validate inside the tool function:

- **Sanitize paths** — reject or normalize `..`, absolute paths, symlinks.
- **Cap numeric ranges** — the timer tool clamps `seconds` to 1–3600; follow this pattern.
- **Reject unexpected values** — check enums, string lengths, and character sets.

### API keys & secrets

Store secrets in `.env`, never hardcode them. Don't return raw API keys or secrets in tool output — the LLM will speak them aloud. Follow the existing pattern: `OVA_SEARCH_API_KEY` is read via `os.getenv()` inside the tool, never exposed in responses.

### Network exposure

When OVA is exposed beyond localhost, tools become remotely invocable by anyone who can reach the API. Always:

- Set `OVA_API_KEY` to require `Authorization: Bearer <key>` on all requests.
- Use HTTPS via a reverse proxy (nginx, Caddy, etc.) if serving over a network.

### Auto-discovery caution

Any `.py` file dropped in `ova/tools/` (that doesn't start with `_`) becomes a callable tool. Be careful about what code lives in that directory, especially in shared or production environments.

### External requests

Tools that call external APIs (web search, weather, etc.) can leak user query content to third parties. Review what data each tool sends externally and document it for your users.

### No filesystem/shell tools

Never create tools that execute shell commands, read/write arbitrary files, or query databases without strict input validation and allowlists. The LLM can be prompted to pass any argument — an unrestricted shell tool is a remote code execution vulnerability.

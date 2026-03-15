# OVA Security Overview

OVA is a **single-user, localhost voice assistant**. It is not designed for internet exposure. The default configuration is safe for local use — it binds to `localhost`, requires no authentication, and disables tools and MCP by default.

The frontend uses Python's built-in static file server, which has no TLS, no rate limiting, and no security headers. The Content Security Policy hardcodes `localhost` URLs. These are deliberate choices for a local-only application.

---

## Table of Contents

- [Security Defaults](#security-defaults)
- [Single-Tenant Limitations](#single-tenant-limitations)
- [Tool & MCP Safety](#tool--mcp-safety)
- [API-Only Mode (SDK / Headless)](#api-only-mode-sdk--headless)
- [Exposing OVA Beyond Localhost](#exposing-ova-beyond-localhost)
- [Configuration Reference](#configuration-reference)
- [Not Implemented](#not-implemented)
- [Reporting Vulnerabilities](#reporting-vulnerabilities)

---

## Security Defaults

Out of the box, OVA provides the following protections:

- **Localhost binding** — both the backend API and frontend bind to `localhost` only
- **Input size limits** — text input capped at 4096 characters, audio and image uploads capped at 20 MB each
- **Tools disabled** — LLM tool calling is off by default (`OVA_ENABLE_TOOLS=false`)
- **MCP disabled** — external MCP server connections are off by default (`OVA_ENABLE_MCP=false`)
- **ASR isolation** — speech recognition runs in a separate subprocess with restricted inter-process communication
- **Offline models** — `HF_HUB_OFFLINE=1` prevents model downloads after initial install
- **Settings validation** — the settings API rejects unknown values for TTS engine, language, and stream format via allowlists

---

## Single-Tenant Limitations

OVA is architecturally single-tenant. These are not bugs — they are inherent to the design:

- **One conversation context** shared across all connected clients
- **One TTS instance and one interrupt flag** — shared by all clients
- **Voice and language changes are global** — one client switching language affects everyone
- **System prompt changes affect all clients**

**For separate users, run separate OVA instances.**

---

## Tool & MCP Safety

### Tools

When `OVA_ENABLE_TOOLS=true`, the LLM can invoke Python functions during conversation.

**What to know:**

- **No sandbox** — tools run in the OVA process with full host access
- **Auto-discovery** — any `.py` file in `ova/tools/` becomes LLM-callable
- **Prompt injection risk** — crafted user input can trick the LLM into calling tools in unintended ways
- **External data exposure** — tools that make network requests (e.g., web search) can leak query content to third parties

**How to manage:**

| Variable | Purpose |
|----------|---------|
| `OVA_ENABLE_TOOLS` | Master toggle (default: `false`) |
| `OVA_DISABLED_TOOLS` | Comma-separated list of tool names to disable |
| `OVA_TOOL_<NAME>_ENABLED` | Per-tool override (`true`/`false`) |
| `OVA_MAX_TOOL_ITERATIONS` | Cap on LLM-tool round-trips per request (default: `5`) |

### MCP (Model Context Protocol)

When `OVA_ENABLE_MCP=true`, OVA connects to external MCP servers as a client.

**What to know:**

- MCP servers run as subprocesses with a filtered environment — OVA secrets are stripped automatically
- Failed MCP connections are logged but do not block startup
- On tool name collision, native OVA tools always take priority over MCP tools

**How to manage:**

| Variable | Purpose |
|----------|---------|
| `OVA_ENABLE_MCP` | Master toggle (default: `false`) |
| `OVA_MCP_CONFIG` | Path to MCP server config file (default: `mcp_servers.json`) |
| `OVA_MCP_CONNECT_TIMEOUT` | Per-server connection timeout in seconds (default: `10`) |
| `OVA_MCP_TOOL_TIMEOUT` | Per-tool call timeout in seconds (default: `30`) |

---

## API-Only Mode (SDK / Headless)

Setting `OVA_DISABLE_FRONTEND_ACCESS=true` skips launching the static file server (port 8080). The API (port 5173) remains fully exposed — this flag only removes the frontend, it does not add any security by itself.

**API-only mode still requires hardening:**

- `OVA_API_KEY` is mandatory — without it, all API endpoints are unauthenticated
- All endpoints remain accessible: chat, TTS, ASR, settings, restart, etc.
- The browser authentication bypass does not apply — SDK clients must always provide the API key
- Single-tenant limitations still apply — multiple SDK clients share context, voice state, and interrupt

**Recommended API-only settings:**

```
OVA_API_KEY=<strong-random-key>
OVA_DISABLE_RESTART_ENDPOINT=true
OVA_DISABLE_FRONTEND_SETTINGS=true
OVA_DISABLE_FRONTEND_ACCESS=true
```

API-only is the safer way to expose OVA on a network compared to exposing the frontend, but it still requires all hardening steps from the next section.

---

## Exposing OVA Beyond Localhost

> **This is discouraged.** OVA is designed for local use. If you expose it to a network, you do so at your own risk.

### Why you should not expose the frontend

- The frontend server is not production-grade — no TLS, no rate limiting, no security headers
- The Content Security Policy hardcodes `localhost` — WebSocket and fetch calls will break on other hostnames without manual HTML edits
- No built-in rate limiting
- The browser authentication bypass can be spoofed by non-browser clients

### Why API-only exposure also has risks

- No rate limiting — a client can flood the server with requests
- No per-user isolation — all clients share state
- No built-in TLS — requires a reverse proxy

### If you do it anyway — minimum required

1. **`OVA_API_KEY=<strong-random-key>`** — mandatory, without this everything is unauthenticated
2. **`OVA_DISABLE_RESTART_ENDPOINT=true`** — prevents remote server restarts
3. **`OVA_DISABLE_FRONTEND_SETTINGS=true`** — prevents remote configuration changes
4. **`OVA_DISABLE_FRONTEND_ACCESS=true`** — strongly recommended, skip the frontend and use API-only
5. **HTTPS via reverse proxy** — use nginx or Caddy for TLS termination
6. **Rate limiting** — use a reverse proxy (nginx `limit_req`, Caddy `rate_limit`) or application-level with [slowapi](https://github.com/laurentS/slowapi) (`pip install slowapi`)
7. **`OVA_DEBUG=false`** — this is the default, but verify it is not set to `true`
8. **Firewall to trusted IPs** — restrict network access to known clients

### Additional hardening

- `OVA_DISABLE_MULTIMODAL=true` — if vision/image input is not needed
- `OVA_ENABLE_TOOLS=false` and `OVA_ENABLE_MCP=false` — these are the defaults, keep them disabled unless needed
- Keep `HF_HUB_OFFLINE=1` — prevents the server from making outbound model download requests
- Audit the `ova/tools/` directory — guard against unauthorized file drops, since any Python file there becomes LLM-callable
- Never set `OVA_MAX_TEXT_LENGTH`, `OVA_MAX_AUDIO_SIZE`, or `OVA_MAX_IMAGE_SIZE` to `0` — this disables the limit entirely
- Review `OVA_MAX_CONTEXT_MESSAGES` — lower values reduce memory usage per session

---

## Configuration Reference

All security-relevant environment variables, grouped by category.

### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_API_KEY` | *(empty)* | API key for all endpoints. Empty = authentication disabled. |

### Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_DISABLE_RESTART_ENDPOINT` | `false` | Disable `POST /v1/restart` endpoint |
| `OVA_DISABLE_FRONTEND_SETTINGS` | `false` | Disable settings panel and `POST /v1/settings` |
| `OVA_DISABLE_MULTIMODAL` | `false` | Disable text/image input, restrict to voice-only |
| `OVA_DISABLE_FRONTEND_ACCESS` | `false` | API-only mode, skip static file server |

### Input Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_MAX_TEXT_LENGTH` | `4096` | Max text input in characters (0 = no limit) |
| `OVA_MAX_AUDIO_SIZE` | `20971520` | Max audio upload in bytes (20 MB, 0 = no limit) |
| `OVA_MAX_IMAGE_SIZE` | `20971520` | Max image upload in bytes (20 MB, 0 = no limit) |
| `OVA_MAX_CONTEXT_MESSAGES` | `50` | Conversation history sliding window (0 = unlimited) |

### Tools & MCP

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_ENABLE_TOOLS` | `false` | Enable LLM tool/function calling |
| `OVA_MAX_TOOL_ITERATIONS` | `5` | Max LLM-tool round-trips per request |
| `OVA_DISABLED_TOOLS` | *(empty)* | Comma-separated tool names to disable |
| `OVA_ENABLE_MCP` | `false` | Enable MCP server connections |
| `OVA_MCP_CONFIG` | `mcp_servers.json` | Path to MCP server config file |
| `OVA_MCP_CONNECT_TIMEOUT` | `10` | Per-server connection timeout (seconds) |
| `OVA_MCP_TOOL_TIMEOUT` | `30` | Per-tool call timeout (seconds) |

### Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_DEBUG` | `false` | Enable debug endpoints and verbose error messages |

---

## Not Implemented

The following are out of scope for OVA and would require external infrastructure:

- **Rate limiting** — use a reverse proxy (nginx `limit_req`, Caddy `rate_limit`) or application-level with [slowapi](https://github.com/laurentS/slowapi) for FastAPI-native rate limiting
- **Per-user session isolation** — OVA is single-tenant by design
- **Built-in TLS** — use a reverse proxy for HTTPS termination
- **WebSocket connection limits** — use a reverse proxy to cap concurrent connections

---

## Reporting Vulnerabilities

Report security issues via [GitHub Issues](https://github.com/rekuenkdr/ova/issues).

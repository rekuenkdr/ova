# MCP (Model Context Protocol) Client

OVA can act as an MCP **client**, connecting to external MCP servers to expand its tool capabilities. This lets you add filesystem access, databases, APIs, and more without writing Python code — just drop a server config in `mcp_servers.json`.

MCP tools work alongside OVA's native tool system (`ova/tools/*.py`). The LLM sees all tools (native + MCP) in a single list and can call any of them during conversation.

---

## Quick Start

1. Enable MCP and tools in `.env`:
   ```ini
   OVA_ENABLE_MCP=true
   OVA_ENABLE_TOOLS=true
   ```

2. Add a server to `mcp_servers.json`:
   ```json
   {
     "mcpServers": {
       "everything": {
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-everything"]
       }
     }
   }
   ```

3. Restart OVA.

4. Verify via the info endpoint:
   ```bash
   curl http://localhost:5173/v1/info | jq '{mcp_enabled, mcp_servers, tools_available}'
   ```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│ OVAPipeline                                              │
│                                                          │
│  ToolRegistry                                            │
│  ├── Native tools (ova/tools/*.py)     → direct call     │
│  └── MCP tools (from MCPClientManager) → session.call_tool│
│                                                          │
│  MCPClientManager (ova/mcp_client.py)                    │
│  ├── Background asyncio event loop thread                │
│  ├── Sessions: {server_name → ClientSession}             │
│  └── Tool map: {tool_name → server_name}                 │
│       ↓ stdio          ↓ SSE/HTTP                        │
│  ┌─────────┐    ┌──────────────┐                         │
│  │ Local    │    │ Remote MCP   │                         │
│  │ MCP srv  │    │ server       │                         │
│  └─────────┘    └──────────────┘                         │
└──────────────────────────────────────────────────────────┘
```

### How It Works

1. At startup, `OVAPipeline.__init__()` checks `OVA_ENABLE_MCP`
2. If enabled, `MCPClientManager` starts a background daemon thread running an asyncio event loop
3. All configured servers connect **in parallel** via `asyncio.gather()` — a slow server doesn't delay others. Each server has an individual timeout (`OVA_MCP_CONNECT_TIMEOUT`).
4. Each server gets a persistent `ClientSession` managed by `AsyncExitStack`
5. `session.list_tools()` discovers each server's tools
6. MCP tool schemas are converted to proper `ollama.Tool` objects and registered with `ToolRegistry`
7. When the LLM calls a tool, `ToolRegistry.execute()` checks native tools first, then routes to MCP via `session.call_tool()`
8. If a session dies, the next tool call triggers **auto-reconnect** before returning an error
9. Results are extracted as text and returned to the LLM

### Async/Sync Bridge

MCP sessions are async, but OVA's tool calling runs in sync threads (FastAPI threadpool). The bridge pattern:

```python
# Background thread runs: asyncio.run_forever()
# Sync callers use:
future = asyncio.run_coroutine_threadsafe(coro, self._loop)
result = future.result(timeout=30)
```

This is the same pattern OVA uses for EventBus (but in reverse direction).

---

## Configuration

### Config File (`mcp_servers.json`)

Uses the standard MCP config format:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"],
      "env": {
        "CUSTOM_VAR": "value"
      }
    },
    "remote-sse": {
      "url": "http://localhost:8081/sse",
      "transport": "sse"
    },
    "remote-http": {
      "url": "http://localhost:8082/mcp",
      "transport": "streamable-http"
    }
  }
}
```

#### Server Types

| Type | Config | Description |
|------|--------|-------------|
| **stdio** | `command` + `args` | Spawns a local subprocess, communicates over stdin/stdout |
| **SSE** | `url` + `transport: "sse"` | Connects to a remote SSE-based MCP server |
| **Streamable HTTP** | `url` + `transport: "streamable-http"` | Connects to a remote Streamable HTTP MCP server (recommended for new HTTP servers) |

#### Server Config Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `command` | string | For stdio | Executable to run (e.g., `"npx"`, `"python"`) |
| `args` | list[string] | No | Command-line arguments |
| `env` | object | No | Extra environment variables (merged into a **filtered** base env — OVA secrets are stripped, see Security) |
| `url` | string | For remote | Server URL |
| `transport` | string | No | `"sse"` (default for url) or `"streamable-http"` |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_ENABLE_MCP` | `false` | Master toggle — must be `true` for MCP to initialize |
| `OVA_MCP_CONFIG` | `mcp_servers.json` | Path to config file (relative to project root or absolute) |
| `OVA_MCP_CONNECT_TIMEOUT` | `10` | Per-server connection timeout in seconds |
| `OVA_MCP_TOOL_TIMEOUT` | `30` | Per-tool call timeout in seconds |

### Prerequisites

- `OVA_ENABLE_TOOLS=true` must also be set (MCP tools use the same tool calling pipeline)
- For stdio servers: the `command` must be available in `PATH` (e.g., `npx` requires Node.js)
- For remote servers: the server must be running and reachable

---

## Tool Priority and Name Collisions

When both native and MCP tools exist:

1. **Native tools always win** — if a native tool and an MCP tool have the same name, the native tool is used and a warning is logged
2. **First MCP server wins** — if two MCP servers provide tools with the same name, the first server (by config order) takes priority
3. **Tool names** appear in a single flat list in `GET /v1/info` → `tools_available`

---

## API Integration

### `/v1/info` Response

When MCP is active, the info endpoint includes:

```json
{
  "tools_enabled": true,
  "tools_available": ["get_current_datetime", "set_timer", "mcp_filesystem_read", "..."],
  "mcp_enabled": true,
  "mcp_servers": {
    "filesystem": {
      "status": "connected",
      "tools": ["read_file", "write_file", "list_directory"],
      "error": null
    },
    "broken-server": {
      "status": "error",
      "tools": [],
      "error": "Connection refused"
    }
  }
}
```

### Tool Execution Flow

```
User: "Read the file /tmp/notes.txt"
  │
  ▼
LLM (Ollama) receives tools=[native_tools + MCP_tools]
  │
  ▼
LLM calls tool: read_file(path="/tmp/notes.txt")
  │
  ▼
ToolRegistry.execute("read_file", {"path": "/tmp/notes.txt"})
  │
  ├── Not in native tools
  │
  ▼
MCPClientManager.call_tool("read_file", {"path": "/tmp/notes.txt"})
  │
  ▼
run_coroutine_threadsafe → session.call_tool() on background loop
  │
  ▼
MCP server processes request, returns CallToolResult
  │
  ▼
_extract_text_content() → "Contents of notes.txt..."
  │
  ▼
LLM incorporates result into spoken response
```

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| MCP config file missing | Warning logged, MCP skipped, OVA starts normally |
| Server fails to connect at startup | Warning logged, server marked as "error", other servers unaffected (parallel connect) |
| Tool call times out | Returns `"Error: MCP tool '...' timed out after 30s"` to LLM |
| Server crashes during tool call | Returns error string to LLM, session marked dead for auto-reconnect on next call |
| Dead session on next tool call | Automatic reconnect attempt via `_try_reconnect()` using stored server configs |
| `OVA_ENABLE_MCP=false` (default) | No MCP initialization at all, zero overhead |

All errors are **non-fatal** — OVA starts and serves requests even if all MCP servers fail.

### Auto-Reconnect

When an MCP session fails during a tool call (connection dropped, server crashed), the manager:

1. Removes the dead session and marks the server status as `"error"`
2. On the **next** tool call to that server, detects the missing session
3. Attempts a fresh reconnect using the stored server config
4. If reconnect succeeds, the tool call proceeds normally
5. If reconnect fails, returns an error string to the LLM

This handles transient failures (server restarts, network blips) without requiring an OVA restart.

---

## Key Files

| File | Role |
|------|------|
| `ova/mcp_client.py` | `MCPClientManager` — background loop, sessions, schema conversion |
| `ova/tools/__init__.py` | `ToolRegistry` — `register_mcp()`, routes `execute()` to MCP |
| `ova/pipeline.py` | Init MCP after tool discovery, cleanup on shutdown |
| `ova/api.py` | Exposes MCP status in `GET /v1/info` |
| `mcp_servers.json` | Server configuration (empty by default) |
| `tests/test_mcp_client.py` | Unit tests |

---

## Examples

### Filesystem Server

Access local files from conversations:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/documents"]
    }
  }
}
```

### Everything Server (Testing)

A test server that exposes a variety of MCP features:

```json
{
  "mcpServers": {
    "everything": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-everything"]
    }
  }
}
```

### Multiple Servers

Connect to several servers at once:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
      }
    }
  }
}
```

### Remote SSE Server

Connect to a remote MCP server over HTTP/SSE:

```json
{
  "mcpServers": {
    "remote-api": {
      "url": "http://192.168.1.100:8081/sse",
      "transport": "sse"
    }
  }
}
```

---

## Testing

Run the MCP client tests:

```bash
pytest tests/test_mcp_client.py -v
```

Tests cover (28 tests):
- Config loading (valid, missing, empty, invalid JSON, env var override)
- Environment filtering (`_build_safe_env` — strips secrets, preserves safe vars, merges extra env)
- Schema conversion (MCP Tool → `ollama.Tool` instances, property types, enums)
- Content extraction (text, multiple blocks, errors)
- Registry integration (tool names, routing, collision priority, enabled state)
- Auto-reconnect (dead session handling, server config storage)
- Default disabled state

---

## Security

### Environment Filtering

Stdio MCP servers are spawned as subprocesses that inherit the parent's environment. OVA uses `_build_safe_env()` to prevent accidental secret leakage:

- **Stripped**: All `OVA_*` environment variables except explicitly safe ones
- **Preserved**: `OVA_LANGUAGE`, `OVA_DEBUG` (non-secret operational config)
- **Always stripped**: `OVA_API_KEY`, `OVA_SEARCH_API_KEY` (explicit secret vars)
- **Merged**: Server-specific `env` from config is applied on top (intentional — user chose to pass these)

Non-`OVA_` system variables (`PATH`, `HOME`, `LANG`, etc.) are passed through normally.

### MCP-Specific Risks

- **MCP servers have host access** — stdio servers run as local subprocesses with the same permissions as OVA. A malicious server can read/write files, access the network, etc.
- **Tool arguments come from the LLM** — the LLM decides what arguments to pass. An MCP server should validate its own inputs, but you're trusting both the LLM and the server.
- **Environment variables in config** — `env` values in `mcp_servers.json` may contain secrets (API keys, tokens). Don't commit this file with real credentials. OVA's base environment is filtered (see above), but server-specific `env` values are passed as-is.
- **Remote servers** — SSE/HTTP MCP connections send tool call data over the network. Use HTTPS for remote servers.
- **Config file security** — anyone who can edit `mcp_servers.json` can add arbitrary MCP servers (including malicious ones). Protect this file in shared environments.

### Recommendations

1. Only configure MCP servers you trust
2. Use `OVA_MCP_TOOL_TIMEOUT` to prevent long-running tool calls from blocking responses
3. Keep `OVA_MAX_TOOL_ITERATIONS` low to limit cascading tool calls
4. Store secrets in environment variables, not directly in `mcp_servers.json`
5. For production, consider running MCP servers in containers with restricted permissions

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| MCP tools not appearing in `/v1/info` | `OVA_ENABLE_MCP` or `OVA_ENABLE_TOOLS` not set | Set both to `true` in `.env` |
| Server shows "error" status | Connection failed at startup | Check server command/URL, increase `OVA_MCP_CONNECT_TIMEOUT` |
| Tool calls timing out | Server is slow or unresponsive | Increase `OVA_MCP_TOOL_TIMEOUT` |
| `npx` command not found | Node.js not installed | Install Node.js for stdio-based MCP servers |
| Config file not found | Wrong path | Check `OVA_MCP_CONFIG` or verify `mcp_servers.json` exists in project root |
| Tool name collision warning | Native and MCP tool share a name | Rename the native tool or accept that native wins |
| "MCP init failed (non-fatal)" | Config parse error or other startup issue | Check logs for details, fix config |

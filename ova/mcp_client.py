"""MCP (Model Context Protocol) client manager for OVA.

Connects to external MCP servers (stdio or SSE) and exposes their tools
alongside OVA's native tool registry. Uses a dedicated background asyncio
event loop thread so that long-lived async MCP sessions can coexist with
OVA's sync tool-calling code.

Environment variables:
    OVA_ENABLE_MCP: Master toggle (default: "false")
    OVA_MCP_CONFIG: Config file path (default: "mcp_servers.json")
    OVA_MCP_CONNECT_TIMEOUT: Per-server connect timeout in seconds (default: 10)
    OVA_MCP_TOOL_TIMEOUT: Per-tool call timeout in seconds (default: 30)
"""

import asyncio
import json
import os
import re
import threading
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from ollama import Tool as OllamaTool

from .utils import get_logger

logger = get_logger("mcp")

# Defaults
_DEFAULT_CONFIG = "mcp_servers.json"
_DEFAULT_CONNECT_TIMEOUT = 10
_DEFAULT_TOOL_TIMEOUT = 30
_RECONNECT_COOLDOWN = 60  # seconds between reconnect attempts per server

# Valid tool name pattern — alphanumeric, underscores, hyphens, max 128 chars
_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")

# System env vars safe to pass to MCP subprocesses (allowlist approach)
_SAFE_SYSTEM_ENV_KEYS = {
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL",
    "LC_CTYPE", "TMPDIR", "TMP", "TEMP", "XDG_RUNTIME_DIR",
    "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",  # TLS
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",  # proxy config
    "http_proxy", "https_proxy", "no_proxy",
}
# OVA env vars safe to pass through (non-secret operational config)
_SAFE_OVA_PREFIXES = ("OVA_LANGUAGE", "OVA_DEBUG")
# OVA secrets to strip even if extra_env injects them
_SECRET_OVA_KEYS = ("OVA_API_KEY", "OVA_SEARCH_API_KEY", "OVA_LLM_API_KEY")


def _build_safe_env(extra_env: dict | None = None) -> dict[str, str]:
    """Build a filtered environment for stdio subprocesses.

    Uses an allowlist approach: only passes known-safe system env vars
    and safe OVA vars. All other env vars (HF_TOKEN, AWS_SECRET_ACCESS_KEY,
    OPENAI_API_KEY, etc.) are dropped. MCP servers that need specific env
    vars should declare them in mcp_servers.json's "env" field.

    Args:
        extra_env: Additional env vars from the server config to merge in.

    Returns:
        Filtered environment dict safe for subprocess use.
    """
    env = {}
    for key, value in os.environ.items():
        if key in _SAFE_SYSTEM_ENV_KEYS:
            env[key] = value
        elif key.startswith("OVA_") and any(key.startswith(p) for p in _SAFE_OVA_PREFIXES):
            env[key] = value
        # Everything else is dropped
    # Merge server-specific env (these are intentional — user chose to pass them)
    if extra_env:
        env.update(extra_env)
    # Final safety: strip OVA secrets even if extra_env injected them
    for secret in _SECRET_OVA_KEYS:
        env.pop(secret, None)
    return env


def _validate_mcp_url(url: str) -> None:
    """Block MCP URLs targeting cloud metadata endpoints or link-local addresses.

    Raises ValueError for SSRF-prone targets. Legitimate MCP servers
    (localhost, real hostnames) are unaffected.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")

    # Block cloud metadata endpoints
    _BLOCKED_HOSTS = {"169.254.169.254", "metadata.google.internal"}
    if hostname in _BLOCKED_HOSTS:
        raise ValueError(f"MCP URL blocked (cloud metadata endpoint): {hostname}")

    # Block link-local IP range (169.254.0.0/16)
    try:
        import ipaddress
        addr = ipaddress.ip_address(hostname)
        if addr.is_link_local:
            raise ValueError(f"MCP URL blocked (link-local address): {hostname}")
    except ValueError as e:
        if "blocked" in str(e):
            raise
        # Not a valid IP — it's a hostname, which is fine


def load_mcp_config(config_path: str | None = None) -> dict:
    """Load MCP server configuration from JSON file.

    Args:
        config_path: Path to config file. If relative, resolved from project root.

    Returns:
        Parsed config dict with "mcpServers" key.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        json.JSONDecodeError: If config file is invalid JSON.
    """
    path_str = config_path or os.getenv("OVA_MCP_CONFIG", _DEFAULT_CONFIG)
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / path
    if not path.exists():
        raise FileNotFoundError(f"MCP config not found: {path}")
    with open(path) as f:
        config = json.load(f)
    if "mcpServers" not in config:
        config["mcpServers"] = {}
    return config


def _mcp_tool_to_ollama(tool) -> OllamaTool:
    """Convert an MCP Tool object to a proper ollama.Tool instance.

    Uses ollama.Tool class directly instead of raw dicts to ensure
    compatibility with Ollama's _copy_tools() validation.

    Args:
        tool: An mcp.types.Tool object.

    Returns:
        ollama.Tool instance.
    """
    input_schema = tool.inputSchema if hasattr(tool, "inputSchema") else (tool.input_schema or {})
    properties_raw = input_schema.get("properties", {})

    properties = {}
    for prop_name, prop_def in properties_raw.items():
        prop_type = prop_def.get("type", "string")
        # ollama.Tool.Function.Parameters.Property.type expects str or Sequence[str]
        properties[prop_name] = OllamaTool.Function.Parameters.Property(
            type=prop_type,
            description=prop_def.get("description", ""),
            enum=prop_def.get("enum"),
        )

    return OllamaTool(
        type="function",
        function=OllamaTool.Function(
            name=tool.name,
            description=tool.description or "",
            parameters=OllamaTool.Function.Parameters(
                type="object",
                required=input_schema.get("required"),
                properties=properties if properties else None,
            ),
        ),
    )


def _extract_text_content(result) -> str:
    """Extract text from a CallToolResult, joining multiple content blocks.

    Args:
        result: An mcp.types.CallToolResult object.

    Returns:
        String representation of the result content.
    """
    from mcp import types

    parts = []
    for block in result.content:
        if isinstance(block, types.TextContent):
            parts.append(block.text)
        elif isinstance(block, types.ImageContent):
            parts.append(f"[image: {block.mimeType}]")
        elif isinstance(block, types.AudioContent):
            parts.append(f"[audio: {block.mimeType}]")
        else:
            parts.append(f"[{block.type} content]")

    text = "\n".join(parts)
    if result.isError:
        text = f"Error: {text}"
    return text


class MCPClientManager:
    """Manages connections to MCP servers and bridges async sessions to sync callers.

    Starts a dedicated daemon thread running an asyncio event loop. MCP sessions
    (long-lived async context managers) live on this loop. Sync callers use
    ``run_coroutine_threadsafe`` to invoke async operations.
    """

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._sessions: dict[str, Any] = {}          # server_name → ClientSession
        self._tool_map: dict[str, str] = {}           # tool_name → server_name
        self._tools: dict[str, Any] = {}              # tool_name → MCP Tool object
        self._status: dict[str, dict] = {}            # server_name → {status, tools, error}
        self._server_configs: dict[str, dict] = {}    # server_name → config (for reconnect)
        self._last_reconnect: dict[str, float] = {}  # server_name → monotonic timestamp
        self._lock = threading.Lock()                 # protects shared state
        self._connect_timeout = max(1, min(int(os.getenv("OVA_MCP_CONNECT_TIMEOUT", _DEFAULT_CONNECT_TIMEOUT)), 300))
        self._tool_timeout = max(1, min(int(os.getenv("OVA_MCP_TOOL_TIMEOUT", _DEFAULT_TOOL_TIMEOUT)), 600))

    def start(self):
        """Start the background asyncio event loop thread."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="mcp-event-loop")
        self._thread.start()
        # Initialize exit stack on the background loop
        future = asyncio.run_coroutine_threadsafe(self._init_exit_stack(), self._loop)
        future.result(timeout=5)

    def _run_loop(self):
        """Run the asyncio event loop forever (daemon thread entry point)."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _init_exit_stack(self):
        """Initialize the async exit stack on the background loop."""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

    def connect_all(self, config: dict):
        """Connect to all servers defined in config concurrently (sync wrapper).

        Uses asyncio.gather() so servers connect in parallel — a slow server
        doesn't delay others.

        Args:
            config: Parsed MCP config dict with "mcpServers" key.
        """
        servers = config.get("mcpServers", {})
        if not servers:
            logger.info("No MCP servers configured")
            return

        # Store configs for reconnect
        self._server_configs = dict(servers)

        # Connect all servers in parallel
        future = asyncio.run_coroutine_threadsafe(
            self._connect_all_async(servers), self._loop
        )
        # Total timeout: per-server timeout + buffer for gather overhead
        total_timeout = self._connect_timeout + 10
        try:
            future.result(timeout=total_timeout)
        except Exception as e:
            logger.warning(f"MCP connect_all failed: {e}")

        total = len(self._tools)
        connected = sum(1 for s in self._status.values() if s["status"] == "connected")
        logger.info(f"MCP: {connected}/{len(servers)} servers connected, {total} tools available")

    async def _connect_all_async(self, servers: dict):
        """Connect to all servers concurrently via asyncio.gather."""
        tasks = [
            self._connect_server_safe(name, config)
            for name, config in servers.items()
        ]
        await asyncio.gather(*tasks)

    async def _connect_server_safe(self, name: str, config: dict):
        """Connect to a server with error handling (won't propagate exceptions)."""
        try:
            await asyncio.wait_for(
                self._connect_server(name, config),
                timeout=self._connect_timeout,
            )
        except Exception as e:
            logger.warning(f"MCP server '{name}' failed to connect: {e}")
            self._status[name] = {"status": "error", "tools": [], "error": str(e)}

    async def _connect_server(self, name: str, config: dict):
        """Connect to a single MCP server and discover its tools.

        Args:
            name: Server name (from config key).
            config: Server config dict (command/args/env or url/transport).
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if "command" in config:
            # stdio transport — use filtered env to prevent secret leakage
            server_params = StdioServerParameters(
                command=config["command"],
                args=config.get("args", []),
                env=_build_safe_env(config.get("env")),
            )
            transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
        elif "url" in config:
            _validate_mcp_url(config["url"])
            transport_type = config.get("transport", "sse")
            if transport_type == "sse":
                from mcp.client.sse import sse_client
                transport = await self._exit_stack.enter_async_context(
                    sse_client(config["url"], timeout=self._connect_timeout)
                )
            else:
                from mcp.client.streamable_http import streamable_http_client
                transport = await self._exit_stack.enter_async_context(
                    streamable_http_client(config["url"])
                )
        else:
            raise ValueError(f"Server '{name}': must specify 'command' (stdio) or 'url' (sse/http)")

        read_stream, write_stream = transport
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await asyncio.wait_for(session.initialize(), timeout=self._connect_timeout)

        # Discover tools
        result = await session.list_tools()
        tool_names = []
        for tool in result.tools:
            # Validate tool name — reject injection attempts
            if not _TOOL_NAME_RE.match(tool.name):
                logger.warning(
                    f"MCP server '{name}': tool name '{tool.name[:64]}' contains "
                    f"invalid characters — skipping (allowed: a-z, A-Z, 0-9, _, -)"
                )
                continue
            if tool.name in self._tool_map:
                existing_server = self._tool_map[tool.name]
                logger.warning(
                    f"MCP tool name collision: '{tool.name}' from '{name}' "
                    f"already registered by '{existing_server}' — skipping"
                )
                continue
            self._tool_map[tool.name] = name
            self._tools[tool.name] = tool
            tool_names.append(tool.name)

        self._sessions[name] = session
        self._status[name] = {"status": "connected", "tools": tool_names, "error": None}
        logger.info(f"MCP server '{name}': connected, {len(tool_names)} tools: {tool_names}")

    def _try_reconnect(self, server_name: str) -> bool:
        """Attempt to reconnect a failed MCP server (sync wrapper).

        Enforces a 60-second cooldown per server to prevent reconnect storms.
        Caller must hold self._lock.

        Returns True if reconnection succeeded.
        """
        config = self._server_configs.get(server_name)
        if not config:
            return False

        # Cooldown: skip if we tried recently
        now = time.monotonic()
        last = self._last_reconnect.get(server_name, 0)
        if now - last < _RECONNECT_COOLDOWN:
            logger.debug(f"MCP server '{server_name}': reconnect cooldown ({_RECONNECT_COOLDOWN - (now - last):.0f}s remaining)")
            return False
        self._last_reconnect[server_name] = now

        logger.info(f"MCP server '{server_name}': attempting reconnect...")

        # Remove stale tool mappings for this server
        stale_tools = [name for name, srv in self._tool_map.items() if srv == server_name]
        for tool_name in stale_tools:
            self._tool_map.pop(tool_name, None)
            self._tools.pop(tool_name, None)
        self._sessions.pop(server_name, None)

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._connect_server_safe(server_name, config), self._loop
            )
            future.result(timeout=self._connect_timeout + 5)
            return self._status.get(server_name, {}).get("status") == "connected"
        except Exception as e:
            logger.warning(f"MCP server '{server_name}' reconnect failed: {e}")
            return False

    def call_tool(self, name: str, arguments: dict | None = None) -> str:
        """Call an MCP tool by name (sync wrapper).

        If the session is dead, attempts one reconnect before failing.
        Thread-safe — uses self._lock to protect shared state.

        Args:
            name: Tool name.
            arguments: Keyword arguments for the tool.

        Returns:
            String result from the tool, or error message.
        """
        with self._lock:
            server_name = self._tool_map.get(name)
            if not server_name:
                return f"Error: unknown MCP tool '{name}'"

            session = self._sessions.get(server_name)
            if not session:
                # Server was never connected or lost — try reconnect
                if self._try_reconnect(server_name):
                    session = self._sessions.get(server_name)
                if not session:
                    return f"Error: MCP server '{server_name}' not connected"

        try:
            future = asyncio.run_coroutine_threadsafe(
                session.call_tool(name, arguments or {}), self._loop
            )
            result = future.result(timeout=self._tool_timeout)
            return _extract_text_content(result)
        except TimeoutError:
            return f"Error: MCP tool '{name}' timed out after {self._tool_timeout}s"
        except Exception as e:
            logger.warning(f"MCP tool '{name}' failed: {e}")
            # Session may be dead — mark for reconnect on next call
            with self._lock:
                self._sessions.pop(server_name, None)
                self._status[server_name] = {
                    "status": "error",
                    "tools": self._status.get(server_name, {}).get("tools", []),
                    "error": str(e),
                }
            return f"Error calling MCP tool '{name}': {e}"

    def get_ollama_tools(self) -> list[OllamaTool]:
        """Convert all MCP tools to Ollama Tool objects.

        Returns:
            List of ollama.Tool instances.
        """
        return [_mcp_tool_to_ollama(tool) for tool in self._tools.values()]

    def get_tool_names(self) -> list[str]:
        """Return sorted list of available MCP tool names."""
        return sorted(self._tools.keys())

    def get_status(self) -> dict:
        """Return connection status for all configured servers."""
        return dict(self._status)

    def shutdown(self):
        """Close all MCP sessions and stop the background event loop."""
        if self._loop and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._exit_stack.aclose(), self._loop
                )
                future.result(timeout=10)
            except Exception as e:
                logger.warning(f"MCP shutdown error: {e}")
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        self._sessions.clear()
        self._tool_map.clear()
        self._tools.clear()
        self._status.clear()
        self._server_configs.clear()
        self._last_reconnect.clear()
        logger.info("MCP client manager shut down")

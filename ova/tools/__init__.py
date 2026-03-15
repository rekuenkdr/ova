"""Modular tool/function calling system for OVA.

Discovers tool functions from sibling modules and provides them to Ollama's
native tool calling interface. Each tool is a plain Python function with a
Google-style docstring containing an ``Args:`` section.

Environment variables:
    OVA_ENABLE_TOOLS: Master toggle (default: "false")
    OVA_MAX_TOOL_ITERATIONS: Max tool-call round-trips per request (default: 5)
    OVA_DISABLED_TOOLS: Comma-separated function names to disable
    OVA_TOOL_<NAME>_ENABLED: Per-tool override ("true"/"false")
"""
import importlib
import inspect
import os
import pkgutil
from pathlib import Path
from typing import Callable

from ..utils import get_logger

logger = get_logger("tools")

TOOLS_ENABLED = os.getenv("OVA_ENABLE_TOOLS", "false").lower() == "true"
MAX_TOOL_ITERATIONS = int(os.getenv("OVA_MAX_TOOL_ITERATIONS", "5"))
DISABLED_TOOLS = {
    name.strip()
    for name in os.getenv("OVA_DISABLED_TOOLS", "").split(",")
    if name.strip()
}


def _has_google_docstring(func: Callable) -> bool:
    """Check if a function has a Google-style docstring with Args section."""
    doc = inspect.getdoc(func)
    return doc is not None and "Args:" in doc


class ToolRegistry:
    """Registry of tool functions discovered from ``ova/tools/*.py`` and MCP servers."""

    def __init__(self):
        self._functions: dict[str, Callable] = {}
        self._mcp_manager = None  # MCPClientManager, set via register_mcp()

    def discover(self):
        """Scan ``ova/tools/*.py`` and register qualifying public functions.

        A function qualifies if it:
        - Does not start with ``_``
        - Has a Google-style docstring (contains ``Args:``)
        - Is not in the disabled list / per-tool env override

        Skips entirely when ``OVA_ENABLE_TOOLS`` is false.
        """
        if not TOOLS_ENABLED:
            logger.debug("Tools disabled (OVA_ENABLE_TOOLS=false)")
            return
        package_dir = Path(__file__).parent
        for module_info in pkgutil.iter_modules([str(package_dir)]):
            if module_info.name.startswith("_"):
                continue
            try:
                module = importlib.import_module(f".{module_info.name}", package=__package__)
            except Exception as exc:
                logger.warning(f"Failed to import tool module {module_info.name}: {exc}")
                continue

            # Module-level default enable flag
            module_default = getattr(module, "TOOL_ENABLED_DEFAULT", True)

            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if name.startswith("_"):
                    continue
                if obj.__module__ != module.__name__:
                    continue
                if not _has_google_docstring(obj):
                    continue

                # Check per-tool env override (OVA_TOOL_<NAME>_ENABLED)
                env_key = f"OVA_TOOL_{name.upper()}_ENABLED"
                env_val = os.getenv(env_key)
                if env_val is not None:
                    if env_val.lower() != "true":
                        logger.debug(f"Tool {name} disabled via {env_key}")
                        continue
                elif name in DISABLED_TOOLS:
                    logger.debug(f"Tool {name} disabled via OVA_DISABLED_TOOLS")
                    continue
                elif not module_default:
                    logger.debug(f"Tool {name} disabled (module default)")
                    continue

                self._functions[name] = obj
                logger.debug(f"Registered tool: {name}")

        if self._functions:
            logger.info(f"Discovered {len(self._functions)} tools: {sorted(self._functions)}")
        else:
            logger.debug("No tools discovered")

    def register_mcp(self, manager):
        """Register an MCPClientManager to provide additional tools.

        Args:
            manager: An MCPClientManager instance with connected sessions.
        """
        self._mcp_manager = manager
        mcp_names = manager.get_tool_names()
        # Warn about collisions (native tools always win)
        for name in mcp_names:
            if name in self._functions:
                logger.warning(f"MCP tool '{name}' shadowed by native tool — native wins")
        if mcp_names:
            logger.info(f"MCP tools registered: {mcp_names}")

    @property
    def enabled(self) -> bool:
        """True if the master toggle is on AND at least one tool is available."""
        has_native = bool(self._functions)
        has_mcp = self._mcp_manager is not None and bool(self._mcp_manager.get_tool_names())
        return TOOLS_ENABLED and (has_native or has_mcp)

    def get_enabled_functions(self) -> list:
        """Return list of tools for ``ollama.chat(tools=...)``.

        Native tools are returned as callables. MCP tools are returned as
        Ollama-compatible dicts. Ollama's ``_copy_tools`` handles both formats.
        """
        if not TOOLS_ENABLED:
            return []
        tools = list(self._functions.values())
        if self._mcp_manager:
            tools.extend(self._mcp_manager.get_ollama_tools())
        return tools

    def get_tool_names(self) -> list[str]:
        """Return sorted list of all registered tool names (native + MCP)."""
        names = set(self._functions.keys())
        if self._mcp_manager:
            names.update(self._mcp_manager.get_tool_names())
        return sorted(names)

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Call a registered tool function and return its string result.

        Native tools are checked first (always win on name collision).
        Falls back to MCP tools if available.

        Args:
            tool_name: The function name.
            arguments: Keyword arguments to pass to the function.

        Returns:
            String result from the tool, or an error message.
        """
        # Native tools first
        func = self._functions.get(tool_name)
        if func is not None:
            try:
                result = func(**arguments)
                return str(result)
            except Exception as exc:
                logger.warning(f"Tool {tool_name} failed: {exc}")
                return f"Error: {tool_name} failed"

        # MCP tools
        if self._mcp_manager and tool_name in self._mcp_manager.get_tool_names():
            return self._mcp_manager.call_tool(tool_name, arguments)

        return f"Error: unknown tool '{tool_name}'"


# Module-level singleton
registry = ToolRegistry()

"""Web search tool via Tavily API.

Requires OVA_SEARCH_API_KEY set to a Tavily API key (https://app.tavily.com).
Disabled by default — enable with OVA_TOOL_WEB_SEARCH_ENABLED=true.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

TOOL_ENABLED_DEFAULT = False

_API_URL = "https://api.tavily.com/search"
_MAX_QUERY_LENGTH = 400
_DEFAULT_MAX_RESULTS = 5
_TIMEOUT = 15  # seconds


def web_search(query: str) -> str:
    """Search the web for current, real-time, or recent information using Tavily.

    Only use this tool when the user asks about something you cannot answer from
    your own knowledge: breaking news, live scores, current prices, today's weather,
    recent events, or anything that changes frequently. Do NOT search for general
    knowledge, historical facts, definitions, or anything you already know well.

    Args:
        query: The search query string.

    Returns:
        str: Formatted search results or an error message.
    """
    api_key = os.getenv("OVA_SEARCH_API_KEY", "")
    if not api_key:
        return "Web search is not configured. Set OVA_SEARCH_API_KEY to enable it."

    # Sanitize query — cap length, strip control characters
    query = query.strip()
    if not query:
        return "Error: empty search query."
    query = "".join(c for c in query if c.isprintable())
    if len(query) > _MAX_QUERY_LENGTH:
        query = query[:_MAX_QUERY_LENGTH]

    max_results = max(1, min(
        int(os.getenv("OVA_SEARCH_MAX_RESULTS", _DEFAULT_MAX_RESULTS)), 20
    ))

    payload = json.dumps({
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": True,
    }).encode()

    req = urllib.request.Request(
        _API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return "Error: invalid Tavily API key."
        if e.code == 429:
            return "Error: Tavily rate limit exceeded. Try again later."
        return f"Error: Tavily search failed (HTTP {e.code})."
    except (urllib.error.URLError, TimeoutError):
        return "Error: could not reach Tavily search API."

    return _format_results(data)


def _domain_from_url(url: str) -> str:
    """Extract bare domain from a URL (e.g., 'https://en.wikipedia.org/wiki/...' -> 'wikipedia.org')."""
    try:
        host = urllib.parse.urlparse(url).hostname or ""
        # Strip leading 'www.' and common subdomains for cleaner source attribution
        parts = host.split(".")
        if len(parts) > 2 and parts[0] in ("www", "en", "es", "fr", "de", "m", "mobile"):
            parts = parts[1:]
        return ".".join(parts)
    except Exception:
        return ""


def _format_results(data: dict) -> str:
    """Format Tavily response for voice-first LLM consumption.

    Optimized for spoken output: no full URLs, summary first, source domains only.
    """
    parts = []

    answer = data.get("answer")
    if answer:
        parts.append(f"Summary: {answer}")

    results = data.get("results", [])
    if results:
        parts.append("")
        parts.append("Sources:")
        for r in results:
            title = r.get("title", "Untitled")
            domain = _domain_from_url(r.get("url", ""))
            content = r.get("content", "")
            source = f" ({domain})" if domain else ""
            snippet = f": {content}" if content else ""
            parts.append(f"- {title}{source}{snippet}")

    if not parts:
        return "No results found."

    return "\n".join(parts)

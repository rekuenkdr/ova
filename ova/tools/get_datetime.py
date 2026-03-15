"""Date and time tool."""
import re
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo, available_timezones

import dateparser

from ._base import get_pipeline_timezone

# Patterns that dateparser can't handle but map to PREFER_DATES_FROM
_NEXT_RE = re.compile(r"^(next|this|coming)\s+", re.IGNORECASE)
_LAST_RE = re.compile(r"^(last|past|previous)\s+", re.IGNORECASE)

TOOL_ENABLED_DEFAULT = True


def _normalize(s: str) -> str:
    """Strip accents/diacritics and lowercase."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


# Build lookup once: normalized city part → full IANA zone name
_CITY_TO_ZONE: dict[str, str] = {}
for _tz in available_timezones():
    # IANA zones like "America/New_York" → key on "new york"
    if "/" in _tz:
        _city = _normalize(_tz.rsplit("/", 1)[-1].replace("_", " "))
        _CITY_TO_ZONE[_city] = _tz


def _find_timezone(location: str) -> str | None:
    """Find an IANA timezone for a city/location name. Returns zone name or None."""
    loc = _normalize(location.strip())

    # Direct IANA city match
    if loc in _CITY_TO_ZONE:
        return _CITY_TO_ZONE[loc]

    # Partial match: "york" matches "new york" → "America/New_York"
    for city, tz_name in _CITY_TO_ZONE.items():
        if loc in city or city in loc:
            return tz_name

    return None


def get_current_datetime(query: str | None = None, location: str | None = None) -> str:
    """Get the current date and time, or resolve a natural-language date expression.

    Call this tool for ANY question about dates, times, days of the week, or
    "what day is it?". Examples of when to call:
      - "what time is it?" → call with no query
      - "what's today's date?" → call with no query
      - "what day is today?" → call with no query
      - "what was yesterday?" → call with query="yesterday"
      - "when is next Friday?" → call with query="next Friday"
      - "3 days ago" → call with query="3 days ago"
      - "what time is it in Tokyo?" → call with location="Tokyo"
      - "hora en Londres" → call with location="London"

    Args:
        query: Natural language date expression (e.g. "yesterday", "next Tuesday"). Leave empty for current date and time.
        location: City name in English for timezone lookup (e.g. "Tokyo", "Lisbon", "New York").
                  Always translate to the English city name, even if the user speaks another language.

    Returns:
        str: The current date and time, or the resolved date.
    """
    # Location-specific time
    if location:
        tz_name = _find_timezone(location)
        if tz_name:
            tz = ZoneInfo(tz_name)
            now = datetime.now(tz)
            tz_abbr = now.strftime("%Z")
            return now.strftime(f"%A, %B %d, %Y at %H:%M {tz_abbr} ({location})")
        return f"Unknown location: '{location}'"

    tz = ZoneInfo(get_pipeline_timezone())
    now = datetime.now(tz)
    if not query:
        tz_abbr = now.strftime("%Z")
        return now.strftime(f"%A, %B %d, %Y at %H:%M {tz_abbr}")
    settings = {
        "TIMEZONE": get_pipeline_timezone(),
        "RETURN_AS_TIMEZONE_AWARE": True,
    }
    parsed = dateparser.parse(query, settings=settings)
    # Fallback: strip "next/last" prefix and use PREFER_DATES_FROM
    if parsed is None and _NEXT_RE.match(query):
        stripped = _NEXT_RE.sub("", query)
        settings["PREFER_DATES_FROM"] = "future"
        parsed = dateparser.parse(stripped, settings=settings)
    elif parsed is None and _LAST_RE.match(query):
        stripped = _LAST_RE.sub("", query)
        settings["PREFER_DATES_FROM"] = "past"
        parsed = dateparser.parse(stripped, settings=settings)
    # Fallback: try as location/timezone if dateparser can't parse
    if parsed is None:
        tz_name = _find_timezone(query)
        if tz_name:
            tz = ZoneInfo(tz_name)
            now = datetime.now(tz)
            tz_abbr = now.strftime("%Z")
            return now.strftime(f"%A, %B %d, %Y at %H:%M {tz_abbr} ({query})")
        return f"Could not understand the date: '{query}'"
    return parsed.strftime("%A, %B %d, %Y")

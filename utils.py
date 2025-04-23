"""
Reusable helpers for MiniAgent (colourful logging, schema generation, etc.).
"""

import inspect
import re
from typing import Any, Dict

# ── ANSI colour tags ──────────────────────────────────────────────
_CLR: Dict[str, str] = {
    "reasoning": "\033[90m",
    "message": "\033[36m",
    "endmessage": "\033[36m",
    "function_call": "\033[33m",
    "function_output": "\033[32m",
}
_RESET = "\033[0m"
_TAG_W = 16  # width for the coloured label inside [...]


def tag(kind: str) -> str:
    """Return a coloured, fixed‑width tag:  '[function_call ] '."""
    colour = _CLR.get(kind, "")
    return f"[{colour}{kind.ljust(_TAG_W)}{_RESET}] "


def strip_ansi(s: str) -> str:
    """Remove ANSI escape codes (useful for width calculations)."""
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ── schema helper ────────────────────────────────────────────────
def fn_to_schema(fn) -> Dict[str, Any]:
    """
    Build a minimal function‑tool schema from a python callable.
    All parameters are typed as string for brevity.
    """
    props = {p: {"type": "string"} for p in inspect.signature(fn).parameters}
    docstring = inspect.getdoc(fn) or ""
    return {
        "type": "function",
        "name": fn.__name__,
        "description": docstring,
        "parameters": {
            "type": "object",
            "properties": props,
            "required": list(props),
            "additionalProperties": True,
        },
    }

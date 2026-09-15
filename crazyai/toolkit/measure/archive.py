"""archive.* - bookkeeping tools available to the model (the pipeline writes the canonical files itself)."""

from __future__ import annotations

import json
from pathlib import Path

from crazyai.toolkit.registry import tool


@tool("archive", "measure")
def write_note(ctx: dict, name: str, content: str) -> dict:
    """Save a named note into the current run folder (e.g. 'scratch', 'derivation').

    Args:
        name: File name without extension (letters, digits, - and _).
        content: Text to save.
    """
    run_dir = ctx.get("run_dir")
    if not run_dir:
        return {"error": "no run directory in context"}
    safe = "".join(c for c in name if c.isalnum() or c in "-_")
    if not safe:
        return {"error": "bad name"}
    p = Path(run_dir) / f"note_{safe}.md"
    p.write_text(content)
    return {"written": str(p), "bytes": len(content.encode())}


@tool("archive", "measure")
def read_key(ctx: dict) -> dict:
    """Read the current run's answer key, if it has been written (generator session only)."""
    run_dir = ctx.get("run_dir")
    if not run_dir:
        return {"error": "no run directory in context"}
    p = Path(run_dir) / "key.json"
    if not p.exists():
        return {"error": "no key yet"}
    return json.loads(p.read_text())

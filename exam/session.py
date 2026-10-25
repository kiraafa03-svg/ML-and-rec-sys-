"""Inspect/copy an actual native export; never manufacture a session transcript."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .data import sha256


def inspect_session(path: Path) -> dict:
    raw = path.read_bytes()
    try:
        parsed = json.loads(raw)
        records = parsed if isinstance(parsed, list) else [parsed]
        container = "json"
    except json.JSONDecodeError:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        container = "jsonl"
    calls = results = 0

    def walk(value):
        nonlocal calls, results
        if isinstance(value, dict):
            kind = str(value.get("type", value.get("role", ""))).lower()
            if kind in {"function_call", "custom_tool_call", "tool_call", "tool_use"}:
                calls += 1
            if kind in {"function_call_output", "custom_tool_call_output", "tool_result", "tool"}:
                results += 1
            for inner in value.values():
                walk(inner)
        elif isinstance(value, list):
            for inner in value:
                walk(inner)

    walk(records)
    if not calls or not results:
        raise ValueError(
            "Export has no recognised raw tool calls/results; a results JSON or session link is insufficient"
        )
    return {
        "filename": path.name,
        "sha256": sha256(path),
        "size_bytes": len(raw),
        "container": container,
        "recognised_tool_calls": calls,
        "recognised_tool_results": results,
        "limitation": "Structural inspection cannot independently prove export provenance",
    }


def copy_native_session(source: Path, destination: Path) -> dict:
    info = inspect_session(source)
    if source.resolve() == destination.resolve():
        raise ValueError("Choose a separate destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256(source) != sha256(destination):
        raise IOError("Export copy is not byte-identical")
    return {**info, "destination": str(destination), "byte_identical_copy": True}

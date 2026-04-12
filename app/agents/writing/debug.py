"""
Write-pipeline debug tracer.

When ``settings.WRITING_DEBUG`` is *True* (env var ``WRITING_DEBUG=true``),
every write-mode request produces a JSON trace file in
``settings.WRITING_DEBUG_DIR`` (default ``debug/write_pipeline/``).

When disabled (the default), every method is a **no-op** — zero overhead.

Usage
-----
In ``_handle_write_mode``::

    dbg = WritePipelineDebugger.from_settings()
    dbg.set_request_info(session_id=..., section_target=...)
    # ... pass *dbg* to each agent method ...
    dbg.finalize()          # writes the JSON file (or does nothing)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Truncation helper (keeps debug files reasonable)
# ---------------------------------------------------------------------------

def _truncate(value: Any, max_len: int) -> Any:
    """Recursively truncate strings in a value (dict / list / str)."""
    if isinstance(value, str):
        if len(value) <= max_len:
            return value
        return value[:max_len] + f"... ({len(value)} chars total)"
    if isinstance(value, dict):
        return {k: _truncate(v, max_len) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(v, max_len) for v in value]
    return value


# ---------------------------------------------------------------------------
# LLM-call timer (context manager)
# ---------------------------------------------------------------------------

@dataclass
class _LLMCallRecord:
    agent: str
    method: str
    start: float = 0.0
    duration_ms: float = 0.0


class LLMTimer:
    """Thin context manager that records elapsed time for an LLM call.

    Usage::

        async with dbg.llm_timer("orchestrator", "classify") as t:
            response = await self._llm.achat(messages)
        # t.duration_ms is now set
    """

    def __init__(self, debugger: WritePipelineDebugger, agent: str, method: str) -> None:
        self._debugger = debugger
        self._record = _LLMCallRecord(agent=agent, method=method)

    async def __aenter__(self) -> _LLMCallRecord:
        self._record.start = time.perf_counter()
        return self._record

    async def __aexit__(self, *_exc: Any) -> None:
        elapsed = time.perf_counter() - self._record.start
        self._record.duration_ms = round(elapsed * 1000, 1)
        self._debugger._llm_calls.append({
            "agent": self._record.agent,
            "method": self._record.method,
            "duration_ms": self._record.duration_ms,
        })


# ---------------------------------------------------------------------------
# Noop timer (when debug is off)
# ---------------------------------------------------------------------------

class _NoopTimer:
    """Async context manager that does nothing."""

    async def __aenter__(self) -> _NoopTimer:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        pass

    # Allow attribute access without error (e.g. t.duration_ms)
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Main debugger
# ---------------------------------------------------------------------------

class WritePipelineDebugger:
    """Accumulates a structured trace of the write pipeline, then dumps to JSON.

    All public methods are safe to call even when disabled — they simply return
    immediately.
    """

    def __init__(
        self,
        enabled: bool = False,
        output_dir: str = "debug/write_pipeline",
        max_content: int = 5000,
    ) -> None:
        self._enabled = enabled
        self._output_dir = output_dir
        self._max_content = max_content

        # Trace data
        self._request_id: str = str(uuid.uuid4())[:8]
        self._start_time: float = time.perf_counter()
        self._timestamp: str = datetime.now(timezone.utc).isoformat()
        self._meta: dict[str, Any] = {}
        self._phases: dict[str, dict[str, Any]] = {}
        self._llm_calls: list[dict[str, Any]] = []

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> WritePipelineDebugger:
        """Create a debugger using app settings."""
        from app.core.config import settings
        return cls(
            enabled=settings.WRITING_DEBUG,
            output_dir=settings.WRITING_DEBUG_DIR,
            max_content=settings.WRITING_DEBUG_MAX_CONTENT,
        )

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Request-level metadata ───────────────────────────────────────────

    def set_request_info(self, **kwargs: Any) -> None:
        """Store top-level metadata (session_id, section_target, etc.)."""
        if not self._enabled:
            return
        for k, v in kwargs.items():
            self._meta[k] = str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v

    # ── Phase / step logging ─────────────────────────────────────────────

    def log_step(self, phase: str, step: str, data: Any) -> None:
        """Record a debug data point under ``phases[phase][step]``.

        ``data`` is truncated according to ``max_content``.
        """
        if not self._enabled:
            return
        if phase not in self._phases:
            self._phases[phase] = {}
        self._phases[phase][step] = _truncate(data, self._max_content)

    # ── LLM call timer ───────────────────────────────────────────────────

    def llm_timer(self, agent: str, method: str) -> LLMTimer | _NoopTimer:
        """Return an async context manager that times an LLM call.

        When disabled, returns a no-op timer.
        """
        if not self._enabled:
            return _NoopTimer()
        return LLMTimer(self, agent, method)

    # ── Finalize ─────────────────────────────────────────────────────────

    def finalize(self) -> Optional[str]:
        """Write the trace to a JSON file. Returns the file path, or None if disabled."""
        if not self._enabled:
            return None

        elapsed_ms = round((time.perf_counter() - self._start_time) * 1000, 1)

        trace = {
            "request_id": self._request_id,
            "timestamp": self._timestamp,
            "duration_ms": elapsed_ms,
            **self._meta,
            "phases": self._phases,
            "llm_calls": self._llm_calls,
            "llm_calls_total_ms": round(sum(c["duration_ms"] for c in self._llm_calls), 1),
        }

        # Ensure output directory exists
        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"{ts}_{self._request_id}.json"
        filepath = out_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False, default=str)

        return str(filepath)

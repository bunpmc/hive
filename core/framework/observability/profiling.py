"""
Event loop profiling for debug mode.

Activated by ``hive open --debug`` (or ``hive serve --debug``).

What this provides
------------------
- **Event loop lag sampling** — a background task wakes every 100 ms and
  measures how late it actually woke.  Lag > 50 ms means something blocked
  the loop (sync file I/O, CPU-bound work, etc.).
- **asyncio slow-callback capture** — when asyncio debug mode is on the
  runtime logs every callback that held the loop for > ``slow_callback_duration``
  seconds.  We intercept those log records and store them in a ring buffer so
  the ``/api/debug/profile`` endpoint can surface them.
- **asyncio debug mode** — ``loop.set_debug(True)`` enables asyncio's own
  built-in coroutine-origin tracking and slow-callback warnings.

Usage (framework-internal)
--------------------------
Inside an async context (after ``asyncio.run()`` enters):

    from framework.observability.profiling import start_debug_profiling
    monitor = await start_debug_profiling()
    ...
    # on shutdown:
    await monitor.stop()
    monitor.print_summary()
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Ring-buffer sizes
_MAX_LAG_SAMPLES = 600  # 60 s at 100 ms interval
_MAX_SLOW_CALLBACKS = 100

# Thresholds
_LAG_WARN_THRESHOLD_MS = 50.0  # log a warning when loop lag exceeds this
_SAMPLE_INTERVAL_S = 0.1  # how often to probe the event loop (100 ms)


class _SlowCallbackCapture(logging.Handler):
    """Intercepts asyncio's built-in slow-callback warnings.

    asyncio emits them at WARNING level from the ``asyncio`` logger when
    ``loop.set_debug(True)`` is active and a callback holds the loop for
    longer than ``loop.slow_callback_duration`` seconds.

    Each captured entry is tagged ``blocking=True`` only if the lag sampler
    measured a spike at the same time — which is the definitive signal that
    the loop was actually held rather than the task being slow async I/O.
    """

    def __init__(self, monitor: EventLoopMonitor) -> None:
        super().__init__(level=logging.WARNING)
        self._monitor = monitor

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        # asyncio formats these as: "Executing <...> took X.XXX seconds"
        if "took" in msg and "second" in msg:
            # Check whether the lag sampler saw a spike in the past two probe
            # intervals.  If yes this is a real blocking call; if no the task
            # was just slow async I/O that properly yielded the loop.
            now = time.monotonic()
            window = _SAMPLE_INTERVAL_S * 2
            recent_lag = max(
                (lag for ts, lag in self._monitor._lag_spikes if now - ts <= window),
                default=0.0,
            )
            self._monitor._record_slow_callback(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "message": msg,
                    "source": "asyncio",
                    "lag_during_ms": round(recent_lag, 2),
                    # True only when lag also spiked → real event loop block
                    "blocking": recent_lag > _LAG_WARN_THRESHOLD_MS,
                }
            )


class EventLoopMonitor:
    """Measures event-loop lag by scheduling a repeating probe task.

    The probe sleeps for ``_SAMPLE_INTERVAL_S`` seconds.  The *actual* sleep
    duration minus the *expected* duration gives the lag introduced by any
    blocking work that ran between two consecutive iterations of the loop.

    All stats are available via :meth:`snapshot` and are exposed through the
    ``/api/debug/profile`` HTTP endpoint.
    """

    def __init__(self) -> None:
        self._samples: list[float] = []  # lag in ms, newest last
        self._lag_spikes: list[tuple[float, float]] = []  # (monotonic_ts, lag_ms)
        self._slow_callbacks: list[dict[str, Any]] = []  # ring buffer
        self._start_time: float = 0.0
        self._task: asyncio.Task | None = None
        self._log_handler: _SlowCallbackCapture | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background probe and attach the slow-callback log handler."""
        self._start_time = time.monotonic()

        # Attach handler to asyncio logger to capture slow-callback warnings
        self._log_handler = _SlowCallbackCapture(self)
        asyncio_logger = logging.getLogger("asyncio")
        asyncio_logger.addHandler(self._log_handler)

        self._task = asyncio.create_task(self._probe_loop(), name="hive-loop-monitor")
        logger.info(
            "Event loop monitor started — lag threshold %.0f ms, sample interval %.0f ms",
            _LAG_WARN_THRESHOLD_MS,
            _SAMPLE_INTERVAL_S * 1000,
        )

    async def stop(self) -> None:
        """Cancel the probe task and detach the log handler."""
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

        if self._log_handler is not None:
            asyncio_logger = logging.getLogger("asyncio")
            asyncio_logger.removeHandler(self._log_handler)
            self._log_handler = None

    # ------------------------------------------------------------------
    # Internal probe
    # ------------------------------------------------------------------

    async def _probe_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            expected_wake = loop.time() + _SAMPLE_INTERVAL_S
            await asyncio.sleep(_SAMPLE_INTERVAL_S)
            actual_wake = loop.time()
            lag_ms = max(0.0, (actual_wake - expected_wake) * 1000)

            self._samples.append(lag_ms)
            if len(self._samples) > _MAX_LAG_SAMPLES:
                self._samples = self._samples[-_MAX_LAG_SAMPLES:]

            if lag_ms > _LAG_WARN_THRESHOLD_MS:
                self._lag_spikes.append((time.monotonic(), lag_ms))
                if len(self._lag_spikes) > 200:
                    self._lag_spikes = self._lag_spikes[-200:]
                logger.warning(
                    "Event loop lag %.1f ms — possible blocking call on main thread",
                    lag_ms,
                    extra={"event": "loop_lag", "lag_ms": round(lag_ms, 1)},
                )

    def _record_slow_callback(self, entry: dict[str, Any]) -> None:
        self._slow_callbacks.append(entry)
        if len(self._slow_callbacks) > _MAX_SLOW_CALLBACKS:
            self._slow_callbacks = self._slow_callbacks[-_MAX_SLOW_CALLBACKS:]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return current profiling stats as a plain dict (JSON-serialisable)."""
        samples = list(self._samples)
        uptime = round(time.monotonic() - self._start_time, 1) if self._start_time else 0.0

        if not samples:
            return {
                "enabled": True,
                "uptime_seconds": uptime,
                "samples": 0,
                "slow_callbacks": list(self._slow_callbacks),
            }

        sorted_s = sorted(samples)
        n = len(sorted_s)

        def pct(p: float) -> float:
            return round(sorted_s[min(int(n * p), n - 1)], 2)

        over = sum(1 for s in samples if s > _LAG_WARN_THRESHOLD_MS)

        callbacks = list(self._slow_callbacks)
        return {
            "enabled": True,
            "uptime_seconds": uptime,
            "samples": n,
            "mean_lag_ms": round(sum(samples) / n, 2),
            "p50_lag_ms": pct(0.50),
            "p95_lag_ms": pct(0.95),
            "p99_lag_ms": pct(0.99),
            "worst_lag_ms": round(max(samples), 2),
            "slow_samples": over,
            "slow_sample_pct": round(over / n * 100, 1),
            "slow_threshold_ms": _LAG_WARN_THRESHOLD_MS,
            # All asyncio slow-task warnings captured
            "slow_callbacks": callbacks,
            # Subset where lag also spiked → real event loop blocking
            "blocking_callbacks": [cb for cb in callbacks if cb.get("blocking")],
            # Subset that are just slow async I/O — loop was free the whole time
            "benign_slow_tasks": [cb for cb in callbacks if not cb.get("blocking")],
        }

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout (called on server shutdown)."""
        snap = self.snapshot()
        if snap["samples"] == 0:
            print("[profiler] No samples collected.")
            return

        print("\n[profiler] Event loop summary")
        print(f"  Uptime:        {snap['uptime_seconds']}s")
        print(f"  Samples:       {snap['samples']}")
        print(f"  Mean lag:      {snap['mean_lag_ms']} ms")
        p50, p95, p99 = snap["p50_lag_ms"], snap["p95_lag_ms"], snap["p99_lag_ms"]
        print(f"  p50 / p95 / p99:  {p50} / {p95} / {p99} ms")
        print(f"  Worst lag:     {snap['worst_lag_ms']} ms")
        slow_pct = snap["slow_sample_pct"]
        thresh = snap["slow_threshold_ms"]
        print(f"  Slow samples:  {snap['slow_samples']} ({slow_pct}%) > {thresh} ms")
        blocking = snap.get("blocking_callbacks", [])
        benign = snap.get("benign_slow_tasks", [])
        if blocking:
            print(f"  BLOCKING callbacks: {len(blocking)}  ← investigate these")
            for cb in blocking[-5:]:
                lag = cb.get("lag_during_ms", "?")
                print(f"    [lag {lag}ms] {cb['timestamp']}  {cb['message'][:120]}")
        if benign:
            print(
                f"  Slow async tasks (benign): {len(benign)}"
                " — loop was free, just slow I/O"
            )
        print()


# Module-level singleton so routes can access it without passing it around.
_monitor: EventLoopMonitor | None = None


async def start_debug_profiling() -> EventLoopMonitor:
    """Enable asyncio debug mode and start the event loop monitor.

    Must be called from inside a running event loop (i.e. inside
    ``asyncio.run()``).  Returns the monitor so the caller can stop it and
    print a summary on shutdown.
    """
    global _monitor

    loop = asyncio.get_running_loop()

    # asyncio built-in debug mode: enables coroutine-origin tracking and
    # logs any callback/coroutine that holds the loop for > slow_callback_duration.
    loop.set_debug(True)
    loop.slow_callback_duration = 0.1  # 100 ms — asyncio default; avoids noise

    _monitor = EventLoopMonitor()
    await _monitor.start()
    return _monitor


def get_profile_snapshot() -> dict[str, Any]:
    """Return the current monitor snapshot, or a disabled sentinel if not running."""
    if _monitor is None:
        return {"enabled": False}
    return _monitor.snapshot()

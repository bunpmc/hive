"""Debug/profiling routes — only meaningful when ``hive open --debug`` is active.

Routes
------
GET /api/debug/profile
    Returns the current event loop profiling snapshot as JSON.
    When profiling is not active (``--debug`` flag absent) the response is
    ``{"enabled": false}``.

GET /api/debug/slow-callbacks
    Returns the ring buffer of slow callbacks captured from asyncio's
    built-in slow-callback logger.  Empty list when not active.
"""

import logging

from aiohttp import web

from framework.observability.profiling import get_profile_snapshot

logger = logging.getLogger(__name__)


async def handle_profile(request: web.Request) -> web.Response:
    """GET /api/debug/profile — current event loop profiling stats."""
    return web.json_response(get_profile_snapshot())


async def handle_slow_callbacks(request: web.Request) -> web.Response:
    """GET /api/debug/slow-callbacks — recent slow asyncio callbacks."""
    snap = get_profile_snapshot()
    return web.json_response(
        {
            "enabled": snap.get("enabled", False),
            "slow_callbacks": snap.get("slow_callbacks", []),
            "slow_threshold_ms": snap.get("slow_threshold_ms", 50),
        }
    )


def register_routes(app: web.Application) -> None:
    app.router.add_get("/api/debug/profile", handle_profile)
    app.router.add_get("/api/debug/slow-callbacks", handle_slow_callbacks)

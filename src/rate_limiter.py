"""Rate limiter for Gemini API calls.

Ensures we don't exceed 15 requests per minute (RPM) by enforcing
a minimum interval between calls.

Usage::

    from src.rate_limiter import rate_limiter

    # In any async function that calls Gemini:
    await rate_limiter.wait()
    result = await llm.ainvoke(...)

    # Or as a context manager:
    async with rate_limiter:
        result = await llm.ainvoke(...)
"""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple token-bucket rate limiter for API calls.

    Args:
        rpm: Maximum requests per minute.
    """

    def __init__(self, rpm: int = 15):
        self.min_interval = 60.0 / rpm  # seconds between calls
        self._last_call: float = 0.0
        self._lock = asyncio.Lock()

    async def wait(self):
        """Wait until the next API call is allowed."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                delay = self.min_interval - elapsed
                logger.debug("[rate_limiter] Sleeping %.1fs to respect RPM limit", delay)
                await asyncio.sleep(delay)
            self._last_call = time.monotonic()

    async def __aenter__(self):
        await self.wait()
        return self

    async def __aexit__(self, *args):
        pass


# Module-level singleton — import this in agents
rate_limiter = RateLimiter(rpm=14)  # slightly under 15 for safety margin

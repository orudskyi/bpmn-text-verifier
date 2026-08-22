"""LLM call retry utility for Gemini API rate limiting.

Drop this file into src/ and use the @with_retry decorator or
wrap_with_retry() function in your agent modules.

Usage in mapper_agent.py / formalizer_agent.py:

    from src.llm_retry import with_retry

    class MapperAgent:
        ...
        @with_retry(max_attempts=4, initial_wait=30)
        async def _run_llm(self, ...):
            return await self.chain.ainvoke(...)

OR — without touching class methods — wrap the chain call:

    from src.llm_retry import call_with_retry

    result = await call_with_retry(self.chain.ainvoke, prompt_input)
"""

import asyncio
import functools
import logging
import time

logger = logging.getLogger(__name__)

# Error message substrings that indicate a rate-limit (429) error
_RATE_LIMIT_SIGNALS = [
    "429",
    "resource_exhausted",
    "resourceexhausted",
    "quota",
    "rate limit",
    "ratelimit",
    "too many requests",
]


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if exception looks like a Gemini 429 / quota error."""
    msg = str(exc).lower()
    return any(signal in msg for signal in _RATE_LIMIT_SIGNALS)


def with_retry(max_attempts: int = 4, initial_wait: float = 30.0, backoff: float = 2.0):
    """Decorator: retry an async method on rate-limit errors with exponential backoff.

    Args:
        max_attempts: Total number of tries (including the first attempt).
        initial_wait: Seconds to wait after the first failure.
        backoff: Multiplier applied to wait time after each failure.

    Example::

        @with_retry(max_attempts=4, initial_wait=30)
        async def _run_llm(self, prompt):
            return await self.chain.ainvoke(prompt)
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            wait = initial_wait
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_attempts:
                        logger.error(
                            "%s: all %d attempts failed. Last error: %s",
                            func.__qualname__, max_attempts, exc
                        )
                        raise
                    if _is_rate_limit_error(exc):
                        logger.warning(
                            "%s: rate limit hit (attempt %d/%d). Waiting %.0fs...",
                            func.__qualname__, attempt, max_attempts, wait
                        )
                        await asyncio.sleep(wait)
                        wait *= backoff
                    else:
                        # Non-rate-limit error — re-raise immediately
                        raise
        return wrapper
    return decorator


async def call_with_retry(
    coro_fn,
    *args,
    max_attempts: int = 4,
    initial_wait: float = 30.0,
    backoff: float = 2.0,
    **kwargs,
):
    """Call an async function with retry on rate-limit errors.

    Use this when you cannot use the @with_retry decorator (e.g. lambdas,
    inline chain calls).

    Args:
        coro_fn: The async callable to invoke.
        *args: Positional arguments passed to coro_fn.
        max_attempts: Total attempts.
        initial_wait: Seconds to wait after first failure.
        backoff: Exponential backoff multiplier.
        **kwargs: Keyword arguments passed to coro_fn.

    Returns:
        The result of coro_fn(*args, **kwargs) on first success.

    Example::

        result = await call_with_retry(
            self.chain.ainvoke,
            {"text": fragment_text},
            max_attempts=4,
            initial_wait=30,
        )
    """
    wait = initial_wait
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            if _is_rate_limit_error(exc):
                logger.warning(
                    "Rate limit hit on attempt %d/%d. Waiting %.0fs...",
                    attempt, max_attempts, wait,
                )
                await asyncio.sleep(wait)
                wait *= backoff
            else:
                raise
    raise last_exc

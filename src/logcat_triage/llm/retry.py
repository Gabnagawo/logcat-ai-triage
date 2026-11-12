"""Small retry helper shared by every network call (LLM and embeddings)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

log = logging.getLogger(__name__)


def call_with_retries[T](
    fn: Callable[[], T],
    *,
    retries: int,
    retry_on: tuple[type[BaseException], ...],
    backoff_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn``; on ``retry_on`` errors retry up to ``retries`` times with linear backoff."""
    attempt = 0
    while True:
        try:
            return fn()
        except retry_on as exc:
            if attempt >= retries:
                raise
            attempt += 1
            log.warning("attempt %d failed (%s); retrying", attempt, exc)
            sleep(backoff_s * attempt)

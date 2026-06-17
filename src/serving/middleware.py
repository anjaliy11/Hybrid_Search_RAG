"""Custom middleware — latency tracking and request logging."""

import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class LatencyMiddleware(BaseHTTPMiddleware):
    """Logs request latency for monitoring."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.time()
        response = await call_next(request)
        elapsed = (time.time() - start) * 1000  # ms

        logger.info(
            f"{request.method} {request.url.path} — {response.status_code} — {elapsed:.0f}ms"
        )
        response.headers["X-Response-Time-Ms"] = f"{elapsed:.0f}"
        return response
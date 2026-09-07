"""Correlation-aware, structured request telemetry for the MINCO API."""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request
from starlette.responses import Response

from src.contracts.common import new_correlation_id
from src.observability.metrics import API_METRICS


CORRELATION_HEADER = "X-Correlation-ID"
_CORRELATION_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,96}$")
logger = logging.getLogger("minco.api")


def resolve_correlation_id(value: str | None) -> str:
    """Keep safe caller IDs and replace malformed values with a generated ID."""
    candidate = value.strip() if isinstance(value, str) else ""
    return candidate if _CORRELATION_PATTERN.fullmatch(candidate) else new_correlation_id()


def correlation_id_for_request(request: Request) -> str:
    """Return the ID assigned by middleware, with a safe fallback for direct calls."""
    return str(
        getattr(request.state, "correlation_id", None)
        or resolve_correlation_id(request.headers.get(CORRELATION_HEADER))
    )


async def correlation_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Attach a correlation ID and emit one structured completion event per request."""
    correlation_id = resolve_correlation_id(request.headers.get(CORRELATION_HEADER))
    request.state.correlation_id = correlation_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
        API_METRICS.record(
            method=request.method,
            path=request.url.path,
            status_code=500,
            duration_ms=duration_ms,
        )
        logger.exception(
            "api_request_failed",
            extra={
                "minco_event": {
                    "event": "api_request_failed",
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "duration_ms": duration_ms,
                }
            },
        )
        raise

    duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
    API_METRICS.record(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    response.headers[CORRELATION_HEADER] = correlation_id
    logger.info(
        "api_request_completed",
        extra={
            "minco_event": {
                "event": "api_request_completed",
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            }
        },
    )
    return response

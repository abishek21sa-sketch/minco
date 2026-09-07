"""Low-cardinality in-process API metrics with Prometheus exposition."""
from __future__ import annotations

import re
from collections import Counter
from threading import Lock
from time import monotonic


_ROUTE_PATTERNS = (
    (re.compile(r"^/v1/audit/[^/]+/review$"), "/v1/audit/{run_id}/review"),
    (re.compile(r"^/v1/audit/[^/]+/reviews$"), "/v1/audit/{run_id}/reviews"),
    (re.compile(r"^/v1/scenarios/[^/]+$"), "/v1/scenarios/{scenario_id}"),
)


def normalize_route(path: str) -> str:
    for pattern, route in _ROUTE_PATTERNS:
        if pattern.fullmatch(path):
            return route
    return path


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class ApiMetrics:
    """Thread-safe request counters; no payloads, query strings, or IDs are retained."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._started = monotonic()
        self._request_count = 0
        self._error_count = 0
        self._status_counts: Counter[str] = Counter()
        self._route_counts: Counter[str] = Counter()
        self._latency_count = 0
        self._latency_sum = 0.0
        self._latency_max = 0.0

    def record(self, *, method: str, path: str, status_code: int, duration_ms: float) -> None:
        route_key = f"{method.upper()} {normalize_route(path)}"
        status_class = f"{status_code // 100}xx"
        with self._lock:
            self._request_count += 1
            self._error_count += int(status_code >= 500)
            self._status_counts[status_class] += 1
            self._route_counts[route_key] += 1
            self._latency_count += 1
            self._latency_sum += duration_ms
            self._latency_max = max(self._latency_max, duration_ms)

    def reset(self) -> None:
        with self._lock:
            self._started = monotonic()
            self._request_count = 0
            self._error_count = 0
            self._status_counts.clear()
            self._route_counts.clear()
            self._latency_count = 0
            self._latency_sum = 0.0
            self._latency_max = 0.0

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "uptime_seconds": round(max(0.0, monotonic() - self._started), 3),
                "request_count": self._request_count,
                "error_count": self._error_count,
                "status_counts": dict(sorted(self._status_counts.items())),
                "route_counts": dict(sorted(self._route_counts.items())),
                "latency_ms": {
                    "count": self._latency_count,
                    "sum": round(self._latency_sum, 3),
                    "max": round(self._latency_max, 3),
                },
            }

    def prometheus(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP minco_api_requests_total Total HTTP requests observed by MINCO.",
            "# TYPE minco_api_requests_total counter",
            f"minco_api_requests_total {snapshot['request_count']}",
            "# HELP minco_api_errors_total Total HTTP 5xx responses observed by MINCO.",
            "# TYPE minco_api_errors_total counter",
            f"minco_api_errors_total {snapshot['error_count']}",
            "# HELP minco_api_request_status_total Requests grouped by status class.",
            "# TYPE minco_api_request_status_total counter",
        ]
        for status_class, count in snapshot["status_counts"].items():
            lines.append(
                f'minco_api_request_status_total{{status_class="{_escape_label(status_class)}"}} {count}'
            )
        lines.extend([
            "# HELP minco_api_request_route_total Requests grouped by normalized route.",
            "# TYPE minco_api_request_route_total counter",
        ])
        for route_key, count in snapshot["route_counts"].items():
            method, route = route_key.split(" ", 1)
            lines.append(
                "minco_api_request_route_total{method=\"%s\",route=\"%s\"} %s"
                % (_escape_label(method), _escape_label(route), count)
            )
        latency = snapshot["latency_ms"]
        lines.extend([
            "# HELP minco_api_request_duration_ms Request duration summary in milliseconds.",
            "# TYPE minco_api_request_duration_ms summary",
            f"minco_api_request_duration_ms_sum {latency['sum']}",
            f"minco_api_request_duration_ms_count {latency['count']}",
            f"minco_api_request_duration_ms_max {latency['max']}",
            "",
        ])
        return "\n".join(lines)


API_METRICS = ApiMetrics()


__all__ = ["API_METRICS", "ApiMetrics", "normalize_route"]

import time
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

REQUEST_COUNT = Counter("app_requests_total", "Total HTTP requests", ["method","endpoint","status"])
REQUEST_LAT = Histogram("app_request_latency_seconds", "Request latency", ["method","endpoint"])

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        endpoint = request.url.path
        REQUEST_COUNT.labels(request.method, endpoint, str(response.status_code)).inc()
        REQUEST_LAT.labels(request.method, endpoint).observe(elapsed)
        response.headers["X-Elapsed-ms"] = f"{elapsed*1000:.2f}"
        return response

async def metrics_endpoint(_):
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

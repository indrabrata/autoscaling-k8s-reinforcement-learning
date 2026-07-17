# Resource Intensive Service

This is a simple REST API service written in Go that burns CPU and memory on demand. It is the workload the
autoscaling agent scales during experiments — it exists to produce controllable, repeatable load rather than
to do anything useful, and it exports the Prometheus metrics the agent reads back as state.

## Features

- Generate CPU load on request (prime sieve, transcendental math, 50×50 matrix multiplication)
- Generate memory load on request (nested structures, transaction matrix, optional heavy aggregation)
- Export request count and latency histogram for Prometheus

## Endpoints

| Endpoint                             | Purpose                                      |
| ------------------------------------ | -------------------------------------------- |
| `POST /cpu?iterations=N`             | CPU-intensive work, `N` defaults to 100      |
| `POST /memory?size=N&heavy_agg=true` | Memory-intensive work, `N` defaults to 10000 |
| `GET /healthz`                       | Health check                                 |
| `GET /metrics`                       | Prometheus metrics                           |

Both workload endpoints also accept `GET`, and every response carries an `X-Elapsed-ms` header. The service
listens on port 8080.

Latency is exported as `app_request_latency_seconds` and request count as `app_requests_total`, both labelled
by method and endpoint. The agent's P90 response time comes from `histogram_quantile` over the former.

## Deployment

[deployment/](deployment/) carries one deployment + service + ServiceMonitor per experiment arm
(`resource-intensive-q`, `resource-intensive-qfuzzy`), so both agents can train concurrently against their own
copy without interfering. [ingress.rule.yaml](deployment/ingress.rule.yaml) routes `/api/q` and `/api/qfuzzy`
to the matching service, and [app.hpa.yaml](deployment/app.hpa.yaml) provides the HPA baseline to compare
against.

Each container sets `resources.limits` (500m CPU, 512Mi memory). These are **required** — the agent measures
utilization as a percentage of limit, so a pod without them reports no usable metrics at all.

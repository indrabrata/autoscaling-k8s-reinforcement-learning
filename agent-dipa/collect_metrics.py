from __future__ import annotations

import csv
import logging
import os
import signal
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect

from utils.logger import setup_logger
from utils.metrics import (
    _build_cpu_limits_query,
    _build_cpu_usage_query,
    _build_memory_limits_query,
    _build_memory_usage_query,
    _build_request_rate_query,
    _build_scope_ready_query,
    _extract_limits_by_pod,
    _get_request_rate,
    _get_response_time,
    _metrics_result,
)

load_dotenv()

CSV_COLUMNS = [
    "timestamp",
    "cpu_usage",
    "memory_usage",
    "response_time_ms",
    "request_rate_rps",
    "replica_state",
    "pod_count",
    "cpu_usage_cores",
    "cpu_limit_cores",
    "memory_usage_mb",
    "memory_limit_mb",
]

_running = True


def _handle_signal(signum, frame):
    global _running
    _running = False


def get_current_replicas(
    namespace: str,
    deployment_name: str,
    logger: logging.Logger,
) -> int:
    """Get current ready replica count from Kubernetes."""
    try:
        apps_v1 = client.AppsV1Api()
        deployment = apps_v1.read_namespaced_deployment(
            name=deployment_name, namespace=namespace
        )
        if deployment.status is None:
            return 0
        return deployment.status.ready_replicas or 0
    except Exception as e:
        logger.error(f"Failed to get replicas: {e}")
        return 0


def collect_once(
    prometheus: PrometheusConnect,
    namespace: str,
    deployment_name: str,
    replicas: int,
    interval: int,
    quantile: float,
    endpoints_method: list,
    logger: logging.Logger,
) -> dict:
    """Collect a single snapshot of metrics from Prometheus."""
    scope_ready = _build_scope_ready_query(namespace, deployment_name)
    cpu_query = _build_cpu_usage_query(namespace, scope_ready, interval)
    memory_query = _build_memory_usage_query(namespace, scope_ready)
    cpu_limits_query = _build_cpu_limits_query(namespace, scope_ready)
    memory_limits_query = _build_memory_limits_query(namespace, scope_ready)
    request_rate_query = _build_request_rate_query(namespace, deployment_name, interval)

    try:
        cpu_usage_results = prometheus.custom_query(cpu_query)
        memory_usage_results = prometheus.custom_query(memory_query)
        cpu_limits_results = prometheus.custom_query(cpu_limits_query)
        memory_limits_results = prometheus.custom_query(memory_limits_query)
    except Exception as e:
        logger.error(f"Prometheus query failed: {e}")
        return {}

    if not cpu_usage_results or not memory_usage_results:
        logger.warning("No CPU/memory metrics returned")
        return {}

    try:
        import numpy as np

        cpu_pcts, mem_pcts, pod_names = _metrics_result(
            cpu_limits_results,
            memory_limits_results,
            cpu_usage_results,
            memory_usage_results,
            logger=logger,
        )

        cpu_mean = float(np.nanmean(cpu_pcts)) if cpu_pcts else 0.0
        mem_mean = float(np.nanmean(mem_pcts)) if mem_pcts else 0.0

        # Sum raw usage and limits across all pods
        cpu_usage_by_pod = _extract_limits_by_pod(cpu_usage_results)
        cpu_limits_by_pod = _extract_limits_by_pod(cpu_limits_results)
        mem_usage_by_pod = _extract_limits_by_pod(memory_usage_results)
        mem_limits_by_pod = _extract_limits_by_pod(memory_limits_results)

        BYTES_PER_MB = 1024 * 1024
        cpu_usage_cores = sum(cpu_usage_by_pod.values())
        cpu_limit_cores = sum(cpu_limits_by_pod.values())
        memory_usage_mb = sum(mem_usage_by_pod.values()) / BYTES_PER_MB
        memory_limit_mb = sum(mem_limits_by_pod.values()) / BYTES_PER_MB
        pod_count = len(pod_names)
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        cpu_mean, mem_mean = 0.0, 0.0
        cpu_usage_cores, cpu_limit_cores = 0.0, 0.0
        memory_usage_mb, memory_limit_mb = 0.0, 0.0
        pod_count = 0

    response_time = _get_response_time(
        prometheus=prometheus,
        deployment_name=deployment_name,
        namespace=namespace,
        endpoints_method=endpoints_method,
        interval=interval,
        quantile=quantile,
        logger=logger,
    )

    request_rate = _get_request_rate(
        prometheus=prometheus,
        rps_query=request_rate_query,
        logger=logger,
    )

    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_usage": round(cpu_mean, 3),
        "memory_usage": round(mem_mean, 3),
        "response_time_ms": round(response_time, 3),
        "request_rate_rps": round(request_rate, 3),
        "replica_state": replicas,
        "pod_count": pod_count,
        "cpu_usage_cores": round(cpu_usage_cores, 6),
        "cpu_limit_cores": round(cpu_limit_cores, 6),
        "memory_usage_mb": round(memory_usage_mb, 3),
        "memory_limit_mb": round(memory_limit_mb, 3),
    }


def run_collector(
    prometheus_url: str,
    namespace: str,
    deployment_name: str,
    output_path: str,
    collect_interval: int = 30,
    metrics_interval: int = 15,
    quantile: float = 0.90,
    endpoints_method: list | None = None,
    logger: logging.Logger | None = None,
) -> str:
    """
    Run metrics collection loop. Collects every `collect_interval` seconds
    and saves results to CSV. Runs until stopped (Ctrl+C or _running=False).

    Args:
        prometheus_url: URL for Prometheus.
        namespace: Kubernetes namespace.
        deployment_name: Name of the deployment.
        output_path: Path to output CSV file.
        collect_interval: Seconds between each collection (e.g. 30 = collect every 30s).
        metrics_interval: Prometheus rate() interval in seconds.
        quantile: Response time quantile.
        endpoints_method: List of [endpoint, method] pairs.
        logger: Logger instance.

    Returns:
        Path to the output CSV file.
    """
    global _running
    _running = True

    if logger is None:
        logger = logging.getLogger(__name__)

    if endpoints_method is None:
        endpoints_method = [("/", "GET"), ("/docs", "GET")]

    prometheus = PrometheusConnect(url=prometheus_url, disable_ssl=True)
    config.load_kube_config()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output.exists()

    logger.info(f"Metrics collector started -> {output}")
    logger.info(
        f"  namespace={namespace}, deployment={deployment_name}, "
        f"collecting every {collect_interval}s"
    )

    count = 0

    # signal.signal() only works in the main thread; skip when called from a thread
    import threading

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    with open(output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()

        while _running:
            replicas = get_current_replicas(namespace, deployment_name, logger)
            row = collect_once(
                prometheus=prometheus,
                namespace=namespace,
                deployment_name=deployment_name,
                replicas=replicas,
                interval=metrics_interval,
                quantile=quantile,
                endpoints_method=endpoints_method,
                logger=logger,
            )

            if row:
                writer.writerow(row)
                f.flush()
                count += 1
                logger.info(
                    f"[{count}] CPU={row['cpu_usage']:.1f}%  MEM={row['memory_usage']:.1f}%  "
                    f"RT={row['response_time_ms']:.1f}ms  RPS={row['request_rate_rps']:.1f}  "
                    f"Replicas={row['replica_state']}"
                )
            else:
                logger.warning("No metrics collected this cycle, skipping row.")

            time.sleep(collect_interval)

    logger.info(f"Collector stopped. {count} rows written to {output}")
    return str(output)


if __name__ == "__main__":
    import ast

    logger = setup_logger(
        "metrics_collector",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_to_file=True,
    )

    try:
        endpoints_str = os.getenv(
            "METRICS_ENDPOINTS_METHOD", "[['/','GET'],['/docs','GET']]"
        )
        endpoints_method = ast.literal_eval(endpoints_str)
    except (ValueError, SyntaxError):
        endpoints_method = [("/", "GET"), ("/docs", "GET")]

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"metrics_output/metrics_{now}.csv"

    run_collector(
        prometheus_url=os.getenv("PROMETHEUS_URL", "http://localhost:1234/prom"),
        namespace=os.getenv("NAMESPACE", "default"),
        deployment_name=os.getenv("DEPLOYMENT_NAME", "ecom-api"),
        output_path=os.getenv("METRICS_OUTPUT_PATH", default_output),
        collect_interval=int(os.getenv("COLLECT_INTERVAL", "30")),
        metrics_interval=int(os.getenv("METRICS_INTERVAL", "15")),
        quantile=float(os.getenv("METRICS_QUANTILE", "0.90")),
        endpoints_method=endpoints_method,
        logger=logger,
    )

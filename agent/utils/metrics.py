import logging
import time

import numpy as np
from prometheus_api_client import PrometheusApiClientException, PrometheusConnect

def _build_scope_ready_query(namespace: str, deployment_name: str) -> str:
    """Build the base query for filtering ready pods in a deployment."""
    return f"""
    (
        (kube_pod_status_ready{{namespace="{namespace}", condition="true"}} == 1)
        and on(pod)
        (
            label_replace(
                kube_pod_owner{{namespace="{namespace}", owner_kind="ReplicaSet"}},
                "replicaset", "$1", "owner_name", "(.*)"
            )
            * on(namespace, replicaset) group_left(owner_name)
            kube_replicaset_owner{{
                namespace="{namespace}", owner_kind="Deployment",
                owner_name="{deployment_name}"
            }}
        )
    )
    """


def _build_cpu_usage_query(namespace: str, scope_ready: str, interval: int = 15) -> str:
    """Build CPU usage query for pods."""
    return f"""
        sum by (pod) (
        rate(container_cpu_usage_seconds_total{{
            namespace="{namespace}",
            container!="", container!="POD"
        }}[{interval}s])
        )
        AND on(pod)
        {scope_ready}
        """


def _build_memory_usage_query(namespace: str, scope_ready: str) -> str:
    """Build memory usage query for pods."""
    return f"""
        sum by (pod) (
            container_memory_working_set_bytes{{
                namespace="{namespace}",
                container!="",
                container!="POD"
            }}
        )
        AND on(pod)
        {scope_ready}
        """


def _build_cpu_limits_query(namespace: str, scope_ready: str) -> str:
    """Build CPU limits query for pods."""
    return f"""
        sum by (pod) (
            kube_pod_container_resource_limits{{
                namespace="{namespace}",
                resource="cpu",
                unit="core"
            }}
        )
        AND on(pod)
        {scope_ready}
        """


def _build_memory_limits_query(namespace: str, scope_ready: str) -> str:
    """Build memory limits query for pods."""
    return f"""
        sum by (pod) (
            kube_pod_container_resource_limits{{
                namespace="{namespace}",
                resource="memory",
                unit="byte"
            }}
        )
        AND on(pod)
        {scope_ready}
        """


def _build_request_rate_query(
    namespace: str, deployment_name: str, interval: int = 15
) -> str:
    """Build request rate query for the deployment."""
    return f"""
    sum(
        rate(app_requests_total{{
            namespace="{namespace}",
            pod=~"{deployment_name}-.*"
        }}[{interval}s])
    )
    """

def _extract_limits_by_pod(results: list) -> dict[str, float]:
    """Extract resource limits indexed by pod name."""
    limits_by_pod = {}
    for result in results:
        pod_name = result["metric"].get("pod")
        if pod_name:
            limits_by_pod[pod_name] = float(result["value"][1])
    return limits_by_pod


def _calculate_cpu_percentages(
    cpu_usage_results: list,
    cpu_limits_by_pod: dict[str, float],
    logger: logging.Logger = logging.getLogger(__name__),
) -> tuple[list[float], set[str]]:
    """Calculate CPU usage percentages for all pods."""
    cpu_percentages = []
    pod_names = set()

    for result in cpu_usage_results:
        pod_name = result["metric"].get("pod")
        if not pod_name or pod_name not in cpu_limits_by_pod:
            logger.warning(f"Skipping pod {pod_name}: CPU limit missing")
            continue

        rate_cores = float(result["value"][1])
        limit_cores = cpu_limits_by_pod[pod_name]

        if limit_cores <= 0:
            logger.warning(f"CPU limit not set or zero for pod {pod_name}")
            continue

        cpu_percentage = (rate_cores / limit_cores) * 100
        cpu_percentages.append(cpu_percentage)
        pod_names.add(pod_name)
        logger.debug(
            f"Pod {pod_name}: CPU {rate_cores:.4f} cores / "
            f"{limit_cores} -> {cpu_percentage:.2f}%"
        )

    return cpu_percentages, pod_names


def _calculate_memory_percentages(
    memory_usage_results: list,
    memory_limits_by_pod: dict[str, float],
    valid_pod_names: set[str],
    logger: logging.Logger = logging.getLogger(__name__),
) -> list[float]:
    """Calculate memory usage percentages for all valid pods."""
    memory_percentages = []

    for result in memory_usage_results:
        pod_name = result["metric"].get("pod")
        if (
            not pod_name
            or pod_name not in memory_limits_by_pod
            or pod_name not in valid_pod_names
        ):
            logger.warning(f"Skipping pod {pod_name}: Memory limit missing")
            continue

        used_bytes = float(result["value"][1])
        limit_bytes = memory_limits_by_pod[pod_name]

        if limit_bytes <= 0:
            logger.warning(f"Memory limit not set or zero for pod {pod_name}")
            continue

        memory_percentage = (used_bytes / limit_bytes) * 100.0
        memory_percentages.append(memory_percentage)
        logger.debug(
            f"Pod {pod_name}: Memory {used_bytes:.2f} bytes / "
            f"{limit_bytes} -> {memory_percentage:.2f}%"
        )

    return memory_percentages


def _metrics_result(
    cpu_limits_results: list,
    memory_limits_results: list,
    cpu_usage_results: list,
    memory_usage_results: list,
    logger: logging.Logger = logging.getLogger(__name__),
) -> tuple[list[float], list[float], set[str]]:
    """Process metrics results and calculate resource usage percentages."""
    # Extract limits for each pod
    cpu_limits_by_pod = _extract_limits_by_pod(cpu_limits_results)
    memory_limits_by_pod = _extract_limits_by_pod(memory_limits_results)

    # Calculate CPU percentages and get valid pod names
    cpu_percentages, pod_names = _calculate_cpu_percentages(
        cpu_usage_results, cpu_limits_by_pod, logger
    )

    # Calculate memory percentages only for pods with valid CPU metrics
    memory_percentages = _calculate_memory_percentages(
        memory_usage_results, memory_limits_by_pod, pod_names, logger
    )

    logger.debug(f"Pod names with metrics: {pod_names}")
    logger.debug(f"CPU percentages: {cpu_percentages}")
    logger.debug(f"Memory percentages: {memory_percentages}")

    return cpu_percentages, memory_percentages, pod_names


def _get_response_time(
    prometheus: PrometheusConnect,
    deployment_name: str,
    namespace: str = "default",
    endpoints_method: list[tuple[str, str]] = (("/", "GET"), ("/docs", "GET")),
    interval: int = 15,
    quantile: float = 0.90,
    logger: logging.Logger = logging.getLogger(__name__),
) -> float:
    result = []
    for endpoint, method in endpoints_method:
        q = f"""
            1000 *
            histogram_quantile(
            {quantile},
            sum by (le) (
                rate(app_request_latency_seconds_bucket{{
                job="{deployment_name}",
                namespace="{namespace}",
                method="{method}",
                exported_endpoint="{endpoint}"
                }}[{interval}s])
            )
            )

        """
        try:
            response = prometheus.custom_query(q)
            if response:
                for res in response:
                    if "value" in res and len(res["value"]) > 1:
                        result.append(float(res["value"][1]))
                    else:
                        result.append(0.0)
            else:
                response = 0.0
                result.append(response)
        except PrometheusApiClientException as e:
            if "404 page not found" in str(e):
                logger.warning(
                    f"Prometheus custom query returned 404 for app={deployment_name}, "
                    f"namespace={namespace}, endpoint={endpoint}, "
                    f"method={method}. Error: {e}"
                )
                result.append(0.0)
            else:
                logger.error(
                    f"Prometheus custom query failed for app={deployment_name}, "
                    f"namespace={namespace}, endpoint={endpoint}, "
                    f"method={method}. Error: {e}"
                )
                result.append(0.0)
        except Exception as e:
            logger.error(
                f"Prometheus custom query failed for app={deployment_name}, "
                f"namespace={namespace}, endpoint={endpoint}, "
                f"method={method}. Error: {e}"
            )
            result.append(0.0)

    response_time = float(np.mean(result)) if result else 0.0
    logger.debug(
        f"Response time (quantile {quantile}) for endpoints "
        f"{endpoints_method}: {response_time} ms"
    )

    return response_time


def _get_request_rate(
    prometheus: PrometheusConnect,
    rps_query: str,
    logger: logging.Logger = logging.getLogger(__name__),
) -> float:
    """Get current request rate (RPS) for the deployment."""
    try:
        result = prometheus.custom_query(rps_query)
        if result and len(result) > 0:
            rps = float(result[0]["value"][1])
            logger.debug(f"Current RPS: {rps:.2f}")
            return rps
        logger.debug("No RPS data available, returning 0.0")
        return 0.0
    except PrometheusApiClientException as e:
        logger.warning(f"Failed to fetch RPS: {e}. Returning 0.0")
        return 0.0
    except Exception as e:
        logger.error(f"Unexpected error fetching RPS: {e}. Returning 0.0")
        return 0.0


def _fetch_metric_with_retry(
    prometheus: PrometheusConnect,
    query: str,
    metric_name: str,
    expected_count: int,
    timeout: int,
    logger: logging.Logger = logging.getLogger(__name__),
) -> list:
    """Fetch a single metric with retry until timeout or expected count is reached."""
    fetch_start = time.time()

    while time.time() - fetch_start < timeout:
        results = prometheus.custom_query(query)
        logger.debug(f"Fetched {len(results)} {metric_name} entries")

        if len(results) != expected_count:
            logger.debug(
                f"Expected {expected_count} {metric_name} results, "
                f"got {len(results)}"
            )
            time.sleep(1)
            continue
        return results

    return results


def _scrape_metrics(
    fetch_timeout: int,
    prometheus: PrometheusConnect,
    cpu_query: str,
    memory_query: str,
    cpu_limits_query: str,
    memory_limits_query: str,
    replicas: int,
    logger: logging.Logger = logging.getLogger(__name__),
) -> tuple[list, list, list, list]:
    """Scrape all required metrics from Prometheus with retry logic."""
    cpu_usage_results = _fetch_metric_with_retry(
        prometheus, cpu_query, "CPU usage", replicas, fetch_timeout, logger
    )
    memory_usage_results = _fetch_metric_with_retry(
        prometheus, memory_query, "Memory usage", replicas, fetch_timeout, logger
    )
    cpu_limits_results = _fetch_metric_with_retry(
        prometheus, cpu_limits_query, "CPU limits", replicas, fetch_timeout, logger
    )
    memory_limits_results = _fetch_metric_with_retry(
        prometheus,
        memory_limits_query,
        "Memory limits",
        replicas,
        fetch_timeout,
        logger,
    )

    logger.debug(
        f"Fetched metrics: CPU usage {len(cpu_usage_results)} entries, "
        f"Memory usage {len(memory_usage_results)} entries"
    )
    logger.debug(
        f"Fetched metrics: CPU limits {len(cpu_limits_results)} entries, "
        f"Memory limits {len(memory_limits_results)} entries"
    )

    return (
        cpu_usage_results,
        memory_usage_results,
        cpu_limits_results,
        memory_limits_results,
    )


def get_metrics(
    replicas: int,
    timeout: int,
    namespace: str,
    deployment_name: str,
    wait_time: int,
    prometheus: PrometheusConnect,
    interval: int = 30,
    quantile: float = 0.90,
    endpoints_method: list[tuple[str, str]] = (("/", "GET"), ("/docs", "GET")),
    increase: bool = False,
    logger: logging.Logger = logging.getLogger(__name__),
) -> tuple[float, float, float, float, int]:
    """
    Collect metrics from Prometheus for a deployment.

    Returns:
        tuple: (cpu_mean, mem_mean, response_time, request_rate, collected_pods)
    """
    if increase or wait_time > 0:
        time.sleep(wait_time)

    start = time.time()
    while time.time() - start < timeout:
        try:
            prometheus.check_prometheus_connection()
        except Exception as e:
            logger.warning(f"Prometheus connectivity issue: {e}")
            time.sleep(1)
            continue

        # Build individual queries
        scope_ready = _build_scope_ready_query(namespace, deployment_name)
        cpu_query = _build_cpu_usage_query(namespace, scope_ready, interval)
        memory_query = _build_memory_usage_query(namespace, scope_ready)
        cpu_limits_query = _build_cpu_limits_query(namespace, scope_ready)
        memory_limits_query = _build_memory_limits_query(namespace, scope_ready)
        request_rate_query = _build_request_rate_query(namespace, deployment_name, interval)

        logger.debug("Metrics queries prepared, querying Prometheus...")
        logger.debug(f"CPU Query: {cpu_query}")
        logger.debug(f"Memory Query: {memory_query}")
        logger.debug(f"CPU Limits Query: {cpu_limits_query}")
        logger.debug(f"Memory Limits Query: {memory_limits_query}")

        try:
            fetch_timeout = timeout / 2

            (
                cpu_usage_results,
                memory_usage_results,
                cpu_limits_results,
                memory_limits_results,
            ) = _scrape_metrics(
                fetch_timeout=fetch_timeout,
                prometheus=prometheus,
                cpu_query=cpu_query,
                memory_query=memory_query,
                cpu_limits_query=cpu_limits_query,
                memory_limits_query=memory_limits_query,
                replicas=replicas,
                logger=logger,
            )
            if not cpu_usage_results or not memory_usage_results:
                logger.debug("No metrics found, retrying...")
                time.sleep(1)
                continue

            logger.debug(
                f"metrics found: CPU usage {len(cpu_usage_results)} entries, "
                f"Memory usage {len(memory_usage_results)} entries"
            )

            cpu_percentages, memory_percentages, pod_names = _metrics_result(
                cpu_limits_results,
                memory_limits_results,
                cpu_usage_results,
                memory_usage_results,
                logger=logger,
            )

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

            collected = len(pod_names)
            if collected == 0:
                logger.warning(
                    "No eligible pods (limits missing or no Ready pods). Retrying..."
                )
                time.sleep(1)
                continue

            if collected == replicas:
                cpu_mean = (
                    float(np.nanmean(cpu_percentages)) if cpu_percentages else 0.0
                )
                mem_mean = (
                    float(np.nanmean(memory_percentages)) if memory_percentages else 0.0
                )

                if not cpu_percentages:
                    logger.warning("No valid CPU percentages calculated.")
                if not memory_percentages:
                    logger.warning("No valid Memory percentages calculated.")

                logger.debug(
                    f"Metrics collected from {collected} pods: \n"
                    f"CPU usage mean {cpu_mean:.3f}%, \n"
                    f"Memory usage mean {mem_mean:.3f}%, \n"
                    f"Response time {response_time:.3f} ms, \n"
                    f"Request rate {request_rate:.3f} req/s"
                )
                return cpu_mean, mem_mean, response_time, request_rate, collected
            logger.warning(
                f"Only collected metrics from {collected} pods, expected {replicas}"
            )
            continue

        except PrometheusApiClientException as e:
            logger.error(f"Prometheus query failed: {e}")
        except Exception as e:
            logger.error(f"Error processing Prometheus metrics: {e}")

        time.sleep(1)

    logger.error("Timeout reached while fetching metrics.")
    return 0.0, 0.0, 0.0, 0.0, 0
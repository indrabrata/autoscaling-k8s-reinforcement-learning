import logging
import math
import time
from logging import Logger
from typing import Dict, Optional, Tuple, Union

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from prometheus_api_client import PrometheusConnect

from database.influxdb import InfluxDB
from rl.fuzzy import Fuzzy
from utils import get_metrics, wait_for_pods_ready


class KubernetesEnv:
    def __init__(
        self,
        min_replicas: int = 1,
        max_replicas: int = 50,
        total_episode: int = 30,
        iteration: int = 100,
        namespace: str = "default",
        deployment_name: str = "default",
        min_cpu: float = 20,
        min_memory: float = 20,
        max_cpu: float = 90,
        max_memory: float = 90,
        max_response_time: float = 100.0,
        timeout: int = 120,
        wait_time: int = 30,
        verbose: bool = False,
        logger: Optional[Logger] = None,
        influxdb: Optional[InfluxDB] = None,
        prometheus_url: str = "http://localhost:1234/prometheus",
        metrics_endpoints_method: list[tuple[str, str]] = [
            ("/", "GET"),
            ("/docs", "GET"),
        ],
        metrics_interval: int = 15,
        metrics_quantile: float = 0.90,
        max_scaling_retries: int = 1000,
        algorithm: str = "Q",
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        config.load_kube_config()
        self.cluster = client.AppsV1Api()
        self.api = client.CustomObjectsApi()
        self.core = client.CoreV1Api()
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.range_replicas = max(1, self.max_replicas - self.min_replicas)
        self.total_episode=total_episode
        self.iteration = iteration
        self.initial_iteration = iteration
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.min_cpu = min_cpu
        self.min_memory = min_memory
        self.max_cpu = max_cpu
        self.max_memory = max_memory
        self.max_response_time = max_response_time
        self.verbose = verbose
        self.timeout = timeout
        self.wait_time = wait_time
        self.last_replica = self.min_replicas
        self.influxdb = influxdb
        self.prometheus = PrometheusConnect(
            url=prometheus_url,
            disable_ssl=True,
        )
        self.metrics_endpoints_method = metrics_endpoints_method
        self.metrics_interval = metrics_interval
        self.metrics_quantile = metrics_quantile
        self.max_scaling_retries = max_scaling_retries

        self.action_space = list(range(1, self.max_replicas + 1))

        self.algorithm = algorithm

        # Track cumulative reward for sample efficiency analysis
        self.cumulative_reward = 0.0
        self.episode_number = 0
        self.episode_reward = (
            0.0  # Reset per episode - total reward for current episode only
        )

        if self.algorithm == "Q-LEARNING-FUZZY":
            self.fuzzy = Fuzzy(logger=logger, max_replicas=self.max_replicas)
        else:
            self.fuzzy = None

        self._log_environment_config()

    def _log_environment_config(self):
        """Log comprehensive environment configuration for debugging"""
        self.logger.info("=" * 100)
        self.logger.info("KUBERNETES ENVIRONMENT CONFIGURATION")
        self.logger.info("=" * 100)

        self.logger.info("📊 TRAINING PARAMETERS:")
        self.logger.info(f"  Algorithm:              {self.algorithm}")
        self.logger.info(f"  Episodes:               {self.total_episode}")
        self.logger.info(f"  Iteration:              {self.iteration}")
        self.logger.info(f"  Namespace:              {self.namespace}")
        self.logger.info(f"  Deployment:             {self.deployment_name}")

        self.logger.info("")
        self.logger.info("🎯 SCALING PARAMETERS:")
        self.logger.info(f"  Min Replicas:           {self.min_replicas}")
        self.logger.info(f"  Max Replicas:           {self.max_replicas}")
        self.logger.info(f"  Replica Range:          {self.range_replicas}")

        self.logger.info("")
        self.logger.info("📈 SLO THRESHOLD:")
        self.logger.info(f"  Max Response Time (SLO): {self.max_response_time}ms")

        self.logger.info("")
        self.logger.info("🚦 METRICS:")
        self.logger.info(f"  Metrics Interval:       {self.metrics_interval}s")
        self.logger.info(
            f"  Response Time Quantile: P{int(self.metrics_quantile * 100)}"
        )

        self.logger.info("")
        self.logger.info("⏱️  TIMING PARAMETERS:")
        self.logger.info(f"  Timeout:                {self.timeout}s")
        self.logger.info(f"  Wait Time:              {self.wait_time}s")
        self.logger.info(f"  Max Scaling Retries:    {self.max_scaling_retries}")

        self.logger.info("Environment initialized successfully!")

    def _scale(self) -> None:
        HTTP_INTERNAL_SERVER_ERROR = 500
        HTTP_CONFLICT = 409

        base_timeout = 60
        max_timeout = 300
        base_delay = 1.0
        max_delay = 30.0
        attempt = 0

        self.logger.info(
            f"Scaling to {self.replica_state} replicas | last_replica {self.last_replica}"
        )

        while attempt < self.max_scaling_retries:
            attempt += 1

            current_timeout = min(base_timeout * (1.5 ** (attempt - 1)), max_timeout)
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

            try:
                self.cluster.patch_namespaced_deployment_scale(
                    name=self.deployment_name,
                    body=client.V1Scale(
                        spec=client.V1ScaleSpec(replicas=int(self.replica_state))
                    ),
                    namespace=self.namespace,
                    _request_timeout=current_timeout,
                )

                if attempt > 1:
                    self.logger.info(
                        f"Scaling succeeded on attempt {attempt} "
                        f"(timeout: {current_timeout}s)"
                    )
                return

            except ApiException as e:
                if e.status == HTTP_INTERNAL_SERVER_ERROR:
                    if "etcdserver: request timed out" in str(e):
                        self.logger.warning(
                            f"Etcd timeout on attempt {attempt} "
                            f"(timeout: {current_timeout}s). "
                            f"Retrying in {delay:.1f}s..."
                        )
                    else:
                        self.logger.warning(
                            f"Server error on attempt {attempt}: {e.reason}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                elif e.status == HTTP_CONFLICT:
                    self.logger.warning(
                        f"Conflict on attempt {attempt} "
                        f"(likely concurrent modification). "
                        f"Retrying in {delay:.1f}s..."
                    )
                else:
                    self.logger.warning(
                        f"API error on attempt {attempt} "
                        f"(status: {e.status}): {e.reason}. "
                        f"Retrying in {delay:.1f}s..."
                    )

            except Exception as e:
                self.logger.warning(
                    f"Unexpected error on attempt {attempt}: {type(e).__name__}: "
                    f"{e}. "
                    f"Retrying in {delay:.1f}s..."
                )

            if attempt % 10 == 0:
                self.logger.info(
                    f"Still retrying scaling operation... "
                    f"Attempt {attempt}, next timeout: {current_timeout}s"
                )

            time.sleep(delay)

        self.logger.error(
            f"CRITICAL: Failed to scale after {self.max_scaling_retries} attempts. "
            f"This indicates a serious cluster issue. "
            f"Proceeding with current replica state to avoid blocking training."
        )

    def _calculate_reward_qlearning(self) -> Tuple[float, Dict[str, Union[float, str]]]:
            """
            UNIFIED REWARD FUNCTION - Used by BOTH Q-Learning and Q-Fuzzy

            Reward formula:
                R_total = R_rt - R_cost

            Components:
            1. R_rt   = 1 - (rt / rt_max)  — SLO compliance reward
            2. R_cost = (R - R_min) / (R_max - R_min) — Replica cost penalty

            Design principles:
            - R_rt: Positive when response time is below SLO, negative when above.
                    Provides continuous gradient — agent always incentivized to minimize RT.
            - R_cost: 0.0 at min replicas (cheapest), 1.0 at max replicas (most expensive).
                    Subtracted from R_rt to penalize over-provisioning.
            - No weights needed: both components naturally operate in comparable ranges.
            - No edge-case overrides: formula handles boundaries naturally
            (R_cost = 0 at R_min, so only R_rt drives reward).

            Reference:
            - SLO preservation ratio inspired by Qiu et al. (USENIX OSDI, 2020)
            - Inverse replica normalization based on Zhong (UvA, 2023)
            - Scalarized multi-objective RL normalization per Rossi et al. (IEEE CLOUD, 2019)

            Returns:
                tuple: (reward, breakdown_dict)
            """
            if (
                self.response_time is None
                or math.isnan(self.response_time)
                or math.isinf(self.response_time)
            ):
                self.response_time = 0.0

            # ========================================================================================================
            # COMPONENT 1: RESPONSE TIME REWARD (R_rt)
            # ========================================================================================================
            # R_rt = 1 - (rt / rt_max)
            #
            # Measures SLO compliance. The value 1 is the maximum reward, normalized so
            # that R_rt operates in the same scale as R_cost.
            #
            # - rt = 0ms      → R_rt = +1.0  (best possible response time)
            # - rt = rt_max    → R_rt =  0.0  (exactly at SLO threshold)
            # - rt = 2×rt_max  → R_rt = -1.0  (SLO violated, strong penalty)
            #
            # Continuous gradient ensures agent always has incentive to minimize
            # response time, avoiding zero-gradient dead zones.
            # ========================================================================================================
            r_rt: float = 1.0 - (self.response_time / self.max_response_time)

            # ========================================================================================================
            # COMPONENT 2: REPLICA COST PENALTY (R_cost)
            # ========================================================================================================
            # R_cost = (R - R_min) / (R_max - R_min)
            #
            # Measures operational cost based on replica count.
            #
            # - R = R_min  → R_cost = 0.0  (most efficient, no cost penalty)
            # - R = R_max  → R_cost = 1.0  (most expensive, maximum cost penalty)
            #
            # At R_min, cost is zero — agent is not penalized for something it
            # cannot reduce further. This provides natural boundary handling
            # without explicit edge-case overrides.
            # ========================================================================================================
            r_cost: float = (
                self.replica_state - self.min_replicas
            ) / self.range_replicas

            # ========================================================================================================
            # TOTAL REWARD
            # ========================================================================================================
            # R_total = R_rt - R_cost
            #
            # No weights needed — both components are normalized to comparable ranges:
            #   R_rt   ∈ (-∞, +1.0]  (typically [-1.0, +1.0] under normal conditions)
            #   R_cost ∈ [0.0, +1.0]
            #
            # Best case:  low RT + low replicas  → R_rt ≈ +1.0, R_cost ≈ 0.0 → R ≈ +1.0
            # Worst case: high RT + max replicas → R_rt < 0.0,  R_cost = 1.0 → R << 0.0
            # ========================================================================================================
            reward: float = r_rt - r_cost


            self.logger.info(
                f"[REWARD] Iter={getattr(self, 'iteration', '?')} | "
                f"Replicas={self.replica_state} | "
                f"Reward={reward:.4f}"
            )
            self.logger.info(
                f"  Components: R_rt={r_rt:.4f} R_cost={r_cost:.4f} | "
                f"RT={self.response_time:.1f}ms "
                f"({self.response_time / self.max_response_time * 100:.1f}% of SLO)"
            )

 
            return reward, {
                "reward": reward,
                "r_rt": r_rt,
                "r_cost": r_cost,
                "response_time_ms": self.response_time,
                "response_time_percentage": self.response_time / self.max_response_time * 100.0,
                "replica_count": self.replica_state,
            }


    def _calculate_reward(self) -> Tuple[float, Dict[str, Union[float, str]]]:
        """
        Unified reward calculation for both Q-Learning and Q-Fuzzy.

        IMPORTANT: Both algorithms now use the SAME reward function!
        The ONLY difference is state representation:
        - Q-Learning: Stores Q-values for raw continuous states
        - Q-Fuzzy: Stores Q-values for fuzzified states

        This ensures fair comparison - same reward signal, different state abstraction.
        """
        # Both algorithms use the same reward function now
        return self._calculate_reward_qlearning()

    def _scale_and_get_metrics(self) -> None:
        self._scale()
        ready, desired_replicas, ready_replicas = wait_for_pods_ready(
            prometheus=self.prometheus,
            deployment_name=self.deployment_name,
            desired_replicas=self.replica_state,
            namespace=self.namespace,
            timeout=self.timeout,
            logger=self.logger,
        )
        (
            self.cpu_usage,
            self.memory_usage,
            self.response_time,
            _,  # request_rate (not used in reward/state)
            self.replica,
        ) = get_metrics(
            replicas=ready_replicas,
            timeout=self.timeout,
            namespace=self.namespace,
            deployment_name=self.deployment_name,
            wait_time=self.wait_time,
            prometheus=self.prometheus,
            interval=self.metrics_interval,
            quantile=self.metrics_quantile,
            endpoints_method=self.metrics_endpoints_method,
            logger=self.logger,
        )

        if not ready:
            self.logger.warning(
                f"Pods are not ready, {ready_replicas}/{desired_replicas} ready"
            )

    def _get_observation(self) -> dict[str, Union[float, str]]:
        """
        Returns observation dict used by RL agents.

        State key fields (used by agents for Q-table indexing):
        - cpu_usage, memory_usage, response_time, last_replica

        Auxiliary fields (for logging only, NOT in state key):
        - response_time_ms, current_replicas
        """
        response_time_percentage: float = min(
            (self.response_time / self.max_response_time) * 100.0, 100.0
        )

        return {
            # Core resource metrics (0-100%)
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "response_time": response_time_percentage,
            "response_time_ms": self.response_time,  # Raw value in milliseconds for logging
            # Action and system state
            "last_replica": self.last_replica,  # Previous replica count (1 to max_replicas)
            "current_replicas": float(self.replica_state),  # Absolute replica count
        }

    def step(
        self, action: int, q_table_size: int = 0
    ) -> tuple[dict[str, Union[float, str]], float, bool, dict]:
        # Store current action as last_replica before executing
        self.last_replica = action

        self.replica_state_old = self.replica_state
        self.replica_state = max(self.min_replicas, min(action, self.max_replicas))

        self._scale_and_get_metrics()

        reward, reward_breakdown = self._calculate_reward()

        self.iteration -= 1
        terminated = bool(self.iteration <= 0)

        observation = self._get_observation()

        # Includes all state variables, reward breakdown, and derived metrics
        info = {
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "iteration": self.iteration,
            # System state
            "replica_state": self.replica_state,
            # Core metrics - raw values
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "response_time": self.response_time,
            # Percentages
            "response_time_percentage": reward_breakdown.get(
                "response_time_percentage", 0.0
            ),
            # Include full breakdown for observation (not saved to InfluxDB)
            **observation,
            **reward_breakdown,
        }

        # Clean info for InfluxDB - only essential metrics
        # Accumulate rewards for sample efficiency tracking
        self.cumulative_reward += reward
        self.episode_reward += reward

        influxdb_fields = {
            "action": action,
            "reward": reward,
            "cumulative_reward": self.cumulative_reward,
            "episode_reward": self.episode_reward,
            "episode_number": self.episode_number,
            "q_table_size": q_table_size,
            "terminated": int(terminated),
            "iteration": self.iteration,
            "replica_state": self.replica_state,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "response_time": self.response_time,
            "response_time_percentage": reward_breakdown.get(
                "response_time_percentage", 0.0
            ),
        }

        if self.influxdb:
            self.influxdb.write_point(
                measurement="autoscaling_metrics",
                tags={
                    "namespace": self.namespace,
                    "deployment": self.deployment_name,
                    "algorithm": self.algorithm,
                },
                fields=influxdb_fields,
            )
        return observation, reward, terminated, info

    def reset(self) -> dict[str, Union[float, str]]:
        """
        CRITICAL: Reset environment to initial state for new episode.
        Clears action history and resets all state variables.
        """
        self.iteration = self.initial_iteration
        self.replica_state_old = (
            self.replica_state if hasattr(self, "replica_state") else self.min_replicas
        )
        self.replica_state = self.min_replicas
        self._scale_and_get_metrics()
        self.last_replica = self.min_replicas

        self.episode_reward = 0.0  # Reset for new episode
        self.episode_number += 1

        return self._get_observation()

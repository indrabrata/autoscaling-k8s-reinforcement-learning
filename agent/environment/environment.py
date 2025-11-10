import math
import time
from logging import Logger
from typing import Dict, Optional, Tuple

from database.influxdb import InfluxDB
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from prometheus_api_client import PrometheusConnect
from rl.fuzzy import Fuzzy
from utils import get_metrics, wait_for_pods_ready

class KubernetesEnv:
    def __init__(
        self,
        min_replicas: int = 1,
        max_replicas: int = 50,
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
        prometheus_url: str = "http://localhost:1234/prom",
        metrics_endpoints_method: list[tuple[str, str]] = (
            ("/", "GET"),
            ("/docs", "GET"),
        ),
        metrics_interval: int = 15,
        metrics_quantile: float = 0.90,
        max_scaling_retries: int = 1000,
        response_time_weight: float = 1.0,
        cpu_memory_weight: float = 0.5,
        cost_weight: float = 0.3,
        algorithm: str = "Q"
    ) -> None:
        self.logger = logger
        config.load_kube_config()
        self.cluster = client.AppsV1Api()
        self.api = client.CustomObjectsApi()
        self.core = client.CoreV1Api()
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.range_replicas = max(1, self.max_replicas - self.min_replicas)
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
        self.last_action = 0
        self.influxdb = influxdb
        self.prometheus = PrometheusConnect(
            url=prometheus_url,
            disable_ssl=True,
        )
        self.metrics_endpoints_method = metrics_endpoints_method
        self.metrics_interval = metrics_interval
        self.metrics_quantile = metrics_quantile
        self.max_scaling_retries = max_scaling_retries

        self.action_space = list(range(100))
        self.response_time_weight = response_time_weight
        self.cpu_memory_weight = cpu_memory_weight
        self.cost_weight = cost_weight

        self.algorithm = algorithm

        # Initialize state tracking variables
        self.request_rate = 0.0
        self.previous_request_rate = 0.0
        self.request_rate_trend = 0.0
        self.action_change = 0

        if self.algorithm == "Q-LEARNING-FUZZY":
            self.fuzzy = Fuzzy(logger=logger)
        else:
            self.fuzzy = None

        self.logger.info("Initialized KubernetesEnv environment")
        self.logger.debug(f"Environment configuration: {self.__dict__}")

    def _scale(self) -> None:
        HTTP_INTERNAL_SERVER_ERROR = 500
        HTTP_CONFLICT = 409

        base_timeout = 60
        max_timeout = 300
        base_delay = 1.0
        max_delay = 30.0
        attempt = 0

        self.logger.info(
            f"Scaling to {self.replica_state} replicas | action {self.last_action}%"
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
                        f"✅ Scaling succeeded on attempt {attempt} "
                        f"(timeout: {current_timeout}s)"
                    )
                return

            except ApiException as e:
                if e.status == HTTP_INTERNAL_SERVER_ERROR:
                    if "etcdserver: request timed out" in str(e):
                        self.logger.warning(
                            f"⏰ Etcd timeout on attempt {attempt} "
                            f"(timeout: {current_timeout}s). "
                            f"Retrying in {delay:.1f}s..."
                        )
                    else:
                        self.logger.warning(
                            f"🔄 Server error on attempt {attempt}: {e.reason}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                elif e.status == HTTP_CONFLICT:
                    self.logger.warning(
                        f"⚠️  Conflict on attempt {attempt} "
                        f"(likely concurrent modification). "
                        f"Retrying in {delay:.1f}s..."
                    )
                else:
                    self.logger.warning(
                        f"🚨 API error on attempt {attempt} "
                        f"(status: {e.status}): {e.reason}. "
                        f"Retrying in {delay:.1f}s..."
                    )

            except Exception as e:
                self.logger.warning(
                    f"💥 Unexpected error on attempt {attempt}: {type(e).__name__}: "
                    f"{e}. "
                    f"Retrying in {delay:.1f}s..."
                )

            if attempt % 10 == 0:
                self.logger.info(
                    f"🔄 Still retrying scaling operation... "
                    f"Attempt {attempt}, next timeout: {current_timeout}s"
                )

            time.sleep(delay)

        self.logger.error(
            f"❌ CRITICAL: Failed to scale after {self.max_scaling_retries} attempts. "
            f"This indicates a serious cluster issue. "
            f"Proceeding with current replica state to avoid blocking training."
        )

    def _calculate_reward_qlearning(self) -> Tuple[float, Dict[str, float]]:
        """Calculate reward for Q-Learning algorithm using direct threshold evaluations"""
        if self.response_time is None or math.isnan(self.response_time) or math.isinf(self.response_time):
            self.response_time = 0.0

        response_time_percentage = (self.response_time / self.max_response_time) * 100.0
        replica_ratio = (self.replica_state - self.min_replicas) / self.range_replicas

        # Evaluate state quality using direct thresholds (similar to fuzzy but with real values)

        # OPTIMAL STATE: Resources well-utilized, response time excellent
        # CPU/MEM in target range, response time low
        cpu_in_range = self.min_cpu <= self.cpu_usage <= self.max_cpu
        mem_in_range = self.min_memory <= self.memory_usage <= self.max_memory
        resp_excellent = response_time_percentage <= 60.0  # < 60% of max
        resp_good = response_time_percentage <= 80.0  # < 80% of max

        if cpu_in_range and mem_in_range and resp_excellent:
            optimal_score = 1.0
        elif cpu_in_range and mem_in_range and resp_good:
            optimal_score = 0.7
        elif (self.min_cpu <= self.cpu_usage <= self.max_cpu * 1.1) and \
             (self.min_memory <= self.memory_usage <= self.max_memory * 1.1) and resp_good:
            optimal_score = 0.5
        else:
            optimal_score = 0.0

        # BALANCED STATE: Stable and efficient operation
        # Metrics moderate, response time acceptable
        cpu_moderate = 40 <= self.cpu_usage <= 70
        mem_moderate = 40 <= self.memory_usage <= 70
        resp_acceptable = response_time_percentage <= 90.0

        if cpu_moderate and mem_moderate and resp_good:
            balanced_score = 1.0
        elif cpu_moderate and mem_moderate and resp_acceptable:
            balanced_score = 0.7
        elif (30 <= self.cpu_usage <= 80) and (30 <= self.memory_usage <= 80) and resp_acceptable:
            balanced_score = 0.5
        else:
            balanced_score = 0.0

        # WASTEFUL STATE: Over-provisioned, resources under-utilized
        # CPU/MEM very low, response time low = wasting resources
        cpu_very_low = self.cpu_usage < self.min_cpu * 0.5  # Less than half of min threshold
        mem_very_low = self.memory_usage < self.min_memory * 0.5
        cpu_low = self.cpu_usage < self.min_cpu
        mem_low = self.memory_usage < self.min_memory

        if cpu_very_low and mem_very_low and resp_excellent:
            wasteful_score = 1.0  # Severe waste
        elif (cpu_very_low or mem_very_low) and resp_excellent:
            wasteful_score = 0.8
        elif cpu_low and mem_low and resp_good:
            wasteful_score = 0.6
        elif (cpu_low or mem_low) and resp_excellent:
            wasteful_score = 0.4
        else:
            wasteful_score = 0.0

        # CRITICAL STATE: Under-provisioned, performance degrading
        # Response time very high, or CPU/MEM maxed out
        cpu_very_high = self.cpu_usage > self.max_cpu * 1.1  # Exceeding max by 10%
        mem_very_high = self.memory_usage > self.max_memory * 1.1
        cpu_high = self.cpu_usage > self.max_cpu
        mem_high = self.memory_usage > self.max_memory
        resp_critical = response_time_percentage > 120.0  # > 120% of max
        resp_high = response_time_percentage > 100.0

        if resp_critical:
            critical_score = 1.0  # Very critical
        elif (cpu_very_high or mem_very_high) and resp_high:
            critical_score = 1.0
        elif resp_high and (cpu_high or mem_high):
            critical_score = 0.8
        elif (cpu_very_high or mem_very_high):
            critical_score = 0.6
        elif resp_high:
            critical_score = 0.5
        else:
            critical_score = 0.0

        # Calculate reward based on state quality
        optimal_contribution = optimal_score * 1.0
        balanced_contribution = balanced_score * 0.7
        wasteful_penalty = wasteful_score * self.cost_weight * 1.5
        critical_penalty = critical_score * (self.response_time_weight + self.cpu_memory_weight)

        positive_contribution = optimal_contribution + balanced_contribution
        negative_contribution = wasteful_penalty + critical_penalty

        # Base reward formula
        if negative_contribution > 0:
            reward = positive_contribution / (1.0 + negative_contribution)
        else:
            reward = positive_contribution

        # Additional cost penalty based on replica ratio and state
        # Penalize high replicas in wasteful state more
        if wasteful_score > 0.5 and replica_ratio > 0.6:
            cost_factor = 1.8
            cost_pen = self.cost_weight * cost_factor * replica_ratio
            reward -= cost_pen * 0.3
        # Reduce penalty for high replicas if system is critical
        elif critical_score > 0.5 and replica_ratio > 0.5:
            cost_factor = 0.2
            cost_pen = self.cost_weight * cost_factor * replica_ratio
            reward -= cost_pen * 0.1
        else:
            cost_factor = 1.0
            cost_pen = self.cost_weight * cost_factor * replica_ratio * 0.2
            reward -= cost_pen

        # Bonus for achieving optimal state with efficient replica usage
        if optimal_score > 0.6 and 0.3 <= replica_ratio <= 0.7:
            reward += 0.05

        # Bonus for stable balanced state
        if balanced_score > 0.7:
            reward += 0.03

        # Clamp reward to [0, 1]
        reward = max(min(reward, 1.0), 0.0)

        # Calculate individual penalties for logging
        if self.cpu_usage < self.min_cpu:
            cpu_pen = (self.min_cpu - self.cpu_usage) / self.min_cpu
        elif self.cpu_usage > self.max_cpu:
            cpu_pen = (self.cpu_usage - self.max_cpu) / (100 - self.max_cpu)
        else:
            cpu_pen = 0.0

        if self.memory_usage < self.min_memory:
            mem_pen = (self.min_memory - self.memory_usage) / self.min_memory
        elif self.memory_usage > self.max_memory:
            mem_pen = (self.memory_usage - self.max_memory) / (100 - self.max_memory)
        else:
            mem_pen = 0.0

        if response_time_percentage <= 100.0:
            resp_pen = 0.0
        else:
            resp_pen = min(1.0, (response_time_percentage - 100.0) / 100.0)

        cpu_mem_pen = self.cpu_memory_weight * (cpu_pen + mem_pen)
        total_penalty = negative_contribution

        self.logger.info(
        f"📊 Q-Learning Reward Breakdown | Iter={getattr(self, 'iteration', '?')} | Replicas={self.replica_state}\n"
        f" ├─ State Quality: Optimal={optimal_score:.2f}, Balanced={balanced_score:.2f}, "
        f"Wasteful={wasteful_score:.2f}, Critical={critical_score:.2f}\n"
        f" ├─ Positive Contribution: {positive_contribution:.4f} (Optimal={optimal_contribution:.4f}, Balanced={balanced_contribution:.4f})\n"
        f" ├─ Negative Contribution: {negative_contribution:.4f} (Wasteful={wasteful_penalty:.4f}, Critical={critical_penalty:.4f})\n"
        f" ├─ CPU Usage: {self.cpu_usage:.2f}% | Penalty={cpu_pen:.4f}\n"
        f" ├─ MEM Usage: {self.memory_usage:.2f}% | Penalty={mem_pen:.4f}\n"
        f" ├─ Response Time: {self.response_time:.2f} ms ({response_time_percentage:.2f}%) | Penalty={resp_pen:.4f}\n"
        f" ├─ Replica Ratio: {replica_ratio:.3f} | CostFactor={cost_factor:.2f} | CostPen={cost_pen:.4f}\n"
        f" └─ ✅ Final Reward: {reward:.4f}"
    )

        return reward, {
            "reward": reward,
            "cpu_penalty": cpu_pen,
            "memory_penalty": mem_pen,
            "response_time_penalty": resp_pen,
            "cpu_memory_penalty": cpu_mem_pen,
            "cost_penalty": cost_pen,
            "cost_factor": cost_factor,
            "total_penalty": total_penalty,
            "replica_ratio": replica_ratio,
            "response_time_percentage": response_time_percentage,
            "optimal": optimal_score,
            "balanced": balanced_score,
            "wasteful": wasteful_score,
            "critical": critical_score,
            "positive_contribution": positive_contribution,
            "negative_contribution": negative_contribution,
        }

    def _calculate_reward_fuzzy(self) -> Tuple[float, Dict[str, float]]:
        """Calculate reward for Q-Learning Fuzzy algorithm using fuzzification"""
        if self.response_time is None or math.isnan(self.response_time) or math.isinf(self.response_time):
            self.response_time = 0.0

        response_time_percentage = (self.response_time / self.max_response_time) * 100.0

        # Prepare observation for fuzzification
        observation = {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "response_time": response_time_percentage,
        }

        # Fuzzify the metrics
        fuzzy_state = self.fuzzy.fuzzify(observation)

        # Apply reward-specific fuzzy rules to evaluate state quality
        reward_state = self.fuzzy.apply_reward_rules(fuzzy_state)

        # Also get scaling decisions for additional context
        fuzzy_actions = self.fuzzy.apply_rules(fuzzy_state)
        fuzzy_influence = self.fuzzy.influence(fuzzy_actions)

        # Get fuzzy memberships for logging
        cpu_fz = fuzzy_state["cpu_usage"]
        mem_fz = fuzzy_state["memory_usage"]
        resp_fz = fuzzy_state["response_time"]

        # Calculate reward based on state quality
        # Optimal state gives positive contribution
        optimal_contribution = reward_state["optimal"] * 1.0

        # Balanced state gives moderate positive contribution
        balanced_contribution = reward_state["balanced"] * 0.7

        # Wasteful state creates penalty (over-provisioned)
        wasteful_penalty = reward_state["wasteful"] * self.cost_weight * 1.5

        # Critical state creates strong penalty (under-provisioned, performance issues)
        critical_penalty = reward_state["critical"] * (self.response_time_weight + self.cpu_memory_weight)

        # Combine contributions
        positive_contribution = optimal_contribution + balanced_contribution
        negative_contribution = wasteful_penalty + critical_penalty

        # Base reward formula
        if negative_contribution > 0:
            reward = positive_contribution / (1.0 + negative_contribution)
        else:
            reward = positive_contribution

        # Replica ratio for cost consideration
        replica_ratio = (self.replica_state - self.min_replicas) / self.range_replicas

        # Additional cost penalty based on replica ratio and state
        # Penalize high replicas in wasteful state more
        if reward_state["wasteful"] > 0.5 and replica_ratio > 0.6:
            cost_factor = 1.8
            cost_pen = self.cost_weight * cost_factor * replica_ratio
            reward -= cost_pen * 0.3
        # Reduce penalty for high replicas if system is critical
        elif reward_state["critical"] > 0.5 and replica_ratio > 0.5:
            cost_factor = 0.2
            cost_pen = self.cost_weight * cost_factor * replica_ratio
            reward -= cost_pen * 0.1
        else:
            cost_factor = 1.0
            cost_pen = self.cost_weight * cost_factor * replica_ratio * 0.2
            reward -= cost_pen

        # Bonus for achieving optimal state with efficient replica usage
        if reward_state["optimal"] > 0.6 and 0.3 <= replica_ratio <= 0.7:
            reward += 0.05

        # Bonus for stable balanced state
        if reward_state["balanced"] > 0.7:
            reward += 0.03

        # Clamp reward to [0, 1]
        reward = max(min(reward, 1.0), 0.0)

        # Calculate individual penalties for logging compatibility
        cpu_pen = (
            cpu_fz.get("very_high", 0.0) * 1.0 +
            cpu_fz.get("high", 0.0) * 0.7 +
            cpu_fz.get("very_low", 0.0) * 0.5
        )
        mem_pen = (
            mem_fz.get("very_high", 0.0) * 1.0 +
            mem_fz.get("high", 0.0) * 0.7 +
            mem_fz.get("very_low", 0.0) * 0.5
        )
        resp_pen = (
            resp_fz.get("very_high", 0.0) * 1.0 +
            resp_fz.get("high", 0.0) * 0.8
        )
        cpu_mem_pen = self.cpu_memory_weight * (cpu_pen + mem_pen)
        total_penalty = negative_contribution

        self.logger.info(
        f"📊 Fuzzy Reward Breakdown | Iter={getattr(self, 'iteration', '?')} | Replicas={self.replica_state}\n"
        f" ├─ State Quality: Optimal={reward_state['optimal']:.2f}, Balanced={reward_state['balanced']:.2f}, "
        f"Wasteful={reward_state['wasteful']:.2f}, Critical={reward_state['critical']:.2f}\n"
        f" ├─ Positive Contribution: {positive_contribution:.4f} (Optimal={optimal_contribution:.4f}, Balanced={balanced_contribution:.4f})\n"
        f" ├─ Negative Contribution: {negative_contribution:.4f} (Wasteful={wasteful_penalty:.4f}, Critical={critical_penalty:.4f})\n"
        f" ├─ CPU Fuzzy: vh={cpu_fz.get('very_high', 0):.2f}, h={cpu_fz.get('high', 0):.2f}, "
        f"m={cpu_fz.get('medium', 0):.2f}, l={cpu_fz.get('low', 0):.2f}, vl={cpu_fz.get('very_low', 0):.2f}\n"
        f" ├─ MEM Fuzzy: vh={mem_fz.get('very_high', 0):.2f}, h={mem_fz.get('high', 0):.2f}, "
        f"m={mem_fz.get('medium', 0):.2f}, l={mem_fz.get('low', 0):.2f}, vl={mem_fz.get('very_low', 0):.2f}\n"
        f" ├─ RESP Fuzzy: vh={resp_fz.get('very_high', 0):.2f}, h={resp_fz.get('high', 0):.2f}, "
        f"m={resp_fz.get('medium', 0):.2f}, l={resp_fz.get('low', 0):.2f}, vl={resp_fz.get('very_low', 0):.2f}\n"
        f" ├─ Fuzzy Actions: up={fuzzy_actions['scale_up']:.2f}, down={fuzzy_actions['scale_down']:.2f}, "
        f"stay={fuzzy_actions['no_change']:.2f} | Influence={fuzzy_influence:.3f}\n"
        f" ├─ Replica Ratio: {replica_ratio:.3f} | CostFactor={cost_factor:.2f} | CostPen={cost_pen:.4f}\n"
        f" └─ ✅ Final Reward: {reward:.4f}"
    )

        return reward, {
            "reward": reward,
            "cpu_penalty": cpu_pen,
            "memory_penalty": mem_pen,
            "response_time_penalty": resp_pen,
            "cpu_memory_penalty": cpu_mem_pen,
            "cost_penalty": cost_pen,
            "cost_factor": cost_factor,
            "total_penalty": total_penalty,
            "replica_ratio": replica_ratio,
            "response_time_percentage": response_time_percentage,
            "fuzzy_scale_up": fuzzy_actions["scale_up"],
            "fuzzy_scale_down": fuzzy_actions["scale_down"],
            "fuzzy_no_change": fuzzy_actions["no_change"],
            "fuzzy_influence": fuzzy_influence,
            "fuzzy_optimal": reward_state["optimal"],
            "fuzzy_balanced": reward_state["balanced"],
            "fuzzy_wasteful": reward_state["wasteful"],
            "fuzzy_critical": reward_state["critical"],
            "fuzzy_positive_contribution": positive_contribution,
            "fuzzy_negative_contribution": negative_contribution,
        }

    def _calculate_reward(self) -> Tuple[float, Dict[str, float]]:
        """Route to appropriate reward calculation based on algorithm"""
        if self.algorithm == "Q-LEARNING-FUZZY":
            return self._calculate_reward_fuzzy()
        else:
            return self._calculate_reward_qlearning()


    def _scale_and_get_metrics(self) -> None:
        self._scale()
        increase: int = self.replica_state > self.replica_state_old
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
            self.request_rate,
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
            increase=increase,
            logger=self.logger,
        )

        if not ready:
            self.logger.warning(
                f"Pods are not ready, {ready_replicas}/{desired_replicas} ready"
            )

    def _get_observation(self) -> dict[str, float]:
        response_time_percentage = min(
            (self.response_time / self.max_response_time) * 100.0, 100.0
        )

        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "response_time": response_time_percentage,
            "request_rate": self.request_rate,
            "request_rate_trend": self.request_rate_trend,
            "action_change": self.action_change,
            "last_action": self.last_action,
        }

    def step(self, action: int) -> tuple[dict[str, float], float, bool, dict]:
        # Calculate action change before updating last_action
        self.action_change = action - self.last_action
        self.last_action = action

        self.previous_request_rate = self.request_rate

        percentage = (
            (action / 99.0) if len(self.action_space) > 1 else 0.0
        )
        self.replica_state_old = self.replica_state
        self.replica_state = round(self.min_replicas + percentage * self.range_replicas)
        self.replica_state = max(
            self.min_replicas, min(self.replica_state, self.max_replicas)
        )

        self._scale_and_get_metrics()

        self.request_rate_trend = self.request_rate - self.previous_request_rate

        reward, reward_breakdown = self._calculate_reward()

        self.iteration -= 1
        terminated = bool(self.iteration <= 0)

        observation = self._get_observation()
        info = {
            "iteration": self.iteration,
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "replica_state": self.replica_state,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "response_time": self.response_time,
            "request_rate": self.request_rate,
            "request_rate_trend": self.request_rate_trend,
            "action_change": self.action_change,
            "last_action": self.last_action,
            **reward_breakdown,  # Include detailed reward breakdown
        }
        self.influxdb.write_point(
            measurement="autoscaling_metrics",
            tags={
                "namespace": self.namespace,
                "deployment": self.deployment_name,
                "algorithm" : self.algorithm
            },
            fields={**info},
        ) if self.influxdb else None
        return observation, reward, terminated, info

    def reset(self) -> dict[str, float]:
        self.iteration = self.initial_iteration
        self.replica_state_old = (
            self.replica_state if hasattr(self, "replica_state") else self.min_replicas
        )
        self.replica_state = self.min_replicas
        self._scale_and_get_metrics()
        self.last_action = 0
        return self._get_observation()
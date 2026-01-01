import math
import time
from logging import Logger
from typing import Dict, Optional, Tuple

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
        request_rate_per_pod_capacity: float = 80.0,
        algorithm: str = "Q",
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

        # This value should be determined through load testing on a single pod
        # See: k6 capacity test script for finding this value
        self.per_pod_capacity = (
            request_rate_per_pod_capacity  # requests per second that one pod can handle
        )

        # Initialize state tracking variables
        self.request_rate = 0.0
        self.previous_request_rate = 0.0
        self.request_rate_trend = 0.0
        self.action_change = 0

        # Used to detect scaling patterns (increasing, decreasing, stable, oscillating)
        self.action_history = []
        self.action_trend_window = 5  # Number of previous actions to consider for trend

        # Track cumulative reward for sample efficiency analysis
        self.cumulative_reward = 0.0
        self.episode_number = 0
        self.episode_reward = 0.0  # Reset per episode - total reward for current episode only

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

    def _calculate_reward_qlearning(self) -> Tuple[float, Dict[str, float]]:
        """
        Considers resource utilization, request rate handling, and scaling behavior.
        """
        if (
            self.response_time is None
            or math.isnan(self.response_time)
            or math.isinf(self.response_time)
        ):
            self.response_time = 0.0

        response_time_percentage: float = (
            self.response_time / self.max_response_time
        ) * 100.0
        replica_ratio: float = (
            self.replica_state - self.min_replicas
        ) / self.range_replicas

        current_capacity: float = self.per_pod_capacity * self.replica_state
        request_rate_normalized: float = min(
            100.0, (self.request_rate / current_capacity) * 100.0
        )

        observation: dict = self._get_observation()
        request_trend: str = observation.get("request_rate_trend_category", "stable")
        action_trend: str = observation.get("action_trend_category", "stable")

        # OPTIMAL STATE: Resources well-utilized, response time excellent, request rate handled well
        # CPU/MEM in target range, response time low, request rate not saturating
        cpu_in_range = self.min_cpu <= self.cpu_usage <= self.max_cpu
        mem_in_range = self.min_memory <= self.memory_usage <= self.max_memory
        resp_excellent = response_time_percentage <= 60.0  # < 60% of max
        resp_good = response_time_percentage <= 80.0  # < 80% of max
        req_rate_good = request_rate_normalized <= 70.0  # Not saturating capacity

        if cpu_in_range and mem_in_range and resp_excellent and req_rate_good:
            optimal_score = 1.0
        elif cpu_in_range and mem_in_range and resp_good and req_rate_good:
            optimal_score = 0.8
        elif cpu_in_range and mem_in_range and resp_good:
            optimal_score = 0.7
        elif (
            (self.min_cpu <= self.cpu_usage <= self.max_cpu * 1.1)
            and (self.min_memory <= self.memory_usage <= self.max_memory * 1.1)
            and resp_good
        ):
            optimal_score = 0.5
        else:
            optimal_score = 0.0

        # BALANCED STATE: Stable and efficient operation
        # Metrics moderate, response time acceptable, request rate sustainable
        cpu_moderate = 40 <= self.cpu_usage <= 70
        mem_moderate = 40 <= self.memory_usage <= 70
        resp_acceptable = response_time_percentage <= 90.0
        req_rate_sustainable = request_rate_normalized <= 80.0

        if cpu_moderate and mem_moderate and resp_good and req_rate_sustainable:
            balanced_score = 1.0
        elif cpu_moderate and mem_moderate and resp_acceptable:
            balanced_score = 0.7
        elif (
            (30 <= self.cpu_usage <= 80)
            and (30 <= self.memory_usage <= 80)
            and resp_acceptable
        ):
            balanced_score = 0.5
        else:
            balanced_score = 0.0

        # WASTEFUL STATE: Over-provisioned, resources under-utilized
        # CPU/MEM very low, response time low, request rate very low = wasting resources
        cpu_very_low = self.cpu_usage < self.min_cpu * 0.5
        mem_very_low = self.memory_usage < self.min_memory * 0.5
        cpu_low = self.cpu_usage < self.min_cpu
        mem_low = self.memory_usage < self.min_memory
        req_rate_very_low = request_rate_normalized < 30.0

        if cpu_very_low and mem_very_low and resp_excellent and req_rate_very_low:
            wasteful_score = 1.0  # Severe waste: all metrics very low
        elif (cpu_very_low or mem_very_low) and resp_excellent and req_rate_very_low:
            wasteful_score = 0.9
        elif cpu_low and mem_low and resp_good and req_rate_normalized < 50.0:
            wasteful_score = 0.7
        elif (cpu_low or mem_low) and resp_excellent:
            wasteful_score = 0.5
        else:
            wasteful_score = 0.0

        # Response time very high, CPU/MEM maxed out, or request rate saturating
        cpu_very_high = self.cpu_usage > self.max_cpu * 1.1
        mem_very_high = self.memory_usage > self.max_memory * 1.1
        cpu_high = self.cpu_usage > self.max_cpu
        mem_high = self.memory_usage > self.max_memory
        resp_critical = response_time_percentage > 120.0
        resp_high = response_time_percentage > 100.0
        req_rate_saturating = request_rate_normalized > 90.0  # Near capacity limit

        if resp_critical or req_rate_saturating:
            critical_score = 1.0  # Very critical
        elif (cpu_very_high or mem_very_high) and resp_high:
            critical_score = 1.0
        elif resp_high and (cpu_high or mem_high):
            critical_score = 0.9
        elif request_rate_normalized > 85.0 and (cpu_high or mem_high):
            critical_score = 0.8  # High request rate with high resources
        elif cpu_very_high or mem_very_high:
            critical_score = 0.7
        elif resp_high:
            critical_score = 0.6
        else:
            critical_score = 0.0

        # Calculate reward based on state quality
        optimal_contribution = optimal_score * 1.0
        balanced_contribution = balanced_score * 0.7
        wasteful_penalty = wasteful_score * self.cost_weight * 1.5
        critical_penalty = critical_score * (
            self.response_time_weight + self.cpu_memory_weight
        )

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

        # CRITICAL: Trend-based bonuses and penalties

        # Proactive scaling bonus: reward scaling up when request rate is trending up
        proactive_bonus = 0.0
        if request_trend == "up" and action_trend == "up":
            proactive_bonus = 0.04  # Good: scaling up proactively
        elif request_trend == "down" and action_trend == "down":
            proactive_bonus = 0.03  # Good: scaling down when load decreasing

        # Reactive penalty: penalize late response to trends
        reactive_penalty = 0.0
        if (
            request_trend == "up"
            and request_rate_normalized > 75.0
            and action_trend == "stable"
        ):
            reactive_penalty = 0.05  # Bad: not scaling when should
        elif (
            request_trend == "up"
            and request_rate_normalized > 75.0
            and action_trend == "down"
        ):
            reactive_penalty = 0.08  # Very bad: scaling down when load increasing

        # Oscillation penalty: penalize unstable action patterns
        oscillation_penalty = 0.0
        if action_trend == "stable" and balanced_score < 0.3 and critical_score > 0.5:
            oscillation_penalty = 0.02  # Bad: not taking action when needed

        # Request rate handling bonus
        req_rate_bonus = 0.0
        if 40.0 <= request_rate_normalized <= 70.0 and request_trend == "stable":
            req_rate_bonus = 0.04  # Sweet spot: good utilization, stable load

        # Apply trend modifiers
        reward += proactive_bonus
        reward -= reactive_penalty
        reward -= oscillation_penalty
        reward += req_rate_bonus

        # Bonus for achieving optimal state with efficient replica usage
        if optimal_score > 0.6 and 0.3 <= replica_ratio <= 0.7:
            reward += 0.05

        # Bonus for stable balanced state
        if balanced_score > 0.7:
            reward += 0.03

        # NOTE: No clamping - preserve actual reward values including:
        # - Very small positive values (e.g., 0.0000001)
        # - Negative values (important learning signal for bad states)
        # - Values > 1.0 (rare but possible for exceptional performance)

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

        # CRITICAL: Concise reward logging on 3 lines
        self.logger.info(
            f"[Q-LEARNING] Iter={getattr(self, 'iteration', '?')} | Replicas={self.replica_state} | "
            f"Reward={reward:.4f} | State(O={optimal_score:.2f} B={balanced_score:.2f} W={wasteful_score:.2f} C={critical_score:.2f})"
        )
        self.logger.info(
            f"  Metrics: CPU={self.cpu_usage:.1f}% MEM={self.memory_usage:.1f}% RT={response_time_percentage:.1f}% "
            f"ReqRate={self.request_rate:.1f}rps({request_rate_normalized:.1f}%) | Trends: Req={request_trend.upper()} Act={action_trend.upper()}"
        )
        self.logger.info(
            f"  Modifiers: Proactive={proactive_bonus:.3f} Reactive=-{reactive_penalty:.3f} ReqBonus={req_rate_bonus:.3f} "
            f"Oscillation=-{oscillation_penalty:.3f} Cost=-{cost_pen:.3f}"
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
            "request_rate_normalized": request_rate_normalized,
            "request_trend": request_trend,
            "action_trend": action_trend,
            "proactive_bonus": proactive_bonus,
            "reactive_penalty": reactive_penalty,
            "oscillation_penalty": oscillation_penalty,
            "req_rate_bonus": req_rate_bonus,
            "optimal": optimal_score,
            "balanced": balanced_score,
            "wasteful": wasteful_score,
            "critical": critical_score,
            "positive_contribution": positive_contribution,
            "negative_contribution": negative_contribution,
        }

    def _calculate_reward_qlearningfuzzy(self) -> Tuple[float, Dict[str, float]]:
        """
        Uses fuzzy logic to evaluate state quality considering all metrics and trends.
        """
        if (
            self.response_time is None
            or math.isnan(self.response_time)
            or math.isinf(self.response_time)
        ):
            self.response_time = 0.0

        response_time_percentage: float = (
            self.response_time / self.max_response_time
        ) * 100.0

        current_capacity: float = self.per_pod_capacity * self.replica_state
        request_rate_normalized: float = min(
            100.0, (self.request_rate / current_capacity) * 100.0
        )

        observation: dict = {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "response_time": response_time_percentage,
            "request_rate_normalized": request_rate_normalized,
        }

        # Get observation for trend categories
        full_observation: dict = self._get_observation()
        request_trend: str = full_observation.get(
            "request_rate_trend_category", "stable"
        )
        action_trend: str = full_observation.get("action_trend_category", "stable")

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
        req_rate_fz = fuzzy_state.get("request_rate_normalized", {})

        # Calculate reward based on state quality
        # Optimal state gives positive contribution
        optimal_contribution = reward_state["optimal"] * 1.0

        # Balanced state gives moderate positive contribution
        balanced_contribution = reward_state["balanced"] * 0.7

        # Wasteful state creates penalty (over-provisioned)
        wasteful_penalty = reward_state["wasteful"] * self.cost_weight * 1.5

        # Critical state creates strong penalty (under-provisioned, performance issues)
        critical_penalty = reward_state["critical"] * (
            self.response_time_weight + self.cpu_memory_weight
        )

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

        # Proactive scaling bonus: reward scaling up when request rate is trending up
        proactive_bonus = 0.0
        if request_trend == "up" and action_trend == "up":
            proactive_bonus = 0.04  # Good: scaling up proactively
        elif request_trend == "down" and action_trend == "down":
            proactive_bonus = 0.03  # Good: scaling down when load decreasing

        # Reactive penalty: penalize late response to trends
        reactive_penalty = 0.0
        if (
            request_trend == "up"
            and request_rate_normalized > 75.0
            and action_trend == "stable"
        ):
            reactive_penalty = 0.05  # Bad: not scaling when should
        elif (
            request_trend == "up"
            and request_rate_normalized > 75.0
            and action_trend == "down"
        ):
            reactive_penalty = 0.08  # Very bad: scaling down when load increasing

        # Oscillation penalty: penalize unstable action patterns
        oscillation_penalty = 0.0
        if (
            action_trend == "stable"
            and reward_state["balanced"] < 0.3
            and reward_state["critical"] > 0.5
        ):
            oscillation_penalty = 0.02  # Bad: not taking action when needed

        # Request rate handling bonus (fuzzy-based)
        req_rate_bonus = 0.0
        req_medium = req_rate_fz.get("medium", 0.0)
        if req_medium > 0.7 and request_trend == "stable":
            req_rate_bonus = 0.04  # Sweet spot: good utilization, stable load

        # Apply trend modifiers
        reward += proactive_bonus
        reward -= reactive_penalty
        reward -= oscillation_penalty
        reward += req_rate_bonus

        # Bonus for achieving optimal state with efficient replica usage
        if reward_state["optimal"] > 0.6 and 0.3 <= replica_ratio <= 0.7:
            reward += 0.05

        # Bonus for stable balanced state
        if reward_state["balanced"] > 0.7:
            reward += 0.03

        # NOTE: No clamping - preserve actual reward values including:
        # - Very small positive values (e.g., 0.0000001)
        # - Negative values (important learning signal for bad states)
        # - Values > 1.0 (rare but possible for exceptional performance)

        # Calculate individual penalties for logging compatibility
        cpu_pen = (
            cpu_fz.get("very_high", 0.0) * 1.0
            + cpu_fz.get("high", 0.0) * 0.7
            + cpu_fz.get("very_low", 0.0) * 0.5
        )
        mem_pen = (
            mem_fz.get("very_high", 0.0) * 1.0
            + mem_fz.get("high", 0.0) * 0.7
            + mem_fz.get("very_low", 0.0) * 0.5
        )
        resp_pen = resp_fz.get("very_high", 0.0) * 1.0 + resp_fz.get("high", 0.0) * 0.8
        cpu_mem_pen = self.cpu_memory_weight * (cpu_pen + mem_pen)
        total_penalty = negative_contribution

        # Get dominant fuzzy memberships for compact display
        cpu_dom = max(cpu_fz, key=cpu_fz.get)
        mem_dom = max(mem_fz, key=mem_fz.get)
        resp_dom = max(resp_fz, key=resp_fz.get)
        req_dom = max(req_rate_fz, key=req_rate_fz.get) if req_rate_fz else "n/a"

        self.logger.info(
            f"[FUZZY] Iter={getattr(self, 'iteration', '?')} | Replicas={self.replica_state} | "
            f"Reward={reward:.4f} | State(O={reward_state['optimal']:.2f} B={reward_state['balanced']:.2f} W={reward_state['wasteful']:.2f} C={reward_state['critical']:.2f})"
        )
        self.logger.info(
            f"  Metrics: CPU={self.cpu_usage:.1f}%({cpu_dom}) MEM={self.memory_usage:.1f}%({mem_dom}) "
            f"RT={response_time_percentage:.1f}%({resp_dom}) ReqRate={self.request_rate:.1f}rps({request_rate_normalized:.1f}%,{req_dom})"
        )
        self.logger.info(
            f"  Fuzzy: ScaleUp={fuzzy_actions['scale_up']:.2f} Down={fuzzy_actions['scale_down']:.2f} Stay={fuzzy_actions['no_change']:.2f} "
            f"Influence={fuzzy_influence:+.3f} | Trends: Req={request_trend.upper()} Act={action_trend.upper()}"
        )
        self.logger.info(
            f"  Modifiers: Proactive={proactive_bonus:.3f} Reactive=-{reactive_penalty:.3f} ReqBonus={req_rate_bonus:.3f} "
            f"Oscillation=-{oscillation_penalty:.3f} Cost=-{cost_pen:.3f}"
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
            "request_rate_normalized": request_rate_normalized,
            "request_trend": request_trend,
            "action_trend": action_trend,
            "proactive_bonus": proactive_bonus,
            "reactive_penalty": reactive_penalty,
            "oscillation_penalty": oscillation_penalty,
            "req_rate_bonus": req_rate_bonus,
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
            return self._calculate_reward_qlearningfuzzy()
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

    def _categorize_request_rate_trend(self) -> str:
        """
        Converts continuous trend values into categorical state: up, down, stable.

        Returns:
            str: 'up', 'down', or 'stable'
        """
        if self.request_rate_trend > 5.0:
            return "up"
        elif self.request_rate_trend < -5.0:
            return "down"
        else:
            return "stable"

    def _calculate_action_trend_value(self) -> float:
        """
        Positive value indicates scaling up trend, negative indicates scaling down.

        Returns:
            float: Scaled slope representing action trend magnitude
        """
        if len(self.action_history) < 2:
            return 0.0

        history = self.action_history[-self.action_trend_window :]
        n = len(history)

        if n < 3:
            # Not enough data for linear regression, use simple difference
            return float(history[-1] - history[0])

        # Linear regression slope: measures if actions are trending up/down
        # Formula: (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
        import numpy as np

        x = np.arange(n)
        y = np.array(history)

        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
            n * np.sum(x**2) - np.sum(x) ** 2 + 1e-10
        )

        # Scale slope to meaningful range
        scaled_slope = slope * self.action_trend_window

        return float(scaled_slope)

    def _categorize_action_trend(self) -> str:
        """
        Helps agent detect scaling patterns (increasing, decreasing, stable).

        Returns:
            str: 'up', 'down', or 'stable'
        """
        trend_value = self._calculate_action_trend_value()

        if trend_value > 5.0:
            return "up"
        elif trend_value < -5.0:
            return "down"
        else:
            return "stable"

    def _get_observation(self) -> dict[str, float]:
        """
        This is the state representation that RL agents use for decision making.
        """
        response_time_percentage: float = min(
            (self.response_time / self.max_response_time) * 100.0, 100.0
        )

        # Accounts for current number of replicas to provide meaningful percentage
        # Example: 80 req/s with 1 pod (80 capacity) = 100%, but with 10 pods (800 capacity) = 10%
        current_capacity: float = self.per_pod_capacity * self.replica_state
        request_rate_normalized: float = min(
            100.0, (self.request_rate / current_capacity) * 100.0
        )

        request_rate_trend_category: str = self._categorize_request_rate_trend()
        action_trend_category: str = self._categorize_action_trend()

        return {
            # Core resource metrics (0-100%)
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "response_time": response_time_percentage,
            "response_time_ms": self.response_time,  # Raw value in milliseconds for logging
            # Request rate metrics
            "request_rate": self.request_rate,  # Raw value for logging
            "request_rate_normalized": request_rate_normalized,  # Percentage of current capacity
            "request_rate_trend": self.request_rate_trend,  # Raw difference
            "request_rate_trend_category": request_rate_trend_category,  # Categorical: up/down/stable
            # Action and system state
            "last_action": self.last_action,  # Action as percentage (0-99 representing min to max replicas)
            "action_trend_category": action_trend_category,  # Categorical: up/down/stable
            "current_replicas": float(self.replica_state),  # Absolute replica count
        }

    def step(self, action: int, q_table_size: int = 0) -> tuple[dict[str, float], float, bool, dict]:
        # Calculate action change before updating last_action
        self.action_change = action - self.last_action
        self.last_action = action

        # Maintains sliding window of recent actions to detect patterns
        self.action_history.append(action)
        if len(self.action_history) > self.action_trend_window:
            self.action_history.pop(0)

        self.previous_request_rate = self.request_rate

        percentage = (action / 99.0) if len(self.action_space) > 1 else 0.0
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
            "request_rate": self.request_rate,
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
            "request_rate": self.request_rate,
            "request_rate_normalized": reward_breakdown.get(
                "request_rate_normalized", 0.0
            ),
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

    def reset(self) -> dict[str, float]:
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
        self.last_action = 0

        self.action_history = []

        # Reset episode-specific metrics and increment episode counter
        self.cumulative_reward = 0.0
        self.episode_reward = 0.0  # Reset for new episode
        self.episode_number += 1

        return self._get_observation()

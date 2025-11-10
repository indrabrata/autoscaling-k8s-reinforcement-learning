from logging import Logger
import math
from typing import Dict, Optional


class Fuzzy:
    def __init__(self, logger: Optional[Logger] = None):
        def _trapezoidal(x, a, b, c, d):
            if x < a or x > d:
                return 0.0
            elif b <= x <= c:
                return 1.0
            elif a < x < b:
                return (x - a) / (b - a) if (b - a) != 0 else 0.0
            else: 
                return (d - x) / (d - c) if (d - c) != 0 else 0.0

        self.memberships = {
            "cpu_usage": {
                "very_low": lambda x: _trapezoidal(x, 0, 0, 10, 25),
                "low": lambda x: _trapezoidal(x, 15, 25, 35, 45),
                "medium": lambda x: _trapezoidal(x, 40, 50, 60, 70),
                "high": lambda x: _trapezoidal(x, 65, 75, 85, 90),
                "very_high": lambda x: _trapezoidal(x, 85, 95, 100, 100),
            },
            "memory_usage": {
                "very_low": lambda x: _trapezoidal(x, 0, 0, 10, 25),
                "low": lambda x: _trapezoidal(x, 15, 25, 35, 45),
                "medium": lambda x: _trapezoidal(x, 40, 50, 60, 70),
                "high": lambda x: _trapezoidal(x, 65, 75, 85, 90),
                "very_high": lambda x: _trapezoidal(x, 85, 95, 100, 100),
            },
            "response_time": {
                "very_low": lambda x: _trapezoidal(x, 0, 0, 10, 25),
                "low": lambda x: _trapezoidal(x, 20, 30, 45, 55),
                "medium": lambda x: _trapezoidal(x, 50, 60, 70, 80),
                "high": lambda x: _trapezoidal(x, 75, 85, 90, 95),
                "very_high": lambda x: _trapezoidal(x, 90, 95, 100, 100),
            },
        }

        self.logger = logger or Logger(__name__)

    def fuzzify(self, obs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        fuzzy_state = {}
        for metric, value in obs.items():
            if metric in self.memberships:
                fuzzy_state[metric] = {
                    label: fn(value) for label, fn in self.memberships[metric].items()
                }
        self.logger.info(f"Fuzzified: {fuzzy_state}")
        return fuzzy_state

    def apply_rules(self, fz: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        cpu = fz.get("cpu_usage", {})
        mem = fz.get("memory_usage", {})
        resp = fz.get("response_time", {})

        scale_up = max(
            # High response time dominates
            max(resp.get("high", 0.0), resp.get("medium", 0.0)) * 1.0,

            # CPU or Memory high and response medium/high
            min(max(cpu.get("high", 0.0), mem.get("high", 0.0)),
                max(resp.get("medium", 0.0), resp.get("high", 0.0))) * 0.9,

            # CPU or Memory very high regardless of response
            max(cpu.get("very_high", 0.0), mem.get("very_high", 0.0)) * 1.0,

            # CPU medium, mem medium, but response high
            min(cpu.get("medium", 0.0), mem.get("medium", 0.0), resp.get("high", 0.0)) * 0.85,

            # CPU or Mem high while response is deteriorating (medium)
            max(cpu.get("high", 0.0), mem.get("high", 0.0)) * resp.get("medium", 0.0) * 0.8,
        )

        scale_down = max(
            # All metrics low
            min(cpu.get("low", 0.0), mem.get("low", 0.0), resp.get("low", 0.0)) * 1.0,

            # CPU & Mem low, response medium (still fine)
            min(cpu.get("low", 0.0), mem.get("low", 0.0), resp.get("medium", 0.0)) * 0.9,

            # CPU or Mem very low and response low
            max(cpu.get("very_low", 0.0), mem.get("very_low", 0.0)) * resp.get("low", 0.0) * 1.0,

            # Response very low even if one resource is medium (under-utilized)
            min(resp.get("very_low", 0.0), max(cpu.get("medium", 0.0), mem.get("medium", 0.0))) * 0.8,
        )

        no_change = max(
            # Balanced load
            min(cpu.get("medium", 0.0), mem.get("medium", 0.0), resp.get("medium", 0.0)) * 1.0,

            # CPU high but response still low — wait before scaling
            min(cpu.get("high", 0.0), resp.get("low", 0.0)) * 0.8,

            # CPU & Mem medium while response low — stable
            min(cpu.get("medium", 0.0), mem.get("medium", 0.0), resp.get("low", 0.0)) * 0.9,

            # CPU low, mem medium, response medium — stable
            min(cpu.get("low", 0.0), mem.get("medium", 0.0), resp.get("medium", 0.0)) * 0.85,

            # Mixed load but not dominant
            min(max(cpu.get("medium", 0.0), mem.get("medium", 0.0)), resp.get("medium", 0.0)) * 0.8,
        )

        total = scale_up + scale_down + no_change + 1e-6
        res = {
            "scale_up": scale_up / total,
            "scale_down": scale_down / total,
            "no_change": no_change / total,
        }

        return res

    def apply_reward_rules(self, fz: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Apply fuzzy rules specifically designed for reward calculation.
        Returns state quality metrics: optimal, wasteful, critical, balanced.
        """
        cpu = fz.get("cpu_usage", {})
        mem = fz.get("memory_usage", {})
        resp = fz.get("response_time", {})

        # OPTIMAL STATE: Resources well-utilized, response time excellent
        optimal_state = max(
            # All metrics in ideal ranges
            min(
                max(cpu.get("medium", 0.0), cpu.get("high", 0.0)),
                max(mem.get("medium", 0.0), mem.get("high", 0.0)),
                max(resp.get("very_low", 0.0), resp.get("low", 0.0))
            ) * 1.0,

            # CPU/MEM medium with very low response time (best case)
            min(
                cpu.get("medium", 0.0),
                mem.get("medium", 0.0),
                resp.get("very_low", 0.0)
            ) * 1.0,

            # Slightly high CPU/MEM but response still low (acceptable)
            min(
                max(cpu.get("medium", 0.0), cpu.get("high", 0.0)),
                max(mem.get("medium", 0.0), mem.get("high", 0.0)),
                resp.get("low", 0.0)
            ) * 0.9,
        )

        # WASTEFUL STATE: Over-provisioned, resources under-utilized
        wasteful_state = max(
            # All metrics very low (severe under-utilization)
            min(
                cpu.get("very_low", 0.0),
                mem.get("very_low", 0.0),
                resp.get("very_low", 0.0)
            ) * 1.0,

            # CPU and MEM low with very low response time
            min(
                cpu.get("low", 0.0),
                mem.get("low", 0.0),
                resp.get("very_low", 0.0)
            ) * 0.9,

            # One very low resource is enough to indicate waste
            max(cpu.get("very_low", 0.0), mem.get("very_low", 0.0)) *
            resp.get("very_low", 0.0) * 0.8,

            # Low CPU and low memory with low response time
            min(
                cpu.get("low", 0.0),
                mem.get("low", 0.0),
                resp.get("low", 0.0)
            ) * 0.7,
        )

        # CRITICAL STATE: Under-provisioned, performance degrading
        critical_state = max(
            # Very high response time is critical regardless of resources
            resp.get("very_high", 0.0) * 1.0,

            # High response time with high CPU/MEM (system struggling)
            min(
                max(cpu.get("high", 0.0), cpu.get("very_high", 0.0)),
                max(mem.get("high", 0.0), mem.get("very_high", 0.0)),
                resp.get("high", 0.0)
            ) * 1.0,

            # Very high CPU or MEM with medium or high response time
            max(cpu.get("very_high", 0.0), mem.get("very_high", 0.0)) *
            max(resp.get("medium", 0.0), resp.get("high", 0.0)) * 0.95,

            # High response time alone is concerning
            resp.get("high", 0.0) * 0.85,
        )

        # BALANCED STATE: Stable and efficient operation
        balanced_state = max(
            # All metrics in medium range (balanced)
            min(
                cpu.get("medium", 0.0),
                mem.get("medium", 0.0),
                resp.get("medium", 0.0)
            ) * 1.0,

            # CPU medium, response low (good efficiency)
            min(
                cpu.get("medium", 0.0),
                max(mem.get("medium", 0.0), mem.get("low", 0.0)),
                resp.get("low", 0.0)
            ) * 0.95,

            # Resources moderate with low response time
            min(
                max(cpu.get("low", 0.0), cpu.get("medium", 0.0)),
                max(mem.get("low", 0.0), mem.get("medium", 0.0)),
                resp.get("low", 0.0)
            ) * 0.9,

            # CPU/MEM slightly varied but response time good
            min(
                max(cpu.get("medium", 0.0), cpu.get("high", 0.0)),
                mem.get("medium", 0.0),
                max(resp.get("very_low", 0.0), resp.get("low", 0.0))
            ) * 0.85,
        )

        # Normalize
        total = optimal_state + wasteful_state + critical_state + balanced_state + 1e-6
        res = {
            "optimal": optimal_state / total,
            "wasteful": wasteful_state / total,
            "critical": critical_state / total,
            "balanced": balanced_state / total,
        }

        self.logger.debug(f"Reward rules: {res}")
        return res

    def influence(self, act: Dict[str, float]) -> float:
        up, down, stay = act["scale_up"], act["scale_down"], act["no_change"]
        total = up + down + stay + 1e-6

        # Base direction
        direction = (up - down) / total

        # Confidence: how dominant one decision is
        confidence = max(up, down, stay)
        neutrality = stay / total

        # Hysteresis: reduce influence near neutral
        hysteresis = 1.0 - 0.6 * neutrality

        # Weighted final influence
        influence = direction * hysteresis * confidence

        # Clip
        influence = max(-1.0, min(1.0, influence))

        self.logger.debug(
            f"Influence={influence:.3f} (up={up:.2f}, down={down:.2f}, stay={stay:.2f}, conf={confidence:.2f})"
        )
        return influence

    def decide(self, obs: Dict[str, float]) -> Dict[str, float]:
        fz = self.fuzzify(obs)
        acts = self.apply_rules(fz)
        infl = self.influence(acts)

        if infl > 0.4:
            rec = "scale_up"
        elif infl < -0.4:
            rec = "scale_down"
        else:
            rec = "maintain"

        result = {"memberships": acts, "influence": infl, "recommendation": rec}
        self.logger.info(f"Decision: {result}")
        return result

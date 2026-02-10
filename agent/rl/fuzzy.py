from logging import Logger
from typing import Dict, Optional, Union


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

        # All metrics use 0-100% scale for consistency
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
            "request_rate_normalized": {
                "very_low": lambda x: _trapezoidal(x, 0, 0, 10, 25),
                "low": lambda x: _trapezoidal(x, 15, 25, 35, 45),
                "medium": lambda x: _trapezoidal(x, 40, 50, 60, 70),
                "high": lambda x: _trapezoidal(x, 65, 75, 85, 90),
                "very_high": lambda x: _trapezoidal(x, 85, 95, 100, 100),
            },
            "last_action": {
                "very_low": lambda x: _trapezoidal(x, 0, 0, 10, 25),
                "low": lambda x: _trapezoidal(x, 15, 25, 35, 45),
                "medium": lambda x: _trapezoidal(x, 40, 50, 60, 70),
                "high": lambda x: _trapezoidal(x, 65, 75, 85, 90),
                "very_high": lambda x: _trapezoidal(x, 85, 95, 100, 100),
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

        return fuzzy_state

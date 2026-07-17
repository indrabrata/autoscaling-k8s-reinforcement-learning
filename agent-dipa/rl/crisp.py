from logging import Logger
from typing import Dict, Optional

from .fuzzy import CRISP_BOUNDARIES, METRICS, StateKey, normalize_observation


class Crisp:
    """
    Hard partition of the metric space at the fuzzy sets' own boundaries.

    This is the ablation between the fuzzy and conventional agents: it keeps
    the fuzzy study's discretization (the same three bands, cut at the same
    places) but drops the fuzziness. A value belongs to exactly one band, with
    no membership degree and no overlap, so every observation maps to a single
    state key.

    The cut-offs come from CRISP_BOUNDARIES, i.e. the midpoints of the zones
    where the trapezoids overlap (35.0 and 65.0). They are derived rather than
    hand-picked so that any change to the trapezoids moves the fuzzy and crisp
    agents together.
    """

    def __init__(self, logger: Optional[Logger] = None, max_replicas: int = 12):
        self.low_threshold, self.high_threshold = CRISP_BOUNDARIES
        self.max_replicas = max_replicas
        self.logger = logger or Logger(__name__)

    def label(self, value: float) -> str:
        if value < self.low_threshold:
            return "low"
        if value < self.high_threshold:
            return "medium"
        return "high"

    def crispify(self, obs: Dict[str, float]) -> Dict[str, str]:
        """The single band each metric falls into."""
        return {
            metric: self.label(
                normalize_observation(metric, obs[metric], self.max_replicas)
            )
            for metric in METRICS
            if metric in obs
        }

    def get_state_key(self, obs: Dict[str, float]) -> StateKey:
        crisp_state = self.crispify(obs)
        return tuple(crisp_state[metric] for metric in METRICS)

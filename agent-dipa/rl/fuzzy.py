import itertools
from logging import Logger
from typing import Dict, List, Optional, Tuple

# Membership degrees at or below this are treated as inactive.
EPSILON = 1e-6

# A state key is one label per metric, ordered as METRICS.
StateKey = Tuple[str, str, str, str]

# Trapezoid parameters (a, b, c, d) on a 0-100% scale, shared by every metric.
# These are the single source of truth for all three agents: the fuzzy agent
# reads the membership degrees, the crisp agent reads the boundaries derived
# from them (see CRISP_BOUNDARIES). The conventional agent ignores both.
# A Ruspini partition: each descending edge coincides exactly with the next
# set's ascending edge (low falls over [25, 40] as medium rises over [25, 40];
# medium falls over [60, 75] as high rises over [60, 75]). So the membership
# degrees sum to 1 everywhere, the overlap midpoint equals the crossover point,
# and the weight normalization in get_activations is a no-op rather than a
# correction. See CRISP_BOUNDARIES for how the crisp agent reuses these edges.
TRAPEZOIDS: Dict[str, Tuple[float, float, float, float]] = {
    "low": (0.0, 0.0, 25.0, 40.0),
    "medium": (25.0, 40.0, 60.0, 75.0),
    "high": (60.0, 75.0, 100.0, 100.0),
}

METRICS: Tuple[str, str, str, str] = (
    "cpu_usage",
    "memory_usage",
    "response_time",
    "last_replica",
)

LABELS: Tuple[str, str, str] = ("low", "medium", "high")


def _trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    # Rising edge, a <= x < b. The bound must be `x < b` rather than
    # `a < x < b`: at exactly x == a a strict lower bound falls through to the
    # falling-edge formula below and returns (d-a)/(d-c), which for `medium`
    # is 3.33 -- not a membership degree at all. Reaching here already implies
    # x >= a, so the lower bound is redundant as well as wrong.
    if x < b:
        return (x - a) / (b - a) if (b - a) != 0 else 0.0
    return (d - x) / (d - c) if (d - c) != 0 else 0.0


def _overlap_midpoint(lower: str, upper: str) -> float:
    """
    Midpoint of the zone where two adjacent fuzzy sets overlap.

    `low` fades out over [25, 40] while `medium` fades in over the same span,
    so the two sets share the interval [a_upper, d_lower]. Because the
    trapezoids form a Ruspini partition (the two edges coincide), this midpoint
    is also the crossover where the two membership degrees are equal -- the
    point where argmax changes hands -- so it is the natural crisp cut-off.
    """
    d_lower = TRAPEZOIDS[lower][3]
    a_upper = TRAPEZOIDS[upper][0]
    return (a_upper + d_lower) / 2.0


# Crisp bin edges derived from the trapezoids above, so the crisp agent
# partitions the metric space at the same places the fuzzy sets change hands.
# On a Ruspini partition the overlap midpoint is exactly the membership
# crossover, so these are the true argmax hand-over points:
#   low/medium overlap [25, 40] -> 32.5
#   medium/high overlap [60, 75] -> 67.5
# Yielding: low = x < 32.5, medium = 32.5 <= x < 67.5, high = x >= 67.5.
CRISP_BOUNDARIES: Tuple[float, float] = (
    _overlap_midpoint("low", "medium"),
    _overlap_midpoint("medium", "high"),
)


def normalize_observation(
    metric: str, value: float, max_replicas: int
) -> float:
    """
    Put a raw observation on the 0-100% scale the membership functions expect.

    `last_replica` arrives as a replica count (1..max_replicas) rather than a
    percentage, so it is rescaled. Everything else is already a percentage but
    is clamped: CPU or memory can legitimately exceed its limit and report
    above 100%, which would otherwise fall outside every trapezoid's support
    and leave the observation with no active membership at all.
    """
    if metric == "last_replica" and max_replicas > 1:
        value = ((value - 1) / (max_replicas - 1)) * 100.0
    return max(0.0, min(100.0, value))


class Fuzzy:
    def __init__(self, logger: Optional[Logger] = None, max_replicas: int = 12):
        self.memberships = {
            metric: {
                label: (lambda x, p=params: _trapezoidal(x, *p))
                for label, params in TRAPEZOIDS.items()
            }
            for metric in METRICS
        }
        self.max_replicas = max_replicas
        self.logger = logger or Logger(__name__)

    def fuzzify(self, obs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Membership degree of every label, per metric."""
        fuzzy_state = {}
        for metric, value in obs.items():
            if metric in self.memberships:
                fuzz_value = normalize_observation(metric, value, self.max_replicas)
                fuzzy_state[metric] = {
                    label: fn(fuzz_value)
                    for label, fn in self.memberships[metric].items()
                }
        return fuzzy_state

    def get_activations(self, obs: Dict[str, float]) -> List[Tuple[StateKey, float]]:
        """
        Every state the observation belongs to, with its normalized firing strength.

        Unlike the previous study, a metric sitting in an overlap zone is NOT
        collapsed to its strongest label. It stays a member of both sets, and
        each of its labels is combined with every label of the other metrics.
        A metric with 2 active labels therefore contributes a factor of 2 to
        the number of active states, up to 2^4 = 16 when all four overlap.

        Firing strength uses the product t-norm:
            w(l_cpu, l_mem, l_rt, l_rep) = mu_cpu(l_cpu) * ... * mu_rep(l_rep)
        and is normalized so the weights sum to 1, making Q-values across
        states directly comparable.

        Returns:
            List of (state_key, weight), weights summing to 1.
        """
        fuzzy_state = self.fuzzify(obs)

        active_per_metric: List[List[Tuple[str, float]]] = []
        for metric in METRICS:
            degrees = fuzzy_state[metric]
            active = [(label, mu) for label, mu in degrees.items() if mu > EPSILON]
            if not active:
                # The trapezoids cover 0-100 without gaps and inputs are clamped
                # to that range, so this is unreachable; fall back to the
                # dominant label rather than emit a stateless observation.
                dominant = max(degrees, key=lambda k: degrees[k])
                self.logger.warning(
                    f"No active membership for {metric}={obs.get(metric)}; "
                    f"falling back to '{dominant}'"
                )
                active = [(dominant, 1.0)]
            active_per_metric.append(active)

        activations: List[Tuple[StateKey, float]] = []
        for combo in itertools.product(*active_per_metric):
            weight = 1.0
            for _, mu in combo:
                weight *= mu
            if weight > EPSILON:
                labels = tuple(label for label, _ in combo)
                activations.append((labels, weight))

        total = sum(weight for _, weight in activations)
        if total <= EPSILON:
            uniform = 1.0 / len(activations)
            return [(labels, uniform) for labels, _ in activations]

        return [(labels, weight / total) for labels, weight in activations]

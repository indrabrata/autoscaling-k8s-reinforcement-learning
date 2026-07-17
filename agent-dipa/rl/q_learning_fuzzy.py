from logging import Logger
from typing import List, Optional, Tuple

import numpy as np

from .base import QLearningBase
from .fuzzy import Fuzzy, StateKey


class QLearningFuzzy(QLearningBase):
    """
    Fuzzy Q-learning with multi-membership states (Glorennec & Jouffe, FQL).

    Where the previous study defuzzified early -- each metric collapsed to its
    single strongest label via max(), yielding exactly one state per
    observation -- this agent keeps every label a metric belongs to. A metric
    in an overlap zone is a partial member of two bands at once, and those
    labels are combined with every label of the other metrics, so one
    observation activates up to 2^4 = 16 states simultaneously.

    Each active state carries a normalized firing strength w_i (product t-norm
    over the membership degrees, see Fuzzy.get_activations). Those weights
    drive both halves of the algorithm:

        action:  a* = argmax_a  sum_i w_i * Q(s_i, a)
        update:  dQ(s_i, a) = lr * w_i * [r + gamma * max_a' Q(s', a') - Q(s, a)]
                 where Q(s, a) = sum_i w_i * Q(s_i, a)

    So a state that the observation barely belongs to is barely updated, and
    an observation sitting on a boundary contributes to both neighbours in
    proportion to how much it belongs to each. This is what removes the
    all-or-nothing jump the crisp agent has at 35 and 65: the policy varies
    continuously across a boundary instead of switching states outright.
    """

    agent_type = "Q-LEARNING-FUZZY"
    state_column_header = "State: (CPU, MEM, RESP, LastReplica) - fuzzy labels"

    def __init__(self, *args, logger: Optional[Logger] = None, **kwargs):
        super().__init__(*args, logger=logger, **kwargs)
        self.fuzzy = Fuzzy(logger=logger, max_replicas=self.n_actions)

    def get_activations(self, observation: dict) -> List[Tuple[StateKey, float]]:
        """Every state this observation belongs to, with normalized weights."""
        self.sanitize(observation)
        return self.fuzzy.get_activations(observation)

    def get_state_key(self, observation: dict) -> StateKey:
        """
        The single most strongly activated state.

        Only for logging and inspection -- learning uses the full activation
        set, never this alone.
        """
        activations = self.get_activations(observation)
        return max(activations, key=lambda item: item[1])[0]

    def get_q_values(self, observation: dict) -> np.ndarray:
        """Firing-strength-weighted Q-values: Q(s, a) = sum_i w_i * Q(s_i, a)."""
        q_values = np.zeros(self.n_actions)
        for state_key, weight in self.get_activations(observation):
            q_values += weight * self.ensure_state(state_key)
        return q_values

    def get_action(self, observation: dict) -> int:
        return self.select_action(self.get_q_values(observation))

    def update_q_table(
        self, observation: dict, action: int, reward: float, next_observation: dict
    ) -> None:
        activations = self.get_activations(observation)
        action_idx = action - 1

        # Both the current estimate and the bootstrap target are aggregated
        # across active states, so the TD error is computed once against the
        # policy's actual estimate rather than per-state.
        current_q = sum(
            weight * self.ensure_state(state_key)[action_idx]
            for state_key, weight in activations
        )
        best_next_q = float(np.max(self.get_q_values(next_observation)))
        td_error = reward + self.discount_factor * best_next_q - current_q

        # Credit is shared out by membership: each state moves only as far as
        # the observation actually belonged to it.
        for state_key, weight in activations:
            self.q_table[state_key][action_idx] += (
                self.learning_rate * weight * td_error
            )

        self.decay_epsilon()

        self.logger.debug(
            f"Q-update | {len(activations)} active states | A={action} | "
            f"R={reward:.3f} | Q={current_q:.3f} | TD={td_error:.3f}"
        )
        for state_key, weight in activations:
            self.logger.debug(
                f"    w={weight:.4f} | S={state_key} | "
                f"NewQ={self.q_table[state_key][action_idx]:.3f}"
            )

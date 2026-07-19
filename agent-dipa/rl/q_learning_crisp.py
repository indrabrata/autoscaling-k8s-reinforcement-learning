from logging import Logger
from typing import Optional

import numpy as np

from .base import QLearningBase
from .crisp import Crisp
from .fuzzy import StateKey


class QLearningCrisp(QLearningBase):
    """
    Tabular Q-learning over hard bands cut at the fuzzy sets' boundaries.

    This is the middle term of the comparison. It borrows the fuzzy study's
    discretization -- the same three bands per metric, cut at the same places
    (32.5 and 67.5, the midpoints of the trapezoid overlaps) -- but assigns
    each metric to exactly one band. The state space collapses to at most
    3^4 = 81 keys, but a metric at 32.4 and one at 32.6 land in different
    states and share nothing, which is precisely the boundary brittleness the
    fuzzy agent is meant to smooth over.

    Beyond the state key, the update is identical to conventional Q-learning.
    """

    agent_type = "Q-LEARNING-CRISP"
    state_column_header = "State: (CPU, MEM, RESP, LastReplica) - crisp bands"

    def __init__(self, *args, logger: Optional[Logger] = None, **kwargs):
        super().__init__(*args, logger=logger, **kwargs)
        self.crisp = Crisp(logger=logger, max_replicas=self.n_actions)
        self.logger.info(
            f"Crisp boundaries: low < {self.crisp.low_threshold} <= medium "
            f"< {self.crisp.high_threshold} <= high"
        )

    def get_state_key(self, observation: dict) -> StateKey:
        self.sanitize(observation)
        return self.crisp.get_state_key(observation)

    def get_action(self, observation: dict) -> int:
        return self.select_action(self.get_q_values(observation))

    def update_q_table(
        self, observation: dict, action: int, reward: float, next_observation: dict
    ) -> None:
        state_key = self.get_state_key(observation)
        next_state_key = self.get_state_key(next_observation)

        q_values = self.ensure_state(state_key)
        next_q_values = self.ensure_state(next_state_key)

        action_idx = action - 1
        best_next_action = float(np.max(next_q_values))
        old_value = q_values[action_idx]

        q_values[action_idx] += self.learning_rate * (
            reward + self.discount_factor * best_next_action - old_value
        )

        self.decay_epsilon()

        self.logger.debug(
            f"Q-update | S={state_key} | A={action} | R={reward:.3f} | "
            f"NewQ={q_values[action_idx]:.3f}"
        )

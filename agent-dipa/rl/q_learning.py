from typing import Tuple

import numpy as np

from .base import QLearningBase


class QLearning(QLearningBase):
    """
    Conventional tabular Q-learning: the control baseline.

    The observation is used verbatim as the Q-table key, with no
    discretization of any kind. Each distinct combination of continuous
    readings is therefore its own state, so the table grows with the number of
    distinct observations and nothing is shared between two states that differ
    only marginally. This is the behaviour the crisp and fuzzy agents are
    meant to improve on.
    """

    agent_type = "Q-LEARNING"
    state_column_header = "State: (CPU, MEM, RESP, LastReplica) - continuous"

    def get_state_key(self, observation: dict) -> Tuple[float, float, float, float]:
        """
        Raw continuous values:
        - cpu_usage      CPU usage percentage (0-100)
        - memory_usage   memory usage percentage (0-100)
        - response_time  response time as a percentage of the SLO (0-100)
        - last_replica   previous replica count
        """
        self.sanitize(observation)
        return (
            observation["cpu_usage"],
            observation["memory_usage"],
            observation["response_time"],
            observation["last_replica"],
        )

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

    def format_state(self, state_key: tuple) -> str:
        cpu, mem, resp, last_replica = state_key
        return f"({cpu:.2f}, {mem:.2f}, {resp:.2f}, {last_replica:.2f})"

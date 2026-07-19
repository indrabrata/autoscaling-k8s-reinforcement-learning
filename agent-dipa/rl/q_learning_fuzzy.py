from logging import Logger
from typing import List, Optional, Tuple

import numpy as np

from .base import QLearningBase
from .fuzzy import Fuzzy, StateKey


class QLearningFuzzy(QLearningBase):
    """
    Fuzzy Q-learning as soft state aggregation (Singh, Jaakkola & Jordan, 1995).

    Where the previous study defuzzified early -- each metric collapsed to its
    single strongest label via max(), yielding exactly one state per
    observation -- this agent keeps every label a metric belongs to. A metric
    in an overlap zone is a partial member of two bands at once, and those
    labels are combined with every label of the other metrics, so one
    observation activates up to 2^4 = 16 states simultaneously.

    Each active state carries a normalized firing strength w_i (product t-norm
    over the membership degrees, see Fuzzy.get_activations). These play the
    role of the "clustering probabilities" P(x|s) of soft state aggregation:
    the raw observation is softly assigned to several discrete clusters at
    once. Both halves of the algorithm run on the aggregated Q-values
    Q(s, a) = sum_i w_i * Q(s_i, a):

        action:  a* = argmax_a  sum_i w_i * Q(s_i, a)
        update:  dQ(s_i, a) = lr * w_i * [r + gamma * max_a' Q(s', a') - Q(s, a)]

    The update is just the semi-gradient of the aggregated Q with respect to
    each cluster's value: because Q(s, a) is linear in the memberships, the
    weight w_i is the feature d Q(s,a) / d Q(s_i,a), so the TD error is shared
    out by membership with no extra machinery. A state the observation barely
    belongs to is barely updated, and an observation on a boundary contributes
    to both neighbours in proportion to how much it belongs to each, removing
    the all-or-nothing jump the crisp agent has at its band edges. The crisp
    agent is precisely the degenerate case Singh et al. single out -- "the
    usual state aggregation where each state belongs to only one cluster" -- so
    crisp vs. fuzzy here is their hard vs. soft aggregation.

    Reference
    ---------
        S. P. Singh, T. Jaakkola, M. I. Jordan, "Reinforcement Learning with
        Soft State Aggregation," NIPS 7, 1995, pp. 361-368.
            "We allow soft clustering, where each state s belongs to cluster x
             with probability P(x|s), called the clustering probabilities."
             (p. 362)
            "The Q-value function for the state space can then be constructed
             via Q(s,a) = sum_x P(x|s) Q(x,a) for all (s,a)."  (p. 364, Eq. 3)
        Theorem 1 there gives the convergence guarantee for Q-learning under a
        fixed soft aggregation.

    This is not Glorennec & Jouffe's fuzzy Q-learning (FQL): FQL runs
    epsilon-greedy per rule and interpolates the winning local actions into one
    continuous action, whereas here all states share one discrete action set
    and a single global argmax over the aggregated Q-values picks the action.
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
            self.ensure_state(state_key)[action_idx] += (
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

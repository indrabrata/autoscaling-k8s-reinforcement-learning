from logging import Logger
from typing import Optional, Type

from .base import QLearningBase
from .q_learning import QLearning
from .q_learning_crisp import QLearningCrisp
from .q_learning_fuzzy import QLearningFuzzy

# The three agents under comparison, keyed by the ALGORITHM env var.
AGENTS: dict[str, Type[QLearningBase]] = {
    "Q-LEARNING": QLearning,
    "Q-LEARNING-CRISP": QLearningCrisp,
    "Q-LEARNING-FUZZY": QLearningFuzzy,
}


def create_agent(
    algorithm: str,
    learning_rate: float,
    discount_factor: float,
    epsilon_start: float,
    epsilon_decay: float,
    epsilon_min: float,
    created_at: int,
    n_actions: int,
    logger: Optional[Logger] = None,
) -> QLearningBase:
    agent_cls = AGENTS.get(algorithm.upper())
    if agent_cls is None:
        raise ValueError(
            f"Unsupported algorithm: {algorithm}. "
            f"Expected one of: {', '.join(sorted(AGENTS))}"
        )

    return agent_cls(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon_start=epsilon_start,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        created_at=created_at,
        n_actions=n_actions,
        logger=logger,
    )

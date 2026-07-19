import math
import pickle
from logging import Logger
from pathlib import Path
from typing import Any, Optional

import numpy as np
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class QLearningBase:
    """
    Shared scaffolding for the three agents under comparison.

    All three run the same tabular Q-learning loop against the same reward
    signal and the same action space (replica count 1..n_actions, stored
    0-indexed in the Q-table). They differ only in how an observation becomes
    a Q-table key:

    - QLearning       raw continuous metrics, no discretization
    - QLearningCrisp  one hard band per metric, cut at the fuzzy boundaries
    - QLearningFuzzy  every band the metrics belong to, weighted by membership

    Subclasses must implement get_state_key, get_action and update_q_table.
    """

    agent_type = "Q-LEARNING-BASE"
    # Header shown above the state column in show_model_summary.
    state_column_header = "State: (CPU, MEM, RESP, LastReplica)"

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 0.9,
        epsilon_decay: float = 0.99,
        epsilon_min: float = 0.01,
        created_at: int = 0,
        n_actions: int = 100,
        logger: Optional[Logger] = None,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.created_at = created_at
        self.n_actions = n_actions
        self.episodes_trained = 0
        self.q_table = {}
        self.logger = logger or Logger(__name__)

        self.logger.info(f"Initialized {self.agent_type} agent")
        self.logger.debug(f"Agent parameters: {self.__dict__}")

    # ------------------------------------------------------------------
    # To be provided by each agent
    # ------------------------------------------------------------------

    def get_state_key(self, observation: dict) -> tuple:
        raise NotImplementedError

    def get_action(self, observation: dict) -> int:
        raise NotImplementedError

    def update_q_table(
        self, observation: dict, action: int, reward: float, next_observation: dict
    ) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared behaviour
    # ------------------------------------------------------------------

    @staticmethod
    def sanitize(observation: dict) -> dict:
        """Response time is absent until the first scrape lands; treat it as 0."""
        response_time = observation.get("response_time")
        if response_time is None or np.isnan(response_time):
            observation["response_time"] = 0.0
        return observation

    def ensure_state(self, state_key: tuple) -> np.ndarray:
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key]

    def decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def select_action(self, q_values: np.ndarray) -> int:
        """Epsilon-greedy over the given Q-values, returning a replica count."""
        if np.random.rand() < self.epsilon:
            action = int(np.random.randint(1, self.n_actions + 1))
            kind = "EXPLORATION"
        else:
            # Break ties uniformly at random rather than with np.argmax, which
            # always returns the first max index. A freshly allocated state is
            # all zeros, so argmax would deterministically pick action 1 (the
            # minimum replica count) for every untrained state -- a systematic
            # under-provisioning bias whose strength differs per agent (worst
            # for the continuous agent, which sees mostly novel states), and so
            # a confound in a comparison that is precisely about state reuse.
            max_val = np.max(q_values)
            candidates = np.flatnonzero(q_values == max_val)
            action = int(np.random.choice(candidates)) + 1
            kind = "EXPLOITATION"
        self.logger.info(
            f"[Epsilon] value={self.epsilon:.6f} | action={kind} ({action})"
        )
        return action

    def get_q_values(self, observation: dict) -> np.ndarray:
        """Q-values over all actions for this observation."""
        return self.ensure_state(self.get_state_key(observation))

    def add_episode_count(self, count: int = 1) -> None:
        self.episodes_trained += count

    def save_model(self, filepath: str, episode_count: int = 0) -> None:
        try:
            model_data = {
                "agent_type": self.agent_type,
                "q_table": self.q_table,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "n_actions": self.n_actions,
                "created_at": self.created_at,
                "episodes_trained": episode_count,
            }
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with Path(filepath).open("wb") as f:
                pickle.dump(model_data, f)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save model to {filepath}: {e}")
            raise

    def load_model(self, filepath: str) -> None:
        try:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")

            with Path(filepath).open("rb") as f:
                model_data = pickle.load(f)  # noqa: S301

            saved_type = model_data.get("agent_type")
            if saved_type is not None and saved_type != self.agent_type:
                raise ValueError(
                    f"Model at {filepath} was trained as {saved_type}, "
                    f"but is being loaded into a {self.agent_type} agent. "
                    f"Their state keys are not interchangeable."
                )

            self.q_table = model_data["q_table"]
            self.learning_rate = model_data["learning_rate"]
            self.discount_factor = model_data["discount_factor"]
            self.epsilon = model_data["epsilon"]
            self.epsilon_min = model_data["epsilon_min"]
            self.epsilon_decay = model_data["epsilon_decay"]
            self.n_actions = model_data["n_actions"]
            self.created_at = model_data.get("created_at", None)
            self.episodes_trained = model_data.get("episodes_trained", 0)

            self.logger.info(f"Model loaded from {filepath}")
            self.logger.info(f"Q-table size: {len(self.q_table)} states")
        except Exception as e:
            self.logger.error(f"Failed to load model from {filepath}: {e}")
            raise

    def format_state(self, state_key: tuple) -> str:
        return f"({', '.join(str(part) for part in state_key)})"

    def show_model_summary(self, max_states: int = 100) -> None:
        print("\n" + "=" * 120)  # noqa: T201
        print(f"Agent Summary ({self.agent_type})".center(120))  # noqa: T201
        print("=" * 120)  # noqa: T201
        print(f"Created At        : {self.created_at}")  # noqa: T201
        print(f"Episodes Trained  : {self.episodes_trained}")  # noqa: T201
        print(f"Learning Rate     : {self.learning_rate}")  # noqa: T201
        print(f"Discount Factor   : {self.discount_factor}")  # noqa: T201
        print(f"Epsilon (current) : {self.epsilon:.5f}")  # noqa: T201
        print(f"Epsilon Min       : {self.epsilon_min}")  # noqa: T201
        print(f"Epsilon Decay     : {self.epsilon_decay}")  # noqa: T201
        print(f"Number of Actions : {self.n_actions}")  # noqa: T201
        print(f"Q-Table Size      : {len(self.q_table)} states")  # noqa: T201
        print("=" * 120)  # noqa: T201

        if not self.q_table:
            print("Q-table is empty - no states have been learned yet.")  # noqa: T201
            print("=" * 120)  # noqa: T201
            return

        total_states = len(self.q_table)
        states_to_show = (
            total_states if max_states is None else min(max_states, total_states)
        )

        print(f"Showing {states_to_show}/{total_states} Q-table states:\n")  # noqa: T201
        print("-" * 150)  # noqa: T201
        print(  # noqa: T201
            f"{'Idx':<5} {self.state_column_header:<85} "
            f"{'BestAct':<8} {'BestQ':<10} {'AvgQ':<10}"
        )
        print("-" * 150)  # noqa: T201

        for i, (state_key, actions) in enumerate(self.q_table.items()):
            if i >= states_to_show:
                break

            try:
                state_str = self.format_state(state_key)
            except Exception:
                state_str = str(state_key)

            best_action = int(np.argmax(actions)) + 1  # 1-based replica count
            best_value = float(np.max(actions))
            avg_value = float(np.mean(actions))

            print(  # noqa: T201
                f"{i + 1:<5} {state_str:<45} {best_action:<8} "
                f"{best_value:<10.4f} {avg_value:<10.4f}"
            )

            print(" " * 7 + "Q-Values (Replica -> Value):")  # noqa: T201
            actions_per_row = 10
            n_actions = len(actions)
            rows = math.ceil(n_actions / actions_per_row)

            for r in range(rows):
                start_idx = r * actions_per_row
                end_idx = min(start_idx + actions_per_row, n_actions)
                segment = actions[start_idx:end_idx]

                row_str = ""
                for j, q_val in enumerate(segment, start=start_idx):
                    row_str += f"[R{j + 1:02d}] {q_val:7.4f}   "
                print(" " * 9 + row_str)  # noqa: T201
            print("-" * 120)  # noqa: T201

        print("End of Q-table summary.")  # noqa: T201
        print("=" * 120 + "\n")  # noqa: T201


def resolve_model_type(agent: Any) -> str:
    """Directory name used to keep each agent's checkpoints apart."""
    return {
        "Q-LEARNING": "qlearning",
        "Q-LEARNING-CRISP": "qlearningcrisp",
        "Q-LEARNING-FUZZY": "qlearningfuzzy",
    }.get(getattr(agent, "agent_type", ""), "unknown")

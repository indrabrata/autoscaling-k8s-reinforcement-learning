import pickle
from logging import Logger
from pathlib import Path
from typing import Optional

import numpy as np
import urllib3
import numpy as np
import math
import numpy as np
import math

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class QLearning:
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 0.1,
        epsilon_decay: float = 0.0,
        epsilon_min: float = 0.01,
        created_at: int = 0,
        logger: Optional[Logger] = None,
    ):
        self.n_actions = 100
        self.agent_type = "Q"
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.created_at = created_at
        self.episodes_trained = 0
        self.q_table = {}
        self.logger = logger or Logger(__name__)

        self.logger.info("Initialized Q-learning agent")
        self.logger.info(f"Agent parameters: {self.__dict__}")

    def add_episode_count(self, count: int = 1):
        self.episodes_trained += count

    def get_state_key(self, observation: dict) -> tuple[int, int, int, int]:
        cpu = int(observation["cpu_usage"])
        memory = int(observation["memory_usage"])
        action = int(observation["last_action"])

        response_time_raw = observation["response_time"]
        if np.isnan(response_time_raw) or response_time_raw is None:
            response_time = 0
        else:
            response_time = int(response_time_raw)

        return (cpu, memory, response_time, action)

    def get_action(self, observation: dict) -> int:
        state_key = self.get_state_key(observation)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)

        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action = np.argmax(self.q_table[state_key])

        return action

    def update_q_table(
        self, observation: dict, action: int, reward: float, next_observation: dict
    ):
        state_key = self.get_state_key(observation)
        next_state_key = self.get_state_key(next_observation)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.n_actions)

        best_next_action = np.max(
            self.q_table[next_state_key]
        )
        self.q_table[state_key][action] += self.learning_rate * (
            reward
            + self.discount_factor * best_next_action
            - self.q_table[state_key][action]
        )
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filepath: str, episode_count: int = 0):
        """Save Q-table and parameters to file"""
        try:
            model_data = {
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

    def load_model(self, filepath: str):
        """Load Q-table and parameters from file"""
        try:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")

            with Path(filepath).open("rb") as f:
                model_data = pickle.load(f)  # noqa: S301

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

    def show_model_summary(self, max_states: int = None):
        print("\n" + "=" * 120)
        print(f"🧠 Q-Learning Agent Summary ({self.agent_type})".center(120))
        print("=" * 120)
        print(f"📅 Created At        : {self.created_at}")
        print(f"📈 Episodes Trained  : {self.episodes_trained}")
        print(f"🎯 Learning Rate     : {self.learning_rate}")
        print(f"💸 Discount Factor   : {self.discount_factor}")
        print(f"🎲 Epsilon (current) : {self.epsilon:.5f}")
        print(f"↘️  Epsilon Min       : {self.epsilon_min}")
        print(f"🌀 Epsilon Decay     : {self.epsilon_decay}")
        print(f"🔢 Number of Actions : {self.n_actions}")
        print(f"📊 Q-Table Size      : {len(self.q_table)} states")
        print("=" * 120)

        if not self.q_table:
            print("⚠️  Q-table is empty — no states have been learned yet.")
            print("=" * 120)
            return

        total_states = len(self.q_table)
        states_to_show = total_states if max_states is None else min(max_states, total_states)

        print(f"🧩 Showing {states_to_show}/{total_states} Q-table states:\n")
        print("-" * 120)
        print(f"{'Idx':<5} {'State (CPU, MEM, RESP, ACT)':<45} {'BestAct':<8} {'BestQ':<10} {'AvgQ':<10}")
        print("-" * 120)

        for i, (state_key, actions) in enumerate(self.q_table.items()):
            if i >= states_to_show:
                break

            try:
                cpu, mem, resp, act = state_key
                state_str = f"({cpu}, {mem}, {resp}, {act})"
            except Exception:
                state_str = str(state_key)

            best_action = int(np.argmax(actions))
            best_value = float(np.max(actions))
            avg_value = float(np.mean(actions))

            print(f"{i+1:<5} {state_str:<45} {best_action:<8} {best_value:<10.4f} {avg_value:<10.4f}")

            print(" " * 7 + "Q-Values (Action → Value):")
            actions_per_row = 10
            n_actions = len(actions)
            rows = math.ceil(n_actions / actions_per_row)

            for r in range(rows):
                start_idx = r * actions_per_row
                end_idx = min(start_idx + actions_per_row, n_actions)
                segment = actions[start_idx:end_idx]

                row_str = ""
                for j, q_val in enumerate(segment, start=start_idx):
                    row_str += f"[A{j:02d}] {q_val:7.4f}   "
                print(" " * 9 + row_str)
            print("-" * 120)

        print("✅ End of full Q-table summary.")
        print("=" * 120 + "\n")
                                
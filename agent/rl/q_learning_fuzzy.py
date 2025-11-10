import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple
from logging import Logger
import numpy as np
import urllib3

from .fuzzy import Fuzzy

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pprint

import numpy as np
import math



class QLearningFuzzy:
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 0.95,
        epsilon_decay: float = 0.9,
        epsilon_min: float = 0.1,
        created_at: int = 0,
        n_actions: int = 100,
        logger: Optional[Logger] = None,
    ):
        self.agent_type = "QFuzzyHybrid"
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.created_at = created_at
        self.episodes_trained = 0
        self.n_actions = n_actions
        self.q_table: Dict[Tuple[str, str, str], np.ndarray] = {}
        self.fuzzy = Fuzzy(logger=logger)
        self.logger = logger or Logger(__name__)

        self.logger.info("Initialized QFuzzyHybrid agent")
        self.logger.info(f"Agent parameters: {self.__dict__}")

    def get_state_key(self, observation: dict) -> Tuple[str, str, str]:
        response_time_raw = observation["response_time"]
        if np.isnan(response_time_raw) or response_time_raw is None:
            response_time = 0
        else:
            response_time = int(response_time_raw)

        observation["response_time"] = response_time
        fuzzy_state = self.fuzzy.fuzzify(observation)
        cpu_label = max(fuzzy_state["cpu_usage"], key=fuzzy_state["cpu_usage"].get)
        mem_label = max(fuzzy_state["memory_usage"], key=fuzzy_state["memory_usage"].get)
        resp_label = max(fuzzy_state["response_time"], key=fuzzy_state["response_time"].get)
        return (cpu_label, mem_label, resp_label)

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
        self,
        observation: dict,
        action: int,
        reward: float,
        next_observation: dict,
    ):
        state_key = self.get_state_key(observation)
        next_state_key = self.get_state_key(next_observation)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.n_actions)

        best_next_action = np.max(self.q_table[next_state_key])
        old_value = self.q_table[state_key][action]

        self.q_table[state_key][action] += self.learning_rate * (
            reward + self.discount_factor * best_next_action - old_value
        )

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.logger.debug(
            f"Q-update | S={state_key} | A={action} | R={reward:.3f} | "
            f"NewQ={self.q_table[state_key][action]:.3f}"
        )

    def add_episode_count(self, count: int = 1):
        self.episodes_trained += count

    def save_model(self, filepath: str, episode_count: int = 0):
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
            self.logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, filepath: str):
        try:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")

            with Path(filepath).open("rb") as f:
                model_data = pickle.load(f)

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
            self.logger.error(f"Failed to load model: {e}")
            raise

    def show_model_summary(self, max_states: int = None):
        print("\n" + "=" * 120)
        print(f"🧠 Q-Fuzzy Hybrid Agent Summary ({self.agent_type})".center(120))
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
            print("⚠️  Q-table is empty — no fuzzy states have been learned yet.")
            print("=" * 120)
            return

        total_states = len(self.q_table)
        states_to_show = total_states if max_states is None else min(max_states, total_states)

        print(f"🧩 Showing {states_to_show}/{total_states} fuzzy Q-table states:\n")
        print("-" * 120)
        print(f"{'Idx':<5} {'Fuzzy State (CPU, MEM, RESP)':<45} {'BestAct':<8} {'BestQ':<10} {'AvgQ':<10}")
        print("-" * 120)

        for i, (state_key, actions) in enumerate(self.q_table.items()):
            if i >= states_to_show:
                break

            try:
                cpu_label, mem_label, resp_label = state_key
                fuzzy_state_str = f"({cpu_label}, {mem_label}, {resp_label})"
            except Exception:
                fuzzy_state_str = str(state_key)

            best_action = int(np.argmax(actions))
            best_value = float(np.max(actions))
            avg_value = float(np.mean(actions))

            print(f"{i+1:<5} {fuzzy_state_str:<45} {best_action:<8} {best_value:<10.4f} {avg_value:<10.4f}")

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

        print("✅ End of full fuzzy Q-table summary.")
        print("=" * 120 + "\n")



import csv
import logging
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Optional

from environment import KubernetesEnv
from rl import QLearningBase, resolve_model_type
from utils import log_verbose_details

CSV_COLUMNS = [
    "timestamp",
    "episode",
    "step",
    "action",
    "reward",
    "cumulative_reward",
    "episode_reward",
    "epsilon",
    "q_table_size",
    "active_states",
    "iteration",
    "replica_state",
    "cpu_usage",
    "memory_usage",
    "response_time",
    "response_time_percentage",
    "r_rt",
    "r_cost",
]


@dataclass
class SaveConfig:
    note: str
    start_time: int


class Trainer:
    def __init__(
        self,
        agent: QLearningBase,
        env: KubernetesEnv,
        logger: Optional[logging.Logger] = None,
        resume: bool = False,
        resume_path: str = "",
        reset_epsilon: bool = True,
        change_epsilon_decay: Optional[float] = None,
        csv_output: str = "",
    ) -> None:
        self.agent = agent
        self.env = env
        self.logger = logger or logging.getLogger(__name__)
        self.savecfg: Optional[SaveConfig] = None
        self._old_sigint = None
        self._old_sigterm = None
        self._csv_path = csv_output
        self._csv_file = None
        self._csv_writer = None
        if resume and resume_path:
            try:
                start_epsilon = self.agent.epsilon
                self.agent.load_model(resume_path)
                self.logger.info(f"Resumed training from model at: {resume_path}")
                if reset_epsilon:
                    self.agent.epsilon = start_epsilon
                    self.logger.info(f"Epsilon reset to {self.agent.epsilon}")
                if change_epsilon_decay is not None:
                    self.agent.epsilon_decay = change_epsilon_decay
                    self.logger.info(
                        f"Epsilon decay changed to {self.agent.epsilon_decay}"
                    )
            except Exception as e:
                self.logger.error(f"Failed to load model from {resume_path}: {e}")
                raise

    def _interrupted_save(self) -> Optional[Path]:
        if not (self.agent and self.savecfg):
            self.logger.warning("Nothing to save yet.")
            return None

        model_type = resolve_model_type(self.agent)
        interrupted_dir = Path(
            f"model/{model_type}/"
            f"{self.savecfg.start_time}_{self.savecfg.note}/interrupted"
        )
        interrupted_dir.mkdir(parents=True, exist_ok=True)

        ts = int(time.time())
        ep = getattr(self.agent, "episodes_trained", 0)
        path = interrupted_dir / f"interrupted_episode_{ep}_{ts}.pkl"

        self.agent.save_model(str(path), ep)
        self.logger.info(f"Model saved to: {path}")
        self.logger.info(f"Episodes completed: {ep}")
        return path

    def _signal_handler(self, signum: int, frame: FrameType) -> None:
        self.logger.warning(f"\nSignal {signum} received. Saving model...")
        try:
            self._interrupted_save()
        finally:
            signal.signal(signal.SIGINT, self._old_sigint or signal.SIG_DFL)
            signal.signal(signal.SIGTERM, self._old_sigterm or signal.SIG_DFL)
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(0)

    def _install_signal_handlers(self) -> None:
        self._old_sigint = signal.getsignal(signal.SIGINT)
        self._old_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _restore_signal_handlers(self) -> None:
        if self._old_sigint is not None:
            signal.signal(signal.SIGINT, self._old_sigint)
        if self._old_sigterm is not None:
            signal.signal(signal.SIGTERM, self._old_sigterm)

    def _open_csv(self) -> None:
        if not self._csv_path:
            return
        path = Path(self._csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_COLUMNS)
        self._csv_writer.writeheader()
        self._csv_file.flush()
        self.logger.info(f"Training CSV logging to: {path}")

    def _count_active_states(self, observation: dict) -> int:
        """How many Q-table states this observation touched (1 unless fuzzy)."""
        if not hasattr(self.agent, "get_activations"):
            return 1
        try:
            return len(self.agent.get_activations(observation))
        except Exception:
            return 1

    def _write_csv_row(
        self, episode: int, step: int, action: int, observation: dict, info: dict
    ) -> None:
        if self._csv_writer is None:
            return
        row = {
            "timestamp": int(time.time()),
            "episode": episode,
            "step": step,
            "action": action,
            "reward": info.get("reward", 0.0),
            "cumulative_reward": self.env.cumulative_reward,
            "episode_reward": self.env.episode_reward,
            "epsilon": self.agent.epsilon,
            "q_table_size": len(self.agent.q_table),
            "active_states": self._count_active_states(observation),
            "iteration": info.get("iteration", 0),
            "replica_state": info.get("replica_state", 0),
            "cpu_usage": info.get("cpu_usage", 0.0),
            "memory_usage": info.get("memory_usage", 0.0),
            "response_time": info.get("response_time", 0.0),
            "response_time_percentage": info.get("response_time_percentage", 0.0),
            "r_rt": info.get("r_rt", 0.0),
            "r_cost": info.get("r_cost", 0.0),
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def _close_csv(self) -> None:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def train(self, episodes: int, note: str, start_time: int) -> None:
        self.savecfg = SaveConfig(note=note, start_time=start_time)
        self._install_signal_handlers()
        self._open_csv()

        try:
            total_best = float("-inf")
            for ep in range(episodes):
                self.agent.add_episode_count()
                self.logger.info(f"\nEpisode {ep + 1}/{episodes}")
                self.logger.info(
                    f"Total episodes trained: {self.agent.episodes_trained}"
                )

                obs = self.env.reset()
                total = 0.0
                step = 0
                while True:
                    step += 1
                    act = self.agent.get_action(obs)
                    nxt, rew, term, info = self.env.step(
                        act, q_table_size=len(self.agent.q_table)
                    )
                    self.agent.update_q_table(obs, act, rew, nxt)
                    total += rew

                    self._write_csv_row(ep + 1, step, act, obs, info)
                    obs = nxt

                    self.logger.info(
                        f"Action: {act}, Reward: {rew}, Total: {total} | "
                        f"Iteration: {info['iteration']}"
                    )

                    self.logger.debug(f"Observation type: {type(obs)}, value: {obs}")

                    log_verbose_details(
                        observation=obs,
                        agent=self.agent,
                        verbose=True,
                        logger=self.logger,
                    )

                    if term:
                        break

                self.logger.info(f"Episode {ep + 1} completed. Total reward: {total}")
                if total > total_best:
                    total_best = total
                    self._save_checkpoint(ep, total_best, note, start_time)

        except KeyboardInterrupt:
            self.logger.warning("Interrupted by user. Attempting to save...")
            self._interrupted_save()
            raise
        finally:
            self._close_csv()
            self._restore_signal_handlers()
            self.logger.info("Training finished (cleanup done).")

    def _save_checkpoint(
        self, episode: int, score: float, note: str, start_time: int
    ) -> None:
        model_type = resolve_model_type(self.agent)
        path = (
            f"model/{model_type}/{start_time}_{note}/checkpoints/"
            f"episode_{episode}_total_{score}.pkl"
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.agent.save_model(path, episode + 1)
        self.logger.info(f"New best model saved with total reward: {score}")

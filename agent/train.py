import ast
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from database import InfluxDB
from dotenv import load_dotenv
from environment import KubernetesEnv
from rl import QLearning, QLearningFuzzy
from trainer import Trainer
from utils import setup_logger

load_dotenv()


def find_latest_checkpoint(algorithm: str) -> Optional[str]:
    """
    Find the latest checkpoint for the given algorithm.
    Searches in model/{algorithm_type}/**/checkpoints/ and model/{algorithm_type}/**/interrupted/
    Returns the most recent checkpoint path or None if not found.
    """
    model_type = "qlearningfuzzy" if algorithm == "Q-LEARNING-FUZZY" else "qlearning"
    model_base = Path("model") / model_type

    if not model_base.exists():
        return None

    # Search for all checkpoint files
    checkpoint_files = []

    # Search in checkpoints directories
    for checkpoint_file in model_base.rglob("checkpoints/*.pkl"):
        checkpoint_files.append(checkpoint_file)

    # Search in interrupted directories
    for interrupted_file in model_base.rglob("interrupted/*.pkl"):
        checkpoint_files.append(interrupted_file)

    if not checkpoint_files:
        return None

    # Sort by modification time, most recent first
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    return str(latest_checkpoint)

if __name__ == "__main__":
    start_time = int(time.time())
    logger = setup_logger(
        "kubernetes_agent",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_to_file=True,
    )

    influxdb = InfluxDB(
        logger=logger,
        url=os.getenv("INFLUXDB_URL", "http://localhost:8086"),
        token=os.getenv("INFLUXDB_TOKEN", "my-token"),
        org=os.getenv("INFLUXDB_ORG", "my-org"),
        bucket=os.getenv("INFLUXDB_BUCKET", "my-bucket"),
    )

    try:
        metrics_endpoints_str = os.getenv(
            "METRICS_ENDPOINTS_METHOD", "[['/', 'GET'], ['/docs', 'GET']]"
        )
        metrics_endpoints_method = ast.literal_eval(metrics_endpoints_str)
    except (ValueError, SyntaxError):
        logger.warning("Invalid METRICS_ENDPOINTS_METHOD format, using default")
        metrics_endpoints_method = [["/", "GET"], ["/docs", "GET"]]

    choose_algorithm = os.getenv("ALGORITHM", "Q-LEARNING").upper()

    env = KubernetesEnv(
        min_replicas=int(os.getenv("MIN_REPLICAS", "1")),
        max_replicas=int(os.getenv("MAX_REPLICAS", "12")),
        iteration=int(os.getenv("ITERATION", "10")),
        namespace=os.getenv("NAMESPACE", "default"),
        deployment_name=os.getenv("DEPLOYMENT_NAME", "ecom-api"),
        min_cpu=int(os.getenv("MIN_CPU", "10")),
        min_memory=int(os.getenv("MIN_MEMORY", "10")),
        max_cpu=int(os.getenv("MAX_CPU", "90")),
        max_memory=int(os.getenv("MAX_MEMORY", "90")),
        max_response_time=float(os.getenv("MAX_RESPONSE_TIME", "100.0")),
        timeout=int(os.getenv("TIMEOUT", "120")),
        wait_time=int(os.getenv("WAIT_TIME", "60")),
        verbose=True,
        logger=logger,
        influxdb=influxdb,
        prometheus_url=os.getenv("PROMETHEUS_URL", "http://localhost:1234/prom"),
        metrics_endpoints_method=metrics_endpoints_method,
        metrics_interval=int(os.getenv("METRICS_INTERVAL", "15")),
        metrics_quantile=float(os.getenv("METRICS_QUANTILE", "0.90")),
        max_scaling_retries=int(os.getenv("MAX_SCALING_RETRIES", "1000")),
        response_time_weight=float(os.getenv("RESPONSE_TIME_WEIGHT", "1.0")),
        cpu_memory_weight=float(os.getenv("CPU_MEMORY_WEIGHT", "0.5")),
        cost_weight=float(os.getenv("COST_WEIGHT", "0.3")),
        algorithm=choose_algorithm,
    )

    if choose_algorithm == "Q-LEARNING":
        algorithm = QLearning(
            learning_rate=float(os.getenv("LEARNING_RATE", 0.1)),
            discount_factor=float(os.getenv("DISCOUNT_FACTOR", 0.95)),
            epsilon_start=float(os.getenv("EPSILON_START", 0.1)),
            epsilon_decay=float(os.getenv("EPSILON_DECAY", 0.99)),
            epsilon_min=float(os.getenv("EPSILON_MIN", 0.01)),
            created_at=start_time,
            logger=logger,
        )

    elif choose_algorithm == "Q-LEARNING-FUZZY":
        algorithm = QLearningFuzzy(
            learning_rate=float(os.getenv("LEARNING_RATE", 0.1)),
            discount_factor=float(os.getenv("DISCOUNT_FACTOR", 0.95)),
            epsilon_start=float(os.getenv("EPSILON_START", 0.1)),
            epsilon_decay=float(os.getenv("EPSILON_DECAY", 0.99)),
            epsilon_min=float(os.getenv("EPSILON_MIN", 0.01)),
            created_at=start_time,
            logger=logger,
        )

    else:
        raise ValueError(f"Unsupported algorithm: {choose_algorithm}")

    note = os.getenv("NOTE", "default")

    # Handle AUTO_RESUME and RESUME logic
    auto_resume = ast.literal_eval(os.getenv("AUTO_RESUME", "False"))
    resume = ast.literal_eval(os.getenv("RESUME", "False"))
    resume_path = os.getenv("RESUME_PATH", "")

    # Determine final resume settings
    final_resume = False
    final_resume_path = ""

    if auto_resume:
        logger.info("AUTO_RESUME is enabled. Searching for latest checkpoint...")
        latest_checkpoint = find_latest_checkpoint(choose_algorithm)
        if latest_checkpoint:
            final_resume = True
            final_resume_path = latest_checkpoint
            logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        else:
            logger.warning("No checkpoint found. Starting from scratch.")
    elif resume and resume_path:
        final_resume = True
        final_resume_path = resume_path
        logger.info(f"Resuming from specified path: {resume_path}")
    elif resume and not resume_path:
        logger.warning("RESUME is True but RESUME_PATH is empty. Starting from scratch.")

    trainer = Trainer(
        agent=algorithm,
        env=env,
        logger=logger,
        resume=final_resume,
        resume_path=final_resume_path,
        reset_epsilon=ast.literal_eval(os.getenv("RESET_EPSILON", "True")),
        change_epsilon_decay=float(os.getenv("EPSILON_DECAY", 0.90)),
    )

    episodes = int(os.getenv("EPISODES", "10"))
    trainer.train(episodes=episodes, note=note, start_time=start_time)

    if hasattr(trainer.agent, "q_table"):
        logger.info(f"\nQ-table size: {len(trainer.agent.q_table)} states")
        logger.info("Sample Q-values:")
        for i, (state, q_values) in enumerate(list(trainer.agent.q_table.items())[:5]):
            max_q = np.max(q_values)
            best_action = np.argmax(q_values)
            logger.info(
                f"State {state}: Best action = {best_action}, Max Q-value = {max_q:.3f}"
            )
    else:
        logger.info("\nNo Q-table to display")

    model_type = (
        "qlearningfuzzy" if isinstance(trainer.agent, QLearningFuzzy)
        else "qlearning" if isinstance(trainer.agent, QLearning)
        else "unknown"
    )
    model_dir = Path(f"model/{model_type}/{start_time}_{note}/final")
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    model_file = model_dir / f"{model_type}_{timestamp}.pkl"
    trainer.agent.save_model(str(model_file), trainer.agent.episodes_trained)

    logger.info(f"Model saved to: {model_file}")

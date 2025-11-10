
import os
import time
from dotenv import load_dotenv

from rl.q_learning import QLearning
from rl.q_learning_fuzzy import QLearningFuzzy
from utils.logger import setup_logger


load_dotenv()

if __name__ == "__main__":
    start_time = int(time.time())
    logger = setup_logger(
        "kubernetes_agent",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_to_file=True,
    )

    choose_algorithm = os.getenv("ALGORITHM", "Q").upper()
  
    if choose_algorithm == "Q":
        agent = QLearning(
            learning_rate=float(os.getenv("LEARNING_RATE", 0.1)),
            discount_factor=float(os.getenv("DISCOUNT_FACTOR", 0.95)),
            epsilon_start=0,
            epsilon_decay=0,
            epsilon_min=0,
            created_at=start_time,
            logger=logger,
        )

    elif choose_algorithm in ("Q-FUZZY", "QFUZZYHYBRID", "Q_FUZZY"):
        agent = QLearningFuzzy(
            learning_rate=float(os.getenv("LEARNING_RATE", 0.1)),
            discount_factor=float(os.getenv("DISCOUNT_FACTOR", 0.95)),
            epsilon_start=0,
            epsilon_decay=0,
            epsilon_min=0,
            created_at=start_time,
            logger=logger,
        )
        
    else:
        raise ValueError(f"Unsupported algorithm: {choose_algorithm}")
        
    model_path = os.getenv("MODEL_PATH", "")
    if model_path == "":
        raise ValueError(f"Invalid model path: {model_path}")
      
    agent.load_model(model_path)
    agent.show_model_summary(max_states=125)
        
import ast
import os
import threading
import time
from datetime import datetime

from dotenv import load_dotenv

from collect_metrics import run_collector
from environment import KubernetesEnv
from rl import AGENTS, create_agent, resolve_model_type
from utils import log_verbose_details, setup_logger

load_dotenv()

if __name__ == "__main__":
    start_time = int(time.time())
    logger = setup_logger(
        "test_model",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_to_file=True,
    )

    try:
        metrics_endpoints_str = os.getenv(
            "METRICS_ENDPOINTS_METHOD", "[['/', 'GET'], ['/docs', 'GET']]"
        )
        metrics_endpoints_method = ast.literal_eval(metrics_endpoints_str)
    except (ValueError, SyntaxError):
        logger.warning("Invalid METRICS_ENDPOINTS_METHOD format, using default")
        metrics_endpoints_method = [("/", "GET"), ("/docs", "GET")]

    choose_algorithm = os.getenv("ALGORITHM", "Q-LEARNING").upper()
    if choose_algorithm not in AGENTS:
        raise ValueError(
            f"Unsupported algorithm: {choose_algorithm}. "
            f"Expected one of: {', '.join(sorted(AGENTS))}"
        )

    max_replicas = int(os.getenv("MAX_REPLICAS", "10"))

    env = KubernetesEnv(
        min_replicas=int(os.getenv("MIN_REPLICAS", "1")),
        max_replicas=max_replicas,
        iteration=int(os.getenv("ITERATION", "10")),
        namespace=os.getenv("NAMESPACE", "default"),
        deployment_name=os.getenv("DEPLOYMENT_NAME", "resource-intensive-q"),
        min_cpu=int(os.getenv("MIN_CPU", "10")),
        min_memory=int(os.getenv("MIN_MEMORY", "10")),
        max_cpu=int(os.getenv("MAX_CPU", "90")),
        max_memory=int(os.getenv("MAX_MEMORY", "90")),
        max_response_time=float(os.getenv("MAX_RESPONSE_TIME", "100.0")),
        timeout=int(os.getenv("TIMEOUT", "120")),
        wait_time=int(os.getenv("WAIT_TIME", "60")),
        verbose=True,
        logger=logger,
        prometheus_url=os.getenv("PROMETHEUS_URL", "http://localhost:1234/prom"),
        metrics_endpoints_method=metrics_endpoints_method,
        metrics_interval=int(os.getenv("METRICS_INTERVAL", "15")),
        metrics_quantile=float(os.getenv("METRICS_QUANTILE", "0.90")),
        max_scaling_retries=int(os.getenv("MAX_SCALING_RETRIES", "1000")),
        algorithm=choose_algorithm,
    )

    agent = create_agent(
        algorithm=choose_algorithm,
        learning_rate=float(os.getenv("LEARNING_RATE", "0.1")),
        discount_factor=float(os.getenv("DISCOUNT_FACTOR", "0.95")),
        epsilon_start=float(os.getenv("EPSILON_START", "0.1")),
        epsilon_decay=float(os.getenv("EPSILON_DECAY", "0.99")),
        epsilon_min=float(os.getenv("EPSILON_MIN", "0.01")),
        created_at=start_time,
        n_actions=max_replicas,
        logger=logger,
    )

    model_path = os.getenv("MODEL_PATH", "")
    if model_path == "":
        raise ValueError(f"Invalid model path: {model_path}")

    agent.load_model(model_path)
    agent.epsilon = 0  # Pure exploitation, no exploration

    # Start metrics collector in background thread
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_output = os.getenv(
        "METRICS_OUTPUT_PATH",
        f"metrics_output/test_{resolve_model_type(agent)}_{now}.csv",
    )
    collect_interval = int(os.getenv("COLLECT_INTERVAL", "15"))

    collector_thread = threading.Thread(
        target=run_collector,
        kwargs={
            "prometheus_url": os.getenv("PROMETHEUS_URL", "http://localhost:1234/prom"),
            "namespace": os.getenv("NAMESPACE", "default"),
            "deployment_name": os.getenv("DEPLOYMENT_NAME", "resource-intensive-q"),
            "output_path": csv_output,
            "collect_interval": collect_interval,
            "metrics_interval": int(os.getenv("METRICS_INTERVAL", "15")),
            "quantile": float(os.getenv("METRICS_QUANTILE", "0.90")),
            "endpoints_method": metrics_endpoints_method,
            "logger": logger,
        },
        daemon=True,
    )
    collector_thread.start()
    logger.info(f"Background metrics collector started -> {csv_output}")

    # Run the model
    obs = env.reset()
    logger.info("=" * 80)
    logger.info("MODEL TESTING STARTED (CSV metrics collection active)")
    logger.info(f"  Algorithm:  {choose_algorithm}")
    logger.info(f"  Model:      {model_path}")
    logger.info(f"  Q-table:    {len(agent.q_table)} states")
    logger.info(f"  Epsilon:    {agent.epsilon} (pure exploitation)")
    logger.info(f"  CSV output: {csv_output}")
    logger.info("=" * 80)

    step_count = 0
    while True:
        act = agent.get_action(obs)
        nxt, rew, term, info = env.step(act, q_table_size=len(agent.q_table))
        obs = nxt
        step_count += 1

        logger.info(
            f"[Step {step_count}] action={act}  reward={rew:.4f}  "
            f"replicas={info.get('replica_state', '?')}  "
            f"CPU={info.get('cpu_usage', 0):.1f}%  "
            f"MEM={info.get('memory_usage', 0):.1f}%  "
            f"RT={info.get('response_time', 0):.1f}ms"
        )

        log_verbose_details(
            observation=obs,
            agent=agent,
            verbose=True,
            logger=logger,
        )

        if term:
            logger.info(f"Testing completed after {step_count} steps.")
            break

    # Stop the collector gracefully
    import collect_metrics

    collect_metrics._running = False
    collector_thread.join(timeout=10)

    logger.info(f"Results saved to {csv_output}")

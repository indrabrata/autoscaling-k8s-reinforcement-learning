"""
Test reward calculation to debug why rewards are all 0
"""
import os
import sys
from dotenv import load_dotenv

# Add agent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from environment.environment import KubernetesEnv
from utils.logger import setup_logger

logger = setup_logger("test_reward", log_level="DEBUG")

# Create mock environment
env = KubernetesEnv(
    min_replicas=int(os.getenv("MIN_REPLICAS", 1)),
    max_replicas=int(os.getenv("MAX_REPLICAS", 50)),
    iteration=int(os.getenv("ITERATION", 50)),
    namespace=os.getenv("NAMESPACE", "default"),
    deployment_name=os.getenv("DEPLOYMENT_NAME", "resource-intensive"),
    min_cpu=float(os.getenv("MIN_CPU", 15)),
    min_memory=float(os.getenv("MIN_MEMORY", 15)),
    max_cpu=float(os.getenv("MAX_CPU", 90)),
    max_memory=float(os.getenv("MAX_MEMORY", 90)),
    max_response_time=float(os.getenv("MAX_RESPONSE_TIME", 1500)),
    timeout=int(os.getenv("TIMEOUT", 120)),
    wait_time=int(os.getenv("WAIT_TIME", 45)),
    logger=logger,
    influxdb=None,  # No InfluxDB for testing
    prometheus_url=os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
    response_time_weight=float(os.getenv("RESPONSE_TIME_WEIGHT", 1.0)),
    cpu_memory_weight=float(os.getenv("CPU_MEMORY_WEIGHT", 0.5)),
    cost_weight=float(os.getenv("COST_WEIGHT", 0.3)),
    request_rate_per_pod_capacity=float(os.getenv("REQUEST_RATE_PER_POD_CAPACITY", 25.0)),
    algorithm=os.getenv("ALGORITHM", "Q-LEARNING"),
)

# Test scenarios
test_cases = [
    {
        "name": "Optimal State",
        "cpu": 50.0,
        "memory": 50.0,
        "response_time": 500.0,  # 33% of 1500ms max
        "request_rate": 60.0,
        "replicas": 10,
    },
    {
        "name": "Wasteful State (Over-provisioned)",
        "cpu": 5.0,
        "memory": 5.0,
        "response_time": 100.0,
        "request_rate": 10.0,
        "replicas": 30,
    },
    {
        "name": "Critical State (Under-provisioned)",
        "cpu": 95.0,
        "memory": 92.0,
        "response_time": 1800.0,  # 120% of max
        "request_rate": 200.0,
        "replicas": 3,
    },
    {
        "name": "Balanced State",
        "cpu": 60.0,
        "memory": 55.0,
        "response_time": 800.0,
        "request_rate": 80.0,
        "replicas": 8,
    },
]

print("\n" + "=" * 100)
print("REWARD FUNCTION TEST")
print("=" * 100)
print(f"Algorithm: {env.algorithm}")
print(f"Request Rate Per Pod Capacity: {env.per_pod_capacity} RPS")
print(f"Weights: RT={env.response_time_weight}, CPU/MEM={env.cpu_memory_weight}, COST={env.cost_weight}")
print("=" * 100 + "\n")

for test in test_cases:
    print(f"\n{'='*100}")
    print(f"Test: {test['name']}")
    print(f"{'='*100}")

    # Set environment state
    env.cpu_usage = test["cpu"]
    env.memory_usage = test["memory"]
    env.response_time = test["response_time"]
    env.request_rate = test["request_rate"]
    env.replica_state = test["replicas"]
    env.replica_state_old = test["replicas"]

    # Calculate reward
    reward, breakdown = env._calculate_reward()

    print(f"\nMetrics:")
    print(f"  CPU:           {test['cpu']:.1f}%")
    print(f"  Memory:        {test['memory']:.1f}%")
    print(f"  Response Time: {test['response_time']:.1f}ms")
    print(f"  Request Rate:  {test['request_rate']:.1f} RPS")
    print(f"  Replicas:      {test['replicas']}")
    print(f"  Current Capacity: {env.per_pod_capacity * test['replicas']:.1f} RPS")
    print(f"  Request Rate Normalized: {breakdown.get('request_rate_normalized', 0):.1f}%")

    print(f"\nReward Breakdown:")
    print(f"  Total Reward:  {reward:.4f}")
    print(f"  Optimal:       {breakdown.get('optimal', 0):.3f}")
    print(f"  Balanced:      {breakdown.get('balanced', 0):.3f}")
    print(f"  Wasteful:      {breakdown.get('wasteful', 0):.3f}")
    print(f"  Critical:      {breakdown.get('critical', 0):.3f}")
    print(f"  Cost Penalty:  {breakdown.get('cost_penalty', 0):.3f}")

    if reward == 0.0:
        print(f"\n⚠️  WARNING: Reward is EXACTLY 0.0!")
        print(f"  Positive Contribution: {breakdown.get('positive_contribution', 0):.3f}")
        print(f"  Negative Contribution: {breakdown.get('negative_contribution', 0):.3f}")

print("\n" + "=" * 100)
print("END OF TESTS")
print("=" * 100 + "\n")

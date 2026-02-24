# Training and Prediction Flow Analysis: Kubernetes Autoscaling with RL

## Table of Contents

- [PART A: TRAINING FLOW](#part-a-training-flow)
  - [1. System Initialization](#1-system-initialization)
  - [2. Key Components](#2-key-components)
  - [3. Training Loop — Trainer Class](#3-training-loop--trainer-class)
  - [4. Environment: reset()](#4-environment-reset)
  - [5. Agent: get\_action()](#5-agent-get_action)
  - [6. Environment: step()](#6-environment-step)
  - [7. Kubernetes Cluster Interaction](#7-kubernetes-cluster-interaction)
  - [8. Prometheus Metrics Collection](#8-prometheus-metrics-collection)
  - [9. Observation (State Representation)](#9-observation-state-representation)
  - [10. Agent: update\_q\_table()](#10-agent-update_q_table)
  - [11. State Key Differences: Q-Learning vs Q-Fuzzy](#11-state-key-differences-q-learning-vs-q-fuzzy)
  - [12. Fuzzy Logic — State Fuzzification](#12-fuzzy-logic--state-fuzzification)
  - [13. Checkpoint and Model Saving](#13-checkpoint-and-model-saving)
  - [14. Training Hyperparameters](#14-training-hyperparameters)
  - [15. End-to-End Training Diagram](#15-end-to-end-training-diagram)
- [PART B: PREDICTION FLOW](#part-b-prediction-flow)
  - [16. Prediction Initialization](#16-prediction-initialization)
  - [17. Prediction Loop](#17-prediction-loop)
  - [18. Training vs Prediction Differences](#18-training-vs-prediction-differences)
  - [19. End-to-End Prediction Diagram](#19-end-to-end-prediction-diagram)
- [PART C: THESIS DEFENSE Q&A](#part-c-thesis-defense-qa)

---

## PART A: TRAINING FLOW

## 1. System Initialization

Training starts from `train.py`. The initialization has 5 stages:

### 1.1 Logger Setup

```python
logger = setup_logger(
    "kubernetes_agent",
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_to_file=True,
)
```

The logger writes to both console and a rotating file (max 10 MB per file, 5 backups). Log files are stored in `logs/{YYYY-MM-DD-HH-MM}/`.

### 1.2 InfluxDB Connection

```python
influxdb = InfluxDB(
    url="http://localhost:8086",
    token="my-token",
    org="my-org",
    bucket="my-bucket",
)
```

InfluxDB stores training metrics for every iteration — reward, CPU, memory, response time, replica count, Q-table size, etc. This data is used for post-training analysis.

### 1.3 Environment Initialization

```python
env = KubernetesEnv(
    min_replicas=1,
    max_replicas=12,
    iteration=10,
    max_response_time=100.0,
    timeout=120,
    wait_time=60,
    algorithm="Q-LEARNING",  # or "Q-LEARNING-FUZZY"
)
```

The environment connects directly to:

- **Kubernetes API** — to scale deployments (add/remove pods)
- **Prometheus** — to collect real-time metrics (CPU, memory, response time)
- **InfluxDB** — to log training metrics

### 1.4 Agent Initialization

```python
# Q-Learning
algorithm = QLearning(
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon_start=0.1,
    epsilon_decay=0.99,
    epsilon_min=0.01,
)

# OR Q-Learning Fuzzy
algorithm = QLearningFuzzy(
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon_start=0.1,
    epsilon_decay=0.99,
    epsilon_min=0.01,
)
```

Both agents have:

- **Q-table**: dictionary `{state_key: numpy.array(n_actions)}` — stores Q-values for each action
- **n\_actions = max\_replicas**: action space maps directly to replica count (1 to max\_replicas)
- Q-table is 0-indexed: index `i` stores the Q-value for replica count `i+1`
- `get_action()` returns 1-based replica count

### 1.5 Trainer Initialization

```python
trainer = Trainer(
    agent=algorithm,
    env=env,
    resume=True,
    resume_path="path/to/model.pkl",
    reset_epsilon=True,
    change_epsilon_decay=0.90,
)
```

The Trainer supports **resume training** — continuing from a previous checkpoint. When resuming:

- Q-table is loaded from the pickle file
- Epsilon can be reset (restart exploration) or continued
- Epsilon decay can be changed (e.g., more aggressive)

---

## 2. Key Components

### System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                          train.py                            │
│  (Entry point: initialize all components, start training)   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                         Trainer                              │
│  (Orchestrator: episode loop, checkpoint, signal handling)  │
│                                                             │
│  ┌───────────────┐     ┌───────────────────────────────┐   │
│  │     Agent     │     │         Environment            │   │
│  │               │     │                               │   │
│  │  QLearning    │     │  KubernetesEnv                │   │
│  │  OR           │◄───►│                               │   │
│  │  QLearningF.  │     │  ┌──────────┐  ┌──────────┐  │   │
│  │               │     │  │ K8s API  │  │Prometheus│  │   │
│  │  ┌──────────┐ │     │  └──────────┘  └──────────┘  │   │
│  │  │ Q-Table  │ │     │                               │   │
│  │  └──────────┘ │     │  ┌───────────────────────┐   │   │
│  │               │     │  │       InfluxDB         │   │   │
│  │  ┌──────────┐ │     │  └───────────────────────┘   │   │
│  │  │  Fuzzy   │ │     │                               │   │
│  │  │(Q-Fuzzy) │ │     │                               │   │
│  │  └──────────┘ │     │                               │   │
│  └───────────────┘     └───────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### File Responsibilities

| File                         | Class/Function          | Responsibility                                        |
| ---------------------------- | ----------------------- | ----------------------------------------------------- |
| `train.py`                   | —                       | Entry point, initialization, configuration            |
| `trainer.py`                 | `Trainer`               | Episode loop orchestration, checkpoint, signals       |
| `environment/environment.py` | `KubernetesEnv`         | Kubernetes interface, reward calculation, state       |
| `rl/q_learning.py`           | `QLearning`             | Q-Learning agent with continuous state                |
| `rl/q_learning_fuzzy.py`     | `QLearningFuzzy`        | Q-Learning agent with fuzzified state                 |
| `rl/fuzzy.py`                | `Fuzzy`                 | State fuzzification, membership functions             |
| `utils/metrics.py`           | `get_metrics()`         | Prometheus queries, CPU/memory/RT computation         |
| `utils/cluster.py`           | `wait_for_pods_ready()` | Wait for pods to become ready after scaling           |
| `database/influxdb.py`       | `InfluxDB`              | Metrics storage to InfluxDB                           |

---

## 3. Training Loop — Trainer Class

The training loop is in `trainer.py`:

```python
def train(self, episodes, note, start_time):
    self._install_signal_handlers()
    total_best = float("-inf")

    for ep in range(episodes):
        agent.add_episode_count()
        obs = env.reset()
        total = 0.0

        while True:
            act = agent.get_action(obs)
            nxt, rew, term, info = env.step(act)
            agent.update_q_table(obs, act, rew, nxt)
            total += rew
            obs = nxt

            if term:
                break

        if total > total_best:
            total_best = total
            self._save_checkpoint(ep, total_best, note, start_time)
```

### Episode Structure

```text
Training Session
├── Episode 1  (N iterations)
│   ├── Step 1:  get_action → step → update_q_table
│   ├── Step 2:  get_action → step → update_q_table
│   ├── ...
│   └── Step N:  get_action → step → update_q_table → terminated
│
├── Episode 2  (N iterations)
│   ├── Step 1:  reset → get_action → step → update_q_table
│   ├── ...
│   └── Step N:  terminated
│
└── Episode M
```

Each step involves **real interaction with the Kubernetes cluster** — scaling the deployment, waiting for pods to become ready, and collecting metrics from Prometheus. This is not a simulation.

### Signal Handling

The Trainer handles SIGINT (Ctrl+C) and SIGTERM gracefully:

```python
def _signal_handler(self, signum, frame):
    self._interrupted_save()
    raise KeyboardInterrupt
```

The model saved during interruption goes to `model/{type}/{timestamp}/interrupted/`.

---

## 4. Environment: reset()

Called at the start of each episode:

```python
def reset(self):
    self.iteration = self.initial_iteration
    self.replica_state = self.min_replicas
    self._scale_and_get_metrics()
    self.last_replica = self.min_replicas
    self.episode_reward = 0.0
    self.episode_number += 1
    return self._get_observation()
```

What happens during reset:

1. **Deployment is scaled to minimum replicas** (e.g., 1 pod) — consistent starting point
2. **Pods are waited on** via Prometheus query
3. **Initial metrics are collected** — CPU, memory, response time
4. **Observation is returned** — dict containing all state variables

Every episode starts from the same condition (minimum replicas), giving the agent a "clean slate" to learn from.

---

## 5. Agent: get\_action()

Action selection uses the **epsilon-greedy policy**:

```python
def get_action(self, observation):
    state_key = self.get_state_key(observation)

    if state_key not in self.q_table:
        self.q_table[state_key] = np.zeros(self.n_actions)

    if np.random.rand() < self.epsilon:
        action = np.random.randint(1, self.n_actions + 1)  # explore
    else:
        action = int(np.argmax(self.q_table[state_key])) + 1  # exploit

    return action  # 1-based replica count
```

### Action Interpretation

The action space is integers 1 to `max_replicas`, representing the **number of replicas directly** — no percentage conversion.

| Action (return value) | Q-table Index | Replicas |
| --------------------- | ------------- | -------- |
| 1                     | 0             | 1        |
| 2                     | 1             | 2        |
| 5                     | 4             | 5        |
| 12                    | 11            | 12       |

### Epsilon-Greedy Decay

```text
epsilon_start=0.1  →  decay per step  →  epsilon_min=0.01
```

Epsilon decays after every `update_q_table()` call:

```python
epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

---

## 6. Environment: step()

The core agent-environment interaction:

```python
def step(self, action, q_table_size=0):
    self.last_replica = action

    self.replica_state = max(self.min_replicas, min(action, self.max_replicas))

    self._scale_and_get_metrics()

    reward, reward_breakdown = self._calculate_reward()

    self.iteration -= 1
    terminated = self.iteration <= 0

    observation = self._get_observation()

    self.influxdb.write_point(...)

    return observation, reward, terminated, info
```

### Single Step Timeline

```text
t=0s    Agent selects action (e.g., action=6 → 6 replicas)
        │
t=0s    Environment calls Kubernetes API to scale
        │
t=1-?s  Wait for pods to be ready (wait_for_pods_ready)
        │ - Poll Prometheus every second
        │ - Check: ready_replicas == desired_replicas?
        │
t=Xs    Wait for metrics to stabilize (wait_time=60s)
        │ - Pods are running but metrics not yet stable
        │ - Need time for CPU/memory/RT to reflect actual load
        │
t=X+    Collect metrics from Prometheus
        │ - CPU usage (mean across all pods)
        │ - Memory usage (mean across all pods)
        │ - Response time (P90 quantile)
        │
t=...   Calculate reward
        │
t=...   Return (observation, reward, terminated, info)
```

Each step takes **~1-3 minutes** in wall-clock time because it involves real scaling and live metric collection.

---

## 7. Kubernetes Cluster Interaction

### 7.1 Scaling — `_scale()`

```python
def _scale(self):
    self.cluster.patch_namespaced_deployment_scale(
        name=self.deployment_name,
        body=V1Scale(spec=V1ScaleSpec(replicas=int(self.replica_state))),
        namespace=self.namespace,
    )
```

Uses the **Kubernetes Python Client** to call `PATCH /apis/apps/v1/namespaces/{ns}/deployments/{name}/scale`.

**Retry logic:**

- Exponential backoff: delay starts at 1s, max 30s
- Timeout also increases: 60s → max 300s
- Retry up to `max_scaling_retries` (default: 1000)
- Handles specific errors: etcd timeout (500), conflict (409)

### 7.2 Waiting for Pods — `wait_for_pods_ready()`

After scaling, the environment waits until all pods are **Ready**:

```python
def wait_for_pods_ready(prometheus, deployment_name, desired_replicas, ...):
    while time.time() - start_time < timeout:
        desired = prometheus.custom_query(q_desired)
        ready = prometheus.custom_query(q_ready)

        if ready_replicas == desired_replicas:
            return True, desired, ready

        time.sleep(1)

    return False, desired, ready  # timeout
```

---

## 8. Prometheus Metrics Collection

`get_metrics()` collects 4 main metrics:

### 8.1 CPU Usage

```promql
sum by (pod) (
    rate(container_cpu_usage_seconds_total{
        namespace="default", container!="", container!="POD"
    }[15s])
)
```

- Uses `rate()` with 15-second window
- Computed as **percentage of CPU limit** per pod
- Result: `np.nanmean(cpu_percentages)` — average across all pods

### 8.2 Memory Usage

```promql
sum by (pod) (
    container_memory_working_set_bytes{
        namespace="default", container!="", container!="POD"
    }
)
```

- Uses `working_set_bytes` (more accurate than RSS for Kubernetes)
- Computed as **percentage of memory limit** per pod
- Result: `np.nanmean(memory_percentages)`

### 8.3 Response Time

```promql
1000 * histogram_quantile(
    0.90,
    sum by (le) (
        rate(app_request_latency_seconds_bucket{
            job="ecom-api", namespace="default",
            method="GET", exported_endpoint="/"
        }[15s])
    )
)
```

- Uses **P90 quantile** — 90% of requests complete below this value
- Result in **milliseconds** (multiplied by 1000)
- Health check endpoints (`/metrics`, `/healthz`) excluded

### 8.4 Request Rate

```promql
sum(
    rate(app_requests_total{
        namespace="default", pod=~"ecom-api-.*",
        exported_endpoint!~"/metrics|/healthz"
    }[15s])
)
```

- Total **Requests Per Second (RPS)** to the deployment
- **Note:** Request rate is collected but **not used** in the state key or reward function — only available for logging and post-training analysis.

---

## 9. Observation (State Representation)

`_get_observation()` returns the dict that becomes input to the agent:

```python
{
    # === STATE KEY COMPONENTS (4 components, used for Q-table key) ===
    "cpu_usage": 55.2,       # Mean CPU usage across all pods (0-100%)
    "memory_usage": 62.1,    # Mean memory usage across all pods (0-100%)
    "response_time": 45.0,   # RT as % of max_response_time, capped at 100
    "last_replica": 6,       # Previous action (1 to max_replicas)

    # === AUXILIARY FIELDS (NOT in state key — logging only) ===
    "response_time_ms": 45.0,     # Raw RT in milliseconds
    "current_replicas": 6.0,      # Current pod count
}
```

State representation uses **4 components**. The `response_time` field implicitly captures the effect of request rate — when request rate is high and pods are insufficient, response time rises. By removing redundancy, the state space is smaller and convergence is faster.

### Metric Normalization

| Metric          | Formula                            | Example                        |
| --------------- | ---------------------------------- | ------------------------------ |
| `response_time` | `min((RT_ms / max_RT) * 100, 100)` | `min((45/100)*100, 100) = 45%` |
| `last_replica`  | Direct value (1 to max\_replicas)  | `6` (6 replicas)               |

---

## 10. Agent: update\_q\_table()

After receiving the reward, Q-values are updated using the **Bellman equation**:

```python
def update_q_table(self, observation, action, reward, next_observation):
    state_key = self.get_state_key(observation)
    next_state_key = self.get_state_key(next_observation)

    action_idx = action - 1  # convert 1-based action to 0-based index

    if state_key not in self.q_table:
        self.q_table[state_key] = np.zeros(self.n_actions)
    if next_state_key not in self.q_table:
        self.q_table[next_state_key] = np.zeros(self.n_actions)

    best_next = np.max(self.q_table[next_state_key])
    self.q_table[state_key][action_idx] += learning_rate * (
        reward + discount_factor * best_next - self.q_table[state_key][action_idx]
    )

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

### Q-Learning Formula

```text
Q(s, a) ← Q(s, a) + α * [r + γ * max_a'(Q(s', a')) - Q(s, a)]
                      │    │   │                        │
                      │    │   │                        └── Old estimate
                      │    │   └── Best value in next state
                      │    └── Reward received
                      └── Learning rate
```

### Update Example

```text
State: (CPU=55%, MEM=60%, RT=45%, last_replica=6)
Action: 6 (index=5), Reward: 0.80, Next state best Q: 0.90

Before: Q[(55,60,45,6), index=5] = 0.50

Update: Q += 0.1 * (0.80 + 0.95 * 0.90 - 0.50)
        Q += 0.1 * (0.80 + 0.855 - 0.50)
        Q += 0.1 * 1.155 = 0.1155

After:  Q[(55,60,45,6), index=5] = 0.6155
```

---

## 11. State Key Differences: Q-Learning vs Q-Fuzzy

### Q-Learning: Continuous State Key (4 components)

```python
def get_state_key(self, observation):
    return (
        cpu_usage,      # float: 55.23
        memory_usage,   # float: 62.10
        response_time,  # float: 45.00
        last_replica,   # int:   6
    )
```

**Example state key:** `(55.23, 62.10, 45.00, 6)`

**Problem:** Because CPU, memory, and response time are continuous floats, two states being *identical* is nearly impossible. The Q-table grows very large but each state is rarely revisited. Generalization is difficult — experience from one state does not transfer to similar states.

### Q-Learning Fuzzy: Discrete State Key (4 components)

```python
def get_state_key(self, observation):
    fuzzy_state = self.fuzzy.fuzzify(observation)

    cpu_label = max(fuzzy_state["cpu_usage"], key=...)      # "medium"
    mem_label = max(fuzzy_state["memory_usage"], key=...)   # "high"
    resp_label = max(fuzzy_state["response_time"], key=...) # "low"
    last_label = max(fuzzy_state["last_replica"], key=...)  # "medium"

    return (cpu_label, mem_label, resp_label, last_label)
```

**Example state key:** `("medium", "high", "low", "medium")`

**Advantage:** State space is much smaller:

- 3 fuzzy labels × 4 metrics (cpu, mem, response\_time, last\_replica) = 3^4 = **81 theoretical states**
- In practice, many combinations never occur — very compact
- **Generalization**: CPU=55.2% and CPU=52.8% both map to "medium" — sharing the same Q-value

### Comparison

| Aspect               | Q-Learning (Continuous)     | Q-Fuzzy (Discrete)                    |
| -------------------- | --------------------------- | ------------------------------------- |
| State key            | (55.23, 62.10, 45.00, 6)   | ("medium", "high", "low", "medium")   |
| State components     | 4 (3 float + 1 int)         | 4 (4 fuzzy labels)                    |
| Q-table size         | Potentially unbounded       | Max 81 states (3^4)                   |
| State revisits       | Very rare                   | Frequent                              |
| Generalization       | None                        | Automatic via fuzzification           |
| Convergence speed    | Slower                      | Faster                                |
| Reward function      | **Identical**               | **Identical**                         |

---

## 12. Fuzzy Logic — State Fuzzification

### 12.1 Membership Function

The `Fuzzy` class (`rl/fuzzy.py`) defines **trapezoidal membership functions** for each metric:

```text
Membership degree
1.0 |      _________
    |     /         \
    |    /           \
0.0 |___/             \___
    a   b             c   d  → input value
```

Trapezoidal formula:

```python
def _trapezoidal(x, a, b, c, d):
    if x < a or x > d:  return 0.0   # outside range
    elif b <= x <= c:   return 1.0   # fully member
    elif a < x < b:     return (x - a) / (b - a)  # rising
    else:               return (d - x) / (d - c)  # falling
```

### 12.2 Membership Definitions (3 Levels: low, medium, high)

Each metric uses **3 membership levels**:

```text
1.0  ┬──low──┐         ┌─medium──┐         ┌──high──┬
     │        │\       /│         │\       /│         │
     │        │ \     / │         │ \     / │         │
0.0  └────────┴──\───/──┴─────────┴──\───/──┴─────────┘
     0    20  30  40  45    55   60  70  80       100 (%)
```

| Label  | a  | b  | c   | d   | Fully Member Range |
| ------ | -- | -- | --- | --- | ------------------ |
| low    | 0  | 0  | 20  | 40  | 0–20%              |
| medium | 30 | 45 | 55  | 70  | 45–55%             |
| high   | 60 | 80 | 100 | 100 | 80–100%            |

### 12.3 Fuzzification Examples

Input: `CPU = 55%`

```text
low(55)    = 0.0   (55 > 40)
medium(55) = 1.0   (45 <= 55 <= 55 → fully member)
high(55)   = 0.0   (55 < 60)

Dominant label: "medium" (degree 1.0)
```

Input: `CPU = 35%`

```text
low(35)    = 0.25  ((40-35)/(40-20) = 5/20 = 0.25)
medium(35) = 0.33  ((35-30)/(45-30) = 5/15 = 0.33)
high(35)   = 0.0   (35 < 60)

Dominant label: "medium" (degree 0.33 > 0.25)
```

Input: `CPU = 72%`

```text
low(72)    = 0.0   (72 > 40)
medium(72) = 0.0   (72 > 70)
high(72)   = 0.6   ((72-60)/(80-60) = 12/20 = 0.6)

Dominant label: "high" (degree 0.6)
```

### 12.4 Fuzzified Metrics

All four metrics use **identical** membership definitions (all on a 0–100% scale):

1. `cpu_usage` — CPU utilization
2. `memory_usage` — memory utilization
3. `response_time` — response time as % of SLO
4. `last_replica` — last action (normalized to 0–100% based on max\_replicas)

Total Q-Fuzzy state space: 3 labels × 4 metrics = 3^4 = **81 theoretical combinations**.

---

## 13. Checkpoint and Model Saving

### 13.1 Best Model Checkpoint

Each time an episode achieves the highest total reward, the model is saved:

```python
def _save_checkpoint(self, episode, score, note, start_time):
    path = f"model/{model_type}/{start_time}_{note}/checkpoints/"
           f"episode_{episode}_total_{score}.pkl"
    agent.save_model(path, episode + 1)
```

**Directory structure:**

```text
model/
├── qlearning/
│   └── 1706000000_experiment_1/
│       ├── checkpoints/
│       │   ├── episode_0_total_5.23.pkl
│       │   ├── episode_3_total_8.45.pkl
│       │   └── episode_7_total_12.10.pkl
│       ├── interrupted/
│       │   └── interrupted_episode_5_1706003600.pkl
│       └── final/
│           └── qlearning_1706007200.pkl
└── qlearningfuzzy/
    └── ...
```

### 13.2 Model File Contents (.pkl)

```python
model_data = {
    "q_table": dict,             # {state_key: np.array(n_actions)}
    "learning_rate": 0.1,
    "discount_factor": 0.95,
    "epsilon": 0.05,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.99,
    "n_actions": 12,             # = max_replicas
    "created_at": 1706000000,
    "episodes_trained": 10,
}
```

Serialized using Python **pickle** — stores the entire agent state including the Q-table.

---

## 14. Training Hyperparameters

| Parameter             | Default | Source | Effect                                              |
| --------------------- | ------- | ------ | --------------------------------------------------- |
| learning_rate (α)     | 0.1     | .env   | How quickly Q-values update per step               |
| discount_factor (γ)   | 0.95    | .env   | Weight of future vs immediate rewards              |
| epsilon_start         | 0.1     | .env   | Initial random exploration probability             |
| epsilon_decay         | 0.99    | .env   | Epsilon reduction rate per step                    |
| epsilon_min           | 0.01    | .env   | Minimum epsilon (always 1% exploration)            |
| n_actions             | max_r   | .env   | Number of actions = max replicas                   |
| episodes              | 10      | .env   | Number of episodes per training session            |
| iteration             | 10      | .env   | Number of steps per episode                        |
| wait_time             | 60s     | .env   | Wait time after scaling before collecting metrics  |
| timeout               | 120s    | .env   | Timeout waiting for pods to become ready           |
| metrics_interval      | 15s     | .env   | PromQL rate() window                               |
| metrics_quantile      | 0.90    | .env   | Quantile for response time (P90)                   |

### Time Estimation

```text
episodes × iteration = total steps per session
10 × 10 = 100 steps

Each step ≈ 1-3 minutes (scaling + wait_time + metric collection)
100 steps × ~2 min = ~200 minutes (~3.3 hours) per training session
```

### Epsilon Decay Behavior

```text
epsilon_decay^(total_steps) = final epsilon
0.99^100  = 0.366  → still significant exploration after 100 steps
0.99^1000 = 0.00004 → near-full exploitation after 1000 steps
```

---

## 15. End-to-End Training Diagram

```text
┌──────────────────────────────────────────────────────────────┐
│                          train.py                             │
│  1. Setup Logger                                              │
│  2. Connect InfluxDB                                          │
│  3. Initialize Environment (KubernetesEnv)                    │
│  4. Initialize Agent (QLearning / QLearningFuzzy)             │
│  5. Initialize Trainer                                        │
│  6. trainer.train(episodes=N)                                 │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    EPISODE LOOP (1..N)                        │
│                                                              │
│  env.reset()                                                  │
│    ├── replica_state = min_replicas                           │
│    ├── K8s API: scale deployment to min                       │
│    ├── Prometheus: wait_for_pods_ready()                      │
│    ├── sleep(wait_time)                                       │
│    ├── Prometheus: get_metrics(CPU, MEM, RT, RPS)             │
│    ├── Reset: rewards, counters                               │
│    └── Return: observation (dict)                             │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │             ITERATION LOOP (1..M)                    │    │
│  │                                                      │    │
│  │  STEP 1: agent.get_action(observation)               │    │
│  │    ├── state_key = get_state_key(obs) (4 components) │    │
│  │    │   Q-Learning: (55.2, 62.1, 45.0, 6)            │    │
│  │    │   Q-Fuzzy:    ("medium","high","low","med")     │    │
│  │    ├── if new state: Q[state] = zeros(n_actions)     │    │
│  │    ├── epsilon-greedy:                               │    │
│  │    │   random < epsilon → random action (1..max_r)  │    │
│  │    │   else → argmax(Q[state]) + 1                  │    │
│  │    └── Return: action (1-based replica count)        │    │
│  │                         │                           │    │
│  │  STEP 2: env.step(action)                            │    │
│  │    ├── replica_state = action (no conversion)        │    │
│  │    ├── K8s API: patch_deployment_scale(replicas)     │    │
│  │    ├── Prometheus: poll ready_replicas == desired    │    │
│  │    ├── sleep(wait_time)                              │    │
│  │    ├── CPU: rate(cpu_usage) / limits * 100           │    │
│  │    ├── MEM: working_set_bytes / limits * 100         │    │
│  │    ├── RT:  histogram_quantile(0.90, latency)        │    │
│  │    ├── reward = R_rt - R_cost                        │    │
│  │    ├── Write to InfluxDB                             │    │
│  │    └── Return: (next_obs, reward, terminated, info)  │    │
│  │                         │                           │    │
│  │  STEP 3: agent.update_q_table(obs, act, rew, nxt)   │    │
│  │    ├── Q[s,a] += α * (r + γ * max_Q(s') - Q[s,a])  │    │
│  │    └── epsilon *= epsilon_decay                      │    │
│  │                         │                           │    │
│  │  terminated? → No: loop | Yes: break                │    │
│  └──────────────────────────────────────────────────────┘    │
│                         │                                    │
│  total_reward > best?                                         │
│    Yes → _save_checkpoint(episode, total_reward)              │
│    No  → next episode                                         │
└──────────────────────────────────────────────────────────────┘
                         │
                         ▼
  Save final model to model/{type}/{timestamp}/final/
```

---

## PART B: PREDICTION FLOW

## 16. Prediction Initialization

Prediction starts from `predict.py`. The key difference from training:

```python
# 1-4: Same as training (logger, influxdb, environment, agent)

# 5. Load trained model
model_path = os.getenv("MODEL_PATH", "")
agent.load_model(model_path)

# 6. CRITICAL: Set epsilon = 0 (no exploration)
agent.epsilon = 0

# 7. Reset environment
obs = env.reset()
```

### Model Loading

```python
def load_model(self, filepath):
    with open(filepath, "rb") as f:
        model_data = pickle.load(f)

    self.q_table = model_data["q_table"]
    self.learning_rate = model_data["learning_rate"]
    self.discount_factor = model_data["discount_factor"]
    self.epsilon = model_data["epsilon"]    # will be overridden to 0
    self.n_actions = model_data["n_actions"]
    self.episodes_trained = model_data["episodes_trained"]
```

---

## 17. Prediction Loop

The prediction loop runs **indefinitely** — continuously controlling the cluster:

```python
while True:
    act = agent.get_action(obs)
    nxt, rew, term, info = env.step(act, q_table_size=len(agent.q_table))
    obs = nxt
    log_verbose_details(obs, agent, verbose=True, logger=logger)
```

### Critical Differences from Training

1. **epsilon = 0** → No random exploration. Every action is `argmax(Q[state])` — always the best known action.
2. **No `update_q_table()`** → Q-table is read-only. The agent does not learn.
3. **Infinite loop** → No episode concept or termination. The agent runs continuously.
4. **No checkpoint saving** → Model is not saved again.

### Handling Unseen States

When prediction encounters a state **not in the Q-table**:

```python
if state_key not in self.q_table:
    self.q_table[state_key] = np.zeros(self.n_actions)

# epsilon = 0, so always:
action = int(np.argmax(self.q_table[state_key])) + 1
# argmax([0, 0, ..., 0]) = 0 → +1 = 1 (minimum replicas)
```

For an unseen state, the agent defaults to **action 1** (minimum replicas). This is conservative behavior — better to start at minimum than to over-provision on unknown states.

For Q-Fuzzy, the state space is limited (81 combinations = 3^4), so unseen states during prediction are far less likely than in Q-Learning with continuous states.

---

## 18. Training vs Prediction Differences

| Aspect              | Training                          | Prediction                      |
| ------------------- | --------------------------------- | ------------------------------- |
| **File**            | `train.py`                        | `predict.py`                    |
| **Epsilon**         | 0.1 → decay → 0.01                | **0** (fixed)                   |
| **Exploration**     | Yes (epsilon-greedy)              | **None** (pure exploitation)    |
| **Q-table update**  | Every step (`update_q_table()`)   | **Never**                       |
| **Loop**            | Episode-based (bounded)           | **Infinite**                    |
| **Termination**     | After N episodes                  | Manual only (Ctrl+C)            |
| **Checkpoint**      | Saves best model                  | **None**                        |
| **InfluxDB**        | Writes all metrics                | Writes all metrics              |
| **Model**           | Built from scratch or resumed     | **Loaded from file**            |
| **Goal**            | Learn optimal policy              | **Apply learned policy**        |
| **reset()**         | Every episode (scale to min)      | Once at startup only            |

---

## 19. End-to-End Prediction Diagram

```text
┌──────────────────────────────────────────────────────────────┐
│                          predict.py                           │
│  1. Setup Logger                                              │
│  2. Connect InfluxDB                                          │
│  3. Initialize Environment (KubernetesEnv)                    │
│  4. Initialize Agent (QLearning / QLearningFuzzy)             │
│  5. agent.load_model(MODEL_PATH)  ← Load trained Q-table     │
│  6. agent.epsilon = 0             ← CRITICAL: no exploration  │
│  7. obs = env.reset()             ← Scale to min, init obs   │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                     INFINITE LOOP                             │
│                                                              │
│  agent.get_action(obs)                                        │
│    ├── state_key = get_state_key(obs) (4 components)          │
│    ├── if new state: Q[state] = zeros → action = 1            │
│    └── action = argmax(Q[state_key]) + 1 ← always best       │
│                         │                                    │
│  env.step(action)                                             │
│    ├── K8s API: scale deployment                              │
│    ├── Prometheus: wait_for_pods_ready()                      │
│    ├── sleep(wait_time)                                       │
│    ├── Prometheus: get_metrics(CPU, MEM, RT, RPS)             │
│    ├── _calculate_reward()  ← computed but not used to learn  │
│    ├── Write to InfluxDB                                      │
│    └── Return: (next_obs, reward, terminated, info)           │
│                         │                                    │
│  obs = next_obs                                               │
│  log_verbose_details()  ← monitoring output                   │
│                                                              │
│  NOT performed:                                               │
│    ✗ update_q_table()                                         │
│    ✗ epsilon decay                                            │
│    ✗ checkpoint saving                                        │
│                         │                                    │
│                    loop forever                               │
└──────────────────────────────────────────────────────────────┘
```

---

## PART C: THESIS DEFENSE Q&A

### Q1: "Describe the training flow briefly."

> Training consists of an episode loop. Each episode starts with **reset** (scale to minimum replicas), then the agent selects an action using **epsilon-greedy**, the environment executes the action on the **real Kubernetes cluster** (not a simulation), collects metrics from **Prometheus**, calculates the reward, and the agent updates the **Q-table** using the Bellman equation. This repeats until the iteration count is exhausted. The best model (highest total reward per episode) is saved as a checkpoint.

### Q2: "Why is the action space directly the replica count (1..max\_replicas)?"

> The action space is designed as the **direct replica count** (1 to max\_replicas). This is more intuitive and efficient: action=6 means deploy 6 pods with no conversion needed. With `n_actions = max_replicas` (e.g., 12), the Q-table per state only stores a small array (12 elements), and each action is visited more frequently, accelerating convergence. Compared to a percentage-based approach (100 actions), many percentage values would map to the same replica count — spreading Q-values across redundant actions and slowing learning.

### Q3: "Why use a real cluster rather than a simulation?"

> Autoscaling depends heavily on real-world dynamics that are difficult to simulate accurately: network latency, pod cold-start time, resource contention between pods, JIT/cache warm-up effects, and load balancer behavior. A simulation would require very accurate modeling of all these factors, and the result may not transfer to a real cluster (sim-to-real gap). By learning directly on the cluster, the agent captures **end-to-end dynamics** as they actually occur.

### Q4: "What is the main advantage of Q-Learning Fuzzy over plain Q-Learning?"

> The main advantage is **state space reduction and automatic generalization**. In continuous Q-Learning, `CPU=55.2%` and `CPU=55.3%` are different states — the Q-table grows large but each state is rarely revisited. Q-Fuzzy maps both values to "medium", sharing Q-values. With 3 labels × 4 metrics, the theoretical state space is only 81 combinations (3^4), so experience is reused efficiently across similar situations, leading to faster convergence.

### Q5: "How does the agent handle states not seen during training (at prediction time)?"

> When prediction encounters a new state (not in the Q-table), Q-values are initialized to **zero for all actions**. Since `argmax([0, ..., 0])` returns index 0, this maps to action 1 (+1 from 0-based) — minimum replicas. This is conservative — defaulting to minimum rather than maximum avoids severe over-provisioning. For Q-Fuzzy, this risk is much smaller because the bounded state space (81 states) means most states were likely visited during training.

### Q6: "Why is wait\_time = 60 seconds after scaling?"

> **Wait time** is needed because of the **propagation delay** between scaling and stable metrics. When a new pod is created, it takes time to: (1) pull the container image, (2) start the application, (3) warm up JIT/caches, (4) begin receiving traffic from the load balancer, and (5) have enough data accumulated in Prometheus. Collecting metrics too soon captures a transient startup spike rather than the true steady-state. 60 seconds provides sufficient buffer for stabilization.

### Q7: "Why use P90 for response time rather than the mean?"

> The mean is sensitive to outliers and can hide problems. If 99 requests complete in 10ms but 1 takes 10,000ms, the mean = 109ms — misleading since most users had a good experience. **P90 (90th percentile)** means "90% of requests complete below this value" — more representative of the typical user experience. It is also the industry standard for SLA monitoring (used by Google, Amazon, etc. to measure service performance).

### Q8: "What are the limitations of this approach?"

> The main limitations are:
>
> 1. **Low sample efficiency**: Each step takes ~2 minutes of real interaction, so 1000 steps ≈ 33 hours. Deep RL with replay buffers could be more efficient, but tabular Q-Learning requires many state visits.
> 2. **Large state space for continuous Q-Learning**: Continuous state keys mean the Q-table grows fast but states are rarely revisited. Q-Fuzzy addresses this with its bounded state space (81 states, 3^4).
> 3. **No transfer learning**: If the cluster configuration or workload type changes, the model must be retrained.
> 4. **Default action for new states**: Prediction defaults to 1 replica (minimum) for unseen states, which may cause temporary under-provisioning.
> 5. **Metric dependency**: If Prometheus has delays or data gaps, the agent learns from incorrect signals.

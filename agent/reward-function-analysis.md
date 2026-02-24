# Reward Function Analysis: Kubernetes Autoscaling with Reinforcement Learning

## Table of Contents

- [1. Overview](#1-overview)
- [2. Input Sanitization](#2-input-sanitization)
- [3. Component 1: SLO Compliance Reward (R_rt)](#3-component-1-slo-compliance-reward-r_rt)
- [4. Component 2: Replica Cost Penalty (R_cost)](#4-component-2-replica-cost-penalty-r_cost)
- [5. Total Reward](#5-total-reward)
- [6. Reward Properties](#6-reward-properties)
- [7. End-to-End Calculation Flow](#7-end-to-end-calculation-flow)
- [8. Numerical Examples](#8-numerical-examples)
- [9. Recommended Thesis Defense Answers](#9-recommended-thesis-defense-answers)

---

## 1. Overview

The reward function is defined in `_calculate_reward_qlearning()` in `environment.py`. This single function is used by **both algorithms** (Q-Learning and Q-Learning Fuzzy) through the `_calculate_reward()` wrapper. The difference between both algorithms is **not** the reward — it is the state representation:

- **Q-Learning:** Stores Q-values for 4-component continuous states: `(cpu_usage, memory_usage, response_time, last_replica)`
- **Q-Fuzzy:** Stores Q-values for fuzzified states: `(cpu_label, mem_label, resp_label, last_replica_label)` — 3 membership levels each

Using an **identical** reward function makes the comparison fair — the only variable is state abstraction.

### Overall Formula

```text
R_total = R_rt - R_cost
```

| Component | Role                  | Range       |
| --------- | --------------------- | ----------- |
| `R_rt`    | SLO compliance reward | (-∞, +1.0]  |
| `R_cost`  | Replica cost penalty  | [0.0, +1.0] |

**Design principles:**

- `R_rt` is positive when response time is below the SLO, negative when above — providing a continuous gradient.
- `R_cost` is 0.0 at minimum replicas (cheapest) and 1.0 at maximum replicas (most expensive), subtracted from `R_rt` to penalize over-provisioning.
- No weights needed: both components are naturally normalized to comparable ranges.
- No edge-case overrides: the formula handles boundaries naturally (`R_cost = 0` at `R_min`, so only `R_rt` drives reward).

**References:**

- SLO preservation ratio inspired by Qiu et al. (USENIX OSDI, 2020)
- Inverse replica normalization based on Zhong (UvA, 2023)
- Scalarized multi-objective RL normalization per Rossi et al. (IEEE CLOUD, 2019)

---

## 2. Input Sanitization

Before computing the reward, `response_time` is sanitized:

```python
if response_time is None or math.isnan(response_time) or math.isinf(response_time):
    response_time = 0.0
```

| Input Value  | After Sanitization | Reason                                     |
| ------------ | ------------------ | ------------------------------------------ |
| `None`       | `0.0`              | No data from Prometheus                    |
| `NaN`        | `0.0`              | Malformed metric value                     |
| `Inf`        | `0.0`              | Overflow from histogram quantile edge case |
| Normal float | Unchanged          | Valid measurement                          |

Sanitizing to `0.0` treats the absence of data as best-case response time. The agent will not be unfairly penalized when metrics are temporarily unavailable.

---

## 3. Component 1: SLO Compliance Reward (R_rt)

```python
r_rt: float = 1.0 - (self.response_time / self.max_response_time)
```

Measures SLO compliance. The value 1 is the maximum reward, normalized so that `R_rt` operates in the same scale as `R_cost`.

| Response Time    | R_rt Value | Interpretation                        |
| ---------------- | ---------- | ------------------------------------- |
| 0 ms             | +1.0       | Best possible (theoretically instant) |
| 50% of `rt_max`  | +0.5       | Fast, well within SLO                 |
| `rt_max`         | 0.0        | Exactly at SLO threshold              |
| 150% of `rt_max` | -0.5       | SLO violated                          |
| 200% of `rt_max` | -1.0       | Heavy SLO violation                   |

**Continuous gradient:** The formula always gives the agent an incentive to minimize response time — there are no zero-gradient dead zones (Qiu et al., 2020). Any improvement in response time, however small, increases the reward.

**No cap on violation:** There is no cap on `R_rt`. If response time is 10× the SLO, `R_rt = -9.0`. This ensures the agent receives a proportionally strong negative signal for extreme violations.

---

## 4. Component 2: Replica Cost Penalty (R_cost)

```python
r_cost: float = (self.replica_state - self.min_replicas) / self.range_replicas
```

Where `range_replicas = max_replicas - min_replicas`.

Measures operational cost based on replica count (Zhong, 2023).

| Replicas  | R_cost Value | Interpretation                       |
| --------- | ------------ | ------------------------------------ |
| `R_min`   | 0.0          | Most efficient, no cost penalty      |
| Mid-range | 0.5          | Moderate cost                        |
| `R_max`   | 1.0          | Most expensive, maximum cost penalty |

**At `R_min`, cost is zero** — the agent is not penalized for something it cannot reduce further. This provides natural boundary handling without explicit edge-case overrides.

**Subtracted from `R_rt`:** The agent must balance keeping response time low (increasing `R_rt`) against using fewer replicas (decreasing `R_cost`). This is the core trade-off the agent learns to optimize.

---

## 5. Total Reward

```python
reward: float = r_rt - r_cost
```

No weights needed — both components are normalized to comparable ranges:

- `R_rt ∈ (-∞, +1.0]` — typically `[-1.0, +1.0]` under normal conditions
- `R_cost ∈ [0.0, +1.0]`

**Scenario summary:**

| Scenario                                 | R_rt | R_cost | Reward   |
| ---------------------------------------- | ---- | ------ | -------- |
| Low RT + min replicas (ideal)            | +1.0 | 0.0    | **+1.0** |
| Low RT + max replicas (over-provisioned) | +1.0 | 1.0    | **0.0**  |
| At SLO + min replicas                    | 0.0  | 0.0    | **0.0**  |
| SLO violated + min replicas              | -1.0 | 0.0    | **-1.0** |
| SLO violated + max replicas (worst)      | -1.0 | 1.0    | **-2.0** |

**Best case:** Low RT + minimum replicas → `R_rt ≈ +1.0`, `R_cost = 0.0` → `Reward ≈ +1.0`

**Worst case:** High RT + maximum replicas → `R_rt < 0.0`, `R_cost = 1.0` → `Reward << 0.0`

---

## 6. Reward Properties

### 6.1 Reward Range

**Upper bound (theoretical):** Response time = 0ms at minimum replicas.

```text
R_rt = 1.0,  R_cost = 0.0  →  Reward = +1.0
```

**Lower bound:** Unbounded negative (proportional to SLO violation severity).

```text
Example: RT = 5× SLO at max replicas → R_rt = -4.0, R_cost = 1.0 → Reward = -5.0
```

**Practical range:** Under normal operating conditions, reward falls within **[-2.0, +1.0]**.

### 6.2 No Clamping

The reward is **not** clamped to any fixed range. Raw values are preserved because:

1. **Negative rewards** are important signals — the agent needs to know a state is bad.
2. **Reward magnitude** carries information: the difference between `-0.5` and `-2.0` is meaningful for learning.
3. **Clamping destroys information** — the agent loses the ability to distinguish between severity levels.

In tabular Q-Learning, Q-values remain stable as long as the learning rate is sufficiently small.

### 6.3 Design Characteristics

| Property                 | Description                                                      |
| ------------------------ | ---------------------------------------------------------------- |
| **Two-component design** | SLO compliance and operational cost, naturally normalized        |
| **Continuous gradient**  | Not binary (good/bad) — always incentivizes improvement          |
| **Bounded positive**     | Maximum reward = +1.0 (at zero RT, min replicas)                 |
| **Unbounded negative**   | Proportional to violation severity — no artificial caps          |
| **No external weights**  | Only uses response time and replica count                        |
| **Boundary-aware**       | `R_cost = 0` at `R_min` — no penalty for the irreducible minimum |

### 6.4 Return Breakdown Dictionary

`_calculate_reward_qlearning()` returns `(reward, breakdown_dict)`:

| Key                        | Type  | Description                         |
| -------------------------- | ----- | ----------------------------------- |
| `reward`                   | float | Final reward (`R_rt - R_cost`)      |
| `r_rt`                     | float | SLO compliance component            |
| `r_cost`                   | float | Replica cost component              |
| `response_time_ms`         | float | Raw response time in milliseconds   |
| `response_time_percentage` | float | Response time as % of SLO threshold |
| `replica_count`            | int   | Current number of replicas          |

---

## 7. End-to-End Calculation Flow

```text
+-----------------------------------------------------+
|  INPUT:                                              |
|    response_time (ms), replica_state                 |
|    min_replicas, max_replicas, max_response_time     |
+---------------------------+-------------------------+
                            |
                            v
+-----------------------------------------------------+
|  STEP 1: Sanitize response_time                     |
|  - None / NaN / Inf  →  0.0                         |
+---------------------------+-------------------------+
                            |
                            v
+-----------------------------------------------------+
|  STEP 2: Compute R_rt (SLO compliance)              |
|  R_rt = 1.0 - (response_time / max_response_time)  |
|  - RT = 0ms      → R_rt = +1.0                     |
|  - RT = rt_max   → R_rt =  0.0                     |
|  - RT > rt_max   → R_rt <  0.0 (unbounded)         |
+---------------------------+-------------------------+
                            |
                            v
+-----------------------------------------------------+
|  STEP 3: Compute R_cost (replica cost)              |
|  R_cost = (replicas - min) / (max - min)            |
|  - replicas = min → R_cost = 0.0                   |
|  - replicas = max → R_cost = 1.0                   |
+---------------------------+-------------------------+
                            |
                            v
+-----------------------------------------------------+
|  STEP 4: Compute total reward                       |
|  reward = R_rt - R_cost                             |
+---------------------------+-------------------------+
                            |
                            v
+-----------------------------------------------------+
|  OUTPUT: reward (float, unclamped)                  |
|          + breakdown dictionary (for logging)       |
+-----------------------------------------------------+
```

---

## 8. Numerical Examples

### Example 1: Ideal State

Low response time, minimum replicas.

```text
Input:
  response_time = 20ms,  max_response_time = 100ms
  replicas = 1,  min = 1,  max = 12

R_rt   = 1.0 - (20 / 100) = +0.80
R_cost = (1 - 1) / (12 - 1) = 0.00

reward = 0.80 - 0.00 = +0.80
```

The agent is rewarded for fast response time with minimal resource usage.

---

### Example 2: Over-provisioned State

Low response time, but using too many replicas.

```text
Input:
  response_time = 20ms,  max_response_time = 100ms
  replicas = 12,  min = 1,  max = 12

R_rt   = 1.0 - (20 / 100) = +0.80
R_cost = (12 - 1) / (12 - 1) = 1.00

reward = 0.80 - 1.00 = -0.20
```

Even though response time is good, the heavy replica cost pulls the reward negative, signaling to the agent to scale down.

---

### Example 3: SLO Violated State

Response time above threshold.

```text
Input:
  response_time = 180ms,  max_response_time = 100ms
  replicas = 3,  min = 1,  max = 12

R_rt   = 1.0 - (180 / 100) = -0.80
R_cost = (3 - 1) / (12 - 1) = 0.18

reward = -0.80 - 0.18 = -0.98
```

The SLO violation dominates the reward. The agent must scale up to reduce response time.

---

### Example 4: At SLO Boundary

Response time exactly at the SLO threshold.

```text
Input:
  response_time = 100ms,  max_response_time = 100ms
  replicas = 5,  min = 1,  max = 12

R_rt   = 1.0 - (100 / 100) = 0.00
R_cost = (5 - 1) / (12 - 1) = 0.36

reward = 0.00 - 0.36 = -0.36
```

At the SLO boundary `R_rt = 0`, so the cost penalty makes the total reward negative. Staying exactly at the SLO boundary with mid-range replicas is suboptimal — the agent is incentivized to either improve response time or reduce replicas.

---

## 9. Recommended Thesis Defense Answers

### Q1: "Why does the reward function use only response time and replica count, not CPU or memory?"

> The reward function focuses on the two most directly meaningful signals for the autoscaling objective:
>
> 1. **Response time** is the primary user-facing SLO metric. It naturally captures the effect of CPU and memory pressure — if resources are insufficient, response time increases. Using response time as a proxy avoids the need to tune separate thresholds for CPU and memory.
> 2. **Replica count** captures operational cost. More replicas mean more resource consumption, so penalizing high replica counts incentivizes the agent to find the minimum sufficient capacity.
>
> CPU and memory utilization are still observable in the state (the agent uses them to make decisions), but they do not appear in the reward. This simplifies the reward signal and ensures the agent optimizes for the actual objective — fast responses at low cost.

### Q2: "Why use a linear formula instead of a more complex reward design?"

> The linear design `R_total = R_rt - R_cost` was chosen for several reasons:
>
> 1. **Interpretability:** Each component has a clear, intuitive meaning.
> 2. **No dead zones:** The continuous linear gradient always gives the agent an incentive to improve.
> 3. **Natural normalization:** Both components are bounded within comparable ranges without needing explicit weights.
> 4. **Stable learning:** Simpler reward functions generally produce more stable Q-value updates in tabular Q-Learning.

### Q3: "What happens when response time is very large (e.g., system outage)?"

> The reward has no cap on negative values. If `response_time = 1000ms` with `max_response_time = 100ms`:
>
> `R_rt = 1 - (1000/100) = -9.0`
>
> This produces a very large negative reward, rapidly decreasing the Q-value for the action that led to this state. Tabular Q-Learning with a small learning rate (e.g., 0.1) handles this gracefully — the Q-update is always dampened by the learning rate factor.

### Q4: "Why is the cost penalty subtracted rather than weighted?"

> Subtraction works because both components are already normalized to comparable scales:
>
> - `R_rt ∈ (-∞, +1.0]` — typically `[-1.0, +1.0]` under normal conditions
> - `R_cost ∈ [0.0, +1.0]`
>
> A weighted sum `w1 * R_rt + w2 * R_cost` would require tuning `w1` and `w2`, introducing extra hyperparameters. With subtraction, the implicit weight is 1:1, which directly encodes the trade-off: at maximum replicas, even perfect response time only gives zero reward, forcing the agent to find a more efficient operating point.

### Q5: "Is not clamping the reward dangerous for Q-Learning?"

> Not in practice for tabular Q-Learning. Q-value updates use the formula:
>
> `Q(s,a) += lr * (reward + gamma * max_Q(s') - Q(s,a))`
>
> As long as `lr` (learning rate) is small (e.g., 0.1), even large reward values produce bounded Q-updates per step. The key stabilizing factors are: (1) the learning rate dampening effect, (2) the discount factor `gamma < 1.0`, and (3) the fact that extreme negative rewards only occur in genuinely bad states — the agent quickly learns to avoid them.

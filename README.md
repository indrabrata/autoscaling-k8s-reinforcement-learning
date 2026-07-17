# Autoscaling Kubernetes with Reinforcement Learning (Q-Learning)

For my thesis, wish me luck, hopefully not cooked.

## What this research is about

Kubernetes' default Horizontal Pod Autoscaler scales on fixed resource thresholds, which reacts to load
but never learns from the outcome of its own decisions. This research replaces that rule with a
reinforcement learning agent that sets the replica count directly, learning from live cluster feedback
which replica count keeps latency under the SLO without over-provisioning.

The core question is whether the **state representation** — how a raw cluster observation becomes a Q-table
key — is what decides if tabular Q-learning is practical for autoscaling. Every arm shares the same reward,
action space, environment and hyperparameters, so any difference in results is attributable to the state
abstraction alone.

### Shared formulation

- **State**: CPU utilization, memory utilization, P90 response time (as % of the SLO), and the previous
  replica count — each on a 0–100 scale. Metrics come from Prometheus.
- **Action**: the desired replica count, chosen directly from `1..MAX_REPLICAS` (not an increment), applied
  by patching the deployment's scale subresource.
- **Reward**: `R = R_rt − R_cost`, where `R_rt = 1 − (rt / rt_max)` rewards SLO compliance and
  `R_cost = (R − R_min) / (R_max − R_min)` penalizes over-provisioning. Both terms are normalized to
  comparable ranges, so no hand-tuned weights are needed. The reward is positive when response time is under
  the SLO and goes negative when it is violated, giving a continuous gradient with no dead zones.

Training runs against a real cluster, not a simulator: the agent scales an actual deployment, waits for pods
to become ready, then reads the resulting metrics back from Prometheus. Load is generated with k6 against a
CPU- and memory-intensive service.

## Two studies

The work was done in two passes. [agent/](agent/) is the first study; [agent-dipa/](agent-dipa/) is the
second, and supersedes it.

### First study — [agent/](agent/)

Two arms, testing whether discretizing the state space helps at all:

| `ALGORITHM` | State key | Q-table size |
| ----------- | --------- | ------------ |
| `Q-LEARNING` | raw continuous values | unbounded |
| `Q-LEARNING-FUZZY` | trapezoidal membership, collapsed to the single strongest label via `max()` | ≤ 3⁴ = 81 |

The conventional agent treats every distinct reading as its own state, so the table grows without bound and
almost no state is revisited often enough to converge. The fuzzy agent maps similar cluster conditions onto
the same `low` / `medium` / `high` state, so experience accumulates.

### Second study — [agent-dipa/](agent-dipa/)

Three arms. Two things changed, and both address weaknesses in the first study:

| `ALGORITHM` | State key | States per observation |
| ----------- | --------- | ---------------------- |
| `Q-LEARNING` | raw continuous values | 1 (unbounded table) |
| `Q-LEARNING-CRISP` | one hard band per metric | 1 |
| `Q-LEARNING-FUZZY` | every band it belongs to, weighted | 1–16 |

**A crisp arm was added to isolate the variable.** The first study's comparison confounds two things:
discretization and fuzziness. `Q-LEARNING-CRISP` keeps the discretization but drops the fuzziness — each
metric lands in exactly one band, with cut-offs *derived from the trapezoids* (the midpoints of the overlap
zones, 35 and 65) rather than hand-picked, so retuning a membership function moves both arms together. Any
remaining gap between crisp and fuzzy is then fuzziness alone.

**The fuzzy agent became actual Fuzzy Q-Learning.** Collapsing to the strongest label via `max()`, as the
first study did, throws the membership degrees away — the result is just a discretization with extra steps,
and it still switches states abruptly at a boundary. In the second study a metric in an overlap zone stays a
partial member of *both* bands, so one observation activates up to 2⁴ = 16 states at once, each carrying a
normalized firing strength `w_i` (product t-norm over the membership degrees):

```text
action:  a* = argmax_a  Σ_i w_i · Q(s_i, a)
update:  ΔQ(s_i, a) = lr · w_i · [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

A state the observation barely belongs to is barely updated. This is what removes the all-or-nothing jump
the crisp agent has at a boundary — the policy varies continuously across it instead of switching outright.
Following Glorennec & Jouffe, *Fuzzy Q-Learning* (IEEE FUZZ, 1997).

## Repository layout

| Path | Contents |
| ---- | -------- |
| [agent/](agent/) | First study — two agents, [environment](agent/environment/environment.py) (cluster interaction, metrics, reward), [rl/](agent/rl/), training and evaluation entry points, k6 load scripts |
| [agent-dipa/](agent-dipa/) | Second study — three agents behind a [factory](agent-dipa/rl/), shared `QLearningBase`, and an [architecture.md](agent-dipa/architecture.md) covering the internals in depth |
| [inference-app/](inference-app/) | First study's workload (Python/FastAPI, resnet inference), plus the Helm/kubectl setup for Prometheus, ingress-nginx and metrics-server |
| [inference-app-dipa/](inference-app-dipa/) | Second study's workload — a Go service exposing `app_requests_total` and `app_request_latency_seconds`, with one deployment + ServiceMonitor per arm and an HPA manifest for the baseline |
| [analysis/](analysis/) | Notebooks and run data — training curves, testing results, statistical comparison, Q-table inspection |
| [analysis-dipa/](analysis-dipa/) | Second study's analysis (empty so far) |

Each arm trains against its own deployment (`resource-intensive-q`, `-qcrisp`, `-qfuzzy`) so the three can run
concurrently without interfering, with one k6 script per arm using identical load profiles differing only in
route. Setup and run instructions live in each directory's own README.

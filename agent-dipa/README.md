# agent-dipa

Kubernetes autoscaling agent comparing **three state representations** of tabular
Q-learning. All three share the same reward function, action space, environment
and hyperparameters — they differ *only* in how an observation becomes a Q-table
key, so any difference in results is attributable to the state abstraction alone.

Unlike the previous study, this agent does not write to InfluxDB. Metrics go to
CSV (`metrics_output/`) via [`collect_metrics.py`](collect_metrics.py).

## The three models

| `ALGORITHM` | State key | Q-table size | Deployment |
|---|---|---|---|
| `Q-LEARNING` | raw continuous values | unbounded | `resource-intensive-q` |
| `Q-LEARNING-CRISP` | one hard band per metric | ≤ 3⁴ = 81 | `resource-intensive-qcrisp` |
| `Q-LEARNING-FUZZY` | every band it belongs to, weighted | ≤ 3⁴ = 81 | `resource-intensive-qfuzzy` |

Observation: `cpu_usage`, `memory_usage`, `response_time` (as % of SLO) and
`last_replica`, each on a 0–100% scale. Action: replica count `1..MAX_REPLICAS`.

### 1. `Q-LEARNING` — conventional (control)

The observation is the key, verbatim. Every distinct reading is its own state,
so the table grows without bound and nothing is shared between two states that
differ marginally. This is the baseline the other two aim to improve on.

### 2. `Q-LEARNING-CRISP` — fuzzy boundaries, no fuzziness

Keeps the fuzzy study's discretization but assigns each metric to exactly one
band. The cut-offs are **derived from the trapezoids**, not hand-picked: they are
the midpoints of the zones where adjacent fuzzy sets overlap.

```
low    = (0, 0, 20, 40)      low/medium overlap [30, 40] -> 35.0
medium = (30, 45, 55, 70)    medium/high overlap [60, 70] -> 65.0
high   = (60, 80, 100, 100)

-> low: x < 35 | medium: 35 <= x < 65 | high: x >= 65
```

See `CRISP_BOUNDARIES` in [`rl/fuzzy.py`](rl/fuzzy.py). Changing a trapezoid moves
the crisp and fuzzy agents together, by construction.

The point of this arm is to isolate *fuzziness* from *discretization*: CPU at
34.9 and 35.1 land in different states and share nothing, which is exactly the
boundary brittleness the fuzzy agent is meant to smooth over.

### 3. `Q-LEARNING-FUZZY` — multi-membership (FQL)

**This is what differs from the previous study.** Previously each metric was
collapsed to its single strongest label via `max()`, giving one state per
observation. Here a metric in an overlap zone stays a partial member of *both*
bands, and those labels combine with every label of the other metrics — so one
observation activates up to **2⁴ = 16 states at once**.

Each active state carries a normalized firing strength `w_i` (product t-norm over
the membership degrees):

```
action:  a* = argmax_a  Σ_i w_i · Q(s_i, a)
update:  ΔQ(s_i, a) = lr · w_i · [ r + γ · max_a' Q(s', a') − Q(s, a) ]
         where     Q(s, a) = Σ_i w_i · Q(s_i, a)
```

A state the observation barely belongs to is barely updated; an observation on a
boundary contributes to both neighbours in proportion to its membership. This is
what removes the all-or-nothing jump the crisp agent has at 35 and 65 — the
policy varies continuously across a boundary instead of switching outright.

Worked example (`cpu=35, mem=62, rt=50, replica=1`):

```
cpu_usage       {'low': 0.25, 'medium': 0.3333}
memory_usage    {'medium': 0.5333, 'high': 0.1}
response_time   {'medium': 1.0}
last_replica    {'low': 1.0}

-> 4 active states (weights sum to 1.0):
   w=0.4812  ('medium', 'medium', 'medium', 'low')
   w=0.3609  ('low',    'medium', 'medium', 'low')
   w=0.0902  ('medium', 'high',   'medium', 'low')
   w=0.0677  ('low',    'high',   'medium', 'low')
```

Reference: Glorennec & Jouffe, *Fuzzy Q-Learning* (IEEE FUZZ, 1997).

## Layout

```
rl/
  fuzzy.py            membership functions, firing strengths, CRISP_BOUNDARIES
  crisp.py            hard bands cut at the fuzzy boundaries
  base.py             shared loop, persistence, reporting
  q_learning.py       conventional
  q_learning_crisp.py crisp
  q_learning_fuzzy.py multi-membership FQL
  factory.py          create_agent(ALGORITHM, ...)
environment/          KubernetesEnv: scaling, metrics, reward
utils/                Prometheus queries, readiness, logging
train.py              training entry point
test_model.py         inference + background CSV collection
collect_metrics.py    standalone metrics collector
```

`rl/fuzzy.py` is the single source of truth for the trapezoids. The fuzzy agent
reads their membership degrees, the crisp agent reads the boundaries derived from
them, and the conventional agent ignores both.

## Running

```bash
cp .env.example .env    # set PROMETHEUS_URL, ALGORITHM, DEPLOYMENT_NAME
uv sync

make train             # trains ALGORITHM against DEPLOYMENT_NAME
make test              # requires MODEL_PATH
make lint
```

`ALGORITHM` and `DEPLOYMENT_NAME` must be set together — each agent trains
against its own deployment so the three can run concurrently without
interfering. Checkpoints are written per model to
`model/{qlearning,qlearningcrisp,qlearningfuzzy}/{start_time}_{note}/`.

Models are tagged with their `agent_type` on save; loading a fuzzy model into a
crisp agent is refused rather than silently producing nonsense, since their state
keys are not interchangeable.

Load generation lives in [`load-resource-intensive-ii/`](load-resource-intensive-ii):
one script per model (`k6-test-q-learning.js`, `k6-test-qlearning-crisp.js`,
`k6-test-qlearning-fuzzy.js`), identical load profiles differing only in route.

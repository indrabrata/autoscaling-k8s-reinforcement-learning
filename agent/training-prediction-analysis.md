# Analisis Detail Alur Training dan Prediction: Autoscaling Kubernetes dengan Reinforcement Learning

## Daftar Isi

- [BAGIAN A: ALUR TRAINING](#bagian-a-alur-training)
  - [1. Inisialisasi Sistem](#1-inisialisasi-sistem)
  - [2. Komponen Utama Training](#2-komponen-utama-training)
  - [3. Training Loop — Trainer Class](#3-training-loop--trainer-class)
  - [4. Environment: reset()](#4-environment-reset)
  - [5. Agent: get_action()](#5-agent-get_action)
  - [6. Environment: step()](#6-environment-step)
  - [7. Interaksi dengan Kubernetes Cluster](#7-interaksi-dengan-kubernetes-cluster)
  - [8. Pengumpulan Metrik dari Prometheus](#8-pengumpulan-metrik-dari-prometheus)
  - [9. Observasi (State Representation)](#9-observasi-state-representation)
  - [10. Agent: update_q_table()](#10-agent-update_q_table)
  - [11. Perbedaan State Key: Q-Learning vs Q-Fuzzy](#11-perbedaan-state-key-q-learning-vs-q-fuzzy)
  - [12. Fuzzy Logic — Fuzzifikasi State](#12-fuzzy-logic--fuzzifikasi-state)
  - [13. Checkpoint dan Model Saving](#13-checkpoint-dan-model-saving)
  - [14. Hyperparameter Training](#14-hyperparameter-training)
  - [15. Diagram Alur Training End-to-End](#15-diagram-alur-training-end-to-end)
- [BAGIAN B: ALUR PREDICTION](#bagian-b-alur-prediction)
  - [16. Inisialisasi Prediction](#16-inisialisasi-prediction)
  - [17. Prediction Loop](#17-prediction-loop)
  - [18. Perbedaan Training vs Prediction](#18-perbedaan-training-vs-prediction)
  - [19. Diagram Alur Prediction End-to-End](#19-diagram-alur-prediction-end-to-end)
- [BAGIAN C: REKOMENDASI JAWABAN SIDANG](#bagian-c-rekomendasi-jawaban-sidang)

---

# BAGIAN A: ALUR TRAINING

## 1. Inisialisasi Sistem

Training dimulai dari `train.py` (line 50–169). Proses inisialisasi terdiri dari 5 tahap:

### 1.1 Setup Logger (line 52–56)

```python
logger = setup_logger(
    "kubernetes_agent",
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_to_file=True,
)
```

Logger mencatat semua aktivitas ke console dan file rotasi (max 10 MB per file, 5 backup). File log disimpan di `logs/{YYYY-MM-DD-HH-MM}/`.

### 1.2 Koneksi InfluxDB (line 58–64)

```python
influxdb = InfluxDB(
    url="http://localhost:8086",
    token="my-token",
    org="my-org",
    bucket="my-bucket",
)
```

InfluxDB digunakan untuk **menyimpan metrik training** setiap iterasi — reward, CPU, memory, response time, request rate, jumlah replica, ukuran Q-table, dan lainnya. Data ini digunakan untuk analisis post-training.

### 1.3 Inisialisasi Environment (line 77–102)

```python
env = KubernetesEnv(
    min_replicas=1,       # Minimum pod yang diizinkan
    max_replicas=12,      # Maksimum pod yang diizinkan
    iteration=10,         # Jumlah step per episode
    min_cpu=10,           # Batas bawah CPU usage (%) — di bawah ini = wasteful
    max_cpu=90,           # Batas atas CPU usage (%) — di atas ini = critical
    min_memory=10,        # Batas bawah memory usage (%)
    max_memory=90,        # Batas atas memory usage (%)
    max_response_time=100.0,  # Target SLA response time (ms)
    timeout=120,          # Timeout menunggu pod ready (detik)
    wait_time=60,         # Waktu tunggu setelah scaling sebelum ambil metrik (detik)
    request_rate_per_pod_capacity=80.0,  # Kapasitas RPS per pod
    algorithm="Q-LEARNING",  # Pilihan: "Q-LEARNING" atau "Q-LEARNING-FUZZY"
)
```

Environment terhubung langsung ke:
- **Kubernetes API** — untuk scaling deployment (menambah/mengurangi pod)
- **Prometheus** — untuk mengumpulkan metrik real-time (CPU, memory, response time, request rate)
- **InfluxDB** — untuk logging metrik training

### 1.4 Inisialisasi Agent (line 104–127)

```python
# Q-Learning
algorithm = QLearning(
    learning_rate=0.1,      # Alpha (α) — seberapa cepat belajar
    discount_factor=0.95,   # Gamma (γ) — seberapa penting future reward
    epsilon_start=0.1,      # Probabilitas awal eksplorasi
    epsilon_decay=0.99,     # Laju penurunan epsilon per step
    epsilon_min=0.01,       # Batas bawah epsilon
)

# ATAU Q-Learning Fuzzy
algorithm = QLearningFuzzy(
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon_start=0.1,
    epsilon_decay=0.99,
    epsilon_min=0.01,
)
```

Kedua agent memiliki:
- **Q-table**: dictionary `{state_key: numpy.array(100)}` — menyimpan Q-value untuk 100 aksi
- **100 aksi** (action 0–99): masing-masing merepresentasikan persentase dari rentang replica

### 1.5 Inisialisasi Trainer (line 158–166)

```python
trainer = Trainer(
    agent=algorithm,
    env=env,
    resume=True/False,           # Lanjutkan dari checkpoint
    resume_path="path/to/model.pkl",
    reset_epsilon=True,          # Reset epsilon saat resume
    change_epsilon_decay=0.90,   # Ubah decay saat resume
)
```

Trainer mendukung **resume training** — melanjutkan dari model checkpoint sebelumnya. Saat resume:
- Q-table dimuat dari file pickle
- Epsilon bisa di-reset (mulai eksplorasi ulang) atau dilanjutkan
- Epsilon decay bisa diubah (misalnya lebih agresif)

---

## 2. Komponen Utama Training

### Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────────┐
│                          train.py                                   │
│  (Entry point: inisialisasi semua komponen, jalankan training)      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Trainer                                     │
│  (Orkestrator: episode loop, checkpoint, signal handling)            │
│                                                                     │
│  ┌──────────────────┐     ┌──────────────────────────────────────┐  │
│  │      Agent       │     │          Environment                 │  │
│  │                  │     │                                      │  │
│  │  QLearning       │     │  KubernetesEnv                       │  │
│  │  ATAU            │◄───►│                                      │  │
│  │  QLearningFuzzy  │     │  ┌──────────┐  ┌──────────────────┐  │  │
│  │                  │     │  │Kubernetes│  │   Prometheus     │  │  │
│  │  ┌────────────┐  │     │  │  API     │  │   (Metrics)      │  │  │
│  │  │  Q-Table   │  │     │  └──────────┘  └──────────────────┘  │  │
│  │  │  {s: Q[a]} │  │     │                                      │  │
│  │  └────────────┘  │     │  ┌──────────────────────────────────┐ │  │
│  │                  │     │  │         InfluxDB                 │ │  │
│  │  ┌────────────┐  │     │  │    (Metrics Storage)             │ │  │
│  │  │   Fuzzy    │  │     │  └──────────────────────────────────┘ │  │
│  │  │  (Q-Fuzzy) │  │     │                                      │  │
│  │  └────────────┘  │     │                                      │  │
│  └──────────────────┘     └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### File dan Tanggung Jawab

| File | Kelas/Fungsi | Tanggung Jawab |
|------|-------------|----------------|
| `train.py` | - | Entry point, inisialisasi, konfigurasi |
| `trainer.py` | `Trainer` | Orkestrator episode loop, checkpoint, signal handling |
| `environment/environment.py` | `KubernetesEnv` | Interface ke Kubernetes, hitung reward, kelola state |
| `rl/q_learning.py` | `QLearning` | Agent Q-Learning dengan state kontinu |
| `rl/q_learning_fuzzy.py` | `QLearningFuzzy` | Agent Q-Learning dengan state fuzzy |
| `rl/fuzzy.py` | `Fuzzy` | Fuzzifikasi state, membership function |
| `utils/metrics.py` | `get_metrics()` | Query Prometheus, hitung CPU/MEM/RT/RPS |
| `utils/cluster.py` | `wait_for_pods_ready()` | Tunggu pod ready setelah scaling |
| `utils/logger.py` | `log_verbose_details()` | Visualisasi metrik per iterasi |
| `database/influxdb.py` | `InfluxDB` | Penyimpanan metrik ke InfluxDB |

---

## 3. Training Loop — Trainer Class

Training loop berada di `trainer.py:102-151`:

```python
def train(self, episodes, note, start_time):
    self._install_signal_handlers()       # Tangani SIGINT/SIGTERM
    total_best = float("-inf")

    for ep in range(episodes):
        agent.add_episode_count()          # Increment episode counter
        obs = env.reset()                  # Reset environment
        total = 0.0

        while True:                        # Iteration loop
            act = agent.get_action(obs)    # Pilih aksi (epsilon-greedy)
            nxt, rew, term, info = env.step(act)  # Eksekusi aksi
            agent.update_q_table(obs, act, rew, nxt)  # Update Q-value
            total += rew
            obs = nxt

            if term:                       # Episode selesai?
                break

        if total > total_best:             # Best model?
            total_best = total
            self._save_checkpoint(ep, total_best, note, start_time)
```

### Struktur Episode

Satu training session terdiri dari beberapa episode, dan setiap episode terdiri dari beberapa iterasi (step):

```
Training Session
├── Episode 1  (iteration = 10 step)
│   ├── Step 1:  get_action → step → update_q_table
│   ├── Step 2:  get_action → step → update_q_table
│   ├── ...
│   └── Step 10: get_action → step → update_q_table → terminated
│
├── Episode 2  (iteration = 10 step)
│   ├── Step 1:  reset → get_action → step → update_q_table
│   ├── ...
│   └── Step 10: terminated
│
└── Episode N
```

Setiap step melibatkan **interaksi nyata dengan cluster Kubernetes** — scaling deployment, menunggu pod ready, mengumpulkan metrik. Ini bukan simulasi.

### Signal Handling (line 79–100)

Trainer menangani SIGINT (Ctrl+C) dan SIGTERM secara graceful:

```python
def _signal_handler(self, signum, frame):
    self._interrupted_save()   # Simpan model sebelum exit
    raise KeyboardInterrupt
```

Model yang tersimpan saat interupsi disimpan di `model/{type}/{timestamp}/interrupted/`.

---

## 4. Environment: reset()

Fungsi `reset()` dipanggil di awal setiap episode (`environment.py:1120-1140`):

```python
def reset(self):
    self.iteration = self.initial_iteration    # Reset counter iterasi
    self.replica_state = self.min_replicas      # Kembalikan ke replica minimum
    self._scale_and_get_metrics()               # Scale ke min & ambil metrik
    self.last_action = 0                        # Reset aksi terakhir
    self.action_history = []                    # Kosongkan riwayat aksi
    self.cumulative_reward = 0.0                # Reset cumulative reward
    self.episode_reward = 0.0                   # Reset episode reward
    self.episode_number += 1                    # Increment episode counter
    return self._get_observation()              # Return observasi awal
```

### Apa yang Terjadi Saat Reset

1. **Deployment di-scale ke minimum** (misalnya 1 pod) — ini adalah starting point yang konsisten
2. **Menunggu pod ready** via Prometheus query
3. **Mengumpulkan metrik awal** — CPU, memory, response time, request rate
4. **Mengembalikan observasi** — dictionary berisi semua state variable

Ini berarti setiap episode dimulai dari kondisi yang sama: jumlah replica minimum, memberikan agent "clean slate" untuk belajar.

---

## 5. Agent: get_action()

Pemilihan aksi menggunakan **epsilon-greedy policy** (`q_learning.py:83-94`):

```python
def get_action(self, observation):
    state_key = self.get_state_key(observation)  # Konversi observasi ke state key

    if state_key not in self.q_table:
        self.q_table[state_key] = np.zeros(100)  # Inisialisasi Q-value = 0

    if np.random.rand() < self.epsilon:          # Eksplorasi
        action = np.random.randint(0, 100)       # Aksi random (0-99)
    else:                                         # Eksploitasi
        action = np.argmax(self.q_table[state_key])  # Aksi terbaik

    return action
```

### Interpretasi Aksi

Action space: integer 0–99, diinterpretasikan sebagai **persentase dari rentang replica**:

```python
# Di environment.step():
percentage = action / 99.0
replica_state = round(min_replicas + percentage * range_replicas)
```

| Action | Percentage | Replicas (min=1, max=12) |
|--------|-----------|-------------------------|
| 0 | 0.0% | 1 |
| 9 | 9.1% | 2 |
| 18 | 18.2% | 3 |
| 27 | 27.3% | 4 |
| 45 | 45.5% | 6 |
| 63 | 63.6% | 8 |
| 81 | 81.8% | 10 |
| 99 | 100.0% | 12 |

### Epsilon-Greedy Exploration

```
Epsilon = 0.1 (awal)  →  Epsilon = 0.01 (minimum)
         │                          │
         ▼                          ▼
   10% random action          1% random action
   90% best Q-value          99% best Q-value
```

Epsilon di-decay setiap kali `update_q_table()` dipanggil:

```python
epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

Contoh decay: `0.1 → 0.099 → 0.098 → ... → 0.01 (min)`

---

## 6. Environment: step()

Fungsi `step()` adalah inti interaksi agent-environment (`environment.py:1027-1118`):

```python
def step(self, action, q_table_size=0):
    # 1. Catat perubahan aksi
    self.action_change = action - self.last_action
    self.last_action = action
    self.action_history.append(action)

    # 2. Simpan request rate sebelumnya (untuk trend detection)
    self.previous_request_rate = self.request_rate

    # 3. Konversi aksi ke jumlah replica
    percentage = action / 99.0
    self.replica_state = round(min_replicas + percentage * range_replicas)
    self.replica_state = clamp(self.replica_state, min_replicas, max_replicas)

    # 4. Scale cluster dan ambil metrik
    self._scale_and_get_metrics()

    # 5. Hitung trend request rate
    self.request_rate_trend = self.request_rate - self.previous_request_rate

    # 6. Hitung reward
    reward, reward_breakdown = self._calculate_reward()

    # 7. Kurangi iterasi, cek terminasi
    self.iteration -= 1
    terminated = self.iteration <= 0

    # 8. Buat observasi baru
    observation = self._get_observation()

    # 9. Simpan ke InfluxDB
    self.influxdb.write_point(...)

    return observation, reward, terminated, info
```

### Timeline Satu Step

```
t=0s    Agent memilih action (mis: action=45 → 6 replicas)
        │
t=0s    Environment memanggil Kubernetes API untuk scale
        │
t=1-60s Menunggu pod ready (wait_for_pods_ready)
        │ - Query Prometheus setiap detik
        │ - Cek: ready_replicas == desired_replicas?
        │
t=60s   Menunggu stabilisasi metrik (wait_time=60s)
        │ - Pod sudah running tapi metrik belum stabil
        │ - Butuh waktu agar CPU/MEM/RT mencerminkan load sebenarnya
        │
t=60s+  Mengumpulkan metrik dari Prometheus
        │ - CPU usage (rata-rata semua pod)
        │ - Memory usage (rata-rata semua pod)
        │ - Response time (P90 quantile)
        │ - Request rate (total RPS)
        │
t=...   Hitung reward berdasarkan metrik
        │
t=...   Return (observation, reward, terminated, info)
```

**Catatan penting:** Setiap step memakan waktu **~1-3 menit** di dunia nyata karena melibatkan scaling dan pengumpulan metrik real-time. Ini bukan simulasi — agent belajar dari cluster Kubernetes yang sebenarnya.

---

## 7. Interaksi dengan Kubernetes Cluster

### 7.1 Scaling — `_scale()` (environment.py:210-292)

```python
def _scale(self):
    self.cluster.patch_namespaced_deployment_scale(
        name=self.deployment_name,
        body=V1Scale(spec=V1ScaleSpec(replicas=int(self.replica_state))),
        namespace=self.namespace,
    )
```

Mekanisme scaling menggunakan **Kubernetes Python Client** untuk memanggil API `PATCH /apis/apps/v1/namespaces/{ns}/deployments/{name}/scale`.

**Retry logic:**
- Exponential backoff: delay mulai 1s, max 30s
- Timeout juga naik: 60s → max 300s
- Retry hingga `max_scaling_retries` (default: 1000)
- Menangani error spesifik: etcd timeout (500), conflict (409), API error lainnya

### 7.2 Menunggu Pod Ready — `wait_for_pods_ready()` (cluster.py:7-99)

Setelah scaling, environment menunggu hingga semua pod dalam status **Ready**:

```python
def wait_for_pods_ready(prometheus, deployment_name, desired_replicas, ...):
    while time.time() - start_time < timeout:
        # Query Prometheus untuk desired replicas
        desired = prometheus.custom_query(q_desired)

        # Query Prometheus untuk ready replicas
        ready = prometheus.custom_query(q_ready)

        if ready_replicas == desired_replicas:
            return True, desired, ready

        time.sleep(1)  # Polling setiap detik

    return False, desired, ready  # Timeout
```

Query Prometheus menggunakan PromQL yang memfilter pod berdasarkan:
1. Pod harus punya `kube_pod_status_ready{condition="true"}`
2. Pod harus milik ReplicaSet yang dimiliki Deployment target
3. Pod harus di namespace yang benar

---

## 8. Pengumpulan Metrik dari Prometheus

Fungsi `get_metrics()` (`utils/metrics.py:375-514`) mengumpulkan 4 metrik utama:

### 8.1 CPU Usage

```promql
sum by (pod) (
    rate(container_cpu_usage_seconds_total{
        namespace="default", container!="", container!="POD"
    }[15s])
)
```

- Menggunakan `rate()` dengan interval 15 detik
- Dihitung sebagai **persentase dari CPU limit** per pod
- Hasil akhir: `np.nanmean(cpu_percentages)` — rata-rata semua pod

### 8.2 Memory Usage

```promql
sum by (pod) (
    container_memory_working_set_bytes{
        namespace="default", container!="", container!="POD"
    }
)
```

- Menggunakan `working_set_bytes` (bukan RSS) — lebih akurat untuk Kubernetes
- Dihitung sebagai **persentase dari memory limit** per pod
- Hasil akhir: `np.nanmean(memory_percentages)`

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

- Menggunakan **histogram quantile P90** — 90% request selesai di bawah nilai ini
- Dihitung dalam **milidetik** (dikali 1000 dari seconds)
- Bisa multi-endpoint: rata-rata dari semua endpoint yang di-monitor
- Endpoint health check (`/metrics`, `/healthz`) dikecualikan

### 8.4 Request Rate

```promql
sum(
    rate(app_requests_total{
        namespace="default", pod=~"ecom-api-.*",
        exported_endpoint!~"/metrics|/healthz"
    }[15s])
)
```

- Total **Requests Per Second (RPS)** ke deployment
- Mengecualikan health check dan metrics endpoint
- Nilai ini akan dinormalisasi terhadap kapasitas cluster di `_get_observation()`

### 8.5 Retry Logic

Setiap metrik di-query dengan retry hingga:
1. Jumlah pod yang terdeteksi == jumlah replica yang diharapkan
2. Timeout tercapai

Ini memastikan metrik yang dikumpulkan **konsisten** dan mencerminkan semua pod.

### 8.6 Alur Pengumpulan Metrik

```
_scale_and_get_metrics()
    │
    ├── _scale()                          ← Kubernetes API: patch deployment
    │
    ├── wait_for_pods_ready()             ← Prometheus: cek pod readiness
    │
    └── get_metrics()
        │
        ├── sleep(wait_time)              ← Tunggu metrik stabil
        │
        ├── check_prometheus_connection() ← Pastikan Prometheus reachable
        │
        ├── _scrape_metrics()
        │   ├── CPU usage query           ← rate(container_cpu_usage)
        │   ├── Memory usage query        ← container_memory_working_set
        │   ├── CPU limits query          ← kube_pod_container_resource_limits
        │   └── Memory limits query       ← kube_pod_container_resource_limits
        │
        ├── _metrics_result()
        │   ├── CPU % = (usage / limit) * 100   per pod
        │   └── MEM % = (usage / limit) * 100   per pod
        │
        ├── _get_response_time()          ← histogram_quantile P90
        │
        └── _get_request_rate()           ← rate(app_requests_total)
        │
        └── Return: (cpu_mean, mem_mean, response_time, request_rate, pod_count)
```

---

## 9. Observasi (State Representation)

Fungsi `_get_observation()` (`environment.py:992-1025`) menghasilkan dictionary yang menjadi **input** bagi agent:

```python
{
    # Core resource metrics (0-100%)
    "cpu_usage": 55.2,                    # Rata-rata CPU usage semua pod
    "memory_usage": 62.1,                 # Rata-rata memory usage semua pod
    "response_time": 45.0,               # RT sebagai % dari max_response_time
    "response_time_ms": 45.0,            # RT dalam milidetik (untuk logging)

    # Request rate metrics
    "request_rate": 200.5,               # Total RPS (raw value)
    "request_rate_normalized": 62.7,     # RPS sebagai % dari kapasitas cluster
    "request_rate_trend": 15.3,          # Delta RPS dari step sebelumnya
    "request_rate_trend_category": "up", # Kategori: up/down/stable

    # Action dan system state
    "last_action": 45,                   # Aksi terakhir (0-99)
    "action_trend_category": "up",       # Tren aksi: up/down/stable
    "current_replicas": 6.0,             # Jumlah pod saat ini
}
```

### Normalisasi Metrik

| Metrik | Formula | Contoh |
|--------|---------|--------|
| `response_time` | `min((RT_ms / max_RT) * 100, 100)` | `min((45/100)*100, 100) = 45%` |
| `request_rate_normalized` | `min((RPS / (pod_capacity * replicas)) * 100, 100)` | `min((200/(80*4))*100, 100) = 62.5%` |

### Trend Detection

**Request Rate Trend** (`environment.py:929-941`):

```python
trend = request_rate - previous_request_rate

if trend > 5.0:      return "up"
elif trend < -5.0:   return "down"
else:                return "stable"
```

Threshold ±5.0 RPS — perubahan di bawah ini dianggap noise.

**Action Trend** (`environment.py:943-990`):

Menggunakan **linear regression** pada 5 aksi terakhir (sliding window):

```python
history = action_history[-5:]  # 5 aksi terakhir
slope = linear_regression(history)
scaled_slope = slope * 5  # Scale ke meaningful range

if scaled_slope > 5.0:    return "up"    # Agent cenderung scale up
elif scaled_slope < -5.0: return "down"  # Agent cenderung scale down
else:                     return "stable"
```

Ini mendeteksi **pola scaling** — apakah agent konsisten menambah, mengurangi, atau mempertahankan replica.

---

## 10. Agent: update_q_table()

Setelah menerima reward, Q-value di-update menggunakan **Bellman equation** (`q_learning.py:96-116`):

```python
def update_q_table(self, observation, action, reward, next_observation):
    state_key = self.get_state_key(observation)
    next_state_key = self.get_state_key(next_observation)

    # Inisialisasi state baru jika belum ada
    if state_key not in self.q_table:
        self.q_table[state_key] = np.zeros(100)
    if next_state_key not in self.q_table:
        self.q_table[next_state_key] = np.zeros(100)

    # Q-Learning update rule (Bellman equation)
    best_next = np.max(self.q_table[next_state_key])
    self.q_table[state_key][action] += learning_rate * (
        reward + discount_factor * best_next - self.q_table[state_key][action]
    )

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

### Formula Q-Learning

```
Q(s, a) ← Q(s, a) + α * [r + γ * max_a'(Q(s', a')) - Q(s, a)]
                      │    │   │                        │
                      │    │   │                        └── Estimasi lama
                      │    │   └── Estimasi terbaik di state berikutnya
                      │    └── Reward yang diterima
                      └── Learning rate
```

### Contoh Perhitungan Update

```
State saat ini: (CPU=55%, MEM=60%, RT=45%, ...)
Action: 45 (→ 6 replicas)
Reward: 1.75
State berikutnya: (CPU=50%, MEM=55%, RT=40%, ...)

Sebelum update:
  Q[(55,60,45,...), 45] = 0.5
  max Q[(50,55,40,...), *] = 0.8

Update:
  Q(s,a) += 0.1 * (1.75 + 0.95 * 0.8 - 0.5)
  Q(s,a) += 0.1 * (1.75 + 0.76 - 0.5)
  Q(s,a) += 0.1 * 2.01
  Q(s,a) += 0.201
  Q(s,a) = 0.5 + 0.201 = 0.701
```

Setelah update:
- Q-value untuk aksi 45 di state (55,60,45,...) naik dari 0.5 menjadi 0.701
- Agent akan lebih cenderung memilih aksi 45 (6 replicas) di state serupa ke depannya

---

## 11. Perbedaan State Key: Q-Learning vs Q-Fuzzy

### Q-Learning: State Key Kontinu

```python
# q_learning.py:41-81
def get_state_key(self, observation):
    return (
        cpu_usage,                    # float: 55.23
        memory_usage,                 # float: 62.10
        response_time,                # float: 45.00
        request_rate_normalized,      # float: 62.50
        request_rate_trend_category,  # str:   "up"
        last_action,                  # float: 45
        action_trend_category,        # str:   "stable"
    )
```

**Contoh state key:** `(55.23, 62.10, 45.00, 62.50, "up", 45, "stable")`

**Masalah:** Karena CPU, memory, dll. adalah float kontinu, kemungkinan dua state yang **identik** sangat kecil. Misalnya:
- `(55.23, 62.10, ...)` dan `(55.24, 62.10, ...)` adalah state berbeda
- Q-table tumbuh sangat besar tapi setiap state jarang dikunjungi ulang
- Generalisasi sulit — pengalaman di satu state tidak mentransfer ke state mirip

### Q-Learning Fuzzy: State Key Diskrit

```python
# q_learning_fuzzy.py:44-94
def get_state_key(self, observation):
    fuzzy_state = self.fuzzy.fuzzify(observation)

    cpu_label = max(fuzzy_state["cpu_usage"], key=...)     # "medium"
    mem_label = max(fuzzy_state["memory_usage"], key=...)  # "high"
    resp_label = max(fuzzy_state["response_time"], key=...) # "low"
    req_rate_label = max(fuzzy_state["request_rate_normalized"], key=...)
    last_action_label = max(fuzzy_state["last_action"], key=...)

    return (
        cpu_label,                    # str: "medium"
        mem_label,                    # str: "high"
        resp_label,                   # str: "low"
        req_rate_label,               # str: "medium"
        request_rate_trend_category,  # str: "up"
        last_action_label,            # str: "medium"
        action_trend_category,        # str: "stable"
    )
```

**Contoh state key:** `("medium", "high", "low", "medium", "up", "medium", "stable")`

**Keuntungan:** State space jauh lebih kecil:
- 5 label fuzzy x 5 metrik = 5^5 = 3.125 kombinasi resource
- x 3 trend request x 3 trend action = 3.125 x 9 = 28.125 state teoritis
- Dalam praktik, banyak kombinasi yang tidak terjadi → lebih compact
- **Generalisasi**: CPU=55.2% dan CPU=52.8% sama-sama "medium" → berbagi Q-value

### Perbandingan

| Aspek | Q-Learning (Kontinu) | Q-Fuzzy (Diskrit) |
|-------|---------------------|-------------------|
| State key | `(55.23, 62.10, 45.00, ...)` | `("medium", "high", "low", ...)` |
| Ukuran Q-table | Potensial tak terbatas | Maksimal ~28.125 state |
| Kunjungan ulang state | Sangat jarang | Sering |
| Generalisasi | Tidak ada | Otomatis via fuzzifikasi |
| Presisi | Tinggi (per titik) | Rendah (per kategori) |
| Konvergensi | Lebih lambat | Lebih cepat |
| Reward function | **SAMA** | **SAMA** |

---

## 12. Fuzzy Logic — Fuzzifikasi State

### 12.1 Membership Function

Kelas `Fuzzy` (`rl/fuzzy.py`) mendefinisikan **trapezoidal membership function** untuk setiap metrik:

```
Derajat keanggotaan
1.0 ─────┐        ┌─────
         │       /│\
         │      / │ \
         │     /  │  \
         │    /   │   \
0.0 ─────┘───/────│────\────
         a   b    c    d     → nilai input
```

Formula trapezoidal:

```python
def _trapezoidal(x, a, b, c, d):
    if x < a or x > d:    return 0.0    # Di luar range
    elif b <= x <= c:      return 1.0    # Fully member
    elif a < x < b:        return (x - a) / (b - a)  # Naik
    else:                  return (d - x) / (d - c)  # Turun
```

### 12.2 Definisi Membership untuk CPU Usage

```
1.0  ┬─very_low──┐    ┌──low───┐    ┌─medium──┐    ┌──high───┐    ┌─very_high─┬
     │           │   /│\       │\  /│         │\  /│\        │\  /│           │
     │           │  / │ \     / │\/ │         │ \/ │ \      / │\/ │           │
     │           │ /  │  \   /  │/\ │         │ /\ │  \    /  │/\ │           │
     │           │/   │   \ /   │  \│         │/  \│   \  /   │  \│           │
0.0  └───────────┴────┴────┴────┴───┴─────────┴────┴────┴─────┴───┴───────────┘
     0    10    25   35   45   50   60   65   70   75   85   90   95  100 (%)
```

| Label | a | b | c | d | Fully Member Range |
|-------|---|---|---|---|--------------------|
| very_low | 0 | 0 | 10 | 25 | 0–10% |
| low | 15 | 25 | 35 | 45 | 25–35% |
| medium | 40 | 50 | 60 | 70 | 50–60% |
| high | 65 | 75 | 85 | 90 | 75–85% |
| very_high | 85 | 95 | 100 | 100 | 95–100% |

### 12.3 Contoh Fuzzifikasi

Input: `CPU = 55%`

```
very_low(55)  = 0.0    (55 > 25)
low(55)       = 0.0    (55 > 45)
medium(55)    = 1.0    (50 ≤ 55 ≤ 60 → fully member)
high(55)      = 0.0    (55 < 65)
very_high(55) = 0.0    (55 < 85)

Dominant label: "medium" (derajat 1.0)
```

Input: `CPU = 42%`

```
very_low(42)  = 0.0
low(42)       = 0.3    ((45-42)/(45-35) = 3/10 = 0.3 → turun dari low)
medium(42)    = 0.2    ((42-40)/(50-40) = 2/10 = 0.2 → naik ke medium)
high(42)      = 0.0
very_high(42) = 0.0

Dominant label: "low" (derajat 0.3 > 0.2)
```

Input: `CPU = 72%`

```
very_low(72)  = 0.0
low(72)       = 0.0
medium(72)    = 0.0    (72 > 70)
high(72)      = 0.7    ((72-65)/(75-65) = 7/10 = 0.7 → naik ke high)
very_high(72) = 0.0    (72 < 85)

Dominant label: "high" (derajat 0.7)
```

### 12.4 Metrik yang Di-fuzzifikasi

Kelima metrik menggunakan definisi membership yang **identik** (semua di skala 0–100%):

1. `cpu_usage` — utilisasi CPU
2. `memory_usage` — utilisasi memory
3. `response_time` — response time sebagai % dari SLA
4. `request_rate_normalized` — utilisasi kapasitas cluster
5. `last_action` — aksi terakhir (0–99, diperlakukan sebagai persentase)

Dua metrik kategorikal **tidak di-fuzzifikasi**:
- `request_rate_trend_category` — sudah diskrit (up/down/stable)
- `action_trend_category` — sudah diskrit (up/down/stable)

---

## 13. Checkpoint dan Model Saving

### 13.1 Best Model Checkpoint

Setiap kali episode menghasilkan total reward tertinggi, model disimpan (`trainer.py:153-168`):

```python
def _save_checkpoint(self, episode, score, note, start_time):
    path = f"model/{model_type}/{start_time}_{note}/checkpoints/"
           f"episode_{episode}_total_{score}.pkl"
    agent.save_model(path, episode + 1)
```

**Struktur direktori:**

```
model/
├── qlearning/
│   └── 1706000000_experiment_1/
│       ├── checkpoints/
│       │   ├── episode_0_total_5.23.pkl
│       │   ├── episode_3_total_8.45.pkl    ← Semakin tinggi = semakin baik
│       │   └── episode_7_total_12.10.pkl
│       ├── interrupted/
│       │   └── interrupted_episode_5_1706003600.pkl
│       └── final/
│           └── qlearning_1706007200.pkl
└── qlearningfuzzy/
    └── ...
```

### 13.2 Isi Model File (.pkl)

```python
model_data = {
    "q_table": dict,            # {state_key: np.array(100)}
    "learning_rate": 0.1,
    "discount_factor": 0.95,
    "epsilon": 0.05,            # Epsilon saat disimpan
    "epsilon_min": 0.01,
    "epsilon_decay": 0.99,
    "n_actions": 100,
    "created_at": 1706000000,   # Unix timestamp
    "episodes_trained": 10,     # Total episode yang sudah dilatih
}
```

Diserialisasi menggunakan Python **pickle** — menyimpan seluruh state agent termasuk Q-table.

### 13.3 Auto-Resume

`train.py` mendukung auto-resume (`line 131-156`):

```python
if auto_resume:
    latest = find_latest_checkpoint(algorithm)  # Cari checkpoint terbaru
    if latest:
        resume_path = latest
```

`find_latest_checkpoint()` mencari file `.pkl` terbaru di `checkpoints/` dan `interrupted/`, sorted by modification time.

---

## 14. Hyperparameter Training

### 14.1 Tabel Hyperparameter

| Parameter | Nilai Default | Sumber | Pengaruh |
|-----------|--------------|--------|----------|
| `learning_rate` (α) | 0.1 | `.env` | Seberapa cepat Q-value berubah per update |
| `discount_factor` (γ) | 0.95 | `.env` | Seberapa penting future reward dibanding immediate |
| `epsilon_start` | 0.1 | `.env` | Probabilitas awal random exploration |
| `epsilon_decay` | 0.99 | `.env` | Laju penurunan epsilon per step |
| `epsilon_min` | 0.01 | `.env` | Batas bawah epsilon (selalu ada 1% eksplorasi) |
| `episodes` | 10 | `.env` | Jumlah episode per training session |
| `iteration` | 10 | `.env` | Jumlah step per episode |
| `wait_time` | 60s | `.env` | Waktu tunggu setelah scaling |
| `timeout` | 120s | `.env` | Timeout menunggu pod ready |
| `metrics_interval` | 15s | `.env` | Window rate() di PromQL |
| `metrics_quantile` | 0.90 | `.env` | Quantile untuk response time (P90) |

### 14.2 Interaksi Antar Hyperparameter

```
episodes x iteration = total_steps per training session
10 x 10 = 100 steps

Setiap step ≈ 1-3 menit (scaling + wait_time + metrics collection)
100 steps x ~2 menit = ~200 menit (~3.3 jam) per training session
```

```
epsilon_decay^(total_steps) = epsilon akhir
0.99^100 = 0.366 → masih banyak eksplorasi setelah 100 step
0.99^1000 = 0.00004 → hampir full exploitation setelah 1000 step
```

### 14.3 Discount Factor (γ = 0.95)

Pengaruh future reward terhadap keputusan saat ini:

| Step di masa depan | Kontribusi ke Q-value | Kalkulasi |
|--------------------|----------------------|-----------|
| t+1 | 95% | 0.95^1 = 0.950 |
| t+2 | 90.2% | 0.95^2 = 0.902 |
| t+3 | 85.7% | 0.95^3 = 0.857 |
| t+5 | 77.4% | 0.95^5 = 0.774 |
| t+10 | 59.9% | 0.95^10 = 0.599 |

γ = 0.95 berarti agent memperhitungkan masa depan secara signifikan — reward 10 step ke depan masih bernilai ~60% dari reward sekarang. Ini cocok untuk autoscaling karena keputusan scaling berdampak jangka panjang.

---

## 15. Diagram Alur Training End-to-End

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              train.py                                        │
│                                                                              │
│  1. Setup Logger                                                             │
│  2. Koneksi InfluxDB                                                         │
│  3. Inisialisasi Environment (KubernetesEnv)                                 │
│  4. Inisialisasi Agent (QLearning / QLearningFuzzy)                          │
│  5. Inisialisasi Trainer                                                     │
│  6. trainer.train(episodes=N)                                                │
└──────────────────────┬───────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         EPISODE LOOP (1..N)                                  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  env.reset()                                                           │  │
│  │    ├── replica_state = min_replicas                                    │  │
│  │    ├── Kubernetes API: scale deployment ke min                         │  │
│  │    ├── Prometheus: wait_for_pods_ready()                               │  │
│  │    ├── sleep(wait_time)                                                │  │
│  │    ├── Prometheus: get_metrics(CPU, MEM, RT, RPS)                      │  │
│  │    ├── Reset: action_history, rewards, counters                        │  │
│  │    └── Return: observation (dict)                                      │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                       │                                                      │
│                       ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  ITERATION LOOP (1..M per episode)                                     │  │
│  │                                                                         │  │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │  │
│  │  │  STEP 1: agent.get_action(observation)                           │  │  │
│  │  │    ├── Extract state_key:                                        │  │  │
│  │  │    │   Q-Learning: (55.2, 62.1, 45.0, 62.5, "up", 45, "stable")│  │  │
│  │  │    │   Q-Fuzzy:    ("medium","high","low","medium","up",...)     │  │  │
│  │  │    ├── if state_key not in Q-table: Q[state] = zeros(100)       │  │  │
│  │  │    ├── Epsilon-greedy:                                           │  │  │
│  │  │    │   random < epsilon? → random action (0-99)                  │  │  │
│  │  │    │   else            → argmax(Q[state])                        │  │  │
│  │  │    └── Return: action (int 0-99)                                 │  │  │
│  │  └───────────────────────────┬───────────────────────────────────────┘  │  │
│  │                              │                                          │  │
│  │                              ▼                                          │  │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │  │
│  │  │  STEP 2: env.step(action)                                        │  │  │
│  │  │    ├── Konversi action → replica count                           │  │  │
│  │  │    │   percentage = action / 99                                   │  │  │
│  │  │    │   replicas = round(min + percentage * range)                 │  │  │
│  │  │    │                                                              │  │  │
│  │  │    ├── _scale()                                                   │  │  │
│  │  │    │   └── Kubernetes API: patch_deployment_scale(replicas)       │  │  │
│  │  │    │                                                              │  │  │
│  │  │    ├── wait_for_pods_ready()                                      │  │  │
│  │  │    │   └── Prometheus: poll ready_replicas == desired (loop)      │  │  │
│  │  │    │                                                              │  │  │
│  │  │    ├── get_metrics()                                              │  │  │
│  │  │    │   ├── sleep(wait_time)   ← metrik stabilization              │  │  │
│  │  │    │   ├── CPU: rate(container_cpu_usage) / limits * 100          │  │  │
│  │  │    │   ├── MEM: working_set_bytes / limits * 100                  │  │  │
│  │  │    │   ├── RT:  histogram_quantile(0.90, latency_bucket)          │  │  │
│  │  │    │   └── RPS: rate(app_requests_total)                          │  │  │
│  │  │    │                                                              │  │  │
│  │  │    ├── _calculate_reward()                                        │  │  │
│  │  │    │   ├── Evaluate: optimal, balanced, wasteful, critical        │  │  │
│  │  │    │   ├── Base reward = positive / (1 + negative)                │  │  │
│  │  │    │   ├── Apply cost penalty                                     │  │  │
│  │  │    │   ├── Apply trend modifiers                                  │  │  │
│  │  │    │   └── Apply achievement bonuses                              │  │  │
│  │  │    │                                                              │  │  │
│  │  │    ├── Write to InfluxDB (all metrics + reward)                   │  │  │
│  │  │    │                                                              │  │  │
│  │  │    └── Return: (next_observation, reward, terminated, info)       │  │  │
│  │  └───────────────────────────┬───────────────────────────────────────┘  │  │
│  │                              │                                          │  │
│  │                              ▼                                          │  │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │  │
│  │  │  STEP 3: agent.update_q_table(obs, action, reward, next_obs)     │  │  │
│  │  │    ├── state_key = get_state_key(obs)                            │  │  │
│  │  │    ├── next_key  = get_state_key(next_obs)                       │  │  │
│  │  │    ├── best_next = max(Q[next_key])                              │  │  │
│  │  │    ├── Q[s,a] += α * (r + γ * best_next - Q[s,a])               │  │  │
│  │  │    └── epsilon *= epsilon_decay                                   │  │  │
│  │  └───────────────────────────┬───────────────────────────────────────┘  │  │
│  │                              │                                          │  │
│  │                              ▼                                          │  │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │  │
│  │  │  STEP 4: Logging & Tracking                                      │  │  │
│  │  │    ├── total_reward += reward                                     │  │  │
│  │  │    ├── log_verbose_details() → visual bars, trends, Q-values     │  │  │
│  │  │    └── obs = next_obs                                            │  │  │
│  │  └───────────────────────────┬───────────────────────────────────────┘  │  │
│  │                              │                                          │  │
│  │                     terminated?                                         │  │
│  │                    /          \                                          │  │
│  │                  No            Yes                                       │  │
│  │                  │              │                                        │  │
│  │            Loop kembali    Keluar loop                                   │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                       │                                                      │
│                       ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  total_reward > best?                                                  │  │
│  │    Yes → _save_checkpoint(episode, total_reward)                       │  │
│  │    No  → lanjut episode berikutnya                                     │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  SETELAH TRAINING SELESAI                                                    │
│    ├── Log Q-table summary (5 state pertama)                                 │
│    ├── Simpan model final ke model/{type}/{timestamp}/final/                 │
│    └── Log: "Model saved to: ..."                                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

# BAGIAN B: ALUR PREDICTION

## 16. Inisialisasi Prediction

Prediction dimulai dari `predict.py` (line 15–113). Berbeda dari training:

### 16.1 Langkah-langkah Inisialisasi

```python
# 1-4: Sama seperti training (logger, influxdb, environment, agent)

# 5. Load model yang sudah dilatih
model_path = os.getenv("MODEL_PATH", "")
agent.load_model(model_path)    # Load Q-table dari file .pkl

# 6. KRITIS: Set epsilon = 0 (full exploitation, tanpa eksplorasi)
agent.epsilon = 0

# 7. Reset environment
obs = env.reset()
```

### 16.2 Load Model

```python
def load_model(self, filepath):
    with open(filepath, "rb") as f:
        model_data = pickle.load(f)

    self.q_table = model_data["q_table"]       # Q-table terlatih
    self.learning_rate = model_data["learning_rate"]
    self.discount_factor = model_data["discount_factor"]
    self.epsilon = model_data["epsilon"]         # Akan di-override ke 0
    self.n_actions = model_data["n_actions"]
    self.episodes_trained = model_data["episodes_trained"]
```

---

## 17. Prediction Loop

Loop prediction berjalan **tanpa batas** (infinite loop) — terus menerus mengontrol cluster (`predict.py:101-113`):

```python
while True:
    act = agent.get_action(obs)                              # Selalu pilih best action
    nxt, rew, term, info = env.step(act, q_table_size=len(agent.q_table))
    obs = nxt
    log_verbose_details(obs, agent, verbose=True, logger=logger)
```

### Perbedaan Kritis dari Training

1. **epsilon = 0** → Tidak ada eksplorasi random. Setiap aksi adalah `argmax(Q[state])` — aksi terbaik berdasarkan apa yang dipelajari
2. **Tidak ada `update_q_table()`** → Q-table tidak berubah. Model bersifat read-only
3. **Infinite loop** → Tidak ada konsep episode atau terminasi. Agent terus berjalan
4. **Tidak ada checkpoint** → Model tidak disimpan ulang

### Alur Per Step di Prediction

```
1. Observasi state saat ini
   obs = {cpu: 65%, mem: 55%, rt: 40%, rps_norm: 70%, ...}

2. Agent memilih aksi terbaik
   state_key = get_state_key(obs)
   action = argmax(Q[state_key])     # Misal: action = 54

3. Environment mengeksekusi
   replicas = round(1 + (54/99) * 11) = round(1 + 6.0) = 7
   → Scale ke 7 pod
   → Tunggu pod ready
   → Tunggu metrik stabil
   → Ambil metrik baru

4. Terima observation baru
   next_obs = {cpu: 50%, mem: 48%, rt: 35%, ...}

5. Log metrik
   ▶ Iter 05 | CPU 50.12% █████░░░░░ | MEM 48.33% ████░░░░░░ | ...

6. Ulangi dari langkah 1 dengan next_obs
```

### Handling State Baru (Tidak Ada di Q-table)

Saat prediction menemui state yang **belum pernah dilihat** saat training:

```python
def get_action(self, observation):
    state_key = self.get_state_key(observation)

    if state_key not in self.q_table:
        self.q_table[state_key] = np.zeros(100)  # Semua Q-value = 0

    # epsilon = 0, jadi selalu ke sini:
    action = np.argmax(self.q_table[state_key])  # argmax([0,0,...,0]) = 0
    return action  # Action = 0 → min replicas
```

**Implikasi:** Jika Q-Fuzzy menemui state baru, agent default ke action 0 (minimum replicas). Ini bisa bermasalah jika state baru adalah kondisi critical. Namun karena Q-Fuzzy memiliki state space terbatas, kemungkinan state baru jauh lebih kecil dibanding Q-Learning kontinu.

---

## 18. Perbedaan Training vs Prediction

| Aspek | Training | Prediction |
|-------|---------|-----------|
| **File** | `train.py` | `predict.py` |
| **Epsilon** | 0.1 → decay → 0.01 | **0** (fixed) |
| **Eksplorasi** | Ada (epsilon-greedy) | **Tidak ada** (pure exploitation) |
| **Q-table update** | Setiap step (`update_q_table()`) | **Tidak pernah** |
| **Loop** | Episode-based (terbatas) | **Infinite** (tanpa henti) |
| **Terminasi** | Setelah N episode | Hanya manual (Ctrl+C) |
| **Checkpoint** | Simpan best model | **Tidak ada** |
| **InfluxDB** | Simpan semua metrik | Simpan semua metrik |
| **Model** | Dibuat dari nol / resume | **Dimuat dari file** |
| **Tujuan** | Belajar kebijakan optimal | **Menerapkan kebijakan** |
| **reset()** | Setiap episode (scale ke min) | Sekali di awal saja |

---

## 19. Diagram Alur Prediction End-to-End

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              predict.py                                      │
│                                                                              │
│  1. Setup Logger                                                             │
│  2. Koneksi InfluxDB                                                         │
│  3. Inisialisasi Environment (KubernetesEnv)                                 │
│  4. Inisialisasi Agent (QLearning / QLearningFuzzy)                          │
│  5. agent.load_model(MODEL_PATH)   ← Load Q-table terlatih                  │
│  6. agent.epsilon = 0              ← KRITIS: tanpa eksplorasi               │
│  7. obs = env.reset()              ← Scale ke min, ambil metrik awal        │
└──────────────────────┬───────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       INFINITE LOOP                                          │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  agent.get_action(obs)                                               │    │
│  │    ├── state_key = get_state_key(obs)                                │    │
│  │    ├── if new state: Q[state] = zeros(100) → action = 0 (default)   │    │
│  │    └── action = argmax(Q[state_key])  ← SELALU best action          │    │
│  └────────────────────────────┬─────────────────────────────────────────┘    │
│                               │                                              │
│                               ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  env.step(action)                                                    │    │
│  │    ├── Konversi action → replica count                               │    │
│  │    ├── Kubernetes API: scale deployment                              │    │
│  │    ├── Prometheus: wait_for_pods_ready()                             │    │
│  │    ├── sleep(wait_time)                                              │    │
│  │    ├── Prometheus: get_metrics(CPU, MEM, RT, RPS)                    │    │
│  │    ├── _calculate_reward()  ← dihitung tapi tidak dipakai update     │    │
│  │    ├── Write to InfluxDB                                             │    │
│  │    └── Return: (next_obs, reward, terminated, info)                  │    │
│  └────────────────────────────┬─────────────────────────────────────────┘    │
│                               │                                              │
│                               ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  obs = next_obs                                                      │    │
│  │  log_verbose_details()  ← Visual monitoring                          │    │
│  │                                                                       │    │
│  │  TIDAK ADA:                                                           │    │
│  │    ✗ update_q_table()                                                │    │
│  │    ✗ epsilon decay                                                    │    │
│  │    ✗ checkpoint saving                                                │    │
│  └────────────────────────────┬─────────────────────────────────────────┘    │
│                               │                                              │
│                          Loop kembali                                        │
│                          (selamanya)                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

# BAGIAN C: REKOMENDASI JAWABAN SIDANG

### Q1: "Jelaskan alur training secara singkat"

> Training terdiri dari episode loop. Setiap episode dimulai dengan **reset** (scale ke minimum replica), lalu agent memilih aksi menggunakan **epsilon-greedy**, environment mengeksekusi aksi tersebut di **cluster Kubernetes nyata** (bukan simulasi), mengumpulkan metrik dari **Prometheus**, menghitung reward, lalu agent meng-update **Q-table** menggunakan Bellman equation. Proses ini berulang hingga iterasi habis. Model terbaik (total reward tertinggi per episode) disimpan sebagai checkpoint.

### Q2: "Kenapa action space 0-99? Bukan langsung jumlah replica?"

> Action space 0–99 didesain sebagai **persentase dari rentang replica**. Ini memberikan **granularitas konsisten** terlepas dari konfigurasi min/max replicas. Jika kita menggunakan jumlah replica langsung (1–12), action space hanya 12 aksi. Dengan 100 aksi, agent bisa memilih jumlah replica secara lebih halus. Selain itu, persentase membuat model **portable** — Q-table yang dilatih dengan max_replicas=12 secara konseptual bisa digunakan pada konfigurasi berbeda, karena aksi diinterpretasikan secara relatif.

### Q3: "Kenapa environment menggunakan cluster nyata, bukan simulasi?"

> Pendekatan **online learning** di cluster nyata dipilih karena autoscaling sangat bergantung pada kondisi yang sulit disimulasikan: latensi jaringan, cold start pod, resource contention antar pod, perilaku garbage collector, caching effect, dan interaksi dengan load balancer. Simulasi akan memerlukan **modeling** yang sangat akurat untuk semua faktor ini, dan hasilnya belum tentu mentransfer ke cluster nyata (sim-to-real gap). Dengan belajar langsung di cluster, agent menangkap dinamika **end-to-end** yang sebenarnya.

### Q4: "Apa keuntungan Q-Learning Fuzzy dibanding Q-Learning biasa?"

> Keuntungan utama adalah **reduksi state space** dan **generalisasi otomatis**. Q-Learning kontinu menghasilkan state key unik untuk setiap variasi kecil metrik — `CPU=55.2%` dan `CPU=55.3%` adalah state berbeda. Akibatnya, Q-table tumbuh besar tapi setiap state jarang dikunjungi ulang, memperlambat konvergensi. Q-Fuzzy memetakan range kontinu ke 5 label diskrit, sehingga `CPU=55.2%` dan `CPU=55.3%` sama-sama "medium" dan berbagi Q-value yang sama. Ini membuat pengalaman **digunakan ulang** secara efektif, mempercepat konvergensi dengan lebih sedikit data.

### Q5: "Bagaimana agent menangani state yang belum pernah dilihat saat prediction?"

> Saat prediction menemui state baru (tidak ada di Q-table), Q-value diinisialisasi ke **nol untuk semua aksi**. Karena `argmax([0, 0, ..., 0])` mengembalikan indeks 0, agent default ke action 0, yaitu **minimum replica**. Ini adalah perilaku konservatif — lebih baik di minimum (berisiko sedikit under-provisioned) daripada over-provisioning di state yang tidak dipahami. Untuk Q-Fuzzy, risiko ini jauh lebih kecil karena state space terbatas (~28.125 kombinasi), sehingga lebih banyak state sudah tercover saat training.

### Q6: "Kenapa wait_time = 60 detik setelah scaling?"

> **Wait time** diperlukan karena ada **propagation delay** antara scaling dan metrik yang stabil. Saat pod baru dibuat, ia membutuhkan waktu untuk: (1) pull container image, (2) startup aplikasi, (3) warm up JIT/cache, (4) mulai menerima traffic dari load balancer, dan (5) metrik Prometheus ter-scrape dan terakumulasi cukup data. Jika kita mengambil metrik terlalu cepat, CPU/memory masih mencerminkan transient state (startup spike), bukan kondisi steady-state yang sebenarnya. 60 detik memberikan buffer yang cukup untuk stabilisasi.

### Q7: "Kenapa menggunakan P90 untuk response time, bukan rata-rata?"

> Rata-rata (mean) sensitif terhadap outlier dan bisa menyembunyikan masalah. Misalnya, jika 99 request selesai dalam 10ms tapi 1 request butuh 10.000ms, rata-rata = 109ms — terlihat buruk padahal mayoritas request baik. **P90 (persentil ke-90)** berarti "90% request selesai di bawah nilai ini" — lebih representatif terhadap pengalaman mayoritas user. Ini juga standar industri untuk **SLA monitoring**: Google, Amazon, dan perusahaan besar menggunakan P90/P95/P99 untuk mengukur performa layanan.

### Q8: "Apa perbedaan training dan prediction di level kode?"

> Perbedaan fundamental ada di tiga hal:
> 1. **Epsilon = 0**: Prediction tidak pernah melakukan aksi random. Setiap keputusan adalah yang terbaik menurut Q-table
> 2. **Tidak ada `update_q_table()`**: Q-table bersifat frozen/read-only. Agent tidak belajar lagi
> 3. **Infinite loop**: Tidak ada konsep episode atau terminasi — agent berjalan terus menerus mengontrol cluster
>
> Secara arsitektural, prediction adalah **deployment** dari model yang sudah dilatih, analog dengan serving di machine learning tradisional.

### Q9: "Bagaimana sistem menangani interupsi saat training?"

> Trainer memasang **signal handler** untuk SIGINT (Ctrl+C) dan SIGTERM. Saat sinyal diterima, handler secara otomatis menyimpan model ke direktori `interrupted/` sebelum proses berhenti. Model yang tersimpan mencakup seluruh Q-table, hyperparameter, dan jumlah episode yang sudah dilatih. Training bisa dilanjutkan (**resume**) dari checkpoint ini menggunakan flag `AUTO_RESUME=True` yang mencari file `.pkl` terbaru, atau `RESUME=True` dengan `RESUME_PATH` spesifik. Saat resume, epsilon bisa di-reset untuk memberikan eksplorasi tambahan pada knowledge yang sudah ada.

### Q10: "Apa kelemahan pendekatan training ini?"

> Beberapa kelemahan yang diakui:
> 1. **Sample efficiency rendah**: Setiap step membutuhkan ~2 menit interaksi nyata, sehingga 1000 step = ~33 jam. Deep RL dengan replay buffer bisa lebih efisien, tapi Q-Learning tabular memerlukan banyak kunjungan per state
> 2. **State space Q-Learning kontinu sangat besar**: State key menggunakan float kontinu, sehingga Q-table tumbuh cepat tapi jarang revisit state yang sama. Ini membuat konvergensi lambat
> 3. **Tidak ada transfer learning**: Jika konfigurasi cluster berubah (pod resource limit, jenis workload), model harus dilatih ulang
> 4. **Default action untuk state baru**: Prediction default ke minimum replicas untuk state yang tidak dikenali, yang bisa menyebabkan under-provisioning sementara
> 5. **Bergantung pada kualitas metrik**: Jika Prometheus mengalami delay atau data gap, metrik yang dikumpulkan bisa misleading dan agent belajar dari sinyal yang salah

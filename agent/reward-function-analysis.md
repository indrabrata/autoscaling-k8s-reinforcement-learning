# Analisis Detail Reward Function: Autoscaling Kubernetes dengan Reinforcement Learning

## Daftar Isi

- [1. Gambaran Umum](#1-gambaran-umum)
- [2. Pre-processing Metrik](#2-pre-processing-metrik)
- [3. Empat State Scoring](#3-empat-state-scoring)
- [4. Bobot (Weights)](#4-bobot-weights)
- [5. Kontribusi Positif vs Negatif](#5-kontribusi-positif-vs-negatif)
- [6. Neutral Baseline — Mengatasi Sparse Reward](#6-neutral-baseline--mengatasi-sparse-reward)
- [7. Formula Base Reward](#7-formula-base-reward)
- [8. Cost Penalty](#8-cost-penalty)
- [9. Trend-based Modifiers](#9-trend-based-modifiers)
- [10. Achievement Bonuses](#10-achievement-bonuses)
- [11. Properti: Tidak Ada Clamping](#11-properti-tidak-ada-clamping)
- [12. Alur Perhitungan End-to-End](#12-alur-perhitungan-end-to-end)
- [13. Contoh Perhitungan Numerik](#13-contoh-perhitungan-numerik)
- [14. Rekomendasi Jawaban Sidang Skripsi](#14-rekomendasi-jawaban-sidang-skripsi)

---

## 1. Gambaran Umum

Reward function didefinisikan pada method `_calculate_reward_qlearning()` di file `environment.py` (line 293–637). Fungsi ini digunakan oleh **kedua algoritma** (Q-Learning murni dan Q-Learning Fuzzy). Perbedaan kedua algoritma **bukan** pada reward, melainkan pada representasi state:

- **Q-Learning:** menyimpan Q-value untuk state kontinu mentah (CPU=45.5%, MEM=60.2%, ...)
- **Q-Fuzzy:** menyimpan Q-value untuk state yang sudah di-fuzzifikasi (CPU=medium, MEM=high, ...)

Dengan menggunakan reward function yang **identik**, perbandingan kedua algoritma menjadi **fair** — satu-satunya variabel adalah abstraksi state.

### Formula Keseluruhan

```
Reward = Base_Reward
       - Cost_Penalty
       + Proactive_Bonus
       - Reactive_Penalty
       - Oscillation_Penalty
       + Request_Rate_Bonus
       + Achievement_Bonus
```

Di mana:

```
Base_Reward = positive_contribution / (1 + negative_contribution)    jika negative > 0
            = positive_contribution                                   jika negative = 0
```

---

## 2. Pre-processing Metrik

Sebelum menghitung reward, tiga metrik utama dinormalisasi (line 326–340):

### 2.1 Response Time Percentage

```python
response_time_percentage = (response_time / max_response_time) * 100
```

Mengubah response time menjadi persentase terhadap target SLA. Jika `max_response_time = 100ms`:

| Response Time Aktual | Persentase | Interpretasi |
|---------------------|-----------|--------------|
| 50 ms | 50% | Excellent |
| 80 ms | 80% | Good |
| 100 ms | 100% | Batas SLA |
| 120 ms | 120% | Melanggar SLA |

### 2.2 Replica Ratio

```python
replica_ratio = (replica_state - min_replicas) / range_replicas
```

Normalisasi jumlah replica ke rentang [0, 1]. Contoh dengan `min=1, max=12`:

| Replicas | Ratio | Interpretasi |
|----------|-------|-------------|
| 1 | 0.00 | Minimum |
| 4 | 0.27 | Rendah |
| 7 | 0.55 | Menengah |
| 12 | 1.00 | Maksimum |

### 2.3 Request Rate Normalized

```python
current_capacity = per_pod_capacity * replica_state
request_rate_normalized = min(100, (request_rate / current_capacity) * 100)
```

Mengukur utilisasi kapasitas cluster. Contoh dengan `per_pod_capacity = 80 RPS`:

| Replicas | Kapasitas Total | RPS Aktual | Normalized | Interpretasi |
|----------|----------------|------------|------------|-------------|
| 2 | 160 | 80 | 50% | Baik |
| 4 | 320 | 240 | 75% | Tinggi |
| 4 | 320 | 304 | 95% | Saturating |

---

## 3. Empat State Scoring

Reward function mengklasifikasi kondisi sistem ke 4 kategori skor. Setiap skor bernilai antara 0.0 sampai 1.0.

### 3.1 Optimal Score (line 342–363)

Mengukur **seberapa baik sistem berjalan**. Kondisi terbaik: resource terpakai dalam rentang target, response time rendah, request rate terkendali.

```
Variabel yang diperiksa:
- cpu_in_range    : min_cpu <= CPU <= max_cpu         (contoh: 10% <= CPU <= 90%)
- mem_in_range    : min_memory <= MEM <= max_memory   (contoh: 10% <= MEM <= 90%)
- resp_excellent  : RT_percentage <= 60%
- resp_good       : RT_percentage <= 80%
- req_rate_good   : request_rate_normalized <= 70%
```

| Tier | Kondisi | Skor | Interpretasi |
|------|---------|------|-------------|
| 1 | cpu_in_range AND mem_in_range AND resp_excellent AND req_rate_good | **1.0** | Sempurna — semua metrik di zona ideal |
| 2 | cpu_in_range AND mem_in_range AND resp_good AND req_rate_good | **0.8** | Sangat baik — RT sedikit lebih tinggi tapi masih aman |
| 3 | cpu_in_range AND mem_in_range AND resp_good | **0.7** | Baik — request rate mungkin tinggi tapi resource masih aman |
| 4 | CPU/MEM hingga 110% max AND resp_good | **0.5** | Cukup — mendekati batas tapi masih bisa diterima |
| 5 | Tidak memenuhi kondisi di atas | **0.0** | Tidak optimal |

### 3.2 Balanced Score (line 365–383)

Mengukur **stabilitas operasi**. Mencari "sweet spot" di mana utilisasi tidak terlalu tinggi dan tidak terlalu rendah.

```
Variabel yang diperiksa:
- cpu_moderate       : 40 <= CPU <= 70
- mem_moderate       : 40 <= MEM <= 70
- resp_good          : RT_percentage <= 80%
- resp_acceptable    : RT_percentage <= 90%
- req_rate_sustainable : request_rate_normalized <= 80%
```

| Tier | Kondisi | Skor | Interpretasi |
|------|---------|------|-------------|
| 1 | cpu_moderate AND mem_moderate AND resp_good AND req_rate_sustainable | **1.0** | Operasi ideal — semua metrik di zona stabil |
| 2 | cpu_moderate AND mem_moderate AND resp_acceptable | **0.7** | Stabil — RT sedikit tinggi tapi resource terkontrol |
| 3 | CPU [30–80] AND MEM [30–80] AND resp_acceptable | **0.5** | Cukup stabil — rentang lebih lebar |
| 4 | Tidak memenuhi kondisi di atas | **0.0** | Tidak balanced |

### 3.3 Wasteful Score (line 385–419)

Mengukur **pemborosan resource** (over-provisioning). Skor tinggi berarti terlalu banyak replica untuk beban yang rendah.

```
Variabel yang diperiksa:
- cpu_very_low    : CPU < min_cpu * 0.5     (contoh: CPU < 5%)
- mem_very_low    : MEM < min_memory * 0.5  (contoh: MEM < 5%)
- cpu_low         : CPU < min_cpu           (contoh: CPU < 10%)
- mem_low         : MEM < min_memory        (contoh: MEM < 10%)
- req_rate_very_low : request_rate_normalized < 30%
```

| Tier | Kondisi | Skor | Interpretasi |
|------|---------|------|-------------|
| 1 | cpu_very_low AND mem_very_low AND resp_excellent AND req_rate_very_low | **1.0** | Pemborosan parah — semua metrik sangat rendah |
| 2 | (cpu_very_low OR mem_very_low) AND resp_excellent AND req_rate_very_low | **0.9** | Pemborosan tinggi — salah satu resource sangat rendah |
| 3 | cpu_low AND mem_low AND resp_good AND req_rate < 50% | **0.7** | Pemborosan menengah |
| 4 | (cpu_low OR mem_low) AND resp_excellent | **0.5** | Pemborosan ringan |
| 5 | Tidak memenuhi kondisi di atas | **0.0** | Tidak wasteful |

#### Mekanisme Waive di MIN_REPLICAS (line 412–419)

```python
if wasteful_score > 0 and replica_state <= min_replicas:
    wasteful_score = 0.0  # Dibatalkan
```

**Alasan:** Jika agent sudah di jumlah replica minimum (misalnya 1 pod) dan load rendah, agent tidak bisa melakukan apa-apa lagi untuk mengurangi pemborosan. Memberikan penalti pada kondisi yang tidak bisa diperbaiki oleh agent akan menyebabkan **unactionable negative reward** yang kontraproduktif untuk learning.

### 3.4 Critical Score (line 421–443)

Mengukur **tingkat bahaya sistem** (under-provisioning / pelanggaran SLA).

```
Variabel yang diperiksa:
- cpu_very_high    : CPU > max_cpu * 1.1    (contoh: CPU > 99%)
- mem_very_high    : MEM > max_memory * 1.1 (contoh: MEM > 99%)
- cpu_high         : CPU > max_cpu          (contoh: CPU > 90%)
- mem_high         : MEM > max_memory       (contoh: MEM > 90%)
- resp_critical    : RT_percentage > 120%
- resp_high        : RT_percentage > 100%
- req_rate_saturating : request_rate_normalized > 90%
```

| Tier | Kondisi | Skor | Interpretasi |
|------|---------|------|-------------|
| 1 | resp_critical OR req_rate_saturating | **1.0** | Sangat kritis — SLA dilanggar atau kapasitas habis |
| 2 | (cpu_very_high OR mem_very_high) AND resp_high | **1.0** | Sangat kritis — resource maxed + RT tinggi |
| 3 | resp_high AND (cpu_high OR mem_high) | **0.9** | Kritis — RT melebihi SLA dengan resource tinggi |
| 4 | req_rate > 85% AND (cpu_high OR mem_high) | **0.8** | Berbahaya — mendekati saturasi |
| 5 | cpu_very_high OR mem_very_high | **0.7** | Warning — resource sangat tinggi |
| 6 | resp_high (saja) | **0.6** | Warning — RT melebihi SLA |
| 7 | Tidak memenuhi kondisi di atas | **0.0** | Tidak critical |

---

## 4. Bobot (Weights)

Tiga bobot utama didefinisikan di `__init__` (line 76–78):

```python
response_time_weight = 1.0    # Prioritas tertinggi
cpu_memory_weight    = 0.5    # Prioritas menengah
cost_weight          = 0.3    # Prioritas terendah
```

### Filosofi Desain

```
Response Time (SLA)  >  Utilisasi Resource  >  Efisiensi Biaya
      1.0                    0.5                   0.3
```

Rasio ini mencerminkan prioritas di production environment:
1. **User experience** (response time) adalah yang paling kritis — langsung berdampak pada pelanggan
2. **Utilisasi resource** penting untuk memastikan sistem berjalan efisien
3. **Biaya** (jumlah replica) adalah pertimbangan terakhir — lebih baik sedikit boros daripada SLA dilanggar

### Penggunaan Bobot dalam Perhitungan

```python
wasteful_penalty = wasteful_score * cost_weight * 1.5
                 = wasteful_score * 0.3 * 1.5
                 = wasteful_score * 0.45

critical_penalty = critical_score * (response_time_weight + cpu_memory_weight)
                 = critical_score * (1.0 + 0.5)
                 = critical_score * 1.5
```

Dari sini terlihat bahwa critical penalty **3.33x lebih berat** dari wasteful penalty pada skor yang sama (1.5 vs 0.45). Ini konsisten dengan filosofi: melanggar SLA jauh lebih buruk daripada memboroskan resource.

---

## 5. Kontribusi Positif vs Negatif

### 5.1 Positive Contribution (line 446–453)

```python
optimal_contribution  = optimal_score * 1.0
balanced_contribution = balanced_score * 0.7

positive_contribution = optimal_contribution + balanced_contribution
```

Optimal diberi bobot lebih tinggi (1.0) dari balanced (0.7) karena mencapai state optimal lebih bernilai daripada sekadar balanced.

**Rentang nilai:** 0.0 sampai 1.7 (jika optimal=1.0 dan balanced=1.0)

### 5.2 Negative Contribution (line 448–454)

```python
wasteful_penalty = wasteful_score * 0.3 * 1.5    # max = 0.45
critical_penalty = critical_score * (1.0 + 0.5)   # max = 1.50

negative_contribution = wasteful_penalty + critical_penalty
```

**Rentang nilai:** 0.0 sampai 1.95 (jika wasteful=1.0 dan critical=1.0)

---

## 6. Neutral Baseline — Mengatasi Sparse Reward

### 6.1 Masalah Sparse Reward

Banyak state yang tidak masuk kategori optimal, balanced, wasteful, maupun critical — semua skor = 0. Tanpa intervensi, state-state ini menghasilkan reward = 0, dan agent tidak belajar apa-apa dari pengalaman tersebut.

### 6.2 Solusi: Baseline Reward (line 464–499)

Jika `positive_contribution == 0` DAN `negative_contribution == 0`:

**Kasus 1 — Di MIN_REPLICAS dengan RT baik:**

```python
if at_min_replicas and resp_good:
    neutral_baseline = 0.3
    positive_contribution = 0.3
```

Agent sudah melakukan hal yang benar (menggunakan replica minimum saat load rendah), jadi diberi reward positif.

**Kasus 2 — State acceptable:**

```python
cpu_acceptable  = 20 <= CPU <= 80
mem_acceptable  = 20 <= MEM <= 80
resp_acceptable = RT_percentage <= 100%
req_acceptable  = request_rate_normalized <= 85%

if semua_acceptable:
    neutral_baseline = 0.2
    positive_contribution = 0.2
```

### 6.3 Gradient Reward yang Dihasilkan

```
Negatif (buruk) → 0 (tidak terkategori) → 0.2 (acceptable) → 0.3 (min replicas)
                → 0.5+ (balanced/optimal) → 1.0+ (sangat optimal)
```

Gradient ini membantu agent membedakan kualitas state secara bertahap, bukan binary (baik/buruk).

---

## 7. Formula Base Reward

### 7.1 Dampening Function (line 501–505)

```python
if negative_contribution > 0:
    reward = positive_contribution / (1.0 + negative_contribution)
else:
    reward = positive_contribution
```

### 7.2 Kenapa Bukan Subtraction (positive - negative)?

| Aspek | `P / (1 + N)` (dipakai) | `P - N` (alternatif) |
|-------|--------------------------|----------------------|
| Sifat | Non-linear, asimptotik | Linear |
| Jika N sangat besar | Mendekati 0 dari atas | Bisa sangat negatif |
| Gradient | Smooth, menurun landai | Konstan |
| Risiko divergence | Rendah | Tinggi (Q-value meledak) |

**Contoh perbandingan:**

| P | N | `P/(1+N)` | `P-N` |
|---|---|-----------|-------|
| 1.0 | 0.0 | 1.000 | 1.000 |
| 1.0 | 0.5 | 0.667 | 0.500 |
| 1.0 | 1.0 | 0.500 | 0.000 |
| 1.0 | 1.5 | 0.400 | -0.500 |
| 1.0 | 3.0 | 0.250 | -2.000 |
| 0.5 | 1.5 | 0.200 | -1.000 |

Formula pembagian memberikan **bounded decay** — reward menurun tapi tidak meledak ke negatif. Penalti negatif yang terkontrol ditambahkan secara terpisah melalui cost penalty dan trend modifiers.

---

## 8. Cost Penalty

Cost penalty menginsentifkan efisiensi jumlah replica (line 507–521).

### 8.1 Tiga Skenario

**Skenario 1 — Wasteful + Replica Tinggi (penalti berat):**

```python
if wasteful_score > 0.5 and replica_ratio > 0.6:
    cost_factor = 1.8
    cost_pen = cost_weight * 1.8 * replica_ratio
    reward -= cost_pen * 0.3
```

Contoh: `wasteful=0.7, replica_ratio=0.8` → `penalti = 0.3 * 1.8 * 0.8 * 0.3 = 0.1296`

**Skenario 2 — Critical + Replica Tinggi (penalti ringan):**

```python
elif critical_score > 0.5 and replica_ratio > 0.5:
    cost_factor = 0.2
    cost_pen = cost_weight * 0.2 * replica_ratio
    reward -= cost_pen * 0.1
```

Contoh: `critical=0.9, replica_ratio=0.8` → `penalti = 0.3 * 0.2 * 0.8 * 0.1 = 0.0048`

**Logika:** Saat sistem critical, replica tinggi **dimaklumi** karena diperlukan untuk performa.

**Skenario 3 — Default:**

```python
else:
    cost_factor = 1.0
    cost_pen = cost_weight * 1.0 * replica_ratio * 0.2
    reward -= cost_pen
```

Contoh: `replica_ratio=0.5` → `penalti = 0.3 * 1.0 * 0.5 * 0.2 = 0.03`

### 8.2 Rangkuman Dampak Cost Penalty

| Kondisi | Multiplier Efektif | Dampak pada Reward |
|---------|-------------------|-------------------|
| Wasteful + banyak replica | `cost_weight * 1.8 * ratio * 0.3` | Berat (dorong scale down) |
| Critical + banyak replica | `cost_weight * 0.2 * ratio * 0.1` | Sangat ringan (izinkan banyak replica) |
| Normal | `cost_weight * ratio * 0.2` | Moderat |

---

## 9. Trend-based Modifiers

Trend analysis memungkinkan agent belajar **anticipatory behavior** — bertindak proaktif, bukan reaktif (line 523–561).

### 9.1 Proactive Bonus

Memberikan reward ketika agent **scaling searah** dengan tren request:

| Kondisi | Bonus | Interpretasi |
|---------|-------|-------------|
| request_trend = "up" AND action_trend = "up" | **+0.04** | Agent menambah replica saat load naik |
| request_trend = "down" AND action_trend = "down" | **+0.03** | Agent mengurangi replica saat load turun |

Bonus untuk scale up (+0.04) lebih besar dari scale down (+0.03) karena **gagal scale up** risikonya lebih tinggi (SLA violation) daripada **gagal scale down** (sedikit boros).

### 9.2 Reactive Penalty

Menghukum agent yang **terlambat** atau **salah arah** saat merespons tren:

| Kondisi | Penalti | Interpretasi |
|---------|---------|-------------|
| request naik + ReqRate > 75% + action stable | **-0.05** | Agent diam saat seharusnya scale up |
| request naik + ReqRate > 75% + action turun | **-0.08** | Agent scale down saat load naik (kontraproduktif) |

Penalti scale down saat load naik (-0.08) adalah **penalti trend terbesar**, karena ini keputusan yang paling berbahaya.

### 9.3 Oscillation Penalty

```python
if action_trend == "stable" AND balanced_score < 0.3 AND critical_score > 0.5:
    oscillation_penalty = 0.02
```

Menghukum agent yang **tidak bertindak** saat sistem dalam kondisi critical dan tidak balanced. "Stable" di sini berarti agent tidak melakukan perubahan scaling — yang salah jika sistem sedang bermasalah.

### 9.4 Request Rate Bonus

```python
if 40% <= request_rate_normalized <= 70% AND request_trend == "stable":
    req_rate_bonus = 0.04
```

Memberikan bonus ketika utilisasi berada di **sweet spot** (40–70%) dan stabil. Ini zona di mana resource terpakai efisien tanpa risiko saturasi.

### 9.5 Rangkuman Seluruh Modifiers

| Modifier | Rentang | Tujuan |
|----------|---------|--------|
| Proactive bonus | +0.03 s/d +0.04 | Reward scaling proaktif |
| Reactive penalty | -0.05 s/d -0.08 | Hukum respons lambat/salah |
| Oscillation penalty | -0.02 | Hukum inaction saat critical |
| Request rate bonus | +0.04 | Reward utilisasi sweet spot |

---

## 10. Achievement Bonuses

Bonus tambahan untuk kombinasi kondisi yang sangat diinginkan (line 563–569):

### 10.1 Optimal + Efisien

```python
if optimal_score > 0.6 and 0.3 <= replica_ratio <= 0.7:
    reward += 0.05
```

Agent mencapai state optimal **dengan jumlah replica yang efisien** (bukan terlalu sedikit, bukan terlalu banyak). Ini mengarahkan agent ke solusi yang **pareto-optimal** — performa baik dengan biaya wajar.

### 10.2 Balanced Stabil

```python
if balanced_score > 0.7:
    reward += 0.03
```

Bonus untuk mempertahankan kondisi balanced yang tinggi. Mendorong **stabilitas** operasi.

---

## 11. Properti: Tidak Ada Clamping

```python
# NOTE: No clamping - preserve actual reward values including:
# - Very small positive values (e.g., 0.0000001)
# - Negative values (important learning signal for bad states)
# - Values > 1.0 (rare but possible for exceptional performance)
```

### Kenapa Tidak Di-clamp ke [0, 1]?

1. **Reward negatif** adalah sinyal penting bahwa state buruk — membantu agent menghindari state tersebut
2. **Reward > 1.0** (jarang) menandakan performa luar biasa — memberikan insentif kuat
3. **Reward sangat kecil** (mendekati 0) membedakan state marginal dari state netral
4. **Clamping menghilangkan informasi** — perbedaan antara reward -0.5 dan -2.0 bermakna untuk learning

---

## 12. Alur Perhitungan End-to-End

```
┌─────────────────────────────────────────────────────┐
│  INPUT: CPU, Memory, Response Time, Request Rate,   │
│         Replicas, Request Trend, Action Trend        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STEP 1: Normalisasi Metrik                         │
│  - response_time_percentage = (RT / max_RT) * 100   │
│  - replica_ratio = (replicas - min) / range         │
│  - request_rate_normalized = (RPS / capacity) * 100 │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STEP 2: Hitung 4 State Scores                      │
│  - optimal_score   : [0.0 - 1.0]                   │
│  - balanced_score   : [0.0 - 1.0]                   │
│  - wasteful_score   : [0.0 - 1.0] (waive di min)   │
│  - critical_score   : [0.0 - 1.0]                   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STEP 3: Hitung Kontribusi                          │
│  positive = optimal * 1.0 + balanced * 0.7          │
│  negative = wasteful * 0.45 + critical * 1.5        │
│                                                     │
│  Jika keduanya = 0 → tambah neutral baseline        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STEP 4: Base Reward                                │
│  reward = positive / (1 + negative)                 │
│         atau                                        │
│  reward = positive        (jika negative = 0)       │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STEP 5: Kurangi Cost Penalty                       │
│  reward -= f(cost_weight, replica_ratio, state)     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STEP 6: Trend Modifiers                            │
│  reward += proactive_bonus     (+0.03 / +0.04)      │
│  reward -= reactive_penalty    (-0.05 / -0.08)      │
│  reward -= oscillation_penalty (-0.02)              │
│  reward += req_rate_bonus      (+0.04)              │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STEP 7: Achievement Bonuses                        │
│  optimal > 0.6 + replica efisien  → +0.05           │
│  balanced > 0.7                   → +0.03           │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  OUTPUT: reward (float, tidak di-clamp)             │
│          + breakdown dictionary (untuk logging)     │
└─────────────────────────────────────────────────────┘
```

---

## 13. Contoh Perhitungan Numerik

### Contoh 1: State Optimal

```
Input:
  CPU = 55%, MEM = 60%, RT = 45ms (45%), RPS = 200, Replicas = 4
  per_pod_capacity = 80, min_replicas = 1, max_replicas = 12
  request_trend = "stable", action_trend = "stable"

Normalisasi:
  response_time_percentage = (45 / 100) * 100 = 45%
  replica_ratio = (4 - 1) / 11 = 0.273
  current_capacity = 80 * 4 = 320
  request_rate_normalized = (200 / 320) * 100 = 62.5%

State Scores:
  optimal:  CPU [10-90]✓ MEM [10-90]✓ RT<=60%✓ ReqRate<=70%✓ → 1.0
  balanced: CPU [40-70]✓ MEM [40-70]✓ RT<=80%✓ ReqRate<=80%✓ → 1.0
  wasteful: CPU > min✓ → 0.0
  critical: RT < 100%✓ → 0.0

Kontribusi:
  positive = 1.0 * 1.0 + 1.0 * 0.7 = 1.7
  negative = 0.0

Base Reward:
  reward = 1.7 (negative = 0)

Cost Penalty (default):
  cost_pen = 0.3 * 1.0 * 0.273 * 0.2 = 0.0164
  reward = 1.7 - 0.0164 = 1.6836

Trend Modifiers:
  proactive = 0 (both stable)
  reactive = 0
  oscillation = 0
  req_rate_bonus = 0.04 (62.5% in [40-70] + stable)
  reward = 1.6836 + 0.04 = 1.7236

Achievement Bonuses:
  optimal(1.0) > 0.6 AND ratio(0.273) NOT in [0.3-0.7] → no bonus
  balanced(1.0) > 0.7 → +0.03
  reward = 1.7236 + 0.03 = 1.7536

FINAL REWARD = 1.7536
```

### Contoh 2: State Critical

```
Input:
  CPU = 95%, MEM = 88%, RT = 150ms (150%), RPS = 350, Replicas = 4
  request_trend = "up", action_trend = "stable"

Normalisasi:
  response_time_percentage = 150%
  replica_ratio = 0.273
  request_rate_normalized = (350 / 320) * 100 = 100% (capped)

State Scores:
  optimal:  RT > 60% → 0.0
  balanced: CPU > 70 → 0.0
  wasteful: CPU > min → 0.0
  critical: req_rate_saturating(100% > 90%) → 1.0

Kontribusi:
  positive = 0.0
  negative = 0 + 1.0 * 1.5 = 1.5

Neutral check: negative > 0, skip baseline

Base Reward:
  reward = 0.0 / (1.0 + 1.5) = 0.0

Cost Penalty (critical path):
  critical(1.0) > 0.5 BUT replica_ratio(0.273) < 0.5 → default path
  cost_pen = 0.3 * 1.0 * 0.273 * 0.2 = 0.0164
  reward = 0.0 - 0.0164 = -0.0164

Trend Modifiers:
  proactive = 0 (action not "up")
  reactive = 0.05 (request up + normalized > 75% + action stable)
  oscillation = 0.02 (action stable + balanced(0) < 0.3 + critical(1.0) > 0.5)
  req_rate_bonus = 0 (normalized > 70%)
  reward = -0.0164 - 0.05 - 0.02 = -0.0864

Achievement Bonuses:
  optimal(0.0) < 0.6 → no bonus
  balanced(0.0) < 0.7 → no bonus

FINAL REWARD = -0.0864
```

### Contoh 3: State Wasteful di MIN_REPLICAS

```
Input:
  CPU = 3%, MEM = 4%, RT = 20ms (20%), RPS = 5, Replicas = 1 (= min)
  request_trend = "stable", action_trend = "stable"

Normalisasi:
  response_time_percentage = 20%
  replica_ratio = 0.0
  request_rate_normalized = (5 / 80) * 100 = 6.25%

State Scores:
  optimal:  CPU < 10% (min_cpu) → out of range → 0.0
  balanced: CPU < 40 → 0.0
  wasteful: cpu_very_low(3% < 5%)✓ mem_very_low(4% < 5%)✓
            resp_excellent(20%)✓ req_rate_very_low(6.25%)✓ → 1.0
            TAPI: replica_state(1) <= min_replicas(1) → WAIVED ke 0.0
  critical: semua rendah → 0.0

Kontribusi:
  positive = 0.0, negative = 0.0

Neutral Baseline:
  at_min_replicas(1 <= 1)✓ AND resp_good(20% <= 80%)✓
  → neutral_baseline = 0.3
  positive = 0.3

Base Reward:
  reward = 0.3

Cost Penalty (default):
  cost_pen = 0.3 * 1.0 * 0.0 * 0.2 = 0.0
  reward = 0.3

Trend Modifiers: semua 0 (stable, score rendah)

Achievement Bonuses: tidak memenuhi

FINAL REWARD = 0.3
```

Tanpa mekanisme waive + baseline, reward akan menjadi 0.0 dan agent tidak belajar bahwa berada di minimum replica saat load rendah adalah perilaku yang benar.

---

## 14. Rekomendasi Jawaban Sidang Skripsi

### Q1: "Kenapa menggunakan reward shaping dengan 4 kategori, bukan formula matematika langsung?"

> Pendekatan rule-based scoring dengan 4 kategori (optimal, balanced, wasteful, critical) dipilih karena domain autoscaling memiliki **multi-objective** yang saling bertentangan — performa vs biaya. Formula matematika tunggal (misalnya weighted sum sederhana) sulit menangkap nuansa seperti: "CPU rendah + response time bagus bukan berarti optimal, melainkan wasteful." Dengan scoring berbasis kategori, semantik bisnis bisa didefinisikan secara eksplisit. Selain itu, pendekatan ini lebih **interpretable** — dari log, kita langsung tahu kenapa reward tinggi atau rendah melalui komponen O, B, W, C.

### Q2: "Bagaimana cara menentukan nilai bobot (1.0, 0.5, 0.3)?"

> Bobot mencerminkan **prioritas bisnis**: response time (1.0) > utilisasi resource (0.5) > efisiensi biaya (0.3). Ini mengikuti prinsip bahwa dalam production environment, SLA dan user experience adalah prioritas utama. Jika response time tinggi, penalti critical sebesar `critical_score * 1.5` jauh lebih besar dari penalti wasteful `wasteful_score * 0.45`. Nilai spesifik dipilih berdasarkan rasio proporsional dan validasi melalui eksperimen training — diobservasi apakah agent konvergen ke kebijakan yang masuk akal.

### Q3: "Kenapa formula `positive / (1 + negative)` bukan `positive - negative`?"

> Formula `positive / (1 + negative)` memberikan **dampening non-linear**. Dengan subtraction, reward bisa menjadi sangat negatif secara unbounded (misalnya -2.0), yang berisiko menyebabkan Q-value divergence — terutama berbahaya pada Q-Learning tabular di mana tidak ada mekanisme stabilisasi seperti di Deep RL. Dengan formula pembagian, reward terdampak secara **asimptotis**: semakin besar penalti, reward mendekati 0 dari sisi positif, memberikan gradien yang smooth. Komponen negatif yang terkontrol kemudian ditambahkan secara terpisah melalui cost penalty dan trend modifiers.

### Q4: "Kenapa wasteful penalty di-waive saat MIN_REPLICAS?"

> Ini mengatasi masalah **unactionable penalty**. Jika agent sudah di replica minimum dan load rendah, agent mendapat penalti wasteful padahal tidak ada aksi yang bisa memperbaiki kondisi tersebut (tidak bisa scale down lagi). Ini menyebabkan agent menerima sinyal negatif tanpa solusi, yang berpotensi membuat Q-value untuk state tersebut terus menurun tanpa batas bawah. Dengan waiving, agent belajar bahwa "berada di minimum replica saat load rendah" adalah perilaku yang benar, bukan kesalahan.

### Q5: "Apa fungsi neutral baseline 0.2 dan 0.3?"

> Neutral baseline mengatasi **sparse reward problem** yang umum di reinforcement learning. Banyak state jatuh di antara kategori — tidak optimal, tidak balanced, tidak wasteful, tidak critical. Tanpa baseline, state-state ini menghasilkan reward = 0 dan agent tidak memiliki gradien untuk belajar. Nilai 0.3 (di min replicas, RT baik) lebih tinggi dari 0.2 (acceptable) karena berada di minimum replica saat aman adalah **keputusan aktif yang benar**, sedangkan state acceptable hanyalah kondisi yang "tidak buruk." Gradient reward yang dihasilkan: negatif → 0 → 0.2 → 0.3 → 0.5+ → 1.0+.

### Q6: "Kenapa ada trend-based bonus/penalty? Bukankah reward seharusnya hanya berdasarkan state saat ini?"

> Dalam Markov Decision Process (MDP) murni, reward memang hanya berdasarkan state saat ini. Namun autoscaling adalah masalah yang **inherently temporal** — scaling yang baik harus proaktif, bukan reaktif. Ketika request rate sedang naik, menunggu sampai sistem critical baru scale up sudah terlambat (ada delay propagasi pod). Trend-based modifier menerapkan konsep **reward shaping** (Ng et al., 1999), menambahkan sinyal tambahan yang mempercepat konvergensi tanpa mengubah optimal policy. Dengan magnitude kecil (+0.03 sampai -0.08 dibandingkan base reward 0.3–1.7), modifier ini berfungsi sebagai **auxiliary signal**, bukan pengubah kebijakan utama.

### Q7: "Apakah reward function ini bisa menyebabkan reward hacking?"

> Risiko reward hacking diminimalkan melalui beberapa mekanisme:
> 1. **Cost penalty** mencegah agent terus menambah replica untuk skor optimal (semakin banyak replica, semakin besar penalti biaya)
> 2. **Wasteful scoring** mendeteksi over-provisioning (jika resource rendah karena terlalu banyak replica, skor wasteful naik)
> 3. **Tidak ada clamping** sehingga sinyal negatif tetap tersampaikan secara proporsional
> 4. **Achievement bonus** memerlukan **kombinasi** kondisi (optimal > 0.6 DAN replica ratio [0.3–0.7]), bukan optimasi satu metrik saja
> 5. **Trend penalty** menghukum scaling yang tidak sesuai konteks (misalnya scale up terus-menerus tanpa load yang naik tidak mendapat proactive bonus)

### Q8: "Kenapa reward tidak di-clamp ke [0, 1]? Apakah ini tidak menyebabkan masalah pada Q-Learning?"

> Tidak di-clamp karena **informasi magnitude penting** untuk learning. Reward -0.08 (critical parah) harus dibedakan dari -0.01 (sedikit suboptimal) — clamping ke 0 menghilangkan perbedaan ini. Pada Q-Learning tabular, Q-value update menggunakan formula `Q(s,a) += lr * (reward + gamma * max_Q(s') - Q(s,a))`. Selama learning rate cukup kecil dan reward tidak meledak (yang terjamin karena formula dampening), Q-values tetap stabil. Dalam eksperimen, rentang reward yang diobservasi sekitar [-0.1, 1.8], yang masih dalam rentang wajar untuk konvergensi Q-Learning.

### Q9: "Bagaimana reward function ini dibandingkan dengan HPA (Horizontal Pod Autoscaler) bawaan Kubernetes?"

> HPA menggunakan pendekatan **reactive threshold-based**: jika CPU > target, tambah replica berdasarkan formula `desiredReplicas = ceil(currentReplicas * (currentMetric / targetMetric))`. HPA hanya melihat satu metrik (CPU atau memory), tidak mempertimbangkan response time atau request rate, dan tidak memiliki konsep biaya. Reward function yang dibangun memungkinkan agent mempelajari kebijakan yang **multi-objective** (mempertimbangkan 4+ metrik sekaligus) dan **proaktif** (melalui trend analysis). Ini adalah keunggulan fundamental RL dibanding rule-based autoscaling.

### Q10: "Apakah ada kelemahan dari desain reward function ini?"

> Ada beberapa kelemahan yang diakui:
> 1. **Threshold bersifat manual** — nilai-nilai seperti 60%, 80%, 40-70% ditentukan berdasarkan domain knowledge, bukan learned secara otomatis. Perubahan beban kerja mungkin memerlukan re-tuning.
> 2. **Sensitivitas terhadap per_pod_capacity** — jika nilai kapasitas per pod tidak akurat (dari load testing), request_rate_normalized akan misleading, dan seluruh scoring terdampak.
> 3. **Trend window tetap** — action_trend_window = 5 mungkin tidak optimal untuk semua pola beban. Beban yang berubah sangat cepat atau sangat lambat mungkin butuh window berbeda.
> 4. **Reward function yang kompleks** memerlukan lebih banyak iterasi training untuk konvergensi dibanding reward sederhana, karena agent harus mempelajari interaksi banyak komponen.

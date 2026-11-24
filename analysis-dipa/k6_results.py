"""
k6 end-of-test summaries, transcribed from `data/testing/<dataset>/<run>/image.png`.

k6 writes its summary to the terminal only, so the screenshot in each run folder is
the sole record of the client-side numbers. They are transcribed here once, rather
than in each notebook, so `testing_analysis.ipynb` (one repetition) and
`aggregate_analysis.ipynb` (every repetition) can never disagree.

All durations are milliseconds, taken from `http_req_duration` over every request of
both scenarios in that k6 process. This is NOT the same measurement as the
`response_time_ms` column of the CSVs: k6 measures client-side and weights by
request, while the CSV holds the mean of the per-endpoint P90s that Prometheus
reported for one ~15 s observation window. The two are not interchangeable.

Datasets
--------
There are two collection campaigns, and their run folders reuse the same names
(`1`, `2`, `3` and `hpa` exist in both), so EVERY lookup here is keyed by dataset
first and run folder second:

    60-episodes    batches A, B, E, F, G - the original campaign plus later
                   single-day follow-up runs
    60-episodes-2  batch C - the rerun, and the one the notebooks default to; plus
                   later follow-up batches E, F, G on a different trained policy

Pass the dataset directory name (the last component of the notebook's `DATA_DIR`,
via `dataset_of`) to `k6_frame()` / `k6_for()`, and index `BATCHES[dataset][run]`,
`RL_RUNS[dataset][batch]`, `HPA_RUNS[dataset][batch]`.

Repetition structure
--------------------
Each RL run folder holds one k6 process per method, run back to back. Each HPA
folder instead holds three k6 processes running at once, one per HPA-managed
deployment, matching the three `metrics_*.csv` files in that folder; `instance`
distinguishes them, in the pane order of the screenshot. The one exception is
`60-episodes/hpa-3`, a single-process baseline carried over from an earlier
campaign - see EXCLUDED_RUNS.

Batches are collection sessions, not repetitions: pooling across them hides real
shifts in cluster conditions. In `60-episodes`, batch B is markedly faster at the
median (~27-65 ms vs ~127-290 ms) and at P90 than batch A - compare within a batch,
and treat a combined mean as a summary rather than as six samples of one population.
"""

from typing import Dict, List

# k6 threshold under test, and the agents' MAX_RESPONSE_TIME (agent-dipa/.env).
SLO_MS = 1000.0

# Dataset directory under `data/testing/`, newest last.
DATASETS: List[str] = ["60-episodes", "60-episodes-2"]

# dataset -> run folder -> collection batch.
BATCHES: Dict[str, Dict[str, str]] = {
    "60-episodes": {
        "1": "A",
        "2": "A",
        "3": "A",
        "hpa": "A",
        "1.1": "B",
        "2.1": "B",
        "3.1": "B",
        "hpa-2": "B",
        # Batch D is an HPA baseline and nothing else: the folder is a copy of the
        # 2026-02-22 run from analysis/data-testing/19-02-2026/hpa (byte-identical
        # CSV), dropped in here for reference. It has no RL counterpart, so there is
        # no D entry in RL_RUNS.
        "hpa-3": "D",
        # Batches E and F are single extra RL repetitions, one per collection day
        # (2026-08-24 and 2026-08-26), after the A/B campaign and its own analysis
        # pass. Kept as their own batches rather than folded into B: both are days
        # later than B, and each is a single run with no same-day sibling in this
        # dataset to compare conditions against.
        "4": "E",
        "5": "F",
        "6": "G",
    },
    # Runs 1-5 were collected on 2026-08-21 and run 6 plus the HPA baseline on the
    # morning of 2026-08-22. They are kept as ONE batch because the two days agree
    # closely on the client-side median (~28-50 ms in every process) and on P90,
    # unlike the A/B split above where conditions genuinely moved.
    "60-episodes-2": {
        "1": "C",
        "2": "C",
        "3": "C",
        "4-anomaly": "C",
        "5": "C",
        "6": "C",
        "hpa": "C",
        # Same 2026-08-21 collection session as 60-episodes/hpa-2 (byte-identical
        # CSVs and screenshot), so the same batch letter - but batch B has no RL
        # counterpart in THIS dataset (the batch-B RL runs are 60-episodes/1.1 etc,
        # which belong to the other campaign), so there is no B entry in RL_RUNS
        # for 60-episodes-2 and this baseline is left out of HPA_RUNS too, same
        # reasoning as batch D below.
        "hpa-2": "B",
        # Same 2026-02-22 collection session as 60-episodes/hpa-3, so the same
        # batch: the folder is copied into both datasets, not collected twice.
        "hpa-3": "D",
        # Batches E (2026-08-24) and F (2026-08-26) are two more collection
        # sessions, each with its own HPA baseline - same split as 60-episodes/4
        # vs 5. Every run and baseline in E/F loads its models from a `model-2`
        # folder, unlike every batch-C run (`model`), so treat E/F as a different
        # trained policy throughout, not more repetitions of runs 1-6.
        "7": "E",
        "8": "E",
        "hpa-4": "E",
        "9": "F",
        "hpa-5": "F",
        "10": "G",
    },
}

BATCH_DATES: Dict[str, str] = {
    "A": "2026-08-07",
    "B": "2026-08-21",
    "C": "2026-08-21/22",
    "D": "2026-02-22",
    "E": "2026-08-24",
    "F": "2026-08-26",
    "G": "2026-08-31",
}

# dataset -> batch -> run folders holding the RL repetitions.
RL_RUNS: Dict[str, Dict[str, List[str]]] = {
    "60-episodes": {
        "A": ["1", "2", "3"],
        "B": ["1.1", "2.1", "3.1"],
        "E": ["4"],
        "F": ["5"],
        "G": ["6"],
    },
    # `4-anomaly` is deliberately absent: see EXCLUDED_RUNS. Its k6 summary is still
    # transcribed below, so k6_for('ql', '4-anomaly', '60-episodes-2') works and the
    # run can be pulled in explicitly when the anomaly itself is the subject.
    #
    # Batches E ('7', '8'), F ('9'), and G ('10') are their own batches rather than
    # folded into C: they load their models from a `model-2` folder, unlike runs
    # 1-6 (`model`), so they are a different trained policy and a mean over 1-6
    # plus 7-10 would silently compare two policies as if they were repetitions of
    # one.
    "60-episodes-2": {
        "C": ["1", "2", "3", "5", "6"],
        "E": ["7", "8"],
        "F": ["9"],
        "G": ["10"],
    },
}

# dataset -> batch -> the HPA baseline folder for that batch. This map is what
# aggregate_analysis.ipynb enumerates, so a folder listed here is pooled with the
# rest of its dataset - which is why batch D is deliberately absent; see
# EXCLUDED_RUNS.
HPA_RUNS: Dict[str, Dict[str, str]] = {
    "60-episodes": {"A": "hpa", "B": "hpa-2"},
    "60-episodes-2": {"C": "hpa", "E": "hpa-4", "F": "hpa-5"},
}

# Run folders present on disk but left out of RL_RUNS / HPA_RUNS, with the reason.
# Kept explicit so an aggregate that silently has one run fewer still explains
# itself. Their k6 summaries are still transcribed below, so k6_frame(dataset,
# runs=[...]) and testing_analysis.ipynb can pull one in when it IS the subject.
EXCLUDED_RUNS: Dict[str, Dict[str, str]] = {
    "60-episodes": {
        "hpa-3": (
            "HPA baseline copied in from the 2026-02-22 campaign "
            "(analysis/data-testing/19-02-2026/hpa), six months before batches A "
            "and B and on a cluster whose conditions are not recorded here; it also "
            "ran ONE k6 process where every other HPA folder ran three, so it "
            "carries no per-deployment spread. Pooling it into the 60-episodes "
            "aggregate would compare campaigns, not autoscalers"
        ),
    },
    "60-episodes-2": {
        "hpa-3": (
            "the same copied-in 2026-02-22 baseline as 60-episodes/hpa-3, filed "
            "under this dataset too; excluded here for the same reasons"
        ),
        "4-anomaly": (
            "flagged as anomalous when collected; Q-Learning and Q-Learning Crisp "
            "degrade to P90 3.24 s / 2.12 s, several times any other run in the "
            "batch, so it would dominate a mean over five repetitions"
        ),
    },
}

# Glob that picks each method's CSV out of a run folder. `test_qlearning_*` does not
# match `test_qlearningcrisp_*` or `test_qlearningfuzzy_*` - the trailing underscore
# separates them.
CSV_PATTERNS: Dict[str, str] = {
    "ql": "test_qlearning_*.csv",
    "qcrisp": "test_qlearningcrisp_*.csv",
    "qfuzzy": "test_qlearningfuzzy_*.csv",
    "hpa": "metrics_*.csv",
}

METHODS: List[str] = ["ql", "qcrisp", "qfuzzy", "hpa"]

# Glob that picks each agent's trained model out of data/model/<dataset>/, and the
# name that agent is registered under in agent-dipa/rl/factory.py. Same trailing
# underscore trick as CSV_PATTERNS: `qlearning_*` does not match `qlearningcrisp_*`.
# HPA is absent on purpose - it is a controller, not a learned policy, so it has no
# model to load and no Q-table to hit.
MODEL_PATTERNS: Dict[str, str] = {
    "ql": "qlearning_*.pkl",
    "qcrisp": "qlearningcrisp_*.pkl",
    "qfuzzy": "qlearningfuzzy_*.pkl",
}

AGENT_ALGORITHMS: Dict[str, str] = {
    "ql": "Q-LEARNING",
    "qcrisp": "Q-LEARNING-CRISP",
    "qfuzzy": "Q-LEARNING-FUZZY",
}

# k6 ramping-vus stages, PER SCENARIO, from SINGLE_CYCLE in
# agent-dipa/load-resource-intensive-ii/k6-train-patterns-loop.js. Two scenarios
# (cpu_stress and memory_stress) run at once, so total concurrency is twice these
# targets - a 150 VU peak, matching the vus_max = 150 every k6 summary reports.
LOAD_STAGES: List[Dict[str, int]] = [
    {"duration_min": 6, "target": 10},
    {"duration_min": 6, "target": 40},
    {"duration_min": 6, "target": 75},
    {"duration_min": 6, "target": 30},
    {"duration_min": 6, "target": 5},
]

METHOD_LABELS: Dict[str, str] = {
    "ql": "Q-Learning",
    "qcrisp": "Q-Learning Crisp",
    "qfuzzy": "Q-Learning Fuzzy",
    "hpa": "HPA",
}

# One record per k6 process.
#   dataset             : directory under data/testing/ - run names repeat across them
#   avg/med/p90/p95/max : http_req_duration, ms
#   reqs                : http_reqs
#   failed_pct          : http_req_failed, percent
K6_RESULTS: List[dict] = [
    # ====================== dataset 60-episodes ===============================
    # ---- batch A, 2026-08-07 -------------------------------------------------
    # run 1 (finished ~15:02)
    {"dataset": "60-episodes", "method": "ql", "run": "1", "instance": 1,
     "avg": 1100.00, "med": 266.41, "p90": 2680.0, "p95": 4830.0, "max": 60000.0,
     "reqs": 53862, "failed_pct": 0.72},
    {"dataset": "60-episodes", "method": "qcrisp", "run": "1", "instance": 1,
     "avg": 696.58, "med": 228.42, "p90": 1410.0, "p95": 2310.0, "max": 52050.0,
     "reqs": 66737, "failed_pct": 0.73},
    {"dataset": "60-episodes", "method": "qfuzzy", "run": "1", "instance": 1,
     "avg": 662.93, "med": 284.48, "p90": 1730.0, "p95": 2570.0, "max": 13630.0,
     "reqs": 68018, "failed_pct": 0.34},
    # run 2 (finished ~15:37)
    {"dataset": "60-episodes", "method": "ql", "run": "2", "instance": 1,
     "avg": 769.89, "med": 176.84, "p90": 1730.0, "p95": 3110.0, "max": 42840.0,
     "reqs": 63951, "failed_pct": 0.29},
    {"dataset": "60-episodes", "method": "qcrisp", "run": "2", "instance": 1,
     "avg": 547.29, "med": 186.81, "p90": 1400.0, "p95": 2360.0, "max": 12230.0,
     "reqs": 73100, "failed_pct": 0.46},
    {"dataset": "60-episodes", "method": "qfuzzy", "run": "2", "instance": 1,
     "avg": 819.55, "med": 145.40, "p90": 1580.0, "p95": 3800.0, "max": 44910.0,
     "reqs": 62289, "failed_pct": 0.43},
    # run 3 (finished ~16:11)
    {"dataset": "60-episodes", "method": "ql", "run": "3", "instance": 1,
     "avg": 525.07, "med": 137.12, "p90": 1110.0, "p95": 2590.0, "max": 25240.0,
     "reqs": 74181, "failed_pct": 0.35},
    {"dataset": "60-episodes", "method": "qcrisp", "run": "3", "instance": 1,
     "avg": 517.23, "med": 162.42, "p90": 1350.0, "p95": 2390.0, "max": 12410.0,
     "reqs": 74543, "failed_pct": 0.72},
    {"dataset": "60-episodes", "method": "qfuzzy", "run": "3", "instance": 1,
     "avg": 808.23, "med": 127.02, "p90": 1500.0, "p95": 3980.0, "max": 55470.0,
     "reqs": 62678, "failed_pct": 0.32},
    # HPA baseline (finished ~16:52), three concurrent k6 processes
    {"dataset": "60-episodes", "method": "hpa", "run": "hpa", "instance": 1,
     "avg": 1200.0, "med": 933.89, "p90": 2690.0, "p95": 3320.0, "max": 9550.0,
     "reqs": 51187, "failed_pct": 0.00},
    {"dataset": "60-episodes", "method": "hpa", "run": "hpa", "instance": 2,
     "avg": 1200.0, "med": 920.65, "p90": 2690.0, "p95": 3320.0, "max": 9460.0,
     "reqs": 51293, "failed_pct": 0.00},
    {"dataset": "60-episodes", "method": "hpa", "run": "hpa", "instance": 3,
     "avg": 1190.0, "med": 917.80, "p90": 2680.0, "p95": 3320.0, "max": 9550.0,
     "reqs": 51421, "failed_pct": 0.00},

    # ---- batch B, 2026-08-21 -------------------------------------------------
    # run 1.1 (started ~12:52)
    {"dataset": "60-episodes", "method": "ql", "run": "1.1", "instance": 1,
     "avg": 617.46, "med": 30.32, "p90": 1490.0, "p95": 3180.0, "max": 39830.0,
     "reqs": 70017, "failed_pct": 0.50},
    {"dataset": "60-episodes", "method": "qcrisp", "run": "1.1", "instance": 1,
     "avg": 515.12, "med": 58.20, "p90": 1360.0, "p95": 2610.0, "max": 30790.0,
     "reqs": 74671, "failed_pct": 0.43},
    {"dataset": "60-episodes", "method": "qfuzzy", "run": "1.1", "instance": 1,
     "avg": 672.96, "med": 27.30, "p90": 1090.0, "p95": 3300.0, "max": 48220.0,
     "reqs": 67738, "failed_pct": 0.33},
    # run 2.1 (started ~13:30)
    {"dataset": "60-episodes", "method": "ql", "run": "2.1", "instance": 1,
     "avg": 749.03, "med": 29.37, "p90": 1810.0, "p95": 3850.0, "max": 45890.0,
     "reqs": 64812, "failed_pct": 0.41},
    {"dataset": "60-episodes", "method": "qcrisp", "run": "2.1", "instance": 1,
     "avg": 424.32, "med": 48.86, "p90": 1210.0, "p95": 2320.0, "max": 14590.0,
     "reqs": 79402, "failed_pct": 0.49},
    {"dataset": "60-episodes", "method": "qfuzzy", "run": "2.1", "instance": 1,
     "avg": 700.81, "med": 27.53, "p90": 1370.0, "p95": 3780.0, "max": 54290.0,
     "reqs": 66632, "failed_pct": 0.36},
    # run 3.1 (started ~14:05) - the only k6 process in either batch that passed
    # the p(90)<1000 threshold: Q-Learning at 897.06 ms.
    {"dataset": "60-episodes", "method": "ql", "run": "3.1", "instance": 1,
     "avg": 602.81, "med": 27.26, "p90": 897.06, "p95": 3290.0, "max": 53040.0,
     "reqs": 70693, "failed_pct": 0.11},
    {"dataset": "60-episodes", "method": "qcrisp", "run": "3.1", "instance": 1,
     "avg": 455.09, "med": 64.51, "p90": 1290.0, "p95": 2330.0, "max": 15190.0,
     "reqs": 77733, "failed_pct": 0.54},
    {"dataset": "60-episodes", "method": "qfuzzy", "run": "3.1", "instance": 1,
     "avg": 724.51, "med": 27.36, "p90": 1300.0, "p95": 4000.0, "max": 49450.0,
     "reqs": 65768, "failed_pct": 0.35},
    # HPA baseline (started ~12:09), three concurrent k6 processes
    {"dataset": "60-episodes", "method": "hpa", "run": "hpa-2", "instance": 1,
     "avg": 279.85, "med": 27.84, "p90": 555.12, "p95": 1500.0, "max": 12760.0,
     "reqs": 88358, "failed_pct": 0.00},
    {"dataset": "60-episodes", "method": "hpa", "run": "hpa-2", "instance": 2,
     "avg": 283.41, "med": 27.85, "p90": 574.75, "p95": 1510.0, "max": 15330.0,
     "reqs": 88121, "failed_pct": 0.00},
    {"dataset": "60-episodes", "method": "hpa", "run": "hpa-2", "instance": 3,
     "avg": 290.76, "med": 27.82, "p90": 565.26, "p95": 1500.0, "max": 18080.0,
     "reqs": 87628, "failed_pct": 0.00},

    # ---- batch D, 2026-02-22 -------------------------------------------------
    # HPA baseline from the earlier campaign, transcribed from hpa-3/hpa-6.png. ONE
    # k6 process, not three: the folder holds a single metrics CSV and the summary
    # shows a single pane, so instance 1 is the whole run and there is no spread
    # between deployments to report. The only HPA process in either dataset that
    # passed p(90)<1000, at 303.91 ms - six months and an unknown cluster apart from
    # batches A/B, so it is a reference point, not a comparable baseline.
    {"dataset": "60-episodes", "method": "hpa", "run": "hpa-3", "instance": 1,
     "avg": 187.33, "med": 72.99, "p90": 303.91, "p95": 574.35, "max": 8430.0,
     "reqs": 95250, "failed_pct": 0.00},

    # ---- batch E, 2026-08-24 --------------------------------------------------
    # run 4 (started ~12:49) - no HPA process collected alongside it. None of the
    # three methods passed p(90)<1000.
    {"dataset": "60-episodes", "method": "ql", "run": "4", "instance": 1,
     "avg": 1380.0, "med": 545.19, "p90": 3900.0, "p95": 5870.0, "max": 23760.0,
     "reqs": 47434, "failed_pct": 0.00},
    {"dataset": "60-episodes", "method": "qcrisp", "run": "4", "instance": 1,
     "avg": 434.49, "med": 62.15, "p90": 1210.0, "p95": 2210.0, "max": 15290.0,
     "reqs": 78851, "failed_pct": 0.51},
    {"dataset": "60-episodes", "method": "qfuzzy", "run": "4", "instance": 1,
     "avg": 570.88, "med": 92.42, "p90": 1680.0, "p95": 3100.0, "max": 20850.0,
     "reqs": 72038, "failed_pct": 0.15},

    # ---- batch F, 2026-08-26 --------------------------------------------------
    # run 5 (started ~00:52) - no HPA process collected alongside it. Q-Learning
    # Crisp passed p(90)<1000.
    {"dataset": "60-episodes", "method": "ql", "run": "5", "instance": 1,
     "avg": 1380.0, "med": 528.09, "p90": 3860.0, "p95": 5890.0, "max": 23970.0,
     "reqs": 47471, "failed_pct": 0.00},
    {"dataset": "60-episodes", "method": "qcrisp", "run": "5", "instance": 1,
     "avg": 356.93, "med": 48.09, "p90": 869.51, "p95": 1980.0, "max": 19310.0,
     "reqs": 83357, "failed_pct": 0.39},
    {"dataset": "60-episodes", "method": "qfuzzy", "run": "5", "instance": 1,
     "avg": 512.08, "med": 77.63, "p90": 1410.0, "p95": 2780.0, "max": 15270.0,
     "reqs": 74839, "failed_pct": 0.23},

    # ---- batch G, 2026-08-31 --------------------------------------------------
    # run 6 (started ~02:57) - no HPA process collected alongside it. None of the
    # three methods passed p(90)<1000.
    {"dataset": "60-episodes", "method": "ql", "run": "6", "instance": 1,
     "avg": 1380.0, "med": 547.03, "p90": 3850.0, "p95": 5840.0, "max": 25240.0,
     "reqs": 47550, "failed_pct": 0.00},
    {"dataset": "60-episodes", "method": "qcrisp", "run": "6", "instance": 1,
     "avg": 398.18, "med": 54.11, "p90": 1090.0, "p95": 2190.0, "max": 13960.0,
     "reqs": 80900, "failed_pct": 0.57},
    {"dataset": "60-episodes", "method": "qfuzzy", "run": "6", "instance": 1,
     "avg": 554.41, "med": 97.61, "p90": 1620.0, "p95": 2800.0, "max": 16160.0,
     "reqs": 72785, "failed_pct": 0.12},

    # ====================== dataset 60-episodes-2 =============================
    # ---- batch C, 2026-08-21/22 ----------------------------------------------
    # Every process in this batch logs "Episodes: 30, Iteration: 30", not 60 - the
    # folder name is inherited from the earlier campaign, the agents are not.
    # run 1 (started ~14:46 on 2026-08-21) - Q-Learning Fuzzy passed p(90)<1000.
    {"dataset": "60-episodes-2", "method": "ql", "run": "1", "instance": 1,
     "avg": 605.45, "med": 28.69, "p90": 1440.0, "p95": 3650.0, "max": 44310.0,
     "reqs": 70541, "failed_pct": 0.22},
    {"dataset": "60-episodes-2", "method": "qcrisp", "run": "1", "instance": 1,
     "avg": 578.51, "med": 32.02, "p90": 1170.0, "p95": 3050.0, "max": 31500.0,
     "reqs": 71737, "failed_pct": 0.17},
    {"dataset": "60-episodes-2", "method": "qfuzzy", "run": "1", "instance": 1,
     "avg": 371.39, "med": 41.11, "p90": 785.15, "p95": 1780.0, "max": 15000.0,
     "reqs": 82481, "failed_pct": 0.42},
    # run 2 (started ~15:28) - Q-Learning Crisp passed p(90)<1000.
    {"dataset": "60-episodes-2", "method": "ql", "run": "2", "instance": 1,
     "avg": 647.38, "med": 37.27, "p90": 1730.0, "p95": 3580.0, "max": 25700.0,
     "reqs": 68679, "failed_pct": 0.44},
    {"dataset": "60-episodes-2", "method": "qcrisp", "run": "2", "instance": 1,
     "avg": 487.46, "med": 28.16, "p90": 897.43, "p95": 2660.0, "max": 39190.0,
     "reqs": 76099, "failed_pct": 0.15},
    {"dataset": "60-episodes-2", "method": "qfuzzy", "run": "2", "instance": 1,
     "avg": 577.74, "med": 28.93, "p90": 1130.0, "p95": 2990.0, "max": 35290.0,
     "reqs": 71766, "failed_pct": 0.14},
    # run 3 (started ~16:07) - all three methods passed p(90)<1000.
    {"dataset": "60-episodes-2", "method": "ql", "run": "3", "instance": 1,
     "avg": 372.17, "med": 34.60, "p90": 884.79, "p95": 1970.0, "max": 18190.0,
     "reqs": 82428, "failed_pct": 0.35},
    {"dataset": "60-episodes-2", "method": "qcrisp", "run": "3", "instance": 1,
     "avg": 508.26, "med": 30.22, "p90": 992.24, "p95": 2680.0, "max": 32920.0,
     "reqs": 75040, "failed_pct": 0.23},
    {"dataset": "60-episodes-2", "method": "qfuzzy", "run": "3", "instance": 1,
     "avg": 307.01, "med": 49.40, "p90": 850.60, "p95": 1520.0, "max": 14340.0,
     "reqs": 86515, "failed_pct": 0.17},
    # run 4-anomaly (started ~16:45) - excluded from RL_RUNS, see EXCLUDED_RUNS.
    {"dataset": "60-episodes-2", "method": "ql", "run": "4-anomaly", "instance": 1,
     "avg": 1070.0, "med": 50.78, "p90": 3240.0, "p95": 5930.0, "max": 47080.0,
     "reqs": 54646, "failed_pct": 0.14},
    {"dataset": "60-episodes-2", "method": "qcrisp", "run": "4-anomaly", "instance": 1,
     "avg": 892.39, "med": 63.33, "p90": 2120.0, "p95": 5300.0, "max": 32680.0,
     "reqs": 59920, "failed_pct": 0.08},
    {"dataset": "60-episodes-2", "method": "qfuzzy", "run": "4-anomaly", "instance": 1,
     "avg": 263.29, "med": 44.99, "p90": 690.89, "p95": 1420.0, "max": 9500.0,
     "reqs": 89516, "failed_pct": 0.10},
    # run 5 (started ~17:18) - Q-Learning Fuzzy passed p(90)<1000.
    {"dataset": "60-episodes-2", "method": "ql", "run": "5", "instance": 1,
     "avg": 591.14, "med": 29.46, "p90": 1340.0, "p95": 3120.0, "max": 38100.0,
     "reqs": 71286, "failed_pct": 0.21},
    {"dataset": "60-episodes-2", "method": "qcrisp", "run": "5", "instance": 1,
     "avg": 544.67, "med": 29.29, "p90": 1190.0, "p95": 3060.0, "max": 27000.0,
     "reqs": 73292, "failed_pct": 0.13},
    {"dataset": "60-episodes-2", "method": "qfuzzy", "run": "5", "instance": 1,
     "avg": 283.53, "med": 50.52, "p90": 771.91, "p95": 1400.0, "max": 12570.0,
     "reqs": 88110, "failed_pct": 0.21},
    # run 6 (started ~09:31 on 2026-08-22) - Q-Learning Fuzzy passed p(90)<1000.
    {"dataset": "60-episodes-2", "method": "ql", "run": "6", "instance": 1,
     "avg": 706.67, "med": 29.43, "p90": 1590.0, "p95": 3810.0, "max": 54300.0,
     "reqs": 66350, "failed_pct": 0.28},
    {"dataset": "60-episodes-2", "method": "qcrisp", "run": "6", "instance": 1,
     "avg": 511.68, "med": 28.70, "p90": 1040.0, "p95": 2960.0, "max": 27990.0,
     "reqs": 74858, "failed_pct": 0.13},
    {"dataset": "60-episodes-2", "method": "qfuzzy", "run": "6", "instance": 1,
     "avg": 252.23, "med": 38.17, "p90": 726.05, "p95": 1350.0, "max": 8550.0,
     "reqs": 90299, "failed_pct": 0.09},
    # HPA baseline (finished ~10:37 on 2026-08-22), three concurrent k6 processes
    {"dataset": "60-episodes-2", "method": "hpa", "run": "hpa", "instance": 1,
     "avg": 284.99, "med": 27.85, "p90": 589.17, "p95": 1590.0, "max": 15330.0,
     "reqs": 88018, "failed_pct": 0.00},
    {"dataset": "60-episodes-2", "method": "hpa", "run": "hpa", "instance": 2,
     "avg": 281.08, "med": 27.88, "p90": 590.59, "p95": 1510.0, "max": 13300.0,
     "reqs": 88286, "failed_pct": 0.00},
    {"dataset": "60-episodes-2", "method": "hpa", "run": "hpa", "instance": 3,
     "avg": 280.88, "med": 27.68, "p90": 556.11, "p95": 1440.0, "max": 17210.0,
     "reqs": 88314, "failed_pct": 0.00},

    # ---- batch B, 2026-08-21 -------------------------------------------------
    # data/testing/60-episodes-2/hpa-2 is a byte-identical copy of the hpa-2
    # folder in 60-episodes (same MD5 per CSV, same screenshot) - the SAME three
    # concurrent k6 processes, filed under both datasets. See BATCHES for why it
    # is not pooled into this dataset's aggregate.
    {"dataset": "60-episodes-2", "method": "hpa", "run": "hpa-2", "instance": 1,
     "avg": 279.85, "med": 27.84, "p90": 555.12, "p95": 1500.0, "max": 12760.0,
     "reqs": 88358, "failed_pct": 0.00},
    {"dataset": "60-episodes-2", "method": "hpa", "run": "hpa-2", "instance": 2,
     "avg": 283.41, "med": 27.85, "p90": 574.75, "p95": 1510.0, "max": 15330.0,
     "reqs": 88121, "failed_pct": 0.00},
    {"dataset": "60-episodes-2", "method": "hpa", "run": "hpa-2", "instance": 3,
     "avg": 290.76, "med": 27.82, "p90": 565.26, "p95": 1500.0, "max": 18080.0,
     "reqs": 87628, "failed_pct": 0.00},
    # ---- batch D, 2026-02-22 -------------------------------------------------
    # data/testing/60-episodes-2/hpa-3 is a byte-identical copy of the hpa-3 folder
    # in 60-episodes: the SAME single k6 process, filed under both datasets so
    # either notebook can reach it. Same caveats as its twin - see EXCLUDED_RUNS.
    {"dataset": "60-episodes-2", "method": "hpa", "run": "hpa-3", "instance": 1,
     "avg": 187.33, "med": 72.99, "p90": 303.91, "p95": 574.35, "max": 8430.0,
     "reqs": 95250, "failed_pct": 0.00},

    # ---- batch E, 2026-08-24 --------------------------------------------------
    # Every run and baseline below loads its models from a `model-2` folder,
    # unlike every batch-C run (`model`); see RL_RUNS.
    # run 7 (started ~11:51) - Q-Learning Crisp and Q-Learning Fuzzy both passed
    # p(90)<1000.
    {"dataset": "60-episodes-2", "method": "ql", "run": "7", "instance": 1,
     "avg": 1360.0, "med": 544.11, "p90": 3800.0, "p95": 5810.0, "max": 22810.0,
     "reqs": 47815, "failed_pct": 0.00},
    {"dataset": "60-episodes-2", "method": "qcrisp", "run": "7", "instance": 1,
     "avg": 239.62, "med": 42.03, "p90": 653.91, "p95": 1190.0, "max": 14100.0,
     "reqs": 91217, "failed_pct": 0.10},
    {"dataset": "60-episodes-2", "method": "qfuzzy", "run": "7", "instance": 1,
     "avg": 561.05, "med": 28.14, "p90": 996.22, "p95": 2650.0, "max": 29190.0,
     "reqs": 72543, "failed_pct": 0.07},
    # run 8 (started ~13:54) - Q-Learning Crisp passed p(90)<1000.
    {"dataset": "60-episodes-2", "method": "ql", "run": "8", "instance": 1,
     "avg": 1390.0, "med": 544.13, "p90": 3840.0, "p95": 5900.0, "max": 28700.0,
     "reqs": 47363, "failed_pct": 0.00},
    {"dataset": "60-episodes-2", "method": "qcrisp", "run": "8", "instance": 1,
     "avg": 252.09, "med": 43.17, "p90": 652.19, "p95": 1220.0, "max": 10790.0,
     "reqs": 90317, "failed_pct": 0.11},
    {"dataset": "60-episodes-2", "method": "qfuzzy", "run": "8", "instance": 1,
     "avg": 623.79, "med": 27.08, "p90": 1140.0, "p95": 3470.0, "max": 31830.0,
     "reqs": 69721, "failed_pct": 0.03},
    # HPA baseline (finished ~16:27), three concurrent k6 processes
    {"dataset": "60-episodes-2", "method": "hpa", "run": "hpa-4", "instance": 1,
     "avg": 306.84, "med": 31.00, "p90": 723.21, "p95": 1620.0, "max": 13280.0,
     "reqs": 86546, "failed_pct": 0.00},
    {"dataset": "60-episodes-2", "method": "hpa", "run": "hpa-4", "instance": 2,
     "avg": 313.22, "med": 30.31, "p90": 728.97, "p95": 1640.0, "max": 14990.0,
     "reqs": 86127, "failed_pct": 0.00},
    {"dataset": "60-episodes-2", "method": "hpa", "run": "hpa-4", "instance": 3,
     "avg": 306.44, "med": 31.36, "p90": 745.07, "p95": 1730.0, "max": 12210.0,
     "reqs": 86580, "failed_pct": 0.00},

    # ---- batch F, 2026-08-26 --------------------------------------------------
    # run 9 (started ~00:16) - Q-Learning Crisp passed p(90)<1000.
    {"dataset": "60-episodes-2", "method": "ql", "run": "9", "instance": 1,
     "avg": 1400.0, "med": 549.01, "p90": 3900.0, "p95": 5880.0, "max": 23090.0,
     "reqs": 47108, "failed_pct": 0.00},
    {"dataset": "60-episodes-2", "method": "qcrisp", "run": "9", "instance": 1,
     "avg": 247.91, "med": 36.29, "p90": 589.19, "p95": 1150.0, "max": 24360.0,
     "reqs": 90602, "failed_pct": 0.06},
    {"dataset": "60-episodes-2", "method": "qfuzzy", "run": "9", "instance": 1,
     "avg": 609.39, "med": 27.26, "p90": 1130.0, "p95": 3710.0, "max": 32490.0,
     "reqs": 70353, "failed_pct": 0.05},
    # HPA baseline (finished ~02:41), three concurrent k6 processes
    {"dataset": "60-episodes-2", "method": "hpa", "run": "hpa-5", "instance": 1,
     "avg": 274.53, "med": 30.05, "p90": 635.73, "p95": 1540.0, "max": 11570.0,
     "reqs": 88733, "failed_pct": 0.00},
    {"dataset": "60-episodes-2", "method": "hpa", "run": "hpa-5", "instance": 2,
     "avg": 270.52, "med": 31.13, "p90": 646.92, "p95": 1480.0, "max": 11110.0,
     "reqs": 89001, "failed_pct": 0.00},
    {"dataset": "60-episodes-2", "method": "hpa", "run": "hpa-5", "instance": 3,
     "avg": 272.53, "med": 30.67, "p90": 677.37, "p95": 1560.0, "max": 12360.0,
     "reqs": 88866, "failed_pct": 0.00},

    # ---- batch G, 2026-08-31 --------------------------------------------------
    # run 10 (started ~02:19) - models loaded from the same `model-2` folder as
    # runs 7-9. No HPA process collected alongside it. Q-Learning Crisp and
    # Q-Learning Fuzzy both passed p(90)<1000.
    {"dataset": "60-episodes-2", "method": "ql", "run": "10", "instance": 1,
     "avg": 1340.0, "med": 511.86, "p90": 3730.0, "p95": 5710.0, "max": 25790.0,
     "reqs": 48366, "failed_pct": 0.00},
    {"dataset": "60-episodes-2", "method": "qcrisp", "run": "10", "instance": 1,
     "avg": 244.70, "med": 40.52, "p90": 557.72, "p95": 1160.0, "max": 26000.0,
     "reqs": 90841, "failed_pct": 0.15},
    {"dataset": "60-episodes-2", "method": "qfuzzy", "run": "10", "instance": 1,
     "avg": 240.83, "med": 44.13, "p90": 612.58, "p95": 1220.0, "max": 8400.0,
     "reqs": 91145, "failed_pct": 0.14},
]

# (dataset, method, run, instance) is the primary key of K6_RESULTS. A re-pasted
# block is easy to miss by eye and does not fail loudly: k6_frame() just returns the
# run twice, doubling its weight in every aggregate, while k6_for() starts reporting
# a single-process RL run as "concurrent". Checked once at import instead.
_k6_keys = [(r["dataset"], r["method"], r["run"], r["instance"]) for r in K6_RESULTS]
if len(_k6_keys) != len(set(_k6_keys)):
    _dupes = sorted({k for k in _k6_keys if _k6_keys.count(k) > 1})
    raise ValueError(f"duplicate k6 summaries in K6_RESULTS: {_dupes}")

# Every transcribed run also needs a BATCHES entry, or k6_frame() cannot resolve its
# batch. Without this the omission surfaces as a KeyError from deep inside
# k6_frame(), on whichever notebook happens to touch that dataset first - and a
# record pasted into the wrong dataset section looks fine until then.
_unregistered = sorted(
    {(r["dataset"], r["run"]) for r in K6_RESULTS
     if r["run"] not in BATCHES.get(r["dataset"], {})}
)
if _unregistered:
    raise ValueError(
        f"k6 summaries whose run folder is missing from BATCHES: {_unregistered}. "
        f"Add each (dataset, run) to BATCHES with its collection batch."
    )

DEFAULT_DATASET = DATASETS[-1]


def dataset_of(data_dir: str) -> str:
    """
    The dataset key for a notebook's `DATA_DIR`, e.g. 'data/testing/60-episodes-2'.

    Takes the last path component, so a notebook keeps one `DATA_DIR` constant and
    derives the key from it instead of repeating the dataset name.
    """
    key = data_dir.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if key not in BATCHES:
        raise KeyError(
            f"{key!r} is not a known dataset; expected one of {DATASETS}. Add its "
            f"runs to BATCHES/RL_RUNS/HPA_RUNS and its k6 summaries to K6_RESULTS "
            f"before pointing a notebook at it."
        )
    return key


def batch_of(run: str, dataset: str = DEFAULT_DATASET) -> str:
    """The collection batch of one run folder, erroring with the dataset named."""
    try:
        return BATCHES[dataset][run]
    except KeyError:
        known = ", ".join(sorted(BATCHES[dataset]))
        raise KeyError(
            f"run folder {run!r} is not registered for dataset {dataset!r}; "
            f"known runs are: {known}"
        ) from None


def last_traffic_elapsed_s(df) -> float:
    """
    Elapsed seconds of the last observation window in `df` that carried traffic.

    The collector keeps sampling after k6 finishes and was stopped by hand, so
    every file ends with a stretch of idle windows: `response_time_ms` is NaN,
    which lines up exactly with `request_rate_rps == 0`. Those windows measure
    nothing, yet they still count toward replica-minutes and pull the utilisation
    means down, and how many of them a run has depends only on when someone hit
    Ctrl+C - 0.3 to 4.8 minutes across this dataset.

    Cutting each run here instead lines them all up on the load itself. Every run
    in the dataset stops carrying traffic at 30.53-30.75 min: the 30-minute k6
    profile plus the collector's lag.

    `df` must already carry an `elapsed_s` column.
    """
    rt = df["response_time_ms"]
    carried = df.loc[rt.notna() & (rt > 0), "elapsed_s"]
    if carried.empty:
        raise ValueError("no observation window in this run carried traffic")
    return float(carried.iloc[-1])


def k6_frame(dataset: str = DEFAULT_DATASET, runs=None, include_excluded: bool = False):
    """
    The k6 summaries as a DataFrame, with `batch` and `passes_slo` filled in.

    EXCLUDED_RUNS are dropped unless asked for. K6_RESULTS transcribes every run on
    disk, excluded ones included, so without this a caller that just wants "the
    dataset" silently pools a run the module already documents as not comparable -
    `4-anomaly` dominating a mean over five repetitions, `hpa-3` mixing in a
    different campaign. Naming a run in `runs` still returns it: that is the
    documented way to analyse an excluded run when it IS the subject.

    Args:
        dataset: dataset directory to read; `None` keeps every dataset, in which
            case `run` is NOT unique and rows have to be grouped by dataset too.
        runs: optional iterable of run folders to keep, in any order. An explicit
            request wins over the EXCLUDED_RUNS filter.
        include_excluded: keep EXCLUDED_RUNS in an otherwise unfiltered frame.
    """
    import pandas as pd

    df = pd.DataFrame(K6_RESULTS)
    if dataset is not None:
        df = df[df["dataset"] == dataset]
    if runs is not None:
        df = df[df["run"].isin(list(runs))]
    elif not include_excluded:
        excluded = {(d, r) for d, rs in EXCLUDED_RUNS.items() for r in rs}
        df = df[[(d, r) not in excluded for d, r in zip(df["dataset"], df["run"])]]
    df = df.copy()
    try:
        df["batch"] = [BATCHES[d][r] for d, r in zip(df["dataset"], df["run"])]
    except KeyError as missing:
        # A bare KeyError here names only the missing key, with no hint of which map
        # it came from or that BATCHES is the thing to edit.
        raise KeyError(
            f"{missing.args[0]!r} has a k6 summary in K6_RESULTS but no entry in "
            f"BATCHES, so its collection batch is unknown - register it under "
            f"BATCHES[dataset]. If it IS registered on disk, the running kernel is "
            f"holding an older copy of this module: rerun the notebook's setup cell, "
            f"which is what calls importlib.reload(k6_results)."
        ) from None
    df["label"] = df["method"].map(METHOD_LABELS)
    df["passes_slo"] = df["p90"] < SLO_MS
    return df.reset_index(drop=True)


def k6_for(method: str, run: str, dataset: str = DEFAULT_DATASET) -> dict:
    """The single k6 record for one method in one run folder (RL runs only)."""
    matches = [
        r for r in K6_RESULTS
        if r["method"] == method and r["run"] == run and r["dataset"] == dataset
    ]
    if not matches:
        raise KeyError(
            f"No k6 summary transcribed for method={method!r} run={run!r} "
            f"dataset={dataset!r}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"{run!r} holds {len(matches)} concurrent k6 processes for {method!r}; "
            f"use k6_frame() and aggregate them instead"
        )
    return matches[0]

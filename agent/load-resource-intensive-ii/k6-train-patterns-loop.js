import http from "k6/http";
import { check, sleep } from "k6";

/**
 * TRAINING WITH REPEATING UNSEEN PATTERNS (INFINITE LOOP)
 *
 * Purpose: Continuously repeat the 5-stage unseen pattern so the RL agent
 * can train on this load profile across many episodes.
 *
 * Pattern per cycle (30 minutes):
 * 1. Very Low  (~20 VUs)  - Minimal load        (0:00 - 6:00)
 * 2. High      (~150 VUs) - Heavy load           (6:00 - 12:00)
 * 3. Very High (200 VUs)  - Peak load            (12:00 - 18:00)
 * 4. Medium    (~100 VUs) - Moderate load         (18:00 - 24:00)
 * 5. Low       (~50 VUs)  - Light load           (24:00 - 30:00)
 *
 * Repeats: Configurable via CYCLES env var (default: 100 = ~50 hours)
 * Max VUs: 200
 *
 * Usage:
 *   k6 run k6-train-unseen-patterns-loop.js
 *   k6 run -e CYCLES=50 k6-train-unseen-patterns-loop.js
 *   k6 run -e CYCLES=200 -e BASE_URL=http://10.34.4.150:30080/api/qfuzzy k6-train-unseen-patterns-loop.js
 */

const CYCLES = parseInt(__ENV.CYCLES || "30", 10);

// Single cycle pattern (30 minutes)
const SINGLE_CYCLE = [
  { duration: "6m", target: 10 },
  { duration: "6m", target: 40 },
  { duration: "6m", target: 75 },
  { duration: "6m", target: 30 },
  { duration: "6m", target: 5 },
];

// Repeat the pattern CYCLES times
const REPEATED_STAGES = [];
for (let i = 0; i < CYCLES; i++) {
  for (const stage of SINGLE_CYCLE) {
    REPEATED_STAGES.push(Object.assign({}, stage));
  }
}

const totalMinutes = CYCLES * 30;
const totalHours = (totalMinutes / 60).toFixed(1);

console.log(`=== K6 Training: Unseen Patterns Loop ===`);
console.log(`Cycles: ${CYCLES} x 30min = ${totalMinutes}min (~${totalHours}h)`);
console.log(`Max VUs: 200`);
console.log(`Pattern: Very Low → High → Very High → Medium → Low (repeat)`);

export const options = {
  scenarios: {
    cpu_train: {
      executor: "ramping-vus",
      exec: "cpuTest",
      startVUs: 0,
      stages: REPEATED_STAGES,
      gracefulRampDown: "30s",
    },

    memory_train: {
      executor: "ramping-vus",
      exec: "memoryTest",
      startVUs: 0,
      stages: REPEATED_STAGES,
      gracefulRampDown: "30s",
    },
  },

  thresholds: {
    http_req_duration: ["p(90)<1000"],
  },
};

const BASE_URL = __ENV.BASE_URL || "http://10.34.4.150:30080/api/q";

console.log("BASE_URL: ", BASE_URL);

export function cpuTest() {
  const url = `${BASE_URL}/cpu?iterations=200`;
  const res = http.post(url);

  check(res, {
    "CPU: status 200": (r) => r.status === 200,
    "CPU: body not empty": (r) => r.body && r.body.length > 0,
  });

  sleep(1);
}

export function memoryTest() {
  const url = `${BASE_URL}/memory?size=2000&heavy_agg=true`;
  const res = http.post(url);

  check(res, {
    "Memory: status 200": (r) => r.status === 200,
    "Memory: body not empty": (r) => r.body && r.body.length > 0,
  });

  sleep(1);
}

import http from "k6/http";
import { check, sleep } from "k6";

/**
 * TESTING WITH UNSEEN PATTERNS (30 MINUTES)
 *
 * Purpose: Quick evaluation of generalization capability
 *
 * This script tests 5 distinct load levels NOT seen during training:
 * 1. Very Low  (~20 VUs)  - Minimal load        (0:00 - 6:00)
 * 2. High      (~150 VUs) - Heavy load           (6:00 - 12:00)
 * 3. Very High (200 VUs)  - Peak load            (12:00 - 18:00)
 * 4. Medium    (~100 VUs) - Moderate load         (18:00 - 24:00)
 * 5. Low       (~50 VUs)  - Light load           (24:00 - 30:00)
 *
 * Duration: 30 minutes total (5 stages × 6 min each)
 * Max VUs: 200
 *
 * CPU and Memory scenarios use the same pattern.
 *
 * Expected Outcome:
 * - Q-Fuzzy: Better generalization due to fuzzy state abstraction
 * - Q-Learning: Poor performance on unseen state combinations
 *
 * Metrics to Compare:
 * - Average reward on unseen patterns
 * - Response time stability
 * - Resource utilization efficiency
 * - Scaling decision quality
 */

// Shared stages for both CPU and Memory scenarios (30 minutes total)
const UNSEEN_STAGES = [
  // === Stage 1: Very Low (0:00 - 6:00) ===
  { duration: "30s", target: 20 }, // Ramp to very low
  { duration: "5m30s", target: 20 }, // Hold very low

  // === Stage 2: High (6:00 - 12:00) ===
  { duration: "30s", target: 150 }, // Ramp to high
  { duration: "5m30s", target: 150 }, // Hold high

  // === Stage 3: Very High (12:00 - 18:00) ===
  { duration: "30s", target: 200 }, // Ramp to very high
  { duration: "5m30s", target: 200 }, // Hold very high

  // === Stage 4: Medium (18:00 - 24:00) ===
  { duration: "30s", target: 100 }, // Ramp to medium
  { duration: "5m30s", target: 100 }, // Hold medium

  // === Stage 5: Low (24:00 - 30:00) ===
  { duration: "30s", target: 50 }, // Ramp to low
  { duration: "5m30s", target: 50 }, // Hold low
];

export const options = {
  scenarios: {
    // CPU-intensive UNSEEN patterns
    cpu_unseen: {
      executor: "ramping-vus",
      exec: "cpuTest",
      startVUs: 0,
      stages: UNSEEN_STAGES,
      gracefulRampDown: "30s",
    },

    // Memory-intensive UNSEEN patterns
    memory_unseen: {
      executor: "ramping-vus",
      exec: "memoryTest",
      startVUs: 0,
      stages: UNSEEN_STAGES,
      gracefulRampDown: "30s",
    },
  },

  thresholds: {
    http_req_duration: ["p(95)<5000"],
  },
};

const BASE_URL = "http://10.34.4.150:30080/api/qfuzzy";

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
  const url = `${BASE_URL}/memory?size_mb=50`;
  const res = http.post(url);

  check(res, {
    "Memory: status 200": (r) => r.status === 200,
    "Memory: body not empty": (r) => r.body && r.body.length > 0,
  });

  sleep(1);
}

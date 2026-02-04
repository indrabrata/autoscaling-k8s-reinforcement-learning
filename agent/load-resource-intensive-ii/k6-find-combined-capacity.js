import http from "k6/http";
import { check, sleep } from "k6";

/**
 * FIND COMBINED CPU + MEMORY CAPACITY
 *
 * Purpose: Find sustainable RPS for combined workload
 * Method: Gradual ramp-up with both CPU and Memory tests
 *
 * Watch for when RT exceeds 1-2 seconds!
 */

export const options = {
  scenarios: {
    cpu_gradual: {
      executor: "ramping-vus",
      exec: "cpuTest",
      startVUs: 0,
      stages: [
        { duration: "1m", target: 10 }, // Stage 1
        { duration: "1m", target: 20 }, // Stage 2
        { duration: "1m", target: 30 }, // Stage 3
        { duration: "1m", target: 40 }, // Stage 4
        { duration: "1m", target: 50 }, // Stage 5
        { duration: "1m", target: 60 }, // Stage 6
        { duration: "30s", target: 0 }, // Ramp down
      ],
      gracefulRampDown: "10s",
    },
    memory_gradual: {
      executor: "ramping-vus",
      exec: "memoryTest",
      startVUs: 0,
      stages: [
        { duration: "1m", target: 10 }, // Stage 1
        { duration: "1m", target: 15 }, // Stage 2
        { duration: "1m", target: 20 }, // Stage 3
        { duration: "1m", target: 25 }, // Stage 4
        { duration: "1m", target: 30 }, // Stage 5
        { duration: "1m", target: 40 }, // Stage 6
        { duration: "30s", target: 0 }, // Ramp down
      ],
      gracefulRampDown: "10s",
    },
  },

  thresholds: {
    http_req_duration: ["p(95)<2000"], // Alert if P95 > 2s
  },
};

const BASE_URL = "http://10.34.4.150:30080/api/q";

console.log("BASE_URL:", BASE_URL);

export function cpuTest() {
  const url = `${BASE_URL}/cpu?iterations=200`;
  const res = http.post(url);

  check(res, {
    "CPU: status 200": (r) => r.status === 200,
    "CPU: RT < 1s": (r) => r.timings.duration < 1000,
  });

  sleep(1);
}

export function memoryTest() {
  const url = `${BASE_URL}/memory?size=2000&heavy_agg=true`;
  const res = http.post(url);

  check(res, {
    "Memory: status 200": (r) => r.status === 200,
    "Memory: RT < 1s": (r) => r.timings.duration < 1000,
  });

  sleep(1);
}

/**
 * INTERPRETATION:
 *
 * Watch the output and find when "RT < 1s" check drops below 80%
 *
 * Example:
 * Stage 3 (30 CPU + 20 Memory = 50 total): RPS=45, RT avg=800ms ✅
 * Stage 4 (40 CPU + 25 Memory = 65 total): RPS=58, RT avg=1.5s ⚠️
 * Stage 5 (50 CPU + 30 Memory = 80 total): RPS=65, RT avg=3.2s ❌
 *
 * Capacity ≈ 45 RPS (at Stage 3 with RT < 1s)
 */

import http from "k6/http";
import { check, sleep } from "k6";

// --- CONFIGURATION ---
export const options = {
  scenarios: {
    // ---- CPU STRESS SCENARIO ----
    cpu_stress: {
      executor: "ramping-vus",
      exec: "cpuTest",
      startVUs: 0,
      stages: [
        { duration: "1h", target: 100 }, // Stage 1
        { duration: "1h", target: 250 }, // Stage 2
        { duration: "1h", target: 500 }, // Stage 3 (max)
        { duration: "1h", target: 500 }, // Stage 4 (steady load)
        { duration: "1h", target: 250 }, // Stage 5
        { duration: "1h", target: 0 }, // Stage 6 (ramp down)
      ],
      gracefulRampDown: "30s",
    },

    // ---- MEMORY STRESS SCENARIO ----
    memory_stress: {
      executor: "ramping-vus",
      exec: "memoryTest",
      startVUs: 0,
      stages: [
        { duration: "1h", target: 150 }, // Stage 1
        { duration: "1h", target: 400 }, // Stage 2
        { duration: "1h", target: 750 }, // Stage 3 (max)
        { duration: "1h", target: 750 }, // Stage 4 (steady load)
        { duration: "1h", target: 400 }, // Stage 5
        { duration: "1h", target: 0 }, // Stage 6 (ramp down)
      ],
      gracefulRampDown: "30s",
    },
  },

  // ---- THRESHOLDS ----
  thresholds: {
    http_req_failed: ["rate<0.01"], // less than 1% should fail
    "http_req_duration{scenario:cpu_stress}": ["p(95)<2000"],
    "http_req_duration{scenario:memory_stress}": ["p(95)<3000"],
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:30080/api";

// ---- TEST FUNCTIONS ----
export function cpuTest() {
  const url = `${BASE_URL}/cpu?iterations=200`;
  const res = http.post(url);
  check(res, {
    "CPU: status 200": (r) => r.status === 200,
    "CPU: body not empty": (r) => r.body.length > 0,
  });
  sleep(1);
}

export function memoryTest() {
  const url = `${BASE_URL}/memory?size=2000&heavy_agg=true`;
  const res = http.post(url);
  check(res, {
    "Memory: status 200": (r) => r.status === 200,
    "Memory: body not empty": (r) => r.body.length > 0,
  });
  sleep(1);
}

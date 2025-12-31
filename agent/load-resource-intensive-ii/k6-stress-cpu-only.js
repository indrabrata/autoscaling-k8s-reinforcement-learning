import http from "k6/http";
import { check, sleep } from "k6";

// Test CPU endpoint ONLY to find maximum capacity
// Purpose: See how many CPU requests one pod can handle alone

export const options = {
  scenarios: {
    cpu_only: {
      executor: "ramping-vus",
      exec: "cpuTest",
      startVUs: 0,
      stages: [
        { duration: "30s", target: 50 }, // Warm up
        { duration: "1m", target: 100 }, // Ramp
        { duration: "1m", target: 150 }, // Build pressure
        { duration: "1m", target: 200 }, // Near peak
        { duration: "1m", target: 250 }, // Max load
        { duration: "30s", target: 0 }, // Ramp down
      ],
      gracefulRampDown: "10s",
    },
  },

  thresholds: {
    // No thresholds - we WANT to see where it breaks
  },
};

const BASE_URL = __ENV.BASE_URL || "http://10.34.4.150:30080/api/qfuzzy";

export function cpuTest() {
  const url = `${BASE_URL}/cpu?iterations=200`;
  const res = http.post(url);

  check(res, {
    "CPU: status 200": (r) => r.status === 200,
    "CPU: body not empty": (r) => r.body && r.body.length > 0,
  });

  sleep(1);
}

import http from "k6/http";
import { check, sleep } from "k6";

// Test Memory endpoint ONLY to find maximum capacity
// Purpose: See how many Memory requests one pod can handle alone

export const options = {
  scenarios: {
    memory_only: {
      executor: "ramping-vus",
      exec: "memoryTest",
      startVUs: 0,
      stages: [
        { duration: "30s", target: 30 }, // Warm up
        { duration: "1m", target: 60 }, // Increase
        { duration: "1m", target: 100 }, // Push harder
        { duration: "1m", target: 150 }, // Find limit
        { duration: "1m", target: 200 }, // Stress to breaking
        { duration: "30s", target: 0 }, // Ramp down
      ],
      gracefulRampDown: "10s",
    },
  },

  thresholds: {
    // No thresholds - we WANT to see where it breaks
  },
};

const BASE_URL = "http://10.34.4.150:30080/api/qfuzzy";

console.log("BASE_URL: ", BASE_URL);

export function memoryTest() {
  const url = `${BASE_URL}/memory?size=2000&heavy_agg=true`;
  const res = http.post(url);

  check(res, {
    "Memory: status 200": (r) => r.status === 200,
    "Memory: body not empty": (r) => r.body && r.body.length > 0,
  });

  sleep(1);
}

// first try 60/s

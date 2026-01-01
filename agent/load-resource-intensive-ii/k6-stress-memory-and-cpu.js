import http from "k6/http";
import { check, sleep } from "k6";

// Quick 5-minute stress test to find ONE POD's capacity
// Purpose: See how many requests one pod can handle before errors occur
// Usage: Scale deployment to 1 replica, then run this test
// Watch for when http_req_failed starts increasing - that's your limit!

export const options = {
  scenarios: {
    cpu_stress: {
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
    memory_stress: {
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
    // No thresholds - i WANT to see where it breaks
    // Monitor the output to see when errors start occurring
  },
};

const BASE_URL = __ENV.BASE_URL || "http://10.34.4.150:30080/api/q";

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
  const url = `${BASE_URL}/memory?size=1500&heavy_agg=true`;
  const res = http.post(url);

  check(res, {
    "Memory: status 200": (r) => r.status === 200,
    "Memory: body not empty": (r) => r.body && r.body.length > 0,
  });

  sleep(1);
}

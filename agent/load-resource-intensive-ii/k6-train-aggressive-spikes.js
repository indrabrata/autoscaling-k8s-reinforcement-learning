import http from "k6/http";
import { check, sleep } from "k6";

/**
 * Aggressive Spike Pattern (6 hours)
 *
 * Purpose: Test autoscaler's ability to handle sudden, dramatic load spikes
 * Pattern: Rapid transitions from very low to very high load
 * Challenge: Quick scale-up decisions, avoiding oscillation
 */

export const options = {
  scenarios: {
    cpu_workload: {
      executor: "ramping-vus",
      exec: "cpuTest",
      startVUs: 0,
      stages: [
        // Spike 1: Calm before the storm
        { duration: "5m", target: 10 },
        { duration: "2m", target: 250 }, // SPIKE!
        { duration: "20m", target: 250 }, // Sustained high
        { duration: "3m", target: 20 }, // Crash down

        // Spike 2: Double spike
        { duration: "5m", target: 15 },
        { duration: "2m", target: 300 }, // SPIKE!
        { duration: "2m", target: 50 }, // Brief drop
        { duration: "2m", target: 350 }, // BIGGER SPIKE!
        { duration: "15m", target: 350 },
        { duration: "3m", target: 25 },

        // Spike 3: Triple spike pattern
        { duration: "5m", target: 20 },
        { duration: "1m", target: 200 },
        { duration: "5m", target: 200 },
        { duration: "1m", target: 50 },
        { duration: "1m", target: 300 },
        { duration: "5m", target: 300 },
        { duration: "1m", target: 40 },
        { duration: "1m", target: 400 },
        { duration: "10m", target: 400 },
        { duration: "3m", target: 30 },

        // Spike 4: Extreme spike
        { duration: "10m", target: 15 },
        { duration: "1m", target: 500 }, // MAX SPIKE!
        { duration: "30m", target: 500 }, // Sustained max
        { duration: "5m", target: 10 },

        // Recovery and final spike
        { duration: "15m", target: 50 },
        { duration: "2m", target: 350 },
        { duration: "20m", target: 350 },
        { duration: "5m", target: 0 },
      ],
      gracefulRampDown: "30s",
    },
    memory_workload: {
      executor: "ramping-vus",
      exec: "memoryTest",
      startVUs: 0,
      stages: [
        // Similar pattern but offset timing
        { duration: "5m", target: 20 },
        { duration: "2m", target: 300 },
        { duration: "20m", target: 300 },
        { duration: "3m", target: 30 },

        { duration: "5m", target: 25 },
        { duration: "2m", target: 350 },
        { duration: "2m", target: 60 },
        { duration: "2m", target: 400 },
        { duration: "15m", target: 400 },
        { duration: "3m", target: 35 },

        { duration: "5m", target: 30 },
        { duration: "1m", target: 250 },
        { duration: "5m", target: 250 },
        { duration: "1m", target: 60 },
        { duration: "1m", target: 350 },
        { duration: "5m", target: 350 },
        { duration: "1m", target: 50 },
        { duration: "1m", target: 450 },
        { duration: "10m", target: 450 },
        { duration: "3m", target: 40 },

        { duration: "10m", target: 25 },
        { duration: "1m", target: 500 },
        { duration: "30m", target: 500 },
        { duration: "5m", target: 20 },

        { duration: "15m", target: 70 },
        { duration: "2m", target: 400 },
        { duration: "20m", target: 400 },
        { duration: "5m", target: 0 },
      ],
      gracefulRampDown: "30s",
    },
  },
};

const BASE_URL = "http://10.34.4.150:30080/api/qfuzzy";

console.log("BASE_URL: ", BASE_URL);

export function cpuTest() {
  const url = `${BASE_URL}/cpu?iterations=200`;
  const res = http.post(url);
  check(res, { "CPU: status 200": (r) => r.status === 200 });
  sleep(1);
}

export function memoryTest() {
  const url = `${BASE_URL}/memory?size=2000&heavy_agg=true`;
  const res = http.post(url);
  check(res, { "Memory: status 200": (r) => r.status === 200 });
  sleep(1);
}

import http from "k6/http";
import { check, sleep } from "k6";

/**
 * 5-Hour Training Script for RL Agent
 *
 * Purpose: Generate diverse load patterns to expose the RL agent to various states:
 * - Low utilization (wasteful state)
 * - Optimal utilization (target state)
 * - High utilization (critical state)
 * - Rapid transitions (test responsiveness)
 * - Sustained load (test stability)
 *
 * This helps the agent learn optimal scaling decisions across different scenarios.
 */

export const options = {
  scenarios: {
    cpu_workload: {
      executor: "ramping-vus",
      exec: "cpuTest",
      startVUs: 0,
      stages: [
        // Stage 1: Warmup & Low Load (30 min) - Learn wasteful state
        { duration: "5m", target: 10 },   // Gradual start
        { duration: "10m", target: 20 },  // Low utilization
        { duration: "15m", target: 15 },  // Slight decrease

        // Stage 2: Ramp to Medium Load (30 min) - Learn optimal state
        { duration: "10m", target: 50 },  // Moderate increase
        { duration: "10m", target: 80 },  // Optimal range
        { duration: "10m", target: 100 }, // High-optimal

        // Stage 3: High Load Stress (45 min) - Learn critical state
        { duration: "15m", target: 150 }, // Push limits
        { duration: "15m", target: 200 }, // Stress test
        { duration: "15m", target: 180 }, // Slight relief

        // Stage 4: Rapid Fluctuations (45 min) - Learn adaptability
        { duration: "5m", target: 50 },   // Sharp drop
        { duration: "5m", target: 150 },  // Sharp spike
        { duration: "5m", target: 30 },   // Drop again
        { duration: "5m", target: 120 },  // Spike again
        { duration: "5m", target: 80 },   // Stabilize
        { duration: "10m", target: 90 },  // Maintain
        { duration: "10m", target: 70 },  // Slight decrease

        // Stage 5: Sustained High Load (60 min) - Learn stability under pressure
        { duration: "20m", target: 140 }, // High sustained
        { duration: "20m", target: 160 }, // Very high sustained
        { duration: "20m", target: 150 }, // Maintain high

        // Stage 6: Gradual Scale Down (45 min) - Learn efficient scale-down
        { duration: "10m", target: 100 }, // Start reducing
        { duration: "10m", target: 60 },  // Continue reducing
        { duration: "10m", target: 40 },  // Lower
        { duration: "10m", target: 20 },  // Very low
        { duration: "5m", target: 0 },    // Ramp down
      ],
      gracefulRampDown: "30s",
    },
    memory_workload: {
      executor: "ramping-vus",
      exec: "memoryTest",
      startVUs: 0,
      stages: [
        // Stage 1: Warmup & Low Load (30 min)
        { duration: "5m", target: 20 },
        { duration: "10m", target: 40 },
        { duration: "15m", target: 30 },

        // Stage 2: Ramp to Medium Load (30 min)
        { duration: "10m", target: 80 },
        { duration: "10m", target: 120 },
        { duration: "10m", target: 150 },

        // Stage 3: High Load Stress (45 min)
        { duration: "15m", target: 200 },
        { duration: "15m", target: 250 },
        { duration: "15m", target: 220 },

        // Stage 4: Rapid Fluctuations (45 min)
        { duration: "5m", target: 80 },
        { duration: "5m", target: 200 },
        { duration: "5m", target: 60 },
        { duration: "5m", target: 180 },
        { duration: "5m", target: 120 },
        { duration: "10m", target: 140 },
        { duration: "10m", target: 110 },

        // Stage 5: Sustained High Load (60 min)
        { duration: "20m", target: 180 },
        { duration: "20m", target: 210 },
        { duration: "20m", target: 190 },

        // Stage 6: Gradual Scale Down (45 min)
        { duration: "10m", target: 130 },
        { duration: "10m", target: 80 },
        { duration: "10m", target: 50 },
        { duration: "10m", target: 25 },
        { duration: "5m", target: 0 },
      ],
      gracefulRampDown: "30s",
    },
  },

  thresholds: {
    // Monitor but don't fail - we want to see how the system behaves
    http_req_duration: ["p(95)<5000"], // 95% of requests should complete within 5s
    http_req_failed: ["rate<0.1"],     // Less than 10% failure rate
  },
};

const BASE_URL = __ENV.BASE_URL || "http://10.34.4.150:30080/api/qfuzzy";

// CPU-intensive endpoint
export function cpuTest() {
  const url = `${BASE_URL}/cpu?iterations=200`;
  const res = http.post(url);

  check(res, {
    "CPU: status is 200": (r) => r.status === 200,
    "CPU: response has body": (r) => r.body && r.body.length > 0,
  });

  sleep(1); // 1 request per second per VU
}

// Memory-intensive endpoint
export function memoryTest() {
  const url = `${BASE_URL}/memory?size=2000&heavy_agg=true`;
  const res = http.post(url);

  check(res, {
    "Memory: status is 200": (r) => r.status === 200,
    "Memory: response has body": (r) => r.body && r.body.length > 0,
  });

  sleep(1); // 1 request per second per VU
}

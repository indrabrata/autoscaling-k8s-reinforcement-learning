import http from "k6/http";
import { check, sleep } from "k6";

/**
 * 24-Hour Comprehensive Training Script for Thesis Comparison
 *
 * Purpose: Generate complete dataset for comparing HPA vs Q-Learning vs Q-Learning Fuzzy
 *
 * Load Pattern Strategy (24 hours = 1440 minutes):
 * - Multiple complete cycles of load patterns
 * - Covers all states: wasteful, optimal, balanced, critical
 * - Includes gradual transitions and sudden spikes
 * - Simulates realistic daily patterns
 * - Repeats patterns to ensure statistical significance
 *
 * Expected States Coverage:
 * ✓ Wasteful (low utilization)
 * ✓ Optimal (target utilization 40-70%)
 * ✓ Balanced (stable moderate load)
 * ✓ Critical (high utilization, performance degradation)
 * ✓ Transition states (scaling events)
 * ✓ Oscillation scenarios (rapid changes)
 */

export const options = {
  scenarios: {
    cpu_workload: {
      executor: "ramping-vus",
      exec: "cpuTest",
      startVUs: 0,
      stages: [
        // CYCLE 1: Morning Ramp-Up Pattern (4 hours)
        { duration: "30m", target: 10 }, // Early morning - low load
        { duration: "30m", target: 30 }, // Users waking up
        { duration: "1h", target: 80 }, // Morning rush
        { duration: "1h", target: 120 }, // Peak morning
        { duration: "1h", target: 100 }, // Settling down

        // CYCLE 2: Midday Steady Load (3 hours)
        { duration: "1h", target: 90 }, // Steady usage
        { duration: "1h", target: 100 }, // Lunch peak
        { duration: "1h", target: 85 }, // Post-lunch

        // CYCLE 3: Afternoon Peak Pattern (3 hours)
        { duration: "30m", target: 130 }, // Building up
        { duration: "1h", target: 160 }, // Afternoon peak
        { duration: "1h", target: 180 }, // Maximum load
        { duration: "30m", target: 140 }, // Cooling down

        // CYCLE 4: Evening Fluctuation (3 hours)
        { duration: "30m", target: 80 }, // Rapid drop
        { duration: "30m", target: 150 }, // Evening spike
        { duration: "30m", target: 60 }, // Drop again
        { duration: "30m", target: 120 }, // Spike again
        { duration: "1h", target: 90 }, // Stabilizing

        // CYCLE 5: Night Pattern (2 hours)
        { duration: "30m", target: 40 }, // Winding down
        { duration: "1h", target: 20 }, // Night low
        { duration: "30m", target: 15 }, // Deep night

        // CYCLE 6: Late Night to Early Morning (3 hours)
        { duration: "2h", target: 10 }, // Minimal load
        { duration: "1h", target: 25 }, // Early risers

        // CYCLE 7: Second Morning Rush (3 hours)
        { duration: "30m", target: 50 },
        { duration: "1h", target: 100 },
        { duration: "1h", target: 140 },
        { duration: "30m", target: 110 },

        // CYCLE 8: Stress Test Pattern (3 hours)
        { duration: "30m", target: 200 }, // Extreme load
        { duration: "1h", target: 250 }, // Maximum stress
        { duration: "1h", target: 220 }, // Sustained high
        { duration: "30m", target: 160 }, // Recovery

        // Final cooldown
        { duration: "30m", target: 50 },
        { duration: "30m", target: 0 },
      ],
      gracefulRampDown: "1m",
    },
    memory_workload: {
      executor: "ramping-vus",
      exec: "memoryTest",
      startVUs: 0,
      stages: [
        // CYCLE 1: Morning Ramp-Up (4 hours)
        { duration: "30m", target: 20 },
        { duration: "30m", target: 50 },
        { duration: "1h", target: 120 },
        { duration: "1h", target: 160 },
        { duration: "1h", target: 140 },

        // CYCLE 2: Midday Steady (3 hours)
        { duration: "1h", target: 130 },
        { duration: "1h", target: 150 },
        { duration: "1h", target: 120 },

        // CYCLE 3: Afternoon Peak (3 hours)
        { duration: "30m", target: 180 },
        { duration: "1h", target: 220 },
        { duration: "1h", target: 250 },
        { duration: "30m", target: 200 },

        // CYCLE 4: Evening Fluctuation (3 hours)
        { duration: "30m", target: 120 },
        { duration: "30m", target: 200 },
        { duration: "30m", target: 90 },
        { duration: "30m", target: 170 },
        { duration: "1h", target: 130 },

        // CYCLE 5: Night Pattern (2 hours)
        { duration: "30m", target: 60 },
        { duration: "1h", target: 30 },
        { duration: "30m", target: 25 },

        // CYCLE 6: Late Night (3 hours)
        { duration: "2h", target: 20 },
        { duration: "1h", target: 40 },

        // CYCLE 7: Second Morning (3 hours)
        { duration: "30m", target: 80 },
        { duration: "1h", target: 140 },
        { duration: "1h", target: 190 },
        { duration: "30m", target: 150 },

        // CYCLE 8: Stress Test (3 hours)
        { duration: "30m", target: 280 },
        { duration: "1h", target: 350 },
        { duration: "1h", target: 300 },
        { duration: "30m", target: 220 },

        // Final cooldown
        { duration: "30m", target: 80 },
        { duration: "30m", target: 0 },
      ],
      gracefulRampDown: "1m",
    },
  },

  thresholds: {
    // Don't fail on errors - we want to see how systems handle overload
    http_req_failed: ["rate<0.15"], // Less than 15% failure rate
    http_req_duration: ["p(95)<8000"], // 95% of requests within 8s
    http_req_duration: ["p(99)<15000"], // 99% of requests within 15s
  },
};

const BASE_URL = __ENV.BASE_URL || "http://10.34.4.150:30080/api/qfuzzy";

export function cpuTest() {
  const url = `${BASE_URL}/cpu?iterations=100`;
  const res = http.post(url);

  check(res, {
    "CPU: status is 200": (r) => r.status === 200,
    "CPU: has response body": (r) => r.body && r.body.length > 0,
  });

  sleep(1);
}

export function memoryTest() {
  const url = `${BASE_URL}/memory?size=1500&heavy_agg=true`;
  const res = http.post(url);

  check(res, {
    "Memory: status is 200": (r) => r.status === 200,
    "Memory: has response body": (r) => r.body && r.body.length > 0,
  });

  sleep(1);
}

/**
 * Load Pattern Summary:
 *
 * Hour 0-4:   Morning ramp-up (10→120 VUs)
 * Hour 4-7:   Midday steady (85-100 VUs)
 * Hour 7-10:  Afternoon peak (130-180 VUs)
 * Hour 10-13: Evening fluctuation (60-150 VUs)
 * Hour 13-15: Night low (15-40 VUs)
 * Hour 15-18: Late night minimal (10-25 VUs)
 * Hour 18-21: Second morning rush (50-140 VUs)
 * Hour 21-24: Stress test (160-250 VUs)
 *
 * Total Duration: 24 hours
 * Total VU range: 0-250 (CPU), 0-350 (Memory)
 *
 * Expected Metrics Collection:
 * - ~2,880 data points (1 per 30 seconds with ITERATION=30s interval)
 * - Full coverage of all autoscaling states
 * - Multiple repetitions for statistical significance
 */

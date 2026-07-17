import http from "k6/http";
import { check, sleep } from "k6";

/**
 * 48-HOUR TRAINING LOAD PATTERN
 *
 * Purpose: Train RL agent with varied but realistic load patterns
 * Strategy: Mix of gradual changes, sustained loads, and moderate variations
 *
 * This pattern is designed to teach the agent common scenarios:
 * - Gradual ramp-ups (morning traffic)
 * - Sustained moderate load (business hours)
 * - Peak periods (lunch, afternoon)
 * - Low activity (night hours)
 * - Weekend patterns
 *
 * IMPORTANT: This is TRAINING data. Testing will use DIFFERENT patterns
 * to prove Q-Fuzzy generalizes better to unseen states.
 */

export const options = {
  scenarios: {
    // CPU-intensive traffic
    cpu_load: {
      executor: "ramping-vus",
      exec: "cpuTest",
      startVUs: 0,
      stages: [
        // === DAY 1 ===

        // Phase 1: Early Morning Ramp-up (0-6 hours)
        { duration: "30m", target: 20 }, // Gradual morning start
        { duration: "1h", target: 40 }, // Building up
        { duration: "1h", target: 60 }, // More users coming
        { duration: "1.5h", target: 80 }, // Pre-business peak
        { duration: "2h", target: 100 }, // Business hours start

        // Phase 2: Morning Business Hours (6-12 hours)
        { duration: "2h", target: 120 }, // Peak morning
        { duration: "1h", target: 140 }, // Late morning surge
        { duration: "1h", target: 120 }, // Slight dip
        { duration: "1h", target: 150 }, // Pre-lunch peak
        { duration: "1h", target: 130 }, // Lunch dip

        // Phase 3: Afternoon Peak (12-18 hours)
        { duration: "2h", target: 180 }, // Post-lunch peak
        { duration: "1h", target: 220 }, // Afternoon high
        { duration: "1h", target: 200 }, // Sustained high
        { duration: "1h", target: 170 }, // Starting to decrease
        { duration: "1h", target: 130 }, // End of business hours

        // Phase 4: Evening Wind-down (18-24 hours)
        { duration: "2h", target: 80 }, // Evening decrease
        { duration: "1h", target: 50 }, // Late evening
        { duration: "1h", target: 30 }, // Night start
        { duration: "2h", target: 15 }, // Deep night

        // === DAY 2 ===

        // Phase 5: Night Hours (24-30 hours)
        { duration: "2h", target: 10 }, // Minimal activity
        { duration: "2h", target: 15 }, // Late night
        { duration: "2h", target: 25 }, // Early morning

        // Phase 6: Second Day Business (30-42 hours)
        { duration: "2h", target: 80 }, // Morning ramp
        { duration: "2h", target: 120 }, // Business start
        { duration: "2h", target: 180 }, // Mid-morning
        { duration: "2h", target: 230 }, // Lunch peak
        { duration: "2h", target: 200 }, // Afternoon
        { duration: "2h", target: 150 }, // Late afternoon

        // Phase 7: Stress Test Period (42-46 hours)
        { duration: "1h", target: 250 }, // Sudden high load
        { duration: "1h", target: 300 }, // Peak stress (MAX)
        { duration: "1h", target: 280 }, // Sustained high
        { duration: "1h", target: 200 }, // Coming down

        // Phase 8: Final Cool-down (46-48 hours)
        { duration: "1h", target: 100 }, // Winding down
        { duration: "1h", target: 50 }, // Final decrease
      ],
      gracefulRampDown: "30s",
    },

    // Memory-intensive traffic
    memory_load: {
      executor: "ramping-vus",
      exec: "memoryTest",
      startVUs: 0,
      stages: [
        // === DAY 1 ===

        // Phase 1: Early Morning (0-6 hours)
        { duration: "1h", target: 30 },
        { duration: "2h", target: 50 },
        { duration: "3h", target: 80 },

        // Phase 2: Morning (6-12 hours)
        { duration: "2h", target: 100 },
        { duration: "2h", target: 120 },
        { duration: "2h", target: 110 },

        // Phase 3: Afternoon (12-18 hours)
        { duration: "2h", target: 160 },
        { duration: "2h", target: 200 },
        { duration: "2h", target: 170 },

        // Phase 4: Evening (18-24 hours)
        { duration: "3h", target: 70 },
        { duration: "3h", target: 30 },

        // === DAY 2 ===

        // Phase 5: Night (24-30 hours)
        { duration: "3h", target: 20 },
        { duration: "3h", target: 40 },

        // Phase 6: Business Day 2 (30-42 hours)
        { duration: "3h", target: 110 },
        { duration: "3h", target: 170 },
        { duration: "3h", target: 210 },
        { duration: "3h", target: 180 },

        // Phase 7: Stress (42-46 hours)
        { duration: "2h", target: 250 },
        { duration: "2h", target: 300 }, // Peak (MAX)

        // Phase 8: Cool-down (46-48 hours)
        { duration: "2h", target: 60 },
      ],
      gracefulRampDown: "30s",
    },
  },

  thresholds: {
    // No strict thresholds - we want to observe agent behavior
    // Even during failures, agent should learn to recover
  },
};

const BASE_URL = "http://10.34.4.150:30080/api/qfuzzy";

console.log("BASE_URL: ", BASE_URL);

export function cpuTest() {
  const url = `${BASE_URL}/cpu?iterations=100`;
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

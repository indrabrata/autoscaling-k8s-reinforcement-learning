import http from "k6/http";
import { check, sleep } from "k6";

/**
 * 5-DAY (120 HOURS) TRAINING LOAD PATTERN - MAX 500 VUs
 *
 * Purpose: Comprehensive RL agent training with extensive varied patterns
 * Strategy: Full business week simulation with realistic daily cycles
 *
 * This pattern provides comprehensive training across:
 * - Multiple day patterns (Monday-Friday business cycles)
 * - Weekday variations (high load days, oscillating days, steady days)
 * - Weekend transitions (Friday evening to Saturday)
 * - Various load levels (minimal night → peak business → stress periods)
 * - Both CPU and Memory workload types
 *
 * Pattern Characteristics:
 * - Day 1 (Mon): Standard business cycle with clear peaks and dips
 * - Day 2 (Tue): High sustained load with stress testing
 * - Day 3 (Wed): Rapid oscillations and variable patterns
 * - Day 4 (Thu): Steady gradual build-up
 * - Day 5 (Fri-Sat): Weekend transition with declining load
 *
 * IMPORTANT: This is TRAINING data (120h). Use k6-test-unseen-patterns.js
 * for generalization testing on completely different patterns.
 *
 * NOTE: This version has been scaled to have a maximum target of 500 VUs
 * (original version had max 300 VUs)
 */

export const options = {
  scenarios: {
    // CPU-intensive traffic
    cpu_load: {
      executor: "ramping-vus",
      exec: "cpuTest",
      startVUs: 0,
      stages: [
        // === DAY 1 - MONDAY: Standard Business Cycle ===

        // Phase 1: Night Hours (0-6 hours)
        { duration: "2h", target: 17 }, // Deep night minimal
        { duration: "2h", target: 20 }, // Late night
        { duration: "2h", target: 33 }, // Pre-dawn warming

        // Phase 2: Morning Ramp (6-12 hours)
        { duration: "1h", target: 83 }, // Morning start
        { duration: "1h", target: 133 }, // Building up
        { duration: "1h", target: 200 }, // Business hours begin
        { duration: "1h", target: 250 }, // Peak morning
        { duration: "1h", target: 217 }, // Pre-lunch dip
        { duration: "1h", target: 167 }, // Lunch dip

        // Phase 3: Afternoon Peak (12-18 hours)
        { duration: "2h", target: 300 }, // Post-lunch recovery
        { duration: "2h", target: 367 }, // Afternoon peak
        { duration: "2h", target: 317 }, // Late afternoon

        // Phase 4: Evening Wind-down (18-24 hours)
        { duration: "2h", target: 233 }, // Evening decline
        { duration: "2h", target: 133 }, // Late evening
        { duration: "2h", target: 50 }, // Night start

        // === DAY 2 - TUESDAY: High Load Day ===

        // Phase 5: Night Minimal (24-30 hours)
        { duration: "3h", target: 33 }, // Night baseline
        { duration: "3h", target: 42 }, // Pre-dawn

        // Phase 6: Heavy Morning (30-36 hours)
        { duration: "2h", target: 167 }, // Strong morning ramp
        { duration: "2h", target: 333 }, // Heavy morning traffic
        { duration: "2h", target: 417 }, // Peak morning load

        // Phase 7: Sustained High Load (36-42 hours)
        { duration: "2h", target: 467 }, // Very high sustained
        { duration: "1h", target: 500 }, // Maximum peak (STRESS)
        { duration: "2h", target: 450 }, // High afternoon
        { duration: "1h", target: 400 }, // Still high

        // Phase 8: Extended Evening (42-48 hours)
        { duration: "2h", target: 267 }, // Gradual decline
        { duration: "2h", target: 150 }, // Evening drop
        { duration: "2h", target: 67 }, // Night settling

        // === DAY 3 - WEDNESDAY: Variable Oscillations ===

        // Phase 9: Night Low (48-54 hours)
        { duration: "3h", target: 42 }, // Night minimal
        { duration: "3h", target: 50 }, // Early morning

        // Phase 10: Rapid Oscillations (54-66 hours)
        { duration: "1h", target: 133 }, // Morning start
        { duration: "1h", target: 250 }, // Spike up
        { duration: "1h", target: 167 }, // Drop back
        { duration: "1h", target: 333 }, // Spike higher
        { duration: "1h", target: 200 }, // Drop again
        { duration: "1h", target: 400 }, // High spike
        { duration: "1h", target: 233 }, // Drop
        { duration: "1h", target: 367 }, // Spike
        { duration: "1h", target: 267 }, // Moderate
        { duration: "1h", target: 333 }, // Up again
        { duration: "1h", target: 300 }, // Settling
        { duration: "1h", target: 250 }, // Decline start

        // Phase 11: Evening Moderate (66-72 hours)
        { duration: "2h", target: 183 }, // Evening moderate
        { duration: "2h", target: 100 }, // Late evening
        { duration: "2h", target: 58 }, // Night

        // === DAY 4 - THURSDAY: Steady Build-up ===

        // Phase 12: Night Minimal (72-78 hours)
        { duration: "3h", target: 30 }, // Deep night
        { duration: "3h", target: 37 }, // Pre-dawn

        // Phase 13: Gradual Business Day (78-90 hours)
        { duration: "2h", target: 117 }, // Morning moderate
        { duration: "2h", target: 200 }, // Business start
        { duration: "2h", target: 283 }, // Mid-morning
        { duration: "2h", target: 333 }, // Lunch time
        { duration: "2h", target: 350 }, // Afternoon steady
        { duration: "2h", target: 300 }, // Late afternoon

        // Phase 14: Evening Moderate (90-96 hours)
        { duration: "2h", target: 217 }, // Evening
        { duration: "2h", target: 117 }, // Late evening
        { duration: "2h", target: 67 }, // Night

        // === DAY 5 - FRIDAY TO SATURDAY: Weekend Transition ===

        // Phase 15: Night Low (96-102 hours)
        { duration: "3h", target: 37 }, // Night baseline
        { duration: "3h", target: 47 }, // Early morning

        // Phase 16: Light Friday Morning (102-108 hours)
        { duration: "2h", target: 117 }, // Lighter Friday start
        { duration: "2h", target: 200 }, // Mid-morning
        { duration: "2h", target: 267 }, // Pre-lunch

        // Phase 17: Early Weekend Decline (108-114 hours)
        { duration: "2h", target: 250 }, // Lunch period
        { duration: "2h", target: 183 }, // Early afternoon drop
        { duration: "2h", target: 117 }, // Friday afternoon exit

        // Phase 18: Weekend Transition (114-120 hours)
        { duration: "2h", target: 83 }, // Evening weekend start
        { duration: "2h", target: 58 }, // Weekend evening
        { duration: "2h", target: 42 }, // Weekend night
      ],
      gracefulRampDown: "30s",
    },

    // Memory-intensive traffic
    memory_load: {
      executor: "ramping-vus",
      exec: "memoryTest",
      startVUs: 0,
      stages: [
        // === DAY 1 - MONDAY ===

        // Phase 1: Night (0-6 hours)
        { duration: "3h", target: 25 },
        { duration: "3h", target: 42 },

        // Phase 2: Morning (6-12 hours)
        { duration: "2h", target: 100 },
        { duration: "2h", target: 167 },
        { duration: "2h", target: 217 },

        // Phase 3: Afternoon (12-18 hours)
        { duration: "2h", target: 300 },
        { duration: "2h", target: 350 },
        { duration: "2h", target: 283 },

        // Phase 4: Evening (18-24 hours)
        { duration: "3h", target: 167 },
        { duration: "3h", target: 67 },

        // === DAY 2 - TUESDAY ===

        // Phase 5: Night (24-30 hours)
        { duration: "3h", target: 33 },
        { duration: "3h", target: 50 },

        // Phase 6: High Load Morning (30-36 hours)
        { duration: "2h", target: 150 },
        { duration: "2h", target: 300 },
        { duration: "2h", target: 400 },

        // Phase 7: Peak Period (36-42 hours)
        { duration: "2h", target: 450 },
        { duration: "1h", target: 500 }, // Peak
        { duration: "2h", target: 433 },
        { duration: "1h", target: 383 },

        // Phase 8: Evening (42-48 hours)
        { duration: "3h", target: 250 },
        { duration: "3h", target: 83 },

        // === DAY 3 - WEDNESDAY ===

        // Phase 9: Night (48-54 hours)
        { duration: "3h", target: 50 },
        { duration: "3h", target: 67 },

        // Phase 10: Oscillating (54-66 hours)
        { duration: "1h", target: 117 },
        { duration: "1h", target: 233 },
        { duration: "1h", target: 150 },
        { duration: "1h", target: 317 },
        { duration: "1h", target: 183 },
        { duration: "1h", target: 383 },
        { duration: "1h", target: 217 },
        { duration: "1h", target: 350 },
        { duration: "1h", target: 250 },
        { duration: "1h", target: 317 },
        { duration: "1h", target: 283 },
        { duration: "1h", target: 233 },

        // Phase 11: Evening (66-72 hours)
        { duration: "3h", target: 150 },
        { duration: "3h", target: 58 },

        // === DAY 4 - THURSDAY ===

        // Phase 12: Night (72-78 hours)
        { duration: "3h", target: 33 },
        { duration: "3h", target: 47 },

        // Phase 13: Business Day (78-90 hours)
        { duration: "2h", target: 133 },
        { duration: "2h", target: 233 },
        { duration: "2h", target: 317 },
        { duration: "2h", target: 333 },
        { duration: "2h", target: 317 },
        { duration: "2h", target: 250 },

        // Phase 14: Evening (90-96 hours)
        { duration: "3h", target: 183 },
        { duration: "3h", target: 83 },

        // === DAY 5 - FRIDAY TO SATURDAY ===

        // Phase 15: Night (96-102 hours)
        { duration: "3h", target: 42 },
        { duration: "3h", target: 53 },

        // Phase 16: Friday Morning (102-108 hours)
        { duration: "2h", target: 125 },
        { duration: "2h", target: 217 },
        { duration: "2h", target: 283 },

        // Phase 17: Friday Afternoon (108-114 hours)
        { duration: "2h", target: 267 },
        { duration: "2h", target: 200 },
        { duration: "2h", target: 133 },

        // Phase 18: Weekend (114-120 hours)
        { duration: "3h", target: 92 },
        { duration: "3h", target: 50 },
      ],
      gracefulRampDown: "30s",
    },
  },

  thresholds: {
    // No strict thresholds - we want to observe agent behavior
    // Even during failures, agent should learn to recover
  },
};

const BASE_URL = __ENV.BASE_URL || "http://10.34.4.150:30080/api/q";

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

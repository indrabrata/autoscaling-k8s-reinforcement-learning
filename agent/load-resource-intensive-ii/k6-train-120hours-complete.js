import http from "k6/http";
import { check, sleep } from "k6";

/**
 * 5-DAY (120 HOURS) TRAINING LOAD PATTERN
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
        { duration: "2h", target: 10 }, // Deep night minimal
        { duration: "2h", target: 12 }, // Late night
        { duration: "2h", target: 20 }, // Pre-dawn warming

        // Phase 2: Morning Ramp (6-12 hours)
        { duration: "1h", target: 50 }, // Morning start
        { duration: "1h", target: 80 }, // Building up
        { duration: "1h", target: 120 }, // Business hours begin
        { duration: "1h", target: 150 }, // Peak morning
        { duration: "1h", target: 130 }, // Pre-lunch dip
        { duration: "1h", target: 100 }, // Lunch dip

        // Phase 3: Afternoon Peak (12-18 hours)
        { duration: "2h", target: 180 }, // Post-lunch recovery
        { duration: "2h", target: 220 }, // Afternoon peak
        { duration: "2h", target: 190 }, // Late afternoon

        // Phase 4: Evening Wind-down (18-24 hours)
        { duration: "2h", target: 140 }, // Evening decline
        { duration: "2h", target: 80 }, // Late evening
        { duration: "2h", target: 30 }, // Night start

        // === DAY 2 - TUESDAY: High Load Day ===

        // Phase 5: Night Minimal (24-30 hours)
        { duration: "3h", target: 20 }, // Night baseline
        { duration: "3h", target: 25 }, // Pre-dawn

        // Phase 6: Heavy Morning (30-36 hours)
        { duration: "2h", target: 100 }, // Strong morning ramp
        { duration: "2h", target: 200 }, // Heavy morning traffic
        { duration: "2h", target: 250 }, // Peak morning load

        // Phase 7: Sustained High Load (36-42 hours)
        { duration: "2h", target: 280 }, // Very high sustained
        { duration: "1h", target: 300 }, // Maximum peak (STRESS)
        { duration: "2h", target: 270 }, // High afternoon
        { duration: "1h", target: 240 }, // Still high

        // Phase 8: Extended Evening (42-48 hours)
        { duration: "2h", target: 160 }, // Gradual decline
        { duration: "2h", target: 90 }, // Evening drop
        { duration: "2h", target: 40 }, // Night settling

        // === DAY 3 - WEDNESDAY: Variable Oscillations ===

        // Phase 9: Night Low (48-54 hours)
        { duration: "3h", target: 25 }, // Night minimal
        { duration: "3h", target: 30 }, // Early morning

        // Phase 10: Rapid Oscillations (54-66 hours)
        { duration: "1h", target: 80 }, // Morning start
        { duration: "1h", target: 150 }, // Spike up
        { duration: "1h", target: 100 }, // Drop back
        { duration: "1h", target: 200 }, // Spike higher
        { duration: "1h", target: 120 }, // Drop again
        { duration: "1h", target: 240 }, // High spike
        { duration: "1h", target: 140 }, // Drop
        { duration: "1h", target: 220 }, // Spike
        { duration: "1h", target: 160 }, // Moderate
        { duration: "1h", target: 200 }, // Up again
        { duration: "1h", target: 180 }, // Settling
        { duration: "1h", target: 150 }, // Decline start

        // Phase 11: Evening Moderate (66-72 hours)
        { duration: "2h", target: 110 }, // Evening moderate
        { duration: "2h", target: 60 }, // Late evening
        { duration: "2h", target: 35 }, // Night

        // === DAY 4 - THURSDAY: Steady Build-up ===

        // Phase 12: Night Minimal (72-78 hours)
        { duration: "3h", target: 18 }, // Deep night
        { duration: "3h", target: 22 }, // Pre-dawn

        // Phase 13: Gradual Business Day (78-90 hours)
        { duration: "2h", target: 70 }, // Morning moderate
        { duration: "2h", target: 120 }, // Business start
        { duration: "2h", target: 170 }, // Mid-morning
        { duration: "2h", target: 200 }, // Lunch time
        { duration: "2h", target: 210 }, // Afternoon steady
        { duration: "2h", target: 180 }, // Late afternoon

        // Phase 14: Evening Moderate (90-96 hours)
        { duration: "2h", target: 130 }, // Evening
        { duration: "2h", target: 70 }, // Late evening
        { duration: "2h", target: 40 }, // Night

        // === DAY 5 - FRIDAY TO SATURDAY: Weekend Transition ===

        // Phase 15: Night Low (96-102 hours)
        { duration: "3h", target: 22 }, // Night baseline
        { duration: "3h", target: 28 }, // Early morning

        // Phase 16: Light Friday Morning (102-108 hours)
        { duration: "2h", target: 70 }, // Lighter Friday start
        { duration: "2h", target: 120 }, // Mid-morning
        { duration: "2h", target: 160 }, // Pre-lunch

        // Phase 17: Early Weekend Decline (108-114 hours)
        { duration: "2h", target: 150 }, // Lunch period
        { duration: "2h", target: 110 }, // Early afternoon drop
        { duration: "2h", target: 70 }, // Friday afternoon exit

        // Phase 18: Weekend Transition (114-120 hours)
        { duration: "2h", target: 50 }, // Evening weekend start
        { duration: "2h", target: 35 }, // Weekend evening
        { duration: "2h", target: 25 }, // Weekend night
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
        { duration: "3h", target: 15 },
        { duration: "3h", target: 25 },

        // Phase 2: Morning (6-12 hours)
        { duration: "2h", target: 60 },
        { duration: "2h", target: 100 },
        { duration: "2h", target: 130 },

        // Phase 3: Afternoon (12-18 hours)
        { duration: "2h", target: 180 },
        { duration: "2h", target: 210 },
        { duration: "2h", target: 170 },

        // Phase 4: Evening (18-24 hours)
        { duration: "3h", target: 100 },
        { duration: "3h", target: 40 },

        // === DAY 2 - TUESDAY ===

        // Phase 5: Night (24-30 hours)
        { duration: "3h", target: 20 },
        { duration: "3h", target: 30 },

        // Phase 6: High Load Morning (30-36 hours)
        { duration: "2h", target: 90 },
        { duration: "2h", target: 180 },
        { duration: "2h", target: 240 },

        // Phase 7: Peak Period (36-42 hours)
        { duration: "2h", target: 270 },
        { duration: "1h", target: 300 }, // Peak
        { duration: "2h", target: 260 },
        { duration: "1h", target: 230 },

        // Phase 8: Evening (42-48 hours)
        { duration: "3h", target: 150 },
        { duration: "3h", target: 50 },

        // === DAY 3 - WEDNESDAY ===

        // Phase 9: Night (48-54 hours)
        { duration: "3h", target: 30 },
        { duration: "3h", target: 40 },

        // Phase 10: Oscillating (54-66 hours)
        { duration: "1h", target: 70 },
        { duration: "1h", target: 140 },
        { duration: "1h", target: 90 },
        { duration: "1h", target: 190 },
        { duration: "1h", target: 110 },
        { duration: "1h", target: 230 },
        { duration: "1h", target: 130 },
        { duration: "1h", target: 210 },
        { duration: "1h", target: 150 },
        { duration: "1h", target: 190 },
        { duration: "1h", target: 170 },
        { duration: "1h", target: 140 },

        // Phase 11: Evening (66-72 hours)
        { duration: "3h", target: 90 },
        { duration: "3h", target: 35 },

        // === DAY 4 - THURSDAY ===

        // Phase 12: Night (72-78 hours)
        { duration: "3h", target: 20 },
        { duration: "3h", target: 28 },

        // Phase 13: Business Day (78-90 hours)
        { duration: "2h", target: 80 },
        { duration: "2h", target: 140 },
        { duration: "2h", target: 190 },
        { duration: "2h", target: 200 },
        { duration: "2h", target: 190 },
        { duration: "2h", target: 150 },

        // Phase 14: Evening (90-96 hours)
        { duration: "3h", target: 110 },
        { duration: "3h", target: 50 },

        // === DAY 5 - FRIDAY TO SATURDAY ===

        // Phase 15: Night (96-102 hours)
        { duration: "3h", target: 25 },
        { duration: "3h", target: 32 },

        // Phase 16: Friday Morning (102-108 hours)
        { duration: "2h", target: 75 },
        { duration: "2h", target: 130 },
        { duration: "2h", target: 170 },

        // Phase 17: Friday Afternoon (108-114 hours)
        { duration: "2h", target: 160 },
        { duration: "2h", target: 120 },
        { duration: "2h", target: 80 },

        // Phase 18: Weekend (114-120 hours)
        { duration: "3h", target: 55 },
        { duration: "3h", target: 30 },
      ],
      gracefulRampDown: "30s",
    },
  },

  thresholds: {
    // No strict thresholds - we want to observe agent behavior
    // Even during failures, agent should learn to recover
  },
};

const BASE_URL = "http://10.34.4.150:30080/api/q";

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

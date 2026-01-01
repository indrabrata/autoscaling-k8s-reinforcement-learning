import http from "k6/http";
import { check, sleep } from "k6";

/**
 * TESTING WITH UNSEEN PATTERNS (30 MINUTES)
 *
 * Purpose: Quick evaluation of generalization capability
 *
 * This script contains patterns NOT seen during training:
 * 1. Sudden spikes (flash crowds)
 * 2. Rapid oscillations (unstable traffic)
 * 3. Extended plateaus at unusual levels
 * 4. Very fast ramp rates
 * 5. Chaotic mixed patterns
 * 6. Extreme stress
 *
 * Duration: 30 minutes total (6 patterns × 5 min each)
 *
 * Expected Outcome:
 * - Q-Fuzzy: Better generalization due to fuzzy state abstraction
 * - Q-Learning: Poor performance on unseen state combinations
 *
 * Metrics to Compare:
 * - Average reward on unseen patterns
 * - Response time stability
 * - Resource utilization efficiency
 * - Scaling decision quality
 */

export const options = {
  scenarios: {
    // CPU-intensive UNSEEN patterns
    cpu_unseen: {
      executor: "ramping-vus",
      exec: "cpuTest",
      startVUs: 0,
      stages: [
        // === PATTERN 1: Flash Crowd Spikes (0-5 min) ===
        // NOT in training: Sudden jumps from low to very high
        { duration: "1m", target: 20 },    // Low baseline
        { duration: "30s", target: 250 },  // SUDDEN SPIKE (unseen)
        { duration: "2m", target: 250 },   // Sustained spike
        { duration: "30s", target: 30 },   // SUDDEN DROP (unseen)
        { duration: "1m", target: 40 },    // Recovery

        // === PATTERN 2: Rapid Oscillations (5-10 min) ===
        // NOT in training: Quick up-down cycles
        { duration: "30s", target: 80 },
        { duration: "1m", target: 160 },   // Up
        { duration: "1m", target: 60 },    // Down
        { duration: "1m", target: 170 },   // Up
        { duration: "1m", target: 50 },    // Down
        { duration: "1m", target: 180 },   // Up
        { duration: "30s", target: 40 },   // Down

        // === PATTERN 3: Unusual Plateaus (10-15 min) ===
        // NOT in training: Sustained at uncommon levels
        { duration: "1.5m", target: 95 },  // Odd level (unseen)
        { duration: "2m", target: 215 },   // Very high plateau (unseen)
        { duration: "1.5m", target: 35 },  // Low plateau (unseen)

        // === PATTERN 4: Extreme Ramp Rates (15-20 min) ===
        // NOT in training: Very fast changes
        { duration: "1m", target: 10 },
        { duration: "30s", target: 240 },  // VERY FAST RAMP UP (unseen)
        { duration: "1m", target: 240 },
        { duration: "30s", target: 15 },   // VERY FAST RAMP DOWN (unseen)
        { duration: "1m", target: 50 },
        { duration: "30s", target: 260 },  // Another fast ramp
        { duration: "1m", target: 180 },
        { duration: "30s", target: 30 },   // Fast drop

        // === PATTERN 5: Chaotic Mix (20-25 min) ===
        // NOT in training: Unpredictable combinations
        { duration: "30s", target: 70 },
        { duration: "30s", target: 190 },
        { duration: "30s", target: 110 },
        { duration: "1m", target: 230 },
        { duration: "30s", target: 60 },
        { duration: "30s", target: 150 },
        { duration: "30s", target: 90 },
        { duration: "30s", target: 200 },
        { duration: "30s", target: 130 },

        // === PATTERN 6: Extreme Stress (25-30 min) ===
        // NOT in training: Higher than training peak
        { duration: "1.5m", target: 270 }, // Higher than training
        { duration: "2m", target: 290 },   // Peak (unseen level)
        { duration: "1m", target: 250 },   // Still very high
        { duration: "30s", target: 100 },  // Final drop
      ],
      gracefulRampDown: "30s",
    },

    // Memory-intensive UNSEEN patterns
    memory_unseen: {
      executor: "ramping-vus",
      exec: "memoryTest",
      startVUs: 0,
      stages: [
        // === PATTERN 1: Spikes (0-5 min) ===
        { duration: "1m", target: 30 },
        { duration: "30s", target: 200 },  // Spike (unseen)
        { duration: "2m", target: 200 },
        { duration: "30s", target: 25 },   // Drop (unseen)
        { duration: "1m", target: 40 },

        // === PATTERN 2: Oscillations (5-10 min) ===
        { duration: "30s", target: 90 },
        { duration: "1m", target: 170 },   // Up
        { duration: "1m", target: 80 },    // Down
        { duration: "1m", target: 160 },   // Up
        { duration: "1m", target: 70 },    // Down
        { duration: "1m", target: 180 },   // Up
        { duration: "30s", target: 50 },   // Down

        // === PATTERN 3: Plateaus (10-15 min) ===
        { duration: "1.5m", target: 110 }, // Unusual level (unseen)
        { duration: "2m", target: 230 },   // High plateau (unseen)
        { duration: "1.5m", target: 45 },  // Low plateau (unseen)

        // === PATTERN 4: Fast Ramps (15-20 min) ===
        { duration: "1m", target: 20 },
        { duration: "30s", target: 210 },  // Fast ramp (unseen)
        { duration: "1m", target: 210 },
        { duration: "30s", target: 30 },   // Fast drop (unseen)
        { duration: "1m", target: 50 },
        { duration: "30s", target: 240 },  // Another fast ramp
        { duration: "1m", target: 190 },
        { duration: "30s", target: 40 },   // Fast drop

        // === PATTERN 5: Chaotic (20-25 min) ===
        { duration: "30s", target: 80 },
        { duration: "30s", target: 190 },
        { duration: "30s", target: 120 },
        { duration: "1m", target: 220 },
        { duration: "30s", target: 70 },
        { duration: "30s", target: 160 },
        { duration: "30s", target: 100 },
        { duration: "30s", target: 200 },
        { duration: "30s", target: 140 },

        // === PATTERN 6: Extreme Stress (25-30 min) ===
        { duration: "1.5m", target: 260 }, // Higher than training
        { duration: "2m", target: 280 },   // Peak (unseen level)
        { duration: "1m", target: 240 },   // Still high
        { duration: "30s", target: 80 },   // Final drop
      ],
      gracefulRampDown: "30s",
    },
  },

  thresholds: {
    // Capture all metrics - even failures show learning quality
    http_req_duration: ["p(95)<5000"], // More lenient for stress test
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

export function memoryTest() {
  const url = `${BASE_URL}/memory?size_mb=50`;
  const res = http.post(url);

  check(res, {
    "Memory: status 200": (r) => r.status === 200,
    "Memory: body not empty": (r) => r.body && r.body.length > 0,
  });

  sleep(1);
}

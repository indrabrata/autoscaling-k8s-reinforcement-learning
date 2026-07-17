import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  scenarios: {
    cpu_stress: {
      executor: "ramping-vus",
      exec: "cpuTest",
      startVUs: 0,
      stages: [
        { duration: "1m", target: 10 },
        { duration: "1m", target: 50 },
        { duration: "1m", target: 100 },
        { duration: "1m", target: 70 },
        { duration: "1m", target: 30 },
      ],
      gracefulRampDown: "10s",
    },
    memory_stress: {
      executor: "ramping-vus",
      exec: "memoryTest",
      startVUs: 0,
      stages: [
        { duration: "1m", target: 10 },
        { duration: "1m", target: 50 },
        { duration: "1m", target: 100 },
        { duration: "1m", target: 70 },
        { duration: "1m", target: 30 },
      ],
      gracefulRampDown: "10s",
    },
  },
  thresholds: {
    http_req_duration: ["p(90)<1000"],
  },
};

const BASE_URL = "http://10.34.4.150:30080/api/qfuzzy";

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
  const url = `${BASE_URL}/memory?size=3000&heavy_agg=true`;
  const res = http.post(url);
  check(res, {
    "Memory: status 200": (r) => r.status === 200,
    "Memory: body not empty": (r) => r.body && r.body.length > 0,
  });
  sleep(1);
}

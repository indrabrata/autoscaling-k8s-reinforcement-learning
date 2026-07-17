import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  tags: { testid: "k6-test-q-learning-crisp" },
  scenarios: {
    cpu_stress: {
      executor: "ramping-vus",
      exec: "cpuTest",
      startVUs: 0,
      stages: [
        { duration: "6m", target: 5 },
        { duration: "6m", target: 15 },
        { duration: "6m", target: 40 },
        { duration: "6m", target: 15 },
        { duration: "6m", target: 5 },
      ],
      gracefulRampDown: "30s",
    },
    memory_stress: {
      executor: "ramping-vus",
      exec: "memoryTest",
      startVUs: 0,
      stages: [
        { duration: "6m", target: 5 },
        { duration: "6m", target: 15 },
        { duration: "6m", target: 40 },
        { duration: "6m", target: 15 },
        { duration: "6m", target: 5 },
      ],
      gracefulRampDown: "30s",
    },
  },
  thresholds: {
    http_req_duration: ["p(90)<1000"],
  },
};

const BASE_URL = "http://10.34.4.150:30080/api/qcrisp";
console.log("BASE_URL: ", BASE_URL);

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
  const url = `${BASE_URL}/memory?size=2000&heavy_agg=true`;
  const res = http.post(url);
  check(res, {
    "Memory: status 200": (r) => r.status === 200,
    "Memory: body not empty": (r) => r.body && r.body.length > 0,
  });
  sleep(1);
}

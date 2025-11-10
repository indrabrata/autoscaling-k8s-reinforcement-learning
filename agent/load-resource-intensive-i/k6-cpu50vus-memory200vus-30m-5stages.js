import { check, sleep } from "k6";
import http from "k6/http";

const BASE_URL = __ENV.BASE_URL || "http://localhost:30080/api";
const imageBin = open("./shoes_resized.png", "b");
const transactionsData = JSON.parse(open("./seed_analyze.json"));

const cpuStages = [
  { target: 14, duration: "6m" }, // low (28%)
  { target: 28, duration: "6m" }, // medium (56%)
  { target: 50, duration: "6m" }, // very high (100%)
  { target: 36, duration: "6m" }, // low (72%)
  { target: 7, duration: "6m" },  // very low (14%)
];

const memoryStages = [
  { target: 70, duration: "6m" }, // low (35%)
  { target: 110, duration: "6m" }, // medium (55%)
  { target: 200, duration: "6m" }, // very high (100%)
  { target: 145, duration: "6m" },  // high (72%)
  { target: 30, duration: "6m" },  // very low (15%)
];

// Order/IO stages - max 200 VUs
const orderStages = [
  { target: 70, duration: "6m" }, // low (35%)
  { target: 110, duration: "6m" }, // medium (55%)
  { target: 200, duration: "6m" }, // very high (100%)
  { target: 145, duration: "6m" },  // low (72%)
  { target: 30, duration: "6m" },  // very low (15%)
];

export const options = {
  scenarios: {
    cpu_scenario: {
      executor: "ramping-vus",
      startVUs: 0,
      gracefulRampDown: "30s",
      stages: cpuStages,
      exec: "predictLoad",
    },
    memory_scenario: {
      executor: "ramping-vus",
      startVUs: 0,
      gracefulRampDown: "30s",
      stages: memoryStages,
      exec: "analyzeRequest",
    },
    order_scenario: {
      executor: "ramping-vus",
      startVUs: 0,
      gracefulRampDown: "30s",
      stages: orderStages,
      exec: "ordersRequest",
    },
  },
};

export function predictLoad() {
  const formData = {
    file: http.file(imageBin, "shoes_resized.png", "image/png"),
  };
  const res = http.post(`${BASE_URL}/predict?topk=1`, formData);
  check(res, { "predict status 200": (r) => r.status === 200 });
  sleep(0.5 + Math.random() * 0.5);
}

export function analyzeRequest() {
  const payload = JSON.stringify(transactionsData);
  const res = http.post(`${BASE_URL}/analyze`, payload, {
    headers: { "Content-Type": "application/json" },
  });
  check(res, { "analyze status 200": (r) => r.status === 200 });
  sleep(0.5 + Math.random() * 0.5);
}

export function ordersRequest() {
  const res = http.get(`${BASE_URL}/orders`);
  check(res, { "orders status 200": (r) => r.status === 200 });
  sleep(0.5 + Math.random() * 0.5);
}
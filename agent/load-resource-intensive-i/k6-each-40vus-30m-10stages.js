import { check, sleep } from "k6";
import http from "k6/http";

const BASE_URL = __ENV.BASE_URL || "http://localhost:30080/api";
const imageBin = open("./shoes.jpg", "b");
const transactionsData = JSON.parse(open("./seed_analyze.json"));

const cpuStages = [
  { target: 6, duration: "2m" },
  { target: 3, duration: "2m" },
  { target: 10, duration: "2m" },
  { target: 15, duration: "2m" },
  { target: 30, duration: "2m" },
  { target: 25, duration: "2m" },
  { target: 38, duration: "2m" },
  { target: 28, duration: "2m" },
  { target: 10, duration: "2m" },
  { target: 34, duration: "2m" },
  { target: 18, duration: "2m" },
  { target: 26, duration: "2m" },
  { target: 14, duration: "2m" },
  { target: 8, duration: "2m" },
  { target: 0, duration: "2m" },
];

const memoryStages = [
  { target: 30, duration: "2m" },
  { target: 3, duration: "2m" },
  { target: 35, duration: "2m" },
  { target: 20, duration: "2m" },
  { target: 40, duration: "2m" },
  { target: 27, duration: "2m" },
  { target: 16, duration: "2m" },
  { target: 30, duration: "2m" },
  { target: 25, duration: "2m" },
  { target: 36, duration: "2m" },
  { target: 11, duration: "2m" },
  { target: 27, duration: "2m" },
  { target: 14, duration: "2m" },
  { target: 9, duration: "2m" },
  { target: 0, duration: "2m" },
];

const orderStages = [
  { target: 11, duration: "2m" },
  { target: 4, duration: "2m" },
  { target: 31, duration: "2m" },
  { target: 18, duration: "2m" },
  { target: 37, duration: "2m" },
  { target: 40, duration: "2m" },
  { target: 26, duration: "2m" },
  { target: 34, duration: "2m" },
  { target: 23, duration: "2m" },
  { target: 14, duration: "2m" },
  { target: 29, duration: "2m" },
  { target: 19, duration: "2m" },
  { target: 12, duration: "2m" },
  { target: 6, duration: "2m" },
  { target: 0, duration: "2m" },
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
    file: http.file(imageBin, "shoes.jpg", "image/jpeg"),
  };
  const res = http.post(`${BASE_URL}/predict?topk=1`, formData);
  check(res, { "predict status 200": (r) => r.status === 200 });
  sleep(0.7 + Math.random() * 1);
}

export function analyzeRequest() {
  const payload = JSON.stringify(transactionsData);
  const res = http.post(`${BASE_URL}/analyze`, payload, {
    headers: { "Content-Type": "application/json" },
  });
  check(res, { "analyze status 200": (r) => r.status === 200 });
  sleep(0.7 + Math.random() * 1);
}

export function ordersRequest() {
  const res = http.get(`${BASE_URL}/orders`);
  check(res, { "orders status 200": (r) => r.status === 200 });
  sleep(0.7 + Math.random() * 1);
}



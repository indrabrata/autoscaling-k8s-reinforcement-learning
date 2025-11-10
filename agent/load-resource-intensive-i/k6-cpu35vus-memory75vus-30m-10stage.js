import { check, sleep } from "k6";
import http from "k6/http";

const BASE_URL = __ENV.BASE_URL || "http://localhost:30080/api";
const imageBin = open("./shoes.jpg", "b");
const transactionsData = JSON.parse(open("./seed_analyze.json"));

const cpuStages = [
  { target: 6, duration: "2m" },
  { target: 2, duration: "2m" },
  { target: 6, duration: "2m" },
  { target: 13, duration: "2m" },
  { target: 27, duration: "2m" },
  { target: 22, duration: "2m" },
  { target: 35, duration: "2m" },
  { target: 25, duration: "2m" },
  { target: 16, duration: "2m" },
  { target: 31, duration: "2m" },
  { target: 15, duration: "2m" },
  { target: 24, duration: "2m" },
  { target: 13, duration: "2m" },
  { target: 7, duration: "2m" },
  { target: 0, duration: "2m" },
];

const memoryStages = [
  { target: 58, duration: "2m" },
  { target: 5, duration: "2m" },
  { target: 65, duration: "2m" },
  { target: 38, duration: "2m" },
  { target: 75, duration: "2m" },
  { target: 50, duration: "2m" },
  { target: 30, duration: "2m" },
  { target: 58, duration: "2m" },
  { target: 47, duration: "2m" },
  { target: 68, duration: "2m" },
  { target: 40, duration: "2m" },
  { target: 50, duration: "2m" },
  { target: 28, duration: "2m" },
  { target: 17, duration: "2m" },
  { target: 0, duration: "2m" },
];

const orderStages = [
  { target: 20, duration: "2m" },
  { target: 8, duration: "2m" },
  { target: 58, duration: "2m" },
  { target: 35, duration: "2m" },
  { target: 70, duration: "2m" },
  { target: 75, duration: "2m" },
  { target: 47, duration: "2m" },
  { target: 65, duration: "2m" },
  { target: 43, duration: "2m" },
  { target: 28, duration: "2m" },
  { target: 55, duration: "2m" },
  { target: 35, duration: "2m" },
  { target: 23, duration: "2m" },
  { target: 13, duration: "2m" },
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
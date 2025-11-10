import { check, sleep } from "k6";
import http from "k6/http";

const BASE_URL = __ENV.BASE_URL || "http://localhost:30080/api";
const imageBin = open("./shoes.jpg", "b");
const transactionsData = JSON.parse(open("./seed_analyze.json"));

const cpuStages = [
  { target: 3, duration: "2m" },
  { target: 8, duration: "2m" },
  { target: 8, duration: "2m" },
  { target: 19, duration: "2m" },
  { target: 39, duration: "2m" },
  { target: 32, duration: "2m" },
  { target: 50, duration: "2m" },
  { target: 36, duration: "2m" },
  { target: 24, duration: "2m" },
  { target: 45, duration: "2m" },
  { target: 22, duration: "2m" },
  { target: 34, duration: "2m" },
  { target: 19, duration: "2m" },
  { target: 10, duration: "2m" },
  { target: 0, duration: "2m" },
];

const memoryStages = [
  { target: 115, duration: "2m" },
  { target: 11, duration: "2m" },
  { target: 130, duration: "2m" },
  { target: 75, duration: "2m" },
  { target: 150, duration: "2m" },
  { target: 100, duration: "2m" },
  { target: 60, duration: "2m" },
  { target: 115, duration: "2m" },
  { target: 95, duration: "2m" },
  { target: 135, duration: "2m" },
  { target: 80, duration: "2m" },
  { target: 100, duration: "2m" },
  { target: 55, duration: "2m" },
  { target: 35, duration: "2m" },
  { target: 0, duration: "2m" },
];

const orderStages = [
  { target: 40, duration: "2m" },
  { target: 15, duration: "2m" },
  { target: 115, duration: "2m" },
  { target: 70, duration: "2m" },
  { target: 140, duration: "2m" },
  { target: 150, duration: "2m" },
  { target: 95, duration: "2m" },
  { target: 130, duration: "2m" },
  { target: 85, duration: "2m" },
  { target: 55, duration: "2m" },
  { target: 110, duration: "2m" },
  { target: 70, duration: "2m" },
  { target: 45, duration: "2m" },
  { target: 25, duration: "2m" },
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
import { check, sleep } from "k6";
import http from "k6/http";

const BASE_URL = __ENV.BASE_URL || "http://localhost:30080/api";
const imageBin = open("./shoes_resized.png", "b");
const transactionsData = JSON.parse(open("./seed_analyze.json"));

const cpuStages = [
  { target: 35, duration: "6m" },
  { target: 55, duration: "6m" },
  { target: 50, duration: "6m" },
  { target: 10, duration: "6m" },
  { target: 5, duration: "6m" },
];

const memoryStages = [
  { target: 250, duration: "6m" },
  { target: 400, duration: "6m" },
  { target: 500, duration: "6m" }, 
  { target: 350, duration: "6m" }, 
  { target: 50, duration: "6m" },
];

const orderStages = [
  { target: 250, duration: "6m" },
  { target: 400, duration: "6m" },
  { target: 500, duration: "6m" },
  { target: 350, duration: "6m" },
  { target: 50, duration: "6m" },
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
import { check, sleep } from "k6";
import http from "k6/http";

const BASE_URL = __ENV.BASE_URL || "http://localhost:30080/api";
const imageBin = open("./shoes.jpg", "b");
const transactionsData = JSON.parse(open("./seed_analyze.json"));

const cpuStages = [
  { target: 300, duration: "2m" },
  { target: 600, duration: "2m" },
  { target: 900, duration: "2m" },
  { target: 1200, duration: "2m" },
  { target: 1600, duration: "2m" },
  { target: 1800, duration: "2m" },
  { target: 1500, duration: "2m" },
  { target: 1000, duration: "2m" },
  { target: 800, duration: "2m" },
  { target: 600, duration: "2m" },
  { target: 1000, duration: "2m" },
  { target: 1400, duration: "2m" },
  { target: 1800, duration: "2m" },
  { target: 1000, duration: "2m" },
  { target: 0, duration: "2m" },
];

const memoryStages = [
  { target: 1800, duration: "2m" },
  { target: 1600, duration: "2m" },
  { target: 1400, duration: "2m" },
  { target: 1000, duration: "2m" },
  { target: 800, duration: "2m" },
  { target: 600, duration: "2m" },
  { target: 900, duration: "2m" },
  { target: 1200, duration: "2m" },
  { target: 1500, duration: "2m" },
  { target: 1700, duration: "2m" },
  { target: 1300, duration: "2m" },
  { target: 1000, duration: "2m" },
  { target: 700, duration: "2m" },
  { target: 500, duration: "2m" },
  { target: 0, duration: "2m" },
];

const orderStages = [
  { target: 400, duration: "2m" },
  { target: 600, duration: "2m" },
  { target: 800, duration: "2m" },
  { target: 1000, duration: "2m" },
  { target: 1200, duration: "2m" },
  { target: 1500, duration: "2m" },
  { target: 1700, duration: "2m" },
  { target: 1900, duration: "2m" },
  { target: 1600, duration: "2m" },
  { target: 1300, duration: "2m" },
  { target: 1000, duration: "2m" },
  { target: 800, duration: "2m" },
  { target: 600, duration: "2m" },
  { target: 400, duration: "2m" },
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
  sleep(0.5 + Math.random() * 1); 
}

export function analyzeRequest() {
  const payload = JSON.stringify(transactionsData);
  const res = http.post(`${BASE_URL}/analyze`, payload, {
    headers: { "Content-Type": "application/json" },
  });
  check(res, {
    "analyze status 200": (r) => r.status === 200,
  });
  sleep(0.5 + Math.random() * 1);
}

export function ordersRequest() {
  const res = http.get(`${BASE_URL}/orders`);
  check(res, { "orders status 200": (r) => r.status === 200 });
  sleep(0.5 + Math.random() * 1);
}

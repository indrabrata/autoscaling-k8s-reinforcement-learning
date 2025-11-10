import { check, sleep } from "k6";
import http from "k6/http";

const BASE_URL = __ENV.BASE_URL || "http://localhost:30080/api";
const imageBin = open("./shoes.jpg", "b");
const transactionsData = JSON.parse(open("./seed_analyze.json"));

const cpuStages = [
  { target: 120, duration: "2m" },
  { target: 280, duration: "2m" },
  { target: 450, duration: "2m" },
  { target: 320, duration: "2m" },
  { target: 680, duration: "2m" },
  { target: 520, duration: "2m" },
  { target: 850, duration: "2m" },
  { target: 640, duration: "2m" },
  { target: 420, duration: "2m" },
  { target: 730, duration: "2m" },
  { target: 380, duration: "2m" },
  { target: 560, duration: "2m" },
  { target: 290, duration: "2m" },
  { target: 180, duration: "2m" },
  { target: 0, duration: "2m" },
];

const memoryStages = [
  { target: 760, duration: "2m" },
  { target: 540, duration: "2m" },
  { target: 820, duration: "2m" },
  { target: 480, duration: "2m" },
  { target: 920, duration: "2m" },
  { target: 650, duration: "2m" },
  { target: 380, duration: "2m" },
  { target: 710, duration: "2m" },
  { target: 590, duration: "2m" },
  { target: 840, duration: "2m" },
  { target: 470, duration: "2m" },
  { target: 620, duration: "2m" },
  { target: 350, duration: "2m" },
  { target: 190, duration: "2m" },
  { target: 0, duration: "2m" },
];

const orderStages = [
  { target: 240, duration: "2m" },
  { target: 510, duration: "2m" },
  { target: 720, duration: "2m" },
  { target: 430, duration: "2m" },
  { target: 860, duration: "2m" },
  { target: 980, duration: "2m" },
  { target: 610, duration: "2m" },
  { target: 790, duration: "2m" },
  { target: 550, duration: "2m" },
  { target: 340, duration: "2m" },
  { target: 670, duration: "2m" },
  { target: 460, duration: "2m" },
  { target: 280, duration: "2m" },
  { target: 150, duration: "2m" },
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
  check(res, { "analyze status 200": (r) => r.status === 200 });
  sleep(0.5 + Math.random() * 1);
}

export function ordersRequest() {
  const res = http.get(`${BASE_URL}/orders`);
  check(res, { "orders status 200": (r) => r.status === 200 });
  sleep(0.5 + Math.random() * 1);
}
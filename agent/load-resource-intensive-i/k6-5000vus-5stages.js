import { check, sleep } from "k6";
import http from "k6/http";

export const options = {
  scenarios: {
    predict_progressive: {
      executor: "ramping-arrival-rate",
      startRate: 50,
      timeUnit: "1s",
      preAllocatedVUs: 1500,
      maxVUs: 5000,
      stages: [
        { target: 400, duration: "4m" },
        { target: 1200, duration: "4m" },
        { target: 2800, duration: "4m" },
        { target: 4500, duration: "4m" },
        { target: 600, duration: "4m" },
      ],
      exec: "predictLoad",
    },

    analyze_and_orders_scaled: {
      executor: "ramping-vus",
      startVUs: 30,
      stages: [
        { target: 800, duration: "4m" },
        { target: 1800, duration: "4m" },
        { target: 3000, duration: "4m" },
        { target: 2500, duration: "4m" },
        { target: 300, duration: "4m" },
      ],
      exec: "hitTwoEndpoints",
    },

    cpu_memory_chaos: {
      executor: "ramping-arrival-rate",
      startRate: 80,
      timeUnit: "1s",
      preAllocatedVUs: 1200,
      maxVUs: 5000,
      stages: [
        { target: 600, duration: "4m" },
        { target: 1800, duration: "3m" },
        { target: 2500, duration: "1m" },
        { target: 3800, duration: "4m" },
        { target: 3200, duration: "3m" },
        { target: 4800, duration: "1m" },
        { target: 800, duration: "4m" },
      ],
      exec: "cpuMemoryMix",
    },
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:30080/api";
const imageBin = open("./shoes.jpg", "b");

export function predictLoad() {
  const formData = {
    file: http.file(imageBin, "shoes.jpg", "image/jpeg"),
  };

  const res = http.post(`${BASE_URL}/predict?topk=1`, formData);
  check(res, { "status is 200": (r) => r.status === 200 });

  sleep(1);
}

const transactionsData = JSON.parse(open("./seed_analyze.json"));

export function analyzeRequest() {
  const payload = JSON.stringify(transactionsData);

  const res = http.post(`${BASE_URL}/analyze`, payload, {
    headers: { "Content-Type": "application/json" },
    tags: { endpoint: "analyze", type: "memory" },
  });

  check(res, {
    "analyze status 200": (r) => r.status === 200,
    "analyze has totals": (r) => r.json("totals_by_category") !== undefined,
  });

  sleep(1);
}

export function ordersRequest() {
  const res = http.get(`${BASE_URL}/orders`, {
    tags: { endpoint: "orders", type: "io" },
  });

  check(res, { "orders status 200": (r) => r.status === 200 });

  sleep(1);
}

export function hitTwoEndpoints() {
  const payload = JSON.stringify(transactionsData);

  const responses = http.batch([
    [
      "POST",
      `${BASE_URL}/analyze`,
      payload,
      {
        headers: { "Content-Type": "application/json" },
        tags: { endpoint: "analyze", type: "memory" },
      },
    ],
    [
      "GET",
      `${BASE_URL}/orders`,
      null,
      { tags: { endpoint: "orders", type: "io" } },
    ],
  ]);

  check(responses[0], { "batch analyze ok": (r) => r.status === 200 });
  check(responses[1], { "batch orders ok": (r) => r.status === 200 });

  sleep(1);
}

export function cpuMemoryMix() {
  const rand = Math.random();

  if (rand < 0.45) {
    predictLoad();
  } else if (rand < 0.9) {
    analyzeRequest();
  } else {
    analyzeRequest();
    sleep(1);
    predictLoad();
  }
}

import { check, sleep } from "k6";
import http from "k6/http";

export let options = {
  scenarios: {
    predict_load: {
      executor: "ramping-arrival-rate",
      startRate: 20,
      timeUnit: "1s",
      preAllocatedVUs: 100,
      maxVUs: 500,
      stages: [
        { target: 100, duration: "2m" },
        { target: 500, duration: "1m" }, 
        { target: 300, duration: "3m" },
        { target: 50, duration: "2m" },
        { target: 800, duration: "1m" }, 
        { target: 200, duration: "2m" },
      ],
      exec: "predictLoad",
    },

    analyze_and_orders: {
      executor: "ramping-vus",
      startVUs: 10,
      stages: [
        { duration: "2m", target: 100 },
        { duration: "3m", target: 300 },
        { duration: "2m", target: 150 },
        { duration: "2m", target: 400 },
        { duration: "2m", target: 50 },
      ],
      exec: "hitTwoEndpoints",
    },

    background_noise: {
      executor: "constant-arrival-rate",
      rate: 20,
      timeUnit: "1s",
      duration: "15m",
      preAllocatedVUs: 20,
      maxVUs: 100,
      exec: "ordersRequest",
    },

    cpu_and_memory_mix: {
      executor: "ramping-arrival-rate",
      startRate: 30,
      timeUnit: "1s",
      preAllocatedVUs: 100,
      maxVUs: 400,
      stages: [
        { target: 150, duration: "2m" }, // ramp up
        { target: 400, duration: "2m" }, // sustained heavy load
        { target: 200, duration: "3m" }, // partial cooldown
        { target: 600, duration: "1m" }, // sudden spike
        { target: 100, duration: "2m" }, // cooldown
      ],
      exec: "cpuMemoryMix",
    },
  },
};

const BASE_URL = __ENV.BASE_URL || "http://100.127.139.26:30080";

const imageBin = open("./shoes.jpg", "b");

export function predictLoad() {
  let imagePayload = http.file(imageBin, "file", "image/jpeg");

  let res = http.post(`${BASE_URL}/predict?topk=1`, imagePayload.body, {
    headers: {
      "Content-Type": `multipart/form-data; boundary=${imagePayload.boundary}`,
    },
  });

  check(res, {
    "status is 200": (r) => r.status === 200,
    "has predictions": (r) => r.json("result") !== undefined,
  });

  sleep(Math.random() * 2);
}

const transactionsData = JSON.parse(open("./seed_analyze.json"));

function analyzeRequest() {
  let payload = JSON.stringify(transactionsData);

  let res = http.post(`${BASE_URL}/analyze`, payload, {
    headers: { "Content-Type": "application/json" },
  });

  check(res, {
    "status is 200": (r) => r.status === 200,
    "has totals_by_category": (r) => r.json("totals_by_category") !== undefined,
  });
}

export function ordersRequest() {
  let res = http.get(`${BASE_URL}/orders`);
  check(res, { "status is 200": (r) => r.status === 200 });
  sleep(Math.random());
} 

export function hitTwoEndpoints() {
  let payload = JSON.stringify(transactionsData);

  let responses = http.batch([
    [
      "POST",
      `${BASE_URL}/analyze`,
      payload,
      { headers: { "Content-Type": "application/json" } },
    ],
    ["GET", `${BASE_URL}/orders`],
  ]);

  check(responses[0], { "analyze ok": (r) => r.status === 200 });
  check(responses[1], { "orders ok": (r) => r.status === 200 });
  sleep(Math.random() * 2);
}

export function cpuMemoryMix() {
  if (Math.random() < 0.5) {
    predictLoad();
  } else {
    analyzeRequest();
  }
}

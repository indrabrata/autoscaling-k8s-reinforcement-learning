import { check, sleep } from "k6";
import http from "k6/http";

const BASE_URL = __ENV.BASE_URL || "http://localhost:30080/api";
const imageBin = open("./shoes_resized.png", "b");

const cpuStages = [
  { target: 35, duration: "6m" },
  { target: 55, duration: "6m" },
  { target: 100, duration: "6m" },
  { target: 75, duration: "6m" },
  { target: 15, duration: "6m" },
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

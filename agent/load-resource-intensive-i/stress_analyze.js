import http from "k6/http";
import { check, sleep } from "k6";

export let options = {
  discardResponseBodies: true,

  scenarios: {
    stress_analyze: {
      executor: "ramping-vus",
      startVUs: 0,                     // mulai dari 0 VU
      stages: [
        { duration: "30s", target: 1000 },   // naik ke 1.000 VU
        { duration: "30s", target: 3000 },   // naik ke 3.000 VU
        { duration: "30s", target: 5000 },   // naik ke 5.000 VU
        { duration: "30s", target: 7000 },   // naik ke 7.000 VU
        { duration: "30s", target: 10000 },  // naik ke 10.000 VU (puncak)
        { duration: "2m", target: 10000 },   // tahan di 10.000 VU (steady phase)
        { duration: "1m", target: 0 },       // turunkan perlahan ke 0
      ],
      gracefulRampDown: "1m",                // biarkan semua iterasi selesai dengan rapi
      exec: "analyzeLoad",
    },
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:8080/api";
const transactionsData = JSON.parse(open("./seed_analyze.json"));

export function analyzeLoad() {
  const payload = JSON.stringify(transactionsData);

  const res = http.post(`${BASE_URL}/analyze`, payload, {
    headers: { "Content-Type": "application/json" },
  });

  check(res, { "status is 200": (r) => r.status === 200 });

  sleep(0.1); // jeda antar request agar server tidak terlalu banjir
}

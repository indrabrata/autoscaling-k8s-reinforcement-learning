import http from "k6/http";
import { check, sleep } from "k6";

export let options = {
  discardResponseBodies: true,

  scenarios: {
    stress_orders: {
      executor: "ramping-vus",
      startVUs: 0,  // mulai dari 0 user
      stages: [
        { duration: "1m", target: 1000 },   // naik ke 1.000 user
        { duration: "1m", target: 2000 },   // naik ke 2.000 user
        { duration: "1m", target: 4000 },   // naik ke 4.000 user
        { duration: "1m", target: 6000 },   // naik ke 6.000 user
        { duration: "1m", target: 8000 },   // naik ke 8.000 user
        { duration: "1m", target: 10000 },  // naik ke 10.000 user (puncak)
        { duration: "2m", target: 10000 },  // tahan di 10.000 user
        { duration: "1m", target: 0 },      // turunkan kembali ke 0 user
      ],
      gracefulRampDown: "1m", // biarkan user menyelesaikan iterasi terakhir
      exec: "ordersLoad",
    },
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:8080/api";

export function ordersLoad() {
  const res = http.get(`${BASE_URL}/orders?limit=2000&offset=0`);
  check(res, { "status is 200": (r) => r.status === 200 });
  sleep(0.1); // jeda antar request agar server tidak terlalu dibanjiri
}

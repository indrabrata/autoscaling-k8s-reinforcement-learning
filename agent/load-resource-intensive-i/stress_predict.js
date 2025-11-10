import http from "k6/http";
import { check, sleep } from "k6";

export let options = {
  discardResponseBodies: true,

  scenarios: {
    stress_predict: {
      executor: "ramping-vus",
      startVUs: 0,  // mulai dari 0 user
      stages: [
        { duration: "30s", target: 500 },    // naik ke 500 VU
        { duration: "30s", target: 1000 },   // naik ke 1000 VU
        { duration: "30s", target: 2000 },   // naik ke 2000 VU
        { duration: "30s", target: 3000 },   // naik ke 3000 VU
        { duration: "30s", target: 4000 },   // naik ke 4000 VU
        { duration: "30s", target: 5000 },   // naik ke 5000 VU (puncak)
        { duration: "2m", target: 5000 },    // tahan di 5000 VU (steady phase)
        { duration: "1m", target: 0 },       // turunkan kembali ke 0
      ],
      gracefulRampDown: "1m",  // biarkan semua iterasi selesai dengan rapi
      exec: "predictLoad",
    },
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:8080/api";
const imageBin = open("./shoes_resized.png", "b");

export function predictLoad() {
  const formData = {
    file: http.file(imageBin, "shoes_resized.png", "image/png"),
  };

  const res = http.post(`${BASE_URL}/predict?topk=1`, formData);

  check(res, {
    "status is 200": (r) => r.status === 200,
  });

  sleep(0.1); // jeda antar request agar tidak terlalu padat
}

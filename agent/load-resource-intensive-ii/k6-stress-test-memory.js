import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  scenarios: {
    memory_stress: {
      executor: 'ramping-vus',
      exec: 'memoryTest',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 10 },
        { duration: '1m', target: 120 },
        { duration: '2m', target: 200 },
        { duration: '1m', target: 100 },
        { duration: '30s', target: 0 },
      ],
      gracefulRampDown: '10s',
    },
  },

  thresholds: {
    http_req_failed: ['rate<0.01'],
    'http_req_duration{scenario:memory_stress}': ['p(95)<3000'],
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:30080/api";

export function memoryTest() {
  const url = `${BASE_URL}/memory?size=5000&heavy_agg=true`;
  const res = http.post(url);

  check(res, {
    'Memory: status 200': (r) => r.status === 200,
    'Memory: body not empty': (r) => r.body.length > 0,
  });

  sleep(1);
}

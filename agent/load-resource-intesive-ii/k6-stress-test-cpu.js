import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  scenarios: {
    cpu_stress: {
      executor: 'ramping-vus',
      exec: 'cpuTest',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 10 },
        { duration: '1m', target: 50 },
        { duration: '2m', target: 150 },
        { duration: '1m', target: 50 },
        { duration: '30s', target: 0 },
      ],
      gracefulRampDown: '10s',
    },
  },

  thresholds: {
    http_req_failed: ['rate<0.01'],
    'http_req_duration{scenario:cpu_stress}': ['p(95)<2000'],
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:30080/api";

export function cpuTest() {
  const url = `${BASE_URL}/cpu?iterations=200`;
  const res = http.post(url);

  check(res, {
    'CPU: status 200': (r) => r.status === 200,
    'CPU: body not empty': (r) => r.body.length > 0,
  });

  sleep(1);
}

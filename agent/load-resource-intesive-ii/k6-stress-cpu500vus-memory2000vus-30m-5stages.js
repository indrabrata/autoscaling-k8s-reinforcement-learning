import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  scenarios: {
    cpu_stress: {
      executor: 'ramping-vus',
      exec: 'cpuTest',
      startVUs: 0,
      stages: [
        { duration: '4m', target: 200 },  // low
        { duration: '4m', target: 300 },  // medium
        { duration: '4m', target: 500 },  // very high
        { duration: '4m', target: 400 },  // high
        { duration: '4m', target: 100 },  // very low
      ],
      gracefulRampDown: '30s',
    },

    memory_stress: {
      executor: 'ramping-vus',
      exec: 'memoryTest',
      startVUs: 0,
      startTime: '0s',
      stages: [
        { duration: '4m', target: 800 },   // low
        { duration: '4m', target: 1200 },  // medium
        { duration: '4m', target: 2000 },  // very high
        { duration: '4m', target: 1600 },  // high
        { duration: '4m', target: 400 },   // very low
      ],
      gracefulRampDown: '30s',
    },
  },

  thresholds: {
    http_req_failed: ['rate<0.01'],
    'http_req_duration{scenario:cpu_stress}': ['p(95)<2000'],
    'http_req_duration{scenario:memory_stress}': ['p(95)<3000'],
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:30080/api";

export function cpuTest() {
  const url = `${BASE_URL}/cpu?iterations=200`;
  const res = http.post(url);

  check(res, {
    'CPU: status 200': (r) => r.status === 200,
    'CPU: body not empty': (r) => r.body && r.body.length > 0,
  });

  sleep(1);
}

export function memoryTest() {
  const url = `${BASE_URL}/memory?size=5000&heavy_agg=true`;
  const res = http.post(url);

  check(res, {
    'Memory: status 200': (r) => r.status === 200,
    'Memory: body not empty': (r) => r.body && r.body.length > 0,
  });

  sleep(1);
}

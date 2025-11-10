package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"runtime"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	requestCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "app_requests_total",
			Help: "Total HTTP requests",
		},
		[]string{"method", "endpoint", "status"},
	)

	requestLatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "app_request_latency_seconds",
			Help:    "Request latency in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "endpoint"},
	)
)

func init() {
	prometheus.MustRegister(requestCount)
	prometheus.MustRegister(requestLatency)
}

type CPUResponse struct {
	Result     []float64 `json:"result"`
	Operations int64     `json:"operations"`
	ElapsedMs  float64   `json:"elapsed_ms"`
}

type MemoryResponse struct {
	Result          map[string]interface{} `json:"result"`
	AllocatedMB     float64                `json:"allocated_mb"`
	StructuresCount int                    `json:"structures_count"`
	ElapsedMs       float64                `json:"elapsed_ms"`
}

func metricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(wrapped, r)

		elapsed := time.Since(start).Seconds()
		endpoint := r.URL.Path

		requestCount.WithLabelValues(r.Method, endpoint, strconv.Itoa(wrapped.statusCode)).Inc()
		requestLatency.WithLabelValues(r.Method, endpoint).Observe(elapsed)

		w.Header().Set("X-Elapsed-ms", fmt.Sprintf("%.2f", elapsed*1000))
	})
}

type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

func cpuIntensiveWork(iterations int) CPUResponse {
	start := time.Now()

	primes := generatePrimes(10000)

	results := make([]float64, 0, iterations)
	operations := int64(0)

	for i := 0; i < iterations; i++ {
		val := float64(i)
		for _, p := range primes[:100] {
			val = math.Pow(val, 1.01) + math.Sin(float64(p))
			val = math.Sqrt(math.Abs(val))
			operations++
		}

		matrixA := make([][]float64, 50)
		matrixB := make([][]float64, 50)
		for j := 0; j < 50; j++ {
			matrixA[j] = make([]float64, 50)
			matrixB[j] = make([]float64, 50)
			for k := 0; k < 50; k++ {
				matrixA[j][k] = float64(j*k + i)
				matrixB[j][k] = float64(j + k + i)
			}
		}

		result := multiplyMatrices(matrixA, matrixB)
		results = append(results, result[0][0])
		operations += 50 * 50 * 50
	}

	elapsed := time.Since(start).Seconds() * 1000

	return CPUResponse{
		Result:     results[:min(10, len(results))],
		Operations: operations,
		ElapsedMs:  elapsed,
	}
}

func generatePrimes(limit int) []int {
	if limit < 2 {
		return []int{}
	}

	sieve := make([]bool, limit+1)
	for i := 2; i <= limit; i++ {
		sieve[i] = true
	}

	for i := 2; i*i <= limit; i++ {
		if sieve[i] {
			for j := i * i; j <= limit; j += i {
				sieve[j] = false
			}
		}
	}

	primes := make([]int, 0)
	for i := 2; i <= limit; i++ {
		if sieve[i] {
			primes = append(primes, i)
		}
	}
	return primes
}

func multiplyMatrices(a, b [][]float64) [][]float64 {
	rows := len(a)
	cols := len(b[0])
	result := make([][]float64, rows)

	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			sum := 0.0
			for k := 0; k < len(b); k++ {
				sum += a[i][k] * b[k][j]
			}
			result[i][j] = sum
		}
	}
	return result
}

func memoryIntensiveWork(size int, heavyAgg bool) MemoryResponse {
	start := time.Now()

	largeSlice := make([]map[string]interface{}, size)
	categoryData := make(map[string][]float64)
	userData := make(map[int][]map[string]interface{})

	for i := 0; i < size; i++ {
		category := fmt.Sprintf("category_%d", i%10)
		userId := i % 100

		item := map[string]interface{}{
			"id":       i,
			"category": category,
			"user_id":  userId,
			"amount":   float64(i) * 1.5,
			"metadata": map[string]interface{}{
				"timestamp": time.Now().Unix(),
				"score":     float64(i) * 2.3,
				"tags":      []string{"tag1", "tag2", "tag3"},
			},
		}

		largeSlice[i] = item
		categoryData[category] = append(categoryData[category], float64(i)*1.5)
		userData[userId] = append(userData[userId], item)
	}

	matrixSize := min(size, 1000)
	transactionMatrix := make([][]float64, matrixSize)
	for i := 0; i < matrixSize; i++ {
		transactionMatrix[i] = make([]float64, 100)
		for j := 0; j < 100; j++ {
			transactionMatrix[i][j] = float64(i*j) * 1.234
		}
	}

	categoryTotals := make(map[string]float64)
	for cat, amounts := range categoryData {
		total := 0.0
		for _, amt := range amounts {
			total += amt
		}
		categoryTotals[cat] = total
	}

	correlationMatrix := [][]float64{}
	if heavyAgg {
		categoryStats := make(map[string]map[string]float64)
		for cat, amounts := range categoryData {
			mean := 0.0
			for _, amt := range amounts {
				mean += amt
			}
			mean /= float64(len(amounts))

			stdDev := 0.0
			for _, amt := range amounts {
				stdDev += math.Pow(amt-mean, 2)
			}
			stdDev = math.Sqrt(stdDev / float64(len(amounts)))

			categoryStats[cat] = map[string]float64{
				"mean":   mean,
				"stddev": stdDev,
			}
		}

		corrSize := min(len(categoryData), 100)
		correlationMatrix = make([][]float64, corrSize)
		categories := make([]string, 0, corrSize)
		for cat := range categoryData {
			if len(categories) < corrSize {
				categories = append(categories, cat)
			}
		}

		for i := 0; i < corrSize; i++ {
			correlationMatrix[i] = make([]float64, corrSize)
			for j := 0; j < corrSize; j++ {
				if i < len(categories) && j < len(categories) {
					correlationMatrix[i][j] = calculateCorrelation(
						categoryData[categories[i]],
						categoryData[categories[j]],
					)
				}
			}
		}
	}

	elapsed := time.Since(start).Seconds() * 1000

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	allocatedMB := float64(m.Alloc) / 1024 / 1024

	result := map[string]interface{}{
		"total_items":      len(largeSlice),
		"categories":       len(categoryData),
		"users":            len(userData),
		"category_totals":  categoryTotals,
		"matrix_size":      matrixSize,
		"heavy_agg_done":   heavyAgg,
		"correlation_size": len(correlationMatrix),
	}

	return MemoryResponse{
		Result:          result,
		AllocatedMB:     allocatedMB,
		StructuresCount: len(largeSlice) + len(categoryData) + len(userData),
		ElapsedMs:       elapsed,
	}
}

func calculateCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0
	}

	meanX, meanY := 0.0, 0.0
	for i := 0; i < len(x); i++ {
		meanX += x[i]
		meanY += y[i]
	}
	meanX /= float64(len(x))
	meanY /= float64(len(y))

	numerator, denomX, denomY := 0.0, 0.0, 0.0
	for i := 0; i < len(x); i++ {
		dx := x[i] - meanX
		dy := y[i] - meanY
		numerator += dx * dy
		denomX += dx * dx
		denomY += dy * dy
	}

	if denomX == 0 || denomY == 0 {
		return 0
	}

	return numerator / math.Sqrt(denomX*denomY)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func cpuHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost && r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	iterations := 100
	if iter := r.URL.Query().Get("iterations"); iter != "" {
		if val, err := strconv.Atoi(iter); err == nil && val > 0 {
			iterations = val
		}
	}

	result := cpuIntensiveWork(iterations)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func memoryHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost && r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	size := 10000
	if s := r.URL.Query().Get("size"); s != "" {
		if val, err := strconv.Atoi(s); err == nil && val > 0 {
			size = val
		}
	}

	heavyAgg := false
	if r.URL.Query().Get("heavy_agg") == "true" {
		heavyAgg = true
	}

	result := memoryIntensiveWork(size, heavyAgg)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func main() {
	mux := http.NewServeMux()

	mux.HandleFunc("/healthz", healthHandler)
	mux.HandleFunc("/cpu", cpuHandler)
	mux.HandleFunc("/memory", memoryHandler)
	mux.Handle("/metrics", promhttp.Handler())

	handler := metricsMiddleware(mux)

	port := ":8080"
	log.Printf("Server starting on port %s", port)
	log.Printf("Endpoints:")
	log.Printf("  GET  /healthz            - Health check")
	log.Printf("  POST /cpu?iterations=N   - CPU intensive workload")
	log.Printf("  POST /memory?size=N&heavy_agg=true - Memory intensive workload")
	log.Printf("  GET  /metrics            - Prometheus metrics")

	if err := http.ListenAndServe(port, handler); err != nil {
		log.Fatal(err)
	}
}

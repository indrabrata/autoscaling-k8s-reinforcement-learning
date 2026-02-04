# Querying InfluxDB for Autoscaling Reinforcement Learning Metrics

1. Total Reward per Episode (Last Value)

   ```influxdb
   from(bucket: "your_bucket")
   |> range(start: -24h)
   |> filter(fn: (r) => r.\_measurement == "autoscaling_metrics")
   |> filter(fn: (r) => r.\_field == "cumulative_reward")
   |> filter(fn: (r) => r.algorithm == "Q-LEARNING" or r.algorithm == "Q-LEARNING-FUZZY")
   |> last()
   |> group(columns: ["algorithm", "episode_number"])
   ```

2. Learning Curve Comparison

   ```influxdb
   from(bucket: "your_bucket")
   |> range(start: -24h)
   |> filter(fn: (r) => r.\_measurement == "autoscaling_metrics")
   |> filter(fn: (r) => r.\_field == "cumulative_reward")
   |> filter(fn: (r) => r.terminated == 1) // Only end of episode
   |> group(columns: ["algorithm"])
   ```

3. Average Reward per Episode Window

   ```influxdb
   from(bucket: "your_bucket")
   |> range(start: -24h)
   |> filter(fn: (r) => r.\_measurement == "autoscaling_metrics")
   |> filter(fn: (r) => r.\_field == "cumulative_reward")
   |> filter(fn: (r) => r.terminated == 1)
   |> group(columns: ["algorithm"])
   |> aggregateWindow(every: 10, fn: mean) // Average per 10 episodes
   ```

4. Exporting Data to CSV

   ```bash
   sudo docker exec member-influxdb2-1 influx query 'from(bucket:"autoscaling-reinforcement-learning") |> range(start:2025-12-27T00:00:00Z, stop: now()) |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "\_value")' --org "" --token "" --raw > metrics_export.csv
   ```

5. Average Cumulative Reward by Algorithm

   ```influxdb
   from(bucket: "autoscaling-reinforcement-learning")
   |> range(
      start: 2026-01-03T03:35:00Z,
      stop: now()
   )
   |> filter(fn: (r) => r["_measurement"] == "autoscaling_metrics")
   |> filter(fn: (r) => r["_field"] == "cumulative_reward")
   |> filter(fn: (r) => r["algorithm"] == "Q-LEARNING-FUZZY")
   |> filter(fn: (r) => r["deployment"] == "resource-intensive-qfuzzy")
   |> group(columns: ["algorithm", "deployment"])
   |> mean()

   ```

# Autoscaling Reinforcement Learning

This repository contains code and resources for the auto scaling reinforcement as my thesis, which focuses on applying reinforcement learning techniques to autoscaling in cloud computing environments for now this applied in kubernetes.

## Reinforcement Learning

### State observation

State is a component that can observe by the agent

- Memory Utilization
- CPU Utilization
- Response Time Inference (Application)
- Latest Action
- Action Trend
- Request Rate
- Request Rate Trend

### Action

Scaling n replica to kubernetes.

### Reward

Formula from the state observation to choose the current replicas is good enough or no.

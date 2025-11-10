# Rest API Inferencing Service

This is a simple REST API service for model inferencing using FastAPI. It allows you to perform inference on a pre-trained machine learning model via HTTP requests.

## Features

- Predict object using resnet pre-trained model
- Get list orders
- Batch analysis

## Deployment

This application is designed to be deployed on a Kubernetes cluster. Below are the steps to deploy the application and set up monitoring using Prometheus and Ingress using nginx.

### Helm Command

```bash
helm install prometheus prometheus-community/kube-prometheus-stack \
--namespace monitoring \
--set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
--set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
--set kubeControllerManager.enabled=false \
--set kubeScheduler.enabled=false \
--set kubeEtcd.enabled=false \
--set kubeProxy.enabled=false \
--set grafana.enabled=false
--set alertmanager.enabled=false
```

```bash
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --set controller.metrics.enabled=true \
  --set-string controller.podAnnotations."prometheus\.io/scrape"="true" \
  --set-string controller.podAnnotations."prometheus\.io/port"="10254" \
  --set-string controller.podAnnotations."prometheus\.io/path"="/metrics" \
  --set controller.metrics.serviceMonitor.enabled=true \
  --set controller.metrics.serviceMonitor.additionalLabels.release="prometheus" \
  --set controller.metrics.serviceMonitor.namespace="ingress-nginx" \
  --set controller.metrics.serviceMonitor.additionalLabels.prometheus="kube-prometheus-stack" \
  --set controller.metrics.serviceMonitor.additionalLabels.app="kube-prometheus-stack" \
  --set controller.metrics.serviceMonitor.interval=5s \
  --set controller.metrics.serviceMonitor.scrapeTimeout=3s \
```

```bash
helm install metrics-server metrics-server/metrics-server \
  --set args={--kubelet-insecure-tls}
```

### Kubectl Command

```bash
kubectl port-forward --address 0.0.0.0 -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
```

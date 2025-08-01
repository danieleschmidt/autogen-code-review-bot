# Monitoring & Observability Guide

Comprehensive monitoring and observability setup for AutoGen Code Review Bot.

## Overview

Our observability stack provides complete visibility into:

- **Metrics**: Performance, resource usage, business metrics
- **Logs**: Structured application and system logs
- **Traces**: Distributed request tracing
- **Health Checks**: Service availability and dependency health
- **Alerts**: Proactive issue detection and notification

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │    Collector    │    │    Storage      │
│                 │    │                 │    │                 │
│  AutoGen Bot    │───▶│  Prometheus     │───▶│  Prometheus     │
│  Metrics        │    │  OTEL Collector │    │  InfluxDB       │
│  Logs           │    │  Fluentd        │    │  Elasticsearch  │
│  Traces         │    │  Jaeger Agent   │    │  Jaeger         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Visualization   │
                       │                 │
                       │  Grafana        │
                       │  Kibana         │
                       │  Jaeger UI      │
                       └─────────────────┘
```

## Quick Start

### Basic Monitoring Stack

```bash
# Start core monitoring services
docker-compose -f monitoring/prometheus.yml up -d

# Start logging stack (optional)
docker-compose -f monitoring/logging-config.yml up -d

# Start tracing stack (optional)
docker-compose -f monitoring/tracing-config.yml up -d
```

### Access Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Jaeger UI**: http://localhost:16686
- **Kibana**: http://localhost:5601
- **Uptime Kuma**: http://localhost:3001

## Metrics Collection

### Application Metrics

The bot exposes metrics at `/metrics` endpoint:

```
# Performance metrics
autogen_bot_request_duration_seconds
autogen_bot_request_total
autogen_bot_errors_total

# Business metrics
autogen_bot_prs_analyzed_total
autogen_bot_reviews_posted_total
autogen_bot_github_api_calls_total

# Resource metrics
autogen_bot_memory_usage_bytes
autogen_bot_cpu_usage_percent
autogen_bot_cache_hit_ratio
```

### System Metrics

Collected via Prometheus exporters:

- **Node Exporter**: System metrics (CPU, memory, disk, network)
- **cAdvisor**: Container metrics
- **Redis Exporter**: Cache metrics
- **Blackbox Exporter**: Endpoint health checks

### Custom Metrics

Add custom metrics to your application:

```python
from prometheus_client import Counter, Histogram, Gauge

# Request counter
REQUEST_COUNT = Counter(
    'autogen_bot_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

# Response time histogram
REQUEST_DURATION = Histogram(
    'autogen_bot_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

# Active connections gauge
ACTIVE_CONNECTIONS = Gauge(
    'autogen_bot_active_connections',
    'Number of active connections'
)
```

## Logging Strategy

### Structured Logging

All logs use structured JSON format:

```json
{
  "timestamp": "2023-12-01T10:00:00Z",
  "level": "INFO",
  "service": "autogen-bot",
  "component": "pr_analyzer",
  "message": "PR analysis completed",
  "pr_number": 123,
  "duration_ms": 1500,
  "files_analyzed": 5,
  "trace_id": "abc123def456"
}
```

### Log Levels

- **ERROR**: System errors, exceptions
- **WARN**: Recoverable issues, deprecations
- **INFO**: General operational messages
- **DEBUG**: Detailed debugging information

### Log Aggregation

Choose from multiple log aggregation options:

#### ELK Stack (Elasticsearch, Logstash, Kibana)
```bash
docker-compose -f monitoring/logging-config.yml up -d
```

#### Fluentd + Elasticsearch
```bash
# Lighter alternative to Logstash
docker-compose -f monitoring/logging-config.yml up -d fluentd elasticsearch kibana
```

#### Simple File-based Logging
```bash
# For development or small deployments
docker-compose logs -f autogen-bot > logs/application.log
```

## Distributed Tracing

### Jaeger Setup

```bash
# Start Jaeger all-in-one
docker-compose -f monitoring/tracing-config.yml up -d jaeger
```

### OpenTelemetry Integration

Add tracing to your application:

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)
span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Use in code
@tracer.start_as_current_span("analyze_pr")
def analyze_pr(pr_data):
    span = trace.get_current_span()
    span.set_attribute("pr.number", pr_data["number"])
    span.set_attribute("pr.files", len(pr_data["files"]))
    
    # Your analysis logic here
    
    span.set_attribute("pr.analysis_duration", duration)
    return result
```

### Trace Sampling

Configure sampling to manage trace volume:

- **Always sample**: Development environments
- **Probabilistic**: Sample percentage (e.g., 10%)
- **Rate limiting**: Fixed number of traces per second
- **Adaptive**: Based on system load

## Health Checks

### Application Health

The bot exposes health endpoints:

- `GET /health`: Overall application health
- `GET /health/ready`: Readiness probe (K8s)
- `GET /health/live`: Liveness probe (K8s)
- `GET /health/deps`: Dependency health check

### External Dependencies

Monitor external service health:

```yaml
# Blackbox exporter configuration
modules:
  github_api:
    prober: http
    http:
      valid_status_codes: [200, 401, 403]
      method: GET
  
  openai_api:
    prober: http
    http:
      valid_status_codes: [200, 401]
      method: GET
```

### Uptime Monitoring

Use Uptime Kuma for comprehensive uptime monitoring:

```bash
# Start uptime monitoring
docker-compose -f monitoring/health-checks.yml up -d uptime-kuma
```

## Alerting

### Prometheus Alerting Rules

Critical alerts are defined in `monitoring/rules/`:

```yaml
# High error rate alert
- alert: HighErrorRate
  expr: rate(autogen_bot_errors_total[5m]) > 0.1
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High error rate detected"
    description: "Error rate is {{ $value }} errors per second"

# Service down alert
- alert: ServiceDown
  expr: up{job="autogen-bot"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "AutoGen Bot service is down"
```

### Alert Channels

Configure multiple notification channels:

- **Slack**: Real-time team notifications
- **PagerDuty**: On-call escalation
- **Email**: Management reports
- **Webhook**: Custom integrations

### Alert Severity Levels

- **Critical**: Service down, data loss, security breach
- **Warning**: Performance degradation, capacity issues
- **Info**: Maintenance events, deployments

## Dashboards

### Grafana Dashboards

Pre-built dashboards available:

1. **Service Overview**: High-level service health
2. **Performance Metrics**: Response times, throughput
3. **Error Tracking**: Error rates, error types
4. **Resource Usage**: CPU, memory, disk, network
5. **Business Metrics**: PRs analyzed, reviews posted
6. **GitHub Integration**: API usage, rate limits
7. **Cache Performance**: Hit rates, response times

### Dashboard as Code

Dashboards are version-controlled in JSON format:

```bash
# Import dashboard
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana-dashboard.json
```

### Custom Dashboards

Create custom dashboards for specific needs:

- **Development**: Debug-focused metrics
- **Operations**: Infrastructure monitoring
- **Business**: KPI tracking
- **Security**: Security event monitoring

## Performance Monitoring

### Key Metrics to Monitor

1. **Latency**: Response time percentiles (p50, p95, p99)
2. **Throughput**: Requests per second
3. **Error Rate**: Failed requests percentage
4. **Saturation**: Resource utilization (CPU, memory)

### SLI/SLO Definition

Service Level Indicators and Objectives:

```yaml
sli:
  availability: 
    metric: up{job="autogen-bot"}
    target: 99.9%
  
  latency:
    metric: histogram_quantile(0.95, rate(autogen_bot_request_duration_seconds_bucket[5m]))
    target: < 2s
  
  error_rate:
    metric: rate(autogen_bot_errors_total[5m]) / rate(autogen_bot_requests_total[5m])
    target: < 1%
```

### Performance Baselines

Establish performance baselines:

- **Small PR**: < 5 seconds analysis time
- **Large PR**: < 60 seconds analysis time
- **API Response**: < 200ms response time
- **Cache Hit**: < 1ms response time

## Troubleshooting

### Common Monitoring Issues

1. **High Cardinality Metrics**
   ```bash
   # Check metric cardinality
   curl http://localhost:9090/api/v1/label/__name__/values
   ```

2. **Missing Metrics**
   ```bash
   # Check Prometheus targets
   curl http://localhost:9090/api/v1/targets
   ```

3. **Log Ingestion Issues**
   ```bash
   # Check Elasticsearch cluster health
   curl http://localhost:9200/_cluster/health
   ```

### Debug Commands

```bash
# Check service health
curl http://localhost:8080/health

# View metrics
curl http://localhost:8080/metrics

# Check Prometheus config
curl http://localhost:9090/api/v1/status/config

# Grafana API health
curl http://localhost:3000/api/health
```

## Security Considerations

### Metrics Security

- **No sensitive data**: Never expose secrets in metrics
- **Access control**: Secure Grafana and Prometheus
- **Network security**: Use TLS for external access
- **Data retention**: Configure appropriate retention policies

### Log Security

- **Log sanitization**: Remove sensitive data from logs
- **Access controls**: Restrict log access to authorized users
- **Encryption**: Encrypt logs in transit and at rest
- **Audit logging**: Track access to sensitive logs

### Monitoring Infrastructure Security

- **Authentication**: Enable authentication for all dashboards
- **Authorization**: Role-based access control
- **Network segmentation**: Isolate monitoring services
- **Regular updates**: Keep monitoring tools updated

## Best Practices

1. **Monitor what matters**: Focus on user-impacting metrics
2. **Use labels wisely**: Avoid high cardinality labels
3. **Set up alerts proactively**: Alert on leading indicators
4. **Regular dashboard reviews**: Keep dashboards current
5. **Documentation**: Document metrics and alerts
6. **Test monitoring**: Regularly test alert mechanisms
7. **Capacity planning**: Monitor growth trends
8. **Cost optimization**: Balance retention with costs
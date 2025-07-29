# Monitoring and Observability

This document outlines the monitoring and observability setup for the AutoGen Code Review Bot.

## Overview

The monitoring stack includes:
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Custom metrics**: Application-specific monitoring
- **Health checks**: Service availability monitoring

## Quick Start

### Docker Compose Monitoring

The included `docker-compose.yml` contains a complete monitoring setup:

```bash
# Start with monitoring enabled
docker-compose up -d

# Access monitoring interfaces
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

### Manual Monitoring Setup

```bash
# Install monitoring tools
pip install prometheus-client psutil

# Configure application metrics
export ENABLE_METRICS=true
export METRICS_PORT=8080

# Start the bot with metrics enabled
python bot.py --enable-metrics
```

## Application Metrics

### Core Metrics

The AutoGen Bot exposes the following metrics at `/metrics`:

#### HTTP Request Metrics
- `http_requests_total`: Total HTTP requests by method and status
- `http_request_duration_seconds`: Request duration histogram
- `http_request_size_bytes`: Request size histogram
- `http_response_size_bytes`: Response size histogram

#### GitHub Integration Metrics
- `github_api_requests_total`: Total GitHub API requests
- `github_rate_limit_remaining`: Remaining GitHub API requests
- `github_rate_limit_limit`: GitHub API rate limit
- `github_webhook_events_total`: Total webhook events processed

#### Agent Performance Metrics
- `agent_conversation_duration_seconds`: Agent conversation duration
- `agent_conversation_failures_total`: Failed agent conversations
- `agent_tokens_used_total`: Tokens consumed by agents
- `code_review_completions_total`: Completed code reviews

#### Cache Performance Metrics
- `cache_hits_total`: Cache hit count
- `cache_misses_total`: Cache miss count
- `cache_operations_duration_seconds`: Cache operation duration
- `cache_size_bytes`: Current cache size

#### Queue Metrics
- `review_queue_size`: Current review queue size
- `review_processing_duration_seconds`: Review processing time
- `review_queue_wait_time_seconds`: Time items spend in queue

### Adding Custom Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Custom counter
custom_operations = Counter(
    'custom_operations_total',
    'Total custom operations',
    ['operation_type', 'status']
)

# Custom histogram
operation_duration = Histogram(
    'operation_duration_seconds',
    'Operation duration',
    ['operation']
)

# Custom gauge
active_connections = Gauge(
    'active_connections',
    'Active connections'
)

# Usage in code
custom_operations.labels(operation_type='review', status='success').inc()
with operation_duration.labels(operation='pr_analysis').time():
    # Your operation here
    pass
```

## Health Checks

### Application Health Check

The bot exposes health check endpoints:

```bash
# Basic health check
curl http://localhost:8080/health

# Detailed health check
curl http://localhost:8080/health/detailed
```

### Docker Health Check

The Dockerfile includes a built-in health check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import autogen_code_review_bot; print('OK')" || exit 1
```

## Alerting Rules

### Critical Alerts

1. **Service Down**: Bot is not responding
2. **High Error Rate**: >10% HTTP 5xx responses
3. **GitHub Rate Limit Critical**: <10 API requests remaining
4. **Memory Exhaustion**: >90% memory usage

### Warning Alerts

1. **High Response Time**: 95th percentile >30 seconds
2. **Queue Backlog**: >50 items in review queue
3. **Low Cache Hit Rate**: <70% hit rate
4. **GitHub Rate Limit Warning**: <100 API requests remaining

### Configuration

Alerts are configured in `monitoring/rules/autogen-bot.yml`:

```yaml
- alert: AutoGenBotDown
  expr: up{job="autogen-bot"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "AutoGen Bot is down"
    description: "Service has been down for more than 1 minute"
```

## Dashboards

### Grafana Dashboard

Import the dashboard from `monitoring/grafana-dashboard.json`:

1. Open Grafana (http://localhost:3000)
2. Login with admin/admin
3. Go to Dashboards → Import
4. Upload `monitoring/grafana-dashboard.json`

### Key Dashboard Panels

- **Service Status**: Up/down indicator
- **Request Rate**: HTTP requests per second
- **Response Time**: 95th percentile latency
- **Memory Usage**: Current memory consumption
- **GitHub API Rate Limit**: Remaining requests
- **Review Queue Size**: Pending reviews
- **Cache Hit Rate**: Cache performance
- **Agent Performance**: Conversation duration by agent type

## Log Aggregation

### Structured Logging

The bot uses structured JSON logging:

```python
import structlog

logger = structlog.get_logger()
logger.info("PR review completed", 
           pr_number=123, 
           repository="owner/repo",
           duration=15.2,
           agent_conversations=3)
```

### Log Levels

- **DEBUG**: Detailed execution information
- **INFO**: General operational messages
- **WARNING**: Potential issues that don't break functionality
- **ERROR**: Errors that prevent specific operations
- **CRITICAL**: Severe errors that may cause service disruption

### Log Aggregation Setup

For production environments, consider:

1. **ELK Stack**: Elasticsearch, Logstash, Kibana
2. **Fluentd**: Log collection and forwarding
3. **Loki**: Grafana's log aggregation system
4. **Cloud Solutions**: AWS CloudWatch, Azure Monitor, GCP Logging

## Performance Monitoring

### Application Performance Monitoring (APM)

Consider integrating with:

1. **OpenTelemetry**: Industry-standard observability
2. **Jaeger**: Distributed tracing
3. **Zipkin**: Request tracing
4. **DataDog**: Full-stack monitoring
5. **New Relic**: Application performance monitoring

### Custom Instrumentation

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def analyze_pr(pr_data):
    with tracer.start_as_current_span("pr_analysis") as span:
        span.set_attribute("pr.number", pr_data.number)
        span.set_attribute("pr.repository", pr_data.repo)
        
        # Your analysis logic here
        result = perform_analysis(pr_data)
        
        span.set_attribute("analysis.duration", result.duration)
        span.set_attribute("analysis.issues_found", len(result.issues))
        
        return result
```

## Troubleshooting

### Common Issues

1. **Metrics not appearing**: Check metrics endpoint and Prometheus config
2. **High memory usage**: Review cache configuration and cleanup policies
3. **Rate limit issues**: Monitor GitHub API usage and implement backoff
4. **Queue buildup**: Check agent performance and processing capacity

### Debug Commands

```bash
# Check metrics endpoint
curl http://localhost:8080/metrics

# Verify Prometheus scraping
curl http://localhost:9090/api/v1/targets

# Check application logs
docker logs autogen-bot --tail 100 -f

# Monitor resource usage
docker stats autogen-bot
```

## Best Practices

1. **Set up alerting early**: Don't wait for problems to occur
2. **Monitor business metrics**: Track review completion rates, not just technical metrics
3. **Use SLIs/SLOs**: Define service level indicators and objectives
4. **Regular dashboard reviews**: Keep dashboards relevant and actionable
5. **Alert on symptoms, not causes**: Focus on user impact
6. **Document runbooks**: Provide clear troubleshooting steps for alerts

## Integration with CI/CD

### Metrics in GitHub Actions

```yaml
- name: Report deployment metrics
  run: |
    curl -X POST "http://prometheus-pushgateway:9091/metrics/job/github-actions" \
         -d "deployment_completed{version=\"${{ github.sha }}\"} 1"
```

### Performance Testing Integration

```yaml
- name: Performance test with metrics
  run: |
    pytest tests/performance/ --benchmark-json=benchmark.json
    python scripts/upload_benchmark_metrics.py benchmark.json
```
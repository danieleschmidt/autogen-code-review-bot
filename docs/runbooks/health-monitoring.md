# Health Monitoring Runbook

## Health Check Endpoints

### `/health`
- **Purpose**: Basic application health check
- **Expected Response**: `{"status": "healthy", "timestamp": "2025-07-27T12:00:00Z"}`
- **Timeout**: 10 seconds
- **Frequency**: Every 30 seconds

### `/metrics`
- **Purpose**: Prometheus metrics endpoint
- **Format**: Prometheus exposition format
- **Key Metrics**:
  - `autogen_requests_total` - Total requests processed
  - `autogen_processing_duration_seconds` - Request processing time
  - `autogen_cache_hits_total` - Cache hit rate
  - `autogen_errors_total` - Error count by type

## Monitoring Scenarios

### High Memory Usage
**Alert**: Memory usage > 80%
**Investigation**:
1. Check `/metrics` for memory metrics
2. Review application logs for memory leaks
3. Check cache size and cleanup policies

### High Response Time
**Alert**: Response time > 30 seconds
**Investigation**:
1. Check OpenAI API latency
2. Review agent conversation logs
3. Analyze linter execution times

### Error Rate Spike
**Alert**: Error rate > 5%
**Investigation**:
1. Check error logs for patterns
2. Verify external service availability
3. Review recent deployments
# Deployment Guide

Comprehensive guide for deploying AutoGen Code Review Bot in various environments.

## Deployment Options

### 1. GitHub Actions (Recommended)
Deploy as a GitHub Action for seamless integration with your repository workflow.

#### Setup Steps
1. **Create Repository Secrets**:
   ```
   GITHUB_TOKEN: <your-github-token>
   OPENAI_API_KEY: <your-openai-key>  # If using OpenAI models
   ```

2. **Add Workflow File**:
   ```yaml
   # .github/workflows/code-review.yml
   name: AutoGen Code Review
   on:
     pull_request:
       types: [opened, synchronize, reopened]
   
   jobs:
     review:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: terragon-labs/autogen-review-bot@v1
           with:
             github-token: ${{ secrets.GITHUB_TOKEN }}
             config-path: '.github/review-config.yaml'
   ```

### 2. Docker Deployment
Deploy using Docker for maximum flexibility and isolation.

#### Using Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  autogen-review-bot:
    image: ghcr.io/terragon-labs/autogen-review-bot:latest
    environment:
      - GITHUB_TOKEN=${GITHUB_TOKEN}
      - WEBHOOK_SECRET=${WEBHOOK_SECRET}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    ports:
      - "8000:8000"
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

#### Direct Docker Run
```bash
docker run -d \
  --name autogen-review-bot \
  -p 8000:8000 \
  -e GITHUB_TOKEN=$GITHUB_TOKEN \
  -e WEBHOOK_SECRET=$WEBHOOK_SECRET \
  -v $(pwd)/config:/app/config:ro \
  ghcr.io/terragon-labs/autogen-review-bot:latest
```

### 3. Kubernetes Deployment
Enterprise-grade deployment with scaling and reliability.

#### Deployment Manifest
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autogen-review-bot
  namespace: default
spec:
  replicas: 3
  selector:
    matchLabels:
      app: autogen-review-bot
  template:
    metadata:
      labels:
        app: autogen-review-bot
    spec:
      containers:
      - name: bot
        image: ghcr.io/terragon-labs/autogen-review-bot:latest
        ports:
        - containerPort: 8000
        env:
        - name: GITHUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: github-secrets
              key: token
        - name: WEBHOOK_SECRET
          valueFrom:
            secretKeyRef:
              name: github-secrets
              key: webhook-secret
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: autogen-review-bot-service
spec:
  selector:
    app: autogen-review-bot
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### 4. Serverless Deployment
Cost-effective deployment using serverless platforms.

#### AWS Lambda
```python
# lambda_handler.py
import json
from autogen_code_review_bot import handle_webhook

def lambda_handler(event, context):
    try:
        result = handle_webhook(
            payload=event['body'],
            signature=event['headers'].get('X-Hub-Signature-256'),
            secret=os.environ['WEBHOOK_SECRET']
        )
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

#### Vercel/Netlify Functions
```javascript
// api/webhook.js
const { execSync } = require('child_process');

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const result = execSync('python -m autogen_code_review_bot.webhook', {
      input: JSON.stringify(req.body),
      encoding: 'utf-8',
      env: { ...process.env, ...req.body }
    });
    
    res.status(200).json(JSON.parse(result));
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}
```

## Configuration Management

### Environment Variables
```bash
# Core Configuration
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
WEBHOOK_SECRET=your-secure-webhook-secret
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx  # Optional

# Performance Tuning
MAX_WORKERS=4
CACHE_TTL_HOURS=24
MAX_PR_FILES=100

# Monitoring
PROMETHEUS_ENABLED=true
LOG_LEVEL=INFO
SENTRY_DSN=https://xxxx@sentry.io/xxxx  # Optional
```

### Configuration Files
```yaml
# config/production.yaml
agents:
  coder:
    model: "gpt-4"
    temperature: 0.3
    max_tokens: 2000
  reviewer:
    model: "gpt-4"
    temperature: 0.1
    max_tokens: 2500

github:
  webhook_secret: "${WEBHOOK_SECRET}"
  bot_token: "${GITHUB_TOKEN}"
  max_files: 100
  timeout: 300

performance:
  cache_enabled: true
  parallel_processing: true
  max_workers: ${MAX_WORKERS:-4}

monitoring:
  prometheus: ${PROMETHEUS_ENABLED:-false}
  log_level: ${LOG_LEVEL:-INFO}
```

## Security Considerations

### Secrets Management
1. **Never commit secrets to version control**
2. **Use environment variables or secret management systems**
3. **Rotate tokens regularly**
4. **Use least-privilege access principles**

### Network Security
```yaml
# Network security policies for Kubernetes
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: autogen-review-bot-netpol
spec:
  podSelector:
    matchLabels:
      app: autogen-review-bot
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []  # Allow all outbound for GitHub API calls
    ports:
    - protocol: TCP
      port: 443
```

### Container Security
```dockerfile
# Security-hardened Dockerfile example
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
RUN groupadd -r appuser && useradd -r -g appuser appuser
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .
RUN chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["python", "-m", "autogen_code_review_bot"]
```

## Monitoring and Observability

### Health Checks
```python
# Health check endpoints
@app.route('/health')
def health_check():
    return {'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()}

@app.route('/ready')
def readiness_check():
    # Check dependencies (GitHub API, database, etc.)
    return {'status': 'ready', 'dependencies': check_dependencies()}
```

### Metrics Collection
```yaml
# Prometheus monitoring configuration
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'autogen-review-bot'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['autogen-review-bot:8000']
```

### Logging Configuration
```yaml
# logging.yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  json:
    format: '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'

handlers:
  console:
    class: logging.StreamHandler
    formatter: default
  file:
    class: logging.handlers.RotatingFileHandler
    filename: logs/bot.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    formatter: json

loggers:
  autogen_code_review_bot:
    level: INFO
    handlers: [console, file]
```

## Troubleshooting

### Common Issues

#### 1. Webhook Not Receiving Events
```bash
# Check webhook configuration
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/owner/repo/hooks

# Test webhook endpoint
curl -X POST http://your-bot-url/webhook -H "Content-Type: application/json" -d '{}'
```

#### 2. Authentication Failures
```bash
# Verify GitHub token permissions
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user

# Check token scopes
curl -I -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user
```

#### 3. Performance Issues
```bash
# Check resource usage
docker stats autogen-review-bot

# Monitor response times
curl -w "@curl-format.txt" -o /dev/null -s http://your-bot-url/health
```

### Debug Mode
```yaml
# Enable debug mode in configuration
debug:
  enabled: true
  log_level: DEBUG
  save_requests: true
  request_log_path: "/tmp/requests"
```

### Support and Maintenance
- **Documentation**: Check [docs/](../docs/) for detailed guides
- **Issues**: Report issues on GitHub Issues
- **Security**: Report security issues to security@terragon-labs.com
- **Updates**: Follow release notes for upgrade instructions

For enterprise support and custom deployment assistance, contact Terragon Labs Professional Services.
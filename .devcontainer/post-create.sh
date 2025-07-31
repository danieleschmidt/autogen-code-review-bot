#!/bin/bash
# Post-create script for AutoGen Code Review Bot development environment

set -e

echo "🚀 Setting up AutoGen Code Review Bot development environment..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -e .[dev,monitoring]

# Install additional development tools
echo "🔧 Installing additional development tools..."
pip install \
    bump2version \
    cyclonedx-bom \
    pip-licenses \
    licensecheck \
    safety \
    semgrep \
    gitpython

# Set up pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Set up git configuration for development
echo "⚙️ Configuring git..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input

# Create development directories
echo "📁 Creating development directories..."
mkdir -p logs tmp data/{prometheus,grafana,alertmanager,loki}

# Set up development secrets baseline
echo "🔐 Setting up secrets detection baseline..."
if [ ! -f .secrets.baseline ]; then
    detect-secrets scan --baseline .secrets.baseline
fi

# Install GitHub CLI extensions
echo "🐙 Installing GitHub CLI extensions..."
gh extension install github/gh-copilot || true

# Set up shell environment
echo "🐚 Setting up shell environment..."
cat >> ~/.zshrc << 'EOF'

# AutoGen Bot development aliases
alias agb-test="pytest -v --cov=src --cov-report=term-missing"
alias agb-lint="ruff check src tests && mypy src"
alias agb-format="ruff format src tests && isort src tests"
alias agb-security="bandit -r src && safety check"
alias agb-start="python bot.py --config config/development.yaml"
alias agb-monitor="docker-compose -f monitoring/enterprise-monitoring.yml up -d"

# Development shortcuts
alias ll="ls -la"
alias la="ls -A"
alias l="ls -CF"
alias grep="grep --color=auto"

# Export environment variables
export ENVIRONMENT=development
export LOG_LEVEL=DEBUG
export PYTHONPATH=/workspace/src
EOF

# Create development configuration
echo "⚙️ Creating development configuration..."
mkdir -p config
cat > config/development.yaml << 'EOF'
# Development Configuration for AutoGen Code Review Bot
environment: development
log_level: DEBUG

agents:
  coder:
    model: "gpt-4"
    temperature: 0.3
    max_tokens: 2000
    focus_areas: ["functionality", "bugs", "edge_cases"]
  
  reviewer:
    model: "gpt-4"
    temperature: 0.1
    max_tokens: 2000
    focus_areas: ["security", "performance", "standards"]

github:
  api_url: "https://api.github.com"
  webhook_secret: "dev-webhook-secret"
  rate_limit_buffer: 100

redis:
  url: "redis://localhost:6379"
  db: 0
  max_connections: 20

review_criteria:
  security_scan: true
  performance_check: true
  test_coverage: true
  documentation: true
  code_style: true

cache:
  enabled: true
  ttl_hours: 24
  max_size_mb: 100

monitoring:
  enabled: true
  metrics_port: 8001
  health_check_port: 8002
EOF

# Set up monitoring stack for development
echo "📊 Setting up monitoring configuration..."
cat > docker-compose.dev.yml << 'EOF'
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
EOF

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  agb-test      - Run tests with coverage"
echo "  agb-lint      - Run linting and type checking"
echo "  agb-format    - Format code with ruff and isort"
echo "  agb-security  - Run security scans"
echo "  agb-start     - Start the bot in development mode"
echo "  agb-monitor   - Start monitoring stack"
echo ""
echo "📚 Documentation: https://docs.company.com/autogen-bot"
echo "🆘 Support: #autogen-bot-dev Slack channel"
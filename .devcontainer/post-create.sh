#!/bin/bash

set -e

echo "🚀 Setting up AutoGen Code Review Bot development environment..."

# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install additional development tools
sudo apt-get install -y \
    curl \
    wget \
    jq \
    tree \
    htop \
    git-extras \
    shellcheck

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install the package in development mode
pip install -e ".[dev,monitoring]"

# Install additional development tools
pip install \
    ipython \
    jupyter \
    pre-commit \
    tox \
    mypy \
    types-PyYAML \
    types-requests

# Setup pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Install language-specific linters for testing
echo "🔍 Installing language-specific linters..."

# JavaScript/TypeScript tools
npm install -g \
    eslint \
    prettier \
    @typescript-eslint/parser \
    @typescript-eslint/eslint-plugin

# Go tools (if Go is available)
if command -v go &> /dev/null; then
    go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
    go install github.com/securecodewarrior/sast-scan@latest
fi

# Rust tools (if Rust is available)
if command -v cargo &> /dev/null; then
    cargo install clippy
    cargo install cargo-audit
fi

# Ruby tools (if Ruby is available)
if command -v gem &> /dev/null; then
    gem install rubocop
    gem install brakeman
fi

# Create necessary directories
mkdir -p \
    logs \
    tmp \
    .cache/autogen-review \
    docs/runbooks \
    tests/integration \
    tests/e2e \
    tests/performance

# Setup git configuration for consistent development
git config --global init.defaultBranch main
git config --global pull.rebase false

# Create development configuration
if [ ! -f config/development.yaml ]; then
    mkdir -p config
    cat > config/development.yaml << EOF
# Development configuration for AutoGen Code Review Bot
agents:
  coder:
    model: "gpt-3.5-turbo"  # Use cheaper model for development
    temperature: 0.3
    focus_areas: ["functionality", "bugs", "edge_cases"]
  
  reviewer:
    model: "gpt-3.5-turbo"  # Use cheaper model for development
    temperature: 0.1
    focus_areas: ["security", "performance", "standards"]

github:
  webhook_secret: "dev_webhook_secret"
  bot_token: "dev_bot_token"
  api_url: "https://api.github.com"

review_criteria:
  security_scan: true
  performance_check: true
  test_coverage: true
  documentation: true

cache:
  enabled: true
  directory: ".cache/autogen-review"
  ttl_hours: 24

logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/development.log"

performance:
  parallel_execution: true
  max_workers: 4
  timeout_seconds: 300
EOF
fi

# Create local environment file template
if [ ! -f .env.example ]; then
    cat > .env.example << EOF
# GitHub Configuration
GITHUB_TOKEN=your_github_token_here
GITHUB_WEBHOOK_SECRET=your_webhook_secret_here

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here
AZURE_OPENAI_API_KEY=your_azure_api_key_here

# Bot Configuration
BOT_CONFIG_PATH=config/development.yaml
LOG_LEVEL=DEBUG
CACHE_ENABLED=true
PARALLEL_EXECUTION=true

# Development Settings
FLASK_ENV=development
FLASK_DEBUG=true
PORT=5000
EOF
fi

# Setup test data and fixtures
echo "🧪 Setting up test fixtures..."
mkdir -p tests/fixtures/repositories
mkdir -p tests/fixtures/pull_requests

# Create sample test repository structure
mkdir -p tests/fixtures/repositories/sample-python
cat > tests/fixtures/repositories/sample-python/main.py << EOF
def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
EOF

cat > tests/fixtures/repositories/sample-python/requirements.txt << EOF
requests>=2.31.0
pyyaml>=6.0.0
EOF

# Make the script executable
chmod +x .devcontainer/post-create.sh

echo "✅ Development environment setup complete!"
echo ""
echo "🔧 Available commands:"
echo "  - make test          # Run all tests"
echo "  - make lint          # Run linting"
echo "  - make format        # Format code"
echo "  - make dev           # Start development server"
echo "  - make clean         # Clean cache and temporary files"
echo ""
echo "📚 Documentation:"
echo "  - README.md          # Project overview"
echo "  - ARCHITECTURE.md    # System architecture"
echo "  - docs/DEVELOPMENT.md # Development guide"
echo ""
echo "🚀 Happy coding!"
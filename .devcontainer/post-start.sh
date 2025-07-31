#!/bin/bash
# Post-start script for AutoGen Code Review Bot development environment

set -e

echo "🌟 Starting AutoGen Code Review Bot development session..."

# Check system health
echo "🔍 Checking system health..."
python --version
pip --version
git --version
gh --version || echo "⚠️  GitHub CLI not available"

# Verify dependencies
echo "📋 Verifying dependencies..."
pip list | grep -E "(pytest|ruff|mypy|bandit)" || echo "⚠️  Some development dependencies may be missing"

# Check pre-commit setup
echo "🪝 Checking pre-commit setup..."
pre-commit --version || echo "⚠️  Pre-commit not available"

# Start background services if not running
echo "🚀 Starting background services..."

# Check if Redis is available
if ! nc -z localhost 6379 2>/dev/null; then
    echo "📡 Starting Redis for development..."
    docker run -d --name dev-redis -p 6379:6379 redis:7-alpine redis-server --appendonly yes || echo "⚠️  Could not start Redis"
fi

# Display service status
echo "📊 Service Status:"
echo "  Redis: $(nc -z localhost 6379 && echo '✅ Running' || echo '❌ Not available')"
echo "  Docker: $(docker version --format '{{.Server.Version}}' 2>/dev/null && echo '✅ Available' || echo '❌ Not available')"

# Show useful development information
echo ""
echo "🔧 Development Environment Ready!"
echo ""
echo "📁 Project Structure:"
echo "  src/              - Source code"
echo "  tests/            - Test suites"
echo "  docs/             - Documentation"
echo "  .github/          - GitHub workflows"
echo "  monitoring/       - Monitoring configuration"
echo ""
echo "🛠️  Available Commands:"
echo "  make test         - Run test suite"
echo "  make lint         - Run linters"
echo "  make format       - Format code"
echo "  make security     - Security checks"
echo "  make docs         - Generate documentation"
echo ""
echo "🐙 GitHub Integration:"
echo "  gh auth status    - Check GitHub CLI authentication"
echo "  gh repo view      - View repository information"
echo ""
echo "🔍 Monitoring:"
echo "  http://localhost:3000  - Grafana (admin/admin)"
echo "  http://localhost:9090  - Prometheus"
echo "  http://localhost:8000  - Bot API (when running)"
echo ""
echo "💡 Tips:"
echo "  - Use 'agb-*' aliases for common tasks"
echo "  - Check DEVELOPMENT.md for detailed setup"
echo "  - Join #autogen-bot-dev Slack for support"
echo ""
echo "Happy coding! 🚀"
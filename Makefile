# AutoGen Code Review Bot - Enterprise Development Makefile
# Provides common development, testing, and deployment commands

.PHONY: help install test lint format security clean build deploy monitor docs
.DEFAULT_GOAL := help

# Configuration
PYTHON := python3
PIP := pip3
PROJECT_NAME := autogen-code-review-bot
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
RED := \033[31m
YELLOW := \033[33m
RESET := \033[0m

help: ## Show this help message
	@echo "$(BLUE)AutoGen Code Review Bot - Development Commands$(RESET)"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

# Installation and Setup
install: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev,monitoring]
	pre-commit install
	@echo "$(GREEN)✅ Installation complete$(RESET)"

install-prod: ## Install production dependencies only
	@echo "$(BLUE)Installing production dependencies...$(RESET)"
	$(PIP) install --upgrade pip
	$(PIP) install -e .
	@echo "$(GREEN)✅ Production installation complete$(RESET)"

# Development and Testing
test: ## Run test suite with coverage
	@echo "$(BLUE)Running test suite...$(RESET)"
	pytest -v --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html --cov-fail-under=95
	@echo "$(GREEN)✅ Tests completed$(RESET)"

test-fast: ## Run tests without coverage (faster)
	@echo "$(BLUE)Running fast test suite...$(RESET)"
	pytest -v -x
	@echo "$(GREEN)✅ Fast tests completed$(RESET)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	pytest -v -m integration
	@echo "$(GREEN)✅ Integration tests completed$(RESET)"

test-benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(RESET)"
	cd benchmarks && python -m pytest test_performance.py --benchmark-only
	@echo "$(GREEN)✅ Benchmarks completed$(RESET)"

# Code Quality
lint: ## Run all linters
	@echo "$(BLUE)Running linters...$(RESET)"
	ruff check $(SRC_DIR) $(TEST_DIR)
	mypy $(SRC_DIR) --strict --ignore-missing-imports
	bandit -r $(SRC_DIR) -ll
	@echo "$(GREEN)✅ Linting completed$(RESET)"

format: ## Format code with ruff and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	ruff format $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✅ Code formatted$(RESET)"

format-check: ## Check if code is properly formatted
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	ruff format --check $(SRC_DIR) $(TEST_DIR)
	isort --check-only $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✅ Format check completed$(RESET)"

# Security
security: ## Run comprehensive security checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	bandit -r $(SRC_DIR) -f json -o security/bandit-report.json || true
	safety check --json --output security/safety-report.json || true
	detect-secrets scan --baseline .secrets.baseline
	@echo "$(GREEN)✅ Security checks completed$(RESET)"

security-update: ## Update security baseline
	@echo "$(BLUE)Updating security baseline...$(RESET)"
	detect-secrets scan --update .secrets.baseline
	@echo "$(GREEN)✅ Security baseline updated$(RESET)"

# Pre-commit and Git
pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files
	@echo "$(GREEN)✅ Pre-commit checks completed$(RESET)"

pre-commit-update: ## Update pre-commit hooks
	@echo "$(BLUE)Updating pre-commit hooks...$(RESET)"
	pre-commit autoupdate
	@echo "$(GREEN)✅ Pre-commit hooks updated$(RESET)"

# Build and Release
build: ## Build package for distribution
	@echo "$(BLUE)Building package...$(RESET)"
	$(PYTHON) -m build
	twine check dist/*
	@echo "$(GREEN)✅ Package built successfully$(RESET)"

build-docker: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -t $(PROJECT_NAME):latest .
	docker build -t $(PROJECT_NAME):dev -f Dockerfile.dev .
	@echo "$(GREEN)✅ Docker images built$(RESET)"

clean: ## Clean build artifacts and cache files
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✅ Cleanup completed$(RESET)"

# Development Environment
dev-setup: install ## Complete development environment setup
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	mkdir -p logs security data/{prometheus,grafana,alertmanager}
	docker-compose -f docker-compose.dev.yml up -d redis
	@echo "$(GREEN)✅ Development environment ready$(RESET)"

dev-start: ## Start the bot in development mode
	@echo "$(BLUE)Starting AutoGen Bot in development mode...$(RESET)"
	$(PYTHON) bot.py --config config/development.yaml

dev-stop: ## Stop development services
	@echo "$(BLUE)Stopping development services...$(RESET)"
	docker-compose -f docker-compose.dev.yml down
	@echo "$(GREEN)✅ Development services stopped$(RESET)"

# Monitoring and Observability
monitor: ## Start monitoring stack
	@echo "$(BLUE)Starting monitoring stack...$(RESET)"
	docker-compose -f monitoring/enterprise-monitoring.yml up -d
	@echo "$(GREEN)✅ Monitoring stack started$(RESET)"
	@echo "$(YELLOW)Grafana: http://localhost:3000 (admin/admin)$(RESET)"
	@echo "$(YELLOW)Prometheus: http://localhost:9090$(RESET)"

monitor-stop: ## Stop monitoring stack
	@echo "$(BLUE)Stopping monitoring stack...$(RESET)"
	docker-compose -f monitoring/enterprise-monitoring.yml down
	@echo "$(GREEN)✅ Monitoring stack stopped$(RESET)"

logs: ## View application logs
	@echo "$(BLUE)Viewing application logs...$(RESET)"
	tail -f logs/*.log 2>/dev/null || echo "No log files found"

# Documentation
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(RESET)"
	sphinx-build -b html $(DOCS_DIR) $(DOCS_DIR)/_build/html
	@echo "$(GREEN)✅ Documentation generated$(RESET)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(RESET)"
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8080

# Compliance and Reporting
compliance: ## Generate compliance reports
	@echo "$(BLUE)Generating compliance reports...$(RESET)"
	cyclonedx-py -o security/sbom.json --format json
	pip-licenses --format=json --output-file=security/licenses.json
	@echo "$(GREEN)✅ Compliance reports generated$(RESET)"

# CI/CD Simulation
ci-local: format lint test security ## Run full CI pipeline locally
	@echo "$(BLUE)Running full CI pipeline locally...$(RESET)"
	@echo "$(GREEN)✅ Local CI pipeline completed successfully$(RESET)"

# Version Management
version-patch: ## Bump patch version
	@echo "$(BLUE)Bumping patch version...$(RESET)"
	bump2version patch
	@echo "$(GREEN)✅ Patch version bumped$(RESET)"

version-minor: ## Bump minor version
	@echo "$(BLUE)Bumping minor version...$(RESET)"
	bump2version minor
	@echo "$(GREEN)✅ Minor version bumped$(RESET)"

version-major: ## Bump major version
	@echo "$(BLUE)Bumping major version...$(RESET)"
	bump2version major
	@echo "$(GREEN)✅ Major version bumped$(RESET)"

# Database and Dependencies
deps-check: ## Check for dependency updates
	@echo "$(BLUE)Checking for dependency updates...$(RESET)"
	pip list --outdated
	@echo "$(GREEN)✅ Dependency check completed$(RESET)"

deps-update: ## Update dependencies (be careful!)
	@echo "$(YELLOW)⚠️  This will update all dependencies. Continue? [y/N]$(RESET)"
	@read -r confirm && [ "$$confirm" = "y" ] || exit 1
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	pip-review --auto
	@echo "$(GREEN)✅ Dependencies updated$(RESET)"

# GitHub Integration
gh-setup: ## Setup GitHub CLI and create initial PR template
	@echo "$(BLUE)Setting up GitHub integration...$(RESET)"
	gh auth status || gh auth login
	@echo "$(GREEN)✅ GitHub CLI configured$(RESET)"

# Performance Analysis
profile: ## Run performance profiling
	@echo "$(BLUE)Running performance profiling...$(RESET)"
	$(PYTHON) -m cProfile -o profile.stats bot.py --profile
	@echo "$(GREEN)✅ Profiling completed - check profile.stats$(RESET)"

# All-in-one commands
all: clean install lint test security build ## Run complete development cycle
	@echo "$(GREEN)✅ Complete development cycle finished$(RESET)"

init: dev-setup pre-commit-update ## Initialize project for new developers
	@echo "$(GREEN)✅ Project initialization completed$(RESET)"
	@echo "$(YELLOW)Next steps:$(RESET)"
	@echo "  1. Copy .env.example to .env and configure"
	@echo "  2. Run 'make dev-start' to start the bot"
	@echo "  3. Check the README.md for more information"
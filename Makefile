# AutoGen Code Review Bot - Makefile
# Standardized commands for development and deployment

.PHONY: help install install-dev test lint format security clean build docs run dev

# Default target
help: ## Show this help message
	@echo "AutoGen Code Review Bot - Available Commands:"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Environment Variables:"
	@echo "  PYTHON_VERSION  Python version to use (default: 3.11)"
	@echo "  ENV            Environment: dev, test, prod (default: dev)"

# Python and environment configuration
PYTHON_VERSION ?= 3.11
PYTHON := python$(PYTHON_VERSION)
PIP := $(PYTHON) -m pip
ENV ?= dev

# Installation targets
install: ## Install package and production dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .

install-dev: ## Install package with development dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev,monitoring,test,docs,performance]"
	pre-commit install
	pre-commit install --hook-type commit-msg

install-ci: ## Install for CI/CD environment
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[test,monitoring]"

# Testing targets
test: ## Run all tests with coverage
	pytest -v --cov=autogen_code_review_bot --cov-report=term-missing --cov-report=html --cov-fail-under=80

test-unit: ## Run only unit tests
	pytest tests/ -m "unit" -v

test-integration: ## Run only integration tests
	pytest tests/ -m "integration" -v

test-security: ## Run security tests
	pytest tests/ -m "security" -v
	bandit -r src/ -f json -o security-report.json

test-performance: ## Run performance tests
	pytest tests/ -m "performance" -v

test-parallel: ## Run tests in parallel
	pytest -n auto --cov=autogen_code_review_bot --cov-report=term-missing

# Code quality targets
lint: ## Run all linting checks
	ruff check src/ tests/
	mypy src/
	bandit -r src/ -q

lint-fix: ## Fix auto-fixable linting issues
	ruff check --fix src/ tests/
	isort src/ tests/

format: ## Format code using black and isort
	black src/ tests/
	isort src/ tests/

format-check: ## Check code formatting without making changes
	black --check src/ tests/
	isort --check-only src/ tests/

security: ## Run security scans
	bandit -r src/
	safety check
	detect-secrets scan --all-files

# Build targets
clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build package for distribution
	$(PYTHON) -m build

build-docker: ## Build Docker image
	docker build -t autogen-code-review-bot:latest .
	docker build -t autogen-code-review-bot:$(shell git rev-parse --short HEAD) .

# Documentation targets
docs: ## Generate documentation
	mkdocs build

docs-serve: ## Serve documentation locally
	mkdocs serve

docs-deploy: ## Deploy documentation to GitHub Pages
	mkdocs gh-deploy

# Development targets
dev: ## Start development server with hot reload
	FLASK_ENV=development FLASK_DEBUG=true $(PYTHON) bot.py --config config/development.yaml

run: ## Run the bot with production settings
	$(PYTHON) bot.py --config config/production.yaml

# Database targets (if applicable)
db-upgrade: ## Run database migrations
	alembic upgrade head

db-downgrade: ## Rollback database migrations
	alembic downgrade -1

# Monitoring and health targets
health-check: ## Check application health
	curl -f http://localhost:8080/health || exit 1

metrics: ## Display application metrics
	curl http://localhost:9090/metrics

# Git and release targets
pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

bump-version: ## Bump version (use VERSION=patch|minor|major)
	bump2version $(VERSION)

release: ## Create a new release
	@echo "Creating release..."
	git tag -a v$(shell grep version pyproject.toml | head -1 | cut -d'"' -f2) -m "Release v$(shell grep version pyproject.toml | head -1 | cut -d'"' -f2)"
	git push origin v$(shell grep version pyproject.toml | head -1 | cut -d'"' -f2)

# Deployment targets
deploy-staging: build ## Deploy to staging environment
	@echo "Deploying to staging..."
	# Add staging deployment commands here

deploy-prod: build ## Deploy to production environment
	@echo "Deploying to production..."
	# Add production deployment commands here

# Utility targets
requirements: ## Generate requirements.txt for deployment
	$(PIP) freeze > requirements.txt

check-deps: ## Check for outdated dependencies
	$(PIP) list --outdated

update-deps: ## Update development dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e ".[dev,monitoring,test,docs,performance]"

# CI/CD simulation
ci: clean install-ci lint test security ## Run CI pipeline locally
	@echo "✅ CI pipeline completed successfully"

cd: ci build ## Run CD pipeline locally
	@echo "✅ CD pipeline completed successfully"

# Environment setup
setup-env: ## Setup development environment
	cp .env.example .env
	@echo "✅ Environment file created. Please edit .env with your settings."

init: install-dev setup-env ## Initialize development environment
	@echo "✅ Development environment initialized"
	@echo "🚀 Run 'make dev' to start the development server"

# Performance profiling
profile: ## Profile application performance
	py-spy record -o profile.svg -- $(PYTHON) bot.py --config config/development.yaml

benchmark: ## Run performance benchmarks
	locust -f tests/performance/locustfile.py --headless -u 10 -r 2 -t 30s

# Troubleshooting
debug: ## Start application in debug mode
	$(PYTHON) -m pdb bot.py --config config/development.yaml

logs: ## View application logs
	tail -f logs/development.log

# All-in-one targets
all: clean install-dev lint test security build docs ## Run complete development workflow

ready: pre-commit test security ## Check if code is ready for commit/PR
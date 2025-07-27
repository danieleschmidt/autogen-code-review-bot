.PHONY: help install dev-install test lint format security clean build docker run-dev run-prod logs stop health check-deps

# Default target
help: ## Show this help message
	@echo "AutoGen Code Review Bot - Available Commands:"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development setup
install: ## Install dependencies
	pip install -e .

dev-install: ## Install development dependencies
	pip install -e .[dev,monitoring]
	pre-commit install

# Code quality
test: ## Run tests
	pytest -v --cov=src --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage
	pytest -v -x

test-integration: ## Run integration tests only
	pytest -v -m integration

test-performance: ## Run performance tests only
	pytest -v -m performance

lint: ## Run linting
	ruff check src tests
	mypy src
	bandit -r src

format: ## Format code
	ruff format src tests
	ruff check --fix src tests

security: ## Run security checks
	bandit -r src -f json -o bandit-report.json
	detect-secrets scan --baseline .secrets.baseline
	safety check

# Build and packaging
clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

build: ## Build Python package
	python -m build

docker-build: ## Build Docker image
	docker build -t autogen-code-review-bot:latest .

docker-build-dev: ## Build Docker image for development
	docker build --target builder -t autogen-code-review-bot:dev .

# Docker operations
run-dev: ## Run development environment with Docker Compose
	docker-compose --profile dev up -d

run-prod: ## Run production environment with Docker Compose
	docker-compose up -d

logs: ## Show Docker Compose logs
	docker-compose logs -f

stop: ## Stop Docker Compose services
	docker-compose down

stop-clean: ## Stop and remove volumes
	docker-compose down -v

health: ## Check service health
	@echo "Checking application health..."
	@curl -f http://localhost:8000/health || echo "Application not healthy"
	@echo "Checking metrics endpoint..."
	@curl -f http://localhost:8080/metrics || echo "Metrics not available"

# Database operations
db-migrate: ## Run database migrations
	python scripts/migrate.py

db-seed: ## Seed database with test data
	python scripts/seed_db.py

db-reset: ## Reset database (development only)
	docker-compose exec postgres psql -U postgres -d autogen_review -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
	$(MAKE) db-migrate

# Monitoring
metrics: ## View metrics dashboard
	@echo "Opening Grafana dashboard at http://localhost:3000"
	@echo "Default credentials: admin/admin"

prometheus: ## View Prometheus at http://localhost:9090
	@echo "Opening Prometheus at http://localhost:9090"

# Development helpers
check-deps: ## Check for dependency vulnerabilities
	safety check
	pip-audit

update-deps: ## Update dependencies
	pip-compile --upgrade requirements.in
	pip-compile --upgrade requirements-dev.in

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

release-check: ## Check if ready for release
	@echo "Running release checks..."
	$(MAKE) test
	$(MAKE) lint
	$(MAKE) security
	@echo "✅ All checks passed - ready for release"

# Documentation
docs: ## Generate documentation
	@echo "Documentation generation not yet implemented"

# CI/CD helpers
ci-test: ## Run CI test suite
	pytest -v --cov=src --cov-report=xml --cov-fail-under=95 -m "not slow"

ci-security: ## Run CI security checks
	bandit -r src -f json -o bandit-report.json
	detect-secrets scan --baseline .secrets.baseline

ci-build: ## Build for CI
	python -m build
	docker build -t autogen-code-review-bot:ci .

# Performance testing
perf-test: ## Run performance tests
	pytest -v tests/performance/ --durations=10

load-test: ## Run load tests (requires additional setup)
	@echo "Load testing not yet implemented"

# Cleanup
deep-clean: clean ## Deep clean including Docker
	docker system prune -f
	docker volume prune -f
	docker image prune -f
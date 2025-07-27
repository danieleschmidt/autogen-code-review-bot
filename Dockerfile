# Multi-stage Docker build for AutoGen Code Review Bot
# Optimized for security, size, and performance

#
# Stage 1: Build stage
#
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Label the image
LABEL org.opencontainers.image.title="AutoGen Code Review Bot"
LABEL org.opencontainers.image.description="Two-agent coder + reviewer loop using Microsoft AutoGen for automated PR critiques"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.vendor="Terragon Labs"
LABEL org.opencontainers.image.url="https://github.com/terragonlabs/autogen-code-review-bot"
LABEL org.opencontainers.image.source="https://github.com/terragonlabs/autogen-code-review-bot"
LABEL org.opencontainers.image.licenses="MIT"

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for building
RUN groupadd -r botuser && useradd -r -g botuser botuser

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml setup.py ./
COPY README.md LICENSE ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir build

# Build the package
COPY src/ src/
RUN python -m build --wheel

#
# Stage 2: Runtime stage
#
FROM python:3.11-slim as runtime

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    # Essential tools
    ca-certificates \
    curl \
    git \
    # Language-specific linters and tools
    nodejs \
    npm \
    # Security tools
    gnupg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install language-specific tools
RUN npm install -g \
    eslint \
    prettier \
    @typescript-eslint/parser \
    @typescript-eslint/eslint-plugin

# Create non-root user
RUN groupadd -r botuser && useradd -r -g botuser -d /home/botuser -m botuser

# Set working directory
WORKDIR /app

# Copy built wheel from builder stage
COPY --from=builder /app/dist/*.whl ./

# Install the application
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ./*.whl[monitoring] && \
    rm ./*.whl

# Copy application files
COPY bot.py review_pr.py setup_webhook.py planner.py ./
COPY config/ config/

# Create necessary directories
RUN mkdir -p \
    /app/logs \
    /app/cache \
    /app/tmp \
    /home/botuser/.cache/autogen-review && \
    chown -R botuser:botuser /app /home/botuser

# Create health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:${PORT:-5000}/health || exit 1' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=5000
ENV WORKERS=1
ENV LOG_LEVEL=INFO
ENV CACHE_ENABLED=true

# Switch to non-root user
USER botuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["/app/healthcheck.sh"]

# Default command
CMD ["python", "bot.py", "--config", "config/production.yaml"]

#
# Stage 3: Development variant
#
FROM runtime as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov ruff bandit mypy pre-commit

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    strace \
    && rm -rf /var/lib/apt/lists/*

# Copy development configuration
COPY .env.example .env
COPY tests/ tests/

# Set development environment
ENV FLASK_ENV=development
ENV FLASK_DEBUG=true
ENV LOG_LEVEL=DEBUG

# Switch back to botuser
USER botuser

# Development command
CMD ["python", "bot.py", "--config", "config/development.yaml"]

#
# Stage 4: Testing variant
#
FROM development as testing

USER root

# Install testing dependencies
RUN pip install --no-cache-dir \
    pytest-xdist \
    pytest-mock \
    pytest-asyncio \
    coverage \
    locust

# Copy test configuration
COPY pytest.ini tox.ini ./

USER botuser

# Run tests by default
CMD ["pytest", "tests/", "-v", "--cov=autogen_code_review_bot"]
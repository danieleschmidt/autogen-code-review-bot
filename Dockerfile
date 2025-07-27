# Multi-stage build for AutoGen Code Review Bot
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILDX_QEMU_ENV
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml setup.py ./
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e .[dev,monitoring]

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --from=builder /app .
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy additional files
COPY bot.py planner.py review_pr.py setup_webhook.py ./
COPY agent_config.yaml ./

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Labels for metadata
LABEL maintainer="AutoGen Team"
LABEL version="0.0.1"
LABEL description="AutoGen Code Review Bot - AI-powered PR analysis"
LABEL org.opencontainers.image.source="https://github.com/your-org/autogen-code-review-bot"
LABEL org.opencontainers.image.description="Dual-agent code review system using Microsoft AutoGen"
LABEL org.opencontainers.image.licenses="MIT"

# Default command
CMD ["python", "bot.py"]
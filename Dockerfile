# Multi-stage build for AutoGen Code Review Bot
# Optimized for production deployment with security hardening

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as builder

# Security: Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml setup.py ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .[monitoring]

# Production stage
FROM python:${PYTHON_VERSION}-slim as production

# Security: Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy application from builder stage
COPY --from=builder --chown=appuser:appuser /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder --chown=appuser:appuser /usr/local/bin /usr/local/bin
COPY --from=builder --chown=appuser:appuser /app /app

# Copy additional application files
COPY --chown=appuser:appuser agent_config.yaml /app/
COPY --chown=appuser:appuser .env.example /app/.env.example

# Set working directory and user
WORKDIR /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import autogen_code_review_bot; print('OK')" || exit 1

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Expose port (if webhook server)
EXPOSE 8080

# Default command
CMD ["python", "bot.py", "--config", "agent_config.yaml"]
# Build & Deployment Guide

Comprehensive guide for building, containerizing, and deploying AutoGen Code Review Bot.

## Quick Start

```bash
# Build production image
make build-docker

# Build development image  
./scripts/build.sh development

# Run with Docker Compose
docker-compose up -d
```

## Build Systems

### Make Targets

```bash
make build              # Build Python package
make build-docker       # Build Docker image
make clean              # Clean build artifacts
make install            # Install development dependencies
make test               # Run test suite
make lint               # Run code quality checks
```

### Build Script

The `scripts/build.sh` script provides advanced build options:

```bash
# Production build
./scripts/build.sh production

# Development build (includes debugging tools)
./scripts/build.sh development

# Custom image name
IMAGE_NAME=mybot ./scripts/build.sh production
```

## Docker Configuration

### Production Dockerfile

- **Multi-stage build**: Optimized for size and security
- **Non-root user**: Runs as unprivileged user (UID 1000)
- **Security hardening**: Minimal attack surface
- **Health checks**: Built-in health monitoring

### Development Dockerfile

- **Development tools**: Includes debugging and profiling tools
- **Hot-reloading**: Source code mounted for live changes
- **Additional ports**: Jupyter (8888), Debug server (5000)
- **Development dependencies**: Full dev environment

### Security Features

- Non-root container execution
- Read-only root filesystem
- Security options: `no-new-privileges:true`
- Minimal base image (Python slim)
- No package caches in final image
- Secrets via Docker secrets or external management

## Container Orchestration

### Docker Compose

#### Production Setup (`docker-compose.yml`)

```bash
# Start production stack
docker-compose up -d

# View logs
docker-compose logs -f autogen-bot

# Scale services
docker-compose up -d --scale autogen-bot=3
```

Services included:
- **autogen-bot**: Main application
- **redis**: Caching layer
- **prometheus**: Metrics collection
- **grafana**: Monitoring dashboards

#### Development Setup (`docker-compose.dev.yml`)

```bash
# Start development stack
docker-compose -f docker-compose.dev.yml up -d

# Attach to development container
docker-compose -f docker-compose.dev.yml exec autogen-bot-dev bash
```

Additional development services:
- **postgres-dev**: Development database
- **grafana-dev**: Development monitoring
- **jupyter**: Notebook environment

### Resource Management

```yaml
# Resource limits and reservations
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 1G
    reservations:
      cpus: '0.5'  
      memory: 256M
```

## Build Optimization

### Image Size Optimization

1. **Multi-stage builds**: Separate build and runtime stages
2. **Minimal base image**: Python slim instead of full Python
3. **Layer caching**: Optimize layer ordering for cache hits
4. **Dependency management**: Only install required packages

### Build Performance

1. **Docker BuildKit**: Enable for faster builds
2. **Build cache**: Use registry cache or local cache
3. **Parallel builds**: Build multiple images concurrently
4. **Dependency caching**: Cache pip dependencies

```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Use build cache
docker build --cache-from autogen-code-review-bot:latest .

# Build with cache mount
docker build --mount=type=cache,target=/root/.cache/pip .
```

## Release Management

### Semantic Versioning

The project uses semantic versioning with automated releases:

- **Major** (1.0.0): Breaking changes
- **Minor** (0.1.0): New features, backwards compatible
- **Patch** (0.0.1): Bug fixes, backwards compatible

### Automated Releases

Configured via `.releaserc.json`:

```bash
# Trigger release (via CI/CD)
git commit -m "feat: add new feature"
git push origin main
```

Release process:
1. Analyze commits for version bump
2. Generate changelog
3. Update version in pyproject.toml
4. Create Git tag and release
5. Build and publish artifacts

### Manual Release

```bash
# Build release artifacts
python -m build

# Check distribution
twine check dist/*

# Upload to PyPI (if configured)
twine upload dist/*
```

## Security Considerations

### Container Security

1. **Vulnerability scanning**: Integrated with build process
2. **SBOM generation**: Software Bill of Materials
3. **Base image updates**: Regular security updates
4. **Secret management**: External secret providers

### Supply Chain Security

1. **Dependency pinning**: Lock file for reproducible builds
2. **Signature verification**: Verify package signatures
3. **Audit trails**: Complete build provenance
4. **Isolation**: Build in clean environments

## Deployment Strategies

### Blue-Green Deployment

```bash
# Deploy new version
docker-compose -f docker-compose.prod.yml up -d --scale autogen-bot=2

# Health check and switch traffic
./scripts/health-check.sh && ./scripts/switch-traffic.sh

# Remove old version
docker-compose -f docker-compose.prod.yml up -d --scale autogen-bot=1
```

### Rolling Updates

```bash
# Kubernetes deployment
kubectl set image deployment/autogen-bot app=autogen-code-review-bot:v1.2.3

# Docker Swarm
docker service update --image autogen-code-review-bot:v1.2.3 autogen-bot
```

### Canary Deployment

```bash
# Deploy canary version (10% traffic)
docker-compose -f docker-compose.canary.yml up -d

# Monitor metrics and gradually increase traffic
# Full rollout or rollback based on metrics
```

## Monitoring & Observability

### Health Checks

- **Application health**: `/health` endpoint
- **Dependency health**: Database, Redis, external APIs
- **Resource health**: Memory, CPU, disk usage

### Metrics Collection

- **Prometheus metrics**: Application and system metrics
- **Custom metrics**: Business logic metrics
- **Performance metrics**: Response times, error rates

### Logging

```yaml
# Structured logging configuration
logging:
  driver: "json-file"
  options:
    max-size: "100m"
    max-file: "3"
    labels: "service=autogen-bot"
```

## Troubleshooting

### Common Build Issues

1. **Out of disk space**: Clean Docker images and volumes
   ```bash
   docker system prune -a --volumes
   ```

2. **Network timeouts**: Configure build-time networking
   ```bash
   docker build --network=host .
   ```

3. **Permission issues**: Check file ownership and Docker daemon access
   ```bash
   sudo chown -R $USER:$USER .
   ```

### Runtime Issues

1. **Container won't start**: Check logs and environment variables
   ```bash
   docker-compose logs autogen-bot
   ```

2. **Performance issues**: Monitor resource usage
   ```bash
   docker stats autogen-bot
   ```

3. **Network connectivity**: Check service discovery and networking
   ```bash
   docker network ls
   docker-compose exec autogen-bot ping redis
   ```

### Debugging

```bash
# Access running container
docker-compose exec autogen-bot bash

# Debug mode
docker-compose -f docker-compose.dev.yml up -d
docker-compose exec autogen-bot-dev bash

# View detailed logs
docker-compose logs -f --tail=100 autogen-bot
```

## Best Practices

### Build Best Practices

1. **Reproducible builds**: Pin all dependencies
2. **Minimal images**: Only include necessary components
3. **Security scanning**: Regular vulnerability assessments
4. **Automated testing**: Test images before deployment
5. **Documentation**: Keep build docs current

### Deployment Best Practices

1. **Zero-downtime deployment**: Use rolling updates
2. **Health checks**: Implement comprehensive health monitoring
3. **Rollback strategy**: Always have rollback plan
4. **Environment parity**: Keep dev/staging/prod similar
5. **Secret management**: Use external secret providers
6. **Resource limits**: Set appropriate resource constraints
7. **Monitoring**: Comprehensive observability stack
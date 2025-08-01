#!/bin/bash
# Build script for AutoGen Code Review Bot
# Handles production and development builds with proper tagging

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="${IMAGE_NAME:-autogen-code-review-bot}"
BUILD_TYPE="${1:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get version from pyproject.toml
get_version() {
    python3 -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
print(data['project']['version'])
" 2>/dev/null || echo "0.0.1"
}

# Get git commit hash
get_git_hash() {
    git rev-parse --short HEAD 2>/dev/null || echo "unknown"
}

# Build production image
build_production() {
    local version=$(get_version)
    local git_hash=$(get_git_hash)
    
    log_info "Building production image..."
    log_info "Version: $version, Git Hash: $git_hash"
    
    docker build \
        --file Dockerfile \
        --tag "${IMAGE_NAME}:${version}" \
        --tag "${IMAGE_NAME}:latest" \
        --build-arg PYTHON_VERSION=3.11 \
        "$PROJECT_ROOT"
    
    log_success "Production image built successfully"
}

# Build development image
build_development() {
    local git_hash=$(get_git_hash)
    
    log_info "Building development image..."
    
    docker build \
        --file Dockerfile.dev \
        --tag "${IMAGE_NAME}:dev" \
        --tag "${IMAGE_NAME}:dev-${git_hash}" \
        --build-arg PYTHON_VERSION=3.11 \
        "$PROJECT_ROOT"
    
    log_success "Development image built successfully"
}

# Main execution
main() {
    cd "$PROJECT_ROOT"
    
    case "${BUILD_TYPE}" in
        "production"|"prod")
            build_production
            ;;
        "development"|"dev")
            build_development
            ;;
        *)
            log_error "Unknown build type: $BUILD_TYPE"
            exit 1
            ;;
    esac
    
    log_success "Build completed successfully!"
}

# Run main function
main "$@"
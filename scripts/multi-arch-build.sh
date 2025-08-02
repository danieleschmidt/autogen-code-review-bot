#!/bin/bash
# Multi-architecture Docker build script
# Builds images for multiple platforms (linux/amd64, linux/arm64)

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="${IMAGE_NAME:-autogen-code-review-bot}"
REGISTRY="${REGISTRY:-}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
PUSH="${PUSH:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if buildx is available
check_buildx() {
    if ! docker buildx version &> /dev/null; then
        log_error "Docker buildx is required but not available"
        log_info "Install with: docker buildx install"
        exit 1
    fi
    
    log_info "Docker buildx is available"
}

# Setup buildx builder
setup_builder() {
    local builder_name="autogen-builder"
    
    log_info "Setting up multi-arch builder..."
    
    # Create builder if it doesn't exist
    if ! docker buildx ls | grep -q "$builder_name"; then
        docker buildx create \
            --name "$builder_name" \
            --driver docker-container \
            --use
        
        log_info "Created new builder: $builder_name"
    else
        docker buildx use "$builder_name"
        log_info "Using existing builder: $builder_name"
    fi
    
    # Bootstrap the builder
    docker buildx inspect --bootstrap
}

# Get version and metadata
get_metadata() {
    local version
    version=$(python3 -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
print(data['project']['version'])
" 2>/dev/null || echo "0.0.1")
    
    local git_hash
    git_hash=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    
    local build_date
    build_date=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    echo "$version,$git_hash,$build_date"
}

# Build multi-architecture images
build_multiarch() {
    local metadata
    metadata=$(get_metadata)
    IFS=',' read -r version git_hash build_date <<< "$metadata"
    
    log_info "Building multi-architecture images..."
    log_info "Version: $version"
    log_info "Git Hash: $git_hash"
    log_info "Platforms: $PLATFORMS"
    log_info "Build Date: $build_date"
    
    local build_args=(
        "--platform=$PLATFORMS"
        "--build-arg=PYTHON_VERSION=3.11"
        "--build-arg=VERSION=$version"
        "--build-arg=GIT_HASH=$git_hash"
        "--build-arg=BUILD_DATE=$build_date"
        "--file=Dockerfile"
    )
    
    # Add registry prefix if specified
    local image_tag="$IMAGE_NAME"
    if [ -n "$REGISTRY" ]; then
        image_tag="$REGISTRY/$IMAGE_NAME"
    fi
    
    # Add tags
    build_args+=(
        "--tag=$image_tag:$version"
        "--tag=$image_tag:latest"
        "--tag=$image_tag:$git_hash"
    )
    
    # Add metadata labels
    build_args+=(
        "--label=org.opencontainers.image.version=$version"
        "--label=org.opencontainers.image.revision=$git_hash"
        "--label=org.opencontainers.image.created=$build_date"
        "--label=org.opencontainers.image.source=https://github.com/danieleschmidt/autogen-code-review-bot"
        "--label=org.opencontainers.image.title=AutoGen Code Review Bot"
        "--label=org.opencontainers.image.description=AI-powered code review bot using Microsoft AutoGen"
        "--label=org.opencontainers.image.vendor=Terragon Labs"
        "--label=org.opencontainers.image.licenses=MIT"
    )
    
    # Add push or load flag
    if [ "$PUSH" = "true" ]; then
        build_args+=("--push")
        log_info "Images will be pushed to registry"
    else
        build_args+=("--load")
        log_info "Images will be loaded locally"
    fi
    
    # Execute build
    docker buildx build "${build_args[@]}" "$PROJECT_ROOT"
    
    log_success "Multi-architecture build completed"
}

# Generate manifest and push
generate_manifest() {
    if [ "$PUSH" != "true" ]; then
        log_warning "Skipping manifest generation (not pushing)"
        return
    fi
    
    local metadata
    metadata=$(get_metadata)
    IFS=',' read -r version git_hash build_date <<< "$metadata"
    
    local image_tag="$IMAGE_NAME"
    if [ -n "$REGISTRY" ]; then
        image_tag="$REGISTRY/$IMAGE_NAME"
    fi
    
    log_info "Generating and pushing manifest lists..."
    
    # Create manifest for latest
    docker buildx imagetools create \
        --tag "$image_tag:latest" \
        "$image_tag:latest"
    
    # Create manifest for version
    docker buildx imagetools create \
        --tag "$image_tag:$version" \
        "$image_tag:$version"
    
    log_success "Manifest lists created and pushed"
}

# Validate images
validate_images() {
    local metadata
    metadata=$(get_metadata)
    IFS=',' read -r version git_hash build_date <<< "$metadata"
    
    local image_tag="$IMAGE_NAME"
    if [ -n "$REGISTRY" ]; then
        image_tag="$REGISTRY/$IMAGE_NAME"
    fi
    
    log_info "Validating built images..."
    
    # Check if images exist for each platform
    IFS=',' read -ra platform_array <<< "$PLATFORMS"
    for platform in "${platform_array[@]}"; do
        log_info "Checking $platform image..."
        
        if docker buildx imagetools inspect "$image_tag:latest" | grep -q "$platform"; then
            log_success "$platform image exists"
        else
            log_error "$platform image not found"
        fi
    done
}

# Main execution
main() {
    cd "$PROJECT_ROOT"
    
    log_info "Starting multi-architecture build for ${IMAGE_NAME}"
    log_info "Target platforms: $PLATFORMS"
    
    check_buildx
    setup_builder
    build_multiarch
    
    if [ "$PUSH" = "true" ]; then
        generate_manifest
        validate_images
    fi
    
    log_success "Multi-architecture build process completed!"
    
    # Display summary
    echo ""
    log_info "Build Summary:"
    echo "  Image: $IMAGE_NAME"
    echo "  Platforms: $PLATFORMS"
    echo "  Registry: ${REGISTRY:-"local"}"
    echo "  Pushed: $PUSH"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH="true"
            shift
            ;;
        --registry=*)
            REGISTRY="${1#*=}"
            shift
            ;;
        --platforms=*)
            PLATFORMS="${1#*=}"
            shift
            ;;
        --image=*)
            IMAGE_NAME="${1#*=}"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --push                 Push images to registry"
            echo "  --registry=<registry>  Registry to push to"
            echo "  --platforms=<list>     Comma-separated list of platforms"
            echo "  --image=<name>         Image name"
            echo "  --help                 Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"
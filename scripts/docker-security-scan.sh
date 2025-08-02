#!/bin/bash
# Docker Security Scanning Script
# Performs comprehensive security analysis of Docker images

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="${IMAGE_NAME:-autogen-code-review-bot}"
SCAN_TYPE="${1:-all}"

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

# Check if required tools are available
check_dependencies() {
    local deps=("docker" "trivy")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Install with: brew install trivy # or apt-get install trivy"
        exit 1
    fi
}

# Scan for vulnerabilities using Trivy
scan_vulnerabilities() {
    log_info "Scanning for vulnerabilities..."
    
    trivy image \
        --format table \
        --severity HIGH,CRITICAL \
        --ignore-unfixed \
        "${IMAGE_NAME}:latest" || {
        log_warning "Vulnerability scan found issues"
        return 1
    }
    
    log_success "Vulnerability scan completed"
}

# Scan for secrets in the image
scan_secrets() {
    log_info "Scanning for embedded secrets..."
    
    trivy image \
        --scanners secret \
        --format table \
        "${IMAGE_NAME}:latest" || {
        log_warning "Secret scan found issues"
        return 1
    }
    
    log_success "Secret scan completed"
}

# Scan for misconfigurations
scan_config() {
    log_info "Scanning for configuration issues..."
    
    trivy image \
        --scanners config \
        --format table \
        "${IMAGE_NAME}:latest" || {
        log_warning "Configuration scan found issues"
        return 1
    }
    
    log_success "Configuration scan completed"
}

# Generate SBOM (Software Bill of Materials)
generate_sbom() {
    log_info "Generating SBOM..."
    
    local output_dir="${PROJECT_ROOT}/reports"
    mkdir -p "$output_dir"
    
    trivy image \
        --format spdx-json \
        --output "${output_dir}/sbom.spdx.json" \
        "${IMAGE_NAME}:latest"
    
    trivy image \
        --format cyclonedx \
        --output "${output_dir}/sbom.cyclonedx.json" \
        "${IMAGE_NAME}:latest"
    
    log_success "SBOM generated in ${output_dir}/"
}

# Run Docker bench security test
run_docker_bench() {
    log_info "Running Docker bench security test..."
    
    if command -v docker-bench-security &> /dev/null; then
        docker run --rm --net host --pid host --userns host --cap-add audit_control \
            -e DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST \
            -v /etc:/etc:ro \
            -v /var/lib:/var/lib:ro \
            -v /var/run/docker.sock:/var/run/docker.sock:ro \
            -v /usr/lib/systemd:/usr/lib/systemd:ro \
            -v /etc/systemd:/etc/systemd:ro \
            --label docker_bench_security \
            docker/docker-bench-security || {
            log_warning "Docker bench security found issues"
            return 1
        }
    else
        log_warning "docker-bench-security not available, skipping..."
    fi
}

# Analyze image layers and size
analyze_image() {
    log_info "Analyzing image structure..."
    
    echo "Image layers:"
    docker history "${IMAGE_NAME}:latest" --format "table {{.CreatedBy}}\t{{.Size}}"
    
    echo ""
    echo "Image size analysis:"
    docker images "${IMAGE_NAME}:latest" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
}

# Main scanning function
run_all_scans() {
    local scan_results=()
    
    # Run individual scans and collect results
    scan_vulnerabilities && scan_results+=("vulnerabilities: PASS") || scan_results+=("vulnerabilities: FAIL")
    scan_secrets && scan_results+=("secrets: PASS") || scan_results+=("secrets: FAIL")
    scan_config && scan_results+=("config: PASS") || scan_results+=("config: FAIL")
    
    generate_sbom
    analyze_image
    
    # Summary
    echo ""
    log_info "Security scan summary:"
    for result in "${scan_results[@]}"; do
        if [[ $result == *"FAIL"* ]]; then
            log_error "$result"
        else
            log_success "$result"
        fi
    done
}

# Main execution
main() {
    cd "$PROJECT_ROOT"
    
    log_info "Starting Docker security scan for ${IMAGE_NAME}:latest"
    check_dependencies
    
    case "${SCAN_TYPE}" in
        "vulnerabilities"|"vulns")
            scan_vulnerabilities
            ;;
        "secrets")
            scan_secrets
            ;;
        "config")
            scan_config
            ;;
        "sbom")
            generate_sbom
            ;;
        "bench")
            run_docker_bench
            ;;
        "analyze")
            analyze_image
            ;;
        "all")
            run_all_scans
            ;;
        *)
            log_error "Unknown scan type: $SCAN_TYPE"
            echo "Usage: $0 [vulnerabilities|secrets|config|sbom|bench|analyze|all]"
            exit 1
            ;;
    esac
    
    log_success "Security scan completed!"
}

# Run main function
main "$@"
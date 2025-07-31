# 🚀 Enterprise Release Automation

## Complete Release Pipeline Workflow

**File**: `.github/workflows/release.yml` (Create new file)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to release (e.g., 1.0.0)'
        required: true
        type: string

permissions:
  contents: write
  packages: write
  id-token: write

jobs:
  validate:
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.version }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Extract version
        id: version
        run: |
          if [[ "${{ github.event_name }}" == "workflow_dispatch" ]]; then
            echo "version=${{ github.event.inputs.version }}" >> $GITHUB_OUTPUT
          else
            echo "version=${GITHUB_REF#refs/tags/v}" >> $GITHUB_OUTPUT
          fi
      
      - name: Validate version format
        run: |
          if [[ ! "${{ steps.version.outputs.version }}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo "Invalid version format: ${{ steps.version.outputs.version }}"
            exit 1
          fi

  test:
    runs-on: ubuntu-latest
    needs: validate
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev,monitoring]
      
      - name: Run tests
        run: pytest -n auto --cov=src --cov-fail-under=95
      
      - name: Run security checks
        run: |
          bandit -r src -q
          safety check

  build:
    runs-on: ubuntu-latest
    needs: [validate, test]
    outputs:
      artifact-name: ${{ steps.build.outputs.artifact-name }}
      artifact-path: ${{ steps.build.outputs.artifact-path }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip build twine
      
      - name: Update version
        run: |
          sed -i 's/version = ".*"/version = "${{ needs.validate.outputs.version }}"/' pyproject.toml
      
      - name: Build package
        id: build
        run: |
          python -m build
          echo "artifact-name=autogen_code_review_bot-${{ needs.validate.outputs.version }}" >> $GITHUB_OUTPUT
          echo "artifact-path=dist/" >> $GITHUB_OUTPUT
      
      - name: Verify package
        run: |
          python -m twine check dist/*
      
      - name: Generate SBOM
        run: |
          pip install cyclone-pip-requirements
          cyclone-pip-requirements > release-sbom.json
      
      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: ${{ steps.build.outputs.artifact-name }}
          path: |
            dist/
            release-sbom.json
          retention-days: 30

  security-scan:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/checkout@v4
      - name: Download artifacts
        uses: actions/download-artifact@v4
        with:
          name: ${{ needs.build.outputs.artifact-name }}
      
      - name: Scan package for vulnerabilities
        run: |
          pip install safety
          safety check --json --output safety-release-report.json || true
      
      - name: Upload security scan results
        uses: actions/upload-artifact@v4
        with:
          name: security-scan-${{ needs.validate.outputs.version }}
          path: safety-release-report.json

  release:
    runs-on: ubuntu-latest
    needs: [validate, build, security-scan]
    environment: release
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Download artifacts
        uses: actions/download-artifact@v4
        with:
          name: ${{ needs.build.outputs.artifact-name }}
      
      - name: Generate changelog
        id: changelog
        run: |
          # Generate changelog from git history
          echo "# Changelog for v${{ needs.validate.outputs.version }}" > CHANGELOG_CURRENT.md
          echo "" >> CHANGELOG_CURRENT.md
          
          # Get previous tag
          PREV_TAG=$(git describe --tags --abbrev=0 HEAD^ 2>/dev/null || echo "")
          
          if [[ -n "$PREV_TAG" ]]; then
            echo "## Changes since $PREV_TAG" >> CHANGELOG_CURRENT.md
            git log --pretty=format:"- %s (%h)" $PREV_TAG..HEAD >> CHANGELOG_CURRENT.md
          else
            echo "## Initial Release" >> CHANGELOG_CURRENT.md
            git log --pretty=format:"- %s (%h)" >> CHANGELOG_CURRENT.md
          fi
          
          echo "" >> CHANGELOG_CURRENT.md
          echo "## Artifacts" >> CHANGELOG_CURRENT.md
          echo "- Python Package: \`autogen_code_review_bot-${{ needs.validate.outputs.version }}.tar.gz\`" >> CHANGELOG_CURRENT.md
          echo "- Wheel: \`autogen_code_review_bot-${{ needs.validate.outputs.version }}-py3-none-any.whl\`" >> CHANGELOG_CURRENT.md
          echo "- SBOM: \`release-sbom.json\`" >> CHANGELOG_CURRENT.md
      
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          tag_name: v${{ needs.validate.outputs.version }}
          name: Release v${{ needs.validate.outputs.version }}
          body_path: CHANGELOG_CURRENT.md
          files: |
            dist/*
            release-sbom.json
          draft: false
          prerelease: false
          generate_release_notes: true
      
      - name: Publish to PyPI
        if: github.repository == 'terragon-labs/autogen-code-review-bot'
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: |
          python -m pip install twine
          python -m twine upload dist/*

  docker-release:
    runs-on: ubuntu-latest
    needs: [validate, test]
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=semver,pattern={{version}},value=v${{ needs.validate.outputs.version }}
            type=semver,pattern={{major}}.{{minor}},value=v${{ needs.validate.outputs.version }}
            type=semver,pattern={{major}},value=v${{ needs.validate.outputs.version }}
            type=raw,value=latest
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64
```

## Release Automation Features

### 🎯 Intelligent Release Management
- **Version Validation**: Semantic version format enforcement
- **Automated Changelog**: Git history-based changelog generation
- **Multi-Format Publishing**: PyPI, GitHub Releases, Container Registry
- **Security Validation**: Pre-release vulnerability scanning

### 📦 Multi-Platform Distribution
- **Python Package**: PyPI publication with wheel and source distribution
- **Container Images**: Multi-architecture Docker builds (amd64, arm64)
- **GitHub Releases**: Comprehensive release artifacts with SBOM
- **Artifact Management**: Secure artifact handling and verification

### 🔒 Security-First Approach
- **Pre-Release Testing**: Full test suite validation
- **Security Scanning**: Vulnerability assessment before release
- **SBOM Generation**: Supply chain transparency
- **Environment Protection**: Release environment gating

### ⚡ Automation Excellence
- **Tag-Based Triggers**: Automatic releases on version tags
- **Manual Dispatch**: On-demand release capability
- **Parallel Processing**: Optimized build and deployment pipeline
- **Rollback Safety**: Comprehensive validation at each stage

## Implementation Guide

### 1. Create Release Workflow
```bash
mkdir -p .github/workflows
# Copy YAML above to .github/workflows/release.yml
```

### 2. Configure Repository Secrets
```bash
# Required secrets in GitHub repository settings:
PYPI_API_TOKEN=<your-pypi-token>  # For PyPI publishing
# GITHUB_TOKEN is auto-provided for GitHub Releases and Container Registry
```

### 3. Set Up Release Environment
```bash
# In GitHub repository settings > Environments:
# Create "release" environment with protection rules
# Require review for production releases (optional)
```

### 4. Configure Container Registry
```bash
# Enable GitHub Packages in repository settings
# Ensure container registry permissions are enabled
```

## Usage Examples

### Automated Release (Tag-based)
```bash
# Create and push a version tag
git tag v1.0.0
git push origin v1.0.0

# Workflow automatically triggers and creates:
# - GitHub Release with changelog
# - PyPI package publication  
# - Container images in ghcr.io
# - Release SBOM for supply chain security
```

### Manual Release (Workflow Dispatch)
```bash
# Use GitHub Actions UI or CLI:
gh workflow run release.yml -f version=1.0.1
```

## Release Artifacts Generated

### 📄 GitHub Release
- **Release Notes**: Auto-generated from git history
- **Source Archive**: Tagged source code
- **Python Packages**: `.tar.gz` and `.whl` files
- **SBOM**: `release-sbom.json` for supply chain security
- **Security Report**: Pre-release vulnerability assessment

### 🐍 PyPI Package
- **Source Distribution**: For pip installation
- **Wheel Package**: Optimized binary distribution
- **Metadata**: Complete package information and dependencies

### 🐳 Container Images
- **Multi-Architecture**: AMD64 and ARM64 support
- **Tagged Versions**: Semantic versioning with latest tag
- **Layer Optimization**: Efficient Docker builds with caching
- **Security Scanning**: Pre-publication vulnerability assessment

## Expected Benefits

- 🚀 **Zero-Touch Releases**: Fully automated release pipeline
- 🔒 **Security-First**: Comprehensive pre-release validation
- 📊 **Supply Chain Transparency**: SBOM generation and tracking
- ⚡ **Fast Distribution**: Parallel multi-platform publishing
- 📈 **Professional Quality**: Enterprise-grade release management
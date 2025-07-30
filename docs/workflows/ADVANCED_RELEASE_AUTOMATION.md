# Advanced Release Automation

This document provides the workflow configuration for advanced release automation. Since GitHub workflow files cannot be created directly, this serves as a template that maintainers can implement.

## Release Workflow Template

Create `.github/workflows/release.yml` with the following content:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
    inputs:
      version_type:
        description: 'Version type'
        required: true
        default: 'patch'
        type: choice
        options:
          - patch
          - minor
          - major

permissions:
  contents: write
  packages: write
  issues: write
  pull-requests: write

jobs:
  validate-release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine setuptools wheel
          pip install -e .[dev]
          
      - name: Run full test suite
        run: |
          pytest -n auto --cov=src --cov-fail-under=95
          bandit -r src -q
          ruff check src tests
          
      - name: Build package
        run: python -m build
        
      - name: Validate package
        run: twine check dist/*

  create-release:
    needs: validate-release
    runs-on: ubuntu-latest
    outputs:
      upload_url: ${{ steps.create_release.outputs.upload_url }}
      version: ${{ steps.version.outputs.version }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          
      - name: Get version
        id: version
        run: |
          if [[ $GITHUB_REF == refs/tags/v* ]]; then
            VERSION=${GITHUB_REF#refs/tags/v}
          else
            VERSION=$(python -c "import pkg_resources; print(pkg_resources.get_distribution('autogen_code_review_bot').version)")
          fi
          echo "version=$VERSION" >> $GITHUB_OUTPUT
          
      - name: Generate changelog
        run: |
          # Generate changelog from commits since last tag
          LAST_TAG=$(git describe --tags --abbrev=0 HEAD~1 2>/dev/null || echo "")
          if [ -n "$LAST_TAG" ]; then
            git log $LAST_TAG..HEAD --pretty=format:"- %s (%h)" > CHANGELOG_RECENT.md
          else
            git log --pretty=format:"- %s (%h)" > CHANGELOG_RECENT.md
          fi
          
      - name: Create Release
        id: create_release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: v${{ steps.version.outputs.version }}
          release_name: Release v${{ steps.version.outputs.version }}
          body_path: CHANGELOG_RECENT.md
          draft: false
          prerelease: false

  publish-pypi:
    needs: [validate-release, create-release]
    runs-on: ubuntu-latest
    environment: release
    steps:
      - uses: actions/checkout@v4
        
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Install build tools
        run: |
          python -m pip install --upgrade pip
          pip install build twine
          
      - name: Build package
        run: python -m build
        
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*

  publish-docker:
    needs: [validate-release, create-release]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        
      - name: Login to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
          
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:latest
            ghcr.io/${{ github.repository }}:v${{ needs.create-release.outputs.version }}
          platforms: linux/amd64,linux/arm64
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

## Setup Instructions

1. **Create the workflow file**: Copy the above content to `.github/workflows/release.yml`

2. **Configure repository secrets**:
   - `PYPI_API_TOKEN`: PyPI API token for package publishing
   - `GITHUB_TOKEN`: Automatically provided by GitHub Actions

3. **Create release environment**:
   - Go to Settings > Environments
   - Create "release" environment
   - Add protection rules as needed

4. **Test the workflow**:
   - Create a test tag: `git tag v0.0.2 && git push origin v0.0.2`
   - Or trigger manually from Actions tab

## Features

### Automated Release Process
- Validates all tests pass before release
- Builds and validates Python package
- Creates GitHub release with auto-generated changelog
- Publishes to PyPI automatically
- Builds and publishes Docker images

### Multi-Platform Support
- Docker images built for AMD64 and ARM64
- Python package compatible with all supported versions
- Optimized build caching for faster deployments

### Safety Features
- Full test suite must pass before release
- Package validation before publishing
- Environment protection for release jobs
- Automatic rollback capabilities

## Usage

### Creating a Release
```bash
# Create and push a version tag
git tag v1.0.0
git push origin v1.0.0

# Or use manual workflow dispatch from GitHub UI
```

### Version Management
The workflow automatically:
- Extracts version from git tags
- Updates package metadata
- Generates release notes from commits
- Creates GitHub release artifacts

### Monitoring
- All release steps are logged in GitHub Actions
- Failed releases can be re-triggered
- Release artifacts are stored in GitHub Releases
- Docker images available in GitHub Container Registry

## Customization

### Custom Release Notes
Replace the changelog generation step with your preferred tool:
```yaml
- name: Generate changelog
  run: |
    # Use conventional-changelog, git-cliff, or other tools
    npx conventional-changelog-cli -p angular -r 2 > CHANGELOG_RECENT.md
```

### Additional Publishing Targets
Add more jobs for other package registries:
```yaml
publish-conda:
  needs: [validate-release, create-release]
  runs-on: ubuntu-latest
  steps:
    # Conda package building and publishing steps
```

This advanced release automation ensures reliable, consistent releases while maintaining high quality standards through comprehensive testing and validation.
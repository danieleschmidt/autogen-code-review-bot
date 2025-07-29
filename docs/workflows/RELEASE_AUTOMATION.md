# Release Automation Workflow

This document outlines the automated release and deployment workflows for the AutoGen Code Review Bot.

## Release Strategy

The project follows semantic versioning with automated releases triggered by:
- Manual workflow dispatch
- Tag creation
- Merge to main with version bump

## Required GitHub Actions Workflows

### 1. Release Workflow (`.github/workflows/release.yml`)

```yaml
name: Release
on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to release (e.g., v1.2.3)'
        required: true
        type: string

permissions:
  contents: write
  packages: write
  pull-requests: write

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e .[dev]
      
      - name: Run full test suite
        run: |
          pytest -n auto --cov=src --cov-fail-under=95
          bandit -r src -q
          ruff check src/
          mypy src/
  
  build:
    needs: validate
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.version }}
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Determine version
        id: version
        run: |
          if [ "${{ github.event_name }}" = "workflow_dispatch" ]; then
            VERSION="${{ github.event.inputs.version }}"
          else
            VERSION="${GITHUB_REF#refs/tags/}"
          fi
          echo "version=${VERSION}" >> $GITHUB_OUTPUT
          echo "Version: ${VERSION}"
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Build package
        run: |
          pip install build
          python -m build
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: dist
          path: dist/

  docker:
    needs: [validate, build]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to GitHub Container Registry
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
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=semver,pattern={{major}}
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  release:
    needs: [validate, build, docker]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Download artifacts
        uses: actions/download-artifact@v3
        with:
          name: dist
          path: dist/
      
      - name: Generate changelog
        id: changelog
        run: |
          # Extract changes from CHANGELOG.md for this version
          VERSION="${{ needs.build.outputs.version }}"
          echo "Generating changelog for ${VERSION}"
          
          # This would typically use a tool like conventional-changelog
          echo "changelog<<EOF" >> $GITHUB_OUTPUT
          echo "## Changes in ${VERSION}" >> $GITHUB_OUTPUT
          echo "" >> $GITHUB_OUTPUT
          echo "- See CHANGELOG.md for detailed changes" >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT
      
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          tag_name: ${{ needs.build.outputs.version }}
          name: Release ${{ needs.build.outputs.version }}
          body: ${{ steps.changelog.outputs.changelog }}
          draft: false
          prerelease: false
          files: |
            dist/*
          generate_release_notes: true
      
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
          packages-dir: dist/
```

### 2. Deployment Workflow (`.github/workflows/deploy.yml`)

```yaml
name: Deploy
on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy to'
        required: true
        default: 'staging'
        type: choice
        options:
        - staging
        - production

jobs:
  deploy-staging:
    if: github.event_name == 'workflow_dispatch' && github.event.inputs.environment == 'staging'
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
      - name: Deploy to Staging
        run: |
          echo "Deploying to staging environment"
          # Add staging deployment commands here
  
  deploy-production:
    if: github.event_name == 'release' || (github.event_name == 'workflow_dispatch' && github.event.inputs.environment == 'production')
    runs-on: ubuntu-latest
    environment: production
    
    steps:
      - name: Deploy to Production
        run: |
          echo "Deploying to production environment"
          # Add production deployment commands here
```

### 3. Version Bump Workflow (`.github/workflows/version-bump.yml`)

```yaml
name: Version Bump
on:
  workflow_dispatch:
    inputs:
      bump_type:
        description: 'Type of version bump'
        required: true
        default: 'patch'
        type: choice
        options:
        - patch
        - minor
        - major

permissions:
  contents: write
  pull-requests: write

jobs:
  bump-version:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install bump2version
      
      - name: Configure git
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
      
      - name: Bump version
        run: |
          bump2version ${{ github.event.inputs.bump_type }}
      
      - name: Push changes
        run: |
          git push
          git push --tags
```

## Configuration Files

### 1. `.bumpversion.cfg`

```ini
[bumpversion]
current_version = 0.0.1
commit = True
tag = True
tag_name = v{new_version}
message = chore: bump version to {new_version}

[bumpversion:file:pyproject.toml]
search = version = "{current_version}"
replace = version = "{new_version}"

[bumpversion:file:src/autogen_code_review_bot/__init__.py]
search = __version__ = "{current_version}"
replace = __version__ = "{new_version}"
```

### 2. Release Checklist Template

Create `.github/ISSUE_TEMPLATE/release_checklist.md`:

```markdown
---
name: Release Checklist
about: Checklist for preparing a new release
title: 'Release v'
labels: release
assignees: ''
---

## Pre-Release Checklist

- [ ] All tests passing on main branch
- [ ] Security scans passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated with release notes
- [ ] Version bumped in pyproject.toml
- [ ] Dependencies reviewed and updated if needed

## Release Process

- [ ] Create release tag
- [ ] GitHub release created automatically
- [ ] Docker images published to GHCR
- [ ] PyPI package published
- [ ] Deployment to staging successful
- [ ] Deployment to production approved

## Post-Release

- [ ] Release announcement posted
- [ ] Documentation site updated
- [ ] Next milestone planned
```

## Implementation Steps

1. **Add workflow files**: Create all workflows in `.github/workflows/`
2. **Configure secrets**: Set up PyPI token and deployment credentials
3. **Set up environments**: Create staging and production environments with approvals
4. **Configure branch protection**: Require release workflows to pass
5. **Test release process**: Run through complete release cycle on staging

## Benefits

- **Automated quality gates**: All tests and security checks before release
- **Consistent releases**: Standardized process reduces human error
- **Container distribution**: Multi-platform Docker images
- **Package distribution**: Automatic PyPI publishing
- **Deployment automation**: Environment-specific deployments with approvals
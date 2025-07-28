# Workflow Requirements

## Overview
This document outlines the GitHub Actions workflows that need manual setup due to permission limitations.

## Required Workflows

### 1. Continuous Integration (`ci.yml`)
**Purpose**: Run tests, linting, and security checks on every PR
**Requirements**:
- Runs on: `[push, pull_request]`
- Python versions: `[3.8, 3.9, 3.10, 3.11]`
- Steps: checkout, setup-python, install deps, run tests, upload coverage

### 2. Security Scanning (`security.yml`)
**Purpose**: Automated security vulnerability scanning
**Requirements**:
- CodeQL analysis for Python
- Dependency vulnerability scanning
- Secret detection
- Schedule: weekly

### 3. Release Automation (`release.yml`)
**Purpose**: Automated releases when tags are pushed
**Requirements**:
- Build and publish to PyPI
- Create GitHub release with changelog
- Trigger on tag push (`v*`)

## Branch Protection Rules
Configure in Settings > Branches for `main`:
- Require PR reviews: 1
- Require status checks: CI tests must pass
- Require up-to-date branches
- Include administrators

## Repository Settings
- Topics: `python`, `code-review`, `automation`, `github-bot`
- Description: "Automated code review bot using AutoGen agents"
- Homepage: Link to documentation

## External Integrations
See [GitHub Marketplace](https://github.com/marketplace) for:
- Code coverage reporting (Codecov)
- Security scanning (Snyk, Dependabot)
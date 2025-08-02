# Manual Setup Required

Due to GitHub App permission limitations, the following setup steps require manual configuration by repository maintainers.

## GitHub Actions Workflows

The following workflow files must be created manually in `.github/workflows/`:

### Required Workflows

1. **CI Workflow** (`ci.yml`)
   - Copy from: `docs/workflows/examples/ci.yml`
   - Purpose: Pull request validation, testing, security scanning
   - Required secrets: `GITHUB_TOKEN` (automatic)

2. **CD Workflow** (`cd.yml`)
   - Copy from: `docs/workflows/examples/cd.yml`
   - Purpose: Deployment automation
   - Required secrets: Deployment-specific tokens

3. **Dependency Update Workflow** (`dependency-update.yml`)
   - Copy from: `docs/workflows/examples/dependency-update.yml`
   - Purpose: Automated dependency management
   - Required secrets: `GITHUB_TOKEN` (automatic)

4. **Security Scan Workflow** (`security-scan.yml`)
   - Copy from: `docs/workflows/examples/security-scan.yml`
   - Purpose: Comprehensive security scanning
   - Required secrets: Security tool tokens (optional)

## Repository Settings

### Branch Protection Rules

Configure the following branch protection rules for `main`:

```yaml
Protection Rules:
  - Require pull request reviews before merging: true
  - Require status checks to pass before merging: true
  - Required status checks:
    - CI / test (ubuntu-latest, 3.9)
    - CI / test (ubuntu-latest, 3.10)
    - CI / test (ubuntu-latest, 3.11)
    - Security Scan / security-check
  - Require branches to be up to date before merging: true
  - Require conversation resolution before merging: true
  - Restrict pushes that create files: false
  - Do not allow bypassing the above settings: true
```

### Repository Secrets

Configure the following repository secrets:

```yaml
Required Secrets:
  GITHUB_TOKEN: # Automatically provided
  
Optional Secrets (for enhanced functionality):
  SLACK_WEBHOOK_URL: # For Slack notifications
  SONAR_TOKEN: # For SonarQube integration
  SNYK_TOKEN: # For Snyk security scanning
```

### Repository Variables

Configure the following repository variables:

```yaml
Variables:
  PYTHON_VERSION: "3.11"
  NODE_VERSION: "18"
  MONITORING_ENABLED: "true"
```

## Issue and PR Templates

The following templates are available but need to be configured:

1. **Bug Report Template**
   - Location: `.github/ISSUE_TEMPLATE/bug_report.md`
   - Purpose: Standardize bug reports

2. **Feature Request Template**
   - Location: `.github/ISSUE_TEMPLATE/feature_request.md`
   - Purpose: Standardize feature requests

3. **Pull Request Template**
   - Location: `.github/PULL_REQUEST_TEMPLATE.md`
   - Purpose: Standardize PR descriptions

## Monitoring and Alerting

### Prometheus Setup

1. Deploy Prometheus using the configuration in `monitoring/prometheus.yml`
2. Configure alerts using rules in `monitoring/rules/autogen-bot.yml`
3. Set up Grafana dashboard from `monitoring/grafana-dashboard.json`

### Health Check Endpoints

Configure health check monitoring for:

```yaml
Endpoints:
  - /health: Basic health check
  - /metrics: Prometheus metrics
  - /ready: Readiness probe
```

## Security Configuration

### Required Security Measures

1. **Enable Dependency Scanning**
   - Go to Settings > Security & analysis
   - Enable Dependabot alerts
   - Enable Dependabot security updates

2. **Enable Code Scanning**
   - Enable CodeQL analysis
   - Configure custom queries if needed

3. **Configure Secret Scanning**
   - Enable secret scanning alerts
   - Configure custom patterns for your organization

### Pre-commit Hooks

Install and configure pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Integration Setup

### GitHub Integration

1. **Repository Topics**
   - Run: `python scripts/integration_scripts.py --setup`
   - Or manually add topics: `autogen`, `code-review`, `automation`, `python`

2. **Repository Description**
   - Update to: "Two-agent AutoGen system for automated code review with dual-agent architecture"

### Monitoring Integration

1. **Prometheus Integration**
   - Configure endpoint: `http://localhost:9090`
   - Set up metrics collection

2. **Grafana Integration**
   - Configure endpoint: `http://localhost:3000`
   - Import dashboard from `monitoring/grafana-dashboard.json`

## Automation Scripts

The following automation scripts are available and should be scheduled:

### Daily Scripts
```bash
# Collect metrics
./scripts/collect_metrics.py

# Monitor code quality
./scripts/code_quality_monitor.py
```

### Weekly Scripts
```bash
# Check for dependency updates
./scripts/dependency_update.py --auto-apply

# Repository maintenance
./scripts/repository_maintenance.py
```

### Monthly Scripts
```bash
# Full integration health check
./scripts/integration_scripts.py --health-check

# Generate comprehensive reports
./scripts/collect_metrics.py && ./scripts/code_quality_monitor.py --report-only
```

## Validation Checklist

After completing manual setup, verify:

- [ ] All GitHub Actions workflows are running successfully
- [ ] Branch protection rules are enforced
- [ ] Monitoring dashboards are accessible
- [ ] Security scanning is active
- [ ] Automation scripts execute without errors
- [ ] Integration health checks pass

## Support

For setup assistance or issues:

1. Check the troubleshooting guide in `docs/`
2. Review workflow logs in GitHub Actions
3. Verify all required secrets and variables are configured
4. Ensure repository permissions are correctly set

---

*This document is automatically maintained. Last updated: 2025-08-02*
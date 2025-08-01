# Comprehensive Workflow Setup Guide

Complete guide for setting up GitHub Actions workflows for the AutoGen Code Review Bot. This guide provides step-by-step instructions for repository maintainers to enable automated CI/CD pipelines.

## 🚨 Important Notice

**GitHub App Permission Limitation**: Due to security restrictions, GitHub Apps cannot directly create workflow files. Repository maintainers must manually copy workflow templates from `docs/workflows/examples/` to `.github/workflows/` to enable automation.

## Quick Setup Checklist

- [ ] Copy workflow templates to `.github/workflows/`
- [ ] Configure repository secrets and environment variables
- [ ] Set up branch protection rules with required status checks
- [ ] Configure GitHub environments (staging, production)
- [ ] Enable security scanning features (Dependabot, CodeQL)
- [ ] Test workflow execution with a sample PR

---

## 1. Workflow Template Installation

### Available Workflows

| Workflow | Purpose | Triggers |
|----------|---------|----------|
| `ci.yml` | Continuous Integration | PR, Push to main/develop |
| `cd.yml` | Continuous Deployment | Push to main, Release |
| `security-scan.yml` | Security Analysis | PR, Push, Schedule, Manual |
| `dependency-update.yml` | Dependency Updates | Schedule (weekly), Manual |

### Installation Steps

```bash
# 1. Create workflows directory
mkdir -p .github/workflows

# 2. Copy all workflow templates
cp docs/workflows/examples/ci.yml .github/workflows/ci.yml
cp docs/workflows/examples/cd.yml .github/workflows/cd.yml
cp docs/workflows/examples/security-scan.yml .github/workflows/security.yml
cp docs/workflows/examples/dependency-update.yml .github/workflows/dependency-update.yml

# 3. Commit the workflows
git add .github/workflows/
git commit -m "feat: add GitHub Actions workflows

- Add comprehensive CI/CD pipeline
- Add security scanning automation
- Add dependency update automation
- Enable automated testing and deployment"
git push origin main
```

---

## 2. Repository Secrets Configuration

### Required Secrets

Configure these secrets in `Settings` → `Secrets and variables` → `Actions`:

#### Core Application Secrets
```bash
# GitHub Bot Operations
GITHUB_BOT_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx  # Personal access token for bot
WEBHOOK_SECRET=your_webhook_secret_here    # GitHub webhook secret

# AI Services
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx  # OpenAI API key
```

#### Container Registry Secrets
```bash
# GitHub Container Registry (recommended)
GITHUB_TOKEN=<automatically_provided>  # GitHub token (auto-provided)

# Docker Hub (alternative)
DOCKERHUB_USERNAME=your_dockerhub_username
DOCKERHUB_TOKEN=your_dockerhub_token
```

#### Security Scanning Secrets
```bash
# Code Security
SNYK_TOKEN=your_snyk_token_here        # Snyk security scanning
CODECOV_TOKEN=your_codecov_token_here  # Code coverage reporting
```

#### Deployment Secrets
```bash
# Staging Environment
STAGING_DEPLOY_KEY=-----BEGIN_OPENSSH_PRIVATE_KEY-----  # SSH key
STAGING_HOST=staging.example.com
STAGING_USER=deploy

# Production Environment  
PROD_DEPLOY_KEY=-----BEGIN_OPENSSH_PRIVATE_KEY-----     # SSH key
PROD_HOST=production.example.com
PROD_USER=deploy
```

#### Notification Secrets (Optional)
```bash
# Team Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/xxx/xxx
TEAMS_WEBHOOK_URL=https://your-teams-webhook-url
```

### Secrets Configuration Commands

```bash
# Using GitHub CLI to set secrets
gh secret set GITHUB_BOT_TOKEN --body "ghp_xxxxxxxxxxxxxxxxxxxx"
gh secret set WEBHOOK_SECRET --body "your_webhook_secret_here"
gh secret set OPENAI_API_KEY --body "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
gh secret set SNYK_TOKEN --body "your_snyk_token_here"
```

---

## 3. Environment Variables Configuration

### Repository Variables

Set these in `Settings` → `Secrets and variables` → `Actions` → `Variables`:

```bash
# Container Configuration
REGISTRY=ghcr.io
IMAGE_NAME=autogen-code-review-bot
PYTHON_VERSION=3.11
NODE_VERSION=20

# Application Configuration
APP_ENV=production
LOG_LEVEL=INFO
CACHE_TTL_HOURS=24

# Deployment Configuration
STAGING_URL=https://staging.example.com
PRODUCTION_URL=https://production.example.com
```

---

## 4. Branch Protection Rules

### Main Branch Protection

Configure branch protection for `main`:

```bash
# Using GitHub CLI
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{
    "strict": true,
    "contexts": [
      "CI Success",
      "Code Quality & Security", 
      "Test Suite",
      "Docker Build & Test",
      "Security Summary"
    ]
  }' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true
  }' \
  --field restrictions=null
```

### Manual Configuration

1. Go to `Settings` → `Branches`
2. Click "Add rule" for `main` branch
3. Configure these settings:

**Branch name pattern**: `main`

**Protect matching branches**:
- [x] Require a pull request before merging
  - Required number of reviewers: 1
  - [x] Dismiss stale PR reviews when new commits are pushed
  - [x] Require review from code owners
- [x] Require status checks to pass before merging
  - [x] Require branches to be up to date before merging
  - Required status checks:
    - CI Success
    - Code Quality & Security
    - Test Suite  
    - Docker Build & Test
    - Security Summary
- [x] Require signed commits
- [x] Include administrators
- [x] Allow force pushes (for maintainers only)

---

## 5. GitHub Environments Setup

### Staging Environment

1. Go to `Settings` → `Environments`
2. Click "New environment"
3. Name: `staging`

**Environment Protection Rules**:
- Required reviewers: None (automatic deployment)
- Wait timer: 0 minutes
- Deployment branches: Only `main` branch

**Environment Secrets**:
```bash
STAGING_API_URL=https://staging-api.example.com
STAGING_DATABASE_URL=postgresql://staging-db-url
STAGING_REDIS_URL=redis://staging-redis-url
```

### Production Environment

1. Create new environment: `production`

**Environment Protection Rules**:
- Required reviewers: 2 maintainers minimum
- Wait timer: 10 minutes (cooling-off period)
- Deployment branches: Only protected branches

**Environment Secrets**:
```bash
PRODUCTION_API_URL=https://api.example.com
PRODUCTION_DATABASE_URL=postgresql://prod-db-url
PRODUCTION_REDIS_URL=redis://prod-redis-url
```

---

## 6. Security Features Configuration

### Enable GitHub Security Features

1. **Go to `Settings` → `Security & analysis`**

2. **Enable these features**:
   - [x] Dependency graph
   - [x] Dependabot alerts
   - [x] Dependabot security updates
   - [x] Dependabot version updates
   - [x] Code scanning alerts
   - [x] Secret scanning alerts
   - [x] Secret scanning push protection

### Configure Dependabot

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
    open-pull-requests-limit: 5
    reviewers:
      - "maintainer-username"
    assignees:
      - "maintainer-username"
    labels:
      - "dependencies"
      - "python"
    commit-message:
      prefix: "chore(deps)"
      include: "scope"

  # Docker dependencies
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
      - "docker"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
      - "github-actions"
```

### Configure CodeQL Analysis

The `ci.yml` workflow includes CodeQL analysis. To customize:

```yaml
# In ci.yml, CodeQL step
- name: Initialize CodeQL
  uses: github/codeql-action/init@v3
  with:
    languages: python
    config-file: .github/codeql/codeql-config.yml  # Optional custom config
```

Create `.github/codeql/codeql-config.yml` for custom rules:

```yaml
name: "CodeQL Config"
disable-default-queries: false
queries:
  - uses: security-and-quality
  - uses: security-experimental
paths-ignore:
  - "tests/"
  - "docs/"
  - "scripts/"
```

---

## 7. Workflow Permissions

### Configure Repository Permissions

1. **Go to `Settings` → `Actions` → `General`**

2. **Set "Workflow permissions"**:
   - [x] Read and write permissions
   - [x] Allow GitHub Actions to create and approve pull requests

3. **Set "Fork pull request workflows"**:
   - Require approval for first-time contributors

### Token Permissions in Workflows

Ensure workflows have proper permissions:

```yaml
# In workflow files
permissions:
  contents: read          # Read repository contents
  packages: write         # Push to GitHub Container Registry
  security-events: write  # Upload security scan results
  actions: read          # Access workflow artifacts
  checks: write          # Create check runs
  pull-requests: write   # Comment on PRs
  issues: write          # Create issues (for dependency updates)
  deployments: write     # Create deployments
```

---

## 8. Testing Your Setup

### Initial Workflow Test

1. **Create a test branch**:
   ```bash
   git checkout -b test/workflow-validation
   echo "# Workflow Test" > WORKFLOW_TEST.md
   git add WORKFLOW_TEST.md
   git commit -m "test: validate workflow setup"
   git push origin test/workflow-validation
   ```

2. **Create a pull request**:
   ```bash
   gh pr create --title "Test: Validate workflow setup" \
     --body "Testing GitHub Actions workflow setup and branch protection rules."
   ```

3. **Verify workflow execution**:
   - Check that all CI workflows run
   - Verify security scans complete
   - Confirm Docker builds succeed
   - Check that PR is blocked until checks pass

### Deployment Test

1. **Test staging deployment**:
   ```bash
   # Merge the test PR to main
   gh pr merge --squash
   
   # Verify staging deployment runs automatically
   gh run list --workflow=cd.yml
   ```

2. **Test production deployment**:
   ```bash
   # Create a release to trigger production deployment
   git tag v0.1.0-test
   git push origin v0.1.0-test
   
   # Or create release via GitHub
   gh release create v0.1.0-test --title "Test Release" --notes "Testing production deployment"
   ```

### Security Scan Test

1. **Trigger security scan**:
   ```bash
   gh workflow run security.yml
   ```

2. **Check scan results**:
   - Go to `Security` tab in GitHub
   - Review any findings in "Code scanning alerts"
   - Check "Dependabot alerts" for dependency issues

---

## 9. Monitoring and Notifications

### Workflow Monitoring

1. **GitHub Insights**:
   - Go to `Insights` → `Actions`
   - Monitor workflow success rates
   - Track workflow performance

2. **Set up notifications**:
   - Go to `Settings` → `Notifications`
   - Configure email notifications for workflow failures
   - Set up Slack/Teams integration if configured

### Custom Monitoring

Add monitoring steps to workflows:

```yaml
# Add to workflow for external monitoring
- name: Report to monitoring system
  if: always()
  run: |
    curl -X POST ${{ secrets.MONITORING_WEBHOOK }} \
      -H 'Content-Type: application/json' \
      -d '{
        "workflow": "${{ github.workflow }}",
        "status": "${{ job.status }}",
        "repository": "${{ github.repository }}",
        "run_id": "${{ github.run_id }}"
      }'
```

---

## 10. Advanced Configuration

### Custom Workflow Triggers

Customize when workflows run:

```yaml
# In workflow files
on:
  push:
    branches: [ main, develop, 'release/*' ]
    paths-ignore:
      - 'docs/**'
      - '*.md'
  pull_request:
    branches: [ main ]
    types: [opened, synchronize, reopened, ready_for_review]
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 2 AM UTC
  workflow_dispatch:
    inputs:
      environment:
        description: 'Target environment'
        required: true
        default: 'staging'
        type: choice
        options: ['staging', 'production']
```

### Matrix Strategies

Customize test matrices:

```yaml
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10', '3.11']
    os: [ubuntu-latest, windows-latest, macos-latest]
  fail-fast: false
  max-parallel: 4
```

### Conditional Steps

Add conditional logic:

```yaml
- name: Deploy to production
  if: |
    github.event_name == 'release' && 
    github.event.action == 'published' &&
    !github.event.release.prerelease
  run: echo "Deploying to production"
```

---

## 11. Troubleshooting

### Common Issues

1. **Workflows not triggering**:
   ```bash
   # Check workflow syntax
   gh workflow view ci.yml
   
   # List recent runs
   gh run list --limit 10
   ```

2. **Permission errors**:
   - Verify `GITHUB_TOKEN` has necessary permissions
   - Check repository settings → Actions → General
   - Ensure secrets are correctly named and scoped

3. **Branch protection bypassed**:
   - Check if "Include administrators" is enabled
   - Verify required status checks are correctly named
   - Ensure status checks are reporting correctly

4. **Secret access issues**:
   ```bash
   # List configured secrets (names only)
   gh secret list
   
   # Check if secret is accessible in workflow
   # Add debug step to workflow:
   - name: Debug secrets
     run: |
       echo "Secret exists: ${{ secrets.SECRET_NAME != '' }}"
   ```

### Debug Workflow Issues

Add debugging steps to workflows:

```yaml
- name: Debug information
  run: |
    echo "Event: ${{ github.event_name }}"
    echo "Ref: ${{ github.ref }}"
    echo "Actor: ${{ github.actor }}"
    echo "Repository: ${{ github.repository }}"
    echo "Run ID: ${{ github.run_id }}"
    env
```

### Performance Optimization

1. **Cache dependencies**:
   ```yaml
   - uses: actions/cache@v3
     with:
       path: ~/.cache/pip
       key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
   ```

2. **Parallel job execution**:
   ```yaml
   jobs:
     test:
       strategy:
         matrix:
           python-version: [3.8, 3.9, 3.10, 3.11]
   ```

3. **Skip unnecessary runs**:
   ```yaml
   on:
     push:
       paths-ignore:
         - 'docs/**'
         - '*.md'
   ```

---

## 12. Maintenance and Updates

### Regular Maintenance Tasks

**Weekly**:
- Review failed workflow runs
- Check Dependabot PRs and merge if tests pass
- Review security scan results

**Monthly**:
- Update workflow action versions
- Review and optimize workflow performance  
- Audit secrets and permissions

**Quarterly**:
- Review branch protection rules
- Update environment configurations
- Security audit of workflow permissions

### Keeping Workflows Updated

1. **Action version updates**:
   ```bash
   # Dependabot will create PRs for action updates
   # Review and merge these regularly
   ```

2. **Manual updates**:
   ```bash
   # Check for new action versions
   gh api repos/actions/checkout/releases/latest
   
   # Update workflow files as needed
   ```

### Backup and Recovery

1. **Backup workflow configurations**:
   ```bash
   # All workflows are in git, ensure regular backups
   git archive --format=tar.gz HEAD .github/ > workflows-backup.tar.gz
   ```

2. **Recovery procedures**:
   - Workflows are version controlled in git
   - Secrets need to be reconfigured manually
   - Environment settings need to be recreated

---

## Summary and Next Steps

After completing this setup, you will have:

✅ **Complete CI/CD Pipeline**: Automated testing, security scanning, and deployment  
✅ **Security Integration**: Comprehensive security analysis and vulnerability management  
✅ **Dependency Management**: Automated updates with testing validation  
✅ **Quality Gates**: Code quality enforcement and review requirements  
✅ **Deployment Automation**: Safe, automated deployments with rollback capability  
✅ **Monitoring**: Workflow performance tracking and failure notifications  

### Immediate Next Steps

1. **Test the complete pipeline** with a sample feature branch
2. **Train your team** on the new CI/CD processes
3. **Monitor workflow performance** and optimize as needed
4. **Document any customizations** specific to your environment
5. **Set up alerting** for critical workflow failures
6. **Plan regular maintenance** schedule for workflow updates

### Long-term Considerations

- **Scale considerations**: Monitor workflow execution times and costs
- **Security updates**: Keep security scanning tools and configurations current
- **Process evolution**: Adapt workflows as team practices evolve
- **Integration expansion**: Consider additional tools and integrations as needs grow

For additional support, refer to the comprehensive documentation in `docs/workflows/` or create an issue in this repository.
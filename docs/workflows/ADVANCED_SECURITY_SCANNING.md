# Advanced Security Scanning

This document provides the workflow configuration for comprehensive security scanning. Since GitHub workflow files cannot be created directly, this serves as a template that maintainers can implement.

## Security Workflow Template

Create `.github/workflows/security.yml` with the following content:

```yaml
name: Security Scan

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6AM UTC
  workflow_dispatch:

permissions:
  contents: read
  security-events: write
  actions: read

jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install safety bandit[toml] semgrep
          pip install -e .
          
      - name: Run Safety (dependency vulnerabilities)
        run: |
          safety check --json --output safety-report.json || true
          safety check
          
      - name: Run Bandit (code security issues)
        run: |
          bandit -r src -f json -o bandit-report.json
          bandit -r src -f txt
          
      - name: Upload Bandit results to GitHub Security
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: bandit-report.json
          
      - name: Archive security reports
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: security-reports
          path: |
            safety-report.json
            bandit-report.json

  code-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: python
          queries: security-extended
          
      - name: Autobuild
        uses: github/codeql-action/autobuild@v3
        
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3
        with:
          category: "/language:python"

  container-scan:
    runs-on: ubuntu-latest
    if: github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository
    steps:
      - uses: actions/checkout@v4
        
      - name: Build Docker image
        run: docker build -t autogen-review-bot:${{ github.sha }} .
        
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'autogen-review-bot:${{ github.sha }}'
          format: 'sarif'
          output: 'trivy-results.sarif'
          
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

  secrets-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Install detect-secrets
        run: pip install detect-secrets
        
      - name: Run detect-secrets
        run: |
          detect-secrets scan --baseline .secrets.baseline --all-files
          detect-secrets audit .secrets.baseline

  license-compliance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Install pip-licenses
        run: pip install pip-licenses
        
      - name: Install project dependencies
        run: pip install -e .
        
      - name: Check licenses
        run: |
          pip-licenses --format=json --output-file=licenses.json
          pip-licenses --fail-on="GPL"
          
      - name: Upload license report
        uses: actions/upload-artifact@v4
        with:
          name: license-report
          path: licenses.json

  sbom-generation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Install cyclone-x
        run: pip install cyclone-dx-bom
        
      - name: Install project
        run: pip install -e .
        
      - name: Generate SBOM
        run: |
          cyclone-dx py --pip -o sbom.json
          cyclone-x py --pip -o sbom.xml --format xml
          
      - name: Upload SBOM
        uses: actions/upload-artifact@v4
        with:
          name: sbom
          path: |
            sbom.json
            sbom.xml
```

## Security Scanning Features

### Multi-Layer Security Analysis

#### 1. Dependency Vulnerability Scanning
- **Safety**: Scans Python dependencies for known vulnerabilities
- **Automated Reports**: JSON reports for integration with security tools
- **Continuous Updates**: Database updated regularly with latest vulnerabilities

#### 2. Static Code Analysis
- **Bandit**: Security-focused static analysis for Python code
- **SARIF Integration**: Results uploaded to GitHub Security tab
- **Custom Rules**: Project-specific security patterns

#### 3. Advanced Code Analysis
- **CodeQL**: GitHub's semantic code analysis engine
- **Security-Extended Queries**: Enhanced security-focused analysis
- **Multi-Language Support**: Extensible to other languages

#### 4. Container Security
- **Trivy Scanner**: Comprehensive container vulnerability scanning
- **Multi-Layer Analysis**: Base image, dependencies, and configuration
- **SARIF Integration**: Results visible in GitHub Security tab

### Compliance & Governance

#### 1. Secrets Detection
- **Detect-Secrets**: Prevents accidental secret commits
- **Baseline Management**: Approved exclusions and false positives
- **Historical Scanning**: Full repository history analysis

#### 2. License Compliance
- **Automated License Checking**: All dependencies analyzed
- **Policy Enforcement**: Blocks problematic licenses (GPL, etc.)
- **Compliance Reports**: JSON reports for legal review

#### 3. SBOM Generation
- **Software Bill of Materials**: Complete dependency inventory
- **Multiple Formats**: JSON and XML for different tools
- **Vulnerability Mapping**: Links to security databases

## Setup Instructions

### 1. Create the Workflow File
Copy the template to `.github/workflows/security.yml`

### 2. Configure Repository Settings
```yaml
# Repository settings required
permissions:
  security-events: write  # For SARIF uploads
  actions: read          # For workflow access
  contents: read         # For repository access
```

### 3. Enable Security Features
- **GitHub Advanced Security**: Required for CodeQL
- **Dependency Graph**: Enable in repository settings
- **Vulnerability Alerts**: Enable Dependabot alerts
- **Security Advisories**: Enable private vulnerability reporting

### 4. Configure Branch Protection
```yaml
# Branch protection rules
required_status_checks:
  strict: true
  contexts:
    - "Security Scan / dependency-scan"
    - "Security Scan / code-analysis"
    - "Security Scan / secrets-scan"
```

## Security Policies

### Vulnerability Response Process

#### High Severity Vulnerabilities
1. **Immediate Assessment**: Triage within 24 hours
2. **Impact Analysis**: Determine exposure and risk
3. **Remediation Planning**: Develop fix strategy
4. **Emergency Patches**: Deploy critical fixes immediately

#### Medium/Low Severity Vulnerabilities
1. **Weekly Review**: Regular vulnerability assessment
2. **Batch Updates**: Group related fixes
3. **Testing Validation**: Ensure fixes don't break functionality
4. **Scheduled Deployment**: Include in regular releases

### Security Metrics Tracking

#### Key Performance Indicators
- **Time to Detection**: Average time to identify vulnerabilities
- **Time to Remediation**: Average time to fix vulnerabilities
- **Coverage Metrics**: Percentage of code/dependencies scanned
- **False Positive Ratio**: Accuracy of security tools

#### Reporting Dashboard
```yaml
# Example Grafana queries for security metrics
queries:
  - name: "Vulnerability Detection Time"
    query: "avg(vulnerability_detection_time_hours)"
  - name: "Critical Vulnerabilities Open"
    query: "count(vulnerabilities{severity='critical',status='open'})"
  - name: "Security Scan Coverage"
    query: "security_scan_coverage_percentage"
```

## Integration with Development Workflow

### Pre-Commit Integration
```yaml
# .pre-commit-config.yaml additions
- repo: local
  hooks:
    - id: security-scan
      name: security-scan
      entry: bandit
      language: system
      args: ['-r', 'src', '-f', 'json']
      types: [python]
```

### IDE Integration
- **VS Code Extensions**: Bandit, Safety, CodeQL
- **PyCharm Plugins**: Security scanning during development
- **Git Hooks**: Prevent commits with security issues

### CI/CD Integration Points
1. **Pull Request Checks**: Security scans on every PR
2. **Merge Requirements**: Must pass security scans to merge
3. **Release Gates**: Enhanced security validation before release
4. **Deployment Monitoring**: Runtime security monitoring

## Advanced Configuration

### Custom Security Rules
```python
# Custom Bandit rules for project-specific patterns
import bandit
from bandit.core import test_properties

@test_properties.checks('Call')
def custom_security_check(context):
    # Custom security validation logic
    pass
```

### Tool Configuration
```toml
# pyproject.toml security tool configuration
[tool.bandit]
exclude_dirs = ["tests", "migrations"]
skips = ["B101"]  # Skip assert_used in tests

[tool.safety]
ignore = ["12345"]  # Ignore specific vulnerability IDs after review
```

This advanced security scanning setup provides enterprise-grade security monitoring and compliance capabilities, ensuring that security vulnerabilities are detected early and addressed promptly throughout the development lifecycle.
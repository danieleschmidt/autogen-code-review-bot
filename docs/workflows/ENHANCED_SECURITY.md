# Enhanced Security Workflow Configuration

This document outlines the enhanced security workflows that should be added to `.github/workflows/` for advanced security scanning and compliance.

## Required GitHub Actions Workflows

### 1. Security Scanning Workflow (`.github/workflows/security.yml`)

```yaml
name: Security Scanning
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 3 * * 1'  # Weekly Monday 3 AM

jobs:
  security-scan:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      actions: read
      contents: read
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e .[dev]
          pip install safety pip-audit
      
      - name: Run Bandit Security Scan
        run: bandit -r src -f sarif -o bandit-results.sarif
      
      - name: Upload Bandit SARIF
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: bandit-results.sarif
      
      - name: Safety Check
        run: safety check --json --output safety-report.json
        continue-on-error: true
      
      - name: Pip Audit
        run: pip-audit --format=json --output=pip-audit-report.json
        continue-on-error: true
      
      - name: Upload Security Reports
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: security-reports
          path: |
            bandit-results.sarif
            safety-report.json
            pip-audit-report.json
```

### 2. CodeQL Analysis Workflow (`.github/workflows/codeql.yml`)

```yaml
name: CodeQL Analysis
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 2'  # Weekly Tuesday 6 AM

jobs:
  analyze:
    name: Analyze Python Code
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write
    
    strategy:
      fail-fast: false
      matrix:
        language: [python]
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: ${{ matrix.language }}
          queries: security-extended,security-and-quality
      
      - name: Autobuild
        uses: github/codeql-action/autobuild@v2
      
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2
        with:
          category: "/language:${{matrix.language}}"
```

### 3. Dependency Review Workflow (`.github/workflows/dependency-review.yml`)

```yaml
name: Dependency Review
on:
  pull_request:
    branches: [main]

permissions:
  contents: read
  pull-requests: write

jobs:
  dependency-review:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4
        
      - name: Dependency Review
        uses: actions/dependency-review-action@v3
        with:
          fail-on-severity: moderate
          allow-licenses: MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause
          deny-licenses: GPL-2.0, GPL-3.0
```

### 4. Container Security Scanning (`.github/workflows/container-security.yml`)

```yaml
name: Container Security
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  container-scan:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Build Docker image
        run: docker build -t autogen-bot:${{ github.sha }} .
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'autogen-bot:${{ github.sha }}'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'
      
      - name: Run Hadolint
        uses: hadolint/hadolint-action@v3.1.0
        with:
          dockerfile: Dockerfile
          format: sarif
          output-file: hadolint-results.sarif
      
      - name: Upload Hadolint results
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: hadolint-results.sarif
```

## Implementation Instructions

1. **Create workflow files**: Add the above workflows to `.github/workflows/` directory
2. **Configure branch protection**: Require security checks to pass before merge
3. **Set up SARIF uploads**: Enable security tab in repository settings
4. **Configure notifications**: Set up alerts for security findings
5. **Regular reviews**: Schedule monthly security report reviews

## Security Benefits

- **Automated vulnerability detection**: Continuous scanning for known vulnerabilities
- **License compliance**: Automatic license checking for dependencies
- **Container security**: Docker image vulnerability scanning
- **Code quality**: Static analysis for security anti-patterns
- **Compliance reporting**: SARIF format for security tools integration

## Configuration Requirements

Enable the following in repository settings:
- Security tab for SARIF uploads
- Dependabot alerts
- Code scanning alerts
- Private vulnerability reporting
- Branch protection rules requiring security checks
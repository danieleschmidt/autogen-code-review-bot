# 🔐 Comprehensive Security Automation

## Advanced Security Scanning Workflow

**File**: `.github/workflows/security.yml` (Create new file)

```yaml
name: Security Scanning

on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM UTC
  push:
    branches: [main]
  pull_request:
    branches: [main]
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
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev,monitoring]
          pip install safety pip-audit
      
      - name: Run Safety check
        run: |
          safety check --json --output safety-report.json
          safety check --output safety-report.txt
        continue-on-error: true
      
      - name: Run pip-audit
        run: |
          pip-audit --format=json --output=pip-audit-report.json
          pip-audit --format=cyclonedx-json --output=pip-audit-sbom.json
        continue-on-error: true
      
      - name: Upload dependency scan results
        uses: actions/upload-artifact@v4
        with:
          name: dependency-scan-results
          path: |
            safety-report.json
            safety-report.txt
            pip-audit-report.json
            pip-audit-sbom.json
          retention-days: 30

  code-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install analysis tools
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]
          pip install semgrep
      
      - name: Run Bandit security scan
        run: |
          bandit -r src -f sarif -o bandit-results.sarif
          bandit -r src -f json -o bandit-results.json
        continue-on-error: true
      
      - name: Run Semgrep
        run: |
          semgrep --config=auto --sarif --output=semgrep-results.sarif src/
          semgrep --config=auto --json --output=semgrep-results.json src/
        continue-on-error: true
      
      - name: Upload SARIF results to GitHub
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: |
            bandit-results.sarif
            semgrep-results.sarif
      
      - name: Upload analysis results
        uses: actions/upload-artifact@v4
        with:
          name: code-analysis-results  
          path: |
            bandit-results.json
            semgrep-results.json
          retention-days: 30

  container-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Build Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          load: true
          tags: autogen-bot:test
          cache-from: type=gha
          cache-to: type=gha,mode=max
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'autogen-bot:test'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Run Trivy JSON scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'autogen-bot:test'
          format: 'json'
          output: 'trivy-results.json'
      
      - name: Upload Trivy SARIF results
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'
      
      - name: Upload container scan results
        uses: actions/upload-artifact@v4
        with:
          name: container-scan-results
          path: trivy-results.json
          retention-days: 30

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
      
      - name: Run secrets scan
        run: |
          detect-secrets scan --all-files --baseline .secrets.baseline
          detect-secrets audit .secrets.baseline
        continue-on-error: true
      
      - name: Validate secrets baseline
        run: |
          detect-secrets scan --baseline .secrets.baseline
      
      - name: Upload updated baseline
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: secrets-baseline
          path: .secrets.baseline
          retention-days: 7

  compliance-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Check required files
        run: |
          echo "Checking compliance requirements..."
          
          # Check required documentation files
          files=(
            "README.md"
            "LICENSE" 
            "SECURITY.md"
            "CONTRIBUTING.md"
            "CODE_OF_CONDUCT.md"
            ".github/CODEOWNERS"
            ".github/dependabot.yml"
          )
          
          missing=()
          for file in "${files[@]}"; do
            if [[ ! -f "$file" ]]; then
              missing+=("$file")
            fi
          done
          
          if [[ ${#missing[@]} -gt 0 ]]; then
            echo "Missing required files:"
            printf '%s\n' "${missing[@]}"
            exit 1
          fi
          
          echo "All required compliance files present ✓"
      
      - name: Check security policy
        run: |
          if grep -q "security@" SECURITY.md; then
            echo "Security contact found ✓"
          else
            echo "Security contact missing in SECURITY.md"
            exit 1
          fi
      
      - name: Validate pyproject.toml
        run: |
          python -c "
          import tomllib
          with open('pyproject.toml', 'rb') as f:
              config = tomllib.load(f)
          
          project = config.get('project', {})
          required = ['name', 'version', 'description', 'authors', 'license']
          
          missing = [field for field in required if field not in project]
          if missing:
              print(f'Missing project fields: {missing}')
              exit(1)
          
          print('pyproject.toml validation passed ✓')
          "

  security-report:
    runs-on: ubuntu-latest
    needs: [dependency-scan, code-analysis, container-scan, secrets-scan, compliance-check]
    if: always()
    steps:
      - uses: actions/checkout@v4
      
      - name: Download all artifacts
        uses: actions/download-artifact@v4
        with:
          path: security-results/
      
      - name: Generate security report
        run: |
          echo "# Security Scan Report - $(date)" > security-report.md
          echo "" >> security-report.md
          
          echo "## Scan Results Summary" >> security-report.md
          echo "- **Dependency Scan**: $(if [[ -f security-results/dependency-scan-results/safety-report.json ]]; then echo '✅ Completed'; else echo '❌ Failed'; fi)" >> security-report.md
          echo "- **Code Analysis**: $(if [[ -f security-results/code-analysis-results/bandit-results.json ]]; then echo '✅ Completed'; else echo '❌ Failed'; fi)" >> security-report.md
          echo "- **Container Scan**: $(if [[ -f security-results/container-scan-results/trivy-results.json ]]; then echo '✅ Completed'; else echo '❌ Failed'; fi)" >> security-report.md
          echo "- **Secrets Scan**: $(if [[ -f security-results/secrets-baseline/.secrets.baseline ]]; then echo '✅ Completed'; else echo '❌ Failed'; fi)" >> security-report.md
          echo "- **Compliance Check**: ${{ needs.compliance-check.result == 'success' && '✅ Passed' || '❌ Failed' }}" >> security-report.md
          
          echo "" >> security-report.md
          echo "## Workflow Status" >> security-report.md
          echo "- Repository: ${{ github.repository }}" >> security-report.md
          echo "- Branch: ${{ github.ref_name }}" >> security-report.md
          echo "- Commit: ${{ github.sha }}" >> security-report.md
          echo "- Triggered by: ${{ github.event_name }}" >> security-report.md
      
      - name: Upload security report
        uses: actions/upload-artifact@v4
        with:
          name: security-report
          path: security-report.md
          retention-days: 90
```

## Security Framework Features

### 🔍 Multi-Tool Analysis
- **Bandit**: SAST for Python security issues
- **Semgrep**: Advanced pattern-based security scanning
- **Safety**: Python dependency vulnerability scanning
- **pip-audit**: OSV vulnerability database scanning
- **Trivy**: Container image vulnerability scanning
- **detect-secrets**: Secrets detection and baseline management

### 📊 Comprehensive Reporting
- **SARIF Integration**: Direct results to GitHub Security tab
- **Artifact Management**: 30-day retention for scan results
- **Consolidated Reports**: Single security dashboard
- **Compliance Automation**: Required file and policy validation

### ⚡ Automation Features
- **Daily Scanning**: Automated security checks
- **PR Integration**: Security validation on code changes
- **Manual Triggers**: On-demand security assessment
- **Compliance Monitoring**: Continuous policy enforcement

## Implementation Guide

1. **Create Security Workflow**:
   ```bash
   mkdir -p .github/workflows
   # Copy YAML above to .github/workflows/security.yml
   ```

2. **Configure Permissions**: Ensure `security-events: write` permission

3. **Test Implementation**:
   ```bash
   git add .github/workflows/security.yml
   git commit -m "feat: comprehensive security automation"
   git push
   ```

## Expected Security Benefits

- 🛡️ **7-Tool Coverage**: Comprehensive vulnerability detection
- 📈 **Daily Monitoring**: Continuous security posture assessment  
- 🔒 **SARIF Integration**: GitHub Security tab visibility
- 📋 **Compliance**: Automated policy validation
- 🚨 **Early Detection**: PR-level security validation
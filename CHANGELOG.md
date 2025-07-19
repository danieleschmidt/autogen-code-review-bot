# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Security
- Enhanced subprocess security in PR analysis with input validation and explicit shell=False
- Added comprehensive parameter validation for _run_command function
- Prevented potential shell injection vulnerabilities

### Added  
- **Configuration management system** following Twelve-Factor App principles
- **Environment variable support** for all configuration values (AUTOGEN_*)
- **File-based configuration** with JSON format support  
- **Configuration validation** with comprehensive error handling
- **Structured logging system** with JSON output and request correlation IDs
- **Request context tracking** for correlated logging across operations
- **Comprehensive metrics system** with counters, gauges, and histograms
- **Metrics collection** and performance timing for observability
- **Multiple export formats** (JSON, Prometheus) for monitoring integration
- **Thread-safe metrics** with label support for dimensional data
- **Log sanitization** to prevent sensitive data exposure
- Added Ruby language detection and default linter configuration
- Autonomous development workflow and backlog management system
- Impact-ranked backlog with WSJF scoring methodology

### Improved
- **Removed hardcoded values**: GitHub API URL, timeouts, linter mappings now configurable
- **Enhanced deployment flexibility** with environment-based configuration
- GitHub integration now includes comprehensive request/response logging
- PR analysis includes detailed operation logging and metrics
- Command execution logging for debugging and monitoring

## [0.2.0] - 2025-06-29
### Added
- Public API function `analyze_and_comment`
- GitHub API retry logic
- CI workflow with ruff, bandit, and pytest
- Contributing guidelines and CODEOWNERS

## [0.1.0] - 2024-07-02
### Added
- Initial project skeleton.

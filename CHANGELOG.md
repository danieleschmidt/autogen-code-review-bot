# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Security
- Enhanced subprocess security in PR analysis with input validation and explicit shell=False
- Added comprehensive parameter validation for _run_command function
- Prevented potential shell injection vulnerabilities

### Added  
- **Structured logging system** with JSON output and request correlation IDs
- **Request context tracking** for correlated logging across operations
- **Metrics collection** and performance timing for observability
- **Log sanitization** to prevent sensitive data exposure
- Added Ruby language detection and default linter configuration
- Autonomous development workflow and backlog management system
- Impact-ranked backlog with WSJF scoring methodology

### Improved
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

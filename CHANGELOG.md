# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Security
- Fixed command injection vulnerability in pr_analysis module
- Added input validation and command allowlisting
- Enhanced path traversal protection
- Improved error handling for malicious inputs

### Added
- Added Ruby language detection and default linter configuration
- Comprehensive security validation functions
- Security-focused test suite
- Complete CLI interface with three entry points:
  - `bot.py` - Webhook server and manual analysis
  - `setup_webhook.py` - GitHub webhook management utility
  - `review_pr.py` - Manual PR review script
- Webhook server with signature verification
- GitHub API integration for automated PR reviews
- Local repository analysis capabilities

## [0.2.0] - 2025-06-29
### Added
- Public API function `analyze_and_comment`
- GitHub API retry logic
- CI workflow with ruff, bandit, and pytest
- Contributing guidelines and CODEOWNERS

## [0.1.0] - 2024-07-02
### Added
- Initial project skeleton.

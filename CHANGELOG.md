# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- **Agent Conversation System**: Revolutionary AI-powered code review enhancement
  - Intelligent agent-to-agent discussions for comprehensive code analysis
  - Sentiment analysis to determine when agents should engage in conversation
  - Automatic resolution detection to prevent infinite discussion loops
  - ConversationManager with configurable parameters and turn limits
  - AgentConversation class for managing multi-turn discussions
  - Enhanced CLI with `--agent-config` flag for AI conversation mode
  - Seamless integration with existing PR analysis workflow
  - Comprehensive test coverage including integration tests
  - Fallback to traditional analysis when conversation system unavailable
### Security
- Fixed command injection vulnerability in pr_analysis module
- Added input validation and command allowlisting
- Enhanced path traversal protection
- Improved error handling for malicious inputs

### Added
- **Structured JSON Logging System**: Comprehensive observability improvements
  - JSON-formatted log output with structured fields
  - Request ID correlation across all operations (webhook → analysis → GitHub API)
  - Operation timing and success/failure tracking
  - Context-aware logging with automatic field enrichment
  - Configurable log levels and service naming
  - Enhanced error tracking with exception details and types
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

### Improved
- **Enhanced Observability**: All major operations now include structured logging
  - Webhook request processing with client IP and user agent tracking
  - PR analysis workflow with repository and commit SHA context
  - Git clone operations with timing and error details
  - GitHub API calls with retry attempts and response metadata
  - Manual analysis operations with comprehensive error handling

## [0.2.0] - 2025-06-29
### Added
- Public API function `analyze_and_comment`
- GitHub API retry logic
- CI workflow with ruff, bandit, and pytest
- Contributing guidelines and CODEOWNERS

## [0.1.0] - 2024-07-02
### Added
- Initial project skeleton.

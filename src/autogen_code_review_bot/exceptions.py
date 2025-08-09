"""Standard exception hierarchy for the code review bot."""


class CodeReviewBotError(Exception):
    """Base exception for all code review bot errors.
    
    This provides a common base for all custom exceptions in the system,
    making it easier to catch and handle bot-specific errors.
    """
    pass


class AnalysisError(CodeReviewBotError):
    """Errors during code analysis operations.
    
    Raised when analysis operations fail due to tool issues,
    code problems, or processing errors.
    """
    pass


class ConfigurationError(CodeReviewBotError):
    """Configuration-related errors.
    
    Raised when configuration files are invalid, missing required
    settings, or contain incompatible values.
    """
    pass


class ExternalServiceError(CodeReviewBotError):
    """Errors from external services (GitHub, etc).
    
    Raised when external API calls fail, rate limits are hit,
    or service connectivity issues occur.
    """
    pass


class ValidationError(CodeReviewBotError):
    """Input validation errors.
    
    Raised when user input, file paths, or data doesn't meet
    expected formats or security requirements.
    """
    pass


class CacheError(CodeReviewBotError):
    """Cache operation errors.
    
    Raised when cache operations fail due to storage issues,
    permissions, or corruption.
    """
    pass


class ToolError(CodeReviewBotError):
    """Errors related to external tools.
    
    Raised when linters, security scanners, or other external
    tools fail to execute or return unexpected results.
    """
    pass


class LinterError(ToolError):
    """Exception for linter-related errors."""
    pass


class SecurityError(AnalysisError):
    """Exception for security-related errors."""
    pass
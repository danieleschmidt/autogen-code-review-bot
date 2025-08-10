"""Token security utilities for safe logging and error handling."""

import re
from typing import Any, Dict, List
from urllib.parse import urlparse, urlunparse


class TokenMasker:
    """Utility class for masking tokens and sensitive data in logs and errors."""

    # Common token patterns to mask
    TOKEN_PATTERNS = [
        # GitHub tokens (ghp_, ghs_, gho_, ghu_, ghr_)
        (re.compile(r'\bghp_[A-Za-z0-9]{36}\b'), '***GITHUB_TOKEN***'),
        (re.compile(r'\bghs_[A-Za-z0-9]{36}\b'), '***GITHUB_TOKEN***'),
        (re.compile(r'\bgho_[A-Za-z0-9]{36}\b'), '***GITHUB_TOKEN***'),
        (re.compile(r'\bghu_[A-Za-z0-9]{36}\b'), '***GITHUB_TOKEN***'),
        (re.compile(r'\bghr_[A-Za-z0-9]{36}\b'), '***GITHUB_TOKEN***'),

        # Generic GitHub personal access tokens (older format)
        (re.compile(r'\b[a-f0-9]{40}\b'), '***TOKEN***'),

        # Authorization headers
        (re.compile(r'Authorization:\s*token\s+[^\s]+', re.IGNORECASE), 'Authorization: token ***'),
        (re.compile(r'Authorization:\s*Bearer\s+[^\s]+', re.IGNORECASE), 'Authorization: Bearer ***'),

        # Basic auth in URLs
        (re.compile(r'https://[^:]+:[^@]+@'), 'https://***:***@'),

        # Generic secrets (patterns that look like API keys/tokens)
        (re.compile(r'\b[A-Za-z0-9_-]{32,}\b'), '***SECRET***'),

        # Webhook secrets in query params or JSON
        (re.compile(r'["\']?secret["\']?\s*[:=]\s*["\']?[^"\'&\s]+["\']?', re.IGNORECASE), 'secret: "***"'),
        (re.compile(r'["\']?webhook_secret["\']?\s*[:=]\s*["\']?[^"\'&\s]+["\']?', re.IGNORECASE), 'webhook_secret: "***"'),

        # Password patterns
        (re.compile(r'["\']?password["\']?\s*[:=]\s*["\']?[^"\'&\s]+["\']?', re.IGNORECASE), 'password: "***"'),
        (re.compile(r'["\']?pass["\']?\s*[:=]\s*["\']?[^"\'&\s]+["\']?', re.IGNORECASE), 'pass: "***"'),
    ]

    @classmethod
    def mask_sensitive_data(cls, text: str) -> str:
        """Mask sensitive data in text using predefined patterns.
        
        Args:
            text: Text that may contain sensitive data
            
        Returns:
            Text with sensitive data masked
        """
        if not isinstance(text, str):
            text = str(text)

        masked_text = text

        for pattern, replacement in cls.TOKEN_PATTERNS:
            masked_text = pattern.sub(replacement, masked_text)

        return masked_text

    @classmethod
    def mask_url(cls, url: str, token: str = None) -> str:
        """Mask tokens in URLs, especially in authentication sections.
        
        Args:
            url: URL that may contain tokens
            token: Specific token to mask (if known)
            
        Returns:
            URL with tokens masked
        """
        if not url:
            return url

        # First, mask any specific token if provided
        if token and token in url:
            url = url.replace(token, "***")

        # Parse URL and mask authentication info
        try:
            parsed = urlparse(url)
            if parsed.username or parsed.password:
                # Replace username and password with masked versions
                netloc = parsed.netloc
                if '@' in netloc:
                    auth_part, host_part = netloc.rsplit('@', 1)
                    masked_netloc = f"***:***@{host_part}"
                    parsed = parsed._replace(netloc=masked_netloc)
                    url = urlunparse(parsed)
        except Exception:
            # If URL parsing fails, fall back to pattern matching
            pass

        # Apply general token masking patterns
        return cls.mask_sensitive_data(url)

    @classmethod
    def mask_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively mask sensitive data in dictionary.
        
        Args:
            data: Dictionary that may contain sensitive data
            
        Returns:
            Dictionary with sensitive data masked
        """
        if not isinstance(data, dict):
            return data

        masked_data = {}

        for key, value in data.items():
            key_lower = key.lower()

            # Keys that should always be masked
            if any(sensitive_key in key_lower for sensitive_key in
                   ['token', 'password', 'secret', 'auth', 'key', 'credential']):
                masked_data[key] = "***"
            elif isinstance(value, dict):
                masked_data[key] = cls.mask_dict(value)
            elif isinstance(value, (list, tuple)):
                masked_data[key] = cls.mask_list(value)
            elif isinstance(value, str):
                masked_data[key] = cls.mask_sensitive_data(value)
            else:
                masked_data[key] = value

        return masked_data

    @classmethod
    def mask_list(cls, data: List[Any]) -> List[Any]:
        """Recursively mask sensitive data in list.
        
        Args:
            data: List that may contain sensitive data
            
        Returns:
            List with sensitive data masked
        """
        if not isinstance(data, (list, tuple)):
            return data

        masked_data = []

        for item in data:
            if isinstance(item, dict):
                masked_data.append(cls.mask_dict(item))
            elif isinstance(item, (list, tuple)):
                masked_data.append(cls.mask_list(item))
            elif isinstance(item, str):
                masked_data.append(cls.mask_sensitive_data(item))
            else:
                masked_data.append(item)

        return type(data)(masked_data)  # Preserve original type (list/tuple)

    @classmethod
    def mask_exception_message(cls, exc: Exception) -> str:
        """Mask sensitive data in exception messages.
        
        Args:
            exc: Exception that may contain sensitive data in its message
            
        Returns:
            Masked exception message
        """
        message = str(exc)
        return cls.mask_sensitive_data(message)

    @classmethod
    def mask_logging_args(cls, *args, **kwargs) -> tuple:
        """Mask sensitive data in logging arguments.
        
        Args:
            *args: Positional arguments for logging
            **kwargs: Keyword arguments for logging
            
        Returns:
            Tuple of (masked_args, masked_kwargs)
        """
        # Mask positional arguments
        masked_args = []
        for arg in args:
            if isinstance(arg, str):
                masked_args.append(cls.mask_sensitive_data(arg))
            elif isinstance(arg, dict):
                masked_args.append(cls.mask_dict(arg))
            elif isinstance(arg, (list, tuple)):
                masked_args.append(cls.mask_list(arg))
            else:
                masked_args.append(arg)

        # Mask keyword arguments
        masked_kwargs = cls.mask_dict(kwargs)

        return tuple(masked_args), masked_kwargs


# Convenience functions for common use cases
def mask_token_in_url(url: str, token: str = None) -> str:
    """Convenience function to mask tokens in URLs."""
    return TokenMasker.mask_url(url, token)


def mask_sensitive_in_text(text: str) -> str:
    """Convenience function to mask sensitive data in text."""
    return TokenMasker.mask_sensitive_data(text)


def safe_log_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to safely log dictionary data."""
    return TokenMasker.mask_dict(data)


def safe_exception_str(exc: Exception) -> str:
    """Convenience function to safely convert exception to string."""
    return TokenMasker.mask_exception_message(exc)

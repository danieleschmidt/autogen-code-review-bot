"""Enhanced input validation and sanitization for security and data integrity."""

import hashlib
import ipaddress
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import yaml

from .exceptions import SecurityError, ValidationError
from .logging_config import get_logger
from .metrics import get_metrics_registry

logger = get_logger(__name__)
metrics = get_metrics_registry()


class SecurityValidator:
    """Security-focused validation utilities."""

    # Dangerous file extensions that should be blocked
    DANGEROUS_EXTENSIONS = {
        '.exe', '.bat', '.cmd', '.scr', '.pif', '.com', '.dll', '.sys',
        '.vbs', '.js', '.jse', '.jar', '.class', '.sh', '.bash', '.zsh',
        '.ps1', '.psm1', '.psd1', '.app', '.deb', '.rpm', '.dmg', '.pkg',
        '.msi', '.apk', '.ipa'
    }

    # Allowed file extensions for code analysis
    ALLOWED_CODE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.rs', '.rb', '.java',
        '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.cs', '.php', '.swift',
        '.kt', '.scala', '.clj', '.hs', '.elm', '.dart', '.lua', '.r',
        '.sql', '.html', '.css', '.scss', '.sass', '.less', '.vue',
        '.json', '.yaml', '.yml', '.toml', '.xml', '.md', '.txt', '.cfg',
        '.conf', '.ini', '.env', '.gitignore', '.dockerignore'
    }

    # Suspicious patterns that might indicate malicious content
    SUSPICIOUS_PATTERNS = [
        r'eval\s*\(',
        r'exec\s*\(',
        r'system\s*\(',
        r'shell_exec\s*\(',
        r'passthru\s*\(',
        r'proc_open\s*\(',
        r'popen\s*\(',
        r'subprocess\.call',
        r'subprocess\.run',
        r'os\.system',
        r'os\.exec',
        r'__import__\s*\(',
        r'importlib\.import_module',
        r'pickle\.loads',
        r'marshal\.loads',
        r'base64\.b64decode',
        r'codecs\.decode'
    ]

    @classmethod
    def validate_file_path(cls, file_path: str, allow_absolute: bool = False, base_path: Optional[str] = None) -> str:
        """Validate and sanitize file paths to prevent directory traversal attacks.
        
        Args:
            file_path: File path to validate
            allow_absolute: Whether to allow absolute paths
            base_path: Base path that all paths must be within
            
        Returns:
            Sanitized file path
            
        Raises:
            ValidationError: If path is invalid or unsafe
        """
        if not file_path or not isinstance(file_path, str):
            raise ValidationError("File path must be a non-empty string")

        # Remove any null bytes
        file_path = file_path.replace('\x00', '')

        # Normalize the path
        try:
            path = Path(file_path).resolve()
        except (OSError, ValueError) as e:
            raise ValidationError(f"Invalid file path: {e}")

        # Check for absolute paths if not allowed
        if not allow_absolute and path.is_absolute():
            # Convert to relative path if base_path provided
            if base_path:
                try:
                    base = Path(base_path).resolve()
                    path = path.relative_to(base)
                except ValueError:
                    raise ValidationError(f"Path {file_path} is outside allowed base path {base_path}")
            else:
                raise ValidationError("Absolute paths not allowed")

        # Check for directory traversal attempts
        path_str = str(path)
        if '..' in path_str or path_str.startswith('/'):
            if not allow_absolute:
                raise ValidationError("Path traversal detected")

        # Validate file extension
        extension = path.suffix.lower()
        if extension in cls.DANGEROUS_EXTENSIONS:
            raise SecurityError(f"Dangerous file extension not allowed: {extension}")

        # Log suspicious extensions
        if extension and extension not in cls.ALLOWED_CODE_EXTENSIONS:
            logger.warning("Unusual file extension detected",
                         file_path=file_path,
                         extension=extension)
            metrics.record_counter("validation_unusual_extension", 1, tags={"extension": extension})

        return str(path)

    @classmethod
    def validate_code_content(cls, content: str, filename: Optional[str] = None) -> bool:
        """Validate code content for suspicious patterns.
        
        Args:
            content: Code content to validate
            filename: Optional filename for context
            
        Returns:
            True if content appears safe
            
        Raises:
            SecurityError: If suspicious patterns are detected
        """
        if not isinstance(content, str):
            raise ValidationError("Content must be a string")

        # Check for suspicious patterns
        suspicious_found = []
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                suspicious_found.append(pattern)

        if suspicious_found:
            logger.warning("Suspicious patterns detected in code",
                         filename=filename or "unknown",
                         patterns=suspicious_found[:5])  # Limit logged patterns
            metrics.record_counter("validation_suspicious_patterns", 1,
                                 tags={"filename": filename or "unknown"})

            # For now, log but don't block - could be legitimate code
            # In stricter environments, this could raise SecurityError

        # Check for excessively long lines (potential obfuscation)
        lines = content.split('\n')
        for i, line in enumerate(lines[:100]):  # Check first 100 lines
            if len(line) > 10000:  # Very long line
                logger.warning("Extremely long line detected",
                             filename=filename or "unknown",
                             line_number=i+1,
                             line_length=len(line))
                metrics.record_counter("validation_long_lines", 1)

        return True

    @classmethod
    def validate_url(cls, url: str) -> str:
        """Validate and sanitize URLs.
        
        Args:
            url: URL to validate
            
        Returns:
            Validated URL
            
        Raises:
            ValidationError: If URL is invalid or unsafe
        """
        if not url or not isinstance(url, str):
            raise ValidationError("URL must be a non-empty string")

        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValidationError(f"Invalid URL format: {e}")

        # Check scheme
        if parsed.scheme not in ['http', 'https']:
            raise ValidationError(f"Invalid URL scheme: {parsed.scheme}")

        # Check for localhost/internal addresses
        if parsed.hostname:
            try:
                ip = ipaddress.ip_address(parsed.hostname)
                if ip.is_private or ip.is_loopback:
                    raise SecurityError("Private/loopback addresses not allowed")
            except ipaddress.AddressValueError:
                # Hostname is not an IP address, check for localhost
                if parsed.hostname.lower() in ['localhost', '127.0.0.1', '::1']:
                    raise SecurityError("Localhost addresses not allowed")

        return url

    @classmethod
    def sanitize_input(cls, value: str, max_length: int = 1000, allow_html: bool = False) -> str:
        """Sanitize input strings.
        
        Args:
            value: Input string to sanitize
            max_length: Maximum allowed length
            allow_html: Whether to allow HTML content
            
        Returns:
            Sanitized string
            
        Raises:
            ValidationError: If input is invalid
        """
        if not isinstance(value, str):
            raise ValidationError("Input must be a string")

        # Check length
        if len(value) > max_length:
            raise ValidationError(f"Input too long: {len(value)} > {max_length}")

        # Remove null bytes and other control characters
        value = ''.join(char for char in value if ord(char) >= 32 or char in '\t\n\r')

        # Normalize Unicode
        value = unicodedata.normalize('NFKC', value)

        # Remove HTML if not allowed
        if not allow_html:
            # Simple HTML tag removal (for basic safety)
            value = re.sub(r'<[^>]*>', '', value)

            # Remove common XSS patterns
            xss_patterns = [
                r'javascript:',
                r'vbscript:',
                r'onload\s*=',
                r'onerror\s*=',
                r'onclick\s*=',
                r'onmouseover\s*='
            ]

            for pattern in xss_patterns:
                value = re.sub(pattern, '', value, flags=re.IGNORECASE)

        return value.strip()


class DataValidator:
    """Data structure and format validation."""

    @staticmethod
    def validate_json(data: str, max_size: int = 1024 * 1024) -> Dict[str, Any]:
        """Validate and parse JSON data.
        
        Args:
            data: JSON string to validate
            max_size: Maximum size in bytes
            
        Returns:
            Parsed JSON data
            
        Raises:
            ValidationError: If JSON is invalid
        """
        if not isinstance(data, str):
            raise ValidationError("JSON data must be a string")

        if len(data.encode('utf-8')) > max_size:
            raise ValidationError(f"JSON data too large: {len(data)} bytes > {max_size}")

        try:
            parsed = json.loads(data)
            return parsed
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {e}")

    @staticmethod
    def validate_yaml(data: str, max_size: int = 1024 * 1024) -> Dict[str, Any]:
        """Validate and parse YAML data.
        
        Args:
            data: YAML string to validate
            max_size: Maximum size in bytes
            
        Returns:
            Parsed YAML data
            
        Raises:
            ValidationError: If YAML is invalid
        """
        if not isinstance(data, str):
            raise ValidationError("YAML data must be a string")

        if len(data.encode('utf-8')) > max_size:
            raise ValidationError(f"YAML data too large: {len(data)} bytes > {max_size}")

        try:
            parsed = yaml.safe_load(data)
            return parsed
        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid YAML: {e}")

    @staticmethod
    def validate_github_webhook_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate GitHub webhook payload structure.
        
        Args:
            payload: Webhook payload dictionary
            
        Returns:
            Validated payload
            
        Raises:
            ValidationError: If payload structure is invalid
        """
        if not isinstance(payload, dict):
            raise ValidationError("Payload must be a dictionary")

        # Check for required fields based on event type
        required_fields = ['action', 'sender']
        for field in required_fields:
            if field not in payload:
                raise ValidationError(f"Missing required field: {field}")

        # Validate sender
        sender = payload.get('sender', {})
        if not isinstance(sender, dict) or 'login' not in sender:
            raise ValidationError("Invalid sender information")

        # Validate repository information if present
        if 'repository' in payload:
            repo = payload['repository']
            if not isinstance(repo, dict) or 'full_name' not in repo:
                raise ValidationError("Invalid repository information")

            # Check repository name format
            repo_name = repo['full_name']
            if not re.match(r'^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$', repo_name):
                raise ValidationError("Invalid repository name format")

        return payload

    @staticmethod
    def validate_configuration(config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate configuration dictionary against schema.
        
        Args:
            config: Configuration dictionary
            schema: Optional schema for validation
            
        Returns:
            Validated configuration
            
        Raises:
            ValidationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValidationError("Configuration must be a dictionary")

        # Basic validation without external schema library
        if schema:
            for key, constraints in schema.items():
                if key in config:
                    value = config[key]
                    value_type = constraints.get('type')

                    if value_type and not isinstance(value, value_type):
                        raise ValidationError(f"Config key '{key}' must be of type {value_type.__name__}")

                    if 'min_value' in constraints and isinstance(value, (int, float)):
                        if value < constraints['min_value']:
                            raise ValidationError(f"Config key '{key}' must be >= {constraints['min_value']}")

                    if 'max_value' in constraints and isinstance(value, (int, float)):
                        if value > constraints['max_value']:
                            raise ValidationError(f"Config key '{key}' must be <= {constraints['max_value']}")

                    if 'allowed_values' in constraints:
                        if value not in constraints['allowed_values']:
                            raise ValidationError(f"Config key '{key}' must be one of {constraints['allowed_values']}")

                elif constraints.get('required', False):
                    raise ValidationError(f"Required configuration key missing: {key}")

        return config


class InputSanitizer:
    """Input sanitization utilities for various data types."""

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to be safe for filesystem operations.
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Sanitized filename
        """
        if not filename:
            return "unnamed_file"

        # Remove path separators and other unsafe characters
        unsafe_chars = r'[<>:"/\\|?*\x00-\x1f]'
        filename = re.sub(unsafe_chars, '_', filename)

        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')

        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            max_name_len = 255 - len(ext)
            filename = name[:max_name_len] + ext

        # Ensure not empty
        if not filename:
            filename = "unnamed_file"

        return filename

    @staticmethod
    def sanitize_log_message(message: str) -> str:
        """Sanitize log messages to prevent log injection.
        
        Args:
            message: Log message to sanitize
            
        Returns:
            Sanitized log message
        """
        if not isinstance(message, str):
            message = str(message)

        # Replace newlines and carriage returns to prevent log injection
        message = message.replace('\n', '\\n').replace('\r', '\\r')

        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        message = ansi_escape.sub('', message)

        # Limit length
        if len(message) > 1000:
            message = message[:997] + "..."

        return message

    @staticmethod
    def hash_sensitive_data(data: str, salt: Optional[str] = None) -> str:
        """Hash sensitive data for logging/storage.
        
        Args:
            data: Sensitive data to hash
            salt: Optional salt for hashing
            
        Returns:
            Hashed data (first 8 characters of SHA-256)
        """
        if salt:
            data = f"{data}:{salt}"

        hash_obj = hashlib.sha256(data.encode('utf-8'))
        return hash_obj.hexdigest()[:8]


def validate_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Validate GitHub webhook signature.
    
    Args:
        payload: Raw webhook payload
        signature: Signature header from GitHub
        secret: Webhook secret
        
    Returns:
        True if signature is valid
    """
    if not signature or not secret:
        return False

    try:
        import hmac
        expected_signature = 'sha256=' + hmac.new(
            secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)
    except Exception as e:
        logger.error("Webhook signature validation failed", error=str(e))
        return False


def create_validation_schema() -> Dict[str, Any]:
    """Create validation schema for configuration.
    
    Returns:
        Configuration validation schema
    """
    return {
        'server': {
            'type': dict,
            'required': False
        },
        'github': {
            'type': dict,
            'required': True
        },
        'logging': {
            'type': dict,
            'required': False
        },
        'cache': {
            'type': dict,
            'required': False
        }
    }

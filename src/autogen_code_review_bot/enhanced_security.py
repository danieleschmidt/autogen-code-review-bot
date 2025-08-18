"""
Enhanced security measures for AutoGen Code Review Bot.

Implements comprehensive security controls including input validation,
rate limiting, secure token handling, and compliance features.
"""

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Union
from urllib.parse import urlparse

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt

from .exceptions import SecurityError, ValidationError
from .global_config import get_config
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SecurityContext:
    """Security context for requests and operations."""
    
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    permissions: Set[str] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = set()


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    # Allowed file extensions for code analysis
    ALLOWED_CODE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c',
        '.h', '.hpp', '.cs', '.rb', '.php', '.go', '.rs', '.kt',
        '.swift', '.scala', '.dart', '.r', '.m', '.mm'
    }
    
    # Maximum file sizes (in bytes)
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    MAX_REPO_SIZE = 100 * 1024 * 1024  # 100MB
    
    # Path traversal patterns
    DANGEROUS_PATTERNS = [
        '../', '..\\', '..%2f', '..%5c', '..%255c',
        '%2e%2e%2f', '%2e%2e%5c', '%c0%ae%c0%ae%c0%af'
    ]
    
    @classmethod
    def validate_file_path(cls, file_path: str) -> str:
        """Validate and sanitize file paths."""
        if not file_path:
            raise ValidationError("File path cannot be empty")
        
        # Check for path traversal attacks
        normalized_path = file_path.lower()
        for pattern in cls.DANGEROUS_PATTERNS:
            if pattern in normalized_path:
                raise SecurityError(f"Path traversal attempt detected: {file_path}")
        
        # Validate path length
        if len(file_path) > 4096:
            raise ValidationError("File path too long")
        
        # Check for null bytes
        if '\x00' in file_path:
            raise SecurityError("Null byte in file path")
        
        return file_path
    
    @classmethod
    def validate_repo_path(cls, repo_path: str) -> str:
        """Validate repository path."""
        validated_path = cls.validate_file_path(repo_path)
        
        # Additional repo-specific validations
        if not validated_path.startswith(('/tmp/', '/var/tmp/', '/workspace/', '/home/', '/Users/')):
            # In production, this would be more restrictive
            logger.warning(f"Repository path outside expected directories: {validated_path}")
        
        return validated_path
    
    @classmethod
    def validate_file_extension(cls, filename: str) -> bool:
        """Check if file extension is allowed for analysis."""
        from pathlib import Path
        ext = Path(filename).suffix.lower()
        return ext in cls.ALLOWED_CODE_EXTENSIONS
    
    @classmethod
    def validate_content_length(cls, content: str, max_length: int = None) -> str:
        """Validate content length."""
        max_length = max_length or cls.MAX_FILE_SIZE
        
        if len(content.encode('utf-8')) > max_length:
            raise ValidationError(f"Content exceeds maximum size: {max_length} bytes")
        
        return content
    
    @classmethod
    def sanitize_user_input(cls, user_input: str) -> str:
        """Sanitize user input to prevent injection attacks."""
        if not isinstance(user_input, str):
            raise ValidationError("Input must be a string")
        
        # Remove null bytes
        sanitized = user_input.replace('\x00', '')
        
        # Limit length
        if len(sanitized) > 10000:
            raise ValidationError("Input too long")
        
        return sanitized
    
    @classmethod
    def validate_url(cls, url: str) -> str:
        """Validate and sanitize URLs."""
        try:
            parsed = urlparse(url)
        except Exception:
            raise ValidationError("Invalid URL format")
        
        # Check scheme
        if parsed.scheme not in ('http', 'https'):
            raise SecurityError("Invalid URL scheme")
        
        # Check for private/local addresses
        hostname = parsed.hostname
        if hostname:
            if hostname in ('localhost', '127.0.0.1', '::1'):
                raise SecurityError("Local URLs not allowed")
            
            # Block private IP ranges (basic check)
            if hostname.startswith(('10.', '172.', '192.168.')):
                raise SecurityError("Private IP addresses not allowed")
        
        return url


class RateLimiter:
    """Token bucket rate limiter for API endpoints."""
    
    def __init__(self):
        self._buckets: Dict[str, Dict] = {}
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    def is_allowed(self, key: str, max_requests: int = 100, window_seconds: int = 3600) -> bool:
        """Check if request is allowed under rate limits."""
        now = time.time()
        
        # Periodic cleanup
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup_old_buckets()
            self._last_cleanup = now
        
        # Get or create bucket
        if key not in self._buckets:
            self._buckets[key] = {
                'tokens': max_requests,
                'last_refill': now,
                'max_tokens': max_requests,
                'refill_rate': max_requests / window_seconds
            }
        
        bucket = self._buckets[key]
        
        # Refill tokens
        time_passed = now - bucket['last_refill']
        tokens_to_add = time_passed * bucket['refill_rate']
        bucket['tokens'] = min(bucket['max_tokens'], bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = now
        
        # Check if request allowed
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True
        
        return False
    
    def _cleanup_old_buckets(self):
        """Remove old, unused buckets."""
        now = time.time()
        cutoff = now - 7200  # 2 hours
        
        keys_to_remove = [
            key for key, bucket in self._buckets.items()
            if bucket['last_refill'] < cutoff
        ]
        
        for key in keys_to_remove:
            del self._buckets[key]


class TokenManager:
    """Secure token generation and validation."""
    
    def __init__(self):
        self.config = get_config()
        self._algorithm = 'HS256'
    
    def generate_session_token(self, user_id: str, permissions: List[str] = None) -> str:
        """Generate a secure session token."""
        if not user_id:
            raise ValidationError("User ID required for token generation")
        
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(hours=self.config.security.token_expiry_hours)
        
        payload = {
            'user_id': user_id,
            'permissions': permissions or [],
            'iat': now.timestamp(),
            'exp': expiry.timestamp(),
            'jti': secrets.token_urlsafe(16)  # JWT ID for revocation
        }
        
        try:
            token = jwt.encode(payload, self._get_secret_key(), algorithm=self._algorithm)
            logger.info(f"Session token generated for user: {user_id}")
            return token
        except Exception as e:
            raise SecurityError(f"Token generation failed: {e}")
    
    def validate_token(self, token: str) -> SecurityContext:
        """Validate and decode a session token."""
        if not token:
            raise SecurityError("Token is required")
        
        try:
            payload = jwt.decode(token, self._get_secret_key(), algorithms=[self._algorithm])
            
            context = SecurityContext(
                user_id=payload.get('user_id'),
                session_id=payload.get('jti'),
                permissions=set(payload.get('permissions', []))
            )
            
            logger.debug(f"Token validated for user: {context.user_id}")
            return context
            
        except jwt.ExpiredSignatureError:
            raise SecurityError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise SecurityError(f"Invalid token: {e}")
    
    def _get_secret_key(self) -> str:
        """Get the JWT secret key."""
        if not self.config.security.encryption_key:
            raise SecurityError("Encryption key not configured")
        return self.config.security.encryption_key


class SecureHasher:
    """Secure hashing utilities for sensitive data."""
    
    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
        """Hash a password with salt using PBKDF2."""
        if not salt:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = kdf.derive(password.encode('utf-8'))
        
        # Return base64 encoded hash and salt
        import base64
        return (
            base64.b64encode(key).decode('utf-8'),
            base64.b64encode(salt).decode('utf-8')
        )
    
    @staticmethod
    def verify_password(password: str, stored_hash: str, salt: str) -> bool:
        """Verify a password against stored hash."""
        import base64
        
        try:
            salt_bytes = base64.b64decode(salt.encode('utf-8'))
            expected_hash = base64.b64decode(stored_hash.encode('utf-8'))
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                iterations=100000,
            )
            
            kdf.verify(password.encode('utf-8'), expected_hash)
            return True
        except Exception:
            return False
    
    @staticmethod
    def hash_data(data: str) -> str:
        """Create a secure hash of data."""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    @staticmethod
    def create_hmac(data: str, key: str) -> str:
        """Create HMAC for data integrity."""
        return hmac.new(
            key.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    @staticmethod
    def verify_hmac(data: str, signature: str, key: str) -> bool:
        """Verify HMAC signature."""
        expected = SecureHasher.create_hmac(data, key)
        return hmac.compare_digest(expected, signature)


class ComplianceManager:
    """Compliance and audit trail management."""
    
    def __init__(self):
        self.config = get_config()
        self._audit_log: List[Dict] = []
    
    def log_security_event(self, event_type: str, details: Dict, context: SecurityContext = None):
        """Log a security event for audit trail."""
        event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': event_type,
            'details': details,
            'user_id': context.user_id if context else None,
            'session_id': context.session_id if context else None,
            'ip_address': context.ip_address if context else None
        }
        
        self._audit_log.append(event)
        logger.info(f"Security event logged: {event_type}")
        
        # In production, this would be sent to a secure logging service
        
    def check_compliance_requirements(self, region: str) -> Dict[str, bool]:
        """Check compliance requirements for a region."""
        region_config = self.config.get_region_config(region)
        requirements = region_config.compliance_requirements
        
        # Simulate compliance checks
        compliance_status = {}
        for requirement in requirements:
            compliance_status[requirement] = self._check_requirement(requirement)
        
        return compliance_status
    
    def _check_requirement(self, requirement: str) -> bool:
        """Check a specific compliance requirement."""
        # This would implement actual compliance checks
        checks = {
            'GDPR': self._check_gdpr_compliance(),
            'SOC2': self._check_soc2_compliance(),
            'HIPAA': self._check_hipaa_compliance(),
            'PDPA': self._check_pdpa_compliance(),
            'CCPA': self._check_ccpa_compliance()
        }
        
        return checks.get(requirement, False)
    
    def _check_gdpr_compliance(self) -> bool:
        """Check GDPR compliance."""
        # Implement GDPR compliance checks
        return True  # Placeholder
    
    def _check_soc2_compliance(self) -> bool:
        """Check SOC2 compliance."""
        # Implement SOC2 compliance checks
        return True  # Placeholder
    
    def _check_hipaa_compliance(self) -> bool:
        """Check HIPAA compliance."""
        # Implement HIPAA compliance checks
        return True  # Placeholder
    
    def _check_pdpa_compliance(self) -> bool:
        """Check PDPA compliance."""
        # Implement PDPA compliance checks
        return True  # Placeholder
    
    def _check_ccpa_compliance(self) -> bool:
        """Check CCPA compliance."""
        # Implement CCPA compliance checks
        return True  # Placeholder


# Global instances
rate_limiter = RateLimiter()
token_manager = TokenManager()
compliance_manager = ComplianceManager()


def require_authentication(func):
    """Decorator to require authentication for functions."""
    def wrapper(*args, **kwargs):
        # This would extract token from request headers in a real API
        # For now, we'll check if security context is provided
        if 'security_context' not in kwargs:
            raise SecurityError("Authentication required")
        return func(*args, **kwargs)
    return wrapper


def require_permission(permission: str):
    """Decorator to require specific permission."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            context = kwargs.get('security_context')
            if not context or permission not in context.permissions:
                raise SecurityError(f"Permission required: {permission}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_input(validator_func):
    """Decorator to validate function inputs."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Apply validation to arguments
            validated_args = []
            for arg in args:
                if isinstance(arg, str):
                    validated_args.append(validator_func(arg))
                else:
                    validated_args.append(arg)
            
            return func(*validated_args, **kwargs)
        return wrapper
    return decorator
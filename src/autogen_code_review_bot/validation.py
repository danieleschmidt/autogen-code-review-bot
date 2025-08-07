#!/usr/bin/env python3
"""
Enterprise Input Validation and Data Sanitization for AutoGen Code Review Bot.

Implements comprehensive validation, sanitization, and security checks
for all user inputs and external data.
"""

import re
import os
import json
import yaml
import html
import urllib.parse
from typing import Dict, List, Optional, Any, Union, Type, Callable
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parseaddr
from ipaddress import ip_address, AddressValueError

from .logging_utils import get_logger
from .exceptions import ValidationError, SecurityError

logger = get_logger(__name__)


@dataclass
class ValidationRule:
    """Represents a validation rule."""
    name: str
    validator: Callable[[Any], bool]
    message: str
    severity: str = "error"  # error, warning, info


@dataclass
class SanitizationRule:
    """Represents a data sanitization rule."""
    name: str
    sanitizer: Callable[[Any], Any]
    description: str


class InputValidator:
    """Comprehensive input validation with security focus."""
    
    # Security patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bunion\b.*\bselect\b)",
        r"(\bselect\b.*\bfrom\b)",
        r"(\binsert\b.*\binto\b)",
        r"(\bdelete\b.*\bfrom\b)",
        r"(\bupdate\b.*\bset\b)",
        r"(\bdrop\b.*\btable\b)",
        r"(;.*--)",
        r"('.*or.*'.*='.*')",
        r"(\bexec\b.*\()",
        r"(\bsp_executesql\b)",
    ]
    
    XSS_PATTERNS = [
        r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
        r"javascript:",
        r"vbscript:",
        r"onload\s*=",
        r"onerror\s*=",
        r"onclick\s*=",
        r"onmouseover\s*=",
        r"<iframe\b",
        r"<object\b",
        r"<embed\b",
        r"<form\b",
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r"(;|\||&)\s*(rm|del|format|dd)\s",
        r"(;|\||&)\s*(cat|type|more)\s",
        r"(;|\||&)\s*(wget|curl|nc|netcat)\s",
        r"\$\([^)]*\)",
        r"`[^`]*`",
        r">\s*/dev/",
        r"<\s*/dev/",
        r"\|\s*sh",
        r"\|\s*bash",
        r"&&\s*(sudo|su)\s",
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\.[\\/]",
        r"[\\/]\.\.",
        r"~[\\/]",
        r"\%2e\%2e[\\/]",
        r"[\\/]\%2e\%2e",
        r"\.\.[\/\\]",
        r"[\/\\]\.\.",
    ]
    
    def __init__(self):
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self.sanitization_rules: Dict[str, List[SanitizationRule]] = {}
        self.logger = get_logger(__name__ + ".InputValidator")
        
        # Register default rules
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default validation and sanitization rules."""
        # String validation rules
        self.add_validation_rule("string", ValidationRule(
            name="max_length",
            validator=lambda x: len(str(x)) <= 10000,  # 10KB limit
            message="String exceeds maximum length of 10,000 characters"
        ))
        
        self.add_validation_rule("string", ValidationRule(
            name="sql_injection",
            validator=self._check_sql_injection,
            message="Potential SQL injection detected",
            severity="error"
        ))
        
        self.add_validation_rule("string", ValidationRule(
            name="xss_check",
            validator=self._check_xss,
            message="Potential XSS payload detected",
            severity="error"
        ))
        
        self.add_validation_rule("string", ValidationRule(
            name="command_injection",
            validator=self._check_command_injection,
            message="Potential command injection detected",
            severity="error"
        ))
        
        # Path validation rules
        self.add_validation_rule("path", ValidationRule(
            name="path_traversal",
            validator=self._check_path_traversal,
            message="Path traversal attempt detected",
            severity="error"
        ))
        
        self.add_validation_rule("path", ValidationRule(
            name="absolute_path",
            validator=lambda x: os.path.isabs(str(x)),
            message="Path must be absolute"
        ))
        
        self.add_validation_rule("path", ValidationRule(
            name="exists",
            validator=lambda x: Path(str(x)).exists(),
            message="Path does not exist"
        ))
        
        # Email validation rules
        self.add_validation_rule("email", ValidationRule(
            name="format",
            validator=self._check_email_format,
            message="Invalid email format"
        ))
        
        # URL validation rules
        self.add_validation_rule("url", ValidationRule(
            name="format",
            validator=self._check_url_format,
            message="Invalid URL format"
        ))
        
        self.add_validation_rule("url", ValidationRule(
            name="scheme",
            validator=lambda x: urllib.parse.urlparse(str(x)).scheme in ['http', 'https'],
            message="Only HTTP and HTTPS URLs are allowed"
        ))
        
        # JSON validation rules
        self.add_validation_rule("json", ValidationRule(
            name="valid_json",
            validator=self._check_valid_json,
            message="Invalid JSON format"
        ))
        
        self.add_validation_rule("json", ValidationRule(
            name="max_depth",
            validator=lambda x: self._check_json_depth(x, max_depth=10),
            message="JSON nesting depth exceeds limit"
        ))
        
        # Sanitization rules
        self.add_sanitization_rule("string", SanitizationRule(
            name="html_escape",
            sanitizer=html.escape,
            description="Escape HTML special characters"
        ))
        
        self.add_sanitization_rule("string", SanitizationRule(
            name="trim_whitespace",
            sanitizer=lambda x: str(x).strip(),
            description="Remove leading and trailing whitespace"
        ))
        
        self.add_sanitization_rule("path", SanitizationRule(
            name="normalize_path",
            sanitizer=lambda x: str(Path(str(x)).resolve()),
            description="Normalize and resolve path"
        ))
        
        self.add_sanitization_rule("filename", SanitizationRule(
            name="safe_filename",
            sanitizer=self._sanitize_filename,
            description="Remove unsafe characters from filename"
        ))
    
    def add_validation_rule(self, data_type: str, rule: ValidationRule):
        """Add a validation rule for a data type."""
        if data_type not in self.validation_rules:
            self.validation_rules[data_type] = []
        self.validation_rules[data_type].append(rule)
        
        self.logger.debug(f"Added validation rule {rule.name} for {data_type}")
    
    def add_sanitization_rule(self, data_type: str, rule: SanitizationRule):
        """Add a sanitization rule for a data type."""
        if data_type not in self.sanitization_rules:
            self.sanitization_rules[data_type] = []
        self.sanitization_rules[data_type].append(rule)
        
        self.logger.debug(f"Added sanitization rule {rule.name} for {data_type}")
    
    def validate(self, value: Any, data_type: str, 
                required: bool = True, custom_rules: List[ValidationRule] = None) -> Dict[str, Any]:
        """Validate input value against type-specific rules."""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check required
        if required and (value is None or (isinstance(value, str) and not value.strip())):
            result['valid'] = False
            result['errors'].append(f"{data_type} is required")
            return result
        
        # Skip validation if value is None and not required
        if value is None and not required:
            return result
        
        # Get rules for data type
        rules = self.validation_rules.get(data_type, [])
        if custom_rules:
            rules.extend(custom_rules)
        
        # Apply validation rules
        for rule in rules:
            try:
                if not rule.validator(value):
                    message = f"{rule.message} (value: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''})"
                    
                    if rule.severity == "error":
                        result['valid'] = False
                        result['errors'].append(message)
                    elif rule.severity == "warning":
                        result['warnings'].append(message)
                    else:
                        result['info'].append(message)
                        
            except Exception as e:
                result['valid'] = False
                result['errors'].append(f"Validation rule {rule.name} failed: {str(e)}")
                self.logger.error(f"Validation rule {rule.name} error: {e}")
        
        return result
    
    def sanitize(self, value: Any, data_type: str, 
                custom_rules: List[SanitizationRule] = None) -> Any:
        """Sanitize input value using type-specific rules."""
        if value is None:
            return None
        
        sanitized_value = value
        
        # Get sanitization rules for data type
        rules = self.sanitization_rules.get(data_type, [])
        if custom_rules:
            rules.extend(custom_rules)
        
        # Apply sanitization rules
        for rule in rules:
            try:
                sanitized_value = rule.sanitizer(sanitized_value)
                self.logger.debug(f"Applied sanitization rule {rule.name}")
            except Exception as e:
                self.logger.error(f"Sanitization rule {rule.name} failed: {e}")
                # Continue with previous value on sanitization error
        
        return sanitized_value
    
    def validate_and_sanitize(self, value: Any, data_type: str, 
                            required: bool = True,
                            custom_validation_rules: List[ValidationRule] = None,
                            custom_sanitization_rules: List[SanitizationRule] = None) -> Dict[str, Any]:
        """Validate and sanitize input value."""
        # First sanitize
        sanitized_value = self.sanitize(value, data_type, custom_sanitization_rules)
        
        # Then validate sanitized value
        validation_result = self.validate(
            sanitized_value, data_type, required, custom_validation_rules
        )
        
        return {
            'original_value': value,
            'sanitized_value': sanitized_value,
            'validation': validation_result
        }
    
    def _check_sql_injection(self, value: str) -> bool:
        """Check for SQL injection patterns."""
        value_lower = str(value).lower()
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                self.logger.warning(f"SQL injection pattern detected: {pattern}")
                return False
        return True
    
    def _check_xss(self, value: str) -> bool:
        """Check for XSS patterns."""
        value_str = str(value)
        for pattern in self.XSS_PATTERNS:
            if re.search(pattern, value_str, re.IGNORECASE):
                self.logger.warning(f"XSS pattern detected: {pattern}")
                return False
        return True
    
    def _check_command_injection(self, value: str) -> bool:
        """Check for command injection patterns."""
        value_str = str(value)
        for pattern in self.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value_str, re.IGNORECASE):
                self.logger.warning(f"Command injection pattern detected: {pattern}")
                return False
        return True
    
    def _check_path_traversal(self, value: str) -> bool:
        """Check for path traversal patterns."""
        value_str = str(value)
        for pattern in self.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, value_str, re.IGNORECASE):
                self.logger.warning(f"Path traversal pattern detected: {pattern}")
                return False
        return True
    
    def _check_email_format(self, value: str) -> bool:
        """Check email format using basic validation."""
        try:
            name, addr = parseaddr(str(value))
            return '@' in addr and '.' in addr.split('@')[1]
        except Exception:
            return False
    
    def _check_url_format(self, value: str) -> bool:
        """Check URL format."""
        try:
            result = urllib.parse.urlparse(str(value))
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _check_valid_json(self, value: str) -> bool:
        """Check if string is valid JSON."""
        try:
            json.loads(str(value))
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    def _check_json_depth(self, value: str, max_depth: int = 10) -> bool:
        """Check JSON nesting depth."""
        try:
            data = json.loads(str(value))
            return self._get_json_depth(data) <= max_depth
        except Exception:
            return False
    
    def _get_json_depth(self, obj: Any, depth: int = 0) -> int:
        """Calculate JSON object depth recursively."""
        if not isinstance(obj, (dict, list)):
            return depth
        
        if isinstance(obj, dict):
            return max(self._get_json_depth(value, depth + 1) for value in obj.values()) if obj else depth
        else:  # list
            return max(self._get_json_depth(item, depth + 1) for item in obj) if obj else depth
    
    def _sanitize_filename(self, value: str) -> str:
        """Sanitize filename by removing unsafe characters."""
        # Remove path separators and other unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        filename = str(value)
        
        for char in unsafe_chars:
            filename = filename.replace(char, '_')
        
        # Remove control characters
        filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext
        
        return filename.strip()


class SchemaValidator:
    """Schema-based validation for complex data structures."""
    
    def __init__(self, validator: InputValidator):
        self.validator = validator
        self.logger = get_logger(__name__ + ".SchemaValidator")
    
    def validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against a schema definition."""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'field_results': {}
        }
        
        try:
            # Check required fields
            required_fields = schema.get('required', [])
            for field in required_fields:
                if field not in data:
                    result['valid'] = False
                    result['errors'].append(f"Required field '{field}' is missing")
            
            # Validate each field
            fields = schema.get('fields', {})
            for field_name, field_schema in fields.items():
                if field_name in data:
                    field_result = self._validate_field(data[field_name], field_schema, field_name)
                    result['field_results'][field_name] = field_result
                    
                    if not field_result['validation']['valid']:
                        result['valid'] = False
                        result['errors'].extend([
                            f"Field '{field_name}': {error}" 
                            for error in field_result['validation']['errors']
                        ])
                        result['warnings'].extend([
                            f"Field '{field_name}': {warning}" 
                            for warning in field_result['validation']['warnings']
                        ])
            
            # Check for unexpected fields if strict mode
            if schema.get('strict', False):
                unexpected_fields = set(data.keys()) - set(fields.keys())
                if unexpected_fields:
                    result['valid'] = False
                    result['errors'].append(f"Unexpected fields: {', '.join(unexpected_fields)}")
            
            self.logger.debug(f"Schema validation completed", extra={
                'valid': result['valid'],
                'error_count': len(result['errors']),
                'warning_count': len(result['warnings'])
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Schema validation error: {e}")
            return {
                'valid': False,
                'errors': [f"Schema validation failed: {str(e)}"],
                'warnings': [],
                'field_results': {}
            }
    
    def _validate_field(self, value: Any, field_schema: Dict[str, Any], field_name: str) -> Dict[str, Any]:
        """Validate individual field against its schema."""
        data_type = field_schema.get('type', 'string')
        required = field_schema.get('required', False)
        
        # Custom validation rules from schema
        custom_rules = []
        if 'min_length' in field_schema:
            min_len = field_schema['min_length']
            custom_rules.append(ValidationRule(
                name="min_length",
                validator=lambda x: len(str(x)) >= min_len,
                message=f"Minimum length is {min_len}"
            ))
        
        if 'max_length' in field_schema:
            max_len = field_schema['max_length']
            custom_rules.append(ValidationRule(
                name="max_length",
                validator=lambda x: len(str(x)) <= max_len,
                message=f"Maximum length is {max_len}"
            ))
        
        if 'pattern' in field_schema:
            pattern = field_schema['pattern']
            custom_rules.append(ValidationRule(
                name="pattern",
                validator=lambda x: re.match(pattern, str(x)) is not None,
                message=f"Value must match pattern: {pattern}"
            ))
        
        if 'enum' in field_schema:
            enum_values = field_schema['enum']
            custom_rules.append(ValidationRule(
                name="enum",
                validator=lambda x: x in enum_values,
                message=f"Value must be one of: {', '.join(map(str, enum_values))}"
            ))
        
        # Custom sanitization rules
        custom_sanitization_rules = []
        if field_schema.get('trim', False):
            custom_sanitization_rules.append(SanitizationRule(
                name="trim",
                sanitizer=lambda x: str(x).strip(),
                description="Trim whitespace"
            ))
        
        if field_schema.get('lowercase', False):
            custom_sanitization_rules.append(SanitizationRule(
                name="lowercase",
                sanitizer=lambda x: str(x).lower(),
                description="Convert to lowercase"
            ))
        
        return self.validator.validate_and_sanitize(
            value, data_type, required, custom_rules, custom_sanitization_rules
        )


class SecurityValidator:
    """Additional security-focused validation."""
    
    def __init__(self):
        self.logger = get_logger(__name__ + ".SecurityValidator")
    
    def validate_ip_address(self, ip_str: str, allow_private: bool = False) -> Dict[str, Any]:
        """Validate IP address with security checks."""
        try:
            ip = ip_address(ip_str)
            
            result = {
                'valid': True,
                'ip_object': ip,
                'version': ip.version,
                'is_private': ip.is_private,
                'is_loopback': ip.is_loopback,
                'is_multicast': ip.is_multicast,
                'errors': [],
                'warnings': []
            }
            
            # Security checks
            if not allow_private and ip.is_private:
                result['valid'] = False
                result['errors'].append("Private IP addresses not allowed")
            
            if ip.is_loopback:
                result['warnings'].append("Loopback address detected")
            
            if ip.is_multicast:
                result['warnings'].append("Multicast address detected")
            
            # Check against known malicious IP ranges (simplified example)
            if self._is_suspicious_ip(ip):
                result['valid'] = False
                result['errors'].append("IP address flagged as suspicious")
            
            return result
            
        except AddressValueError as e:
            return {
                'valid': False,
                'errors': [f"Invalid IP address: {str(e)}"],
                'warnings': []
            }
    
    def validate_user_agent(self, user_agent: str) -> Dict[str, Any]:
        """Validate User-Agent string for suspicious patterns."""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'analysis': {
                'length': len(user_agent),
                'suspicious_patterns': [],
                'bot_indicators': []
            }
        }
        
        # Check length
        if len(user_agent) > 500:
            result['warnings'].append("Unusually long User-Agent string")
        elif len(user_agent) < 10:
            result['warnings'].append("Unusually short User-Agent string")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'curl|wget|python-requests',  # Common scripting tools
            r'bot|crawl|spider|scrape',     # Bot indicators
            r'<script|javascript:',         # XSS attempts
            r'\x00|\x01|\x02|\x03',        # Control characters
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, user_agent, re.IGNORECASE):
                result['analysis']['suspicious_patterns'].append(pattern)
                result['warnings'].append(f"Suspicious pattern detected: {pattern}")
        
        return result
    
    def _is_suspicious_ip(self, ip) -> bool:
        """Check if IP is in suspicious ranges (simplified)."""
        # In production, this would check against threat intelligence feeds
        suspicious_ranges = [
            # Example suspicious ranges - replace with real data
            '192.0.2.0/24',  # TEST-NET-1
            '198.51.100.0/24',  # TEST-NET-2
        ]
        
        # This is a simplified check - production would use proper IP range checking
        return False


# Global validator instance
_input_validator: Optional[InputValidator] = None
_schema_validator: Optional[SchemaValidator] = None
_security_validator: Optional[SecurityValidator] = None


def get_input_validator() -> InputValidator:
    """Get global input validator instance."""
    global _input_validator
    if _input_validator is None:
        _input_validator = InputValidator()
    return _input_validator


def get_schema_validator() -> SchemaValidator:
    """Get global schema validator instance."""
    global _schema_validator
    if _schema_validator is None:
        _schema_validator = SchemaValidator(get_input_validator())
    return _schema_validator


def get_security_validator() -> SecurityValidator:
    """Get global security validator instance."""
    global _security_validator
    if _security_validator is None:
        _security_validator = SecurityValidator()
    return _security_validator


# Convenience functions
def validate_input(value: Any, data_type: str, required: bool = True) -> Dict[str, Any]:
    """Convenience function for input validation."""
    return get_input_validator().validate(value, data_type, required)


def sanitize_input(value: Any, data_type: str) -> Any:
    """Convenience function for input sanitization."""
    return get_input_validator().sanitize(value, data_type)


def validate_and_sanitize_input(value: Any, data_type: str, required: bool = True) -> Dict[str, Any]:
    """Convenience function for validation and sanitization."""
    return get_input_validator().validate_and_sanitize(value, data_type, required)


def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for schema validation."""
    return get_schema_validator().validate_schema(data, schema)
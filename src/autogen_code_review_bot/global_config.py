"""
Global configuration management with multi-region and I18n support.

Implements enterprise-grade configuration management with security,
validation, and global deployment readiness.
"""

import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
from cryptography.fernet import Fernet

from .exceptions import ConfigurationError
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RegionConfig:
    """Configuration for a specific deployment region."""
    
    name: str
    endpoint: str
    timezone: str
    compliance_requirements: List[str]
    data_residency: bool = True
    

@dataclass
class SecurityConfig:
    """Security configuration settings."""
    
    encryption_key: Optional[str] = None
    token_expiry_hours: int = 24
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    allowed_origins: List[str] = None
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = []


@dataclass
class I18nConfig:
    """Internationalization configuration."""
    
    default_locale: str = "en"
    supported_locales: List[str] = None
    translations_path: str = "translations"
    
    def __post_init__(self):
        if self.supported_locales is None:
            self.supported_locales = ["en", "es", "fr", "de", "ja", "zh"]


@dataclass
class GlobalConfig:
    """Comprehensive global configuration management."""
    
    # Core settings
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    
    # Regional deployment
    regions: Dict[str, RegionConfig] = None
    primary_region: str = "us-east-1"
    
    # Security
    security: SecurityConfig = None
    
    # Internationalization
    i18n: I18nConfig = None
    
    # Feature flags
    features: Dict[str, bool] = None
    
    # Performance settings
    max_workers: int = 4
    cache_ttl_seconds: int = 3600
    
    def __post_init__(self):
        if self.regions is None:
            self.regions = self._default_regions()
        if self.security is None:
            self.security = SecurityConfig()
        if self.i18n is None:
            self.i18n = I18nConfig()
        if self.features is None:
            self.features = self._default_features()
    
    def _default_regions(self) -> Dict[str, RegionConfig]:
        """Default regional configurations."""
        return {
            "us-east-1": RegionConfig(
                name="US East",
                endpoint="https://api-us-east.example.com",
                timezone="America/New_York",
                compliance_requirements=["SOC2", "HIPAA"]
            ),
            "eu-west-1": RegionConfig(
                name="EU West",
                endpoint="https://api-eu-west.example.com",
                timezone="Europe/London",
                compliance_requirements=["GDPR", "SOC2"]
            ),
            "ap-southeast-1": RegionConfig(
                name="Asia Pacific",
                endpoint="https://api-ap-southeast.example.com",
                timezone="Asia/Singapore",
                compliance_requirements=["PDPA", "SOC2"]
            )
        }
    
    def _default_features(self) -> Dict[str, bool]:
        """Default feature flags."""
        return {
            "quantum_optimization": True,
            "advanced_caching": True,
            "real_time_collaboration": True,
            "enterprise_monitoring": True,
            "auto_scaling": True,
            "security_scanning": True,
            "performance_analytics": True
        }
    
    def get_region_config(self, region: Optional[str] = None) -> RegionConfig:
        """Get configuration for a specific region."""
        region = region or self.primary_region
        if region not in self.regions:
            raise ConfigurationError(f"Unknown region: {region}")
        return self.regions[region]
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return self.features.get(feature, False)
    
    def validate(self) -> None:
        """Validate configuration settings."""
        errors = []
        
        # Validate environment
        if self.environment not in ["development", "staging", "production"]:
            errors.append(f"Invalid environment: {self.environment}")
        
        # Validate primary region
        if self.primary_region not in self.regions:
            errors.append(f"Primary region not found: {self.primary_region}")
        
        # Validate security settings
        if self.environment == "production":
            if not self.security.encryption_key:
                errors.append("Encryption key required for production")
            if self.debug:
                errors.append("Debug mode not allowed in production")
        
        # Validate I18n settings
        if self.i18n.default_locale not in self.i18n.supported_locales:
            errors.append("Default locale not in supported locales")
        
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")


class ConfigManager:
    """Global configuration manager with encryption and validation."""
    
    def __init__(self):
        self._config: Optional[GlobalConfig] = None
        self._cipher: Optional[Fernet] = None
        
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> GlobalConfig:
        """Load configuration from file or environment."""
        if self._config is not None:
            return self._config
        
        config_data = {}
        
        # Load from file if provided
        if config_path:
            config_data = self._load_from_file(config_path)
        
        # Override with environment variables
        config_data.update(self._load_from_env())
        
        # Create configuration object
        self._config = self._create_config(config_data)
        self._config.validate()
        
        logger.info(f"Configuration loaded for environment: {self._config.environment}")
        return self._config
    
    def _load_from_file(self, config_path: Union[str, Path]) -> Dict:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file: {e}")
    
    def _load_from_env(self) -> Dict:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Core settings
        if env_val := os.getenv("APP_ENVIRONMENT"):
            env_config["environment"] = env_val
        if env_val := os.getenv("APP_DEBUG"):
            env_config["debug"] = env_val.lower() in ("true", "1", "yes")
        if env_val := os.getenv("APP_LOG_LEVEL"):
            env_config["log_level"] = env_val
        
        # Security settings
        security_config = {}
        if env_val := os.getenv("ENCRYPTION_KEY"):
            security_config["encryption_key"] = env_val
        if env_val := os.getenv("TOKEN_EXPIRY_HOURS"):
            try:
                security_config["token_expiry_hours"] = int(env_val)
            except ValueError:
                logger.warning(f"Invalid TOKEN_EXPIRY_HOURS: {env_val}")
        
        if security_config:
            env_config["security"] = security_config
        
        # I18n settings
        i18n_config = {}
        if env_val := os.getenv("DEFAULT_LOCALE"):
            i18n_config["default_locale"] = env_val
        if env_val := os.getenv("SUPPORTED_LOCALES"):
            i18n_config["supported_locales"] = env_val.split(",")
        
        if i18n_config:
            env_config["i18n"] = i18n_config
        
        return env_config
    
    def _create_config(self, config_data: Dict) -> GlobalConfig:
        """Create GlobalConfig object from dictionary data."""
        # Handle nested configurations
        if "security" in config_data:
            config_data["security"] = SecurityConfig(**config_data["security"])
        
        if "i18n" in config_data:
            config_data["i18n"] = I18nConfig(**config_data["i18n"])
        
        if "regions" in config_data:
            regions = {}
            for name, region_data in config_data["regions"].items():
                regions[name] = RegionConfig(name=name, **region_data)
            config_data["regions"] = regions
        
        return GlobalConfig(**config_data)
    
    def get_config(self) -> GlobalConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            self.load_config()
        return self._config
    
    def get_cipher(self) -> Fernet:
        """Get encryption cipher for secure data."""
        if self._cipher is None:
            config = self.get_config()
            if not config.security.encryption_key:
                raise ConfigurationError("Encryption key not configured")
            
            key = config.security.encryption_key.encode()
            if len(key) != 44:  # Fernet key length
                # Generate key from provided string
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
                import base64
                
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b'autogen-salt',
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(key))
            
            self._cipher = Fernet(key)
        
        return self._cipher
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a sensitive value."""
        cipher = self.get_cipher()
        return cipher.encrypt(value.encode()).decode()
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a sensitive value."""
        cipher = self.get_cipher()
        return cipher.decrypt(encrypted_value.encode()).decode()


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> GlobalConfig:
    """Get the global configuration instance."""
    return config_manager.get_config()


def load_config(config_path: Optional[Union[str, Path]] = None) -> GlobalConfig:
    """Load configuration from file or environment."""
    return config_manager.load_config(config_path)
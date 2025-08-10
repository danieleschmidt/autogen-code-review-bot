#!/usr/bin/env python3
"""
Enterprise API Gateway for AutoGen Code Review Bot.

Provides REST API endpoints for enterprise integration, authentication,
rate limiting, and comprehensive monitoring.
"""

import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

import jwt
from flask import Flask, g, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import RequestEntityTooLarge

from .logging_utils import ContextLogger, set_request_id
from .logging_utils import get_request_logger as get_logger
from .metrics import get_metrics_registry, record_operation_metrics
from .pr_analysis import analyze_pr, format_analysis_with_agents

logger = get_logger(__name__)
metrics = get_metrics_registry()


@dataclass
class APIUser:
    """Enterprise API user with permissions and quotas."""
    user_id: str
    email: str
    organization: str
    permissions: List[str]
    daily_quota: int
    monthly_quota: int
    created_at: datetime
    last_active: Optional[datetime] = None
    is_active: bool = True

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions or 'admin' in self.permissions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'user_id': self.user_id,
            'email': self.email,
            'organization': self.organization,
            'permissions': self.permissions,
            'daily_quota': self.daily_quota,
            'monthly_quota': self.monthly_quota,
            'created_at': self.created_at.isoformat(),
            'last_active': self.last_active.isoformat() if self.last_active else None,
            'is_active': self.is_active
        }


class AuthenticationManager:
    """Enterprise authentication and authorization manager."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = 'HS256'
        self.token_expiry = timedelta(hours=24)
        self.logger = get_logger(__name__ + ".AuthenticationManager")

        # Mock user database (in production, use proper database)
        self.users = {
            'enterprise_user_1': APIUser(
                user_id='enterprise_user_1',
                email='admin@company.com',
                organization='Enterprise Corp',
                permissions=['read', 'write', 'analyze', 'admin'],
                daily_quota=1000,
                monthly_quota=30000,
                created_at=datetime.now(timezone.utc)
            ),
            'standard_user_1': APIUser(
                user_id='standard_user_1',
                email='dev@startup.com',
                organization='Startup Inc',
                permissions=['read', 'analyze'],
                daily_quota=100,
                monthly_quota=3000,
                created_at=datetime.now(timezone.utc)
            )
        }

    def authenticate_request(self, token: str) -> Optional[APIUser]:
        """Authenticate API request using JWT token."""
        try:
            # Decode JWT token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload.get('user_id')

            if not user_id:
                return None

            # Get user from database
            user = self.users.get(user_id)
            if not user or not user.is_active:
                return None

            # Check token expiration
            exp_timestamp = payload.get('exp')
            if exp_timestamp and datetime.fromtimestamp(exp_timestamp, timezone.utc) < datetime.now(timezone.utc):
                return None

            # Update last active timestamp
            user.last_active = datetime.now(timezone.utc)

            self.logger.info("User authenticated successfully", extra={
                'user_id': user_id,
                'organization': user.organization
            })

            return user

        except jwt.InvalidTokenError as e:
            self.logger.warning(f"JWT token validation failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return None

    def generate_token(self, user_id: str) -> Optional[str]:
        """Generate JWT token for authenticated user."""
        try:
            user = self.users.get(user_id)
            if not user or not user.is_active:
                return None

            payload = {
                'user_id': user_id,
                'email': user.email,
                'organization': user.organization,
                'permissions': user.permissions,
                'iat': datetime.now(timezone.utc),
                'exp': datetime.now(timezone.utc) + self.token_expiry
            }

            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

            self.logger.info("JWT token generated", extra={
                'user_id': user_id,
                'expires_at': payload['exp'].isoformat()
            })

            return token

        except Exception as e:
            self.logger.error(f"Token generation failed: {e}")
            return None


class RateLimitManager:
    """Enterprise rate limiting with per-user quotas."""

    def __init__(self):
        self.usage_tracking = {}  # In production, use Redis
        self.logger = get_logger(__name__ + ".RateLimitManager")

    def check_rate_limit(self, user: APIUser, operation: str = 'default') -> bool:
        """Check if user is within rate limits."""
        current_time = datetime.now(timezone.utc)
        user_key = f"{user.user_id}:{current_time.date()}"

        # Get current usage
        current_usage = self.usage_tracking.get(user_key, 0)

        # Check against daily quota
        if current_usage >= user.daily_quota:
            self.logger.warning("Daily rate limit exceeded", extra={
                'user_id': user.user_id,
                'current_usage': current_usage,
                'daily_quota': user.daily_quota
            })
            return False

        return True

    def record_usage(self, user: APIUser, operation: str = 'default', cost: int = 1):
        """Record API usage for rate limiting."""
        current_time = datetime.now(timezone.utc)
        user_key = f"{user.user_id}:{current_time.date()}"

        # Update usage counter
        self.usage_tracking[user_key] = self.usage_tracking.get(user_key, 0) + cost

        self.logger.info("API usage recorded", extra={
            'user_id': user.user_id,
            'operation': operation,
            'cost': cost,
            'total_usage': self.usage_tracking[user_key]
        })


# Initialize Flask app and components
app = Flask(__name__)
CORS(app, origins=["https://*.company.com", "http://localhost:*"])

# Configuration
app.config['SECRET_KEY'] = os.getenv('API_SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max request size

# Initialize managers
auth_manager = AuthenticationManager(app.config['SECRET_KEY'])
rate_manager = RateLimitManager()

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["1000 per hour", "100 per minute"]
)


def require_auth(required_permission: str = None):
    """Authentication decorator for API endpoints."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Generate request ID
            request_id = set_request_id()
            req_logger = ContextLogger(logger, request_id=request_id)

            # Extract authentication token
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                req_logger.warning("Missing or invalid authorization header")
                return jsonify({
                    'error': 'Authentication required',
                    'message': 'Bearer token required in Authorization header'
                }), 401

            token = auth_header.split(' ')[1]

            # Authenticate user
            user = auth_manager.authenticate_request(token)
            if not user:
                req_logger.warning("Authentication failed")
                return jsonify({
                    'error': 'Authentication failed',
                    'message': 'Invalid or expired token'
                }), 401

            # Check permissions
            if required_permission and not user.has_permission(required_permission):
                req_logger.warning("Insufficient permissions", extra={
                    'user_id': user.user_id,
                    'required_permission': required_permission,
                    'user_permissions': user.permissions
                })
                return jsonify({
                    'error': 'Insufficient permissions',
                    'message': f'Permission {required_permission} required'
                }), 403

            # Check rate limits
            if not rate_manager.check_rate_limit(user):
                req_logger.warning("Rate limit exceeded", extra={'user_id': user.user_id})
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'message': 'Daily quota exceeded'
                }), 429

            # Store user in Flask g object
            g.current_user = user
            g.request_logger = req_logger

            return f(*args, **kwargs)

        return decorated_function
    return decorator


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for load balancers."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': '1.0.0',
        'service': 'autogen-code-review-api'
    })


@app.route('/api/v1/auth/token', methods=['POST'])
@limiter.limit("10 per minute")
def generate_auth_token():
    """Generate authentication token for API access."""
    try:
        data = request.get_json()
        if not data or 'user_id' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'user_id required'
            }), 400

        user_id = data['user_id']
        token = auth_manager.generate_token(user_id)

        if not token:
            return jsonify({
                'error': 'Authentication failed',
                'message': 'Invalid user credentials'
            }), 401

        user = auth_manager.users.get(user_id)

        return jsonify({
            'token': token,
            'expires_at': (datetime.now(timezone.utc) + auth_manager.token_expiry).isoformat(),
            'user': user.to_dict() if user else None
        })

    except Exception as e:
        logger.error(f"Token generation failed: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'Token generation failed'
        }), 500


@app.route('/api/v1/analyze/repository', methods=['POST'])
@require_auth('analyze')
def analyze_repository():
    """Analyze a repository for code quality, security, and performance."""
    req_logger = g.request_logger
    user = g.current_user

    try:
        # Record API usage
        rate_manager.record_usage(user, 'repository_analysis', cost=5)

        # Parse request
        data = request.get_json()
        if not data or 'repository_path' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'repository_path required'
            }), 400

        repo_path = data['repository_path']
        config_path = data.get('config_path')
        use_cache = data.get('use_cache', True)
        use_parallel = data.get('use_parallel', True)
        agent_config = data.get('agent_config_path')

        req_logger.info("Starting repository analysis", extra={
            'repo_path': repo_path,
            'user_id': user.user_id,
            'organization': user.organization
        })

        # Validate repository path
        if not Path(repo_path).exists():
            return jsonify({
                'error': 'Repository not found',
                'message': f'Repository path does not exist: {repo_path}'
            }), 404

        # Perform analysis
        with record_operation_metrics("api_repository_analysis", metrics):
            analysis_result = analyze_pr(
                repo_path=repo_path,
                config_path=config_path,
                use_cache=use_cache,
                use_parallel=use_parallel
            )

        # Format with agents if requested
        formatted_output = None
        if agent_config and Path(agent_config).exists():
            try:
                formatted_output = format_analysis_with_agents(analysis_result, agent_config)
            except Exception as e:
                req_logger.warning(f"Agent formatting failed: {e}")

        # Record success metrics
        metrics.record_counter("api_requests_total", 1, tags={
            "endpoint": "analyze_repository",
            "status": "success",
            "organization": user.organization
        })

        response_data = {
            'analysis_id': f"analysis_{int(time.time())}_{user.user_id}",
            'status': 'completed',
            'analysis_result': {
                'security': asdict(analysis_result.security),
                'style': asdict(analysis_result.style),
                'performance': asdict(analysis_result.performance),
                'metadata': analysis_result.metadata
            },
            'formatted_output': formatted_output,
            'user_context': {
                'user_id': user.user_id,
                'organization': user.organization,
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }
        }

        req_logger.info("Repository analysis completed successfully", extra={
            'analysis_duration': analysis_result.metadata.get('analysis_duration'),
            'security_severity': analysis_result.security.metadata.get('severity')
        })

        return jsonify(response_data)

    except Exception as e:
        # Record error metrics
        metrics.record_counter("api_requests_total", 1, tags={
            "endpoint": "analyze_repository",
            "status": "error",
            "organization": user.organization
        })

        req_logger.error(f"Repository analysis failed: {e}")
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e),
            'analysis_id': f"failed_{int(time.time())}_{user.user_id}"
        }), 500


@app.route('/api/v1/user/profile', methods=['GET'])
@require_auth()
def get_user_profile():
    """Get current user profile and usage statistics."""
    user = g.current_user

    # Get usage statistics
    current_date = datetime.now(timezone.utc).date()
    user_key = f"{user.user_id}:{current_date}"
    daily_usage = rate_manager.usage_tracking.get(user_key, 0)

    return jsonify({
        'user': user.to_dict(),
        'usage_statistics': {
            'daily_usage': daily_usage,
            'daily_quota': user.daily_quota,
            'daily_remaining': max(0, user.daily_quota - daily_usage),
            'monthly_quota': user.monthly_quota,
            'last_reset': current_date.isoformat()
        }
    })


@app.route('/api/v1/metrics', methods=['GET'])
@require_auth('admin')
def get_metrics():
    """Get system metrics (admin only)."""
    try:
        # Collect system metrics
        system_metrics = {
            'total_users': len(auth_manager.users),
            'active_users': len([u for u in auth_manager.users.values() if u.is_active]),
            'total_requests_today': sum(v for k, v in rate_manager.usage_tracking.items()
                                      if k.endswith(str(datetime.now(timezone.utc).date()))),
            'system_status': 'operational',
            'uptime': time.time() - app._start_time if hasattr(app, '_start_time') else 0
        }

        return jsonify({
            'metrics': system_metrics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return jsonify({
            'error': 'Metrics unavailable',
            'message': str(e)
        }), 500


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Handle file upload size limit exceeded."""
    return jsonify({
        'error': 'File too large',
        'message': 'Request size exceeds 50MB limit'
    }), 413


@app.errorhandler(429)
def handle_rate_limit(e):
    """Handle rate limit exceeded."""
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests, please try again later'
    }), 429


@app.errorhandler(500)
def handle_internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {e}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


def create_api_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    """Create and configure the API application."""
    if config:
        app.config.update(config)

    # Set start time for uptime calculation
    app._start_time = time.time()

    logger.info("AutoGen API Gateway initialized", extra={
        'version': '1.0.0',
        'max_content_length': app.config['MAX_CONTENT_LENGTH'],
        'cors_origins': "configured"
    })

    return app


if __name__ == '__main__':
    # Development server
    app.run(host='0.0.0.0', port=8080, debug=False)

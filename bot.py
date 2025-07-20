#!/usr/bin/env python3
"""AutoGen Code Review Bot - Main CLI entry point with webhook server."""

import argparse
import sys
import os
import json
import yaml
import logging
import hmac
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import tempfile
import subprocess

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autogen_code_review_bot.pr_analysis import analyze_pr
from autogen_code_review_bot.github_integration import analyze_and_comment


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the bot."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/default.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file and environment variables."""
        config = {
            "github": {
                "webhook_secret": os.getenv("GITHUB_WEBHOOK_SECRET", ""),
                "bot_token": os.getenv("GITHUB_TOKEN", ""),
            },
            "review_criteria": {
                "security_scan": True,
                "performance_check": True,
                "test_coverage": True,
                "documentation": True,
            },
            "server": {
                "host": os.getenv("HOST", "0.0.0.0"),
                "port": int(os.getenv("PORT", "8080")),
            }
        }
        
        # Load from config file if it exists
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f) or {}
                # Merge file config with defaults (environment takes precedence)
                config.update(file_config)
            except (yaml.YAMLError, OSError) as e:
                logger.warning(f"Failed to load config file {self.config_path}: {e}")
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


class WebhookHandler(BaseHTTPRequestHandler):
    """HTTP request handler for GitHub webhooks."""
    
    def __init__(self, *args, config: Config, **kwargs):
        self.config = config
        super().__init__(*args, **kwargs)
    
    def do_POST(self):
        """Handle incoming webhook POST requests."""
        if self.path != "/webhook":
            self.send_error(404, "Not Found")
            return
        
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            # Verify webhook signature
            if not self._verify_signature(body):
                self.send_error(401, "Unauthorized")
                return
            
            # Parse JSON payload
            try:
                payload = json.loads(body.decode('utf-8'))
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return
            
            # Process webhook event
            self._process_webhook(payload)
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "success"}')
            
        except Exception as e:
            logger.error(f"Webhook processing error: {e}")
            self.send_error(500, "Internal Server Error")
    
    def _verify_signature(self, body: bytes) -> bool:
        """Verify GitHub webhook signature."""
        signature = self.headers.get('X-Hub-Signature-256')
        if not signature:
            logger.warning("Missing webhook signature")
            return False
        
        webhook_secret = self.config.get('github.webhook_secret')
        if not webhook_secret:
            logger.warning("No webhook secret configured")
            return False
        
        expected_signature = 'sha256=' + hmac.new(
            webhook_secret.encode('utf-8'),
            body,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    def _process_webhook(self, payload: Dict[str, Any]) -> None:
        """Process GitHub webhook payload."""
        event_type = self.headers.get('X-GitHub-Event')
        action = payload.get('action')
        
        logger.info(f"Received {event_type} event with action: {action}")
        
        # Handle pull request events
        if event_type == 'pull_request' and action in ['opened', 'synchronize', 'reopened']:
            self._handle_pr_event(payload)
        else:
            logger.info(f"Ignoring {event_type}:{action} event")
    
    def _handle_pr_event(self, payload: Dict[str, Any]) -> None:
        """Handle pull request webhook events."""
        try:
            repo_full_name = payload['repository']['full_name']
            pr_number = payload['number']
            
            logger.info(f"Processing PR #{pr_number} for {repo_full_name}")
            
            # Clone repository temporarily and analyze
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_path = self._clone_repository(payload['repository'], temp_dir)
                if repo_path:
                    # Run analysis and post comment
                    analyze_and_comment(repo_path, repo_full_name, pr_number)
                    logger.info(f"Analysis completed for PR #{pr_number}")
                else:
                    logger.error(f"Failed to clone repository {repo_full_name}")
                    
        except KeyError as e:
            logger.error(f"Missing required field in webhook payload: {e}")
        except Exception as e:
            logger.error(f"Error processing PR event: {e}")
    
    def _clone_repository(self, repo_info: Dict[str, Any], temp_dir: str) -> Optional[str]:
        """Clone repository to temporary directory."""
        try:
            clone_url = repo_info['clone_url']
            repo_name = repo_info['name']
            repo_path = Path(temp_dir) / repo_name
            
            # Use GitHub token for private repos if available
            token = self.config.get('github.bot_token')
            if token:
                # Insert token into clone URL
                parsed_url = urlparse(clone_url)
                auth_url = f"https://{token}@{parsed_url.netloc}{parsed_url.path}"
            else:
                auth_url = clone_url
            
            # Clone repository
            result = subprocess.run(
                ['git', 'clone', '--depth=1', auth_url, str(repo_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return str(repo_path)
            else:
                logger.error(f"Git clone failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Git clone timed out")
            return None
        except Exception as e:
            logger.error(f"Clone error: {e}")
            return None
    
    def log_message(self, format, *args):
        """Override to use our logger instead of stderr."""
        logger.info(f"{self.address_string()} - {format % args}")


def create_webhook_handler(config: Config):
    """Create webhook handler with config dependency injection."""
    def handler(*args, **kwargs):
        return WebhookHandler(*args, config=config, **kwargs)
    return handler


def run_server(config: Config) -> None:
    """Run the webhook server."""
    host = config.get('server.host', '0.0.0.0')
    port = config.get('server.port', 8080)
    
    # Validate required configuration
    if not config.get('github.webhook_secret'):
        logger.error("GITHUB_WEBHOOK_SECRET environment variable is required")
        sys.exit(1)
    
    if not config.get('github.bot_token'):
        logger.error("GITHUB_TOKEN environment variable is required")
        sys.exit(1)
    
    # Create and start server
    handler_class = create_webhook_handler(config)
    server = HTTPServer((host, port), handler_class)
    
    logger.info(f"Starting webhook server on {host}:{port}")
    logger.info("Webhook endpoint: /webhook")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.shutdown()


def manual_analysis(repo_path: str, config_path: Optional[str] = None) -> None:
    """Run manual analysis on a local repository."""
    if not Path(repo_path).is_dir():
        logger.error(f"Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    logger.info(f"Running analysis on {repo_path}")
    
    try:
        result = analyze_pr(repo_path, config_path)
        
        # Print results
        print("\\n=== AutoGen Code Review Results ===\\n")
        
        print(f"🔒 Security Analysis ({result.security.tool}):")
        print(result.security.output or "No issues found")
        print()
        
        print(f"🎨 Style Analysis ({result.style.tool}):")
        print(result.style.output or "No issues found")
        print()
        
        print(f"⚡ Performance Analysis ({result.performance.tool}):")
        print(result.performance.output or "No issues found")
        print()
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AutoGen Code Review Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start webhook server
  python bot.py --server --config config/production.yaml
  
  # Run manual analysis
  python bot.py --analyze /path/to/repo
  
  # Run with custom linter config
  python bot.py --analyze /path/to/repo --linter-config linters.yaml
"""
    )
    
    parser.add_argument(
        '--server', 
        action='store_true',
        help='Start webhook server mode'
    )
    
    parser.add_argument(
        '--analyze',
        metavar='REPO_PATH',
        help='Run manual analysis on local repository'
    )
    
    parser.add_argument(
        '--config',
        metavar='CONFIG_FILE',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--linter-config',
        metavar='LINTER_CONFIG',
        help='Path to linter configuration YAML file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = Config(args.config)
    
    # Execute based on mode
    if args.server:
        run_server(config)
    elif args.analyze:
        manual_analysis(args.analyze, args.linter_config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
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
import time
from pathlib import Path
from typing import Dict, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import tempfile
import subprocess

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autogen_code_review_bot.pr_analysis import analyze_pr, format_analysis_with_agents
from autogen_code_review_bot.github_integration import analyze_and_comment
from autogen_code_review_bot.logging_config import (
    configure_logging, get_logger, set_request_id, 
    log_operation_start, log_operation_end, ContextLogger
)
from autogen_code_review_bot.webhook_deduplication import is_duplicate_event
from autogen_code_review_bot.coverage_metrics import (
    CoverageConfig, generate_coverage_report, validate_coverage_threshold
)
from autogen_code_review_bot.monitoring import MetricsEmitter, MonitoringServer

# Configure structured logging and metrics
configure_logging(level="INFO", service_name="autogen-code-review-bot")
logger = get_logger(__name__)
metrics = MetricsEmitter()


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
                from autogen_code_review_bot.config_validation import validate_config_file, ConfigError, ValidationError
                
                # Validate configuration using our validation framework
                file_config = validate_config_file(self.config_path, "bot")
                
                # Merge file config with defaults (environment takes precedence)
                config.update(file_config)
                logger.info(f"Successfully loaded and validated configuration", 
                           extra={"config_path": self.config_path, "sections": list(file_config.keys())})
                
            except (ConfigError, ValidationError) as e:
                logger.error(f"Configuration validation failed: {e}", 
                           extra={"config_path": self.config_path})
                logger.warning("Using default configuration due to validation errors")
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
        # Generate request ID for this webhook request
        request_id = set_request_id()
        
        # Create context logger for this request
        req_logger = ContextLogger(logger, request_id=request_id, endpoint=self.path)
        
        if self.path != "/webhook":
            req_logger.warning("Invalid endpoint requested", endpoint=self.path)
            self.send_error(404, "Not Found")
            return
        
        # Start operation tracking
        operation_context = log_operation_start(
            logger, 
            "webhook_request",
            request_id=request_id,
            client_ip=self.client_address[0],
            user_agent=self.headers.get('User-Agent', 'unknown')
        )
        
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            req_logger.debug("Webhook request received", content_length=content_length)
            
            # Verify webhook signature
            if not self._verify_signature(body):
                req_logger.warning("Webhook signature verification failed")
                self.send_error(401, "Unauthorized")
                log_operation_end(logger, operation_context, success=False, error="signature_verification_failed")
                return
            
            # Check for duplicate events using GitHub delivery ID
            delivery_id = self.headers.get('X-GitHub-Delivery')
            if is_duplicate_event(delivery_id):
                req_logger.info("Duplicate webhook event detected, skipping processing", 
                               delivery_id=delivery_id)
                # Return success response for duplicates to avoid retries
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "duplicate", "message": "Event already processed"}')
                log_operation_end(logger, operation_context, success=True, result="duplicate_skipped")
                return
            
            req_logger.debug("Webhook event accepted for processing", delivery_id=delivery_id)
            
            # Parse JSON payload
            try:
                payload = json.loads(body.decode('utf-8'))
            except json.JSONDecodeError as e:
                req_logger.error("Invalid JSON payload", error=str(e))
                self.send_error(400, "Invalid JSON")
                log_operation_end(logger, operation_context, success=False, error="invalid_json")
                return
            
            # Process webhook event
            self._process_webhook(payload, req_logger)
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "success"}')
            
            # Record successful webhook processing metrics
            metrics.record_counter("webhook_requests_total", 1, tags={"status": "success"})
            
            log_operation_end(logger, operation_context, success=True)
            
        except Exception as e:
            req_logger.error("Webhook processing error", error=str(e), error_type=type(e).__name__)
            self.send_error(500, "Internal Server Error")
            # Record failed webhook processing metrics
            metrics.record_counter("webhook_requests_total", 1, tags={"status": "error"})
            metrics.record_counter("webhook_errors_total", 1, tags={"error_type": type(e).__name__})
            log_operation_end(logger, operation_context, success=False, error=str(e))
    
    def _verify_signature(self, body: bytes) -> bool:
        """Verify GitHub webhook signature."""
        signature = self.headers.get('X-Hub-Signature-256')
        if not signature:
            return False
        
        webhook_secret = self.config.get('github.webhook_secret')
        if not webhook_secret:
            return False
        
        expected_signature = 'sha256=' + hmac.new(
            webhook_secret.encode('utf-8'),
            body,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    def _process_webhook(self, payload: Dict[str, Any], req_logger: ContextLogger) -> None:
        """Process GitHub webhook payload."""
        event_type = self.headers.get('X-GitHub-Event')
        action = payload.get('action')
        
        # Record webhook event metrics
        metrics.record_counter("webhook_events_total", 1, tags={"event_type": event_type or "unknown", "action": action or "unknown"})
        
        req_logger.info("Webhook event received", 
                       event_type=event_type, 
                       action=action,
                       repository=payload.get('repository', {}).get('full_name'))
        
        # Handle pull request events
        if event_type == 'pull_request' and action in ['opened', 'synchronize', 'reopened']:
            # Record PR event processing metrics
            metrics.record_counter("pr_events_processed_total", 1, tags={"action": action})
            self._handle_pr_event(payload, req_logger)
        else:
            # Record ignored event metrics
            metrics.record_counter("webhook_events_ignored_total", 1, tags={"event_type": event_type or "unknown", "action": action or "unknown"})
            req_logger.info("Ignoring webhook event", 
                          event_type=event_type, 
                          action=action,
                          reason="not_a_relevant_pr_event")
    
    def _handle_pr_event(self, payload: Dict[str, Any], req_logger: ContextLogger) -> None:
        """Handle pull request webhook events."""
        try:
            repo_full_name = payload['repository']['full_name']
            pr_number = payload['number']
            pr_head_sha = payload['pull_request']['head']['sha']
            
            # Create PR-specific context logger
            pr_logger = ContextLogger(
                logger, 
                repository=repo_full_name,
                pr_number=pr_number,
                head_sha=pr_head_sha[:8]  # Short SHA for readability
            )
            
            # Start PR processing operation
            pr_context = log_operation_start(
                logger, 
                "pr_analysis",
                repository=repo_full_name,
                pr_number=pr_number,
                head_sha=pr_head_sha
            )
            
            pr_logger.info("Starting PR analysis")
            
            # Clone repository temporarily and analyze
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_path = self._clone_repository(payload['repository'], temp_dir, pr_logger)
                if repo_path:
                    try:
                        # Record PR analysis start
                        pr_start_time = time.time()
                        
                        # Run analysis and post comment
                        analyze_and_comment(repo_path, repo_full_name, pr_number)
                        
                        # Record successful PR analysis metrics
                        pr_duration = time.time() - pr_start_time
                        metrics.record_histogram("pr_analysis_webhook_duration_seconds", pr_duration)
                        metrics.record_counter("pr_analysis_webhook_completed_total", 1, tags={"status": "success"})
                        
                        pr_logger.info("PR analysis completed successfully", extra={"duration_seconds": pr_duration})
                        log_operation_end(logger, pr_context, success=True)
                    except Exception as analysis_error:
                        # Record failed PR analysis metrics
                        metrics.record_counter("pr_analysis_webhook_completed_total", 1, tags={"status": "error"})
                        metrics.record_counter("pr_analysis_webhook_errors_total", 1, tags={"error_type": type(analysis_error).__name__})
                        
                        pr_logger.error("Analysis failed", error=str(analysis_error))
                        log_operation_end(logger, pr_context, success=False, error=str(analysis_error))
                        raise
                else:
                    error_msg = f"Failed to clone repository {repo_full_name}"
                    pr_logger.error("Repository clone failed")
                    log_operation_end(logger, pr_context, success=False, error="clone_failed")
                    raise RuntimeError(error_msg)
                    
        except KeyError as e:
            req_logger.error("Missing required field in webhook payload", 
                           missing_field=str(e), 
                           available_keys=list(payload.keys()))
            raise
        except Exception as e:
            req_logger.error("Error processing PR event", 
                           error=str(e), 
                           error_type=type(e).__name__)
            raise
    
    def _clone_repository(self, repo_info: Dict[str, Any], temp_dir: str, 
                         pr_logger: ContextLogger) -> Optional[str]:
        """Clone repository to temporary directory."""
        clone_context = log_operation_start(
            logger,
            "git_clone",
            repository=repo_info.get('full_name'),
            clone_url=repo_info.get('clone_url', '').replace(self.config.get('github.bot_token', ''), '***')
        )
        
        try:
            clone_url = repo_info['clone_url']
            repo_name = repo_info['name']
            repo_path = Path(temp_dir) / repo_name
            
            pr_logger.debug("Starting repository clone", 
                          repo_name=repo_name,
                          temp_dir=temp_dir)
            
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
                pr_logger.info("Repository cloned successfully", 
                             repo_path=str(repo_path),
                             clone_depth=1)
                log_operation_end(logger, clone_context, success=True, repo_path=str(repo_path))
                return str(repo_path)
            else:
                pr_logger.error("Git clone command failed", 
                              return_code=result.returncode,
                              stderr=result.stderr)
                log_operation_end(logger, clone_context, success=False, error=result.stderr)
                return None
                
        except subprocess.TimeoutExpired:
            pr_logger.error("Git clone operation timed out", timeout_seconds=60)
            log_operation_end(logger, clone_context, success=False, error="timeout")
            return None
        except Exception as e:
            pr_logger.error("Clone operation failed", 
                          error=str(e), 
                          error_type=type(e).__name__)
            log_operation_end(logger, clone_context, success=False, error=str(e))
            return None
    
    def log_message(self, format, *args):
        """Override to use our logger instead of stderr."""
        # Use structured logging for HTTP server messages
        message = format % args
        logger.info("HTTP request", 
                   client_address=self.address_string(),
                   message=message,
                   component="http_server")


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
        logger.error("Missing required configuration", 
                    missing_config="GITHUB_WEBHOOK_SECRET",
                    component="server_startup")
        sys.exit(1)
    
    if not config.get('github.bot_token'):
        logger.error("Missing required configuration", 
                    missing_config="GITHUB_TOKEN",
                    component="server_startup")
        sys.exit(1)
    
    # Create and start server
    handler_class = create_webhook_handler(config)
    server = HTTPServer((host, port), handler_class)
    
    logger.info("Starting webhook server", 
               host=host, 
               port=port,
               webhook_endpoint="/webhook",
               component="server_startup")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested", component="server_shutdown")
        server.shutdown()


def manual_analysis(repo_path: str, config_path: Optional[str] = None, agent_config_path: Optional[str] = None) -> None:
    """Run manual analysis on a local repository."""
    if not Path(repo_path).is_dir():
        logger.error("Repository path does not exist", 
                    repo_path=repo_path,
                    component="manual_analysis")
        sys.exit(1)
    
    # Start manual analysis operation
    analysis_context = log_operation_start(
        logger,
        "manual_analysis",
        repo_path=repo_path,
        config_path=config_path
    )
    
    logger.info("Starting manual repository analysis", repo_path=repo_path)
    
    try:
        result = analyze_pr(repo_path, config_path)
        
        logger.info("Manual analysis completed successfully",
                   security_tool=result.security.tool,
                   style_tool=result.style.tool,
                   performance_tool=result.performance.tool,
                   using_agent_conversation=agent_config_path is not None)
        
        # Format results with agent conversation if config provided
        if agent_config_path:
            print("\\n=== AutoGen Code Review Results with Agent Conversation ===\\n")
            formatted_output = format_analysis_with_agents(result, agent_config_path)
            print(formatted_output)
        else:
            # Print results in traditional format
            print("\\n=== AutoGen Code Review Results ===\\n")
            print("💡 Tip: Use --agent-config for enhanced AI-powered review discussions\\n")
            
            print(f"🔒 Security Analysis ({result.security.tool}):")
            print(result.security.output or "No issues found")
            print()
            
            print(f"🎨 Style Analysis ({result.style.tool}):")
            print(result.style.output or "No issues found")
            print()
            
            print(f"⚡ Performance Analysis ({result.performance.tool}):")
            print(result.performance.output or "No issues found")
            print()
        
        log_operation_end(logger, analysis_context, success=True)
        
    except Exception as e:
        logger.error("Manual analysis failed", 
                    error=str(e),
                    error_type=type(e).__name__)
        log_operation_end(logger, analysis_context, success=False, error=str(e))
        sys.exit(1)


def run_coverage_analysis(repo_path: str, config_file: Optional[str] = None):
    """Run test coverage analysis on a repository.
    
    Args:
        repo_path: Path to the repository
        config_file: Optional coverage configuration file
    """
    # Validate repository path
    if not os.path.exists(repo_path):
        logger.error("Repository path does not exist", repo_path=repo_path)
        print(f"❌ Error: Repository path '{repo_path}' does not exist")
        sys.exit(1)
    
    # Set up operation tracking
    analysis_context = log_operation_start(
        logger,
        "coverage_analysis",
        repo_path=repo_path,
        config_file=config_file
    )
    
    try:
        logger.info("Starting coverage analysis", repo_path=repo_path)
        print(f"🔍 Running coverage analysis on: {repo_path}")
        
        # Load coverage configuration
        if config_file and os.path.exists(config_file):
            coverage_config = CoverageConfig.from_file(config_file)
            print(f"📋 Loaded coverage config from: {config_file}")
        else:
            coverage_config = CoverageConfig()
            print(f"📋 Using default coverage config (threshold: {coverage_config.minimum_coverage}%)")
        
        # Create output directory for reports
        output_dir = Path(repo_path) / "coverage_reports"
        output_dir.mkdir(exist_ok=True)
        
        # Run coverage analysis
        print("🧪 Discovering and running tests with coverage...")
        result, html_report = generate_coverage_report(
            repo_path, 
            coverage_config, 
            str(output_dir)
        )
        
        # Display results
        print("\n📊 Coverage Analysis Results:")
        print(f"   Total Coverage: {result.total_coverage:.1f}%")
        print(f"   Line Coverage: {result.line_coverage:.1f}%")
        print(f"   Branch Coverage: {result.branch_coverage:.1f}%")
        print(f"   Files Analyzed: {result.files_analyzed}")
        print(f"   Lines Covered: {result.lines_covered} / {result.lines_total}")
        
        # Validate against threshold
        is_valid, message = validate_coverage_threshold(result, coverage_config.minimum_coverage)
        
        if is_valid:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
        
        # Report HTML output
        if html_report:
            print(f"📄 HTML report generated: {html_report}")
        
        # Save JSON report
        json_report_path = output_dir / f"coverage_{int(time.time())}.json"
        with open(json_report_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"💾 JSON report saved: {json_report_path}")
        
        log_operation_end(
            logger, 
            analysis_context, 
            success=True,
            coverage=result.total_coverage,
            meets_threshold=is_valid
        )
        
        # Exit with appropriate code
        if not is_valid:
            print(f"\n❌ Coverage analysis failed: {message}")
            sys.exit(1)
        else:
            print(f"\n🎉 Coverage analysis passed!")
            
    except Exception as e:
        logger.error("Coverage analysis failed", error=str(e), error_type=type(e).__name__)
        print(f"❌ Coverage analysis failed: {str(e)}")
        log_operation_end(logger, analysis_context, success=False, error=str(e))
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
  
  # Run with AI agent conversations for enhanced feedback
  python bot.py --analyze /path/to/repo --agent-config agent_config.yaml
  
  # Run test coverage analysis
  python bot.py --coverage /path/to/repo
  
  # Run coverage with custom config
  python bot.py --coverage /path/to/repo --config coverage.yaml
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
        '--coverage',
        metavar='REPO_PATH',
        help='Run test coverage analysis on repository'
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
        '--agent-config',
        metavar='AGENT_CONFIG',
        help='Path to agent configuration YAML file for enhanced reviews'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    log_level = "DEBUG" if args.verbose else "INFO"
    configure_logging(level=log_level, service_name="autogen-code-review-bot")
    
    # Load configuration
    config = Config(args.config)
    
    # Execute based on mode
    if args.server:
        run_server(config)
    elif args.analyze:
        manual_analysis(args.analyze, args.linter_config, args.agent_config)
    elif args.coverage:
        run_coverage_analysis(args.coverage, args.config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
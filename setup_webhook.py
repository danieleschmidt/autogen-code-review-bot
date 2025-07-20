#!/usr/bin/env python3
"""GitHub Webhook Setup Utility for AutoGen Code Review Bot."""

import argparse
import sys
import os
import requests
from typing import Dict, Any, Optional
from urllib.parse import urlparse


class GitHubWebhookManager:
    """Manager for GitHub webhook operations."""
    
    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'AutoGen-Code-Review-Bot/1.0'
        })
    
    def create_webhook(
        self, 
        repo: str, 
        webhook_url: str, 
        secret: str,
        events: Optional[list] = None
    ) -> Dict[str, Any]:
        """Create a new webhook for the repository.
        
        Args:
            repo: Repository in format 'owner/name'
            webhook_url: URL where webhook events will be sent
            secret: Secret for webhook signature verification
            events: List of events to subscribe to
            
        Returns:
            Webhook creation response
        """
        if events is None:
            events = ['pull_request', 'push']
        
        webhook_config = {
            'name': 'web',
            'active': True,
            'events': events,
            'config': {
                'url': webhook_url,
                'content_type': 'json',
                'secret': secret,
                'insecure_ssl': '0'  # Always require SSL
            }
        }
        
        url = f'https://api.github.com/repos/{repo}/hooks'
        
        try:
            response = self.session.post(url, json=webhook_config)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to create webhook: {e}")
    
    def list_webhooks(self, repo: str) -> list:
        """List existing webhooks for the repository."""
        url = f'https://api.github.com/repos/{repo}/hooks'
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to list webhooks: {e}")
    
    def delete_webhook(self, repo: str, webhook_id: int) -> bool:
        """Delete a webhook by ID."""
        url = f'https://api.github.com/repos/{repo}/hooks/{webhook_id}'
        
        try:
            response = self.session.delete(url)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to delete webhook: {e}")
    
    def test_webhook(self, repo: str, webhook_id: int) -> Dict[str, Any]:
        """Test a webhook by sending a ping event."""
        url = f'https://api.github.com/repos/{repo}/hooks/{webhook_id}/pings'
        
        try:
            response = self.session.post(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to test webhook: {e}")


def validate_webhook_url(url: str) -> bool:
    """Validate that webhook URL is properly formatted and accessible."""
    try:
        parsed = urlparse(url)
        
        # Must be HTTPS for security
        if parsed.scheme != 'https':
            print("❌ Webhook URL must use HTTPS")
            return False
        
        # Must have valid host
        if not parsed.netloc:
            print("❌ Invalid webhook URL format")
            return False
        
        # Test URL accessibility (ping test)
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 404:
                print("✅ Webhook URL is accessible (404 expected for GET)")
                return True
            elif response.status_code < 500:
                print("✅ Webhook URL is accessible")
                return True
            else:
                print(f"⚠️  Webhook URL returned {response.status_code}")
                return True  # Still allow, might be server-specific
        except requests.exceptions.RequestException:
            print("⚠️  Could not verify webhook URL accessibility")
            return True  # Allow anyway, might be behind firewall
        
    except Exception as e:
        print(f"❌ Invalid webhook URL: {e}")
        return False


def validate_repo_format(repo: str) -> bool:
    """Validate repository format (owner/name)."""
    if '/' not in repo:
        print("❌ Repository must be in format 'owner/name'")
        return False
    
    parts = repo.split('/')
    if len(parts) != 2:
        print("❌ Repository must be in format 'owner/name'")
        return False
    
    owner, name = parts
    if not owner or not name:
        print("❌ Repository owner and name cannot be empty")
        return False
    
    return True


def setup_webhook(args) -> None:
    """Set up a new webhook for the repository."""
    # Validate inputs
    if not validate_repo_format(args.repo):
        sys.exit(1)
    
    if not validate_webhook_url(args.url):
        sys.exit(1)
    
    # Get tokens
    github_token = args.token or os.getenv('GITHUB_TOKEN')
    webhook_secret = args.secret or os.getenv('GITHUB_WEBHOOK_SECRET')
    
    if not github_token:
        print("❌ GitHub token is required. Use --token or set GITHUB_TOKEN environment variable")
        sys.exit(1)
    
    if not webhook_secret:
        print("❌ Webhook secret is required. Use --secret or set GITHUB_WEBHOOK_SECRET environment variable")
        sys.exit(1)
    
    # Create webhook manager
    manager = GitHubWebhookManager(github_token)
    
    try:
        print(f"Creating webhook for {args.repo}...")
        result = manager.create_webhook(
            args.repo,
            args.url,
            webhook_secret,
            events=['pull_request', 'push', 'pull_request_review']
        )
        
        print("✅ Webhook created successfully!")
        print(f"   ID: {result['id']}")
        print(f"   URL: {result['config']['url']}")
        print(f"   Events: {', '.join(result['events'])}")
        
        # Test the webhook
        if args.test:
            print("\\nTesting webhook...")
            try:
                manager.test_webhook(args.repo, result['id'])
                print("✅ Webhook test successful!")
            except Exception as e:
                print(f"⚠️  Webhook test failed: {e}")
        
    except Exception as e:
        print(f"❌ Failed to create webhook: {e}")
        sys.exit(1)


def list_webhooks(args) -> None:
    """List existing webhooks for the repository."""
    if not validate_repo_format(args.repo):
        sys.exit(1)
    
    github_token = args.token or os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("❌ GitHub token is required. Use --token or set GITHUB_TOKEN environment variable")
        sys.exit(1)
    
    manager = GitHubWebhookManager(github_token)
    
    try:
        webhooks = manager.list_webhooks(args.repo)
        
        if not webhooks:
            print(f"No webhooks found for {args.repo}")
            return
        
        print(f"Webhooks for {args.repo}:")
        print("-" * 50)
        
        for webhook in webhooks:
            print(f"ID: {webhook['id']}")
            print(f"URL: {webhook['config'].get('url', 'N/A')}")
            print(f"Events: {', '.join(webhook['events'])}")
            print(f"Active: {webhook['active']}")
            print(f"Created: {webhook['created_at']}")
            print("-" * 50)
    
    except Exception as e:
        print(f"❌ Failed to list webhooks: {e}")
        sys.exit(1)


def delete_webhook(args) -> None:
    """Delete a webhook by ID."""
    if not validate_repo_format(args.repo):
        sys.exit(1)
    
    github_token = args.token or os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("❌ GitHub token is required. Use --token or set GITHUB_TOKEN environment variable")
        sys.exit(1)
    
    manager = GitHubWebhookManager(github_token)
    
    try:
        if manager.delete_webhook(args.repo, args.webhook_id):
            print(f"✅ Webhook {args.webhook_id} deleted successfully!")
        else:
            print(f"❌ Failed to delete webhook {args.webhook_id}")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Failed to delete webhook: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GitHub Webhook Setup Utility for AutoGen Code Review Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new webhook
  python setup_webhook.py create --repo owner/repo --url https://example.com/webhook
  
  # List existing webhooks
  python setup_webhook.py list --repo owner/repo
  
  # Delete a webhook
  python setup_webhook.py delete --repo owner/repo --webhook-id 123
  
  # Create webhook with custom secret
  python setup_webhook.py create --repo owner/repo --url https://example.com/webhook --secret mysecret

Environment Variables:
  GITHUB_TOKEN - GitHub personal access token (required)
  GITHUB_WEBHOOK_SECRET - Secret for webhook signature verification (optional)
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create webhook command
    create_parser = subparsers.add_parser('create', help='Create a new webhook')
    create_parser.add_argument('--repo', required=True, help='Repository in format owner/name')
    create_parser.add_argument('--url', required=True, help='Webhook URL')
    create_parser.add_argument('--token', help='GitHub token (or use GITHUB_TOKEN env var)')
    create_parser.add_argument('--secret', help='Webhook secret (or use GITHUB_WEBHOOK_SECRET env var)')
    create_parser.add_argument('--test', action='store_true', help='Test webhook after creation')
    
    # List webhooks command
    list_parser = subparsers.add_parser('list', help='List existing webhooks')
    list_parser.add_argument('--repo', required=True, help='Repository in format owner/name')
    list_parser.add_argument('--token', help='GitHub token (or use GITHUB_TOKEN env var)')
    
    # Delete webhook command
    delete_parser = subparsers.add_parser('delete', help='Delete a webhook')
    delete_parser.add_argument('--repo', required=True, help='Repository in format owner/name')
    delete_parser.add_argument('--webhook-id', type=int, required=True, help='Webhook ID to delete')
    delete_parser.add_argument('--token', help='GitHub token (or use GITHUB_TOKEN env var)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'create':
        setup_webhook(args)
    elif args.command == 'list':
        list_webhooks(args)
    elif args.command == 'delete':
        delete_webhook(args)


if __name__ == "__main__":
    main()
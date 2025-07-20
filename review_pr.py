#!/usr/bin/env python3
"""Manual PR Review Script for AutoGen Code Review Bot."""

import argparse
import sys
import os
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autogen_code_review_bot.pr_analysis import analyze_pr
from autogen_code_review_bot.github_integration import analyze_and_comment


def clone_repository(repo_url: str, temp_dir: str, token: Optional[str] = None) -> Optional[str]:
    """Clone repository to temporary directory.
    
    Args:
        repo_url: Repository URL to clone
        temp_dir: Temporary directory path
        token: Optional GitHub token for private repos
        
    Returns:
        Path to cloned repository or None if failed
    """
    try:
        repo_name = Path(repo_url).stem
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        
        repo_path = Path(temp_dir) / repo_name
        
        # Add token to URL if provided
        if token and 'github.com' in repo_url:
            if repo_url.startswith('https://'):
                auth_url = repo_url.replace('https://', f'https://{token}@')
            else:
                auth_url = repo_url
        else:
            auth_url = repo_url
        
        # Clone with depth 1 for faster cloning
        result = subprocess.run(
            ['git', 'clone', '--depth=1', auth_url, str(repo_path)],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print(f"✅ Repository cloned to {repo_path}")
            return str(repo_path)
        else:
            print(f"❌ Git clone failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ Git clone timed out")
        return None
    except Exception as e:
        print(f"❌ Clone error: {e}")
        return None


def review_repository(repo_path: str, linter_config: Optional[str] = None) -> None:
    """Run analysis on a repository and display results.
    
    Args:
        repo_path: Path to repository to analyze
        linter_config: Optional path to linter configuration file
    """
    if not Path(repo_path).is_dir():
        print(f"❌ Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    print(f"\\n🔍 Analyzing repository: {repo_path}")
    print("=" * 60)
    
    try:
        result = analyze_pr(repo_path, linter_config)
        
        # Display results in a formatted way
        print("\\n" + "🔒 SECURITY ANALYSIS".center(60, "="))
        print(f"Tool: {result.security.tool}")
        if result.security.output.strip():
            print(result.security.output)
        else:
            print("✅ No security issues found")
        
        print("\\n" + "🎨 STYLE ANALYSIS".center(60, "="))
        print(f"Tool: {result.style.tool}")
        if result.style.output.strip():
            print(result.style.output)
        else:
            print("✅ No style issues found")
        
        print("\\n" + "⚡ PERFORMANCE ANALYSIS".center(60, "="))
        print(f"Tool: {result.performance.tool}")
        if result.performance.output.strip():
            print(result.performance.output)
        else:
            print("✅ No performance issues found")
        
        print("\\n" + "=" * 60)
        print("✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


def review_pr_online(repo: str, pr_number: int, token: Optional[str] = None, post_comment: bool = False) -> None:
    """Review a specific PR by number and optionally post comment.
    
    Args:
        repo: Repository in format 'owner/name'
        pr_number: Pull request number
        token: GitHub token
        post_comment: Whether to post results as PR comment
    """
    if not token:
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            print("❌ GitHub token is required. Use --token or set GITHUB_TOKEN environment variable")
            sys.exit(1)
    
    print(f"\\n🔍 Reviewing PR #{pr_number} in {repo}")
    
    # Clone repository temporarily
    repo_url = f"https://github.com/{repo}.git"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = clone_repository(repo_url, temp_dir, token)
        if not repo_path:
            sys.exit(1)
        
        if post_comment:
            try:
                print("\\n📝 Running analysis and posting comment...")
                analyze_and_comment(repo_path, repo, pr_number)
                print("✅ Comment posted successfully!")
            except Exception as e:
                print(f"❌ Failed to post comment: {e}")
                sys.exit(1)
        else:
            # Just show results locally
            review_repository(repo_path)


def review_local_diff(repo_path: str, diff_target: str = "HEAD~1", linter_config: Optional[str] = None) -> None:
    """Review local changes by comparing with a git reference.
    
    Args:
        repo_path: Path to git repository
        diff_target: Git reference to compare against (default: HEAD~1)
        linter_config: Optional linter configuration file
    """
    repo_path = Path(repo_path).resolve()
    
    if not repo_path.is_dir():
        print(f"❌ Path does not exist: {repo_path}")
        sys.exit(1)
    
    # Check if it's a git repository
    if not (repo_path / ".git").exists():
        print(f"❌ Not a git repository: {repo_path}")
        sys.exit(1)
    
    print(f"\\n🔍 Reviewing local changes in {repo_path}")
    print(f"📊 Comparing against: {diff_target}")
    
    try:
        # Get list of changed files
        result = subprocess.run(
            ['git', 'diff', '--name-only', diff_target],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Git diff failed: {result.stderr}")
            sys.exit(1)
        
        changed_files = result.stdout.strip().split('\\n') if result.stdout.strip() else []
        
        if not changed_files:
            print("✅ No changes detected")
            return
        
        print(f"\\n📁 Changed files ({len(changed_files)}):")
        for file in changed_files:
            print(f"  • {file}")
        
        # Run full repository analysis
        review_repository(str(repo_path), linter_config)
        
        # Show diff summary
        print("\\n" + "📊 CHANGE SUMMARY".center(60, "="))
        diff_result = subprocess.run(
            ['git', 'diff', '--stat', diff_target],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if diff_result.returncode == 0 and diff_result.stdout:
            print(diff_result.stdout)
        
    except Exception as e:
        print(f"❌ Local diff analysis failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manual PR Review Script for AutoGen Code Review Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Review a specific PR and post comment
  python review_pr.py --pr-number 123 --repo owner/repo --post-comment
  
  # Review a PR without posting comment
  python review_pr.py --pr-number 123 --repo owner/repo
  
  # Review local repository
  python review_pr.py --path ./my-repo
  
  # Review local changes since last commit
  python review_pr.py --path ./my-repo --diff HEAD~1
  
  # Review with custom linter config
  python review_pr.py --path ./my-repo --linter-config linters.yaml
  
  # Clone and review any repository
  python review_pr.py --clone https://github.com/owner/repo.git

Environment Variables:
  GITHUB_TOKEN - GitHub personal access token (for private repos and posting comments)
"""
    )
    
    # Mutually exclusive group for different modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    
    mode_group.add_argument(
        '--pr-number',
        type=int,
        help='Review specific PR number (requires --repo)'
    )
    
    mode_group.add_argument(
        '--path',
        help='Path to local repository to review'
    )
    
    mode_group.add_argument(
        '--clone',
        help='Clone and review repository from URL'
    )
    
    # Additional options
    parser.add_argument(
        '--repo',
        help='Repository in format owner/name (required with --pr-number)'
    )
    
    parser.add_argument(
        '--token',
        help='GitHub token (or use GITHUB_TOKEN env var)'
    )
    
    parser.add_argument(
        '--post-comment',
        action='store_true',
        help='Post review results as PR comment (only with --pr-number)'
    )
    
    parser.add_argument(
        '--diff',
        default='HEAD~1',
        help='Git reference to compare against for local changes (default: HEAD~1)'
    )
    
    parser.add_argument(
        '--linter-config',
        help='Path to linter configuration YAML file'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.pr_number and not args.repo:
        print("❌ --repo is required when using --pr-number")
        sys.exit(1)
    
    if args.post_comment and not args.pr_number:
        print("❌ --post-comment can only be used with --pr-number")
        sys.exit(1)
    
    # Execute based on mode
    try:
        if args.pr_number:
            review_pr_online(args.repo, args.pr_number, args.token, args.post_comment)
        elif args.path:
            if args.diff != 'HEAD~1':
                review_local_diff(args.path, args.diff, args.linter_config)
            else:
                review_repository(args.path, args.linter_config)
        elif args.clone:
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_path = clone_repository(args.clone, temp_dir, args.token)
                if repo_path:
                    review_repository(repo_path, args.linter_config)
                else:
                    sys.exit(1)
    
    except KeyboardInterrupt:
        print("\\n❌ Review interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
"""Enhanced subprocess security utilities with comprehensive validation."""

import shlex
import subprocess
import unicodedata
from pathlib import Path
from typing import List, Optional

from .exceptions import ToolError, ValidationError
from .logging_config import get_logger
from .system_config import get_system_config

logger = get_logger(__name__)


class SubprocessValidator:
    """Enhanced subprocess validation with security hardening."""

    # Expanded allowlist with version checking
    ALLOWED_EXECUTABLES = {
        # Python tools
        "python", "python3", "pip", "pip3", "pytest", "coverage",
        "ruff", "flake8", "pylint", "mypy", "black", "isort", "bandit",

        # JavaScript/Node.js tools
        "node", "npm", "npx", "yarn", "eslint", "prettier", "jest",

        # Go tools
        "go", "golangci-lint", "gofmt", "goimports",

        # Ruby tools
        "ruby", "gem", "bundle", "rubocop",

        # Rust tools
        "cargo", "rustc", "clippy",

        # Java tools
        "java", "javac", "mvn", "gradle",

        # C/C++ tools
        "gcc", "g++", "clang", "clang++", "make", "cmake",

        # Git operations
        "git",

        # System utilities (carefully curated)
        "ls", "cat", "find", "grep", "wc", "head", "tail", "sort", "uniq",
        "pwd", "which", "env", "echo", "sleep", "timeout",

        # Build and package managers
        "docker", "podman",

        # Analysis tools
        "radon", "sonar-scanner", "shellcheck",
    }

    # Commands that should never have certain arguments
    RESTRICTED_ARGUMENTS = {
        "rm": ["-rf", "--recursive", "--force", "/"],
        "chmod": ["777", "666", "644"],  # Prevent overly permissive permissions
        "git": ["--git-dir=/", "--work-tree=/"],  # Prevent git operations on root
        "find": ["-exec", "-delete"],  # Prevent dangerous find operations
        "grep": ["-r", "/"],  # Prevent recursive searches on root
    }

    # Maximum argument count to prevent argument injection
    MAX_ARGUMENTS = 50

    # Maximum argument length to prevent buffer overflow attempts
    MAX_ARGUMENT_LENGTH = 1000

    # Dangerous characters that should never appear in arguments
    DANGEROUS_CHARS = {
        '\x00',  # Null byte
        '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',  # Control chars
        '\x08', '\x0b', '\x0c', '\x0e', '\x0f',
        '\x10', '\x11', '\x12', '\x13', '\x14', '\x15', '\x16', '\x17',
        '\x18', '\x19', '\x1a', '\x1b', '\x1c', '\x1d', '\x1e', '\x1f',
        '\x7f',  # DEL character
    }

    # Shell metacharacters (expanded list)
    SHELL_METACHARACTERS = {
        '&', '|', ';', '$', '`', '>', '<', '"', "'", '\\',
        '(', ')', '{', '}', '[', ']', '*', '?', '~', '#',
        '!',  # History expansion
        '^',  # Some shells use for negation
    }

    @classmethod
    def validate_command(cls, cmd: List[str], cwd: Optional[str] = None,
                        project_root: Optional[str] = None) -> bool:
        """Comprehensive command validation with enhanced security checks.
        
        Args:
            cmd: Command and arguments to validate
            cwd: Working directory (optional)
            project_root: Project root for path validation (optional)
            
        Returns:
            True if command is safe to execute
            
        Raises:
            ValidationError: If validation fails with details
        """
        try:
            # Basic structure validation
            cls._validate_command_structure(cmd)

            # Executable validation
            cls._validate_executable(cmd[0])

            # Arguments validation
            cls._validate_arguments(cmd)

            # Command-specific restrictions
            cls._validate_command_restrictions(cmd)

            # Path validation if working directory provided
            if cwd:
                cls._validate_working_directory(cwd, project_root)

            # Resource limits validation
            cls._validate_resource_limits(cmd)

            logger.debug("Command validation passed",
                        extra={"command": cmd[0], "arg_count": len(cmd) - 1})
            return True

        except ValidationError:
            raise
        except Exception as e:
            logger.error("Unexpected error during command validation",
                        extra={"command": cmd[0] if cmd else "empty", "error": str(e)})
            raise ValidationError(f"Command validation failed: {e}") from e

    @classmethod
    def _validate_command_structure(cls, cmd: List[str]) -> None:
        """Validate basic command structure."""
        if not cmd:
            raise ValidationError("Empty command")

        if not isinstance(cmd, list):
            raise ValidationError("Command must be a list")

        if len(cmd) > cls.MAX_ARGUMENTS:
            raise ValidationError(f"Too many arguments: {len(cmd)} > {cls.MAX_ARGUMENTS}")

        # Check for empty arguments
        for i, arg in enumerate(cmd):
            if not isinstance(arg, str):
                raise ValidationError(f"Argument {i} is not a string: {type(arg)}")
            if len(arg) > cls.MAX_ARGUMENT_LENGTH:
                raise ValidationError(f"Argument {i} too long: {len(arg)} > {cls.MAX_ARGUMENT_LENGTH}")

    @classmethod
    def _validate_executable(cls, executable: str) -> None:
        """Validate the executable name and path."""
        if not executable:
            raise ValidationError("Empty executable name")

        # Resolve executable path
        try:
            if '/' in executable or '\\' in executable:
                # Path-based executable - validate it resolves to allowed location
                resolved_path = Path(executable).resolve()
                executable_name = resolved_path.name
            else:
                # Simple name - should be in PATH
                executable_name = executable

            # Check against allowlist
            if executable_name not in cls.ALLOWED_EXECUTABLES:
                raise ValidationError(f"Executable not allowed: {executable_name}")

            # Additional path safety for full paths
            if '/' in executable or '\\' in executable:
                # Only allow executables in standard system locations
                allowed_prefixes = [
                    '/usr/bin', '/usr/local/bin', '/bin', '/sbin',
                    '/usr/sbin', '/usr/local/sbin', '/opt',
                    str(Path.home() / '.local' / 'bin'),  # User local bins
                ]

                if not any(str(resolved_path).startswith(prefix) for prefix in allowed_prefixes):
                    raise ValidationError(f"Executable path not in allowed locations: {resolved_path}")

        except (OSError, ValueError) as e:
            raise ValidationError(f"Failed to resolve executable path: {e}")

    @classmethod
    def _validate_arguments(cls, cmd: List[str]) -> None:
        """Validate command arguments for security issues."""
        for i, arg in enumerate(cmd):
            # Check for dangerous characters
            for char in cls.DANGEROUS_CHARS:
                if char in arg:
                    raise ValidationError(f"Dangerous character in argument {i}: {repr(char)}")

            # Check for shell metacharacters (unless properly quoted)
            for char in cls.SHELL_METACHARACTERS:
                if char in arg:
                    # Allow if the argument is properly quoted or escaped
                    if not cls._is_safely_quoted(arg):
                        raise ValidationError(f"Shell metacharacter in argument {i}: {char}")

            # Check for URL encoding attempts
            if '%' in arg and cls._contains_url_encoding_attack(arg):
                raise ValidationError(f"URL encoding attack detected in argument {i}")

            # Check for Unicode normalization attacks
            cls._validate_unicode_safety(arg, i)

            # Check for file path traversal
            if cls._contains_path_traversal(arg):
                raise ValidationError(f"Path traversal detected in argument {i}")

    @classmethod
    def _validate_command_restrictions(cls, cmd: List[str]) -> None:
        """Validate command-specific restrictions."""
        executable = cmd[0]

        if executable in cls.RESTRICTED_ARGUMENTS:
            restricted_args = cls.RESTRICTED_ARGUMENTS[executable]
            for arg in cmd[1:]:
                if arg in restricted_args:
                    raise ValidationError(f"Restricted argument for {executable}: {arg}")

        # Additional command-specific validation
        if executable == "git":
            cls._validate_git_command(cmd)
        elif executable in ["rm", "rmdir"]:
            cls._validate_deletion_command(cmd)
        elif executable in ["chmod", "chown"]:
            cls._validate_permission_command(cmd)

    @classmethod
    def _validate_working_directory(cls, cwd: str, project_root: Optional[str] = None) -> None:
        """Validate working directory safety."""
        if not cwd:
            raise ValidationError("Empty working directory")

        # Check for dangerous characters
        for char in cls.DANGEROUS_CHARS:
            if char in cwd:
                raise ValidationError(f"Dangerous character in working directory: {repr(char)}")

        try:
            cwd_path = Path(cwd).resolve()

            # Ensure directory exists and is accessible
            if not cwd_path.exists():
                raise ValidationError(f"Working directory does not exist: {cwd}")
            if not cwd_path.is_dir():
                raise ValidationError(f"Working directory is not a directory: {cwd}")

            # If project root provided, ensure cwd is within it
            if project_root:
                project_path = Path(project_root).resolve()
                try:
                    cwd_path.relative_to(project_path)
                except ValueError:
                    raise ValidationError(f"Working directory outside project root: {cwd}")

        except (OSError, ValueError) as e:
            raise ValidationError(f"Invalid working directory: {e}")

    @classmethod
    def _validate_resource_limits(cls, cmd: List[str]) -> None:
        """Validate that command won't exceed resource limits."""
        config = get_system_config()

        # Check total command length
        total_length = sum(len(arg) for arg in cmd)
        if total_length > 10000:  # 10KB limit for total command
            raise ValidationError(f"Command too long: {total_length} characters")

        # Check for commands that might consume excessive resources
        resource_intensive_commands = ["find", "grep", "sort", "tar", "zip"]
        if cmd[0] in resource_intensive_commands:
            # Add timeout argument if not present
            if "-timeout" not in cmd and "--timeout" not in cmd:
                logger.warning("Resource-intensive command without timeout",
                             extra={"command": cmd[0]})

    @classmethod
    def _is_safely_quoted(cls, arg: str) -> bool:
        """Check if argument is safely quoted."""
        # Simple heuristic: if starts and ends with quotes and no unescaped quotes inside
        if ((arg.startswith('"') and arg.endswith('"')) or
            (arg.startswith("'") and arg.endswith("'"))):
            return True

        # Check if properly escaped
        try:
            # If shlex can parse it safely, it's probably OK
            parsed = shlex.split(arg)
            return len(parsed) == 1
        except ValueError:
            return False

    @classmethod
    def _contains_url_encoding_attack(cls, arg: str) -> bool:
        """Check for URL encoding attacks."""
        dangerous_encoded = [
            '%2e%2e',  # ..
            '%2f',     # /
            '%5c',     # \
            '%00',     # null byte
            '%0a',     # newline
            '%0d',     # carriage return
        ]

        arg_lower = arg.lower()
        return any(encoded in arg_lower for encoded in dangerous_encoded)

    @classmethod
    def _validate_unicode_safety(cls, arg: str, arg_index: int) -> None:
        """Validate Unicode safety to prevent normalization attacks."""
        try:
            normalized = unicodedata.normalize('NFC', arg)
            if normalized != arg:
                # Allow minor differences but flag major changes
                if abs(len(normalized) - len(arg)) > 5:
                    raise ValidationError(f"Unicode normalization attack in argument {arg_index}")
        except UnicodeError as e:
            raise ValidationError(f"Unicode error in argument {arg_index}: {e}")

    @classmethod
    def _contains_path_traversal(cls, arg: str) -> bool:
        """Check for path traversal attempts."""
        traversal_patterns = [
            '..',
            '/..',
            '..\\',
            '%2e%2e',
            '..%2f',
            '..%5c',
        ]

        return any(pattern in arg for pattern in traversal_patterns)

    @classmethod
    def _validate_git_command(cls, cmd: List[str]) -> None:
        """Validate git-specific command safety."""
        if len(cmd) < 2:
            return

        git_subcommand = cmd[1]

        # Dangerous git operations
        dangerous_operations = [
            "daemon", "upload-pack", "receive-pack",
            "http-backend", "shell"
        ]

        if git_subcommand in dangerous_operations:
            raise ValidationError(f"Dangerous git operation: {git_subcommand}")

        # Check for dangerous flags
        for arg in cmd[2:]:
            if arg.startswith('--git-dir=/') or arg.startswith('--work-tree=/'):
                raise ValidationError(f"Dangerous git flag: {arg}")

    @classmethod
    def _validate_deletion_command(cls, cmd: List[str]) -> None:
        """Validate deletion commands for safety."""
        # Check for dangerous deletion patterns
        dangerous_patterns = ['/', '/usr', '/etc', '/var', '/home', '*']

        for arg in cmd[1:]:
            if arg in dangerous_patterns:
                raise ValidationError(f"Dangerous deletion target: {arg}")
            if arg.startswith('/') and len(arg.split('/')) <= 3:
                raise ValidationError(f"Dangerous system path deletion: {arg}")

    @classmethod
    def _validate_permission_command(cls, cmd: List[str]) -> None:
        """Validate permission change commands."""
        if len(cmd) < 3:
            return

        mode = cmd[1]

        # Dangerous permission modes
        if mode in ['777', '666', '000']:
            raise ValidationError(f"Dangerous permission mode: {mode}")

        # Check for system file targets
        for target in cmd[2:]:
            if target.startswith('/etc') or target.startswith('/usr') or target.startswith('/bin'):
                raise ValidationError(f"System file permission change attempted: {target}")


def safe_subprocess_run(cmd: List[str], cwd: Optional[str] = None,
                       timeout: Optional[int] = None,
                       project_root: Optional[str] = None, **kwargs) -> subprocess.CompletedProcess:
    """Safely execute subprocess with comprehensive validation.
    
    Args:
        cmd: Command and arguments
        cwd: Working directory
        timeout: Timeout in seconds
        project_root: Project root for validation
        **kwargs: Additional subprocess.run arguments
        
    Returns:
        CompletedProcess result
        
    Raises:
        ValidationError: If validation fails
        ToolError: If execution fails
    """
    # Validate command before execution
    SubprocessValidator.validate_command(cmd, cwd, project_root)

    # Get system configuration
    config = get_system_config()
    if timeout is None:
        timeout = config.default_command_timeout

    # Set secure defaults
    secure_kwargs = {
        'shell': False,  # Never use shell
        'capture_output': True,
        'text': True,
        'timeout': timeout,
        'cwd': cwd,
        **kwargs
    }

    try:
        logger.debug("Executing validated subprocess",
                    extra={
                        "command": cmd[0],
                        "arg_count": len(cmd) - 1,
                        "cwd": cwd,
                        "timeout": timeout
                    })

        result = subprocess.run(cmd, **secure_kwargs)

        logger.debug("Subprocess completed",
                    extra={
                        "command": cmd[0],
                        "return_code": result.returncode,
                        "stdout_length": len(result.stdout) if result.stdout else 0,
                        "stderr_length": len(result.stderr) if result.stderr else 0
                    })

        return result

    except subprocess.TimeoutExpired as e:
        logger.error("Subprocess timeout",
                    extra={"command": cmd[0], "timeout": timeout})
        raise ToolError(f"Command '{cmd[0]}' timed out after {timeout}s") from e

    except (OSError, subprocess.SubprocessError) as e:
        logger.error("Subprocess execution failed",
                    extra={"command": cmd[0], "error": str(e)})
        raise ToolError(f"Failed to execute '{cmd[0]}': {e}") from e

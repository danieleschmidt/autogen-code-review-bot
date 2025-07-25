"""Test dependency version constraints."""

import re
import pytest
from pathlib import Path


class TestDependencyVersions:
    """Test that dependencies have proper version constraints."""

    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml should exist"

    def test_core_dependencies_have_version_constraints(self):
        """Test that core dependencies have version constraints."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()
        
        # Check that PyYAML has version constraint
        assert re.search(r'PyYAML>=\d+\.\d+\.\d+,<\d+\.\d+\.\d+', content), \
            "PyYAML should have version constraints"
        
        # Check that requests has version constraint
        assert re.search(r'requests>=\d+\.\d+\.\d+,<\d+\.\d+\.\d+', content), \
            "requests should have version constraints"

    def test_optional_dependencies_exist(self):
        """Test that optional dependencies are defined."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()
        
        # Check for monitoring dependencies
        assert "monitoring" in content, "monitoring optional dependencies should be defined"
        assert "psutil" in content, "psutil should be in monitoring dependencies"
        
        # Check for dev dependencies
        assert "dev" in content, "dev optional dependencies should be defined"

    def test_dev_dependencies_have_version_constraints(self):
        """Test that dev dependencies have version constraints."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()
        
        dev_tools = ["pre-commit", "ruff", "bandit", "pytest", "pytest-cov", "pytest-xdist", "detect-secrets"]
        
        for tool in dev_tools:
            pattern = f'{tool}>=\\d+\\.\\d+\\.\\d+,<\\d+\\.\\d+\\.\\d+'
            assert re.search(pattern, content), \
                f"{tool} should have version constraints in format >=x.y.z,<x.y.z"

    def test_monitoring_dependencies_have_version_constraints(self):
        """Test that monitoring dependencies have version constraints."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()
        
        # Check that psutil has version constraint
        assert re.search(r'psutil>=\d+\.\d+\.\d+,<\d+\.\d+\.\d+', content), \
            "psutil should have version constraints"

    def test_python_version_requirement(self):
        """Test that Python version requirement is specified."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()
        
        assert re.search(r'requires-python\s*=\s*">=3\.\d+"', content), \
            "Python version requirement should be specified"
cat > tests/test_setup.py << 'EOF'
"""Sanity tests to verify the environment is set up correctly."""

import sys
from pathlib import Path


def test_python_version():
    """We require Python 3.10 or higher."""
    assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version}"


def test_project_structure():
    """All expected folders exist."""
    root = Path(__file__).parent.parent
    expected = ["backend", "frontend", "ingestion", "rag",
                "evaluation", "tests", "data", "scripts", "docs"]
    for folder in expected:
        assert (root / folder).is_dir(), f"Missing folder: {folder}"


def test_env_example_exists():
    """We must ship an .env.example so contributors know what's needed."""
    root = Path(__file__).parent.parent
    assert (root / ".env.example").is_file()


def test_requirements_exists():
    """requirements.txt must exist for reproducible installs."""
    root = Path(__file__).parent.parent
    assert (root / "requirements.txt").is_file()
EOF
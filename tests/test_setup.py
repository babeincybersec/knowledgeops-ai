"""Sanity tests."""

import sys
from pathlib import Path


def test_python_version():
    assert sys.version_info >= (3, 10)


def test_project_structure():
    root = Path(__file__).parent.parent
    for folder in ["backend", "frontend", "ingestion", "rag", "evaluation", "tests", "data", "scripts", "docs"]:
        assert (root / folder).is_dir(), f"Missing: {folder}"


def test_env_example_exists():
    root = Path(__file__).parent.parent
    assert (root / ".env.example").is_file()


def test_requirements_exists():
    root = Path(__file__).parent.parent
    assert (root / "requirements.txt").is_file()

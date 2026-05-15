"""Helpers for loading sample email files in tests."""

from pathlib import Path

SAMPLES_DIR = Path(__file__).parent


def load_sample(rel_path: str) -> str:
    """Load a sample file by path relative to tests/samples/."""
    return (SAMPLES_DIR / rel_path).read_text(encoding="utf-8")


def load_sample_bytes(rel_path: str) -> bytes:
    """Load a sample file as bytes (for .eml files)."""
    return (SAMPLES_DIR / rel_path).read_bytes()

"""Shared pytest fixtures for the ticket router test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def raw_csv_path(repo_root: Path) -> Path:
    path = repo_root / "data" / "raw" / "tickets_raw.csv"
    if not path.exists():
        pytest.skip(f"Raw dataset not present at {path}")
    return path


@pytest.fixture(scope="session")
def processed_dir(repo_root: Path) -> Path:
    path = repo_root / "data" / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path
"""Shared pytest fixtures for the tsadbench test suite."""
import json
from pathlib import Path

import pytest

from tsadbench.io.misc import project_root

ROOT = project_root()
GOLDEN_FILE = ROOT / "golden_checksums.json"
DATA_DIR = ROOT / "data"


@pytest.fixture(scope="session")
def golden() -> dict:
    """The committed SMED bit-exact label baseline (see scripts/golden.py)."""
    if not GOLDEN_FILE.exists():
        pytest.skip(f"golden_checksums.json not found at {GOLDEN_FILE}")
    return json.loads(GOLDEN_FILE.read_text())["checksums"]


@pytest.fixture(scope="session")
def has_mission1_raw() -> bool:
    return (DATA_DIR / "ESA-Mission1" / "raw").is_dir()


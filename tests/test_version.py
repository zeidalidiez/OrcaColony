from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import orcacolony


ROOT = Path(__file__).parents[1]


def test_visible_version_matches_package_and_metadata() -> None:
    expected = (ROOT / "VERSION").read_text(encoding="utf-8").strip()

    assert orcacolony.__version__ == expected
    result = subprocess.run(
        [sys.executable, "scripts/check_version.py"],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"version sources agree: {expected}"

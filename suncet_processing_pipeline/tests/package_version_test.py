"""Tests for the independent software release version."""

from pathlib import Path

import suncet_processing_pipeline


def test_package_version_matches_repository_release_version():
    repository = Path(__file__).resolve().parents[2]

    assert suncet_processing_pipeline.__version__ == (
        repository / "version"
    ).read_text(encoding="utf-8").strip()

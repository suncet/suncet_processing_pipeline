"""SunCET data processing pipeline."""

from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path


def _package_version() -> str:
    """Return the release version in a checkout or installed distribution."""
    checkout_version = Path(__file__).resolve().parents[1] / "version"
    if checkout_version.is_file():
        return checkout_version.read_text(encoding="utf-8").strip()
    try:
        return distribution_version("suncet")
    except PackageNotFoundError:
        return "0+unknown"


__version__ = _package_version()

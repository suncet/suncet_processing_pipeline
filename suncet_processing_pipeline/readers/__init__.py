"""Readers for SunCET data products."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .telemetry import TelemetryReader

__all__ = ["TelemetryReader"]


def __getattr__(name: str):
    """Load reader implementations only when requested."""
    if name == "TelemetryReader":
        from .telemetry import TelemetryReader

        return TelemetryReader
    raise AttributeError(name)

"""Helpers for the SunCET CCSDS secondary spacecraft-time header."""

from __future__ import annotations

import math


FINE_MILLISECONDS_MIN = 0
FINE_MILLISECONDS_MAX = 999


def combine_spacecraft_time_seconds(
    coarse_seconds: int | float,
    fine_milliseconds: int | float = 0,
) -> float:
    """Return seconds since the SunCET epoch from coarse and fine fields.

    The transmitted 16-bit fine field is an integer count of milliseconds
    after the coarse whole second. Values outside one second are invalid.
    """

    coarse = float(coarse_seconds)
    fine = float(fine_milliseconds)
    if not math.isfinite(coarse) or coarse < 0:
        raise ValueError("coarse spacecraft time must be finite and nonnegative")
    if not math.isfinite(fine) or not fine.is_integer():
        raise ValueError("fine spacecraft time must be an integer millisecond count")
    if not FINE_MILLISECONDS_MIN <= fine <= FINE_MILLISECONDS_MAX:
        raise ValueError("fine spacecraft time must be between 0 and 999 milliseconds")
    return coarse + fine / 1000.0

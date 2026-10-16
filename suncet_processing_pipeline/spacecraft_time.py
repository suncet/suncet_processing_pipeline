"""Helpers for the SunCET CCSDS secondary spacecraft-time header."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone


FINE_MILLISECONDS_MIN = 0
FINE_MILLISECONDS_MAX = 999
J2000_UTC_EPOCH = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
POST_J2000_LEAP_SECOND_EFFECTIVE_UTC = (
    datetime(2006, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2009, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2012, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2015, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2017, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
)


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


def j2000_seconds_to_iso8601(seconds: int | float) -> str:
    """Convert SunCET J2000 seconds to the pipeline's UTC ISO 8601 form.

    SunCET timestamps count SI seconds from midnight UTC on 2000-01-01, so UTC
    display adds leap seconds that became effective after that epoch.
    """

    seconds_since_epoch = float(seconds)
    if not math.isfinite(seconds_since_epoch) or seconds_since_epoch < 0:
        raise ValueError("J2000 seconds must be finite and nonnegative")

    leap_count = 0
    while True:
        corrected_time = J2000_UTC_EPOCH + timedelta(
            seconds=seconds_since_epoch + leap_count
        )
        new_count = sum(
            1
            for leap_time in POST_J2000_LEAP_SECOND_EFFECTIVE_UTC
            if corrected_time >= leap_time
        )
        if new_count == leap_count:
            break
        leap_count = new_count

    timespec = "milliseconds" if corrected_time.microsecond else "seconds"
    return corrected_time.isoformat(timespec=timespec).replace("+00:00", "Z")

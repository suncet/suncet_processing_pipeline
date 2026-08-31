"""Machine-readable quality flags shared by Level 4 products."""

from __future__ import annotations

from enum import IntFlag, auto
from typing import Iterable


class QualityFlag(IntFlag):
    """Reasons a Level 4 sample or event needs qualification.

    Flags are powers of two so products can preserve both a compact integer
    mask and a readable list of names.  A zero mask means that no known quality
    issue was recorded; it is not a claim of scientific perfection.
    """

    NONE = 0
    ASSUMED_CADENCE = auto()
    ABSOLUTE_TIME_UNAVAILABLE = auto()
    SYNTHETIC_BYPASS = auto()
    LEVEL2_PSF_NOT_APPLIED = auto()
    LEVEL3_GEOMETRY_NOT_APPLIED = auto()
    FRONT_NOT_DETECTED = auto()
    LOW_FRONT_COVERAGE = auto()
    TRACK_GAP = auto()
    LOW_CONFIDENCE = auto()
    PARTIAL_FIELD_OF_VIEW = auto()
    KINEMATICS_NOT_FIT = auto()
    DERIVATIVE_ENDPOINT = auto()
    UNCERTAINTY_UNAVAILABLE = auto()
    INPUT_INTEGRITY_UNVERIFIED = auto()
    KINEMATIC_HEIGHT_OUTLIER = auto()
    TEMPORAL_EVIDENCE_UNSUPPORTED = auto()


def decode_quality_flags(mask: int | QualityFlag) -> tuple[str, ...]:
    """Return stable flag names for a bit mask."""

    flags = QualityFlag(mask)
    return tuple(
        member.name
        for member in QualityFlag
        if member is not QualityFlag.NONE and member & flags
    )


def encode_quality_flags(flags: Iterable[str | QualityFlag]) -> QualityFlag:
    """Build a mask from enum members or their names."""

    mask = QualityFlag.NONE
    for flag in flags:
        if isinstance(flag, str):
            try:
                flag = QualityFlag[flag]
            except KeyError as exc:
                raise ValueError(f"Unknown Level 4 quality flag: {flag}") from exc
        mask |= flag
    return mask

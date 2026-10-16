"""Lossless Level 1 solar-north orientation from APID 46 attitude telemetry."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from suncet_processing_pipeline.spacecraft_time import combine_spacecraft_time_seconds


APID46_TIME_COARSE = "ccsdsSecHeader2_sec_adcs_att_det"
APID46_TIME_FINE = "ccsdsSecHeader2_sub_adcs_att_det"
APID46_QUATERNION_FIELDS = tuple(
    f"adcs_att_det_q_body_wrt_eci{index}" for index in range(1, 5)
)
APID46_ATTITUDE_VALID = "adcs_att_det_attitude_vld"
APID46_RATE_VALID = "adcs_att_det_meas_rate_vld"
APID46_BODY_Z_RATE = "adcs_att_det_body_rate3"

# IAU solar north pole direction in the J2000 inertial frame.  APID 46 names its
# quaternion as body with respect to ECI (J2000), so no of-date frame conversion
# belongs between this vector and that quaternion.
SOLAR_NORTH_RA_J2000_DEG = 286.13
SOLAR_NORTH_DEC_J2000_DEG = 63.87

# A commanded equator roll is expected to exceed this by a wide margin.  This
# conservative provisional value also allows normal fine-pointing jitter.  It is
# deliberately an argument to the planner so an operations-approved value can
# replace it without changing the algorithm.
DEFAULT_ROLL_RATE_THRESHOLD_RAD_S = math.radians(0.1)


@dataclass(frozen=True)
class CoarseOrientationDecision:
    """Auditable decision for a lossless multiple-of-90-degree image rotation."""

    status: str
    quarter_turns_ccw: int | None
    valid_sample_count: int
    invalid_sample_count: int
    max_y_misalignment_deg: float | None = None
    max_attitude_change_deg: float | None = None
    max_abs_body_z_rate_rad_s: float | None = None
    start_attitude_age_s: float | None = None
    end_attitude_age_s: float | None = None
    reason: str = ""
    detector_y_convention: str = "native_array_right_is_detector_plus_y"

    @property
    def rotation_degrees_ccw(self) -> int | None:
        if self.quarter_turns_ccw is None:
            return None
        return 90 * self.quarter_turns_ccw

    @property
    def roll_in_progress(self) -> bool:
        return self.status == "roll_in_progress"

    def fits_keywords(self) -> dict[str, object]:
        """Return short FITS-safe provenance fields for the eventual writer."""
        values: dict[str, object] = {
            "ATTAPID": 46,
            "ROTSTAT": self.status,
            "ROLLFLAG": self.roll_in_progress,
            "NATYDIR": "RIGHT",
            "ATTSAMPS": self.valid_sample_count,
        }
        if self.quarter_turns_ccw is not None:
            values["ROTQTR"] = self.quarter_turns_ccw
            values["ROTDEG"] = self.rotation_degrees_ccw
        if self.max_y_misalignment_deg is not None:
            values["SOLNRES"] = self.max_y_misalignment_deg
        if self.max_abs_body_z_rate_rad_s is not None:
            values["MAXROLL"] = self.max_abs_body_z_rate_rad_s
        return values


def solar_north_j2000_unit_vector() -> np.ndarray:
    """Return the IAU solar north pole as a Cartesian J2000 unit vector."""
    right_ascension = math.radians(SOLAR_NORTH_RA_J2000_DEG)
    declination = math.radians(SOLAR_NORTH_DEC_J2000_DEG)
    return np.array(
        [
            math.cos(declination) * math.cos(right_ascension),
            math.cos(declination) * math.sin(right_ascension),
            math.sin(declination),
        ],
        dtype=float,
    )


def _flag_is_true(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)):
        return bool(value)
    return str(value).strip().casefold() in {
        "1",
        "true",
        "yes",
        "valid",
        "ena",
        "enabled",
    }


def _sample_time(sample: Mapping[str, object]) -> float | None:
    try:
        return combine_spacecraft_time_seconds(
            sample[APID46_TIME_COARSE], sample.get(APID46_TIME_FINE, 0)
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_apid46_attitude_samples(
    database_path: str | Path,
    exposure_start_j2000: float,
    exposure_end_j2000: float,
) -> list[dict[str, object]]:
    """Read the APID 46 samples bracketing and within an exposure from DuckDB."""
    import duckdb

    start = float(exposure_start_j2000)
    end = float(exposure_end_j2000)
    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError("invalid exposure interval")
    database = Path(database_path).expanduser()
    if not database.is_file():
        raise FileNotFoundError(database)

    with duckdb.connect(str(database), read_only=True) as connection:
        table_exists = connection.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            ["apid_0046"],
        ).fetchone()[0]
        if not table_exists:
            raise LookupError("DuckDB does not contain APID 46 table apid_0046")
        columns = {
            row[0] for row in connection.execute('DESCRIBE "apid_0046"').fetchall()
        }
        required = {
            APID46_TIME_COARSE,
            APID46_TIME_FINE,
            APID46_ATTITUDE_VALID,
            *APID46_QUATERNION_FIELDS,
        }
        missing = sorted(required - columns)
        if missing:
            raise LookupError(
                "APID 46 table is missing required field(s): " + ", ".join(missing)
            )

        time_expression = (
            f'CAST("{APID46_TIME_COARSE}" AS DOUBLE) + '
            f'CAST("{APID46_TIME_FINE}" AS DOUBLE) / 1000.0'
        )

        def query(where: str, parameters: list[float], order: str = ""):
            cursor = connection.execute(
                f'SELECT * FROM "apid_0046" WHERE {where} {order}', parameters
            )
            names = [item[0] for item in cursor.description]
            return [dict(zip(names, row)) for row in cursor.fetchall()]

        rows = query(
            f"{time_expression} <= ?",
            [start],
            f"ORDER BY {time_expression} DESC LIMIT 1",
        )
        rows.extend(
            query(
                f"{time_expression} > ? AND {time_expression} < ?",
                [start, end],
                f"ORDER BY {time_expression}",
            )
        )
        rows.extend(
            query(
                f"{time_expression} >= ?",
                [end],
                f"ORDER BY {time_expression} ASC LIMIT 1",
            )
        )

    unique: dict[tuple[float | None, tuple[float, ...]], dict[str, object]] = {}
    for row in rows:
        quaternion = tuple(float(row[field]) for field in APID46_QUATERNION_FIELDS)
        unique[(_sample_time(row), quaternion)] = row
    return sorted(unique.values(), key=lambda row: _sample_time(row) or -math.inf)


def _select_exposure_samples(
    samples: Iterable[Mapping[str, object]],
    start_j2000: float,
    end_j2000: float,
) -> tuple[list[Mapping[str, object]], float | None, float | None, bool]:
    """Select the samples bracketing and falling within one complete exposure."""
    if not math.isfinite(start_j2000) or not math.isfinite(end_j2000):
        raise ValueError("exposure times must be finite")
    if end_j2000 < start_j2000:
        raise ValueError("exposure end precedes exposure start")

    timed = sorted(
        (
            (timestamp, sample)
            for sample in samples
            if (timestamp := _sample_time(sample)) is not None
        ),
        key=lambda item: item[0],
    )
    before = [item for item in timed if item[0] <= start_j2000]
    during = [item for item in timed if start_j2000 < item[0] < end_j2000]
    after = [item for item in timed if item[0] >= end_j2000]
    start_item = before[-1] if before else None
    end_item = after[0] if after else None

    selected: list[tuple[float, Mapping[str, object]]] = []
    if start_item is not None:
        selected.append(start_item)
    selected.extend(during)
    if end_item is not None and (not selected or end_item[0] != selected[-1][0]):
        selected.append(end_item)
    return (
        [sample for _timestamp, sample in selected],
        start_j2000 - start_item[0] if start_item is not None else None,
        end_item[0] - end_j2000 if end_item is not None else None,
        start_item is not None and end_item is not None,
    )


def _solar_north_in_body_plane(
    sample: Mapping[str, object],
    solar_north_eci: np.ndarray,
) -> tuple[np.ndarray, float | None]:
    quaternion = np.asarray(
        [float(sample[field]) for field in APID46_QUATERNION_FIELDS],
        dtype=float,
    )
    if not np.all(np.isfinite(quaternion)) or np.linalg.norm(quaternion) <= 0:
        raise ValueError("invalid APID 46 attitude quaternion")
    quaternion /= np.linalg.norm(quaternion)

    # Provisional CTDB convention: q1,q2,q3,q4 are x,y,z,w and the quaternion
    # rotates J2000 ECI vectors into the spacecraft body frame.  This is isolated
    # here so a contrary FSW confirmation requires one localized change.
    north_body = Rotation.from_quat(quaternion).apply(solar_north_eci)
    north_xy = np.asarray(north_body[:2], dtype=float)
    norm = np.linalg.norm(north_xy)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("solar north projects degenerately onto the detector plane")
    north_xy /= norm

    body_z_rate = None
    if _flag_is_true(sample.get(APID46_RATE_VALID)):
        try:
            candidate = abs(float(sample[APID46_BODY_Z_RATE]))
            if math.isfinite(candidate):
                body_z_rate = candidate
        except (KeyError, TypeError, ValueError):
            pass
    return north_xy, body_z_rate


def plan_coarse_solar_north_rotation(
    attitude_samples: Iterable[Mapping[str, object]],
    *,
    exposure_start_j2000: float | None = None,
    exposure_end_j2000: float | None = None,
    solar_north_eci: np.ndarray | None = None,
    body_y_alignment_tolerance_deg: float = 20.0,
    max_attitude_change_deg: float = 10.0,
    roll_rate_threshold_rad_s: float | None = DEFAULT_ROLL_RATE_THRESHOLD_RAD_S,
) -> CoarseOrientationDecision:
    """Choose a lossless rotation or fail closed when the exposure spans a roll.

    The Level 0.5 image's long native array dimension is detector X.  We
    provisionally designate native array-right as detector +Y; flight imagery
    can reverse this single convention if it proves to put solar south upward.
    Stable mission attitudes align either +Y or -Y with solar north, so samples
    nearer body +/-X are treated as roll-in-progress rather than as valid 0/180
    degree science orientations.
    """
    samples = list(attitude_samples)
    start_age = None
    end_age = None
    if (exposure_start_j2000 is None) != (exposure_end_j2000 is None):
        raise ValueError("provide both exposure start and end, or neither")
    if exposure_start_j2000 is not None and exposure_end_j2000 is not None:
        samples, start_age, end_age, coverage_complete = _select_exposure_samples(
            samples, float(exposure_start_j2000), float(exposure_end_j2000)
        )
        if not coverage_complete:
            return CoarseOrientationDecision(
                status="insufficient_attitude_coverage",
                quarter_turns_ccw=None,
                valid_sample_count=0,
                invalid_sample_count=0,
                start_attitude_age_s=start_age,
                end_attitude_age_s=end_age,
                reason="APID 46 does not bracket the complete exposure",
            )
    if not samples:
        return CoarseOrientationDecision(
            status="no_attitude_samples",
            quarter_turns_ccw=None,
            valid_sample_count=0,
            invalid_sample_count=0,
            start_attitude_age_s=start_age,
            end_attitude_age_s=end_age,
            reason="no APID 46 samples are associated with the exposure",
        )

    pole = np.asarray(
        solar_north_j2000_unit_vector() if solar_north_eci is None else solar_north_eci,
        dtype=float,
    )
    if pole.shape != (3,) or not np.all(np.isfinite(pole)) or np.linalg.norm(pole) <= 0:
        raise ValueError("solar_north_eci must be a finite three-vector")
    pole /= np.linalg.norm(pole)

    directions: list[np.ndarray] = []
    rates: list[float] = []
    invalid = 0
    for sample in samples:
        if not _flag_is_true(sample.get(APID46_ATTITUDE_VALID)):
            invalid += 1
            continue
        try:
            direction, rate = _solar_north_in_body_plane(sample, pole)
        except (KeyError, TypeError, ValueError):
            invalid += 1
            continue
        directions.append(direction)
        if rate is not None:
            rates.append(rate)

    if not directions:
        return CoarseOrientationDecision(
            status="no_valid_attitude",
            quarter_turns_ccw=None,
            valid_sample_count=0,
            invalid_sample_count=invalid,
            start_attitude_age_s=start_age,
            end_attitude_age_s=end_age,
            reason="associated APID 46 samples have no valid usable attitude",
        )
    if invalid:
        return CoarseOrientationDecision(
            status="invalid_attitude_coverage",
            quarter_turns_ccw=None,
            valid_sample_count=len(directions),
            invalid_sample_count=invalid,
            start_attitude_age_s=start_age,
            end_attitude_age_s=end_age,
            reason="at least one APID 46 sample covering the exposure is invalid",
        )

    y_misalignments = [
        math.degrees(math.atan2(abs(direction[0]), abs(direction[1])))
        for direction in directions
    ]
    max_y_misalignment = max(y_misalignments)
    first = directions[0]
    changes = [
        math.degrees(math.acos(float(np.clip(np.dot(first, item), -1.0, 1.0))))
        for item in directions[1:]
    ]
    max_change = max(changes, default=0.0)
    max_rate = max(rates) if rates else None
    y_signs = {1 if direction[1] >= 0 else -1 for direction in directions}

    roll_reasons: list[str] = []
    if max_y_misalignment > body_y_alignment_tolerance_deg:
        roll_reasons.append("solar north is not aligned with body +/-Y")
    if len(y_signs) > 1:
        roll_reasons.append("solar north crosses between body +Y and -Y")
    if max_change > max_attitude_change_deg:
        roll_reasons.append("attitude changes across the exposure")
    if (
        roll_rate_threshold_rad_s is not None
        and max_rate is not None
        and max_rate > roll_rate_threshold_rad_s
    ):
        roll_reasons.append("body Z roll rate exceeds the provisional threshold")
    if roll_reasons:
        return CoarseOrientationDecision(
            status="roll_in_progress",
            quarter_turns_ccw=None,
            valid_sample_count=len(directions),
            invalid_sample_count=0,
            max_y_misalignment_deg=max_y_misalignment,
            max_attitude_change_deg=max_change,
            max_abs_body_z_rate_rad_s=max_rate,
            start_attitude_age_s=start_age,
            end_attitude_age_s=end_age,
            reason="; ".join(roll_reasons),
        )

    # Native array-right is provisionally detector +Y.  np.rot90(..., 1)
    # moves array-right upward; np.rot90(..., 3) moves array-left upward.
    quarter_turns = 1 if next(iter(y_signs)) > 0 else 3
    return CoarseOrientationDecision(
        status="stable",
        quarter_turns_ccw=quarter_turns,
        valid_sample_count=len(directions),
        invalid_sample_count=0,
        max_y_misalignment_deg=max_y_misalignment,
        max_attitude_change_deg=max_change,
        max_abs_body_z_rate_rad_s=max_rate,
        start_attitude_age_s=start_age,
        end_attitude_age_s=end_age,
        reason="stable APID 46 attitude aligned with body +/-Y",
    )


def apply_coarse_solar_north_rotation(
    image: np.ndarray,
    decision: CoarseOrientationDecision,
) -> np.ndarray:
    """Apply only an approved quarter-turn decision; otherwise return unchanged data."""
    data = np.asanyarray(image)
    if data.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    if decision.status != "stable" or decision.quarter_turns_ccw is None:
        return data.copy()
    return np.rot90(data, decision.quarter_turns_ccw, axes=(-2, -1))

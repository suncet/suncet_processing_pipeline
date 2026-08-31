"""Lazy FITS-sequence input boundary for Level 4 CME tracking.

Production Level 4 consumes Level 3 images. Historical simulator images may
bypass Level 2 PSF deconvolution and Level 3 geometric correction only through
the explicitly labeled ``synthetic_bypass`` path. Arrays and FITS headers are
never modified at this boundary.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Sequence

from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np

from .manifest import (
    CadenceStatus,
    CorrectionState,
    InputSourceKind,
    ManifestTimeAxis,
    SequenceManifest,
    TimeAxisKind,
    UpstreamProcessing,
    read_manifest,
    sha256_file,
)


_SOURCE_INDEX_PATTERN = re.compile(r"_(\d+)\.fits$", re.IGNORECASE)
_PRODUCTION_LEVEL_VALUES = {"3", "3.0", "L3", "LEVEL3"}


class SequenceInputError(ValueError):
    """Raised when an image sequence is unsuitable for tracking."""


@dataclass(frozen=True)
class ValidationIssue:
    """Nonfatal input condition retained in sequence provenance."""

    code: str
    message: str
    path: Path | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "message": self.message,
            "path": str(self.path) if self.path is not None else None,
        }


def _unit_vector_yx(
    value: Sequence[float],
    *,
    name: str,
) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise SequenceInputError(f"{name} must contain two finite y/x components.")
    norm = float(np.linalg.norm(array))
    if norm <= 0:
        raise SequenceInputError(f"{name} may not be the zero vector.")
    normalized = array / norm
    return float(normalized[0]), float(normalized[1])


@dataclass(frozen=True)
class SequenceGeometry:
    """Reference geometry and WCS-derived solar cardinal directions.

    Position angle is mapped into array coordinates as::

        center_yx + radius * (
            cos(position_angle) * north_vector_yx
            + sin(position_angle) * east_vector_yx
        )

    Thus zero degrees is solar north and 90 degrees is solar east. The vector
    signs come from the FITS WCS rather than assumptions about image-display
    origin.
    """

    image_shape_yx: tuple[int, int]
    center_x_px: float
    center_y_px: float
    solar_radius_px: float
    north_vector_yx: tuple[float, float]
    east_vector_yx: tuple[float, float]
    orientation_source: str
    pixel_scales_arcsec_xy: tuple[float, float] | None = None
    pixel_scale_arcsec: float | None = None
    pixel_scale_anisotropy_fraction: float | None = None

    def __post_init__(self) -> None:
        height, width = self.image_shape_yx
        if height < 2 or width < 2:
            raise SequenceInputError("Images must be at least 2 by 2 pixels.")
        for name, value in (
            ("center_x_px", self.center_x_px),
            ("center_y_px", self.center_y_px),
            ("solar_radius_px", self.solar_radius_px),
        ):
            if not math.isfinite(float(value)):
                raise SequenceInputError(f"{name} must be finite.")
        if self.solar_radius_px <= 0:
            raise SequenceInputError("solar_radius_px must be greater than zero.")
        if not (-0.5 <= self.center_x_px <= width - 0.5):
            raise SequenceInputError("Solar center x is outside the image.")
        if not (-0.5 <= self.center_y_px <= height - 0.5):
            raise SequenceInputError("Solar center y is outside the image.")

        north = _unit_vector_yx(self.north_vector_yx, name="north_vector_yx")
        east = _unit_vector_yx(self.east_vector_yx, name="east_vector_yx")
        object.__setattr__(self, "north_vector_yx", north)
        object.__setattr__(self, "east_vector_yx", east)
        determinant = north[1] * east[0] - north[0] * east[1]
        if abs(determinant) < 1e-6:
            raise SequenceInputError(
                "north_vector_yx and east_vector_yx may not be parallel."
            )
        if abs(float(np.dot(north, east))) > 0.05:
            raise SequenceInputError(
                "north_vector_yx and east_vector_yx must be orthogonal."
            )
        if self.orientation_source not in {"fits_wcs", "explicit_override"}:
            raise SequenceInputError(
                "orientation_source must be 'fits_wcs' or 'explicit_override'."
            )

        if self.pixel_scales_arcsec_xy is None:
            if self.pixel_scale_arcsec is not None:
                raise SequenceInputError(
                    "pixel_scale_arcsec requires pixel_scales_arcsec_xy."
                )
        else:
            scales = tuple(float(value) for value in self.pixel_scales_arcsec_xy)
            if len(scales) != 2 or any(
                not math.isfinite(value) or value <= 0 for value in scales
            ):
                raise SequenceInputError(
                    "pixel_scales_arcsec_xy must contain two positive finite values."
                )
            object.__setattr__(self, "pixel_scales_arcsec_xy", scales)
            if self.pixel_scale_arcsec is None or (
                not math.isfinite(float(self.pixel_scale_arcsec))
                or self.pixel_scale_arcsec <= 0
            ):
                raise SequenceInputError(
                    "pixel_scale_arcsec must be positive when WCS scales are known."
                )
            if self.pixel_scale_anisotropy_fraction is None or (
                not math.isfinite(float(self.pixel_scale_anisotropy_fraction))
                or self.pixel_scale_anisotropy_fraction < 0
            ):
                raise SequenceInputError(
                    "pixel_scale_anisotropy_fraction must be finite and nonnegative."
                )

    @property
    def center_yx(self) -> tuple[float, float]:
        return self.center_y_px, self.center_x_px


@dataclass(frozen=True)
class FrameRecord:
    """Lazily readable metadata for one FITS frame."""

    path: Path
    ordinal: int
    frame_number: int
    source_index: int | None
    elapsed_seconds: float
    raw_date_obs: str | None
    declared_fits_level: str | None
    shape_yx: tuple[int, int]
    dtype: str
    expected_sha256: str | None = None
    verify_sha256_on_read: bool = False


@dataclass
class LoadedFrame:
    """One loaded frame with its unmodified data values and header."""

    record: FrameRecord
    data: np.ndarray
    header: fits.Header
    wcs: WCS


@dataclass(frozen=True)
class ImageSequence:
    """A validated image sequence whose arrays remain lazy until requested."""

    frames: tuple[FrameRecord, ...]
    elapsed_seconds: np.ndarray
    geometry: SequenceGeometry
    scenario_id: str
    source_kind: InputSourceKind
    time_axis: ManifestTimeAxis
    time_source: str
    observation_times_utc: tuple[str | None, ...]
    upstream_processing: UpstreamProcessing
    issues: tuple[ValidationIssue, ...] = ()
    manifest_path: Path | None = None
    manifest_sha256: str | None = None
    hash_verification_status: str = "not_applicable"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "issues", tuple(self.issues))
        elapsed = np.asarray(self.elapsed_seconds, dtype=np.float64).copy()
        elapsed.setflags(write=False)
        object.__setattr__(self, "elapsed_seconds", elapsed)

        if self.hash_verification_status not in {
            "not_applicable",
            "verified_at_load_and_read",
            "skipped_by_request",
        }:
            raise SequenceInputError("Unknown input hash-verification status.")
        if self.manifest_path is None and self.hash_verification_status != "not_applicable":
            raise SequenceInputError(
                "Manifest hash-verification status requires a manifest path."
            )
        if self.manifest_path is not None and self.manifest_sha256 is None:
            raise SequenceInputError("Manifest-backed input requires manifest_sha256.")

        count = len(self.frames)
        if count < 2:
            raise SequenceInputError("Tracking requires at least two FITS images.")
        if elapsed.shape != (count,):
            raise SequenceInputError(
                "elapsed_seconds must contain one value per image."
            )
        if not np.all(np.isfinite(elapsed)) or not np.all(np.diff(elapsed) > 0):
            raise SequenceInputError(
                "elapsed_seconds must be finite and strictly increasing."
            )
        if len(self.observation_times_utc) != count:
            raise SequenceInputError(
                "observation_times_utc must contain one value per image."
            )
        if any(frame.ordinal != index for index, frame in enumerate(self.frames)):
            raise SequenceInputError("Frame ordinal values must be contiguous.")
        if any(
            frame.elapsed_seconds != elapsed[index]
            for index, frame in enumerate(self.frames)
        ):
            raise SequenceInputError(
                "Frame and sequence elapsed-time coordinates disagree."
            )
        if any(
            frame.shape_yx != self.geometry.image_shape_yx for frame in self.frames
        ):
            raise SequenceInputError("Frame and sequence geometry shapes disagree.")

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(frame.path for frame in self.frames)

    @property
    def header_date_obs(self) -> tuple[str | None, ...]:
        return tuple(frame.raw_date_obs for frame in self.frames)

    @property
    def cadence_seconds(self) -> float | None:
        return self.time_axis.cadence_seconds

    @property
    def input_stage(self) -> str:
        """Compatibility alias; do not confuse the synthetic bypass with Level 3."""

        return self.source_kind.value

    def read_frame(self, index: int) -> LoadedFrame:
        """Load and copy one frame without materializing the sequence."""

        return read_frame(self.frames[index])

    def iter_frames(self) -> Iterator[LoadedFrame]:
        """Yield one copied frame at a time."""

        for frame in self.frames:
            yield read_frame(frame)

    def materialize(
        self,
        dtype: np.dtype | type | str | None = np.float32,
    ) -> np.ndarray:
        """Explicitly load the sequence as ``(time, y, x)``.

        Passing ``dtype=None`` preserves the first FITS array dtype. The caller
        must request this allocation explicitly; ``ImageSequence`` has no
        implicit ``data`` property.
        """

        output_dtype = np.dtype(self.frames[0].dtype) if dtype is None else np.dtype(dtype)
        output = np.empty(
            (self.frame_count, *self.geometry.image_shape_yx),
            dtype=output_dtype,
        )
        for index, loaded in enumerate(self.iter_frames()):
            if not np.any(np.isfinite(loaded.data)):
                raise SequenceInputError(
                    f"FITS frame contains no finite pixels: {loaded.record.path}"
                )
            output[index] = np.asarray(loaded.data, dtype=output_dtype)
        return output

    def provenance_dict(self) -> dict[str, object]:
        """Return JSON-compatible assumptions needed to reproduce the load."""

        return {
            "scenario_id": self.scenario_id,
            "source_kind": self.source_kind.value,
            "time_source": self.time_source,
            "time_axis": self.time_axis.to_dict(),
            "frame_count": self.frame_count,
            "image_shape_yx": list(self.geometry.image_shape_yx),
            "center_yx": list(self.geometry.center_yx),
            "solar_radius_px": self.geometry.solar_radius_px,
            "pixel_scales_arcsec_xy": (
                list(self.geometry.pixel_scales_arcsec_xy)
                if self.geometry.pixel_scales_arcsec_xy is not None
                else None
            ),
            "pixel_scale_arcsec": self.geometry.pixel_scale_arcsec,
            "pixel_scale_anisotropy_fraction": (
                self.geometry.pixel_scale_anisotropy_fraction
            ),
            "north_vector_yx": list(self.geometry.north_vector_yx),
            "east_vector_yx": list(self.geometry.east_vector_yx),
            "orientation_source": self.geometry.orientation_source,
            "upstream_processing": self.upstream_processing.to_dict(),
            "manifest_path": (
                str(self.manifest_path) if self.manifest_path is not None else None
            ),
            "manifest_sha256": self.manifest_sha256,
            "hash_verification_status": self.hash_verification_status,
            "issues": [issue.to_dict() for issue in self.issues],
            "frames": [
                {
                    "path": str(frame.path),
                    "ordinal": frame.ordinal,
                    "frame_number": frame.frame_number,
                    "source_index": frame.source_index,
                    "elapsed_seconds": frame.elapsed_seconds,
                    "raw_date_obs": frame.raw_date_obs,
                    "declared_fits_level": frame.declared_fits_level,
                    "expected_sha256": frame.expected_sha256,
                }
                for frame in self.frames
            ],
        }


def _normalize_source_kind(value: InputSourceKind | str) -> InputSourceKind:
    if value == "level3":
        value = InputSourceKind.PRODUCTION_LEVEL3
    try:
        return InputSourceKind(value)
    except (TypeError, ValueError) as exc:
        raise SequenceInputError(
            "source_kind must be 'synthetic_bypass' or 'production_level3'."
        ) from exc


def _wcs_from_header(header: fits.Header) -> WCS:
    try:
        wcs = WCS(header, naxis=2, relax=True, fix=False)
    except Exception as exc:
        raise SequenceInputError(f"FITS WCS could not be constructed: {exc}") from exc
    if not wcs.has_celestial:
        raise SequenceInputError("FITS header does not define a celestial WCS.")
    return wcs


def _wcs_scale_and_orientation(
    header: fits.Header,
) -> tuple[
    tuple[float, float],
    float,
    float,
    tuple[float, float],
    tuple[float, float],
]:
    wcs = _wcs_from_header(header)
    matrix = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise SequenceInputError("FITS WCS pixel-scale matrix is unusable.")
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise SequenceInputError("FITS WCS pixel-scale matrix is singular.") from exc

    ctype = [str(value).upper() for value in wcs.wcs.ctype]
    try:
        longitude_axis = next(
            index for index, name in enumerate(ctype) if name.startswith("HPLN")
        )
        latitude_axis = next(
            index for index, name in enumerate(ctype) if name.startswith("HPLT")
        )
    except StopIteration as exc:
        raise SequenceInputError(
            "Solar position angle requires HPLN/HPLT helioprojective WCS axes."
        ) from exc

    north_world = np.zeros(2, dtype=np.float64)
    north_world[latitude_axis] = 1.0
    # Helioprojective longitude is positive toward solar west, so solar east is
    # the negative HPLN direction.
    east_world = np.zeros(2, dtype=np.float64)
    east_world[longitude_axis] = -1.0
    north_xy = inverse @ north_world
    east_xy = inverse @ east_world
    north_yx = _unit_vector_yx(
        (north_xy[1], north_xy[0]), name="WCS-derived north vector"
    )
    east_yx = _unit_vector_yx(
        (east_xy[1], east_xy[0]), name="WCS-derived east vector"
    )

    # Astropy returns plane scales in degrees per pixel for this WCS.
    scales_xy = np.asarray(proj_plane_pixel_scales(wcs), dtype=np.float64) * 3600.0
    if scales_xy.shape != (2,) or not np.all(np.isfinite(scales_xy)) or np.any(
        scales_xy <= 0
    ):
        raise SequenceInputError("FITS WCS pixel scales are unusable.")
    mean_scale = float(np.mean(scales_xy))
    anisotropy = float((np.max(scales_xy) - np.min(scales_xy)) / mean_scale)
    return (
        (float(scales_xy[0]), float(scales_xy[1])),
        mean_scale,
        anisotropy,
        north_yx,
        east_yx,
    )


def geometry_from_header(
    header: fits.Header,
    shape_yx: tuple[int, int],
    *,
    center_x_px: float | None = None,
    center_y_px: float | None = None,
    solar_radius_px: float | None = None,
    pixel_scale_arcsec: float | None = None,
    north_vector_yx: Sequence[float] | None = None,
    east_vector_yx: Sequence[float] | None = None,
    maximum_scale_anisotropy_fraction: float = 0.10,
) -> SequenceGeometry:
    """Resolve reference geometry while retaining WCS cardinal directions."""

    height, width = shape_yx
    if center_x_px is None:
        if header.get("CRPIX1") is None:
            raise SequenceInputError(
                "CRPIX1 is missing; provide an explicit center_x_px override."
            )
        center_x_px = float(header["CRPIX1"]) - 1.0
    if center_y_px is None:
        if header.get("CRPIX2") is None:
            raise SequenceInputError(
                "CRPIX2 is missing; provide an explicit center_y_px override."
            )
        center_y_px = float(header["CRPIX2"]) - 1.0

    if (north_vector_yx is None) != (east_vector_yx is None):
        raise SequenceInputError(
            "north_vector_yx and east_vector_yx overrides must be supplied together."
        )

    wcs_error: SequenceInputError | None = None
    try:
        scales_xy, mean_scale, anisotropy, wcs_north, wcs_east = (
            _wcs_scale_and_orientation(header)
        )
    except SequenceInputError as exc:
        wcs_error = exc
        scales_xy = None
        mean_scale = None
        anisotropy = None
        wcs_north = None
        wcs_east = None

    if pixel_scale_arcsec is not None:
        mean_scale = float(pixel_scale_arcsec)
        if not math.isfinite(mean_scale) or mean_scale <= 0:
            raise SequenceInputError(
                "pixel_scale_arcsec override must be finite and greater than zero."
            )
        scales_xy = (mean_scale, mean_scale)
        anisotropy = 0.0

    if north_vector_yx is None:
        if wcs_north is None or wcs_east is None:
            assert wcs_error is not None
            raise SequenceInputError(
                "Solar orientation is unavailable; provide explicit north/east "
                f"vectors. WCS error: {wcs_error}"
            ) from wcs_error
        north = wcs_north
        east = wcs_east
        orientation_source = "fits_wcs"
    else:
        north = _unit_vector_yx(north_vector_yx, name="north_vector_yx")
        east = _unit_vector_yx(east_vector_yx, name="east_vector_yx")
        orientation_source = "explicit_override"

    if mean_scale is None:
        if solar_radius_px is None:
            assert wcs_error is not None
            raise SequenceInputError(
                "Pixel scale is unavailable; provide pixel_scale_arcsec or "
                f"solar_radius_px. WCS error: {wcs_error}"
            ) from wcs_error
    elif anisotropy is not None and anisotropy > maximum_scale_anisotropy_fraction:
        raise SequenceInputError(
            "Pixel-scale anisotropy exceeds the current polar tracker's limit: "
            f"{anisotropy:.3%} > {maximum_scale_anisotropy_fraction:.3%}."
        )

    if solar_radius_px is None:
        apparent_radius = header.get("RSUN")
        if apparent_radius is None:
            apparent_radius = header.get("RSUN_OBS")
        if apparent_radius is None or mean_scale is None:
            raise SequenceInputError(
                "Cannot infer solar radius in pixels; provide solar_radius_px."
            )
        solar_radius_px = float(apparent_radius) / mean_scale

    return SequenceGeometry(
        image_shape_yx=(height, width),
        center_x_px=float(center_x_px),
        center_y_px=float(center_y_px),
        solar_radius_px=float(solar_radius_px),
        pixel_scales_arcsec_xy=scales_xy,
        pixel_scale_arcsec=mean_scale,
        pixel_scale_anisotropy_fraction=anisotropy,
        north_vector_yx=north,
        east_vector_yx=east,
        orientation_source=orientation_source,
    )


def _natural_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    )


def discover_fits_files(
    directory: str | Path,
    pattern: str = "*.fits",
    *,
    recursive: bool = False,
) -> tuple[Path, ...]:
    """Discover a draft ordered list; review it before creating a manifest."""

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise SequenceInputError(f"Input directory does not exist: {root}")
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    paths = tuple(
        sorted(
            (path.resolve() for path in iterator if path.is_file()),
            key=_natural_key,
        )
    )
    if len(paths) < 2:
        raise SequenceInputError(
            f"Expected at least two FITS files matching {pattern!r} below {root}."
        )
    return paths


def _source_index(path: Path) -> int | None:
    match = _SOURCE_INDEX_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


def _validate_numbers(
    values: Sequence[int] | None,
    *,
    count: int,
    name: str,
    default: Sequence[int | None],
) -> tuple[int | None, ...]:
    if values is None:
        result = tuple(default)
    else:
        result = tuple(values)
    if len(result) != count:
        raise SequenceInputError(f"{name} must contain one value per FITS file.")
    if any(
        value is not None
        and (isinstance(value, bool) or not isinstance(value, int))
        for value in result
    ):
        raise SequenceInputError(f"{name} values must be integers or null.")
    return result


def _parse_header_times(
    date_obs_values: Sequence[str | None],
) -> tuple[np.ndarray, tuple[str, ...]]:
    if any(value is None or not str(value).strip() for value in date_obs_values):
        raise SequenceInputError("FITS DATE-OBS timestamps are missing.")
    text_values = tuple(str(value).strip() for value in date_obs_values)
    try:
        times = Time(text_values, format="isot", scale="utc")
    except (TypeError, ValueError) as exc:
        raise SequenceInputError(
            "FITS DATE-OBS timestamps could not be parsed."
        ) from exc
    unix_seconds = np.asarray(times.unix, dtype=np.float64)
    if not np.all(np.diff(unix_seconds) > 0):
        raise SequenceInputError(
            "FITS DATE-OBS timestamps must be strictly increasing."
        )
    return unix_seconds - unix_seconds[0], tuple(times.utc.isot.tolist())


def _fixed_elapsed(
    frame_numbers: Sequence[int], cadence_seconds: float
) -> np.ndarray:
    try:
        cadence = float(cadence_seconds)
    except (TypeError, ValueError) as exc:
        raise SequenceInputError("cadence_seconds must be numeric.") from exc
    if not math.isfinite(cadence) or cadence <= 0:
        raise SequenceInputError(
            "cadence_seconds must be finite and greater than zero."
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in frame_numbers
    ):
        raise SequenceInputError("frame_numbers must contain integers.")
    if any(
        right <= left for left, right in zip(frame_numbers, frame_numbers[1:])
    ):
        raise SequenceInputError("frame_numbers must be strictly increasing.")
    first = frame_numbers[0]
    return (
        np.asarray(frame_numbers, dtype=np.float64) - float(first)
    ) * cadence


def _production_level(header: fits.Header) -> bool:
    value = header.get("LEVEL")
    if value is None:
        return False
    cleaned = str(value).strip().upper()
    if cleaned in _PRODUCTION_LEVEL_VALUES:
        return True
    try:
        return float(cleaned) == 3.0
    except ValueError:
        return False


def _requires_non_memmap_image_read(header: fits.Header) -> bool:
    """Return whether Astropy must scale image values into allocated memory."""

    return (
        header.get("BSCALE", 1) != 1
        or header.get("BZERO", 0) != 0
        or header.get("BLANK") is not None
    )


def _open_image_hdul(path: Path) -> fits.HDUList:
    """Open a FITS image lazily when possible and safely when scaling is needed."""

    hdul = fits.open(
        path,
        mode="readonly",
        memmap=True,
        checksum=False,
        uint=True,
    )
    if _requires_non_memmap_image_read(hdul[0].header):
        hdul.close()
        hdul = fits.open(
            path,
            mode="readonly",
            memmap=False,
            checksum=False,
            uint=True,
        )
    return hdul


def _frame_header(path: Path) -> tuple[fits.Header, tuple[int, int], str]:
    try:
        with _open_image_hdul(path) as hdul:
            image = hdul[0].data
            if image is None or np.ndim(image) != 2:
                raise SequenceInputError(
                    f"Primary HDU must contain a 2-D image: {path}"
                )
            if not np.issubdtype(image.dtype, np.number):
                raise SequenceInputError(f"FITS image must be numeric: {path}")
            shape = tuple(int(value) for value in image.shape)
            dtype = image.dtype.str
            header = hdul[0].header.copy()
    except OSError as exc:
        raise SequenceInputError(f"Could not read FITS file {path}: {exc}") from exc
    _wcs_from_header(header)
    return header, shape, dtype


def _geometry_difference(
    reference: SequenceGeometry,
    candidate: SequenceGeometry,
) -> str | None:
    center_error = math.hypot(
        reference.center_x_px - candidate.center_x_px,
        reference.center_y_px - candidate.center_y_px,
    )
    radius_fraction = abs(
        reference.solar_radius_px - candidate.solar_radius_px
    ) / reference.solar_radius_px
    north_dot = float(
        np.clip(
            np.dot(reference.north_vector_yx, candidate.north_vector_yx), -1, 1
        )
    )
    east_dot = float(
        np.clip(
            np.dot(reference.east_vector_yx, candidate.east_vector_yx), -1, 1
        )
    )
    north_error_deg = math.degrees(math.acos(north_dot))
    east_error_deg = math.degrees(math.acos(east_dot))
    reference_handedness = math.copysign(
        1.0,
        reference.north_vector_yx[1] * reference.east_vector_yx[0]
        - reference.north_vector_yx[0] * reference.east_vector_yx[1],
    )
    candidate_handedness = math.copysign(
        1.0,
        candidate.north_vector_yx[1] * candidate.east_vector_yx[0]
        - candidate.north_vector_yx[0] * candidate.east_vector_yx[1],
    )
    parity_changed = reference_handedness != candidate_handedness
    if (
        center_error > 0.1
        or radius_fraction > 0.005
        or north_error_deg > 0.1
        or east_error_deg > 0.1
        or parity_changed
    ):
        return (
            f"center differs by {center_error:.3f} px, solar radius by "
            f"{radius_fraction:.3%}, north by {north_error_deg:.3f} deg, "
            f"east by {east_error_deg:.3f} deg, and parity_changed={parity_changed}"
        )
    return None


@contextmanager
def open_frame(frame: FrameRecord) -> Iterator[LoadedFrame]:
    """Open one frame whose data remain valid inside the context.

    Ordinary images remain memory mapped. Images with FITS scaling keywords are
    read into memory because Astropy cannot apply BSCALE/BZERO through a memmap.
    """

    if frame.verify_sha256_on_read:
        actual = sha256_file(frame.path)
        if actual != frame.expected_sha256:
            raise SequenceInputError(
                f"SHA-256 mismatch while reading manifest input {frame.path}: "
                f"expected {frame.expected_sha256}, got {actual}."
            )
    try:
        with _open_image_hdul(frame.path) as hdul:
            image = hdul[0].data
            if image is None or tuple(image.shape) != frame.shape_yx:
                raise SequenceInputError(
                    f"FITS frame changed after validation: {frame.path}"
                )
            header = hdul[0].header.copy()
            yield LoadedFrame(
                record=frame,
                data=image,
                header=header,
                wcs=_wcs_from_header(header),
            )
    except OSError as exc:
        raise SequenceInputError(
            f"Could not read FITS file {frame.path}: {exc}"
        ) from exc


def read_frame(frame: FrameRecord) -> LoadedFrame:
    """Copy one FITS array so it remains valid after the file is closed."""

    with open_frame(frame) as loaded:
        return LoadedFrame(
            record=frame,
            data=np.array(loaded.data, copy=True),
            header=loaded.header.copy(),
            wcs=loaded.wcs.deepcopy(),
        )


def load_fits_sequence(
    paths: Sequence[str | Path],
    *,
    scenario_id: str,
    source_kind: InputSourceKind | str | None = None,
    input_stage: InputSourceKind | str | None = None,
    cadence_seconds: float | None = None,
    cadence_status: CadenceStatus | str = CadenceStatus.ASSUMED,
    frame_numbers: Sequence[int] | None = None,
    source_indices: Sequence[int | None] | None = None,
    upstream_processing: UpstreamProcessing | None = None,
    center_x_px: float | None = None,
    center_y_px: float | None = None,
    solar_radius_px: float | None = None,
    pixel_scale_arcsec: float | None = None,
    north_vector_yx: Sequence[float] | None = None,
    east_vector_yx: Sequence[float] | None = None,
    allow_inconsistent_geometry: bool = False,
    maximum_scale_anisotropy_fraction: float = 0.10,
    _expected_hashes: Sequence[str | None] | None = None,
    _verify_hashes_on_read: bool = False,
    _hash_verification_status: str = "not_applicable",
    _time_source: str | None = None,
    _manifest_path: Path | None = None,
    _manifest_sha256: str | None = None,
) -> ImageSequence:
    """Validate an explicitly ordered sequence without loading all image arrays."""

    if source_kind is None and input_stage is None:
        raise SequenceInputError("source_kind is required.")
    if source_kind is not None and input_stage is not None:
        raise SequenceInputError("Use source_kind or input_stage, not both.")
    normalized_source = _normalize_source_kind(
        source_kind if source_kind is not None else input_stage  # type: ignore[arg-type]
    )
    if normalized_source == InputSourceKind.PRODUCTION_LEVEL3 and allow_inconsistent_geometry:
        raise SequenceInputError(
            "Production Level 3 may not allow inconsistent sequence geometry."
        )
    if not scenario_id.strip():
        raise SequenceInputError("scenario_id must be non-empty.")

    resolved_paths = tuple(Path(path).expanduser().resolve() for path in paths)
    if len(resolved_paths) < 2:
        raise SequenceInputError("Tracking requires at least two FITS images.")
    if len(set(resolved_paths)) != len(resolved_paths):
        raise SequenceInputError("Input FITS paths must be unique.")
    for path in resolved_paths:
        if not path.is_file():
            raise SequenceInputError(f"Input FITS file does not exist: {path}")

    count = len(resolved_paths)
    direct_frame_numbers = _validate_numbers(
        frame_numbers,
        count=count,
        name="frame_numbers",
        default=tuple(range(count)),
    )
    if any(value is None for value in direct_frame_numbers):
        raise SequenceInputError("frame_numbers may not contain null values.")
    numbers = tuple(int(value) for value in direct_frame_numbers)
    indices = _validate_numbers(
        source_indices,
        count=count,
        name="source_indices",
        default=tuple(_source_index(path) for path in resolved_paths),
    )
    expected_hashes = (
        tuple(_expected_hashes) if _expected_hashes is not None else (None,) * count
    )
    if len(expected_hashes) != count:
        raise SequenceInputError("Expected hashes must contain one value per file.")

    if normalized_source == InputSourceKind.SYNTHETIC_BYPASS:
        corrections = upstream_processing or UpstreamProcessing(
            level2_psf_deconvolution=CorrectionState.NOT_APPLIED,
            level3_geometric_correction=CorrectionState.NOT_APPLIED,
        )
        if cadence_seconds is None:
            elapsed = np.empty(count, dtype=np.float64)
            observation_times: tuple[str | None, ...] = ()
            time_axis = ManifestTimeAxis(
                kind=TimeAxisKind.FITS_HEADERS,
                absolute_time_valid=True,
            )
            time_source = _time_source or "fits_headers"
        else:
            try:
                normalized_status = CadenceStatus(cadence_status)
            except (TypeError, ValueError) as exc:
                raise SequenceInputError(
                    "cadence_status must be 'assumed' or 'verified'."
                ) from exc
            elapsed = _fixed_elapsed(numbers, cadence_seconds)
            time_axis = ManifestTimeAxis(
                kind=TimeAxisKind.FIXED_CADENCE,
                cadence_seconds=cadence_seconds,
                cadence_status=normalized_status,
                absolute_time_valid=False,
            )
            observation_times = (None,) * count
            time_source = _time_source or "explicit_fixed_cadence"
    else:
        if cadence_seconds is not None:
            raise SequenceInputError(
                "production_level3 may not override FITS timestamps."
            )
        elapsed = np.empty(count, dtype=np.float64)
        observation_times = ()
        time_axis = ManifestTimeAxis(
            kind=TimeAxisKind.FITS_HEADERS,
            absolute_time_valid=True,
        )
        corrections = upstream_processing or UpstreamProcessing(
            level2_psf_deconvolution=CorrectionState.APPLIED,
            level3_geometric_correction=CorrectionState.APPLIED,
        )
        if (
            corrections.level2_psf_deconvolution != CorrectionState.APPLIED
            or corrections.level3_geometric_correction != CorrectionState.APPLIED
        ):
            raise SequenceInputError(
                "production_level3 requires applied Level 2 PSF deconvolution "
                "and Level 3 geometric correction."
            )
        time_source = _time_source or "fits_headers"

    issues: list[ValidationIssue] = []
    headers: list[fits.Header] = []
    shapes: list[tuple[int, int]] = []
    dtypes: list[str] = []
    for path in resolved_paths:
        header, shape, dtype = _frame_header(path)
        headers.append(header)
        shapes.append(shape)
        dtypes.append(dtype)
        datasum = header.get("DATASUM")
        if datasum is not None and not str(datasum).strip().isdigit():
            issues.append(
                ValidationIssue(
                    code="INVALID_FITS_DATASUM",
                    message=(
                        "Historical DATASUM card is not a valid FITS data sum; "
                        "use the manifest/provenance SHA-256 instead."
                    ),
                    path=path,
                )
            )
        if normalized_source == InputSourceKind.PRODUCTION_LEVEL3 and not (
            _production_level(header)
        ):
            raise SequenceInputError(
                f"production_level3 input lacks LEVEL=3 declaration: {path}"
            )

    first_shape = shapes[0]
    for path, shape in zip(resolved_paths[1:], shapes[1:], strict=True):
        if shape != first_shape:
            raise SequenceInputError(
                f"Image shape {shape} in {path} differs from {first_shape}."
            )

    reference_geometry = geometry_from_header(
        headers[0],
        first_shape,
        center_x_px=center_x_px,
        center_y_px=center_y_px,
        solar_radius_px=solar_radius_px,
        pixel_scale_arcsec=pixel_scale_arcsec,
        north_vector_yx=north_vector_yx,
        east_vector_yx=east_vector_yx,
        maximum_scale_anisotropy_fraction=maximum_scale_anisotropy_fraction,
    )
    if (
        reference_geometry.pixel_scale_anisotropy_fraction is not None
        and reference_geometry.pixel_scale_anisotropy_fraction > 0.01
    ):
        issues.append(
            ValidationIssue(
                code="PIXEL_SCALE_ANISOTROPY",
                message=(
                    "Reference WCS pixel scales differ by "
                    f"{reference_geometry.pixel_scale_anisotropy_fraction:.3%}."
                ),
                path=resolved_paths[0],
            )
        )

    for path, header in zip(resolved_paths[1:], headers[1:], strict=True):
        candidate = geometry_from_header(
            header,
            first_shape,
            center_x_px=center_x_px,
            center_y_px=center_y_px,
            solar_radius_px=solar_radius_px,
            pixel_scale_arcsec=pixel_scale_arcsec,
            north_vector_yx=north_vector_yx,
            east_vector_yx=east_vector_yx,
            maximum_scale_anisotropy_fraction=maximum_scale_anisotropy_fraction,
        )
        difference = _geometry_difference(reference_geometry, candidate)
        if difference is not None:
            issue = ValidationIssue(
                code="INCONSISTENT_SEQUENCE_GEOMETRY",
                message=difference,
                path=path,
            )
            if not allow_inconsistent_geometry:
                raise SequenceInputError(
                    "Sequence geometry is inconsistent; use reviewed overrides or "
                    "explicitly allow the synthetic inconsistency. First issue: "
                    + difference
                )
            issues.append(issue)

    raw_date_obs = tuple(
        str(header["DATE-OBS"]).strip() if header.get("DATE-OBS") else None
        for header in headers
    )
    if time_axis.kind == TimeAxisKind.FITS_HEADERS:
        elapsed, parsed_times = _parse_header_times(raw_date_obs)
        observation_times = parsed_times
    else:
        if len(set(raw_date_obs)) < len(raw_date_obs):
            issues.append(
                ValidationIssue(
                    code="FROZEN_OR_DUPLICATE_HEADER_TIME",
                    message=(
                        "FITS DATE-OBS values are not used; elapsed time comes "
                        "from the explicit fixed-cadence axis."
                    ),
                )
            )
        if time_axis.cadence_status == CadenceStatus.ASSUMED:
            issues.append(
                ValidationIssue(
                    code="ASSUMED_CADENCE",
                    message=(
                        f"Kinematics currently assume {time_axis.cadence_seconds:g} "
                        "seconds per frame-number interval."
                    ),
                )
            )

    nonnull_indices = [value for value in indices if value is not None]
    if len(nonnull_indices) == len(indices) and len(nonnull_indices) > 2:
        differences = [
            right - left
            for left, right in zip(nonnull_indices, nonnull_indices[1:])
        ]
        nominal_step = Counter(differences).most_common(1)[0][0]
        if any(step != nominal_step for step in differences):
            issues.append(
                ValidationIssue(
                    code="NONUNIFORM_SOURCE_INDEX_STEP",
                    message=(
                        "Filename/source indices are not uniform. They remain "
                        "provenance only; the declared time axis controls elapsed "
                        "time."
                    ),
                )
            )

    frame_records = tuple(
        FrameRecord(
            path=path,
            ordinal=ordinal,
            frame_number=numbers[ordinal],
            source_index=indices[ordinal],
            elapsed_seconds=float(elapsed[ordinal]),
            raw_date_obs=raw_date_obs[ordinal],
            declared_fits_level=(
                str(headers[ordinal]["LEVEL"]).strip()
                if headers[ordinal].get("LEVEL") is not None
                else None
            ),
            shape_yx=shapes[ordinal],
            dtype=dtypes[ordinal],
            expected_sha256=expected_hashes[ordinal],
            verify_sha256_on_read=_verify_hashes_on_read,
        )
        for ordinal, path in enumerate(resolved_paths)
    )

    return ImageSequence(
        frames=frame_records,
        elapsed_seconds=elapsed,
        geometry=reference_geometry,
        scenario_id=scenario_id,
        source_kind=normalized_source,
        time_axis=time_axis,
        time_source=time_source,
        observation_times_utc=observation_times,
        upstream_processing=corrections,
        issues=tuple(issues),
        manifest_path=_manifest_path,
        manifest_sha256=_manifest_sha256,
        hash_verification_status=_hash_verification_status,
    )


def load_sequence_from_manifest(
    manifest_path: str | Path,
    *,
    data_root: str | Path | None = None,
    verify_hashes: bool = True,
    center_x_px: float | None = None,
    center_y_px: float | None = None,
    solar_radius_px: float | None = None,
    pixel_scale_arcsec: float | None = None,
    north_vector_yx: Sequence[float] | None = None,
    east_vector_yx: Sequence[float] | None = None,
    allow_inconsistent_geometry: bool = False,
    maximum_scale_anisotropy_fraction: float = 0.10,
) -> tuple[ImageSequence, SequenceManifest]:
    """Load a sequence whose selection, timing, and hashes were reviewed."""

    path = Path(manifest_path).expanduser().resolve()
    manifest_sha256 = sha256_file(path)
    manifest = read_manifest(path)
    resolved = manifest.resolve_files(path.parent, data_root=data_root)
    for frame, frame_path in zip(manifest.frames, resolved, strict=True):
        if not frame_path.is_file():
            raise SequenceInputError(f"Manifest FITS file does not exist: {frame_path}")
        if verify_hashes:
            actual = sha256_file(frame_path)
            if actual != frame.sha256:
                raise SequenceInputError(
                    f"SHA-256 mismatch for manifest input {frame_path}: "
                    f"expected {frame.sha256}, got {actual}."
                )

    sequence = load_fits_sequence(
        resolved,
        scenario_id=manifest.scenario_id,
        source_kind=manifest.source_kind,
        cadence_seconds=manifest.time_axis.cadence_seconds,
        cadence_status=(
            manifest.time_axis.cadence_status or CadenceStatus.ASSUMED
        ),
        frame_numbers=tuple(frame.frame_number for frame in manifest.frames),
        source_indices=tuple(frame.source_index for frame in manifest.frames),
        upstream_processing=manifest.upstream_processing,
        center_x_px=center_x_px,
        center_y_px=center_y_px,
        solar_radius_px=solar_radius_px,
        pixel_scale_arcsec=pixel_scale_arcsec,
        north_vector_yx=north_vector_yx,
        east_vector_yx=east_vector_yx,
        allow_inconsistent_geometry=allow_inconsistent_geometry,
        maximum_scale_anisotropy_fraction=maximum_scale_anisotropy_fraction,
        _expected_hashes=tuple(frame.sha256 for frame in manifest.frames),
        _verify_hashes_on_read=verify_hashes,
        _hash_verification_status=(
            "verified_at_load_and_read" if verify_hashes else "skipped_by_request"
        ),
        _time_source=(
            "reviewed_manifest_fixed_cadence"
            if manifest.time_axis.kind == TimeAxisKind.FIXED_CADENCE
            else "reviewed_manifest_fits_headers"
        ),
        _manifest_path=path,
        _manifest_sha256=manifest_sha256,
    )
    return sequence, manifest


def load_synthetic_sequence(
    ordered_paths: Sequence[str | Path] | None = None,
    *,
    scenario_id: str | None = None,
    assumed_cadence_seconds: float | None = None,
    frame_numbers: Sequence[int] | None = None,
    source_indices: Sequence[int | None] | None = None,
    manifest_path: str | Path | None = None,
    data_root: str | Path | None = None,
    verify_hashes: bool = True,
    **geometry_options: object,
) -> ImageSequence:
    """Load synthetic images using FITS time or an explicit historical cadence."""

    if manifest_path is not None:
        if (
            ordered_paths is not None
            or scenario_id is not None
            or assumed_cadence_seconds is not None
            or frame_numbers is not None
            or source_indices is not None
        ):
            raise SequenceInputError(
                "Manifest mode may not be combined with direct paths, scenario, "
                "cadence, or frame-number overrides."
            )
        sequence, _manifest = load_sequence_from_manifest(
            manifest_path,
            data_root=data_root,
            verify_hashes=verify_hashes,
            **geometry_options,
        )
        if sequence.source_kind != InputSourceKind.SYNTHETIC_BYPASS:
            raise SequenceInputError(
                "Synthetic loader received a production_level3 manifest."
            )
        return sequence

    if ordered_paths is None or scenario_id is None:
        raise SequenceInputError(
            "Direct synthetic loading requires ordered_paths and scenario_id."
        )
    return load_fits_sequence(
        ordered_paths,
        scenario_id=scenario_id,
        source_kind=InputSourceKind.SYNTHETIC_BYPASS,
        cadence_seconds=assumed_cadence_seconds,
        cadence_status=CadenceStatus.ASSUMED,
        frame_numbers=frame_numbers,
        source_indices=source_indices,
        **geometry_options,
    )


def load_level3_sequence(
    ordered_paths: Sequence[str | Path],
    *,
    scenario_id: str,
    **geometry_options: object,
) -> ImageSequence:
    """Load production Level 3 images using only their real FITS timestamps."""

    return load_fits_sequence(
        ordered_paths,
        scenario_id=scenario_id,
        source_kind=InputSourceKind.PRODUCTION_LEVEL3,
        **geometry_options,
    )

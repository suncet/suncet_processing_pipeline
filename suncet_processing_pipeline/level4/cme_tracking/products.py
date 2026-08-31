"""Inspectable provisional Level 4 CME product writers.

Astropy ECSV is the authoritative research table format: it is readable as
plain text while preserving column units and table metadata.  The small JSON
summary and PNGs are derived views of the same arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence
import uuid

from astropy.table import Table
import astropy.units as u
import numpy as np

from ..common.quality import QualityFlag, decode_quality_flags


class ProductValidationError(ValueError):
    """Raised when arrays cannot form a self-consistent product."""


_EVENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class TrackProduct:
    """One event-level height and kinematics sample per input frame."""

    event_id: str
    frame_number: np.ndarray
    elapsed_s: np.ndarray
    time_utc: tuple[str | None, ...]
    height_raw_rsun: np.ndarray
    height_raw_sigma_rsun: np.ndarray
    height_fit_rsun: np.ndarray
    height_fit_sigma_rsun: np.ndarray
    speed_fit_km_s: np.ndarray
    speed_sigma_km_s: np.ndarray
    acceleration_fit_m_s2: np.ndarray
    acceleration_sigma_m_s2: np.ndarray
    position_angle_deg: np.ndarray
    angular_width_deg: np.ndarray
    front_coverage_fraction: np.ndarray
    confidence: np.ndarray
    quality_mask: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _EVENT_ID_PATTERN.fullmatch(self.event_id):
            raise ProductValidationError(
                "event_id must be one portable path component."
            )
        count = len(self.elapsed_s)
        if count < 2:
            raise ProductValidationError("A CME track requires at least two rows.")
        one_dimensional = (
            self.frame_number,
            self.elapsed_s,
            self.height_raw_rsun,
            self.height_raw_sigma_rsun,
            self.height_fit_rsun,
            self.height_fit_sigma_rsun,
            self.speed_fit_km_s,
            self.speed_sigma_km_s,
            self.acceleration_fit_m_s2,
            self.acceleration_sigma_m_s2,
            self.position_angle_deg,
            self.angular_width_deg,
            self.front_coverage_fraction,
            self.confidence,
            self.quality_mask,
        )
        if any(np.asarray(values).shape != (count,) for values in one_dimensional):
            raise ProductValidationError(
                "Every track field must be a one-dimensional array of equal length."
            )
        if len(self.time_utc) != count:
            raise ProductValidationError("time_utc must contain one value per row.")
        elapsed = np.asarray(self.elapsed_s, dtype=np.float64)
        if not np.all(np.isfinite(elapsed)) or not np.all(np.diff(elapsed) > 0):
            raise ProductValidationError(
                "elapsed_s must be finite and strictly increasing."
            )
        coverage = np.asarray(self.front_coverage_fraction, dtype=np.float64)
        confidence = np.asarray(self.confidence, dtype=np.float64)
        if np.any((coverage < 0) | (coverage > 1)):
            raise ProductValidationError("front coverage must lie between zero and one.")
        if np.any((confidence < 0) | (confidence > 1)):
            raise ProductValidationError("confidence must lie between zero and one.")


@dataclass(frozen=True)
class FrontSamplesProduct:
    """Long-form retained front samples over time and position angle."""

    event_id: str
    frame_number: np.ndarray
    elapsed_s: np.ndarray
    position_angle_deg: np.ndarray
    radius_rsun: np.ndarray
    radius_sigma_rsun: np.ndarray
    score: np.ndarray
    accepted: np.ndarray
    quality_mask: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _EVENT_ID_PATTERN.fullmatch(self.event_id):
            raise ProductValidationError(
                "event_id must be one portable path component."
            )
        size = np.asarray(self.elapsed_s).size
        arrays = (
            self.frame_number,
            self.elapsed_s,
            self.position_angle_deg,
            self.radius_rsun,
            self.radius_sigma_rsun,
            self.score,
            self.accepted,
            self.quality_mask,
        )
        if any(np.asarray(values).ndim != 1 for values in arrays):
            raise ProductValidationError("Front-sample fields must be one-dimensional.")
        if any(np.asarray(values).size != size for values in arrays):
            raise ProductValidationError("Front-sample fields have different lengths.")


@dataclass(frozen=True)
class FrontOverlayFrame:
    """One raw image and retained polar samples for visual verification."""

    image: np.ndarray
    frame_number: int
    elapsed_s: float
    radius_px: np.ndarray
    position_angle_deg: np.ndarray
    headline_height_rsun: float


@dataclass(frozen=True)
class FrontOverlayProduct:
    """Small verification mosaic derived from the authoritative front samples."""

    frames: tuple[FrontOverlayFrame, ...]
    center_yx: tuple[float, float]
    north_vector_yx: tuple[float, float]
    east_vector_yx: tuple[float, float]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ProductValidationError("An overlay mosaic needs at least one frame.")
        shape = np.asarray(self.frames[0].image).shape
        if len(shape) != 2 or any(np.asarray(frame.image).shape != shape for frame in self.frames):
            raise ProductValidationError("Overlay images must share one 2-D shape.")


def _readable_flags(masks: Sequence[int]) -> list[str]:
    return ["|".join(decode_quality_flags(int(mask))) for mask in masks]


def track_table(product: TrackProduct) -> Table:
    """Convert a track product to its authoritative ECSV table."""

    raw = np.asarray(product.height_raw_rsun, dtype=np.float64)
    fitted = np.asarray(product.height_fit_rsun, dtype=np.float64)
    solar_radius_km = float(product.metadata.get("solar_radius_km", 695_700.0))
    table = Table()
    table["event_id"] = [product.event_id] * len(product.elapsed_s)
    table["frame_number"] = np.asarray(product.frame_number, dtype=np.int64)
    table["time_utc"] = [value or "" for value in product.time_utc]
    table["elapsed_s"] = np.asarray(product.elapsed_s, dtype=np.float64) * u.s
    table["height_raw_rsun"] = raw * u.R_sun
    table["height_raw_km"] = raw * solar_radius_km * u.km
    table["height_raw_sigma_rsun"] = (
        np.asarray(product.height_raw_sigma_rsun, dtype=np.float64) * u.R_sun
    )
    table["height_fit_rsun"] = fitted * u.R_sun
    table["height_fit_km"] = fitted * solar_radius_km * u.km
    table["height_fit_sigma_rsun"] = (
        np.asarray(product.height_fit_sigma_rsun, dtype=np.float64) * u.R_sun
    )
    table["speed_fit_km_s"] = (
        np.asarray(product.speed_fit_km_s, dtype=np.float64) * u.km / u.s
    )
    table["speed_sigma_km_s"] = (
        np.asarray(product.speed_sigma_km_s, dtype=np.float64) * u.km / u.s
    )
    table["acceleration_fit_m_s2"] = (
        np.asarray(product.acceleration_fit_m_s2, dtype=np.float64) * u.m / u.s**2
    )
    table["acceleration_sigma_m_s2"] = (
        np.asarray(product.acceleration_sigma_m_s2, dtype=np.float64) * u.m / u.s**2
    )
    table["position_angle_deg"] = (
        np.asarray(product.position_angle_deg, dtype=np.float64) * u.deg
    )
    table["angular_width_deg"] = (
        np.asarray(product.angular_width_deg, dtype=np.float64) * u.deg
    )
    table["front_coverage_fraction"] = np.asarray(
        product.front_coverage_fraction, dtype=np.float64
    )
    table["confidence"] = np.asarray(product.confidence, dtype=np.float64)
    table["quality_mask"] = np.asarray(product.quality_mask, dtype=np.uint32)
    table["quality_flags"] = _readable_flags(product.quality_mask)
    table.meta = {"schema": "suncet.cme_track", "schema_version": 1}
    table.meta.update(dict(product.metadata))
    return table


def front_samples_table(product: FrontSamplesProduct) -> Table:
    """Convert retained polar front samples to long-form ECSV."""

    table = Table()
    table["event_id"] = [product.event_id] * np.asarray(product.elapsed_s).size
    table["frame_number"] = np.asarray(product.frame_number, dtype=np.int64)
    table["elapsed_s"] = np.asarray(product.elapsed_s, dtype=np.float64) * u.s
    table["position_angle_deg"] = (
        np.asarray(product.position_angle_deg, dtype=np.float64) * u.deg
    )
    table["radius_rsun"] = np.asarray(product.radius_rsun, dtype=np.float64) * u.R_sun
    table["radius_sigma_rsun"] = (
        np.asarray(product.radius_sigma_rsun, dtype=np.float64) * u.R_sun
    )
    table["score"] = np.asarray(product.score, dtype=np.float64)
    table["accepted"] = np.asarray(product.accepted, dtype=bool)
    table["quality_mask"] = np.asarray(product.quality_mask, dtype=np.uint32)
    table["quality_flags"] = _readable_flags(product.quality_mask)
    table.meta = {"schema": "suncet.cme_front_samples", "schema_version": 1}
    table.meta.update(dict(product.metadata))
    return table


def _write_json_atomic(values: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(values, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return [_json_scalar(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    return value


def _zero_centered_detail_limits(values: np.ndarray) -> tuple[float, float]:
    """Return readable zero-centered limits for a fitted acceleration curve.

    Acceleration uncertainty can be orders of magnitude larger than the fitted
    curve during early method development.  Those uncertainties remain
    scientifically important, but using them to set the only y-axis makes the
    fitted variation visually disappear.  Detail views therefore derive their
    limits from the fitted curve and explicitly mark uncertainty that continues
    outside the view.
    """

    finite = np.abs(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return -1.0, 1.0
    magnitude = float(np.max(finite))
    if magnitude == 0.0:
        magnitude = 1.0
    padded = 1.12 * magnitude
    return -padded, padded


def _plot_uncertainty_band(
    axis: Any,
    elapsed_s: np.ndarray,
    values: np.ndarray,
    sigma: np.ndarray,
    *,
    color: str,
    alpha: float,
    label: str | None,
    y_limits: tuple[float, float] | None = None,
) -> bool:
    """Plot a 1-sigma band and mark where a detail view clips it."""

    elapsed = np.asarray(elapsed_s, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(sigma)
    if not np.any(valid):
        return False

    lower = values - sigma
    upper = values + sigma
    if y_limits is None:
        display_lower = lower
        display_upper = upper
        lower_clipped = np.zeros(valid.shape, dtype=bool)
        upper_clipped = np.zeros(valid.shape, dtype=bool)
    else:
        lower_limit, upper_limit = y_limits
        display_lower = np.maximum(lower, lower_limit)
        display_upper = np.minimum(upper, upper_limit)
        lower_clipped = valid & (lower < lower_limit)
        upper_clipped = valid & (upper > upper_limit)

    axis.fill_between(
        elapsed,
        display_lower,
        display_upper,
        where=valid,
        color=color,
        alpha=alpha,
        linewidth=0,
        label=label,
    )
    if y_limits is not None:
        lower_limit, upper_limit = y_limits
        if np.any(upper_clipped):
            axis.scatter(
                elapsed[upper_clipped],
                np.full(np.count_nonzero(upper_clipped), upper_limit),
                marker="^",
                s=11,
                color=color,
                alpha=0.55,
                linewidths=0,
                clip_on=False,
                zorder=4,
            )
        if np.any(lower_clipped):
            axis.scatter(
                elapsed[lower_clipped],
                np.full(np.count_nonzero(lower_clipped), lower_limit),
                marker="v",
                s=11,
                color=color,
                alpha=0.55,
                linewidths=0,
                clip_on=False,
                zorder=4,
            )
    return bool(np.any(lower_clipped) or np.any(upper_clipped))


def _plot_series(
    elapsed_s: np.ndarray,
    values: np.ndarray,
    sigma: np.ndarray,
    *,
    ylabel: str,
    title: str,
    path: Path,
    raw_values: np.ndarray | None = None,
    raw_quality_mask: np.ndarray | None = None,
    y_limits: tuple[float, float] | None = None,
    zero_reference: bool = False,
) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    finite = np.isfinite(values)
    if raw_values is not None:
        raw_finite = np.isfinite(raw_values)
        if raw_quality_mask is None:
            raw_quality = np.zeros(raw_values.shape, dtype=np.uint32)
        else:
            raw_quality = np.asarray(raw_quality_mask, dtype=np.uint32)
            if raw_quality.shape != raw_values.shape:
                raise ValueError("raw_quality_mask must match raw_values")
        outlier = raw_finite & (
            raw_quality & int(QualityFlag.KINEMATIC_HEIGHT_OUTLIER)
        ).astype(bool)
        censored = raw_finite & ~outlier & (
            raw_quality & int(QualityFlag.PARTIAL_FIELD_OF_VIEW)
        ).astype(bool)
        retained = raw_finite & ~outlier & ~censored
        axis.scatter(
            elapsed_s[retained],
            raw_values[retained],
            s=14,
            color="0.45",
            label="measured",
            zorder=2,
        )
        if np.any(censored):
            axis.scatter(
                elapsed_s[censored],
                raw_values[censored],
                s=20,
                facecolors="none",
                edgecolors="#d17a00",
                linewidths=0.8,
                label="FOV-censored",
                zorder=2,
            )
            first_censored = float(elapsed_s[np.flatnonzero(censored)[0]])
            axis.axvspan(
                first_censored,
                float(elapsed_s[-1]),
                color="#f2a33a",
                alpha=0.08,
                linewidth=0,
                zorder=0,
            )
        if np.any(outlier):
            axis.scatter(
                elapsed_s[outlier],
                raw_values[outlier],
                s=30,
                marker="x",
                color="#c62828",
                linewidths=1.0,
                label="kinematic outlier",
                zorder=4,
            )
    # Retain NaNs in the plotted arrays so Matplotlib visibly breaks lines at
    # unsupported intervals instead of connecting across missing data.
    axis.plot(elapsed_s, values, color="tab:blue", label="fit")
    uncertainty = finite & np.isfinite(sigma)
    clipped = _plot_uncertainty_band(
        axis,
        elapsed_s,
        values,
        sigma,
        color="tab:blue",
        alpha=0.2,
        label="1-sigma" if y_limits is None else "1-sigma (clipped)",
        y_limits=y_limits,
    )
    if zero_reference:
        axis.axhline(0.0, color="0.35", linewidth=0.8, alpha=0.55, zorder=1)
    axis.set(xlabel="Elapsed time (s)", ylabel=ylabel, title=title)
    if y_limits is not None:
        axis.set_ylim(*y_limits)
    if clipped:
        axis.text(
            0.01,
            0.02,
            "▲/▼: 1-sigma interval continues beyond detail view",
            transform=axis.transAxes,
            fontsize=8.5,
            color="0.3",
            ha="left",
            va="bottom",
        )
    axis.grid(alpha=0.25)
    if raw_values is not None or np.any(uncertainty):
        axis.legend(loc="best")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _pyplot():
    """Load a noninteractive plotting backend only when plots are requested."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as pyplot

    return pyplot


def _write_front_overlay(product: FrontOverlayProduct, path: Path) -> None:
    """Write a compact raw-image/front verification mosaic."""

    plt = _pyplot()
    count = len(product.frames)
    columns = min(3, count)
    rows = int(math.ceil(count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.4 * columns, 4.4 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    center_y, center_x = product.center_yx
    north_y, north_x = product.north_vector_yx
    east_y, east_x = product.east_vector_yx
    for axis, frame in zip(axes.flat, product.frames, strict=False):
        image = np.asarray(frame.image, dtype=np.float64)
        finite = image[np.isfinite(image)]
        low, high = np.percentile(finite, [1.0, 99.7])
        linear_width = max((high - low) * 0.03, np.finfo(float).eps)
        display_image = np.arcsinh((image - low) / linear_width)
        axis.imshow(
            display_image,
            origin="lower",
            cmap="gray",
            vmin=0.0,
            vmax=float(np.arcsinh((high - low) / linear_width)),
        )
        angles = np.deg2rad(np.asarray(frame.position_angle_deg, dtype=np.float64))
        radii = np.asarray(frame.radius_px, dtype=np.float64)
        sample_y = center_y + radii * (
            np.cos(angles) * north_y + np.sin(angles) * east_y
        )
        sample_x = center_x + radii * (
            np.cos(angles) * north_x + np.sin(angles) * east_x
        )
        axis.scatter(sample_x, sample_y, s=8, color="cyan", edgecolors="none")
        height_text = (
            f"{frame.headline_height_rsun:.2f} R_sun"
            if math.isfinite(frame.headline_height_rsun)
            else "no headline height"
        )
        axis.set_title(
            f"frame {frame.frame_number}, t={frame.elapsed_s:g} s, {height_text}"
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[count:]:
        axis.set_visible(False)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_event_products(
    output_root: str | Path,
    track: TrackProduct,
    front_samples: FrontSamplesProduct,
    summary: Mapping[str, Any],
    *,
    front_overlay: FrontOverlayProduct | None = None,
    diagnostic_movie_writer: Callable[[Path], None] | None = None,
    diagnostic_movie_metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Stage a complete product set, then publish it on one filesystem.

    The final event directory is never populated file by file.  If generation
    fails, the prior product (when present) remains untouched and the staging
    directory is removed.  ``overwrite=True`` performs a recoverable directory
    swap before deleting the old product.
    """

    if diagnostic_movie_metadata is not None and diagnostic_movie_writer is None:
        raise ProductValidationError(
            "diagnostic_movie_metadata requires diagnostic_movie_writer."
        )

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    event_directory = root / track.event_id
    if event_directory.is_symlink() or (
        event_directory.exists() and not event_directory.is_dir()
    ):
        raise ProductValidationError(
            f"Event product path must be a real directory: {event_directory}"
        )
    if (
        event_directory.exists()
        and any(event_directory.iterdir())
        and not overwrite
    ):
        raise FileExistsError(
            f"Refusing to overwrite non-empty event directory: {event_directory}"
        )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{track.event_id}.staging-", dir=root)
    )
    backup: Path | None = None
    try:
        _write_product_set(
            staging,
            track,
            front_samples,
            summary,
            front_overlay=front_overlay,
            diagnostic_movie_writer=diagnostic_movie_writer,
            diagnostic_movie_metadata=diagnostic_movie_metadata,
        )

        if event_directory.is_symlink() or (
            event_directory.exists() and not event_directory.is_dir()
        ):
            raise ProductValidationError(
                f"Event product path changed to a non-directory: {event_directory}"
            )
        if event_directory.exists():
            if any(event_directory.iterdir()) and not overwrite:
                raise FileExistsError(
                    "Another writer published this event while products were "
                    f"being staged: {event_directory}"
                )
            backup = root / f".{track.event_id}.backup-{uuid.uuid4().hex}"
            event_directory.replace(backup)
        try:
            staging.replace(event_directory)
        except Exception:
            if backup is not None and backup.exists() and not event_directory.exists():
                backup.replace(event_directory)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return event_directory
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _write_product_set(
    event_directory: Path,
    track: TrackProduct,
    front_samples: FrontSamplesProduct,
    summary: Mapping[str, Any],
    *,
    front_overlay: FrontOverlayProduct | None,
    diagnostic_movie_writer: Callable[[Path], None] | None,
    diagnostic_movie_metadata: Mapping[str, Any] | None,
) -> None:
    """Write one complete event product inside an unpublished directory."""

    track_path = event_directory / "track.ecsv"
    samples_path = event_directory / "front_samples.ecsv"
    track_table(track).write(track_path, format="ascii.ecsv", overwrite=True)
    front_samples_table(front_samples).write(
        samples_path, format="ascii.ecsv", overwrite=True
    )

    elapsed = np.asarray(track.elapsed_s, dtype=np.float64)
    solar_radius_km = float(track.metadata.get("solar_radius_km", 695_700.0))
    derivative_exclusions = int(
        QualityFlag.DERIVATIVE_ENDPOINT | QualityFlag.TRACK_GAP
    )
    derivative_valid = (
        np.asarray(track.quality_mask, dtype=np.uint32) & derivative_exclusions
    ) == 0
    _plot_series(
        elapsed,
        np.asarray(track.height_fit_rsun, dtype=np.float64),
        np.asarray(track.height_fit_sigma_rsun, dtype=np.float64),
        raw_values=np.asarray(track.height_raw_rsun, dtype=np.float64),
        raw_quality_mask=np.asarray(track.quality_mask, dtype=np.uint32),
        ylabel="Projected height (R_sun)",
        title=f"{track.event_id}: CME height-time",
        path=event_directory / "height_time.png",
    )
    _plot_series(
        elapsed,
        np.where(
            derivative_valid,
            np.asarray(track.speed_fit_km_s, dtype=np.float64),
            np.nan,
        ),
        np.where(
            derivative_valid,
            np.asarray(track.speed_sigma_km_s, dtype=np.float64),
            np.nan,
        ),
        ylabel="Projected speed (km/s)",
        title=f"{track.event_id}: CME speed-time",
        path=event_directory / "speed_time.png",
    )
    _plot_series(
        elapsed,
        np.where(
            derivative_valid,
            np.asarray(track.acceleration_fit_m_s2, dtype=np.float64),
            np.nan,
        ),
        np.where(
            derivative_valid,
            np.asarray(track.acceleration_sigma_m_s2, dtype=np.float64),
            np.nan,
        ),
        ylabel="Projected acceleration (m/s^2)",
        title=f"{track.event_id}: CME acceleration-time",
        path=event_directory / "acceleration_time.png",
        zero_reference=True,
    )
    acceleration_values = np.where(
        derivative_valid,
        np.asarray(track.acceleration_fit_m_s2, dtype=np.float64),
        np.nan,
    )
    _plot_series(
        elapsed,
        acceleration_values,
        np.where(
            derivative_valid,
            np.asarray(track.acceleration_sigma_m_s2, dtype=np.float64),
            np.nan,
        ),
        ylabel="Projected acceleration (m/s^2)",
        title=(
            f"{track.event_id}: CME acceleration-time detail "
            "(1-sigma clipped)"
        ),
        path=event_directory / "acceleration_time_detail.png",
        y_limits=_zero_centered_detail_limits(acceleration_values),
        zero_reference=True,
    )
    if front_overlay is not None:
        _write_front_overlay(front_overlay, event_directory / "front_overlay.png")
    movie_path = event_directory / "front_tracking.mp4"
    if diagnostic_movie_writer is not None:
        diagnostic_movie_writer(movie_path)
        if (
            movie_path.is_symlink()
            or not movie_path.is_file()
            or movie_path.stat().st_size == 0
        ):
            raise ProductValidationError(
                "The diagnostic movie writer did not create a non-empty regular "
                f"file at {movie_path}."
            )

    height_plot = event_directory / "height_time.png"
    speed_plot = event_directory / "speed_time.png"
    acceleration_plot = event_directory / "acceleration_time.png"
    acceleration_detail_plot = event_directory / "acceleration_time_detail.png"
    overlay_plot = event_directory / "front_overlay.png"
    product_index: dict[str, Any] = {
        "track": {"path": track_path.name, "sha256": _sha256(track_path)},
        "front_samples": {
            "path": samples_path.name,
            "sha256": _sha256(samples_path),
        },
        "height_time_plot": {
            "path": height_plot.name,
            "sha256": _sha256(height_plot),
        },
        "speed_time_plot": {
            "path": speed_plot.name,
            "sha256": _sha256(speed_plot),
        },
        "acceleration_time_plot": {
            "path": acceleration_plot.name,
            "sha256": _sha256(acceleration_plot),
        },
        "acceleration_time_detail_plot": {
            "path": acceleration_detail_plot.name,
            "sha256": _sha256(acceleration_detail_plot),
            "uncertainty_display": "clipped_with_boundary_markers",
        },
        "front_overlay": (
            {
                "path": overlay_plot.name,
                "sha256": _sha256(overlay_plot),
            }
            if front_overlay is not None
            else None
        ),
    }
    if diagnostic_movie_writer is not None:
        product_index["front_tracking_movie"] = {
            **dict(diagnostic_movie_metadata or {}),
            "path": movie_path.name,
            "sha256": _sha256(movie_path),
        }

    summary_values = dict(summary)
    summary_values.update(
        {
            "schema": "suncet.cme_event_summary",
            "schema_version": 1,
            "event_id": track.event_id,
            "frame_count": len(track.elapsed_s),
            "solar_radius_km": solar_radius_km,
            "products": product_index,
        }
    )
    summary_path = event_directory / "summary.json"
    _write_json_atomic(
        _json_scalar(summary_values), summary_path
    )
    _write_json_atomic(
        {
            "schema": "suncet.cme_product_completion",
            "schema_version": 1,
            "event_id": track.event_id,
            "summary_sha256": _sha256(summary_path),
        },
        event_directory / "COMPLETE.json",
    )

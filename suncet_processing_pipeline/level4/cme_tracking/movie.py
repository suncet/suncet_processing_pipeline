"""Synchronized image/front and kinematics diagnostic movies."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..common.quality import QualityFlag
from .products import (
    TrackProduct,
    _plot_uncertainty_band,
    _zero_centered_detail_limits,
)

if TYPE_CHECKING:
    from .pipeline import CMETrackingRun


class MovieWriterUnavailable(RuntimeError):
    """Raised when the selected Matplotlib animation writer is unavailable."""


def _pyplot_and_animation():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.animation as animation
    import matplotlib.pyplot as pyplot

    return pyplot, animation


def _display_limits(run: "CMETrackingRun") -> tuple[float, float, float]:
    """Estimate one stable asinh display stretch from representative frames."""

    sample_count = min(12, run.sequence.frame_count)
    indices = np.unique(
        np.linspace(0, run.sequence.frame_count - 1, sample_count)
        .round()
        .astype(int)
    )
    lows: list[float] = []
    highs: list[float] = []
    for index in indices:
        image = np.asarray(run.sequence.read_frame(int(index)).data, dtype=np.float64)
        finite = image[np.isfinite(image)]
        if finite.size:
            low, high = np.percentile(finite, [1.0, 99.7])
            lows.append(float(low))
            highs.append(float(high))
    if not lows:
        raise ValueError("Diagnostic movie input contains no finite image pixels.")
    low = float(np.median(lows))
    high = float(np.median(highs))
    if not high > low:
        high = low + max(abs(low), 1.0) * np.finfo(np.float64).eps
    linear_width = max((high - low) * 0.03, np.finfo(np.float64).eps)
    return low, high, linear_width


def _asinh_display(
    image: np.ndarray,
    *,
    low: float,
    linear_width: float,
) -> np.ndarray:
    return np.arcsinh((np.asarray(image, dtype=np.float64) - low) / linear_width)


def _finite_limits(
    values: np.ndarray,
    sigma: np.ndarray | None = None,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    candidates = [finite_values]
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=np.float64)
        valid = np.isfinite(values) & np.isfinite(sigma)
        candidates.extend((values[valid] - sigma[valid], values[valid] + sigma[valid]))
    nonempty = [candidate for candidate in candidates if candidate.size]
    if not nonempty:
        return -1.0, 1.0
    finite = np.concatenate(nonempty)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    span = upper - lower
    padding = 0.06 * (span if span > 0 else max(abs(lower), 1.0))
    return lower - padding, upper + padding


def _set_current_point(artist: Any, time: float, value: float) -> None:
    if math.isfinite(value):
        artist.set_data([time], [value])
        artist.set_visible(True)
    else:
        artist.set_data([], [])
        artist.set_visible(False)


def _value_text(value: float, unit: str, precision: int) -> str:
    return f"{value:.{precision}f} {unit}" if math.isfinite(value) else "unavailable"


def write_cme_tracking_movie(
    run: "CMETrackingRun",
    track: TrackProduct,
    path: str | Path,
    *,
    fps: float = 10.0,
    dpi: int = 100,
    writer_name: str = "ffmpeg",
) -> Path:
    """Write every input frame with synchronized tracking and kinematic plots.

    The default output is an H.264 MP4 through the external ``ffmpeg`` writer.
    ``writer_name='pillow'`` is retained for small dependency-free GIF tests.
    Images are loaded one at a time so movie generation does not duplicate the
    full image cube in memory.
    """

    if not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("fps must be finite and positive.")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    plt, animation = _pyplot_and_animation()
    if not animation.writers.is_available(writer_name):
        raise MovieWriterUnavailable(
            f"Matplotlib animation writer {writer_name!r} is unavailable. "
            "Install ffmpeg for MP4 output."
        )
    if writer_name == "ffmpeg":
        writer = animation.FFMpegWriter(
            fps=float(fps),
            codec="libx264",
            metadata={
                "title": f"{track.event_id} CME tracking diagnostic",
                "artist": "SunCET processing pipeline",
                "comment": "Provisional known-window Level 4 research output",
            },
            extra_args=[
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                "-movflags",
                "+faststart",
            ],
        )
    elif writer_name == "pillow":
        writer = animation.PillowWriter(fps=float(fps))
    else:
        writer = animation.writers[writer_name](fps=float(fps))

    elapsed = np.asarray(track.elapsed_s, dtype=np.float64)
    raw_height = np.asarray(track.height_raw_rsun, dtype=np.float64)
    fitted_height = np.asarray(track.height_fit_rsun, dtype=np.float64)
    height_sigma = np.asarray(track.height_fit_sigma_rsun, dtype=np.float64)
    quality = np.asarray(track.quality_mask, dtype=np.uint32)
    derivative_exclusions = int(
        QualityFlag.DERIVATIVE_ENDPOINT | QualityFlag.TRACK_GAP
    )
    derivative_valid = (quality & derivative_exclusions) == 0
    speed = np.where(
        derivative_valid,
        np.asarray(track.speed_fit_km_s, dtype=np.float64),
        np.nan,
    )
    speed_sigma = np.where(
        derivative_valid,
        np.asarray(track.speed_sigma_km_s, dtype=np.float64),
        np.nan,
    )
    acceleration = np.where(
        derivative_valid,
        np.asarray(track.acceleration_fit_m_s2, dtype=np.float64),
        np.nan,
    )
    acceleration_sigma = np.where(
        derivative_valid,
        np.asarray(track.acceleration_sigma_m_s2, dtype=np.float64),
        np.nan,
    )

    low, high, linear_width = _display_limits(run)
    display_maximum = float(np.arcsinh((high - low) / linear_width))
    initial = run.sequence.read_frame(0).data

    figure = plt.figure(figsize=(16.0, 9.0), constrained_layout=True)
    grid = figure.add_gridspec(
        3,
        2,
        width_ratios=(1.45, 1.0),
        height_ratios=(1.0, 1.0, 1.0),
    )
    image_axis = figure.add_subplot(grid[:, 0])
    height_axis = figure.add_subplot(grid[0, 1])
    speed_axis = figure.add_subplot(grid[1, 1], sharex=height_axis)
    acceleration_axis = figure.add_subplot(grid[2, 1], sharex=height_axis)
    plot_axes = (height_axis, speed_axis, acceleration_axis)

    image_artist = image_axis.imshow(
        _asinh_display(initial, low=low, linear_width=linear_width),
        origin="lower",
        cmap="gray",
        vmin=0.0,
        vmax=display_maximum,
    )
    front_artist = image_axis.scatter(
        [],
        [],
        s=16,
        color="#00d7df",
        edgecolors="none",
        label="Retained front samples",
    )
    image_axis.set_xticks([])
    image_axis.set_yticks([])
    image_axis.legend(loc="lower left", framealpha=0.75)

    raw_finite = np.isfinite(raw_height)
    height_outlier = raw_finite & (
        quality & int(QualityFlag.KINEMATIC_HEIGHT_OUTLIER)
    ).astype(bool)
    fov_censored = raw_finite & ~height_outlier & (
        quality & int(QualityFlag.PARTIAL_FIELD_OF_VIEW)
    ).astype(bool)
    retained_height = raw_finite & ~height_outlier & ~fov_censored
    height_axis.scatter(
        elapsed[retained_height],
        raw_height[retained_height],
        s=13,
        color="0.45",
        label="Measured height",
        zorder=3,
    )
    if np.any(fov_censored):
        height_axis.scatter(
            elapsed[fov_censored],
            raw_height[fov_censored],
            s=18,
            facecolors="none",
            edgecolors="#d17a00",
            linewidths=0.7,
            label="FOV-censored",
            zorder=3,
        )
    if np.any(height_outlier):
        height_axis.scatter(
            elapsed[height_outlier],
            raw_height[height_outlier],
            s=26,
            marker="x",
            color="#c62828",
            linewidths=0.9,
            label="Kinematic outlier",
            zorder=4,
        )
    height_axis.plot(elapsed, fitted_height, color="#1f77b4", label="Fit")
    height_uncertainty = np.isfinite(fitted_height) & np.isfinite(height_sigma)
    height_axis.fill_between(
        elapsed,
        fitted_height - height_sigma,
        fitted_height + height_sigma,
        where=height_uncertainty,
        color="#1f77b4",
        alpha=0.18,
        linewidth=0,
        label="1-sigma",
    )
    height_axis.legend(loc="best", ncol=3, fontsize="small")

    speed_axis.plot(elapsed, speed, color="#1f77b4")
    speed_uncertainty = np.isfinite(speed) & np.isfinite(speed_sigma)
    speed_axis.fill_between(
        elapsed,
        speed - speed_sigma,
        speed + speed_sigma,
        where=speed_uncertainty,
        color="#1f77b4",
        alpha=0.18,
        linewidth=0,
    )
    acceleration_axis.plot(elapsed, acceleration, color="#1f77b4")
    acceleration_limits = _zero_centered_detail_limits(acceleration)
    acceleration_uncertainty_clipped = _plot_uncertainty_band(
        acceleration_axis,
        elapsed,
        acceleration,
        acceleration_sigma,
        color="#1f77b4",
        alpha=0.18,
        label=None,
        y_limits=acceleration_limits,
    )
    acceleration_axis.axhline(
        0.0, color="0.35", linewidth=0.8, alpha=0.55, zorder=1
    )
    if acceleration_uncertainty_clipped:
        acceleration_axis.text(
            0.01,
            0.03,
            "▲/▼: 1σ continues beyond detail view",
            transform=acceleration_axis.transAxes,
            fontsize=7.5,
            color="0.3",
            ha="left",
            va="bottom",
        )

    height_axis.set_ylabel(r"Projected height ($R_\odot$)")
    speed_axis.set_ylabel("Projected speed (km/s)")
    acceleration_axis.set_ylabel(r"Projected acceleration (m/s$^2$)")
    acceleration_axis.set_xlabel("Elapsed time (s)")
    height_for_limits = np.concatenate((fitted_height, raw_height))
    height_sigma_for_limits = np.concatenate(
        (height_sigma, np.full_like(raw_height, np.nan))
    )
    height_axis.set_ylim(*_finite_limits(height_for_limits, height_sigma_for_limits))
    speed_axis.set_ylim(*_finite_limits(speed, speed_sigma))
    acceleration_axis.set_ylim(*acceleration_limits)
    for axis in plot_axes:
        axis.grid(alpha=0.25)
        axis.set_xlim(float(elapsed[0]), float(elapsed[-1]))
    height_axis.tick_params(labelbottom=False)
    speed_axis.tick_params(labelbottom=False)

    limited_indices = np.flatnonzero(run.field_of_view_limited_mask)
    if limited_indices.size:
        start = float(elapsed[limited_indices[0]])
        for axis in plot_axes:
            axis.axvspan(
                start,
                float(elapsed[-1]),
                color="#f2a33a",
                alpha=0.10,
                linewidth=0,
            )

    cursor_lines = tuple(
        axis.axvline(elapsed[0], color="#d62728", linewidth=1.1, alpha=0.9)
        for axis in plot_axes
    )
    height_point, = height_axis.plot([], [], "o", color="#00a9b5", markersize=6)
    speed_point, = speed_axis.plot([], [], "o", color="#00a9b5", markersize=6)
    acceleration_point, = acceleration_axis.plot(
        [], [], "o", color="#00a9b5", markersize=6
    )

    cadence_status = track.metadata.get("cadence_status")
    timing_label = (
        f"{cadence_status} cadence" if cadence_status else "observed timing"
    )
    figure.suptitle(
        f"{track.event_id} | provisional known-window CME tracking | {timing_label}"
    )
    center_y, center_x = run.sequence.geometry.center_yx
    north_y, north_x = run.sequence.geometry.north_vector_yx
    east_y, east_x = run.sequence.geometry.east_vector_yx

    def update(index: int):
        loaded = run.sequence.read_frame(index)
        image_artist.set_data(
            _asinh_display(loaded.data, low=low, linear_width=linear_width)
        )
        observed = run.front.observed_mask[index]
        angles = np.deg2rad(run.front.position_angle_deg[observed])
        radii = run.front.radius_px[index, observed]
        sample_y = center_y + radii * (
            np.cos(angles) * north_y + np.sin(angles) * east_y
        )
        sample_x = center_x + radii * (
            np.cos(angles) * north_x + np.sin(angles) * east_x
        )
        offsets = (
            np.column_stack((sample_x, sample_y))
            if sample_x.size
            else np.empty((0, 2), dtype=np.float64)
        )
        front_artist.set_offsets(offsets)

        time = float(elapsed[index])
        for cursor in cursor_lines:
            cursor.set_xdata([time, time])
        _set_current_point(height_point, time, float(raw_height[index]))
        _set_current_point(speed_point, time, float(speed[index]))
        _set_current_point(
            acceleration_point,
            time,
            float(acceleration[index]),
        )

        status: list[str] = []
        if run.field_of_view_limited_mask[index]:
            status.append("FOV-limited / censored")
        if height_outlier[index]:
            status.append("height outlier / excluded from fit")
        if not status:
            status.append("uncensored")
        image_axis.set_title(
            f"Frame {track.frame_number[index]} | sequence "
            f"{index + 1}/{run.sequence.frame_count} | t={time:g} s | "
            f"{'; '.join(status)}"
        )
        height_axis.set_title(
            "Height: " + _value_text(float(raw_height[index]), r"$R_\odot$", 2)
        )
        speed_axis.set_title(
            "Speed: " + _value_text(float(speed[index]), "km/s", 1)
        )
        acceleration_axis.set_title(
            "Acceleration: "
            + _value_text(float(acceleration[index]), r"m/s$^2$", 1)
        )
        return (
            image_artist,
            front_artist,
            *cursor_lines,
            height_point,
            speed_point,
            acceleration_point,
        )

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=run.sequence.frame_count,
        interval=1000.0 / float(fps),
        blit=False,
        repeat=False,
        cache_frame_data=False,
    )
    try:
        movie.save(output, writer=writer, dpi=dpi)
    finally:
        plt.close(figure)
    return output

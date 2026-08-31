"""Transparent temporal screening of provisional headline CME heights.

This module deliberately does not alter front samples or manufacture a
replacement height.  A rejected scalar headline height becomes ``NaN`` only
in the array supplied to the kinematic fit; the measured value remains in the
raw track product and is accompanied by a quality flag.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class HeadlineHeightFilterResult:
    """Kinematic input plus inspectable decisions from the temporal filter."""

    raw_height: FloatArray
    kinematic_height: FloatArray
    support_mask: BoolArray
    candidate_mask: BoolArray
    evaluated_mask: BoolArray
    outlier_mask: BoolArray
    local_median_height: FloatArray
    residual: FloatArray
    method_metadata: dict[str, Any]


def _boolean_mask(
    values: ArrayLike | None,
    *,
    size: int,
    name: str,
    default: BoolArray,
) -> BoolArray:
    if values is None:
        return default.copy()
    mask = np.asarray(values)
    if mask.ndim != 1 or mask.size != size:
        raise ValueError(f"{name} must be a one-dimensional mask matching height")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise ValueError(f"{name} must contain boolean values")
    return np.asarray(mask, dtype=np.bool_).copy()


def filter_headline_height_outliers(
    elapsed_seconds: ArrayLike,
    raw_height: ArrayLike,
    *,
    support_mask: ArrayLike | None = None,
    candidate_mask: ArrayLike | None = None,
    enabled: bool = False,
    window_samples: int = 7,
    absolute_tolerance: float = 0.2,
    minimum_neighbors: int = 4,
    maximum_gap_seconds: float | None = None,
) -> HeadlineHeightFilterResult:
    """Exclude isolated temporal headline-height outliers from a later fit.

    For each eligible candidate, the reference is the median of finite support
    heights in an odd, centered row window, excluding the candidate itself.
    At least one support sample must occur on each side, so this method never
    extrapolates at the edge of a sequence or temporal segment.  A candidate
    is rejected when its absolute residual from that median is strictly larger
    than ``absolute_tolerance``.

    ``support_mask`` and ``candidate_mask`` are intentionally distinct.  A
    finite, uncensored but low-coverage headline height may stabilize the
    median while remaining ineligible for either rejection or kinematic use.
    No values are interpolated: rejected and otherwise ineligible samples are
    represented by ``NaN`` in ``kinematic_height``.
    """

    time = np.asarray(elapsed_seconds, dtype=np.float64)
    height = np.asarray(raw_height, dtype=np.float64)
    if time.ndim != 1 or height.ndim != 1 or time.size != height.size:
        raise ValueError(
            "elapsed_seconds and raw_height must be equal-length one-dimensional arrays"
        )
    if not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0.0):
        raise ValueError("elapsed_seconds must be finite and strictly increasing")
    if np.any(np.isinf(height)):
        raise ValueError("raw_height may contain finite values or NaN, not infinity")
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("enabled must be boolean")
    if isinstance(window_samples, bool) or not isinstance(
        window_samples, (int, np.integer)
    ):
        raise ValueError("window_samples must be an odd integer of at least three")
    if window_samples < 3 or window_samples % 2 != 1:
        raise ValueError("window_samples must be an odd integer of at least three")
    if isinstance(minimum_neighbors, bool) or not isinstance(
        minimum_neighbors, (int, np.integer)
    ):
        raise ValueError("minimum_neighbors must be an integer")
    if minimum_neighbors < 2 or minimum_neighbors >= window_samples:
        raise ValueError(
            "minimum_neighbors must be at least two and smaller than window_samples"
        )
    absolute_tolerance = float(absolute_tolerance)
    if not math.isfinite(absolute_tolerance) or absolute_tolerance <= 0.0:
        raise ValueError("absolute_tolerance must be finite and positive")
    if maximum_gap_seconds is not None:
        maximum_gap_seconds = float(maximum_gap_seconds)
        if not math.isfinite(maximum_gap_seconds) or maximum_gap_seconds <= 0.0:
            raise ValueError("maximum_gap_seconds must be finite and positive")

    finite = np.isfinite(height)
    support = _boolean_mask(
        support_mask,
        size=height.size,
        name="support_mask",
        default=finite,
    )
    candidate = _boolean_mask(
        candidate_mask,
        size=height.size,
        name="candidate_mask",
        default=finite,
    )
    support &= finite
    candidate &= finite

    kinematic_height = np.where(candidate, height, np.nan)
    evaluated = np.zeros(height.shape, dtype=np.bool_)
    outlier = np.zeros(height.shape, dtype=np.bool_)
    local_median = np.full(height.shape, np.nan, dtype=np.float64)
    residual = np.full(height.shape, np.nan, dtype=np.float64)

    segment = np.zeros(height.shape, dtype=np.int64)
    if maximum_gap_seconds is not None and height.size > 1:
        segment[1:] = np.cumsum(np.diff(time) > maximum_gap_seconds)

    if enabled:
        half_window = int(window_samples) // 2
        for index in np.flatnonzero(candidate):
            start = max(0, int(index) - half_window)
            stop = min(height.size, int(index) + half_window + 1)
            neighbors = np.arange(start, stop, dtype=np.int64)
            neighbor_mask = (
                support[neighbors]
                & (neighbors != index)
                & (segment[neighbors] == segment[index])
            )
            neighbors = neighbors[neighbor_mask]
            if neighbors.size < minimum_neighbors:
                continue
            if not np.any(neighbors < index) or not np.any(neighbors > index):
                continue

            reference = float(np.median(height[neighbors]))
            difference = float(height[index] - reference)
            evaluated[index] = True
            local_median[index] = reference
            residual[index] = difference
            outlier[index] = abs(difference) > absolute_tolerance

        kinematic_height[outlier] = np.nan

    rejected_indices = np.flatnonzero(outlier).astype(int).tolist()
    metadata: dict[str, Any] = {
        "method": "centered_neighbor_median_absolute_residual",
        "enabled": bool(enabled),
        "provisional": True,
        "window_samples": int(window_samples),
        "absolute_tolerance": absolute_tolerance,
        "minimum_neighbors": int(minimum_neighbors),
        "requires_bidirectional_support": True,
        "support_definition": (
            "caller-selected finite headline heights; pipeline uses all finite "
            "non-FOV-limited rows"
        ),
        "candidate_definition": (
            "caller-selected finite rows otherwise eligible for kinematic fitting"
        ),
        "maximum_gap_seconds": maximum_gap_seconds,
        "evaluated_count": int(np.count_nonzero(evaluated)),
        "rejected_count": len(rejected_indices),
        "rejected_indices": rejected_indices,
        "replacement_policy": (
            "none; rejected values are NaN only in the kinematic fit input"
        ),
        "raw_height_modified": False,
    }
    return HeadlineHeightFilterResult(
        raw_height=height.copy(),
        kinematic_height=kinematic_height,
        support_mask=support,
        candidate_mask=candidate,
        evaluated_mask=evaluated,
        outlier_mask=outlier,
        local_median_height=local_median,
        residual=residual,
        method_metadata=metadata,
    )

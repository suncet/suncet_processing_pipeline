"""Science metrics shared by CME-tracking algorithm comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ErrorMetrics:
    """Finite-overlap errors for one scalar or array-valued quantity."""

    sample_count: int
    truth_count: int
    coverage_fraction: float
    bias: float
    mean_absolute_error: float
    median_absolute_error: float
    root_mean_square_error: float
    percentile_90_absolute_error: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def finite_overlap_metrics(
    predicted: np.ndarray,
    truth: np.ndarray,
) -> ErrorMetrics:
    """Evaluate values where truth exists, retaining missing-prediction coverage."""

    predicted_values = np.asarray(predicted, dtype=np.float64)
    truth_values = np.asarray(truth, dtype=np.float64)
    if predicted_values.shape != truth_values.shape:
        raise ValueError("Predicted and truth arrays must have identical shapes.")
    truth_valid = np.isfinite(truth_values)
    overlap = truth_valid & np.isfinite(predicted_values)
    truth_count = int(np.count_nonzero(truth_valid))
    sample_count = int(np.count_nonzero(overlap))
    coverage = sample_count / truth_count if truth_count else 0.0
    if sample_count == 0:
        return ErrorMetrics(
            sample_count=0,
            truth_count=truth_count,
            coverage_fraction=coverage,
            bias=float("nan"),
            mean_absolute_error=float("nan"),
            median_absolute_error=float("nan"),
            root_mean_square_error=float("nan"),
            percentile_90_absolute_error=float("nan"),
        )
    residual = predicted_values[overlap] - truth_values[overlap]
    absolute = np.abs(residual)
    return ErrorMetrics(
        sample_count=sample_count,
        truth_count=truth_count,
        coverage_fraction=coverage,
        bias=float(np.mean(residual)),
        mean_absolute_error=float(np.mean(absolute)),
        median_absolute_error=float(np.median(absolute)),
        root_mean_square_error=float(np.sqrt(np.mean(np.square(residual)))),
        percentile_90_absolute_error=float(np.percentile(absolute, 90)),
    )


@dataclass(frozen=True)
class FrontEvaluation:
    """Radial front errors plus useful time/angle aggregations."""

    all_samples: ErrorMetrics
    per_frame_rmse: np.ndarray
    per_angle_rmse: np.ndarray


def evaluate_front(
    predicted_radius: np.ndarray,
    truth_radius: np.ndarray,
) -> FrontEvaluation:
    """Evaluate aligned ``(time, position_angle)`` front-radius arrays."""

    predicted = np.asarray(predicted_radius, dtype=np.float64)
    truth = np.asarray(truth_radius, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape != truth.shape:
        raise ValueError(
            "Front arrays must share shape (time, position_angle)."
        )

    def axis_rmse(axis: int) -> np.ndarray:
        overlap = np.isfinite(predicted) & np.isfinite(truth)
        squared = np.where(overlap, np.square(predicted - truth), np.nan)
        counts = np.sum(overlap, axis=axis)
        sums = np.nansum(squared, axis=axis)
        result = np.full(counts.shape, np.nan, dtype=np.float64)
        np.sqrt(
            np.divide(sums, counts, out=result, where=counts > 0),
            out=result,
            where=counts > 0,
        )
        return result

    return FrontEvaluation(
        all_samples=finite_overlap_metrics(predicted, truth),
        per_frame_rmse=axis_rmse(axis=1),
        per_angle_rmse=axis_rmse(axis=0),
    )

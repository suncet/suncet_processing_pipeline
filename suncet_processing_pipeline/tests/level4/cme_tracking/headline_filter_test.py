"""Tests for explicit temporal screening of headline CME heights."""

import numpy as np

from suncet_processing_pipeline.level4.cme_tracking.headline_filter import (
    filter_headline_height_outliers,
)


def _filter(height, *, time=None, support=None, candidates=None):
    values = np.asarray(height, dtype=float)
    if time is None:
        time = 15.0 * np.arange(values.size)
    return filter_headline_height_outliers(
        time,
        values,
        support_mask=support,
        candidate_mask=candidates,
        enabled=True,
        window_samples=7,
        absolute_tolerance=0.2,
        minimum_neighbors=4,
        maximum_gap_seconds=45.0,
    )


def test_smooth_constant_and_accelerating_tracks_are_retained() -> None:
    sample = np.arange(21, dtype=float)
    constant = _filter(np.full(sample.shape, 2.0))
    accelerating = _filter(1.2 + 0.02 * sample + 0.001 * sample**2)

    assert not np.any(constant.outlier_mask)
    assert not np.any(accelerating.outlier_mask)
    assert np.all(np.isfinite(constant.kinematic_height))
    assert np.all(np.isfinite(accelerating.kinematic_height))


def test_isolated_low_and_high_spikes_are_excluded_without_replacement() -> None:
    raw = 1.5 + 0.01 * np.arange(19, dtype=float)
    raw[5] -= 0.8
    raw[13] += 0.7
    unchanged = raw.copy()

    result = _filter(raw)

    np.testing.assert_array_equal(raw, unchanged)
    np.testing.assert_array_equal(np.flatnonzero(result.outlier_mask), [5, 13])
    assert np.isnan(result.kinematic_height[5])
    assert np.isnan(result.kinematic_height[13])
    assert result.raw_height[5] == raw[5]
    assert result.raw_height[13] == raw[13]
    assert result.method_metadata["replacement_policy"].startswith("none;")
    assert result.method_metadata["raw_height_modified"] is False


def test_non_candidate_samples_support_median_but_are_not_rejected() -> None:
    raw = 2.0 + 0.01 * np.arange(11, dtype=float)
    raw[5] -= 0.8
    candidates = np.ones(raw.shape, dtype=bool)
    candidates[4] = False

    result = _filter(raw, support=np.ones(raw.shape, bool), candidates=candidates)

    assert result.outlier_mask[5]
    assert not result.outlier_mask[4]
    assert np.isnan(result.kinematic_height[4])
    assert result.support_mask[4]


def test_temporal_gap_is_not_bridged_and_missing_values_are_not_filled() -> None:
    time = np.array([0.0, 15.0, 30.0, 300.0, 315.0, 330.0, 345.0])
    raw = np.array([1.0, 1.1, 0.2, 2.0, np.nan, 2.2, 2.3])

    result = _filter(raw, time=time)

    assert not np.any(result.outlier_mask)
    assert not result.evaluated_mask[2]
    assert np.isnan(result.kinematic_height[4])
    assert np.isnan(result.local_median_height[4])


def test_disabled_filter_preserves_prior_kinematic_candidate_behavior() -> None:
    raw = np.array([1.0, 1.1, 0.2, 1.3, 1.4, 1.5, 1.6])
    candidate = np.array([True, True, True, False, True, True, True])
    result = filter_headline_height_outliers(
        np.arange(raw.size, dtype=float),
        raw,
        candidate_mask=candidate,
    )

    np.testing.assert_array_equal(result.kinematic_height[candidate], raw[candidate])
    assert np.isnan(result.kinematic_height[3])
    assert not np.any(result.evaluated_mask)
    assert not np.any(result.outlier_mask)

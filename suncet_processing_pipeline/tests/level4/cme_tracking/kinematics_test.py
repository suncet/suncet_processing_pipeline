"""Focused tests for provisional CME kinematic fitting."""

import numpy as np
import pytest

from suncet_processing_pipeline.level4.cme_tracking.kinematics import (
    fit_kinematics,
)


IRREGULAR_TIMES = np.array([0.0, 7.0, 19.0, 34.0, 58.0, 91.0, 137.0])


def test_exact_constant_velocity_at_irregular_times():
    initial_height = 1.35
    velocity = 2.5e-3
    heights = initial_height + velocity * IRREGULAR_TIMES

    result = fit_kinematics(IRREGULAR_TIMES, heights)

    np.testing.assert_allclose(result.fitted_height, heights, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result.speed, velocity, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result.acceleration, 0.0, rtol=0.0, atol=1e-13)
    np.testing.assert_array_equal(
        result.endpoint_mask,
        [True, False, False, False, False, False, True],
    )
    np.testing.assert_array_equal(result.valid_mask, ~result.endpoint_mask)
    assert result.method_metadata["provisional"] is True
    assert result.method_metadata["smoothing_factor"] == 0.0


def test_exact_constant_acceleration_with_weighting():
    initial_height = 1.1
    initial_velocity = 1.8e-3
    acceleration = 7.0e-6
    heights = (
        initial_height
        + initial_velocity * IRREGULAR_TIMES
        + 0.5 * acceleration * IRREGULAR_TIMES**2
    )
    sigmas = np.linspace(1.0e-3, 2.0e-3, IRREGULAR_TIMES.size)

    result = fit_kinematics(IRREGULAR_TIMES, heights, sigmas)

    np.testing.assert_allclose(result.fitted_height, heights, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.speed,
        initial_velocity + acceleration * IRREGULAR_TIMES,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.acceleration,
        acceleration,
        rtol=0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(result.raw_height_sigma, sigmas)
    assert result.method_metadata["weighting"] == "inverse_measurement_sigma"
    assert result.method_metadata["smoothing_factor"] == IRREGULAR_TIMES.size


def test_cadence_rescaling_has_expected_derivative_scaling():
    normalized_time = IRREGULAR_TIMES / IRREGULAR_TIMES[-1]
    heights = 1.2 + 0.4 * normalized_time + 0.15 * normalized_time**2

    nominal = fit_kinematics(IRREGULAR_TIMES, heights)
    four_times_slower = fit_kinematics(4.0 * IRREGULAR_TIMES, heights)

    np.testing.assert_allclose(
        four_times_slower.fitted_height,
        nominal.fitted_height,
        rtol=0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        four_times_slower.speed,
        nominal.speed / 4.0,
        rtol=0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        four_times_slower.acceleration,
        nominal.acceleration / 16.0,
        rtol=0.0,
        atol=1e-13,
    )


def test_missing_heights_are_fit_only_inside_measurement_domain():
    times = np.array([-20.0, *IRREGULAR_TIMES, 160.0])
    initial_height = 1.1
    initial_velocity = 1.8e-3
    acceleration = 7.0e-6
    true_height = (
        initial_height
        + initial_velocity * times
        + 0.5 * acceleration * times**2
    )
    heights = true_height.copy()
    heights[[0, 3, -1]] = np.nan

    result = fit_kinematics(times, heights)

    np.testing.assert_array_equal(
        result.measurement_mask,
        [False, True, True, False, True, True, True, True, False],
    )
    np.testing.assert_array_equal(
        result.fit_domain_mask,
        [False, True, True, True, True, True, True, True, False],
    )
    assert np.all(np.isnan(result.fitted_height[[0, -1]]))
    assert np.all(np.isnan(result.speed[[0, -1]]))
    assert np.all(np.isnan(result.acceleration[[0, -1]]))
    np.testing.assert_allclose(
        result.fitted_height[result.fit_domain_mask],
        true_height[result.fit_domain_mask],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.speed[result.fit_domain_mask],
        initial_velocity + acceleration * times[result.fit_domain_mask],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.acceleration[result.fit_domain_mask],
        acceleration,
        rtol=0.0,
        atol=1e-13,
    )
    np.testing.assert_array_equal(
        result.valid_mask,
        [False, False, True, True, True, True, True, False, False],
    )


def test_uncertainties_may_be_missing_only_with_missing_height():
    heights = 1.0 + 2.0e-3 * IRREGULAR_TIMES
    heights[2] = np.nan
    sigmas = np.full(IRREGULAR_TIMES.shape, 2.0e-3)
    sigmas[2] = np.nan

    result = fit_kinematics(IRREGULAR_TIMES, heights, sigmas)

    assert np.isnan(result.raw_height_sigma[2])
    assert result.measurement_mask[2] == np.bool_(False)
    assert result.fit_domain_mask[2] == np.bool_(True)


def test_uncertainty_sampling_is_seeded_and_finite():
    heights = 1.0 + 2.0e-3 * IRREGULAR_TIMES
    sigmas = np.full(IRREGULAR_TIMES.shape, 2.0e-3)

    first = fit_kinematics(
        IRREGULAR_TIMES,
        heights,
        sigmas,
        uncertainty_samples=32,
        random_seed=20260826,
    )
    second = fit_kinematics(
        IRREGULAR_TIMES,
        heights,
        sigmas,
        uncertainty_samples=32,
        random_seed=20260826,
    )

    for field in (
        "fitted_height_sigma",
        "speed_sigma",
        "acceleration_sigma",
    ):
        first_values = getattr(first, field)
        np.testing.assert_array_equal(first_values, getattr(second, field))
        assert np.all(np.isfinite(first_values))
        assert np.all(first_values > 0.0)
    assert (
        first.method_metadata["uncertainty_method"]
        == "gaussian_measurement_monte_carlo"
    )


def test_derivatives_bordering_an_unsupported_long_gap_are_invalid():
    times = np.array([0.0, 1.0, 2.0, 20.0, 21.0, 22.0])
    heights = 1.0 + 0.01 * times

    result = fit_kinematics(
        times,
        heights,
        endpoint_samples=0,
        maximum_gap_seconds=3.0,
    )

    np.testing.assert_array_equal(
        result.unsupported_gap_mask,
        [False, False, True, True, False, False],
    )
    np.testing.assert_array_equal(
        result.valid_mask,
        [True, True, False, False, True, True],
    )
    assert result.method_metadata["unsupported_gap_count"] == 2


@pytest.mark.parametrize(
    ("times", "heights", "sigmas", "kwargs", "message"),
    [
        ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], None, {}, "at least four"),
        (
            [0.0, 1.0, 1.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            None,
            {},
            "strictly increasing",
        ),
        (
            [0.0, 1.0, np.nan, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            None,
            {},
            "finite",
        ),
        ([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0], None, {}, "equal length"),
        (
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 0.0, 1.0, 1.0],
            {},
            "positive",
        ),
        (
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            None,
            {"smoothing_factor": -1.0},
            "nonnegative",
        ),
        (
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            None,
            {"endpoint_samples": 2},
            "interior",
        ),
        (
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            None,
            {"uncertainty_samples": 1},
            "zero or at least two",
        ),
        (
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            None,
            {"maximum_gap_seconds": 0.0},
            "finite and positive",
        ),
        (
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, np.nan, 4.0, 5.0],
            [1.0, 1.0, np.nan, 1.0, np.nan],
            {},
            "finite and positive at every measurement",
        ),
        (
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, np.nan, np.nan, 4.0, 5.0],
            None,
            {},
            "at least four finite",
        ),
    ],
)
def test_invalid_inputs_raise_value_error(times, heights, sigmas, kwargs, message):
    with pytest.raises(ValueError, match=message):
        fit_kinematics(times, heights, sigmas, **kwargs)

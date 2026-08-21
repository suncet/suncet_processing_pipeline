"""Tests for the SunCET CCSDS secondary-time conversion."""

import pytest

from suncet_processing_pipeline.spacecraft_time import (
    combine_spacecraft_time_seconds,
)


@pytest.mark.parametrize(
    ("fine_milliseconds", "expected"),
    [(0, 100.0), (1, 100.001), (999, 100.999)],
)
def test_combines_integer_milliseconds(fine_milliseconds, expected):
    assert combine_spacecraft_time_seconds(100, fine_milliseconds) == expected


@pytest.mark.parametrize("fine_milliseconds", [-1, 1_000, 1.5, float("inf")])
def test_rejects_invalid_fine_milliseconds(fine_milliseconds):
    with pytest.raises(ValueError):
        combine_spacecraft_time_seconds(100, fine_milliseconds)


@pytest.mark.parametrize("coarse_seconds", [-1, float("inf"), float("nan")])
def test_rejects_invalid_coarse_seconds(coarse_seconds):
    with pytest.raises(ValueError):
        combine_spacecraft_time_seconds(coarse_seconds, 0)

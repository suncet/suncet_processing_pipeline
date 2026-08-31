import pytest

from suncet_processing_pipeline.level4.cme_tracking.config import (
    CMETrackingConfig,
    KinematicsConfig,
    read_configuration,
    write_configuration,
)
from suncet_processing_pipeline.level4.cme_tracking.likelihood import EvidenceConfig
from suncet_processing_pipeline.level4.cme_tracking.tracking import (
    TrackingConfig,
    TrackingError,
)
from suncet_processing_pipeline.level4.common.provenance import configuration_sha256


def test_default_configuration_is_json_hashable() -> None:
    values = CMETrackingConfig().to_dict()

    assert values["algorithm_name"] == "polar_coherent_fragments_v1"
    assert values["polar"]["minimum_radius_rsun"] == 1.05
    assert len(configuration_sha256(values)) == 64


def test_configuration_json_round_trip(tmp_path) -> None:
    configuration = CMETrackingConfig(
        evidence=EvidenceConfig(temporal_median_window_frames=3)
    )
    path = write_configuration(configuration, tmp_path / "tracker.json")

    assert read_configuration(path) == configuration


def test_tracking_configuration_normalizes_position_angle_window() -> None:
    configuration = TrackingConfig(position_angle_window_deg=[220, 330])

    assert configuration.position_angle_window_deg == (220.0, 330.0)


@pytest.mark.parametrize(
    "window",
    (
        (10.0, 10.0),
        (-1.0, 30.0),
        (30.0, 360.0),
        (10.0,),
    ),
)
def test_tracking_configuration_rejects_invalid_position_angle_window(
    window,
) -> None:
    with pytest.raises(TrackingError, match="position_angle_window_deg"):
        TrackingConfig(position_angle_window_deg=window)


@pytest.mark.parametrize("value", (0.0, -0.1, 1.1, float("inf")))
def test_tracking_configuration_rejects_invalid_fragment_occupancy(value) -> None:
    with pytest.raises(TrackingError, match="angular_occupancy"):
        TrackingConfig(minimum_fragment_angular_occupancy=value)


@pytest.mark.parametrize("value", (0.0, -0.1, 1.1, float("inf")))
def test_tracking_configuration_rejects_invalid_fragment_overlap(value) -> None:
    with pytest.raises(TrackingError, match="overlap_fraction"):
        TrackingConfig(minimum_fragment_overlap_fraction=value)


def test_tracking_configuration_rejects_unknown_association_method() -> None:
    with pytest.raises(TrackingError, match="association_method"):
        TrackingConfig(association_method="unknown")


def test_height_outlier_filter_is_disabled_by_default() -> None:
    configuration = KinematicsConfig()

    assert configuration.height_outlier_filter_enabled is False
    assert configuration.height_outlier_filter_window_samples == 7
    assert configuration.height_outlier_filter_absolute_tolerance_rsun == 0.2


@pytest.mark.parametrize("window", (2, 4, 1.5, True))
def test_height_outlier_filter_rejects_invalid_window(window) -> None:
    with pytest.raises(ValueError, match="window_samples"):
        KinematicsConfig(height_outlier_filter_window_samples=window)

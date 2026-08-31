import numpy as np

from suncet_processing_pipeline.level4.cme_tracking.summary import summarize_front
from suncet_processing_pipeline.level4.cme_tracking.tracking import (
    FrontState,
    FrontTrack,
)


def _track(radius_rsun: np.ndarray, angles: np.ndarray) -> FrontTrack:
    radius_px = radius_rsun * 100.0
    observed = np.isfinite(radius_rsun)
    return FrontTrack(
        radius_px=radius_px,
        radius_rsun=radius_rsun,
        radial_sigma_px=np.where(observed, 1.0, np.nan),
        score=np.where(observed, 5.0, np.nan),
        state=np.where(observed, int(FrontState.OBSERVED), int(FrontState.MISSING)),
        position_angle_deg=angles,
        event_detected=True,
        quality_flags=(),
    )


def test_summary_preserves_missing_frames_and_uses_robust_apex() -> None:
    angles = np.arange(0.0, 360.0, 45.0)
    radii = np.full((3, 8), np.nan)
    radii[0, :4] = [1.1, 1.2, 1.3, 1.4]
    radii[2, :4] = [1.3, 1.4, 1.5, 1.6]

    summary = summarize_front(_track(radii, angles))

    assert summary.height_rsun[0] == np.percentile(radii[0, :4], 90)
    assert np.isnan(summary.height_rsun[1])
    assert summary.height_rsun[2] > summary.height_rsun[0]
    assert summary.coverage_fraction[0] == 1.0
    assert summary.observed_angle_count[1] == 0


def test_summary_position_angle_handles_zero_degree_wrap() -> None:
    angles = np.arange(0.0, 360.0, 10.0)
    radii = np.full((1, angles.size), np.nan)
    radii[0, [34, 35, 0, 1, 2]] = 2.0

    summary = summarize_front(_track(radii, angles))

    assert summary.position_angle_deg[0] < 5.0 or summary.position_angle_deg[0] > 355.0
    assert summary.angular_width_deg[0] == 50.0


def test_summary_does_not_invent_position_angle_for_symmetric_support() -> None:
    angles = np.array([0.0, 90.0, 180.0, 270.0])
    radii = np.full((1, angles.size), 2.0)

    summary = summarize_front(_track(radii, angles))

    assert np.isnan(summary.position_angle_deg[0])

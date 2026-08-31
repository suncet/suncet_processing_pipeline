import json

from astropy.table import Table
import numpy as np
import pytest

from suncet_processing_pipeline.level4.cme_tracking import products
from suncet_processing_pipeline.level4.cme_tracking.products import (
    FrontOverlayFrame,
    FrontOverlayProduct,
    FrontSamplesProduct,
    TrackProduct,
    write_event_products,
)
from suncet_processing_pipeline.level4.common.quality import QualityFlag


def test_write_event_products(tmp_path) -> None:
    count = 6
    elapsed = np.arange(count, dtype=float) * 30.0
    raw = 1.5 + 0.05 * np.arange(count)
    fitted = raw + 0.001
    track = TrackProduct(
        event_id="synthetic-event-001",
        frame_number=np.arange(count),
        elapsed_s=elapsed,
        time_utc=(None,) * count,
        height_raw_rsun=raw,
        height_raw_sigma_rsun=np.full(count, 0.01),
        height_fit_rsun=fitted,
        height_fit_sigma_rsun=np.full(count, 0.02),
        speed_fit_km_s=np.full(count, 1159.5),
        speed_sigma_km_s=np.full(count, 20.0),
        acceleration_fit_m_s2=np.zeros(count),
        acceleration_sigma_m_s2=np.full(count, 5.0),
        position_angle_deg=np.full(count, 270.0),
        angular_width_deg=np.full(count, 45.0),
        front_coverage_fraction=np.full(count, 0.8),
        confidence=np.full(count, 0.9),
        quality_mask=np.full(count, int(QualityFlag.ASSUMED_CADENCE)),
        metadata={"cadence_status": "assumed"},
    )
    front = FrontSamplesProduct(
        event_id=track.event_id,
        frame_number=np.repeat(np.arange(count), 2),
        elapsed_s=np.repeat(elapsed, 2),
        position_angle_deg=np.tile([260.0, 280.0], count),
        radius_rsun=np.repeat(raw, 2),
        radius_sigma_rsun=np.full(count * 2, 0.01),
        score=np.full(count * 2, 4.0),
        accepted=np.ones(count * 2, dtype=bool),
        quality_mask=np.zeros(count * 2, dtype=np.uint32),
    )
    overlay = FrontOverlayProduct(
        frames=(
            FrontOverlayFrame(
                image=np.arange(32 * 32, dtype=np.float32).reshape(32, 32),
                frame_number=0,
                elapsed_s=0.0,
                radius_px=np.array([8.0, 9.0]),
                position_angle_deg=np.array([80.0, 100.0]),
                headline_height_rsun=1.5,
            ),
        ),
        center_yx=(15.5, 15.5),
        north_vector_yx=(-1.0, 0.0),
        east_vector_yx=(0.0, -1.0),
    )

    def write_movie(path) -> None:
        path.write_bytes(b"synthetic movie fixture")

    directory = write_event_products(
        tmp_path,
        track,
        front,
        {"scenario_id": "fixture"},
        front_overlay=overlay,
        diagnostic_movie_writer=write_movie,
        diagnostic_movie_metadata={
            "frame_count": count,
            "fps": 10.0,
            "codec": "fixture",
        },
    )

    expected = {
        "track.ecsv",
        "front_samples.ecsv",
        "summary.json",
        "height_time.png",
        "speed_time.png",
        "acceleration_time.png",
        "acceleration_time_detail.png",
        "front_overlay.png",
        "front_tracking.mp4",
        "COMPLETE.json",
    }
    assert expected == {path.name for path in directory.iterdir()}
    recovered = Table.read(directory / "track.ecsv", format="ascii.ecsv")
    assert recovered.meta["cadence_status"] == "assumed"
    assert recovered["quality_flags"][0] == "ASSUMED_CADENCE"
    np.testing.assert_allclose(recovered["height_raw_sigma_rsun"], 0.01)
    np.testing.assert_allclose(recovered["height_fit_sigma_rsun"], 0.02)
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["scenario_id"] == "fixture"
    assert len(summary["products"]["track"]["sha256"]) == 64
    for name in (
        "height_time_plot",
        "speed_time_plot",
        "acceleration_time_plot",
        "acceleration_time_detail_plot",
        "front_overlay",
        "front_tracking_movie",
    ):
        assert len(summary["products"][name]["sha256"]) == 64
    movie = summary["products"]["front_tracking_movie"]
    assert movie["frame_count"] == count
    assert movie["fps"] == 10.0
    assert movie["path"] == "front_tracking.mp4"
    acceleration_detail = summary["products"]["acceleration_time_detail_plot"]
    assert acceleration_detail["path"] == "acceleration_time_detail.png"
    assert (
        acceleration_detail["uncertainty_display"]
        == "clipped_with_boundary_markers"
    )
    completion = json.loads((directory / "COMPLETE.json").read_text())
    assert completion["event_id"] == track.event_id
    assert len(completion["summary_sha256"]) == 64


def test_overwrite_failure_never_publishes_a_partial_product(
    tmp_path, monkeypatch
) -> None:
    count = 4
    elapsed = np.arange(count, dtype=float)
    raw = np.linspace(1.2, 1.5, count)
    track = TrackProduct(
        event_id="transaction-test",
        frame_number=np.arange(count),
        elapsed_s=elapsed,
        time_utc=(None,) * count,
        height_raw_rsun=raw,
        height_raw_sigma_rsun=np.full(count, 0.01),
        height_fit_rsun=raw,
        height_fit_sigma_rsun=np.full(count, 0.02),
        speed_fit_km_s=np.ones(count),
        speed_sigma_km_s=np.ones(count),
        acceleration_fit_m_s2=np.zeros(count),
        acceleration_sigma_m_s2=np.ones(count),
        position_angle_deg=np.full(count, 250.0),
        angular_width_deg=np.full(count, 30.0),
        front_coverage_fraction=np.ones(count),
        confidence=np.ones(count),
        quality_mask=np.zeros(count, dtype=np.uint32),
    )
    empty = np.array([], dtype=float)
    front = FrontSamplesProduct(
        event_id=track.event_id,
        frame_number=np.array([], dtype=int),
        elapsed_s=empty,
        position_angle_deg=empty,
        radius_rsun=empty,
        radius_sigma_rsun=empty,
        score=empty,
        accepted=np.array([], dtype=bool),
        quality_mask=np.array([], dtype=np.uint32),
    )
    published = write_event_products(tmp_path, track, front, {"revision": 1})
    original_summary = (published / "summary.json").read_bytes()

    def fail_plot(*_args, **_kwargs):
        raise RuntimeError("injected plot failure")

    monkeypatch.setattr(products, "_plot_series", fail_plot)
    with pytest.raises(RuntimeError, match="injected plot failure"):
        write_event_products(
            tmp_path,
            track,
            front,
            {"revision": 2},
            overwrite=True,
        )

    assert (published / "summary.json").read_bytes() == original_summary
    assert not list(tmp_path.glob(".transaction-test.staging-*"))
    assert not list(tmp_path.glob(".transaction-test.backup-*"))


def test_movie_failure_never_replaces_a_published_product(tmp_path) -> None:
    count = 4
    elapsed = np.arange(count, dtype=float)
    values = np.linspace(1.2, 1.5, count)
    track = TrackProduct(
        event_id="movie-transaction-test",
        frame_number=np.arange(count),
        elapsed_s=elapsed,
        time_utc=(None,) * count,
        height_raw_rsun=values,
        height_raw_sigma_rsun=np.full(count, 0.01),
        height_fit_rsun=values,
        height_fit_sigma_rsun=np.full(count, 0.02),
        speed_fit_km_s=np.ones(count),
        speed_sigma_km_s=np.ones(count),
        acceleration_fit_m_s2=np.zeros(count),
        acceleration_sigma_m_s2=np.ones(count),
        position_angle_deg=np.full(count, 250.0),
        angular_width_deg=np.full(count, 30.0),
        front_coverage_fraction=np.ones(count),
        confidence=np.ones(count),
        quality_mask=np.zeros(count, dtype=np.uint32),
    )
    empty = np.array([], dtype=float)
    front = FrontSamplesProduct(
        event_id=track.event_id,
        frame_number=np.array([], dtype=int),
        elapsed_s=empty,
        position_angle_deg=empty,
        radius_rsun=empty,
        radius_sigma_rsun=empty,
        score=empty,
        accepted=np.array([], dtype=bool),
        quality_mask=np.array([], dtype=np.uint32),
    )
    published = write_event_products(tmp_path, track, front, {"revision": 1})
    original_summary = (published / "summary.json").read_bytes()

    def fail_movie(path) -> None:
        path.write_bytes(b"partial")
        raise RuntimeError("injected movie failure")

    with pytest.raises(RuntimeError, match="injected movie failure"):
        write_event_products(
            tmp_path,
            track,
            front,
            {"revision": 2},
            diagnostic_movie_writer=fail_movie,
            overwrite=True,
        )

    assert (published / "summary.json").read_bytes() == original_summary
    assert not (published / "front_tracking.mp4").exists()
    assert not list(tmp_path.glob(".movie-transaction-test.staging-*"))
    assert not list(tmp_path.glob(".movie-transaction-test.backup-*"))

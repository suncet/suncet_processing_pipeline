"""Focused tests for the static CME method-explanation renderer."""

import numpy as np
from PIL import Image

from suncet_processing_pipeline.level4.cme_tracking.config import (
    CMETrackingConfig,
    KinematicsConfig,
)
from suncet_processing_pipeline.level4.cme_tracking.explanation import (
    write_cme_method_explanation,
)
from suncet_processing_pipeline.level4.cme_tracking.input import (
    load_synthetic_sequence,
)
from suncet_processing_pipeline.level4.cme_tracking.likelihood import (
    EvidenceConfig,
)
from suncet_processing_pipeline.level4.cme_tracking.pipeline import (
    run_known_window,
)
from suncet_processing_pipeline.level4.cme_tracking.polar import PolarConfig
from suncet_processing_pipeline.level4.cme_tracking.tracking import TrackingConfig

from .pipeline_test import _write_expanding_fits_sequence


def test_method_explanation_writes_numbered_middle_frame_walkthrough(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _write_expanding_fits_sequence(tmp_path, frame_count=10)
    sequence = load_synthetic_sequence(
        paths,
        scenario_id="analytic-explanation-front",
        assumed_cadence_seconds=30.0,
    )
    configuration = CMETrackingConfig(
        polar=PolarConfig(
            position_angle_step_deg=4.0,
            minimum_radius_rsun=1.05,
            maximum_radius_rsun=3.0,
        ),
        evidence=EvidenceConfig(minimum_leading_edge_z=0.5),
        tracking=TrackingConfig(
            score_threshold=1.5,
            maximum_outward_step_px_per_frame=4.0,
            minimum_track_points=5,
            minimum_outward_displacement_px=3.0,
            minimum_angular_support_deg=16.0,
            minimum_event_frames=6,
        ),
        kinematics=KinematicsConfig(
            endpoint_samples=1,
            uncertainty_samples=8,
            random_seed=4,
        ),
        field_of_view_margin_px=3.0,
    )
    run = run_known_window(sequence, configuration)
    middle_index = sequence.frame_count // 2
    score_before = run.likelihood.score.copy()
    front_before = run.front.radius_rsun.copy()
    summary_before = run.summary.height_rsun.copy()

    def forbid_cartesian_materialization(*_args, **_kwargs):
        raise AssertionError("renderer must stream Cartesian input frames")

    monkeypatch.setattr(
        type(sequence),
        "materialize",
        forbid_cartesian_materialization,
    )
    artifacts = write_cme_method_explanation(
        run,
        middle_index,
        tmp_path / "explanation",
        dpi=36,
    )

    assert artifacts.frame_index == middle_index
    assert artifacts.frame_number == sequence.frames[middle_index].frame_number
    assert artifacts.overview_path.name == "00_method_overview.png"
    assert [path.name[:2] for path in artifacts.panel_paths] == [
        f"{number:02d}" for number in range(1, 12)
    ]
    for path in (*artifacts.panel_paths, artifacts.overview_path):
        assert path.is_file()
        with Image.open(path) as image:
            image.verify()

    np.testing.assert_array_equal(run.likelihood.score, score_before)
    np.testing.assert_allclose(run.front.radius_rsun, front_before, equal_nan=True)
    np.testing.assert_allclose(
        run.summary.height_rsun,
        summary_before,
        equal_nan=True,
    )

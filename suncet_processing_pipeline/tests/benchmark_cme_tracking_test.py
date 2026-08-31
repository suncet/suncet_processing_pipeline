import json
import resource
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from suncet_processing_pipeline.benchmark_cme_tracking import (
    PeakRSSMonitor,
    TegrastatsCollector,
    _array_sha256,
    _deadline_metrics,
    _science_arrays,
    _safe_command,
    _summary_statistics,
    compare_science_reference,
    science_signature,
)


def _minimal_run():
    return SimpleNamespace(
        likelihood=SimpleNamespace(score=np.array([[1.0, np.nan]])),
        front=SimpleNamespace(
            state=np.array([[1, 0]], dtype=np.uint8),
            radius_px=np.array([[2.0, np.nan]]),
            score=np.array([[3.0, np.nan]]),
            observed_mask=np.array([[True, False]]),
            event_detected=True,
        ),
        summary=SimpleNamespace(
            height_rsun=np.array([2.0]),
            height_sigma_rsun=np.array([0.1]),
            position_angle_deg=np.array([90.0]),
            angular_width_deg=np.array([20.0]),
            coverage_fraction=np.array([0.5]),
            confidence=np.array([0.8]),
        ),
        field_of_view_limited_mask=np.array([False]),
        kinematics=None,
        kinematics_error="fixture has no fit",
    )


def test_deadline_metrics_keep_replay_cadence_separate_from_science_time():
    metrics = _deadline_metrics(
        100.0,
        frame_count=10,
        cadences_seconds=(15.0, 10.0),
    )

    assert metrics["15"]["batch_utilization_fraction"] == 2 / 3
    assert metrics["15"]["bounded_backlog_by_average_rate"] is True
    assert metrics["10"]["batch_utilization_fraction"] == 1.0
    assert metrics["10"]["bounded_backlog_by_average_rate"] is False


def test_science_signature_is_stable_and_changes_with_science_arrays():
    run = _minimal_run()
    first = science_signature(run)
    second = science_signature(run)

    assert first == second
    assert len(first["overall_sha256"]) == 64

    run.summary.height_rsun[0] = 2.1
    changed = science_signature(run)
    assert changed["overall_sha256"] != first["overall_sha256"]
    assert (
        changed["component_sha256"]["summary_height_rsun"]
        != first["component_sha256"]["summary_height_rsun"]
    )

    run.summary.height_rsun[0] = 2.0
    run.front.event_detected = False
    changed_state = science_signature(run)
    assert changed_state["component_sha256"] == first["component_sha256"]
    assert changed_state["overall_sha256"] != first["overall_sha256"]


def test_array_hash_includes_dtype_and_shape():
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    assert _array_sha256(values) != _array_sha256(values.astype(np.int64))
    assert _array_sha256(values) != _array_sha256(values.reshape(2, 2))


def test_independent_reference_comparison_reports_exact_and_numeric_difference(
    tmp_path,
):
    run = _minimal_run()
    npz_path = tmp_path / "reference.npz"
    json_path = tmp_path / "reference.json"
    np.savez_compressed(npz_path, **_science_arrays(run))
    json_path.write_text(
        json.dumps(science_signature(run)),
        encoding="utf-8",
    )

    exact = compare_science_reference(
        run,
        expected_npz_path=npz_path,
        expected_json_path=json_path,
    )
    assert exact["exact_match"] is True
    assert exact["numerically_equivalent"] is True

    run.likelihood.score[0, 0] += 2e-7
    roundoff = compare_science_reference(
        run,
        expected_npz_path=npz_path,
        expected_json_path=json_path,
    )
    assert roundoff["exact_match"] is False
    assert roundoff["numerically_equivalent"] is True

    run.likelihood.score[0, 0] = 1.0
    run.summary.height_rsun[0] += 0.25
    changed = compare_science_reference(
        run,
        expected_npz_path=npz_path,
        expected_json_path=json_path,
    )
    assert changed["exact_match"] is False
    assert changed["numerically_equivalent"] is False
    assert changed["components"]["summary_height_rsun"][
        "maximum_absolute_difference"
    ] == 0.25


def test_summary_statistics_are_distributional_and_empty_safe():
    summary = _summary_statistics([1.0, 2.0, 10.0])
    assert summary["minimum"] == 1.0
    assert summary["median"] == 2.0
    assert summary["maximum"] == 10.0
    assert summary["iqr"] == 4.5
    assert all(value is None for value in _summary_statistics([]).values())


def test_missing_tegrastats_is_a_recorded_nonfatal_condition(monkeypatch):
    monkeypatch.setattr(
        "suncet_processing_pipeline.benchmark_cme_tracking.shutil.which",
        lambda _name: None,
    )
    collector = TegrastatsCollector(200)
    collector.start()

    assert collector.available is False
    assert collector.error == "tegrastats executable not found"
    assert collector.snapshot() == []


def test_peak_rss_monitor_uses_kernel_high_water_without_polling(monkeypatch):
    maximum_rss_values = iter((1_000, 2_000))
    calls = []

    def fake_getrusage(who):
        calls.append(who)
        return SimpleNamespace(ru_maxrss=next(maximum_rss_values))

    monkeypatch.setattr(
        "suncet_processing_pipeline.benchmark_cme_tracking.resource.getrusage",
        fake_getrusage,
    )
    monitor = PeakRSSMonitor()

    monitor.start()
    monitor.stop()

    multiplier = 1 if sys.platform == "darwin" else 1024
    assert calls == [resource.RUSAGE_SELF, resource.RUSAGE_SELF]
    assert monitor.initial_peak_rss_bytes == 1_000 * multiplier
    assert monitor.peak_rss_bytes == 2_000 * multiplier


def test_timed_out_optional_probe_remains_json_safe(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["probe"],
            timeout=1,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(
        "suncet_processing_pipeline.benchmark_cme_tracking.subprocess.run",
        timeout,
    )
    result = _safe_command(("probe",), timeout=1)

    assert result["timed_out"] is True
    assert result["stdout"] == "partial stdout"
    json.dumps(result, allow_nan=False)

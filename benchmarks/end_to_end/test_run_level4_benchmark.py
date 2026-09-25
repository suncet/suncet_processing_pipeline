"""Focused tests for the benchmark-only Level 4 entry point."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

from astropy.io import fits
import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_level4_benchmark  # noqa: E402

from suncet_processing_pipeline.level4.cme_tracking.manifest import (  # noqa: E402
    CorrectionState,
    InputSourceKind,
    TimeAxisKind,
)
from suncet_processing_pipeline.level4.cme_tracking.pipeline import (  # noqa: E402
    _base_quality_mask,
)
from suncet_processing_pipeline.level4.common.quality import (  # noqa: E402
    QualityFlag,
)


def _write_passthrough(
    path: Path,
    date_obs: str,
    *,
    level: object = 3,
    deconvolved: bool = True,
    benchmark_only: bool = True,
    passthrough: bool = True,
    geometry_applied: bool = False,
    dark_applied: bool = False,
    corrections: str = "NONE",
) -> None:
    data = np.arange(12 * 16, dtype=np.float32).reshape(12, 16)
    header = fits.Header()
    header["LEVEL"] = level
    header["DECONV"] = deconvolved
    header["BMRKONLY"] = benchmark_only
    header["L3PASS"] = passthrough
    header["L3GEOM"] = geometry_applied
    header["L3DARK"] = dark_applied
    header["L3CORR"] = corrections
    header["TIMESYS"] = "UTC"
    header["DATE-OBS"] = date_obs
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CRPIX1"] = 8.5
    header["CRPIX2"] = 6.5
    header["CDELT1"] = -9.6 / 3600.0
    header["CDELT2"] = 9.6 / 3600.0
    header["RSUN"] = 43.9
    fits.PrimaryHDU(data=data, header=header).writeto(path, checksum=True)


def test_loader_preserves_honest_processing_and_quality_state(tmp_path: Path):
    _write_passthrough(tmp_path / "frame_002.fits", "2027-02-15T00:00:15.000")
    _write_passthrough(tmp_path / "frame_001.fits", "2027-02-15T00:00:00.000")

    sequence = run_level4_benchmark.load_benchmark_sequence(
        tmp_path,
        pattern="*.fits",
        scenario_id="synthetic-e2e",
    )

    assert [path.name for path in sequence.paths] == [
        "frame_001.fits",
        "frame_002.fits",
    ]
    np.testing.assert_allclose(sequence.elapsed_seconds, [0.0, 15.0])
    assert sequence.source_kind is InputSourceKind.SYNTHETIC_BYPASS
    assert sequence.time_axis.kind is TimeAxisKind.FITS_HEADERS
    assert sequence.time_axis.absolute_time_valid is True
    assert sequence.time_axis.cadence_status is None
    assert (
        sequence.upstream_processing.level2_psf_deconvolution
        is CorrectionState.APPLIED
    )
    assert (
        sequence.upstream_processing.level3_geometric_correction
        is CorrectionState.NOT_APPLIED
    )

    mask = _base_quality_mask(SimpleNamespace(sequence=sequence))
    assert mask & QualityFlag.SYNTHETIC_BYPASS
    assert mask & QualityFlag.LEVEL3_GEOMETRY_NOT_APPLIED
    assert mask & QualityFlag.INPUT_INTEGRITY_UNVERIFIED
    assert not mask & QualityFlag.LEVEL2_PSF_NOT_APPLIED
    assert not mask & QualityFlag.ASSUMED_CADENCE
    assert not mask & QualityFlag.ABSOLUTE_TIME_UNAVAILABLE


@pytest.mark.parametrize(
    ("options", "message"),
    (
        ({"level": 2}, "LEVEL=3"),
        ({"deconvolved": False}, "DECONV=True"),
        ({"benchmark_only": False}, "BMRKONLY=True"),
        ({"passthrough": False}, "L3PASS=True"),
        ({"geometry_applied": True}, "L3GEOM=False"),
        ({"dark_applied": True}, "L3DARK=False"),
        ({"corrections": "ROTATE"}, "L3CORR='NONE'"),
    ),
)
def test_rejects_ambiguous_or_non_passthrough_headers(
    tmp_path: Path,
    options: dict[str, object],
    message: str,
):
    path = tmp_path / "frame.fits"
    _write_passthrough(path, "2027-02-15T00:00:00.000", **options)

    with pytest.raises(ValueError, match=message):
        run_level4_benchmark.validate_benchmark_level3_header(path)


def test_main_uses_deterministic_event_path_and_disables_diagnostics(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    sequence = SimpleNamespace(scenario_id="synthetic-e2e")
    configuration = object()
    run = SimpleNamespace(
        front=SimpleNamespace(event_detected=True),
        kinematics_error=None,
    )
    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        run_level4_benchmark,
        "load_benchmark_sequence",
        lambda *args, **kwargs: sequence,
    )
    monkeypatch.setattr(
        run_level4_benchmark,
        "read_configuration",
        lambda path: configuration,
    )
    monkeypatch.setattr(
        run_level4_benchmark,
        "run_known_window",
        lambda supplied_sequence, supplied_configuration: run,
    )

    def _write_products(supplied_run, output_root, event_id, **kwargs):
        recorded.update(
            {
                "run": supplied_run,
                "output_root": output_root,
                "event_id": event_id,
                **kwargs,
            }
        )
        return Path(output_root) / event_id

    monkeypatch.setattr(
        run_level4_benchmark,
        "write_known_window_products",
        _write_products,
    )

    result = run_level4_benchmark.main(
        [
            "--input-directory",
            str(tmp_path / "level3"),
            "--output-root",
            str(tmp_path / "level4"),
            "--config",
            str(tmp_path / "tracker.json"),
            "--scenario-id",
            "synthetic-e2e",
        ]
    )

    assert result == 0
    assert recorded["run"] is run
    assert recorded["output_root"] == tmp_path / "level4"
    assert recorded["event_id"] == "synthetic-e2e-event-001"
    assert recorded["include_diagnostic_plots"] is False
    assert recorded["include_diagnostic_movie"] is False
    assert recorded["overwrite"] is False
    assert "detected" in capsys.readouterr().out

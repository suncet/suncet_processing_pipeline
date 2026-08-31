"""Command-line contract tests for Level 4 dispatch."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from suncet_processing_pipeline import make_level4


def test_successful_no_event_window_returns_zero(tmp_path, monkeypatch, capsys):
    sequence = SimpleNamespace(scenario_id="quiet-window")
    run = SimpleNamespace(
        front=SimpleNamespace(event_detected=False),
        kinematics_error="No coherent event was detected.",
    )

    monkeypatch.setattr(
        make_level4,
        "_load_sequence",
        lambda _arguments, _parser: sequence,
    )
    monkeypatch.setattr(
        make_level4,
        "run_known_window",
        lambda _sequence, _configuration: run,
    )
    recorded = {}

    def write_products(*_args, **kwargs):
        recorded.update(kwargs)
        return tmp_path / "quiet-window-event-001"

    monkeypatch.setattr(
        make_level4,
        "write_known_window_products",
        write_products,
    )

    status = make_level4.main(
        [
            "cme-track",
            "--manifest",
            str(Path("scenario.json")),
            "--diagnostic-movie",
            "--movie-fps",
            "12.5",
        ]
    )

    assert status == 0
    output = capsys.readouterr().out
    assert "not detected" in output
    assert "No coherent event was detected" in output
    assert recorded["include_diagnostic_movie"] is True
    assert recorded["movie_fps"] == 12.5


def test_movie_fps_must_be_positive() -> None:
    with pytest.raises(SystemExit):
        make_level4.main(
            [
                "cme-track",
                "--manifest",
                "scenario.json",
                "--movie-fps",
                "0",
            ]
        )


def test_direct_synthetic_cli_omits_cadence_to_use_fits_headers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sentinel = object()
    paths = (tmp_path / "frame_000.fits", tmp_path / "frame_001.fits")
    recorded = {}
    monkeypatch.setattr(
        make_level4,
        "discover_fits_files",
        lambda directory, pattern: paths,
    )

    def load_synthetic(ordered_paths, **kwargs):
        recorded["paths"] = ordered_paths
        recorded.update(kwargs)
        return sentinel

    monkeypatch.setattr(make_level4, "load_synthetic_sequence", load_synthetic)
    parser = make_level4._parser()
    arguments = parser.parse_args(
        [
            "cme-track",
            "--synthetic-directory",
            str(tmp_path),
            "--scenario-id",
            "header-timed",
        ]
    )

    result = make_level4._load_sequence(arguments, parser)

    assert result is sentinel
    assert recorded["paths"] == paths
    assert recorded["assumed_cadence_seconds"] is None

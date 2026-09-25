from pathlib import Path
import sys

import pytest
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_synthetic_level1 as prepare  # noqa: E402


def _write_config(path: Path, *, filtering: bool) -> None:
    path.write_text(
        "\n".join(
            (
                "[behavior]",
                f"filter_out_particle_hits = {filtering}",
                "[shdr]",
                "exposure_time_short = 0.035",
                "exposure_time_long = 15",
                "num_short_exposures_to_stack = 9",
                "num_long_exposures_to_stack = 4",
                "num_shift_bits_32_to_16 = 2",
                "inner_fov_circle_radius = 1.33",
                "[detector]",
                "num_pixels_to_bin = [2, 2]",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_unfiltered_simulator_truth_uses_runtime_single_frame_stacks(tmp_path):
    config = tmp_path / "simulator.ini"
    _write_config(config, filtering=False)

    truth = prepare._read_simulator_truth(config)

    assert truth["configured_short_stack_count"] == 9
    assert truth["configured_long_stack_count"] == 4
    assert truth["short_stack_count"] == 1
    assert truth["long_stack_count"] == 1
    assert truth["effective_exposure_inner_seconds"] == pytest.approx(0.00875)
    assert truth["effective_exposure_outer_seconds"] == pytest.approx(3.75)


def test_filtered_simulator_truth_uses_sum_minus_maximum(tmp_path):
    config = tmp_path / "simulator.ini"
    _write_config(config, filtering=True)

    truth = prepare._read_simulator_truth(config)

    assert truth["short_stack_count"] == 9
    assert truth["long_stack_count"] == 4
    assert truth["effective_exposure_inner_seconds"] == pytest.approx(0.07)
    assert truth["effective_exposure_outer_seconds"] == pytest.approx(11.25)


def test_legacy_elapsed_time_is_derived_from_utc_header_dates():
    header = fits.Header()
    header["TELAPSE"] = "N/A"
    header["DATE-OBS"] = "2023-01-14T17:00:00.000"
    header["DATE-END"] = "2023-01-14T17:00:15.000"

    assert prepare._elapsed_seconds_from_header(header) == 15.0


def test_directory_preparation_is_sorted_and_reuses_static_inputs(
    tmp_path, monkeypatch
):
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    for name in ("b.fits", "a.fits"):
        (input_directory / name).touch()
    simulator_config = tmp_path / "simulator.ini"
    _write_config(simulator_config, filtering=False)
    metadata_definition = tmp_path / "metadata.csv"
    metadata_definition.write_text("placeholder\n", encoding="utf-8")

    observed = []

    def fake_prepare(input_path, output_path, *args, **kwargs):
        observed.append((input_path.name, output_path.name, kwargs))
        return output_path, output_path.with_suffix(".provenance.json")

    monkeypatch.setattr(prepare, "prepare_level1", fake_prepare)
    monkeypatch.setattr(prepare, "_sha256", lambda path: f"hash:{path.name}")

    products = prepare.prepare_level1_directory(
        input_directory,
        output_directory,
        simulator_config,
        metadata_definition,
        simulator_commit="abc123",
        pipeline_version="2.0.0",
    )

    assert [item[0] for item in observed] == ["a.fits", "b.fits"]
    assert [item[1] for item in observed] == [
        "a_level1_v2.0.0_provisional.fits",
        "b_level1_v2.0.0_provisional.fits",
    ]
    assert len(products) == 2
    assert observed[0][2]["_truth"] is observed[1][2]["_truth"]
    assert observed[0][2]["_simulator_config_sha256"] == "hash:simulator.ini"
    assert observed[0][2]["_metadata_definition_sha256"] == "hash:metadata.csv"

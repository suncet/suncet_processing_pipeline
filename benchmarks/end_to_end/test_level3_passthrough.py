"""Focused tests for the benchmark-only Level 3 pass-through adapter."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

from astropy.io import fits
import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import level3_passthrough  # noqa: E402


_DEFINITION_COLUMNS = (
    "Field Name",
    "Internal Variable Name",
    "Description",
    "Minimum Level",
    "FITS variable name",
    "units (human)",
    "units (astropy)",
    "data type",
    "typical value",
    "provenance (fixed value, derived, ancillary, spacecraft data, FITS generated)",
)


def _write_definition(path: Path) -> Path:
    rows = (
        ("Level", "level", "Processing level", "0.5", "LEVEL", "int"),
        ("Minimum", "minimum", "Minimum value", "0.5", "DATAMIN", "float"),
        ("Maximum", "maximum", "Maximum value", "0.5", "DATAMAX", "float"),
        ("PSF", "psf", "PSF manifest", "2", "CALPSF", "string"),
        ("Checksum", "checksum", "Checksum", "0.5", "CHECKSUM", "string"),
        ("Datasum", "datasum", "Datasum", "0.5", "DATASUM", "string"),
        ("History", "history", "History", "0.5", "HISTORY", "string"),
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_DEFINITION_COLUMNS)
        writer.writeheader()
        for field, internal, description, level, fits_name, data_type in rows:
            writer.writerow(
                {
                    "Field Name": field,
                    "Internal Variable Name": internal,
                    "Description": description,
                    "Minimum Level": level,
                    "FITS variable name": fits_name,
                    "data type": data_type,
                }
            )
    return path


def _write_level2(path: Path, *, value: float, level=2, deconvolved=True) -> np.ndarray:
    data = np.arange(30, dtype=np.float32).reshape(5, 6) + value
    header = fits.Header()
    header["LEVEL"] = level
    header["TITLE"] = "SunCET Level 2 Image"
    header["FILENAME"] = path.name
    header["TIMESYS"] = "UTC"
    header["DATE-BEG"] = "2027-02-15T00:00:00.000"
    header["DATE-OBS"] = "2027-02-15T00:00:00.000"
    header["DATE-END"] = "2027-02-15T00:00:15.000"
    header["BUNIT"] = "DN/s"
    header["DECONV"] = deconvolved
    header["CALPSF"] = "level2_psf_calibration_example.json"
    header["DATAMIN"] = float(np.min(data))
    header["DATAMAX"] = float(np.max(data))
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 3.5
    header["CRPIX2"] = 3.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = 4.8
    header["CDELT2"] = 4.8
    header.add_history("Applied diffraction and scatter PSF deconvolution")
    fits.PrimaryHDU(data=data, header=header).writeto(path, checksum=True)
    return data


def _processor(
    tmp_path: Path,
    **options,
) -> level3_passthrough.BenchmarkLevel3PassThrough:
    definition = _write_definition(tmp_path / "metadata.csv")
    return level3_passthrough.BenchmarkLevel3PassThrough(
        metadata_definition_file=definition,
        generated_at=lambda: "2026-09-25T12:34:56.789Z",
        **options,
    )


def test_directory_run_is_sorted_and_preserves_array_wcs_and_checksums(tmp_path):
    input_directory = tmp_path / "level2"
    output_directory = tmp_path / "level3"
    input_directory.mkdir()
    expected_b = _write_level2(
        input_directory / "frame_b_level2_v2.0.0.fits",
        value=100,
    )
    expected_a = _write_level2(
        input_directory / "frame_a_level2_v2.0.0.fits",
        value=0,
    )
    processor = _processor(tmp_path)

    outputs = processor.run(input_directory, output_directory)

    assert [path.name for path in outputs] == [
        "frame_a_level3_benchmark_passthrough_v2.0.0.fits",
        "frame_b_level3_benchmark_passthrough_v2.0.0.fits",
    ]
    for output, parent_name, expected in zip(
        outputs,
        ("frame_a_level2_v2.0.0.fits", "frame_b_level2_v2.0.0.fits"),
        (expected_a, expected_b),
        strict=True,
    ):
        with fits.open(output, checksum=True) as hdul:
            hdul.verify("exception")
            header = hdul[0].header
            np.testing.assert_array_equal(hdul[0].data, expected)
            assert hdul[0].data.dtype.kind == expected.dtype.kind
            assert hdul[0].data.dtype.itemsize == expected.dtype.itemsize
            assert header["BITPIX"] == -32
            assert header["LEVEL"] == 3
            assert header["L2PARENT"] == parent_name
            assert header["PROCSTAT"] == "PROVISIONAL"
            assert header["BMRKONLY"] is True
            assert header["L3PASS"] is True
            assert header["L3GEOM"] is False
            assert header["L3DARK"] is False
            assert header["L3CORR"] == "NONE"
            assert header["DECONV"] is True
            assert header["DATE"] == "2026-09-25T12:34:56.789Z"
            assert header["CTYPE1"] == "HPLN-TAN"
            assert header["CTYPE2"] == "HPLT-TAN"
            assert header["CRPIX1"] == 3.5
            assert header["CRPIX2"] == 3.0
            assert header["CDELT1"] == 4.8
            assert header["CDELT2"] == 4.8
            histories = list(header["HISTORY"])
            assert any("BENCHMARK ONLY" in line for line in histories)
            assert any("No Level 3 fine rotation" in line for line in histories)
            assert hdul[0].verify_checksum() == 1
            assert hdul[0].verify_datasum() == 1


@pytest.mark.parametrize(
    ("level", "deconvolved", "message"),
    (
        (1, True, "must declare LEVEL=2"),
        (2, False, "DECONV=T"),
    ),
)
def test_rejects_non_level2_or_non_deconvolved_input(
    tmp_path,
    level,
    deconvolved,
    message,
):
    input_directory = tmp_path / "level2"
    input_directory.mkdir()
    _write_level2(
        input_directory / "frame.fits",
        value=0,
        level=level,
        deconvolved=deconvolved,
    )
    processor = _processor(tmp_path)

    with pytest.raises(ValueError, match=message):
        processor.run(input_directory, tmp_path / "level3")


def test_refuses_existing_output_without_explicit_overwrite(tmp_path):
    input_directory = tmp_path / "level2"
    output_directory = tmp_path / "level3"
    input_directory.mkdir()
    input_file = input_directory / "frame.fits"
    _write_level2(input_file, value=0)
    processor = _processor(tmp_path)
    [output] = processor.run(input_directory, output_directory)
    original = output.read_bytes()

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        processor.run(input_directory, output_directory)

    assert output.read_bytes() == original
    assert not list(output_directory.glob("*.lock"))

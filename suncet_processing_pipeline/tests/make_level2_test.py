from types import SimpleNamespace
import csv
import json

from astropy.io import fits
import numpy as np
import pytest

from .. import make_level2, metadata_managers


DEFINITION_COLUMNS = (
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


def _write_metadata_definition(path):
    definitions = (
        ("Level", "level", "Processing level", "0.5", "LEVEL", "int"),
        ("Type", "product_type", "Product type", "0.5", "TYPECODE", "string"),
        ("Internal LED", "int_led", "Internal LED", "0.5", "INT_LED", "boolean"),
        ("External LED", "ext_led", "External LED", "0.5", "EXT_LED", "boolean"),
        ("Minimum", "minimum", "Minimum", "0.5", "DATAMIN", "int"),
        ("Maximum", "maximum", "Maximum", "0.5", "DATAMAX", "int"),
        ("Saturated", "saturated", "Saturated pixels", "1", "DATASAT", "int"),
        ("Saturation", "saturation", "Saturation value", "1", "DSATVAL", "float"),
        ("Exposure mask", "mask", "Exposure mask", "1", "EXP_MASK", "string"),
        ("PSF", "psf", "PSF manifest", "2", "CALPSF", "string"),
        ("Checksum", "checksum", "Checksum", "0.5", "CHECKSUM", "string"),
        ("Datasum", "datasum", "Datasum", "0.5", "DATASUM", "string"),
        ("History", "history", "History", "0.5", "HISTORY", "string"),
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=DEFINITION_COLUMNS)
        writer.writeheader()
        for field, internal, description, level, fits_name, data_type in definitions:
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


def _write_input(
    path,
    *,
    level,
    bunit="DN/s",
    shape=make_level2.Level2.EXPECTED_IMAGE_SHAPE,
    synthetic=None,
):
    header = fits.Header()
    header["LEVEL"] = level
    header["BUNIT"] = bunit
    header["TITLE"] = "Stale upstream title"
    header["FILENAME"] = path.name
    header["PIPEVRSN"] = "v0"
    header["TIMESYS"] = ("UTC", "Principal time system: UTC")
    header["DATE-BEG"] = "2023-01-14T17:30:00.000"
    header["DATE-OBS"] = "2023-01-14T17:30:00.000"
    header["DATE-END"] = "2023-01-14T17:30:15.000"
    header["TELAPSE"] = "N/A"
    header["IMAGEW"] = shape[1]
    header["IMAGEH"] = shape[0]
    header["DATASAT"] = 1
    header["DSATVAL"] = 100
    header["EXP_MASK"] = "ExpTime_missing.fits"
    header["TYPECODE"] = 42
    header["INT_LED"] = "FALSE"
    header["EXT_LED"] = "TRUE"
    if synthetic is not None:
        header["SYNTHET"] = synthetic
    data = np.ones(shape, dtype=np.float32)
    if data.size >= 6:
        data.flat[:6] = np.arange(6, dtype=np.float32)
    fits.PrimaryHDU(
        data=data,
        header=header,
    ).writeto(path, checksum=True)
    return data


def _level2(
    tmp_path,
    *,
    input_kind=make_level2.Level2.INPUT_KIND_LEVEL1,
    **options,
):
    assets = []
    for name in (
        "diffraction.fits",
        "scatter.fits",
        "spectrum.genx",
        "response.genx",
    ):
        path = tmp_path / name
        path.touch()
        assets.append(path)
    return make_level2.Level2(
        config=SimpleNamespace(version="2.0.0"),
        diffraction_psf_file=assets[0],
        scatter_psf_file=assets[1],
        spec_file=assets[2],
        resp_file=assets[3],
        input_kind=input_kind,
        metadata_definition_file=_write_metadata_definition(
            tmp_path / "metadata.csv"
        ),
        **options,
    )


def test_level2_requires_declared_level1_by_default(tmp_path):
    input_path = tmp_path / "input.fits"
    _write_input(input_path, level="0.5")
    processor = _level2(tmp_path)

    with pytest.raises(ValueError, match="must declare LEVEL=1"):
        processor.run(input_path, tmp_path / "output")


def test_synthetic_bypass_requires_rate_data(tmp_path):
    input_path = tmp_path / "input.fits"
    _write_input(input_path, level="0.5", bunit="DN")
    processor = _level2(
        tmp_path,
        input_kind=make_level2.Level2.INPUT_KIND_SYNTHETIC_LEVEL0_5_BYPASS,
    )

    with pytest.raises(ValueError, match="normalized to a DN/time rate"):
        processor.run(input_path, tmp_path / "output")


def test_provisional_synthetic_handoff_header_and_checksums(tmp_path, monkeypatch):
    input_path = tmp_path / "input.fits"
    _write_input(input_path, level="0.5", bunit="dn/sec")
    processor = _level2(
        tmp_path,
        input_kind=make_level2.Level2.INPUT_KIND_SYNTHETIC_LEVEL0_5_BYPASS,
    )

    def fake_deconvolution(data, *_args, **_kwargs):
        output = data.copy()
        output[0, 0] = -0.25
        output[0, 1] += 0.25
        output[1, 0] = 0
        output[1, 1] += 1
        return output

    monkeypatch.setattr(
        make_level2.suncet_deconv,
        "apply_deconv",
        fake_deconvolution,
    )

    outputs = processor.run(input_path, tmp_path / "output")

    assert len(outputs) == 1
    output_path = outputs[0]
    with fits.open(output_path, checksum=True) as hdul:
        hdul.verify("exception")
        header = hdul[0].header
        expected = np.ones(make_level2.Level2.EXPECTED_IMAGE_SHAPE)
        expected.flat[:6] = np.arange(6, dtype=np.float64)
        expected[0, 0] = -0.25
        expected[0, 1] += 0.25
        expected[1, 0] = 0
        expected[1, 1] += 1
        np.testing.assert_allclose(hdul[0].data, expected)
        assert header["LEVEL"] == 2
        assert header["TITLE"] == "SunCET Level 2 Image"
        assert header["FILENAME"] == output_path.name
        assert header["PIPEVRSN"] == "v2.0.0"
        assert header["PROCSTAT"] == "PROVISIONAL"
        assert header["TIMESYS"] == "UTC"
        assert "UTC" in header.comments["TIMESYS"]
        assert "UTC" in header.comments["DATE-BEG"]
        assert "UTC" in header.comments["DATE-OBS"]
        assert "UTC" in header.comments["DATE-END"]
        assert header["SYNTHET"] is True
        assert header["L1BYPASS"] is True
        assert header["L0PARENT"] == input_path.name
        assert header["TELAPSE"] == 15.0
        assert header["IMAGEW"] == 1000
        assert header["IMAGEH"] == 750
        assert header["DIFPSF"] == "diffraction.fits"
        assert header["SCATPSF"] == "scatter.fits"
        assert header["SPECFIL"] == "spectrum.genx"
        assert header["RESPFIL"] == "response.genx"
        calibration_manifest = output_path.parent / header["CALPSF"]
        assert header["CALID"] == calibration_manifest.stem.removeprefix(
            "level2_psf_calibration_"
        )
        assert header["DATASAT"] == 1
        assert header["DSATVAL"] == 100.0
        assert isinstance(header["DSATVAL"], float)
        assert header["EXP_MASK"] == "NOT_APPLIED_SYNTHETIC_BYPASS"
        assert header["DOI"] == "NOT_ASSIGNED"
        assert header["OBSTYPE"] == "SYNTHETIC DEVELOPMENT FIXTURE"
        assert header["FILE_RAW"] == "NOT_APPLICABLE_SYNTHETIC"
        assert header["TYPECODE"] == "42"
        assert header["INT_LED"] is False
        assert header["EXT_LED"] is True
        assert header["DECONRAT"] == pytest.approx(1.0)
        assert header["DECONTOL"] == 0.05
        assert header["DATAZER"] == 1
        nonzero = expected[expected != 0]
        assert header["DATASIG"] == pytest.approx(float(np.std(nonzero)))
        assert header["DATAP01"] == pytest.approx(float(np.percentile(nonzero, 1)))
        assert calibration_manifest.is_file()
        calibration = json.loads(calibration_manifest.read_text())
        assert calibration["product_stage"] == "level2_psf_deconvolution"
        assert calibration["correction_factor"] == 0.4
        assert set(calibration["components"]) == {
            "diffraction_psf",
            "scatter_psf",
            "spectral_response",
            "spectrum",
        }
        assert hdul[0].verify_checksum() == 1
        assert hdul[0].verify_datasum() == 1
        metadata_managers.validate_fits_header(
            header,
            processor.metadata_definition_file,
            2,
            float_output_statistics=("DATAMIN", "DATAMAX"),
        )
        wrong_level = header.copy()
        wrong_level["LEVEL"] = 1
        with pytest.raises(
            metadata_managers.FitsMetadataContractError,
            match="LEVEL must equal validated level 2",
        ):
            metadata_managers.validate_fits_header(
                wrong_level,
                processor.metadata_definition_file,
                2,
                float_output_statistics=("DATAMIN", "DATAMAX"),
            )
        nonfinite_metadata = dict(header)
        nonfinite_metadata["DSATVAL"] = float("inf")
        with pytest.raises(
            metadata_managers.FitsMetadataContractError,
            match="DSATVAL must be finite",
        ):
            metadata_managers.validate_fits_header(
                nonfinite_metadata,
                processor.metadata_definition_file,
                2,
                float_output_statistics=("DATAMIN", "DATAMAX"),
            )


def test_level1_data_is_used_without_crude_recalibration(tmp_path, monkeypatch):
    input_path = tmp_path / "input.fits"
    input_data = _write_input(input_path, level=1, synthetic=True)
    processor = _level2(tmp_path)
    observed = {}

    def fake_deconvolution(data, *_args, **_kwargs):
        observed["data"] = data.copy()
        return data

    monkeypatch.setattr(
        make_level2.suncet_deconv,
        "apply_deconv",
        fake_deconvolution,
    )
    output_path = processor.run(input_path, tmp_path / "output")[0]

    np.testing.assert_array_equal(observed["data"], input_data)
    with fits.open(output_path) as hdul:
        assert hdul[0].header["SYNTHET"] is True
        assert hdul[0].header["L1BYPASS"] is False


def test_level2_rejects_non_utc_input_time_system(tmp_path, monkeypatch):
    input_path = tmp_path / "input.fits"
    _write_input(input_path, level=1)
    with fits.open(input_path, mode="update") as hdul:
        hdul[0].header["TIMESYS"] = "TAI"
        hdul[0].add_checksum()
    processor = _level2(tmp_path)
    monkeypatch.setattr(
        make_level2.suncet_deconv,
        "apply_deconv",
        lambda data, *_args, **_kwargs: data,
    )

    with pytest.raises(ValueError, match="use UTC"):
        processor.run(input_path, tmp_path / "output")


def test_level2_rejects_incomplete_cumulative_metadata(tmp_path, monkeypatch):
    input_path = tmp_path / "input.fits"
    _write_input(input_path, level=1)
    with fits.open(input_path) as hdul:
        data = hdul[0].data.copy()
        header = hdul[0].header.copy()
    del header["DATASAT"]
    for key in ("CHECKSUM", "DATASUM"):
        header.remove(key, ignore_missing=True, remove_all=True)
    fits.PrimaryHDU(data=data, header=header).writeto(
        input_path,
        overwrite=True,
        checksum=True,
    )
    processor = _level2(tmp_path)
    monkeypatch.setattr(
        make_level2.suncet_deconv,
        "apply_deconv",
        lambda data, *_args, **_kwargs: data,
    )

    with pytest.raises(
        metadata_managers.FitsMetadataContractError,
        match="DATASAT",
    ):
        processor.run(input_path, tmp_path / "output")

    assert not list((tmp_path / "output").glob("*.fits"))
    assert not list((tmp_path / "output").glob("*.json"))
    assert not list((tmp_path / "output").glob(".*.tmp"))


def test_level2_rejects_corrupt_input_checksum(tmp_path, monkeypatch):
    input_path = tmp_path / "input.fits"
    _write_input(input_path, level=1)
    with fits.open(input_path) as hdul:
        data_offset = hdul[0].fileinfo()["datLoc"]
    with input_path.open("r+b") as stream:
        stream.seek(data_offset)
        byte = stream.read(1)
        stream.seek(data_offset)
        stream.write(bytes([byte[0] ^ 0x01]))

    processor = _level2(tmp_path)
    monkeypatch.setattr(
        make_level2.suncet_deconv,
        "apply_deconv",
        lambda *_args, **_kwargs: pytest.fail("deconvolution must not run"),
    )
    with pytest.raises(ValueError, match="checksums did not validate"):
        processor.run(input_path, tmp_path / "output")
    assert not list((tmp_path / "output").iterdir())


def test_level2_rejects_unexpected_detector_shape(tmp_path, monkeypatch):
    input_path = tmp_path / "input.fits"
    _write_input(input_path, level=1, shape=(2, 3))
    processor = _level2(tmp_path)
    monkeypatch.setattr(
        make_level2.suncet_deconv,
        "apply_deconv",
        lambda *_args, **_kwargs: pytest.fail("deconvolution must not run"),
    )
    with pytest.raises(ValueError, match="calibrated for detector shape"):
        processor.run(input_path, tmp_path / "output")


def test_level2_rejects_deconvolution_shape_change(tmp_path, monkeypatch):
    input_path = tmp_path / "input.fits"
    _write_input(input_path, level=1)
    processor = _level2(tmp_path)
    monkeypatch.setattr(
        make_level2.suncet_deconv,
        "apply_deconv",
        lambda data, *_args, **_kwargs: data[:-1],
    )
    with pytest.raises(ValueError, match="changed the image shape"):
        processor.run(input_path, tmp_path / "output")


def test_level2_rejects_nonconserving_deconvolution(tmp_path, monkeypatch):
    input_path = tmp_path / "input.fits"
    _write_input(input_path, level=1)
    processor = _level2(tmp_path)
    monkeypatch.setattr(
        make_level2.suncet_deconv,
        "apply_deconv",
        lambda data, *_args, **_kwargs: data * 1.1,
    )
    with pytest.raises(ValueError, match="flux-conservation gate failed"):
        processor.run(input_path, tmp_path / "output")


def test_level2_refuses_implicit_overwrite_and_preserves_product(
    tmp_path,
    monkeypatch,
):
    input_path = tmp_path / "input.fits"
    _write_input(input_path, level=1)
    processor = _level2(tmp_path)
    monkeypatch.setattr(
        make_level2.suncet_deconv,
        "apply_deconv",
        lambda data, *_args, **_kwargs: data,
    )
    output_path = processor.run(input_path, tmp_path / "output")[0]
    original = output_path.read_bytes()

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        processor.run(input_path, tmp_path / "output")
    assert output_path.read_bytes() == original
    assert not list((tmp_path / "output").glob(".*.tmp"))


@pytest.mark.parametrize(
    ("keyword", "value"),
    (("correction_factor", float("nan")), ("flux_ratio_tolerance", float("inf"))),
)
def test_level2_rejects_nonfinite_parameters(tmp_path, keyword, value):
    kwargs = {keyword: value}
    with pytest.raises(ValueError, match="must be finite"):
        _level2(tmp_path, **kwargs)


def test_metadata_contract_rejects_invalid_named_minimum_level(tmp_path):
    definition_path = _write_metadata_definition(tmp_path / "metadata.csv")
    with definition_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["Minimum Level"] = "not-a-level"
    with definition_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=DEFINITION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="invalid Minimum Level.*LEVEL"):
        metadata_managers.validate_fits_header(
            fits.Header({"LEVEL": 2}),
            definition_path,
            2,
        )

import os
import numpy as np
import pytest
import duckdb
from astropy.io import fits
from scipy.spatial.transform import Rotation
from types import SimpleNamespace
from .. import config_parser, make_level1
from ..level1_orientation import solar_north_j2000_unit_vector


def test_effective_exposure_uses_stack_counts_filtering_and_right_shift():
    exposures = make_level1.effective_exposure_times_from_metadata(
        {
            "INTTIMEI": 0.035,
            "NSTACKI": 9,
            "STKNORMI": 8,
            "PIXFILTI": True,
            "INTTIMEO": 15.0,
            "NSTACKO": 4,
            "STKNORMO": 4,
            "PIXFILTO": True,
            "TELAPSE": 69.0,
        }
    )
    assert exposures == {"inner": 0.035, "outer": 11.25}


def test_effective_exposure_understands_ctdb_enabled_disabled_states():
    exposures = make_level1.effective_exposure_times_from_metadata(
        {
            "INTTIMEI": 2.0,
            "NSTACKI": 4,
            "STKNORMI": 2,
            "PIXFILTI": "DIS",
            "INTTIMEO": 2.0,
            "NSTACKO": 4,
            "STKNORMO": 2,
            "PIXFILTO": "ENA",
        }
    )

    assert exposures == {"inner": 4.0, "outer": 3.0}


def test_2d_composite_exposure_requires_and_uses_inner_mask():
    image = np.zeros((2, 3))
    exposures = {"inner": 1.0, "outer": 4.0}
    with pytest.raises(ValueError, match="requires an inner_mask"):
        make_level1.create_exposure_time_mask(image, exposures)
    mask = make_level1.create_exposure_time_mask(
        image,
        exposures,
        inner_mask=np.array([[False, True, False], [True, True, False]]),
    )
    np.testing.assert_array_equal(mask, [[4, 1, 4], [1, 1, 4]])


def test_circular_inner_mask_includes_boundary_and_uses_array_coordinates():
    mask = make_level1.create_circular_inner_mask(
        (5, 7),
        center_column=3,
        center_row=2,
        radius_pixels=1,
    )

    expected = np.zeros((5, 7), dtype=bool)
    expected[2, 3] = True
    expected[1, 3] = True
    expected[3, 3] = True
    expected[2, 2] = True
    expected[2, 4] = True
    np.testing.assert_array_equal(mask, expected)


def test_csie_composite_mask_uses_processed_image_circle_coordinates_directly():
    mask = make_level1.create_csie_composite_inner_mask(
        (3, 4),
        {
            "ICMRAD2": 0,
            "ICMCOL": 2,
            "ICMROW": 1,
            # These do not transform the already-binned ICM coordinates.
            "BINCOL": 2,
            "BINROW": 2,
            "ROICOL0": 10,
            "ROIROW0": 20,
        },
    )

    expected = np.zeros((3, 4), dtype=bool)
    expected[1, 2] = True
    np.testing.assert_array_equal(mask, expected)


def test_normalize_to_dn_per_second_uses_pixelwise_effective_exposure():
    image = np.array([[8, 8, 8], [8, 8, 8]], dtype=np.uint16)
    inner_mask = np.array([[False, True, False], [True, True, False]])

    normalized = make_level1.normalize_to_dn_per_second(
        image,
        {"inner": 2.0, "outer": 4.0},
        inner_mask=inner_mask,
    )

    np.testing.assert_array_equal(normalized, [[2, 4, 2], [4, 4, 2]])
    assert normalized.dtype == np.float64


def test_normalize_to_dn_per_second_rejects_nonpositive_exposure():
    with pytest.raises(ValueError, match="strictly positive"):
        make_level1.normalize_to_dn_per_second(
            np.ones((2, 2)),
            {"inner": 0.0},
        )


def test_level1_composite_normalization_requires_explicit_composite_metadata():
    image = np.full((2, 3), 8, dtype=np.uint16)
    metadata = {
        "EFFEXPI": 2.0,
        "EFFEXPO": 4.0,
        "ICMPROC1": "FPM0_INSIDE_FPM1_OUTSIDE",
    }
    inner_mask = np.array([[False, True, False], [True, True, False]])

    normalized = make_level1.normalize_level1_image_to_dn_per_second(
        image,
        metadata,
        inner_mask=inner_mask,
    )

    np.testing.assert_array_equal(normalized, [[2, 4, 2], [4, 4, 2]])


def test_level1_composite_normalization_builds_circle_mask_from_metadata():
    image = np.full((3, 3), 8, dtype=np.uint16)
    normalized = make_level1.normalize_level1_image_to_dn_per_second(
        image,
        {
            "EFFEXPI": 2.0,
            "EFFEXPO": 4.0,
            "ICMPOS": "FPM0_INSIDE_FPM1_OUTSIDE",
            "ICMRAD2": 0,
            "ICMCOL": 1,
            "ICMROW": 1,
            "BINCOL": 1,
            "BINROW": 1,
        },
    )

    expected = np.full((3, 3), 2.0)
    expected[1, 1] = 4.0
    np.testing.assert_array_equal(normalized, expected)


def test_level1_noncomposite_does_not_apply_unequal_inner_outer_exposures():
    with pytest.raises(ValueError, match="non-composite metadata"):
        make_level1.normalize_level1_image_to_dn_per_second(
            np.full((2, 3), 8, dtype=np.uint16),
            {"EFFEXPI": 2.0, "EFFEXPO": 4.0, "ICMPROC1": "OFF"},
            inner_mask=np.ones((2, 3), dtype=bool),
        )


def test_level1_noncomposite_uses_one_unambiguous_exposure():
    normalized = make_level1.normalize_level1_image_to_dn_per_second(
        np.full((2, 3), 8, dtype=np.uint16),
        {"EFFEXPI": 4.0, "ICMPROC1": "OFF"},
    )

    np.testing.assert_array_equal(normalized, np.full((2, 3), 2.0))


def test_coarse_rotation_fails_closed_without_exposure_timestamps():
    level1 = make_level1.Level1.__new__(make_level1.Level1)
    level1.metadata = {}
    image = np.arange(6).reshape(2, 3)

    observed = level1._Level1__coarse_rotate(image, {})

    np.testing.assert_array_equal(observed, image)
    assert level1.coarse_orientation_decision.status == "missing_exposure_time"


@pytest.mark.parametrize(
    "metadata",
    [
        {"ICMPROC1": "FPM0_INSIDE_FPM1_OUTSIDE"},
        {"icm_proc_cfg_posi_select_meta": 3},
        {"csie_meta_icm_proc_config": 3 << 6},
    ],
)
def test_composite_detection_accepts_v1_1_9_metadata_forms(metadata):
    assert make_level1.csie_metadata_indicates_composite(metadata) is True


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"ICMPROC1": "OFF"},
        {"icm_proc_cfg_posi_select_meta": 1},
        {"icm_proc_cfg_posi_select_meta": "FPM0_INSIDE"},
    ],
)
def test_composite_detection_fails_closed_for_ambiguous_metadata(metadata):
    assert make_level1.csie_metadata_indicates_composite(metadata) is False


def test_Level1_object_instantiates(tmp_path, monkeypatch):
    data_root = tmp_path / 'data'
    ctdb_root = tmp_path / 'ctdb'
    data_root.mkdir()
    ctdb_root.mkdir()
    monkeypatch.setenv('suncet_data', str(data_root))
    monkeypatch.setenv('suncet_ctdb', str(ctdb_root))
    default_config = os.path.join(
        os.path.dirname(__file__), '..', 'config_files',
        'config_default.ini'
    )
    config = config_parser.Config(default_config)
    make_level1.Level1(config)


def test_level1_database_resolution_ignores_folder_local_time_corrected_db(tmp_path):
    test_data = tmp_path / "test_data"
    local = test_data / "run" / "run_time_corrected.duckdb"
    local.parent.mkdir(parents=True)
    local.touch()
    config = SimpleNamespace(data_root=str(tmp_path), version_pipeline="2.0.2")

    assert make_level1.resolve_level1_telemetry_database(config) is None

    canonical = test_data / "suncet_telemetry_test_phase_v2.0.2.duckdb"
    canonical.touch()
    assert make_level1.resolve_level1_telemetry_database(config) == canonical


def test_Level1_uses_requested_portable_input_directory(tmp_path, monkeypatch):
    data_root = tmp_path / 'data'
    ctdb_root = tmp_path / 'ctdb'
    input_root = data_root / 'level0_5'
    input_root.mkdir(parents=True)
    ctdb_root.mkdir()
    expected = input_root / 'image.fits'
    fits.PrimaryHDU(
        np.full((2, 3), 8, dtype=np.uint16),
        header=fits.Header({"EFFEXPI": 4.0, "ICMPOS": "OFF"}),
    ).writeto(expected)
    monkeypatch.setenv('suncet_data', str(data_root))
    monkeypatch.setenv('suncet_ctdb', str(ctdb_root))
    default_config = os.path.join(
        os.path.dirname(__file__), '..', 'config_files', 'config_default.ini'
    )
    level1 = make_level1.Level1(config_parser.Config(default_config))
    outputs = level1.make(input_root)

    assert outputs == [input_root / "level1" / "image.fits"]
    with fits.open(outputs[0]) as hdul:
        np.testing.assert_array_equal(hdul[0].data, np.full((2, 3), 2.0))
        assert hdul[0].header["BUNIT"] == "DN / s"
        assert hdul[0].header["ROTSTAT"] == "attitude_unavailable"


def test_level1_pipeline_gets_current_circle_from_canonical_apid537(tmp_path):
    data_root = tmp_path / "data"
    input_root = data_root / "test_data" / "run" / "csie_images"
    input_root.mkdir(parents=True)
    source = input_root / "image_10.fits"
    fits.PrimaryHDU(
        np.full((3, 3), 8, dtype=np.uint16),
        header=fits.Header(
            {
                "EFFEXPI": 2.0,
                "EFFEXPO": 4.0,
                "ICMPOS": "FPM0_INSIDE_FPM1_OUTSIDE",
                "BINCOL": 1,
                "BINROW": 1,
                "ROICOL0": 0,
                "ROIROW0": 0,
                "J2K-BEG": 100.0,
                "J2K-END": 101.0,
            }
        ),
    ).writeto(source)
    database = data_root / "test_data" / "suncet_telemetry_test_phase_v2.0.2.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            """
            CREATE TABLE apid_0537 AS SELECT * FROM (VALUES
                (90, 0, 0, 1, 1)
            ) AS rows(
                SHCOARSE, SHFINE, csie_115_hk_icm_rad_sqr,
                csie_115_hk_icm_cent_col, csie_115_hk_icm_cent_row
            )
            """
        )
    processor = make_level1.Level1(
        SimpleNamespace(data_root=str(data_root), version_pipeline="2.0.2")
    )

    outputs = processor.make(input_root)

    assert outputs == [input_root.parent / "level1" / source.name]
    with fits.open(outputs[0]) as hdul:
        expected = np.full((3, 3), 2.0)
        expected[1, 1] = 4.0
        np.testing.assert_array_equal(hdul[0].data, expected)
        assert hdul[0].header["ICMRAD2"] == 0
        assert hdul[0].header["ICMCOL"] == 1
        assert hdul[0].header["ICMROW"] == 1
        assert hdul[0].header["ICMSRC"] == "APID 537"


def test_level1_pipeline_applies_lossless_apid46_quarter_turn(tmp_path):
    source = tmp_path / "image.fits"
    image = np.arange(6, dtype=np.uint16).reshape(2, 3) * 4
    fits.PrimaryHDU(
        image,
        header=fits.Header(
            {
                "EFFEXPI": 4.0,
                "ICMPOS": "OFF",
                "J2K-BEG": 100.0,
                "J2K-END": 101.0,
            }
        ),
    ).writeto(source)
    database = tmp_path / "telemetry.duckdb"
    pole = solar_north_j2000_unit_vector()
    quaternion = Rotation.align_vectors(
        np.array([[0.0, 1.0, 0.0]]), pole[None, :]
    )[0].as_quat()
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            """
            CREATE TABLE apid_0046(
                ccsdsSecHeader2_sec_adcs_att_det BIGINT,
                ccsdsSecHeader2_sub_adcs_att_det BIGINT,
                adcs_att_det_q_body_wrt_eci1 DOUBLE,
                adcs_att_det_q_body_wrt_eci2 DOUBLE,
                adcs_att_det_q_body_wrt_eci3 DOUBLE,
                adcs_att_det_q_body_wrt_eci4 DOUBLE,
                adcs_att_det_attitude_vld VARCHAR
            )
            """
        )
        for timestamp in (100, 101):
            connection.execute(
                "INSERT INTO apid_0046 VALUES (?, 0, ?, ?, ?, ?, 'YES')",
                [timestamp, *quaternion.tolist()],
            )
    processor = make_level1.Level1(
        SimpleNamespace(data_root=str(tmp_path), version_pipeline="2.0.2")
    )

    [output] = processor.make(
        source,
        output_dir=tmp_path / "level1",
        telemetry_database=database,
    )

    with fits.open(output) as hdul:
        np.testing.assert_array_equal(hdul[0].data, np.rot90(image / 4.0, 1))
        assert hdul[0].header["ROTSTAT"] == "stable"
        assert hdul[0].header["ROTQTR"] == 1
        assert hdul[0].header["ROLLFLAG"] is False

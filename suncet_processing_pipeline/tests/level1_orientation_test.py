import numpy as np
import duckdb

from ..level1_orientation import (
    apply_coarse_solar_north_rotation,
    load_apid46_attitude_samples,
    plan_coarse_solar_north_rotation,
    solar_north_j2000_unit_vector,
)


def _sample(time, quaternion=(0.0, 0.0, 0.0, 1.0), *, valid=True, z_rate=None):
    sample = {
        "ccsdsSecHeader2_sec_adcs_att_det": int(time),
        "ccsdsSecHeader2_sub_adcs_att_det": int(round((time % 1) * 1000)),
        "adcs_att_det_attitude_vld": "YES" if valid else "NO",
        **{
            f"adcs_att_det_q_body_wrt_eci{index}": value
            for index, value in enumerate(quaternion, start=1)
        },
    }
    if z_rate is not None:
        sample["adcs_att_det_meas_rate_vld"] = "YES"
        sample["adcs_att_det_body_rate3"] = z_rate
    return sample


def test_solar_north_j2000_vector_is_unit_length():
    np.testing.assert_allclose(np.linalg.norm(solar_north_j2000_unit_vector()), 1.0)


def test_body_plus_y_rotates_native_array_right_up_without_interpolation():
    decision = plan_coarse_solar_north_rotation(
        [_sample(100), _sample(110)],
        exposure_start_j2000=100,
        exposure_end_j2000=110,
        solar_north_eci=np.array([0.0, 1.0, 0.0]),
    )
    image = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)

    rotated = apply_coarse_solar_north_rotation(image, decision)

    assert decision.status == "stable"
    assert decision.quarter_turns_ccw == 1
    assert decision.rotation_degrees_ccw == 90
    assert rotated.dtype == image.dtype
    np.testing.assert_array_equal(rotated, np.rot90(image, 1))


def test_body_minus_y_selects_three_counterclockwise_quarter_turns():
    decision = plan_coarse_solar_north_rotation(
        [_sample(100), _sample(110)],
        exposure_start_j2000=100,
        exposure_end_j2000=110,
        solar_north_eci=np.array([0.0, -1.0, 0.0]),
    )

    assert decision.status == "stable"
    assert decision.quarter_turns_ccw == 3


def test_body_x_alignment_is_flagged_as_roll_and_image_is_not_rotated():
    decision = plan_coarse_solar_north_rotation(
        [_sample(100), _sample(110)],
        exposure_start_j2000=100,
        exposure_end_j2000=110,
        solar_north_eci=np.array([1.0, 0.0, 0.0]),
    )
    image = np.arange(6).reshape(2, 3)

    assert decision.status == "roll_in_progress"
    assert decision.quarter_turns_ccw is None
    np.testing.assert_array_equal(
        apply_coarse_solar_north_rotation(image, decision), image
    )


def test_attitude_change_across_exposure_is_flagged_as_roll():
    decision = plan_coarse_solar_north_rotation(
        [
            _sample(100),
            _sample(110, quaternion=(1.0, 0.0, 0.0, 0.0)),
        ],
        exposure_start_j2000=100,
        exposure_end_j2000=110,
        solar_north_eci=np.array([0.0, 1.0, 0.0]),
    )

    assert decision.status == "roll_in_progress"
    assert "crosses" in decision.reason


def test_high_valid_body_z_rate_is_flagged_as_roll():
    decision = plan_coarse_solar_north_rotation(
        [_sample(100, z_rate=0.01), _sample(110, z_rate=0.01)],
        exposure_start_j2000=100,
        exposure_end_j2000=110,
        solar_north_eci=np.array([0.0, 1.0, 0.0]),
    )

    assert decision.status == "roll_in_progress"
    assert "roll rate" in decision.reason


def test_invalid_attitude_during_exposure_fails_closed():
    decision = plan_coarse_solar_north_rotation(
        [_sample(100), _sample(110, valid=False)],
        exposure_start_j2000=100,
        exposure_end_j2000=110,
        solar_north_eci=np.array([0.0, 1.0, 0.0]),
    )

    assert decision.status == "invalid_attitude_coverage"
    assert decision.quarter_turns_ccw is None


def test_exposure_requires_bracketing_apid46_samples():
    decision = plan_coarse_solar_north_rotation(
        [_sample(105)],
        exposure_start_j2000=100,
        exposure_end_j2000=110,
        solar_north_eci=np.array([0.0, 1.0, 0.0]),
    )

    assert decision.status == "insufficient_attitude_coverage"


def test_decision_exposes_fits_provenance_fields():
    decision = plan_coarse_solar_north_rotation(
        [_sample(100), _sample(110)],
        exposure_start_j2000=100,
        exposure_end_j2000=110,
        solar_north_eci=np.array([0.0, 1.0, 0.0]),
    )

    assert decision.fits_keywords() == {
        "ATTAPID": 46,
        "ROTSTAT": "stable",
        "ROLLFLAG": False,
        "NATYDIR": "RIGHT",
        "ATTSAMPS": 2,
        "ROTQTR": 1,
        "ROTDEG": 90,
        "SOLNRES": 0.0,
    }


def test_duckdb_loader_returns_only_bracketing_and_in_exposure_samples(tmp_path):
    database = tmp_path / "telemetry.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            """
            CREATE TABLE apid_0046 AS
            SELECT * FROM (VALUES
                (95, 0, 0.0, 0.0, 0.0, 1.0, 'YES'),
                (100, 0, 0.0, 0.0, 0.0, 1.0, 'YES'),
                (105, 0, 0.0, 0.0, 0.0, 1.0, 'YES'),
                (110, 0, 0.0, 0.0, 0.0, 1.0, 'YES'),
                (115, 0, 0.0, 0.0, 0.0, 1.0, 'YES')
            ) AS rows(
                ccsdsSecHeader2_sec_adcs_att_det,
                ccsdsSecHeader2_sub_adcs_att_det,
                adcs_att_det_q_body_wrt_eci1,
                adcs_att_det_q_body_wrt_eci2,
                adcs_att_det_q_body_wrt_eci3,
                adcs_att_det_q_body_wrt_eci4,
                adcs_att_det_attitude_vld
            )
            """
        )

    rows = load_apid46_attitude_samples(database, 100, 110)

    assert [row["ccsdsSecHeader2_sec_adcs_att_det"] for row in rows] == [100, 105, 110]

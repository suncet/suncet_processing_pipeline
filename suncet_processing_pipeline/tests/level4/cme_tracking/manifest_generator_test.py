"""Tests for the timestamp-correct simulator manifest generator."""

from pathlib import Path
import runpy

from astropy.io import fits
import pytest

_REPOSITORY = Path(__file__).resolve().parents[4]
_GENERATOR = runpy.run_path(
    str(
        _REPOSITORY
        / "benchmarks/cme_tracking/"
        "create_config_default_no_particle_filter_manifest.py"
    )
)
_DATE_CARDS_WITH_UTC_COMMENTS = _GENERATOR["_DATE_CARDS_WITH_UTC_COMMENTS"]
_EXPECTED_HEADER_VALUES = _GENERATOR["_EXPECTED_HEADER_VALUES"]
_validate_reviewed_header = _GENERATOR["_validate_reviewed_header"]


def _reviewed_header() -> fits.Header:
    header = fits.Header()
    for key, value in _EXPECTED_HEADER_VALUES.items():
        header[key] = value
    for key in _DATE_CARDS_WITH_UTC_COMMENTS:
        header[key] = (
            "2023-01-14T17:00:00.000",
            "Reviewed simulator date expressed in UTC",
        )
    header["DATE-END"] = (
        "2023-01-14T17:00:15.000",
        "End of the observation on orbit in UTC",
    )
    return header


def test_reviewed_header_accepts_locked_simulator_facts():
    _validate_reviewed_header(
        _reviewed_header(),
        Path("reviewed.fits"),
        date_obs="2023-01-14T17:00:00.000",
        date_end="2023-01-14T17:00:15.000",
    )


@pytest.mark.parametrize("key", tuple(_EXPECTED_HEADER_VALUES))
def test_reviewed_header_rejects_each_locked_header_value_drifting(key: str):
    header = _reviewed_header()
    expected = _EXPECTED_HEADER_VALUES[key]
    header[key] = "changed" if isinstance(expected, str) else expected + 1

    with pytest.raises(RuntimeError, match=key):
        _validate_reviewed_header(
            header,
            Path("changed.fits"),
            date_obs="2023-01-14T17:00:00.000",
            date_end="2023-01-14T17:00:15.000",
        )


def test_reviewed_header_rejects_non_fifteen_second_date_end():
    with pytest.raises(RuntimeError, match=r"DATE-END = DATE-OBS \+ 15 s"):
        _validate_reviewed_header(
            _reviewed_header(),
            Path("changed.fits"),
            date_obs="2023-01-14T17:00:00.000",
            date_end="2023-01-14T17:00:14.000",
        )


@pytest.mark.parametrize("key", _DATE_CARDS_WITH_UTC_COMMENTS)
def test_reviewed_header_rejects_documented_date_comment_drifting(key: str):
    header = _reviewed_header()
    header.comments[key] = "TAI"

    with pytest.raises(RuntimeError, match=f"UTC comment on {key}"):
        _validate_reviewed_header(
            header,
            Path("changed.fits"),
            date_obs="2023-01-14T17:00:00.000",
            date_end="2023-01-14T17:00:15.000",
        )

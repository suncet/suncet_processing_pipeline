"""Tests for the private SatNOGS field-review worksheet generator."""

import csv

from suncet_processing_pipeline.satnogs.field_review import (
    REVIEW_COLUMNS,
    build_review_rows,
    export_review_csv,
)


def _source_rows():
    return [
        {
            "APID": "2",
            "ItemName": "not_beacon",
            "DataType": "U8",
            "ShortDescription": "Other packet",
        },
        {
            "APID": "1",
            "ItemName": "VERSION",
            "DataType": "U3",
            "ShortDescription": "CCSDS version",
        },
        {
            "APID": "1",
            "ItemName": "beac_cmd_count",
            "DataType": "U16",
            "ShortDescription": "Command counter",
        },
        {
            "APID": "1",
            "ItemName": "beac_ana_bat1_v",
            "DataType": "U16",
            "ShortDescription": "Battery voltage",
            "Units": "V",
            "Conversion": "C0=0 C1=1",
        },
    ]


def test_build_review_rows_tracks_packed_bit_offsets_and_triage():
    rows = build_review_rows(_source_rows())

    assert [row["bit_offset"] for row in rows] == [0, 3, 19]
    assert [row["bit_in_byte"] for row in rows] == [0, 3, 3]
    assert [row["proposed_action"] for row in rows] == ["REVIEW", "OMIT", "REVIEW"]
    assert all(row["public_name"] == "" for row in rows)


def test_export_review_csv_is_explicitly_unapproved(tmp_path):
    source = tmp_path / "ct_tlm.csv"
    output = tmp_path / "apid1_private_review.csv"
    source_columns = (
        "APID",
        "ItemName",
        "DataType",
        "ShortDescription",
        "LongDescription",
        "Units",
        "Conversion",
    )
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=source_columns)
        writer.writeheader()
        writer.writerows(_source_rows())

    field_count, total_bits = export_review_csv(source, output)

    with output.open(newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert field_count == 3
    assert total_bits == 35
    assert tuple(exported[0]) == REVIEW_COLUMNS
    assert {row["proposed_action"] for row in exported} == {"REVIEW", "OMIT"}
    assert all(not row["public_name"] for row in exported)

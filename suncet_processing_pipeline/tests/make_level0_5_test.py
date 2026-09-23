"""Focused tests for canonical Level 0.5 ingest and CSIE row policy."""

from collections import Counter
import json

import imagecodecs
import numpy as np
from PIL import Image as PilImage

from .. import make_level0_5 as level0_5
from ..make_level0_5 import (
    INPUT_MODE_CCSDS,
    PacketRecord,
    PLAYBACK_METADATA_LEN,
    SYNC_MARKER,
    UHF_MAX_SEGMENT_PAYLOAD_LEN,
    UHF_PLAYBACK_APID,
    UHF_SEGMENTED_APID,
    UHF_SEGMENT_FLAG_END,
    UHF_SEGMENT_FLAG_MIDDLE,
    UHF_SEGMENT_FLAG_START,
    _csie_preview_rgb_uint8,
    _capture_timestamp_seconds,
    _decode_csie_jpegls_uint16,
    _decode_recovered_csie_jpegls,
    _decoder_object_fields,
    _has_fixed_transfer_frame_checksum_footer_layout,
    _level0_5_output_fields,
    _metadata_expectations_from_records,
    _record_sort_time,
    _scan_csie_data_records_in_stream,
    _trim_jpegls_at_eoi,
    _write_csie_fits,
    _write_csie_meta_json,
    _write_csie_png,
    assemble_csie_uncompressed_image,
    build_fixed_binary,
    ccsds_packet_at,
    csie_data_additive_checksum,
    csie_meta_processing_mode,
    decode_csie_meta_packet,
    fletcher32,
    fletcher32_words_be,
    generic_ctdb_decode_packet,
    infer_csie_uncompressed_dimensions_from_rows,
    packetize_checksum_valid_ccsds,
    parse_csie_data_row_packet,
    reverse_32bit_words,
    strip_xband_frame_records,
    summarize_csie_metadata_packets,
    unwrap_direct_playback_stream,
    unwrap_uhf_playback_stream,
    validate_transfer_frame_checksum_footer,
)


def _csie_row_packet(
    image_id: int,
    row_index: int,
    *,
    cols: int = 1504,
    checksum_valid: bool = True,
) -> bytes:
    pixels = (np.arange(cols, dtype=np.uint16) + row_index).astype(">u2")
    row_payload = pixels.tobytes()
    secondary_header = image_id.to_bytes(4, "big") + b"\x00\x00"
    checksum = csie_data_additive_checksum(row_payload)
    if not checksum_valid:
        checksum ^= 1
    data_field = secondary_header + row_payload + checksum.to_bytes(4, "big")
    first_word = 0x0800 | 536
    sequence_word = 0xC000 | row_index
    return (
        first_word.to_bytes(2, "big")
        + sequence_word.to_bytes(2, "big")
        + (len(data_field) - 1).to_bytes(2, "big")
        + data_field
    )


def test_csie_stream_scan_discovers_checksum_gated_metadata_less_image():
    stream = b"\x55" * 7 + b"".join(
        (
            _csie_row_packet(4357, 5, checksum_valid=False),
            _csie_row_packet(4357, 8),
            _csie_row_packet(4357, 2000),
            _csie_row_packet(4357, 2001, cols=1000),
            _csie_row_packet(9999, 2000),
        )
    )

    records, warnings = _scan_csie_data_records_in_stream(
        stream,
        source="test",
        known_image_ids=set(),
        expected_cols_by_image={},
    )

    summaries = [parse_csie_data_row_packet(record)[0] for record in records]
    assert [(row["image_id"], row["row_index"]) for row in summaries] == [
        (4357, 5),
        (4357, 8),
        (4357, 2000),
        (4357, 2001),
    ]
    assert [row["checksum_status"] for row in summaries] == [
        "failed",
        "valid_be",
        "valid_be",
        "valid_be",
    ]
    assert any("metadata-less" in warning and "4357" in warning for warning in warnings)


def test_csie_stream_scan_repairs_unique_boundary_fill_candidate():
    original_row = _csie_row_packet(4357, 1)
    prefix = b"\x33" * 336
    removed_bytes = 120
    fill_at = 2040 - len(prefix) - removed_bytes
    corrupted_row = (
        original_row[:fill_at] + b"\x55" * removed_bytes + original_row[fill_at:]
    )
    stream = prefix + corrupted_row + _csie_row_packet(4357, 2) + _csie_row_packet(4357, 3)

    known_records, known_warnings = _scan_csie_data_records_in_stream(
        stream,
        source="test",
        known_image_ids={4357},
        expected_cols_by_image={4357: 1504},
    )
    assert known_records[0].packet == original_row
    assert known_records[0].acceptance_mode == (
        "csie_full_stream_boundary_0x55_repair"
    )
    assert any("removed_bytes=120" in warning for warning in known_warnings)

    records, warnings = _scan_csie_data_records_in_stream(
        stream,
        source="test",
        known_image_ids=set(),
        expected_cols_by_image={},
    )

    assert [record.source_offset for record in records] == [336, 3480, 6504]
    assert records[0].packet == original_row
    assert records[0].acceptance_mode == "csie_full_stream_boundary_0x55_repair"
    assert records[0].checksum_validated is True
    summary, _pixels = parse_csie_data_row_packet(records[0])
    assert summary["checksum_status"] == "valid_be"
    assert summary["acceptance_mode"] == "csie_full_stream_boundary_0x55_repair"
    assert any(
        "image_id=4357" in warning
        and "row_index=1" in warning
        and "removed_bytes=120" in warning
        for warning in warnings
    )


def test_csie_boundary_fill_repair_does_not_weaken_metadataless_discovery_gate():
    original_row = _csie_row_packet(4357, 1)
    prefix = b"\x33" * 336
    fill_at = 2040 - len(prefix) - 120
    corrupted_row = original_row[:fill_at] + b"\x55" * 120 + original_row[fill_at:]
    stream = prefix + corrupted_row + _csie_row_packet(4357, 2)

    records, warnings = _scan_csie_data_records_in_stream(
        stream,
        source="test",
        known_image_ids=set(),
        expected_cols_by_image={},
    )

    assert records == []
    assert not any("removed_bytes=" in warning for warning in warnings)


def test_csie_boundary_fill_repair_requires_failed_packet_and_boundary_touch():
    row_one = _csie_row_packet(4357, 1)
    row_two = _csie_row_packet(4357, 2)
    prefix = b"\x33" * 336

    valid_with_interpacket_fill = prefix + row_one + b"\x55" * 120 + row_two
    valid_records, valid_warnings = _scan_csie_data_records_in_stream(
        valid_with_interpacket_fill,
        source="test",
        known_image_ids={4357},
        expected_cols_by_image={4357: 1504},
    )
    assert valid_records[0].packet == row_one
    assert valid_records[0].acceptance_mode == "csie_full_stream_structural"
    assert not any("removed_bytes=" in warning for warning in valid_warnings)

    boundary_fill_at = 2040 - len(prefix) - 120
    boundary_corrupted_row = (
        row_one[:boundary_fill_at] + b"\x55" * 120 + row_one[boundary_fill_at:]
    )
    id_only_records, id_only_warnings = _scan_csie_data_records_in_stream(
        prefix + boundary_corrupted_row + row_two,
        source="test",
        known_image_ids={4357},
        expected_cols_by_image={},
    )
    id_only_summary, _pixels = parse_csie_data_row_packet(id_only_records[0])
    assert id_only_summary["checksum_status"] == "failed"
    assert id_only_records[0].acceptance_mode == "csie_full_stream_structural"
    assert not any("removed_bytes=" in warning for warning in id_only_warnings)

    off_boundary_fill_at = 1500
    corrupted_row = (
        row_one[:off_boundary_fill_at]
        + b"\x55" * 120
        + row_one[off_boundary_fill_at:]
    )
    off_boundary_records, off_boundary_warnings = _scan_csie_data_records_in_stream(
        prefix + corrupted_row + row_two,
        source="test",
        known_image_ids={4357},
        expected_cols_by_image={4357: 1504},
    )
    failed_summary, _pixels = parse_csie_data_row_packet(off_boundary_records[0])
    assert failed_summary["checksum_status"] == "failed"
    assert off_boundary_records[0].acceptance_mode == "csie_full_stream_structural"
    assert not any("removed_bytes=" in warning for warning in off_boundary_warnings)


def test_csie_boundary_fill_repair_requires_unique_checksum_valid_deletion():
    original_row = bytearray(_csie_row_packet(4357, 1))
    prefix = b"\x33" * 336
    fill_at = 2040 - len(prefix) - 120
    original_row[fill_at : fill_at + 120] = b"\x55" * 120
    original_row[-4:] = csie_data_additive_checksum(bytes(original_row[12:-4])).to_bytes(
        4, "big"
    )
    corrupted_row = (
        bytes(original_row[:fill_at])
        + b"\x55" * 120
        + bytes(original_row[fill_at:])
    )

    records, warnings = _scan_csie_data_records_in_stream(
        prefix + corrupted_row + _csie_row_packet(4357, 2),
        source="test",
        known_image_ids={4357},
        expected_cols_by_image={4357: 1504},
    )

    failed_summary, _pixels = parse_csie_data_row_packet(records[0])
    assert failed_summary["checksum_status"] == "failed"
    assert records[0].acceptance_mode == "csie_full_stream_structural"
    assert not any("removed_bytes=" in warning for warning in warnings)


def test_csie_boundary_fill_repair_rejects_bad_fill_and_successor_evidence():
    row_one = _csie_row_packet(4357, 1)
    prefix = b"\x33" * 336
    fill_at = 2040 - len(prefix) - 120
    valid_fill_corruption = row_one[:fill_at] + b"\x55" * 120 + row_one[fill_at:]

    bad_fill = bytearray(b"\x55" * 120)
    bad_fill[17] = 0x54
    streams = [
        prefix
        + row_one[:fill_at]
        + bytes(bad_fill)
        + row_one[fill_at:]
        + _csie_row_packet(4357, 2),
        prefix + valid_fill_corruption + _csie_row_packet(4357, 3),
        prefix
        + valid_fill_corruption
        + _csie_row_packet(4357, 2, checksum_valid=False),
    ]

    for stream in streams:
        records, warnings = _scan_csie_data_records_in_stream(
            stream,
            source="test",
            known_image_ids={4357},
            expected_cols_by_image={4357: 1504},
        )
        failed_summary, _pixels = parse_csie_data_row_packet(records[0])
        assert failed_summary["checksum_status"] == "failed"
        assert records[0].acceptance_mode == "csie_full_stream_structural"
        assert not any("removed_bytes=" in warning for warning in warnings)


def test_csie_boundary_fill_repair_rejects_unaligned_and_oversized_excess():
    row_one = _csie_row_packet(4357, 1)
    row_two = _csie_row_packet(4357, 2)
    prefix = b"\x33" * 336
    unaligned_extra = 121
    unaligned_fill_at = 2040 - len(prefix) - unaligned_extra
    unaligned = (
        prefix
        + row_one[:unaligned_fill_at]
        + b"\x55" * unaligned_extra
        + row_one[unaligned_fill_at:]
        + row_two
    )
    oversized_extra = 2044
    oversized_fill_at = 2040 - len(prefix)
    oversized = (
        prefix
        + row_one[:oversized_fill_at]
        + b"\x55" * oversized_extra
        + row_one[oversized_fill_at:]
        + row_two
    )

    for stream in (unaligned, oversized):
        records, warnings = _scan_csie_data_records_in_stream(
            stream,
            source="test",
            known_image_ids={4357},
            expected_cols_by_image={4357: 1504},
        )
        assert records[0].acceptance_mode == "csie_full_stream_structural"
        assert not any("removed_bytes=" in warning for warning in warnings)


def _metadata_less_inference_args(
    row_indices,
    *,
    cols=1504,
    valid_indices=None,
    terminal_flag=2,
    chunks=None,
):
    row_indices = set(row_indices)
    valid_indices = row_indices if valid_indices is None else set(valid_indices)
    row = np.zeros(cols, dtype=np.uint16)
    rows = {row_index: row for row_index in row_indices}
    quality = {
        row_index: "valid_be" if row_index in valid_indices else "failed"
        for row_index in row_indices
    }
    sequence_flags = {row_index: 1 for row_index in row_indices}
    if 2000 in row_indices:
        sequence_flags[2000] = terminal_flag
    return (
        rows,
        Counter({cols: len(rows)}),
        quality,
        sequence_flags,
        [] if chunks is None else chunks,
    )


def test_metadata_less_inference_accepts_sep10_image_4357_shape():
    row_indices = {5, *range(7, 2001)}
    valid_indices = set(range(8, 2001))
    assert infer_csie_uncompressed_dimensions_from_rows(
        *_metadata_less_inference_args(
            row_indices,
            valid_indices=valid_indices,
        )
    ) == (2000, 1504, 1)


def test_metadata_less_inference_requires_checksum_valid_coverage():
    assert infer_csie_uncompressed_dimensions_from_rows(
        *_metadata_less_inference_args(
            range(1, 2001),
            valid_indices=range(101, 2001),
        )
    ) == (2000, 1504, 1)
    assert infer_csie_uncompressed_dimensions_from_rows(
        *_metadata_less_inference_args(
            range(1, 2001),
            valid_indices=range(102, 2001),
        )
    ) is None
    assert infer_csie_uncompressed_dimensions_from_rows(
        *_metadata_less_inference_args(
            range(1, 2001),
            valid_indices=(1, 2000),
        )
    ) is None


def test_metadata_less_inference_rejects_ambiguous_geometry_and_sequence():
    assert infer_csie_uncompressed_dimensions_from_rows(
        *_metadata_less_inference_args(range(1, 501), cols=376)
    ) is None
    assert infer_csie_uncompressed_dimensions_from_rows(
        *_metadata_less_inference_args(range(1, 2000))
    ) is None
    assert infer_csie_uncompressed_dimensions_from_rows(
        *_metadata_less_inference_args(range(1, 2002))
    ) is None
    assert infer_csie_uncompressed_dimensions_from_rows(
        *_metadata_less_inference_args(range(1, 2001), terminal_flag=1)
    ) is None


def test_metadata_less_inference_rejects_jpegls_first_chunk_header():
    jpegls_header = bytes.fromhex("ffd8fff7000b1007d005e001011100ffda")
    assert infer_csie_uncompressed_dimensions_from_rows(
        *_metadata_less_inference_args(
            range(1, 2001),
            chunks=[{"sequence_count": 1, "payload": jpegls_header}],
        )
    ) is None


def _capture_meta(integration_ms, capture_id, coarse, *, right_shift, filtered):
    return {
        "csie_meta_meta_src": 0,
        "csie_meta_capture_id": capture_id,
        "csie_meta_intg_ms": integration_ms,
        "fpm_proc_cfg_right_shift_meta": right_shift,
        "fpm_proc_cfg_pix_clean_meta": "ENA" if filtered else "DIS",
        "_capture_time_coarse": coarse,
        "_capture_time_fine": 0,
    }


def test_capture_timestamp_combines_integer_milliseconds():
    assert _capture_timestamp_seconds(
        {"_capture_time_coarse": 100, "_capture_time_fine": 234}
    ) == 100.234
    assert _capture_timestamp_seconds(
        {"_capture_time_coarse": 100, "_capture_time_fine": 1_000}
    ) is None


def test_csie_metadata_capture_set_derives_timing_and_normalization():
    packets = [
        *[
            _capture_meta(35, index, 100 + index, right_shift=3, filtered=True)
            for index in range(9)
        ],
        *[
            _capture_meta(15000, 9 + index, 109 + 15 * index, right_shift=2, filtered=True)
            for index in range(4)
        ],
        {"csie_meta_meta_src": 1, "csie_meta_img_id": 42},
    ]

    product, derived, warnings = summarize_csie_metadata_packets(packets)

    assert product["csie_meta_img_id"] == 42
    assert derived["number_stacked_integrations_inner"] == 9
    assert derived["number_stacked_integrations_outer"] == 4
    assert derived["stack_normalization_factor_inner"] == 8
    assert derived["stack_normalization_factor_outer"] == 4
    assert derived["effective_exposure_time_inner"] == 8 * 0.035 / 8
    assert derived["effective_exposure_time_outer"] == 3 * 15 / 4
    assert derived["exposure_time"] == 169 - 100
    assert warnings == []


def test_csie_meta_decode_prefers_generated_engineering_values():
    packet = b"\x0a\x1a\xc0\x07\x00\x01" + (1114).to_bytes(2, "big")
    calls = {}

    class GeneratedCsieMeta:
        def __init__(self, payload, header, file_origin):
            calls["payload"] = payload
            calls["header"] = bytes(header)
            calls["file_origin"] = file_origin
            self.csie_meta_img_id = 4360
            self.csie_meta_bus_volt = 9.8726022
            self.csie_meta_fpga_temp = 33.86074715164372
            self.csie_meta_fpm_row_per_frame = 2000
            self.csie_meta_fpm_pix_per_row = 1504
            self.fpm_proc_cfg_row_bin_meta = 0
            self.fpm_proc_cfg_col_bin_meta = 0

    field_definitions = {
        538: [
            {"ItemName": "header", "DataType": "U48"},
            {"ItemName": "csie_meta_bus_volt", "DataType": "U16"},
        ]
    }

    decoded = decode_csie_meta_packet(
        packet,
        field_definitions,
        GeneratedCsieMeta,
    )

    assert decoded["csie_meta_bus_volt"] == 9.8726022
    assert decoded["csie_meta_fpga_temp"] == 33.86074715164372
    assert calls == {
        "payload": packet[6:],
        "header": packet[:6],
        "file_origin": "csie_image_products",
    }

    record = PacketRecord(
        packet_index=0,
        source_offset=12,
        apid=538,
        packet_len=len(packet),
        source="test",
        acceptance_mode="test",
        checksum_validated=False,
        original_primary_header_endian="big",
        primary_header_normalized=False,
        payload_16bit_words_swapped=False,
        packet=packet,
    )
    image_ids, expected_cols, warnings = _metadata_expectations_from_records(
        [record],
        field_definitions,
        GeneratedCsieMeta,
    )
    assert image_ids == {4360}
    assert expected_cols == {4360: 1504}
    assert warnings == []


def test_csie_meta_decode_falls_back_to_raw_ctdb_fields():
    packet = b"\x0a\x1a\xc0\x07\x00\x01" + (1114).to_bytes(2, "big")
    field_definitions = {
        538: [
            {"ItemName": "header", "DataType": "U48"},
            {"ItemName": "csie_meta_bus_volt", "DataType": "U16"},
        ]
    }

    decoded = decode_csie_meta_packet(packet, field_definitions)

    assert decoded["csie_meta_bus_volt"] == 1114


def test_csie_meta_processing_mode_accepts_generated_state_name():
    assert csie_meta_processing_mode(
        {"icm_proc_cfg_encoding_meta": "JPEG_LS"}
    ) == (True, None, [])


def test_generated_decoder_fields_preserve_assignment_order():
    class GeneratedPacket:
        def __init__(self):
            self.z_field = 1
            self.a_field = 2
            self.middle_field = 3

    assert list(_decoder_object_fields(GeneratedPacket())) == [
        "z_field",
        "a_field",
        "middle_field",
    ]


def test_level0_5_output_projection_filters_spares_and_groups_csie_fields():
    raw = {
        "version": 0,
        "REUSABLE_SPARE_10": 0,
        "csie_meta_bus_volt": 9.87,
        "REUSABLE_SPARE_32": 29.6,
        "csie_meta_detBackside_temp": 23.8,
        "csie_meta_capture_id": 6579,
        "csie_meta_img_id": 4360,
    }

    packet_fields = _level0_5_output_fields(raw, apid=538)
    assert "REUSABLE_SPARE_10" not in packet_fields
    assert packet_fields["csie_meta_adc_thermistor_temp"] == 29.6
    assert "csie_meta_img_id" in packet_fields
    assert "image_counter" not in packet_fields
    assert list(packet_fields).index("csie_meta_capture_id") < list(packet_fields).index(
        "csie_meta_bus_volt"
    )
    assert list(packet_fields).index("csie_meta_bus_volt") < list(packet_fields).index(
        "csie_meta_detBackside_temp"
    )

    product_fields = _level0_5_output_fields(
        raw,
        apid=538,
        canonicalize_csie_product=True,
        engineering_units_applied=True,
    )
    assert product_fields["image_counter"] == 4360
    assert product_fields["detector_temp"] == 23.8
    assert "csie_meta_img_id" not in product_fields
    assert "csie_meta_detBackside_temp" not in product_fields

    raw_fallback = _level0_5_output_fields(
        {"csie_meta_detBackside_temp": 1671},
        apid=538,
        canonicalize_csie_product=True,
        engineering_units_applied=False,
    )
    assert raw_fallback == {"csie_meta_detBackside_temp": 1671}


def test_spare_filter_does_not_change_generic_ctdb_bit_offsets():
    decoded = generic_ctdb_decode_packet(
        b"\x01\xff\x02",
        [
            {"ItemName": "first", "DataType": "U8"},
            {"ItemName": "REUSABLE_SPARE_8", "DataType": "U8"},
            {"ItemName": "last", "DataType": "U8"},
        ],
    )

    assert decoded == {"first": 1, "REUSABLE_SPARE_8": 255, "last": 2}
    assert _level0_5_output_fields(decoded, apid=538) == {"first": 1, "last": 2}


def test_csie_json_preserves_nesting_and_uses_product_crosswalk(tmp_path):
    output = tmp_path / "image_4360_meta.json"
    meta = {
        "version": 0,
        "REUSABLE_SPARE_10": 0,
        "csie_meta_bus_volt": 9.8726022,
        "REUSABLE_SPARE_32": 29.5896,
        "csie_meta_detBackside_temp": 23.8449,
        "csie_meta_detBoard_temp": 24.1275,
        "csie_meta_img_id": 4360,
    }
    packets = [
        {
            "REUSABLE_SPARE_24": 0,
            "REUSABLE_SPARE_32": 29.4,
            "csie_meta_detBackside_temp": 23.7,
            "csie_meta_img_id": 4360,
        }
    ]

    _write_csie_meta_json(
        output,
        4360,
        meta,
        packets,
        engineering_units_applied=True,
    )
    payload = json.loads(output.read_text())

    assert next(iter(payload)) == "image_id"
    assert list(payload)[-1] == "metadata_packets"
    assert payload["image_counter"] == 4360
    assert payload["detector_temp"] == 23.8449
    assert payload["detector_temp_board"] == 24.1275
    assert payload["csie_meta_adc_thermistor_temp"] == 29.5896
    assert not any("reusable_spare" in key.casefold() for key in payload)
    nested = payload["metadata_packets"][0]
    assert nested["csie_meta_img_id"] == 4360
    assert nested["csie_meta_detBackside_temp"] == 23.7
    assert nested["csie_meta_adc_thermistor_temp"] == 29.4
    assert not any("reusable_spare" in key.casefold() for key in nested)


def test_csie_fits_uses_spreadsheet_names_and_omits_spares(tmp_path):
    from astropy.io import fits

    output = tmp_path / "image_4360.fits"
    _write_csie_fits(
        output,
        np.zeros((2, 3), dtype=np.uint16),
        image_id=4360,
        meta={
            "REUSABLE_SPARE_10": 0,
            "REUSABLE_SPARE_32": 29.5896,
            "csie_meta_img_id": 4360,
            "csie_meta_bus_volt": 9.8726022,
            "csie_meta_detBackside_temp": 23.8449,
            "csie_meta_detBoard_temp": 24.1275,
        },
        inventory_row={},
        engineering_units_applied=True,
    )

    header = fits.getheader(output)
    assert header["LEVEL"] == 0.5
    assert header["BUNIT"] == "DN"
    assert header["IMGCTR"] == 4360
    assert header["DET_TEMP"] == 23.8449
    assert header["BRD_TEMP"] == 24.1275
    assert header["ADC_TEMP"] == 29.5896
    assert "BUSVOLT" in header
    assert "IMGID" not in header
    assert "DETBACKS" not in header
    assert "DETBOARD" not in header
    assert not any(key.startswith("REUSABL") for key in header)
    assert header.index("BUSVOLT") < header.index("DET_TEMP")


def _ccsds_packet(
    apid: int,
    payload: bytes,
    *,
    sequence: int = 0,
    secondary_header: bool = True,
) -> bytes:
    first_word = apid | (0x0800 if secondary_header else 0)
    sequence_word = 0xC000 | (sequence & 0x3FFF)
    length_field = len(payload) - 1
    return (
        first_word.to_bytes(2, "big")
        + sequence_word.to_bytes(2, "big")
        + length_field.to_bytes(2, "big")
        + payload
    )


def _uhf_segment_packets(inner_packet: bytes, *, copies: int = 2) -> list[bytes]:
    chunks = [
        inner_packet[offset : offset + UHF_MAX_SEGMENT_PAYLOAD_LEN]
        for offset in range(0, len(inner_packet), UHF_MAX_SEGMENT_PAYLOAD_LEN)
    ]
    packets: list[bytes] = []
    payload_sequence = int.from_bytes(inner_packet[2:4], "big")
    for index, chunk in enumerate(chunks):
        flag = (
            UHF_SEGMENT_FLAG_START
            if index == 0
            else UHF_SEGMENT_FLAG_END
            if index == len(chunks) - 1
            else UHF_SEGMENT_FLAG_MIDDLE
        )
        segment_header = (
            UHF_PLAYBACK_APID.to_bytes(2, "big")
            + payload_sequence.to_bytes(2, "big")
            + bytes((index, flag))
        )
        outer = _ccsds_packet(
            UHF_SEGMENTED_APID,
            segment_header + chunk,
            sequence=index,
            secondary_header=False,
        )
        packets.extend([outer] * copies)
    return packets


def _direct_playback_packet(payload: bytes, *, sequence: int) -> bytes:
    metadata = bytes((sequence & 0xFF,)) * PLAYBACK_METADATA_LEN
    return _ccsds_packet(
        UHF_PLAYBACK_APID,
        metadata + payload,
        sequence=sequence,
        secondary_header=False,
    )


def _xband_frame(
    packet_zone: bytes,
    *,
    master_count: int,
    checksum_footer: bool,
    corrupt_footer: bool = False,
) -> bytes:
    first_word = 66 << 4
    data_field_status = 3 << 11
    protected_prefix = (
        first_word.to_bytes(2, "big")
        + bytes((master_count & 0xFF, master_count & 0xFF))
        + data_field_status.to_bytes(2, "big")
        + b"\x00\x00"
    )
    if checksum_footer:
        assert len(packet_zone) == 2040
        footer = fletcher32_words_be(protected_prefix + packet_zone).to_bytes(
            4, "big"
        )
        if corrupt_footer:
            footer = footer[:-1] + bytes((footer[-1] ^ 0x01,))
        data_field = packet_zone + footer
    else:
        assert len(packet_zone) == 2044
        data_field = packet_zone
    frame = SYNC_MARKER + protected_prefix + data_field
    assert len(frame) == 2056
    return frame


def test_xband_fixed_checksum_footer_layout_strips_invalid_footers_too():
    packet_zones = [bytes((value,)) * 2040 for value in (0x11, 0x22, 0x33)]
    frames = b"".join(
        _xband_frame(
            packet_zone,
            master_count=index,
            checksum_footer=True,
            corrupt_footer=index == 2,
        )
        for index, packet_zone in enumerate(packet_zones)
    )

    fixed, stats = strip_xband_frame_records(
        frames,
        drop_idle_frames=False,
        strip_out_of_phase_xband_artifacts=False,
    )

    assert fixed == reverse_32bit_words(b"".join(packet_zones))
    assert stats.frame_footer_fletcher32_be == 2
    assert stats.frame_footer_checksum_failures == 1
    assert stats.frame_footer_unknown_kept == 0
    assert stats.frame_footer_bytes_removed == 12


def test_xband_footer_layout_probe_stops_after_two_valid_frames(monkeypatch):
    packet_zone = b"\x11" * 2040
    frames = b"".join(
        _xband_frame(
            packet_zone,
            master_count=index,
            checksum_footer=True,
        )
        for index in range(4)
    )
    original_validator = level0_5.validate_transfer_frame_checksum_footer
    calls = 0

    def counting_validator(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_validator(*args, **kwargs)

    monkeypatch.setattr(
        level0_5,
        "validate_transfer_frame_checksum_footer",
        counting_validator,
    )

    assert _has_fixed_transfer_frame_checksum_footer_layout(frames) is True
    assert calls == 2


def test_xband_frame_checksum_rejects_legacy_little_endian_word_variant():
    protected_prefix = b"abcdefgh"
    packet_zone = bytes(range(32))
    protected = protected_prefix + packet_zone
    data_field = packet_zone + fletcher32(protected).to_bytes(4, "big")

    assert validate_transfer_frame_checksum_footer(
        data_field,
        protected_prefix=protected_prefix,
    ) is None


def test_xband_frame_checksum_rejects_little_endian_storage():
    protected_prefix = b"abcdefgh"
    packet_zone = bytes(range(32))
    protected = protected_prefix + packet_zone
    data_field = packet_zone + fletcher32_words_be(protected).to_bytes(4, "little")

    assert validate_transfer_frame_checksum_footer(
        data_field,
        protected_prefix=protected_prefix,
    ) is None


def test_vectorized_fletcher32_matches_scalar_recurrence():
    def scalar(data: bytes, *, big_endian_words: bool) -> int:
        if len(data) % 2:
            data += b"\x00"
        sum1 = 0xFFFF
        sum2 = 0xFFFF
        for index in range(0, len(data), 2):
            if big_endian_words:
                word = (data[index] << 8) | data[index + 1]
            else:
                word = data[index] | (data[index + 1] << 8)
            sum1 = (sum1 + word) % 0xFFFF
            sum2 = (sum2 + sum1) % 0xFFFF
        return (sum2 << 16) | sum1

    source = np.random.default_rng(538).integers(
        0,
        256,
        size=10_001,
        dtype=np.uint8,
    ).tobytes()
    for length in (0, 1, 2, 3, 31, 2048, 2050, 10_001):
        data = source[:length]
        assert fletcher32(data) == scalar(data, big_endian_words=False)
        assert fletcher32_words_be(data) == scalar(data, big_endian_words=True)


def test_vectorized_csie_additive_checksum_matches_scalar_sum():
    source = np.random.default_rng(536).integers(
        0,
        256,
        size=3_011,
        dtype=np.uint8,
    ).tobytes()
    for length in (0, 1, 2, 3, 4, 31, 3008, 3011):
        data = source[:length]
        padded = data + b"\x00" * (-len(data) % 4)
        expected = sum(
            int.from_bytes(padded[index : index + 4], "big")
            for index in range(0, len(padded), 4)
        ) & 0xFFFFFFFF
        assert csie_data_additive_checksum(data) == expected


def test_xband_footerless_layout_keeps_unknown_trailing_payload_bytes():
    packet_zones = [bytes((value,)) * 2044 for value in (0x12, 0x34)]
    frames = b"".join(
        _xband_frame(
            packet_zone,
            master_count=index,
            checksum_footer=False,
        )
        for index, packet_zone in enumerate(packet_zones)
    )

    assert (
        validate_transfer_frame_checksum_footer(
            frames[:2056][12:],
            protected_prefix=frames[:2056][4:12],
        )
        is None
    )
    fixed, stats = strip_xband_frame_records(
        frames,
        drop_idle_frames=False,
        strip_out_of_phase_xband_artifacts=False,
    )

    assert fixed == reverse_32bit_words(b"".join(packet_zones))
    assert stats.frame_footer_checksum_failures == 0
    assert stats.frame_footer_unknown_kept == 2
    assert stats.frame_footer_bytes_removed == 0


def test_reverse_32bit_words_preserves_incomplete_tail():
    assert reverse_32bit_words(bytes(range(11))) == bytes(
        (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10)
    )


def test_packetizer_recovers_valid_packets_across_unaligned_gap():
    first = _ccsds_packet(68, b"first")
    second = _ccsds_packet(72, b"second")

    output, stats = packetize_checksum_valid_ccsds(
        b"\xab\xcd\xef" + first + second,
        {68, 72},
        extract_playback_wrappers=False,
    )

    assert output == first + second
    assert [record.apid for record in stats.records] == [68, 72]
    assert stats.resync_gap_count == 1


def test_packetizer_rejects_packets_outside_configured_apids():
    accepted = _ccsds_packet(68, b"accepted")
    rejected = _ccsds_packet(99, b"rejected")

    output, stats = packetize_checksum_valid_ccsds(
        accepted + rejected,
        {68},
        extract_playback_wrappers=False,
    )

    assert output == accepted
    assert [record.apid for record in stats.records] == [68]


def test_ccsds_candidate_rejects_nonzero_version():
    packet = bytearray(_ccsds_packet(68, b"payload"))
    packet[0] |= 0x20

    assert ccsds_packet_at(bytes(packet), 0, {68}) is None


def test_direct_apid72_chain_is_unwrapped_as_continuous_inner_stream():
    chunks = [b"inner packet fragment 1", b" and fragment 2", b"; packet 3"]
    wrappers = b"".join(
        _direct_playback_packet(chunk, sequence=100 + index)
        for index, chunk in enumerate(chunks)
    )
    raw = b"leading-direct-ccsds" + wrappers + b"trailing-direct-ccsds"

    fixed, stats = unwrap_direct_playback_stream(raw)

    assert fixed == b"leading-direct-ccsds" + b"".join(chunks) + b"trailing-direct-ccsds"
    assert stats.candidates_found == 3
    assert stats.validated_chains == 1
    assert stats.wrappers_stripped == 3
    assert stats.wrapper_bytes_removed == 3 * (6 + PLAYBACK_METADATA_LEN)
    assert stats.payload_bytes_emitted == sum(map(len, chunks))


def test_direct_apid72_isolated_candidate_is_preserved():
    raw = b"prefix" + _direct_playback_packet(b"payload", sequence=7) + b"suffix"

    fixed, stats = unwrap_direct_playback_stream(raw)

    assert fixed == raw
    assert stats.candidates_found == 1
    assert stats.validated_chains == 0
    assert stats.wrappers_stripped == 0


def test_direct_apid72_sequence_break_is_preserved():
    raw = _direct_playback_packet(
        b"first",
        sequence=10,
    ) + _direct_playback_packet(b"second", sequence=12)

    fixed, stats = unwrap_direct_playback_stream(raw)

    assert fixed == raw
    assert stats.candidates_found == 2
    assert stats.wrappers_stripped == 0


def test_ccsds_fixed_binary_stage_unwraps_direct_apid72_chain():
    wrappers = b"".join(
        _direct_playback_packet(bytes((index,)) * 12, sequence=20 + index)
        for index in range(2)
    )

    fixed, stats = build_fixed_binary(wrappers, input_mode=INPUT_MODE_CCSDS)

    assert fixed == bytes((0,)) * 12 + bytes((1,)) * 12
    assert stats.transfer_frame_strip.mode == "hardline_apid72_playback_unwrap"
    assert stats.direct_playback is not None
    assert stats.direct_playback.wrappers_stripped == 2


def test_csie_jpegls_decode_preserves_uint16_pixels():
    source = (np.arange(48, dtype=np.uint16).reshape(6, 8) * 997) % 65535
    codestream = imagecodecs.jpegls_encode(source)

    decoded = _decode_csie_jpegls_uint16(codestream)

    assert decoded.dtype == np.uint16
    np.testing.assert_array_equal(decoded, source)


def test_csie_jpegls_recovery_decompresses_reversed_32bit_words():
    source = (np.arange(48, dtype=np.uint16).reshape(6, 8) * 997) % 65535
    codestream = imagecodecs.jpegls_encode(source)
    padded = codestream + b"\x00" * (-len(codestream) % 4)
    reversed_words = b"".join(
        padded[i : i + 4][::-1] for i in range(0, len(padded), 4)
    )

    decoded, recovered, mode, _prefix_removed, _trailing_removed = (
        _decode_recovered_csie_jpegls(reversed_words)
    )

    assert mode == "reverse_32bit_words"
    assert recovered == codestream
    np.testing.assert_array_equal(decoded, source)


def test_jpegls_trim_ignores_eoi_without_preceding_soi():
    recovered, eoi_found, trimmed = _trim_jpegls_at_eoi(
        b"partial entropy data\xff\xd9trailing bytes"
    )

    assert recovered == b"partial entropy data\xff\xd9trailing bytes"
    assert eoi_found is False
    assert trimmed == 0


def test_csie_png_uses_rotated_inferno_preview(tmp_path):
    source = np.arange(48, dtype=np.uint16).reshape(6, 8)
    output_path = tmp_path / "preview.png"

    _write_csie_png(output_path, source)

    with PilImage.open(output_path) as image:
        actual = np.asarray(image.convert("RGB"))
    expected = _csie_preview_rgb_uint8(source)
    assert actual.shape == (8, 6, 3)
    np.testing.assert_array_equal(actual, expected)


def test_uhf_apid73_reassembly_deduplicates_and_strips_playback_headers():
    direct_packet = _ccsds_packet(1, b"D" * 10)
    playback_payload = bytes((index * 7) & 0xFF for index in range(700))
    metadata = bytes(range(PLAYBACK_METADATA_LEN))
    inner_packet = _ccsds_packet(
        UHF_PLAYBACK_APID,
        metadata + playback_payload,
        sequence=57,
        secondary_header=False,
    )
    segment_packets = _uhf_segment_packets(inner_packet, copies=2)
    raw = direct_packet + b"side-channel-junk" + b"".join(segment_packets)

    fixed, stats = unwrap_uhf_playback_stream(
        raw,
        valid_apids={1},
        expected_packet_bytes={1: len(direct_packet)},
    )

    assert fixed == direct_packet + playback_payload
    assert stats.complete_playback_packets == 1
    assert stats.incomplete_playback_packets == 0
    assert stats.unique_segments_seen == len(segment_packets) // 2
    assert stats.duplicate_segment_packets == len(segment_packets) // 2
    assert stats.conflicting_segment_indices == 0
    assert stats.direct_packets_preserved == 1
    assert stats.non_wrapper_bytes_dropped == len(b"side-channel-junk")


def test_uhf_apid73_reassembly_uses_majority_copy_on_conflict():
    playback_payload = bytes(index & 0xFF for index in range(400))
    inner_packet = _ccsds_packet(
        UHF_PLAYBACK_APID,
        bytes(PLAYBACK_METADATA_LEN) + playback_payload,
        sequence=91,
        secondary_header=False,
    )
    packets = _uhf_segment_packets(inner_packet, copies=3)
    corrupt = bytearray(packets[0])
    corrupt[-1] ^= 0xFF
    packets[0] = bytes(corrupt)

    fixed, stats = unwrap_uhf_playback_stream(b"".join(packets), valid_apids=set())

    assert fixed == playback_payload
    assert stats.complete_playback_packets == 1
    assert stats.conflicting_segment_indices == 1
    assert any("majority copy" in warning for warning in stats.warnings)


def test_checksum_failed_csie_row_is_left_zero_filled():
    image = assemble_csie_uncompressed_image(
        {
            1: np.array([1, 2, 3], dtype=np.uint16),
            2: np.array([4, 5, 6], dtype=np.uint16),
        },
        {1: "valid_be", 2: "failed"},
        expected_rows=3,
        expected_cols=3,
    )

    np.testing.assert_array_equal(image[0], np.array([1, 2, 3], dtype=np.uint16))
    np.testing.assert_array_equal(image[1], np.zeros(3, dtype=np.uint16))
    np.testing.assert_array_equal(image[2], np.zeros(3, dtype=np.uint16))


def test_combined_sort_time_requires_ccsds_secondary_header():
    timed_packet = _ccsds_packet(
        1,
        (123456).to_bytes(4, "big") + (789).to_bytes(2, "big") + b"payload",
    )
    untimed_packet = _ccsds_packet(
        2,
        (999999).to_bytes(4, "big") + (999).to_bytes(2, "big") + b"payload",
        secondary_header=False,
    )

    def record(packet: bytes, apid: int) -> PacketRecord:
        return PacketRecord(
            packet_index=0,
            source_offset=0,
            apid=apid,
            packet_len=len(packet),
            source="test",
            acceptance_mode="test",
            checksum_validated=False,
            original_primary_header_endian="big",
            primary_header_normalized=False,
            payload_16bit_words_swapped=False,
            packet=packet,
        )

    assert _record_sort_time(record(timed_packet, 1)) == (123456, 789)
    assert _record_sort_time(record(untimed_packet, 2)) is None

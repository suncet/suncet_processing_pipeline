"""Focused tests for canonical Level 0.5 ingest and CSIE row policy."""

from collections import Counter
import csv
import json
from types import SimpleNamespace

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


def test_level0_5_defaults_to_checksums_and_duckdb():
    args = level0_5._build_argument_parser().parse_args([])

    assert args.require_packet_checksums is True
    assert args.skip_duckdb is False
    assert args.telemetry_database is None
    assert args.write_decoded_csv is False
    assert args.write_packet_manifest is False
    assert args.write_csie_jpeg2000 is False


def test_level0_5_checksum_bypass_is_explicit():
    args = level0_5._build_argument_parser().parse_args(
        ["--bypass-packet-checksums"]
    )

    assert args.require_packet_checksums is False


def test_level0_5_jpeg2000_output_is_explicit():
    args = level0_5._build_argument_parser().parse_args(
        ["--write-csie-jpeg2000"]
    )

    assert args.write_csie_jpeg2000 is True


def test_explicit_input_file_bypasses_prefix_discovery(tmp_path):
    selected = tmp_path / "capture_b.dat"
    selected.write_bytes(b"selected")
    (tmp_path / "capture_a.dat").write_bytes(b"other")

    mode, prefix, paths = level0_5.resolve_input_files_and_mode(
        tmp_path,
        prefix="capture",
        input_mode="xband",
        input_files=[selected.name],
    )

    assert mode == "xband"
    assert prefix == "capture"
    assert paths == [selected]


def test_decode_defaults_to_direct_rows_without_csv_products(tmp_path, monkeypatch):
    monkeypatch.setattr(
        level0_5,
        "import_bus_decoder_bundle",
        lambda _config: level0_5.DecoderBundle(None, None, None),
    )
    monkeypatch.setattr(
        level0_5,
        "read_ctdb_field_definitions_from_config",
        lambda _config: {},
    )
    packet = b"\x08\x01\xc0\x01\x00\x00\x00"
    record = PacketRecord(
        packet_index=0,
        source_offset=0,
        apid=1,
        packet_len=len(packet),
        source="test",
        acceptance_mode="checksum_bypassed_structural",
        checksum_validated=False,
        original_primary_header_endian="big",
        primary_header_normalized=False,
        payload_16bit_words_swapped=False,
        packet=packet,
    )
    decoded_dir = tmp_path / "decoded"

    decoded = level0_5.decode_packet_records(
        [record],
        SimpleNamespace(),
        {1: "beacon"},
        decoded_dir,
    )

    assert decoded.stats.total_packets == 1
    assert list(decoded.rows_by_apid) == [1]
    assert decoded.packet_names == {1: "beacon"}
    assert list(decoded_dir.iterdir()) == []


def test_csie_hk_wire_length_comes_from_v1_1_9_definition(tmp_path):
    bus = tmp_path / "bus"
    csie = tmp_path / "csie"
    bus.mkdir()
    csie.mkdir()
    (bus / "ct_tlm.csv").write_text("APID,DataType\n")
    (csie / "ct_tlm.csv").write_text("APID,DataType\n537,U3552\n")

    expected = level0_5.read_expected_packet_bytes_from_config(
        SimpleNamespace(bus_ctdb_path=bus, csie_ctdb_path=csie)
    )

    assert expected[537] == 444


def test_csie_hk_444_byte_packet_uses_generated_v1_1_9_checksum(tmp_path, monkeypatch):
    class FakeCsieHk:
        def __init__(self, packet, _header, _file_origin):
            self.SHCOARSE = int.from_bytes(packet[:4], "big")
            self.csie_hk_intg_ms = 123
            self.csie_hk_fletcher32 = int.from_bytes(packet[-4:], "big")

    fake_csie = SimpleNamespace(CSIE_HK=FakeCsieHk)
    monkeypatch.setattr(
        level0_5,
        "import_bus_decoder_bundle",
        lambda _config: level0_5.DecoderBundle(None, None, fake_csie),
    )
    monkeypatch.setattr(
        level0_5,
        "read_ctdb_field_definitions_from_config",
        lambda _config: {},
    )
    body = b"\x00" * 432 + b"\x12\x34" + b"\xaa\xbb\xcc\xdd"
    packet = b"\x0a\x19\xc0\x01\x01\xb5" + body
    record = PacketRecord(
        packet_index=0,
        source_offset=0,
        apid=537,
        packet_len=len(packet),
        source="test",
        acceptance_mode="fletcher32_le_words_stored_be",
        checksum_validated=True,
        original_primary_header_endian="big",
        primary_header_normalized=False,
        payload_16bit_words_swapped=False,
        packet=packet,
    )

    decoded = level0_5.decode_packet_records(
        [record],
        SimpleNamespace(),
        {537: "csie_hk"},
        tmp_path / "decoded",
    )

    row = decoded.rows_by_apid[537][0]
    assert row["csie_hk_fletcher32"] == 0xAABBCCDD
    assert row["csie_hk_intg_ms"] == 123
    assert "csie_hk_schema_status" not in row
    assert "csie_hk_unknown_tail_2bytes_hex" not in row


def test_direct_duckdb_source_key_separates_independent_xband_inputs(
    tmp_path, monkeypatch
):
    captured = {}
    captured_rows = {}
    captured_names = {}

    class FakeTelemetryProcessor:
        def __init__(self, version, database_path, software_versions):
            self.database_path = tmp_path / "telemetry.duckdb"

        def process_decoded_rows(self, rows, names, source_paths):
            captured.update(source_paths)
            captured_rows.update(rows)
            captured_names.update(names)
            return {
                "files_ingested": 1,
                "files_replaced": 0,
                "files_skipped": 0,
                "rows_ingested": 1,
                "rows_removed": 0,
            }

    monkeypatch.setattr(level0_5, "TelemetryProcessor", FakeTelemetryProcessor)
    decoded = SimpleNamespace(
        rows_by_apid={
            1: [{"value": 1}],
            536: [{"image_data": 1}],
            538: [{"image_metadata": 1}],
        },
        packet_names={1: "beacon", 536: "csie_data", 538: "csie_meta"},
    )
    config = SimpleNamespace(
        version_pipeline="test",
        version_bus="bus",
        version_csie="csie",
        version_dsps="dsps",
    )
    args = SimpleNamespace(
        skip_duckdb=False,
        telemetry_database=None,
        duckdb_source_key="capture_a",
    )

    level0_5.ingest_decoded_rows_to_duckdb(decoded, tmp_path, config, args)

    assert captured[1].name == "decoded_apid_0001_beacon__xband_capture_a.csv"
    assert captured_rows == {1: [{"value": 1}]}
    assert captured_names == {1: "beacon"}
    assert set(captured) == {1}


def test_level0_5_ingests_decoded_csv_into_duckdb(tmp_path):
    decoded = tmp_path / "decoded_packets"
    decoded.mkdir()
    (decoded / "decoded_apid_0001_beacon.csv").write_text(
        "time,temperature\n1,10\n2,11\n"
    )
    (decoded / "decoded_apid_0538_csie_meta.csv").write_text(
        "time,image_counter\n1,42\n"
    )
    database = tmp_path / "telemetry.duckdb"
    config = SimpleNamespace(
        version_pipeline="test",
        version_bus="bus-test",
        version_csie="csie-test",
        version_dsps="dsps-test",
    )
    args = SimpleNamespace(skip_duckdb=False, telemetry_database=database)

    summary = level0_5.ingest_decoded_telemetry_to_duckdb(
        decoded, config, args
    )

    assert summary == {
        "files_ingested": 1,
        "files_replaced": 0,
        "files_skipped": 0,
        "rows_ingested": 2,
        "rows_removed": 0,
    }
    assert database.is_file()


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
    assert derived["capture_start_j2000"] == 100
    assert derived["capture_average_j2000"] == 134.5
    assert derived["capture_end_j2000"] == 169
    assert derived["capture_start_iso8601"] == "2000-01-01T00:01:40Z"
    assert derived["capture_average_iso8601"] == "2000-01-01T00:02:14.500Z"
    assert derived["capture_end_iso8601"] == "2000-01-01T00:02:49Z"
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


def test_csie_hk_geometry_association_uses_latest_sample_before_capture():
    meta = {"SHCOARSE": 120, "SHFINE": 0}
    samples = [
        {
            "_geometry_j2000": 100.0,
            "csie_115_hk_icm_rad_sqr": 62500,
            "csie_115_hk_icm_cent_col": 752,
            "csie_115_hk_icm_cent_row": 1000,
        },
        {
            "_geometry_j2000": 110.0,
            "csie_115_hk_icm_rad_sqr": 40000,
            "csie_115_hk_icm_cent_col": 700,
            "csie_115_hk_icm_cent_row": 900,
        },
        {
            "_geometry_j2000": 130.0,
            "csie_115_hk_icm_rad_sqr": 1,
            "csie_115_hk_icm_cent_col": 1,
            "csie_115_hk_icm_cent_row": 1,
        },
    ]

    enriched, warning = level0_5.associate_csie_hk_geometry(meta, samples)

    assert warning is None
    assert enriched["icm_circle_radius_squared_pixels"] == 40000
    assert enriched["icm_circle_center_column_pixels"] == 700
    assert enriched["icm_circle_center_row_pixels"] == 900
    assert enriched["icm_geometry_hk_j2000"] == 110.0
    assert enriched["icm_geometry_hk_age_seconds"] == 10.0

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
        "SHCOARSE": 843_505_701,
        "SHFINE": 770,
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
    assert list(payload)[1] == "metadata_time_iso8601"
    assert list(payload)[-1] == "metadata_packets"
    assert payload["metadata_time_iso8601"] == "2026-09-23T19:08:26.770Z"
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


def test_csie_json_uses_null_for_unavailable_metadata_time(tmp_path):
    output = tmp_path / "image_4360_meta.json"

    _write_csie_meta_json(output, 4360, {"csie_meta_img_id": 4360}, [])

    payload = json.loads(output.read_text())
    assert payload["metadata_time_iso8601"] is None
    assert level0_5._metadata_time_iso8601(None) is None


def test_csie_inventory_places_iso_time_immediately_after_image_id(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        level0_5, "read_ctdb_field_definitions_from_config", lambda _config: {}
    )
    monkeypatch.setattr(
        level0_5, "read_expected_packet_bytes_from_config", lambda _config: {}
    )
    monkeypatch.setattr(
        level0_5,
        "import_bus_decoder_bundle",
        lambda _config: SimpleNamespace(csie_pkts=None),
    )

    stats = level0_5.write_csie_image_products(
        [],
        SimpleNamespace(also_save_csie_meta_json=True),
        tmp_path,
    )

    with stats.inventory_path.open(newline="") as handle:
        fieldnames = csv.DictReader(handle).fieldnames
    assert fieldnames[:2] == ["image_id", "metadata_time_iso8601"]


def test_csie_image_writer_accepts_apid537_geometry_records(tmp_path, monkeypatch):
    class GeneratedCsieHk:
        def __init__(self, payload, header, file_origin):
            self.csie_115_hk_icm_rad_sqr = 62500
            self.csie_115_hk_icm_cent_col = 752
            self.csie_115_hk_icm_cent_row = 1000

    monkeypatch.setattr(
        level0_5, "read_ctdb_field_definitions_from_config", lambda _config: {}
    )
    monkeypatch.setattr(
        level0_5,
        "import_bus_decoder_bundle",
        lambda _config: SimpleNamespace(
            csie_pkts=SimpleNamespace(CSIE_META=None, CSIE_HK=GeneratedCsieHk)
        ),
    )
    packet = (
        (0x0800 | 537).to_bytes(2, "big")
        + (0xC000).to_bytes(2, "big")
        + (5).to_bytes(2, "big")
        + (100).to_bytes(4, "big")
        + (250).to_bytes(2, "big")
    )
    record = PacketRecord(
        packet_index=0,
        source_offset=0,
        apid=537,
        packet_len=len(packet),
        source="test",
        acceptance_mode="test",
        checksum_validated=True,
        original_primary_header_endian="big",
        primary_header_normalized=False,
        payload_16bit_words_swapped=False,
        packet=packet,
    )

    stats = level0_5.write_csie_image_products(
        [record], SimpleNamespace(also_save_csie_meta_json=False), tmp_path
    )

    assert stats.inventory_path.is_file()
    assert not any("could not supply ICM geometry" in item for item in stats.warnings)


def test_csie_inventory_append_preserves_prior_images(tmp_path, monkeypatch):
    monkeypatch.setattr(
        level0_5, "read_ctdb_field_definitions_from_config", lambda _config: {}
    )
    monkeypatch.setattr(
        level0_5, "read_expected_packet_bytes_from_config", lambda _config: {}
    )
    monkeypatch.setattr(
        level0_5,
        "import_bus_decoder_bundle",
        lambda _config: SimpleNamespace(csie_pkts=None),
    )
    inventory = tmp_path / "csie_image_inventory.csv"
    inventory.write_text(
        "image_id,metadata_time_iso8601,source,warnings\n"
        "41,2026-10-11T00:00:00Z,prior,kept\n"
    )

    level0_5.write_csie_image_products(
        [],
        SimpleNamespace(also_save_csie_meta_json=True),
        tmp_path,
        append_inventory=True,
    )

    with inventory.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["image_id"] == "41"
    assert rows[0]["source"] == "prior"
    assert rows[0]["warnings"] == "kept"


def test_uhf_csie_selection_recovers_nested_xband_stream(monkeypatch, tmp_path):
    recovered = PacketRecord(
        packet_index=0,
        source_offset=0,
        apid=536,
        packet_len=12,
        source="nested",
        acceptance_mode="test",
        checksum_validated=True,
        original_primary_header_endian="big",
        primary_header_normalized=False,
        payload_16bit_words_swapped=False,
        packet=b"\x0a\x18\xc0\x00\x00\x05" + b"\x00" * 6,
    )
    monkeypatch.setattr(
        level0_5,
        "strip_xband_payload_stream",
        lambda _data: (
            b"nested-payload",
            SimpleNamespace(internal_sync_markers_removed=3),
        ),
    )
    monkeypatch.setattr(
        level0_5,
        "recover_xband_csie_records",
        lambda stream, _config: ([recovered], [f"scanned {stream!r}"]),
    )
    spec = level0_5.SourceSpec(
        source_id="uhf",
        input_mode="uhf",
        search_root=tmp_path,
        input_paths=[],
        output_dir=tmp_path,
        prefix="ccsds",
    )

    records, warnings = level0_5.select_csie_records_for_source(
        [], level0_5.SYNC_MARKER + b"payload", spec, object()
    )

    assert records == [recovered]
    assert recovered.source_provenance_quality == "nested_xband_in_playback_stream_scan"
    assert any("3 embedded sync markers" in warning for warning in warnings)


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
            "capture_start_j2000": 843_505_700.0,
            "capture_average_j2000": 843_505_707.5,
            "capture_end_j2000": 843_505_715.0,
            "capture_start_iso8601": "2026-09-23T19:08:25Z",
            "capture_average_iso8601": "2026-09-23T19:08:32.500Z",
            "capture_end_iso8601": "2026-09-23T19:08:40Z",
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
    assert header["DATE-OBS"] == "2026-09-23T19:08:25Z"
    assert header["DATE-BEG"] == "2026-09-23T19:08:25Z"
    assert header["DATE-AVG"] == "2026-09-23T19:08:32.500Z"
    assert header["DATE-END"] == "2026-09-23T19:08:40Z"
    assert header["J2K-BEG"] == 843_505_700.0
    assert header["J2K-AVG"] == 843_505_707.5
    assert header["J2K-END"] == 843_505_715.0
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


def test_packetizer_can_return_records_without_compact_packet_stream():
    packet = _ccsds_packet(1, b"accepted")

    output, stats = packetize_checksum_valid_ccsds(
        packet,
        {1},
        extract_playback_wrappers=False,
        emit_packet_stream=False,
    )

    assert output == b""
    assert stats.packet_bytes == len(packet)
    assert [record.packet for record in stats.records] == [packet]


def test_packetizer_candidate_scan_preserves_overlapping_headers():
    packet = _ccsds_packet(68, b"accepted")

    output, stats = packetize_checksum_valid_ccsds(
        b"\x08" + packet,
        {8, 68},
        expected_packet_bytes={8: 99, 68: len(packet)},
        extract_playback_wrappers=False,
    )

    assert output == packet
    assert [record.source_offset for record in stats.records] == [1]
    assert stats.dropped_non_filler_bytes == 1


def test_packetizer_candidate_scan_accounts_for_skipped_filler_and_noise():
    packet = _ccsds_packet(68, b"accepted")
    prefix = b"\x00\x55\xff\x12"
    suffix = b"\x55\xab\x00"

    output, stats = packetize_checksum_valid_ccsds(
        prefix + packet + suffix,
        {68},
        extract_playback_wrappers=False,
    )

    assert output == packet
    assert stats.dropped_filler_bytes == 5
    assert stats.dropped_non_filler_bytes == 2


def test_packetizer_candidate_scan_does_not_visit_every_noise_byte(monkeypatch):
    packet = _ccsds_packet(68, b"accepted")
    original = level0_5.ccsds_packet_at
    calls = 0

    def counted_packet_at(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(level0_5, "ccsds_packet_at", counted_packet_at)
    output, _stats = packetize_checksum_valid_ccsds(
        b"\x12" * 100_000 + packet,
        {68},
        extract_playback_wrappers=False,
    )

    assert output == packet
    assert calls == 1


def test_level0_5_source_processing_keeps_binary_stages_in_memory(tmp_path):
    packet = _ccsds_packet(1, b"cross-file packet")
    first = tmp_path / "ccsds_001.bin"
    second = tmp_path / "ccsds_002.bin"
    first.write_bytes(packet[:4])
    second.write_bytes(packet[4:])
    output_dir = tmp_path / "source_products" / "hardline"
    spec = level0_5.SourceSpec(
        source_id="hardline",
        input_mode=INPUT_MODE_CCSDS,
        search_root=tmp_path,
        input_paths=[first, second],
        output_dir=output_dir,
        prefix="ccsds",
    )
    args = level0_5._build_argument_parser().parse_args(
        [
            "--bypass-packet-checksums",
            "--skip-decode-csv",
            "--skip-csie-images",
        ]
    )

    product = level0_5.process_source_product(
        spec,
        args,
        config=object(),
        apid_names={1: "beacon"},
        valid_apids={1},
        expected_packet_bytes={1: len(packet)},
    )

    assert [record.packet for record in product.records] == [packet]
    assert product.raw_bytes == len(packet)
    assert product.packet_bytes == len(packet)
    assert not any(
        (output_dir / name).exists()
        for name in level0_5.LEGACY_INTERMEDIATE_BASENAMES
    )


def test_single_source_run_writes_no_intermediate_binaries(tmp_path, monkeypatch):
    packet = _ccsds_packet(1, b"single source packet")
    (tmp_path / "ccsds_001.bin").write_bytes(packet)
    monkeypatch.setattr(
        level0_5,
        "read_apid_names_from_config",
        lambda _config: {1: "beacon"},
    )
    monkeypatch.setattr(
        level0_5,
        "read_expected_packet_bytes_from_config",
        lambda _config: {1: len(packet)},
    )

    level0_5.run(
        [
            "--input-mode",
            "ccsds",
            "--bypass-packet-checksums",
            "--skip-decode-csv",
            "--skip-csie-images",
        ],
        _prepared=(object(), tmp_path),
    )

    assert not any(
        (tmp_path / name).exists()
        for name in level0_5.LEGACY_INTERMEDIATE_BASENAMES
    )


def test_combined_processing_writes_products_but_no_packet_binary(
    tmp_path,
    monkeypatch,
):
    packet = _ccsds_packet(1, b"combined packet")
    (tmp_path / "ccsds_001.bin").write_bytes(packet)
    monkeypatch.setattr(
        level0_5,
        "read_apid_names_from_config",
        lambda _config: {1: "beacon"},
    )
    monkeypatch.setattr(
        level0_5,
        "read_expected_packet_bytes_from_config",
        lambda _config: {1: len(packet)},
    )
    args = level0_5._build_argument_parser().parse_args(
        [
            "--input-mode",
            "combined",
            "--bypass-packet-checksums",
            "--skip-decode-csv",
            "--skip-csie-images",
        ]
    )

    level0_5.run_combined_pipeline(args, config=object(), folder=tmp_path)

    summary = tmp_path / level0_5.COMBINED_SOURCE_SUMMARY_BASENAME
    assert summary.is_file()
    assert "in_memory_from_raw_inputs" in summary.read_text()
    assert not any(
        path.name in level0_5.LEGACY_INTERMEDIATE_BASENAMES
        for path in tmp_path.rglob("*")
    )


def test_legacy_intermediate_is_ignored_and_left_untouched(tmp_path, capsys):
    legacy = tmp_path / "packets_valid.bin"
    legacy.write_bytes(b"stale cache")

    level0_5.warn_about_legacy_intermediates(tmp_path)

    assert legacy.read_bytes() == b"stale cache"
    output = capsys.readouterr().out
    assert "will not be read or updated" in output
    assert "packets_valid.bin" in output


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

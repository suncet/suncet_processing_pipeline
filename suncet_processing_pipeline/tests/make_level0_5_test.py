"""Focused tests for canonical Level 0.5 ingest and CSIE row policy."""

import imagecodecs
import numpy as np
from PIL import Image as PilImage

from ..make_level0_5 import (
    INPUT_MODE_CCSDS,
    PacketRecord,
    PLAYBACK_METADATA_LEN,
    UHF_MAX_SEGMENT_PAYLOAD_LEN,
    UHF_PLAYBACK_APID,
    UHF_SEGMENTED_APID,
    UHF_SEGMENT_FLAG_END,
    UHF_SEGMENT_FLAG_MIDDLE,
    UHF_SEGMENT_FLAG_START,
    _csie_preview_rgb_uint8,
    _decode_csie_jpegls_uint16,
    _record_sort_time,
    _write_csie_png,
    assemble_csie_uncompressed_image,
    build_fixed_binary,
    ccsds_packet_at,
    packetize_checksum_valid_ccsds,
    unwrap_direct_playback_stream,
    unwrap_uhf_playback_stream,
)


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

"""Unit tests for playback inner-stream CCSDS helpers (no full CTDB required)."""

from ..make_level0_5 import Level0_5


def _minimal_ccsds_packet(apid, data_after_primary):
    """
    Build a CCSDS Space Packet (primary header + data_after_primary).
    data_after_primary length L satisfies CCSDS length field = len(data_after_primary) - 1.
    """
    L = len(data_after_primary) - 1
    hdr = bytearray(6)
    w = apid & 0x7FF
    hdr[0] = (w >> 8) & 0xFF
    hdr[1] = w & 0xFF
    hdr[2] = 0
    hdr[3] = 0
    hdr[4] = (L >> 8) & 0xFF
    hdr[5] = L & 0xFF
    return bytes(hdr) + data_after_primary


def test_extract_ccsds_packets_unaligned_with_gap():
    body = b"\x00" * 6 + b"\x00\x00\x00\x00"  # 10 bytes after primary -> L=9
    p1 = _minimal_ccsds_packet(68, body)
    p2 = _minimal_ccsds_packet(72, body)
    junk = b"\xab\xcd\xef"
    stream = junk + p1 + p2
    out = Level0_5.extract_ccsds_packets_unaligned(stream, valid_apids=None)
    assert len(out) == 2
    assert out[0] == p1
    assert out[1] == p2


def test_extract_ccsds_packets_unaligned_respects_valid_apids():
    body = b"\x00" * 6 + b"\x00\x00\x00\x00"
    p_ok = _minimal_ccsds_packet(68, body)
    p_other = _minimal_ccsds_packet(99, body)
    stream = p_ok + p_other
    out = Level0_5.extract_ccsds_packets_unaligned(stream, valid_apids={68})
    assert len(out) == 1
    assert out[0] == p_ok


def test_strip_uhf_segmentation_header():
    # Primary (6) + PAY_APID(2) + PAY_SEQ(2) + SEG_COUNT(1) + SEG_FLAGS(1) + payload
    pkt = bytearray(6 + 6 + 3)
    pkt[6:8] = (0x1234).to_bytes(2, "big")
    pkt[6 + 5] = 2  # end of segment
    pkt[6 + 6 :] = b"\xaa\xbb\xcc"
    pay_apid, seg_flags, payload = Level0_5.strip_uhf_segmentation_header(bytes(pkt))
    assert pay_apid == 0x1234
    assert seg_flags == 2
    assert payload == b"\xaa\xbb\xcc"


def test_is_valid_ccsds_header_accepts_minimal_packet():
    body = b"\x00" * 6 + b"\x00\x00\x00\x00"
    pkt = _minimal_ccsds_packet(68, body)
    ok, plen = Level0_5._is_valid_ccsds_header(pkt, 0, valid_apids=None)
    assert ok is True
    assert plen == len(pkt)


def test_is_valid_ccsds_header_rejects_nonzero_version():
    body = b"\x00" * 6 + b"\x00\x00\x00\x00"
    pkt = bytearray(_minimal_ccsds_packet(68, body))
    pkt[0] = 0xFF
    ok, _ = Level0_5._is_valid_ccsds_header(bytes(pkt), 0, valid_apids=None)
    assert ok is False

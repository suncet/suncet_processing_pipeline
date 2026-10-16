"""Independent HDLC boundary and public-beacon filtering checks."""
from pathlib import Path

import numpy as np
import pytest

from suncet_processing_pipeline.satnogs.ax25 import crc16_x25, SUNCET_AX25_HEADER
from suncet_processing_pipeline.satnogs.iq_audit import (
    FLAG, is_public_beacon, recover_hdlc_frames,
)


def wire_bits(frame):
    """Test transmitter: stuff one zero after every five payload ones."""
    out = list(FLAG)
    ones = 0
    for bit in np.unpackbits(np.frombuffer(frame, dtype=np.uint8), bitorder="little"):
        out.append(bit)
        ones = ones + 1 if bit else 0
        if ones == 5:
            out.append(0)
            ones = 0
    return np.array(out + list(FLAG), dtype=np.uint8)


def test_hdlc_known_x25_check_value_and_fcs_preservation():
    # Published CRC-16/X-25 check value for the canonical nine ASCII bytes.
    frame = b"123456789\x6e\x90"
    wire = np.r_[np.zeros(3, dtype=np.uint8), wire_bits(frame), FLAG]
    recovered = list(recover_hdlc_frames(wire))
    assert len(recovered) == 1
    assert recovered[0][0] == 3
    assert recovered[0][2] == frame
    damaged = wire.copy()
    damaged[3 + 8 + 2] ^= 1
    assert not list(recover_hdlc_frames(damaged))


def test_stuffed_literal_flag_and_abort_are_not_packet_boundaries():
    data = b"\xff\x7e\xff\xf8\x00\x7e"
    frame = data + crc16_x25(data).to_bytes(2, "little")
    wire = wire_bits(frame)
    assert list(recover_hdlc_frames(wire))[0][2] == frame
    assert list(recover_hdlc_frames(wire))[0][3] > 0
    # Destroy the first stuffed zero, yielding an abort run of >=7 ones.
    wire[8 + 5] = 1
    assert not list(recover_hdlc_frames(wire))


@pytest.mark.parametrize("mutation", ["none", "wrong_apid", "bad_inner_checksum"])
def test_public_filter_requires_inner_contract_even_after_valid_rf_crc(mutation):
    path = Path(__file__).parents[1] / "satnogs/test_data/suncet_apid1_synthetic_252.hex"
    packet = bytearray.fromhex(path.read_text())
    if mutation == "wrong_apid":
        packet[1] = 2
    elif mutation == "bad_inner_checksum":
        packet[30] ^= 1
    data = SUNCET_AX25_HEADER + packet
    frame = data + crc16_x25(data).to_bytes(2, "little")
    assert list(recover_hdlc_frames(wire_bits(frame)))[0][2] == frame
    assert is_public_beacon(frame, fcs_present=True) == (mutation == "none")

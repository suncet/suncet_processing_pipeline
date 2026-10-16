"""SunCET CTDB 2.0.5: public decoder tests."""

import json
import math
import struct
from pathlib import Path

import pytest
from kaitaistruct import ValidationFailedError

from satnogsdecoders import decoder
from satnogsdecoders.decode_frame import decode_frame_to_dict, \
    decode_frame_to_fields
from satnogsdecoders.decoder.suncet import Suncet

FIXTURE_DIR = Path(__file__).resolve().parents[1] / 'contrib' / 'suncet'
FRAME = bytes.fromhex((FIXTURE_DIR / 'synthetic.hex').read_text())
EXPECTED = json.loads((FIXTURE_DIR / 'synthetic_expected.json').read_text())


def repair_checksum(frame):
    """Independent weighted-sum Fletcher oracle over little-endian words."""
    result = bytearray(frame)
    words = struct.unpack('<124H', result[16:264])
    low = sum(words) % 65535
    high = sum(
        (124 - index) * word for index, word in enumerate(words)) % 65535
    result[-4:] = struct.pack('>I', high * 65536 + low)
    return bytes(result)


def test_public_values_and_discovery():
    """Expose the 111 reviewed aliases through the normal entrypoint."""
    assert repair_checksum(FRAME) == FRAME
    values = decode_frame_to_fields('suncet', FRAME)
    assert values == pytest.approx(EXPECTED)
    assert len(values) == 111
    assert set(decoder.get_fields(Suncet, empty=True)) == set(EXPECTED)
    for name in ('mode_system_mode', 'eps_pwr_state_csie'):
        assert isinstance(values[name], int)
        assert not isinstance(values[name], bool)


@pytest.mark.parametrize('offset', range(268))
def test_every_frame_byte_is_protected(offset):
    """Check header, public and excluded bytes, and checksum integrity."""
    corrupt = bytearray(FRAME)
    corrupt[offset] ^= 1
    with pytest.raises(ValidationFailedError):
        Suncet.from_bytes(corrupt)


@pytest.mark.parametrize('frame', [
    FRAME[16:],
    FRAME[:-1],
    FRAME + b'\x00',
    FRAME + b'\x00\x00',
    b'\x7e' + FRAME + b'\x7e',
    FRAME * 2,
])
def test_exact_receiver_boundary(frame):
    """No scanning, FCS inference, padding, or concatenated frames."""
    with pytest.raises(ValidationFailedError):
        Suncet.from_bytes(frame)


@pytest.mark.parametrize('offset, value', [
    (0, b'\x08\x02'),
    (0, b'\x18\x01'),
    (0, b'\x00\x01'),
    (0, b'\x28\x01'),
    (4, b'\x00\xf4'),
    (10, b'\x03\xe8'),
])
def test_inner_envelope_checked_on_construction(offset, value):
    """Reject other APIDs/commands and bad header/time with valid checksum."""
    frame = bytearray(FRAME)
    frame[16 + offset:16 + offset + len(value)] = value
    with pytest.raises(ValidationFailedError):
        Suncet.from_bytes(repair_checksum(frame))


def test_excluded_bytes_never_appear_in_raw_structure():
    """The full-structure API hides checksum words and excluded regions."""
    frame = bytearray(FRAME)
    frame[16 + 124:16 + 128] = bytes.fromhex('deadbeef')
    result = decode_frame_to_dict('suncet', repair_checksum(frame))
    assert set(result) == {'beacon'}
    text = json.dumps(result)
    assert 'deadbeef' not in text
    assert 'opaque' not in text
    assert 'fletcher' not in text
    assert 'words' not in result['beacon']


def test_unknown_state_keeps_integer_type():
    """Unknown codes remain observable without changing Influx field type."""
    frame = bytearray(FRAME)
    frame[16 + 227] = 255
    values = decode_frame_to_fields('suncet', repair_checksum(frame))
    assert values['mode_system_mode'] == 255
    assert isinstance(values['mode_system_mode'], int)
    assert not isinstance(values['mode_system_mode'], bool)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_sensor_is_missing_without_rejecting_other_fields(value):
    """Normalize infinities to NaN so DB omits these missing readings."""
    frame = bytearray(FRAME)
    frame[16 + 68:16 + 72] = struct.pack('>f', value)
    values = decode_frame_to_fields('suncet', repair_checksum(frame))
    assert math.isnan(values['dsps_flare_level'])
    assert len(values) == 111
    assert values['battery_1_voltage'] == pytest.approx(
        EXPECTED['battery_1_voltage'])

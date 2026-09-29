"""Exercise the compiled current-layout parser and validated public decoder.

These tests import the real generated parser; they do not emulate Kaitai.
The manual packet below uses fixed CTDB 2.0.5 public offsets and an independent
weighted-sum checksum calculation, rather than the schema/fixture generator.
"""

from enum import Enum
from io import BytesIO
import json
from pathlib import Path
import struct

from kaitaistruct import KaitaiStream, ValidationFailedError
import pytest

from suncet_processing_pipeline.satnogs.beacon_contract import BeaconValidationError
from suncet_processing_pipeline.satnogs.decoder import decode_public_beacon
from suncet_processing_pipeline.satnogs.generated_suncet_apid1 import SuncetApid1
from suncet_processing_pipeline.satnogs.public_schema import load_public_beacon_schema


FIXTURE_DIRECTORY = Path(__file__).parents[1] / "satnogs" / "test_data"


def _with_checksum(packet: bytes | bytearray) -> bytes:
    """Compute Fletcher sums in closed form, independently of the contract."""

    packet = bytearray(packet)
    covered = bytes(packet[:-4])
    covered += b"\x00" if len(covered) % 2 else b""
    words = struct.unpack(f"<{len(covered) // 2}H", covered)
    sum1 = sum(words) % 65535
    sum2 = sum((len(words) - index) * word for index, word in enumerate(words)) % 65535
    packet[-4:] = struct.pack(">I", (sum2 << 16) | sum1)
    return bytes(packet)


def _manual_packet() -> bytes:
    packet = bytearray(252)
    struct.pack_into(">HHHIH", packet, 0, 0x0801, 0xC02A, 245, 833_326_475, 999)
    struct.pack_into(">I", packet, 48, 0x01020304)  # Last retained storage pointer.
    struct.pack_into(">I", packet, 52, 123456)  # Boot time now follows byte48 pointer.
    struct.pack_into(">f", packet, 68, -6.25)
    struct.pack_into(">i", packet, 72, -200_000_000)
    struct.pack_into(">i", packet, 108, -12345)
    struct.pack_into(">i", packet, 120, -32768)
    struct.pack_into(">H", packet, 144, 1000)
    struct.pack_into(">H", packet, 156, 1000)
    struct.pack_into(">h", packet, 204, -321)
    struct.pack_into(">h", packet, 212, -1234)
    packet[220] = 0b10_10_01_00  # Capture2, WP6 enabled, WP5 passive, WP4 disabled.
    packet[226] = 18
    packet[227] = 2
    packet[228] = 0xF6  # Signed -10 Celsius.
    packet[234] = 250  # CTDB2.0.5 unsigned magnitude, no legacy scale.
    packet[235] = 40
    packet[243] = 0b11_1_0_0_0_1_0  # Opaque prefix11, battery1 charging, UHF on.
    packet[244] = 0b0_0_0_0_01_00  # UHF alive.
    packet[245] = 0b1_1_1_0_0_110  # Validity flags, sun-point state6.
    packet[246] = 1
    packet[247] = 0xA5  # Current opaque tail byte is covered by checksum.
    return _with_checksum(packet)


def _parse(packet: bytes) -> SuncetApid1:
    stream = KaitaiStream(BytesIO(packet))
    parsed = SuncetApid1(stream)
    assert stream.is_eof(), "The compiled parser must consume the entire packet"
    return parsed


def test_compiled_parser_and_entrypoint_match_all_public_fixture_values():
    packet = bytes.fromhex(
        (FIXTURE_DIRECTORY / "suncet_apid1_synthetic_252.hex").read_text(encoding="ascii")
    )
    expected = json.loads(
        (FIXTURE_DIRECTORY / "suncet_apid1_synthetic_252_expected.json").read_text(
            encoding="utf-8"
        )
    )
    parsed = _parse(packet)
    decoded = decode_public_beacon(packet)
    public_names = {field.public_name for field in load_public_beacon_schema()}
    assert len(public_names) == 111
    assert set(decoded) == set(expected) == public_names
    assert all(type(value) in (int, float, str) for value in decoded.values())
    for name, reference in expected.items():
        actual = getattr(parsed, name)
        actual = actual.name.upper() if isinstance(actual, Enum) else actual
        wanted = reference["engineering"]
        if isinstance(wanted, float):
            assert actual == pytest.approx(wanted), name
            assert decoded[name] == pytest.approx(wanted), name
        else:
            assert actual == wanted, name
            assert decoded[name] == wanted, name


def test_manual_packet_has_independent_current_offsets_and_engineering_values():
    values = decode_public_beacon(_manual_packet())
    expected = {
        "ccsds_apid": 1,
        "ccsds_sequence_flags": 3,
        "ccsds_sequence_count": 42,
        "ccsds_packet_length_field": 245,
        "spacecraft_time_seconds_since_2000": 833_326_475,
        "spacecraft_time_milliseconds": 999,
        "csie_nand_sci_write_ptr": 0x01020304,
        "time_since_boot": 123456,
        "dsps_flare_level": -6.25,
        "adcs_body_rate_1": -1.0,
        "adcs_wheel_speed_1": -24.69,
        "xband_pa_temp": -32.0,
        "battery_1_voltage": 8.8623,
        "dsps_visible_sps_sun_pos_x": -321,
        "dsps_sensor_board_temp": -12.34,
        "csie_capture_state": 2,
        "fault_protection_watchpoint_6_state": "ENABLED",
        "fault_protection_watchpoint_5_state": "PASSIVE",
        "fault_protection_watchpoint_4_state": "DISABLED",
        "clt_hours_until_reboot": 18,
        "mode_system_mode": "SCIENCE",
        "uhf_temp": -10,
        "dsps_flare_magnitude": 250,
        "dsps_flare_phase": "RISING_FLARE",
        "battery_1_charging_state": "CHARGING",
        "eps_pwr_state_uhf": "ON",
        "eps_pwr_state_csie": "OFF",
        "uhf_alive": "ALIVE",
        "adcs_att_valid": "YES",
        "adcs_ref_valid": "YES",
        "adcs_time_valid": "YES",
        "adcs_sun_point_state": "ON_SUN",
        "xband_data_source": "CDH",
    }
    for name, wanted in expected.items():
        assert values[name] == (pytest.approx(wanted) if isinstance(wanted, float) else wanted)
    # Direct polynomial sum is independent of the generator's Horner expression.
    expected_temperature = sum(
        coefficient * 1000**degree
        for degree, coefficient in enumerate(
            [125.55, -0.13622, 0.000098611, -0.000000044176, 1.0125e-11, -9.3905e-16]
        )
    )
    assert values["cdh_temp"] == pytest.approx(expected_temperature)


def test_private_regions_and_checksum_never_appear_in_public_output():
    packet = bytearray(_manual_packet())
    expected = decode_public_beacon(packet)
    for start, end in [(124, 128), (130, 140), (222, 226), (229, 233), (236, 243), (247, 248)]:
        packet[start:end] = b"\xFF" * (end - start)
    packet[243] ^= 0xC0  # Only the two excluded leading bits.
    values = decode_public_beacon(_with_checksum(packet))
    assert values == expected
    assert not any("opaque" in name or "checksum" in name or name.endswith("_raw") for name in values)
    assert "csie_meta_nand_sci_write_ptr" not in values


def test_unknown_enumerations_remain_numeric():
    packet = bytearray(_manual_packet())
    packet[227] = 254
    packet[235] = 255
    packet[220] = (packet[220] & ~0x30) | 0x30  # Undeclared WP6 state3.
    packet[244] = (packet[244] & ~0x0C) | 0x0C  # Undeclared UHF alive state3.
    values = decode_public_beacon(_with_checksum(packet))
    assert values["mode_system_mode"] == 254
    assert values["dsps_flare_phase"] == 255
    assert values["fault_protection_watchpoint_6_state"] == 3
    assert values["uhf_alive"] == 3


@pytest.mark.parametrize("length", [251, 253, 256])
def test_compiled_parser_rejects_noncurrent_lengths_with_matching_header(length):
    packet = bytearray(_manual_packet())
    packet = packet[:length] if length < len(packet) else packet + bytes(length - len(packet))
    packet[4:6] = (length - 7).to_bytes(2, "big")
    packet = _with_checksum(packet)
    with pytest.raises(ValidationFailedError):
        _parse(packet)
    with pytest.raises(BeaconValidationError, match="requires 252"):
        decode_public_beacon(packet)


@pytest.mark.parametrize("length_field", [0, 244, 246, 65535])
def test_compiled_parser_rejects_wrong_declared_length_on_current_size(length_field):
    packet = bytearray(_manual_packet())
    packet[4:6] = length_field.to_bytes(2, "big")
    packet = _with_checksum(packet)
    with pytest.raises(ValidationFailedError):
        _parse(packet)
    with pytest.raises(BeaconValidationError, match="CCSDS header declares"):
        decode_public_beacon(packet)


@pytest.mark.parametrize("length", [0, 1, 5, 6, 11, 247, 248, 251])
def test_compiled_parser_and_entrypoint_reject_truncation(length):
    packet = _manual_packet()[:length]
    with pytest.raises((EOFError, ValidationFailedError)):
        _parse(packet)
    with pytest.raises(BeaconValidationError, match="requires 252"):
        decode_public_beacon(packet)


@pytest.mark.parametrize("primary_word", [0x0001, 0x0802, 0x1801, 0x2801])
def test_compiled_parser_and_entrypoint_reject_nonbeacon_primary_words(primary_word):
    packet = bytearray(_manual_packet())
    packet[0:2] = primary_word.to_bytes(2, "big")
    packet = _with_checksum(packet)
    with pytest.raises(ValidationFailedError):
        _parse(packet)
    with pytest.raises(BeaconValidationError):
        decode_public_beacon(packet)


@pytest.mark.parametrize("fine", [0, 999])
def test_compiled_parser_accepts_fine_time_boundaries(fine):
    packet = bytearray(_manual_packet())
    packet[10:12] = fine.to_bytes(2, "big")
    packet = _with_checksum(packet)
    assert _parse(packet).spacecraft_time_milliseconds == fine
    assert decode_public_beacon(packet)["spacecraft_time_milliseconds"] == fine


@pytest.mark.parametrize("fine", [1000, 65535])
def test_compiled_parser_rejects_invalid_fine_time(fine):
    packet = bytearray(_manual_packet())
    packet[10:12] = fine.to_bytes(2, "big")
    packet = _with_checksum(packet)
    with pytest.raises(ValidationFailedError):
        _parse(packet)
    with pytest.raises(BeaconValidationError, match="expected 0-999 ms"):
        decode_public_beacon(packet)


@pytest.mark.parametrize("offset", [12, 72, 247, 248, 251])
def test_public_entrypoint_rejects_checksum_corruption_before_parsing(offset, monkeypatch):
    packet = bytearray(_manual_packet())
    packet[offset] ^= 1

    def unexpected_parse(*_args, **_kwargs):
        pytest.fail("Corrupt packet reached the compiled parser")

    monkeypatch.setattr("suncet_processing_pipeline.satnogs.decoder.SuncetApid1", unexpected_parse)
    with pytest.raises(BeaconValidationError, match="Fletcher-32 mismatch"):
        decode_public_beacon(packet)


@pytest.mark.parametrize(
    "offset,format_,raw,name,expected",
    [
        (72, ">i", -(2**31), "adcs_body_rate_1", -(2**31) * 5e-9),
        (72, ">i", 2**31 - 1, "adcs_body_rate_1", (2**31 - 1) * 5e-9),
        (204, ">h", -32768, "dsps_visible_sps_sun_pos_x", -32768),
        (204, ">h", 32767, "dsps_visible_sps_sun_pos_x", 32767),
        (228, ">b", -128, "uhf_temp", -128),
        (228, ">b", 127, "uhf_temp", 127),
        (234, ">B", 0, "dsps_flare_magnitude", 0),
        (234, ">B", 255, "dsps_flare_magnitude", 255),
    ],
)
def test_signed_and_unsigned_field_boundaries(offset, format_, raw, name, expected):
    packet = bytearray(_manual_packet())
    struct.pack_into(format_, packet, offset, raw)
    assert decode_public_beacon(_with_checksum(packet))[name] == pytest.approx(expected)

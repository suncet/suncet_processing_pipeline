"""CCSDS packet validation and CTDB decode bridge for realtime display."""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DecodedPacket:
    apid: int
    packet_name: str
    fields: dict[str, Any]
    header: dict[str, int]
    decode_status: str
    decode_error: str = ""
    decoder_kind: str = ""
    checksum_status: str = "not_checked"
    checksum_validated: bool = False
    source: str = ""


@dataclass(frozen=True)
class DecoderMetadata:
    valid_apids: set[int] | None = None
    apid_names: dict[int, str] = field(default_factory=dict)
    expected_packet_bytes: dict[int, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class RealtimePacketDecoder:
    """Decode packet bytes with the same generated CTDB classes used offline."""

    def __init__(
        self,
        processing_config_path: str | Path | None,
        *,
        require_packet_checksums: bool = False,
        drop_failed_checksums: bool = False,
    ) -> None:
        self.processing_config_path = (
            Path(processing_config_path).expanduser().resolve()
            if processing_config_path
            else None
        )
        self.require_packet_checksums = require_packet_checksums
        self.drop_failed_checksums = drop_failed_checksums
        self.bundle = None
        self.field_definitions: dict[int, list[dict[str, str]]] = {}
        self.metadata = DecoderMetadata()
        self._load_offline_decoder_helpers()

    def _load_offline_decoder_helpers(self) -> None:
        self._helper_error = None
        try:
            from suncet_processing_pipeline.config_parser import Config
            from suncet_processing_pipeline import make_level0_5_slowly_built_up as slow
        except Exception as exc:
            self._helper_error = f"{type(exc).__name__}: {exc}"
            return

        self._slow = slow
        if self.processing_config_path is None:
            return

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                config = Config(str(self.processing_config_path))
            apid_names = slow.read_apid_names_from_config(config)
            expected_packet_bytes = slow.read_expected_packet_bytes_from_config(config)
            bundle = slow.import_bus_decoder_bundle(config)
            self.field_definitions = slow.read_ctdb_field_definitions_from_config(config)
            warnings = tuple(getattr(bundle, "warnings", ()) or ())
            self.bundle = bundle
            self.metadata = DecoderMetadata(
                valid_apids=set(apid_names),
                apid_names=apid_names,
                expected_packet_bytes=expected_packet_bytes,
                warnings=warnings,
            )
        except Exception as exc:
            self._helper_error = f"{type(exc).__name__}: {exc}"
            self.metadata = DecoderMetadata(warnings=(self._helper_error,))

    def decode(self, packet: bytes, *, source: str = "") -> DecodedPacket | None:
        header = packet_header_metadata(packet)
        apid = header["apid"]
        packet_name = self.metadata.apid_names.get(apid, f"apid_{apid}")

        checksum_status = "not_checked"
        checksum_validated = False
        if self.require_packet_checksums:
            algorithm = self._validate_checksum(packet, apid)
            if algorithm is None:
                checksum_status = "failed"
                if self.drop_failed_checksums:
                    return None
            else:
                checksum_status = algorithm
                checksum_validated = True

        fields: dict[str, Any] = {}
        decode_status = "decoded"
        decode_error = ""
        decoder_kind = ""

        packet_class, decoder_kind = self._select_generated_decoder(packet_name)
        if packet_class is not None:
            try:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    packet_object = packet_class(packet[6:], bytearray(packet[:6]), "realtime")
                fields.update(self._decoder_object_fields(packet_object))
                decoder_text = output.getvalue().strip().replace("\n", " | ")
                if decoder_text:
                    fields["decoder_messages"] = decoder_text[:1000]
            except Exception as exc:
                decode_status = "decode_failed"
                decode_error = f"{type(exc).__name__}: {exc}"
        else:
            field_rows = self.field_definitions.get(apid)
            if field_rows:
                try:
                    fields.update(self._generic_decode(packet, field_rows))
                    decoder_kind = "generic_ctdb_csv"
                except Exception as exc:
                    decode_status = "decode_failed"
                    decode_error = f"{type(exc).__name__}: {exc}"
                    decoder_kind = "generic_ctdb_csv"
            else:
                decode_status = "no_decoder"
                decode_error = "no generated decoder or CTDB field rows"
                decoder_kind = "none"

        fields.update(_secondary_header_fields(packet, fields))

        return DecodedPacket(
            apid=apid,
            packet_name=packet_name,
            fields=fields,
            header=header,
            decode_status=decode_status,
            decode_error=decode_error,
            decoder_kind=decoder_kind,
            checksum_status=checksum_status,
            checksum_validated=checksum_validated,
            source=source,
        )

    def _validate_checksum(self, packet: bytes, apid: int) -> str | None:
        slow = getattr(self, "_slow", None)
        if slow is None:
            return None
        return slow.validate_packet_checksum(packet, apid)

    def _select_generated_decoder(self, packet_name: str) -> tuple[object | None, str]:
        if self.bundle is None:
            return None, ""
        slow = getattr(self, "_slow", None)
        if slow is None:
            return None, ""
        return slow._select_generated_decoder(packet_name, self.bundle)

    def _decoder_object_fields(self, packet_object: object) -> dict[str, Any]:
        slow = getattr(self, "_slow", None)
        if slow is not None:
            return slow._decoder_object_fields(packet_object)
        fields: dict[str, Any] = {}
        for attr_name in dir(packet_object):
            if attr_name.startswith("_"):
                continue
            value = getattr(packet_object, attr_name)
            if not callable(value):
                fields[attr_name] = value
        return fields

    def _generic_decode(
        self, packet: bytes, field_rows: list[dict[str, str]]
    ) -> dict[str, Any]:
        slow = getattr(self, "_slow", None)
        if slow is None:
            raise RuntimeError("generic CTDB decoder unavailable")
        return slow.generic_ctdb_decode_packet(packet, field_rows)


def packet_header_metadata(packet: bytes) -> dict[str, int]:
    if len(packet) < 6:
        raise ValueError("packet is shorter than the CCSDS primary header")
    first = int.from_bytes(packet[0:2], "big")
    seq = int.from_bytes(packet[2:4], "big")
    length_field = int.from_bytes(packet[4:6], "big")
    return {
        "ccsds_version": (first >> 13) & 0x07,
        "ccsds_type": (first >> 12) & 0x01,
        "ccsds_secondary_header_flag": (first >> 11) & 0x01,
        "apid": first & 0x07FF,
        "sequence_flags": (seq >> 14) & 0x03,
        "sequence_count": seq & 0x3FFF,
        "ccsds_length_field": length_field,
        "ccsds_total_length_from_header": length_field + 7,
    }


def _secondary_header_fields(packet: bytes, fields: dict[str, Any]) -> dict[str, int]:
    header = packet_header_metadata(packet)
    if header["ccsds_secondary_header_flag"] != 1 or len(packet) < 12:
        return {}
    if any(name.startswith("ccsdsSecHeader2_sec") for name in fields):
        return {}
    return {
        "ccsdsSecHeader2_sec": int.from_bytes(packet[6:10], "big"),
        "ccsdsSecHeader2_sub": int.from_bytes(packet[10:12], "big"),
    }

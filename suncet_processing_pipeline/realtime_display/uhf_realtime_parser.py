"""Incremental UHF realtime frame parser.

The parser is intentionally tolerant while the ground bridge contract is still
settling. It accepts:

* AX.25 wrapper + direct CCSDS packet
* AX.25 wrapper + segmented realtime header + CCSDS packet chunks
* CCSDS attached sync marker (ASM) + CCSDS packet
* Direct CCSDS packets when AX.25 has already been stripped
* Segmented realtime header + chunks when AX.25 has been stripped
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import FramingConfig


AX25_ADDRESS_BYTES = 8
AX25_HEADER_BYTES = AX25_ADDRESS_BYTES * 2 + 2
SEGMENT_HEADER_BYTES = 12
CCSDS_PRIMARY_HEADER_BYTES = 6


@dataclass(frozen=True)
class PacketEnvelope:
    packet: bytes
    source: str
    received_monotonic: float
    ax25_present: bool = False
    segmented: bool = False
    segment_key: tuple[int, int, int, int] | None = None


@dataclass
class ParserStats:
    bytes_seen: int = 0
    bytes_dropped: int = 0
    ax25_frames_seen: int = 0
    ax25_stripped_frames_seen: int = 0
    asm_frames_seen: int = 0
    frame_prefix_frames_seen: int = 0
    direct_packets: int = 0
    segment_frames: int = 0
    segmented_packets_completed: int = 0
    incomplete_candidates: int = 0
    parse_errors: int = 0
    max_buffer_depth: int = 0


@dataclass
class SegmentAssembly:
    total_length: int
    created_monotonic: float
    updated_monotonic: float
    chunks: dict[int, bytes] = field(default_factory=dict)
    arrival_order: list[int] = field(default_factory=list)
    saw_start: bool = False
    saw_end: bool = False


@dataclass(frozen=True)
class SegmentHeader:
    packet_id: int
    sequence_count: int
    packet_length: int
    payload_id: int
    payload_sequence_count: int
    segment_count: int
    segment_flags: int

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (
            self.packet_id,
            self.sequence_count,
            self.payload_id,
            self.payload_sequence_count,
        )


class UHFRealtimeParser:
    """Consume arbitrary TCP byte chunks and emit complete CCSDS packets."""

    def __init__(
        self,
        framing: FramingConfig,
        *,
        valid_apids: set[int] | None = None,
    ) -> None:
        self.framing = framing
        self.valid_apids = valid_apids
        self.buffer = bytearray()
        self.stats = ParserStats()
        self._segments: dict[tuple[int, int, int, int], SegmentAssembly] = {}

    def feed(self, data: bytes) -> list[PacketEnvelope]:
        if not data:
            return []
        self.buffer.extend(data)
        self.stats.bytes_seen += len(data)
        self.stats.max_buffer_depth = max(self.stats.max_buffer_depth, len(self.buffer))
        self._enforce_buffer_limit()

        emitted: list[PacketEnvelope] = []
        while self.buffer:
            packet = self._parse_one()
            if packet is None:
                break
            emitted.extend(packet)
        return emitted

    def flush(self) -> list[PacketEnvelope]:
        emitted: list[PacketEnvelope] = []
        while self.buffer:
            before = len(self.buffer)
            packet = self._parse_one(final=True)
            if packet is None:
                if len(self.buffer) == before:
                    self.stats.bytes_dropped += len(self.buffer)
                    self.buffer.clear()
                break
            emitted.extend(packet)
        return emitted

    def _enforce_buffer_limit(self) -> None:
        extra = len(self.buffer) - self.framing.max_buffer_bytes
        if extra > 0:
            del self.buffer[:extra]
            self.stats.bytes_dropped += extra

    def _parse_one(self, *, final: bool = False) -> list[PacketEnvelope] | None:
        candidates = self._candidate_offsets()
        if not candidates:
            keep = CCSDS_PRIMARY_HEADER_BYTES + AX25_HEADER_BYTES + SEGMENT_HEADER_BYTES
            if len(self.buffer) > keep:
                drop = len(self.buffer) - keep
                del self.buffer[:drop]
                self.stats.bytes_dropped += drop
            return None

        offset, kind = candidates[0]
        if offset > 0:
            del self.buffer[:offset]
            self.stats.bytes_dropped += offset

        if kind == "asm":
            result = self._parse_asm_candidate()
        elif kind == "frame_prefix":
            result = self._parse_frame_prefix_candidate()
        elif kind == "ax25":
            result = self._parse_ax25_candidate()
        elif kind == "direct":
            result = self._parse_direct_candidate(ax25_present=False)
        else:
            result = self._parse_segment_candidate(ax25_present=False)

        if result == "need_more":
            if final:
                self.stats.incomplete_candidates += 1
            return None
        if result == "invalid":
            self.stats.parse_errors += 1
            del self.buffer[:1]
            self.stats.bytes_dropped += 1
            return []
        return result

    def _candidate_offsets(self) -> list[tuple[int, str]]:
        strong_candidates: list[tuple[int, str]] = []
        weak_candidates: list[tuple[int, str]] = []
        if self.framing.allow_ccsds_asm and self.framing.ccsds_asm:
            asm_offset = self.buffer.find(self.framing.ccsds_asm)
            if asm_offset >= 0:
                strong_candidates.append((asm_offset, "asm"))

        if self.framing.allow_frame_prefix and self.framing.frame_prefix:
            frame_prefix_offset = self.buffer.find(self.framing.frame_prefix)
            if frame_prefix_offset >= 0:
                strong_candidates.append((frame_prefix_offset, "frame_prefix"))

        if self.framing.allow_ax25_header:
            ax25_offset = self._find_ax25_prefix()
            if ax25_offset >= 0:
                strong_candidates.append((ax25_offset, "ax25"))

        if strong_candidates:
            return sorted(strong_candidates, key=lambda item: item[0])

        if self.framing.allow_missing_ax25_header:
            direct_offset = self._find_direct_ccsds()
            if direct_offset >= 0:
                weak_candidates.append((direct_offset, "direct"))
            if self.framing.segmentation_enabled:
                segment_offset = self._find_segment_header()
                if segment_offset >= 0:
                    weak_candidates.append((segment_offset, "segment"))

        return sorted(weak_candidates, key=lambda item: item[0])

    def _find_ax25_prefix(self) -> int:
        prefixes = self._ax25_prefixes()
        if not prefixes:
            return -1
        found = [self.buffer.find(prefix) for prefix in prefixes]
        found = [idx for idx in found if idx >= 0]
        return min(found) if found else -1

    def _ax25_prefixes(self) -> list[bytes]:
        ctrl_pid = bytes((self.framing.ax25_ctrl & 0xFF, self.framing.ax25_pid & 0xFF))
        prefixes: list[bytes] = []
        destination = self._padded_address(self.framing.ax25_destination)
        source = self._padded_address(self.framing.ax25_source)
        if destination and source:
            prefixes.append(destination + source + ctrl_pid)
        # Useful when the bridge changes address formatting but leaves AX.25 ctrl/PID.
        prefixes.append(ctrl_pid)
        return prefixes

    @staticmethod
    def _padded_address(value: str) -> bytes:
        text = value.encode("ascii", errors="ignore")[:AX25_ADDRESS_BYTES]
        return text.ljust(AX25_ADDRESS_BYTES, b" ")

    def _find_direct_ccsds(self) -> int:
        limit = max(0, len(self.buffer) - CCSDS_PRIMARY_HEADER_BYTES + 1)
        for offset in range(limit):
            info = self._ccsds_packet_info(offset)
            if info is not None:
                return offset
        return -1

    def _find_segment_header(self) -> int:
        limit = max(0, len(self.buffer) - SEGMENT_HEADER_BYTES + 1)
        for offset in range(limit):
            header = self._segment_header_at(offset)
            if header is not None:
                return offset
        return -1

    def _parse_asm_candidate(self) -> list[PacketEnvelope] | str:
        prefix_len = len(self.framing.ccsds_asm)
        if len(self.buffer) < prefix_len + CCSDS_PRIMARY_HEADER_BYTES:
            return "need_more"
        result = self._parse_direct_candidate(
            ax25_present=False,
            prefix_len=prefix_len,
            source="ccsds_asm_direct",
            require_valid_apid=False,
        )
        if result not in ("need_more", "invalid"):
            self.stats.asm_frames_seen += 1
        return result

    def _parse_frame_prefix_candidate(self) -> list[PacketEnvelope] | str:
        prefix_marker = self.framing.frame_prefix
        prefix_len = self.framing.frame_prefix_bytes
        if prefix_len < len(prefix_marker):
            return "invalid"
        if len(self.buffer) < prefix_len + CCSDS_PRIMARY_HEADER_BYTES:
            return "need_more"
        if not self.buffer.startswith(prefix_marker):
            return "invalid"
        result = self._parse_direct_candidate(
            ax25_present=False,
            prefix_len=prefix_len,
            source="frame_prefix_direct",
            require_valid_apid=False,
        )
        if result not in ("need_more", "invalid"):
            self.stats.frame_prefix_frames_seen += 1
        return result

    def _parse_ax25_candidate(self) -> list[PacketEnvelope] | str:
        ax25_len = self._matched_ax25_prefix_len()
        if ax25_len is None:
            return "invalid"
        if len(self.buffer) < ax25_len + CCSDS_PRIMARY_HEADER_BYTES:
            return "need_more"

        self.stats.ax25_frames_seen += 1
        direct = self._parse_direct_candidate(ax25_present=True, prefix_len=ax25_len)
        if direct != "invalid":
            return direct
        if self.framing.segmentation_enabled:
            return self._parse_segment_candidate(ax25_present=True, prefix_len=ax25_len)
        return "invalid"

    def _matched_ax25_prefix_len(self) -> int | None:
        for prefix in self._ax25_prefixes():
            if self.buffer.startswith(prefix):
                return len(prefix)
        return None

    def _parse_direct_candidate(
        self,
        *,
        ax25_present: bool,
        prefix_len: int = 0,
        source: str | None = None,
        require_valid_apid: bool = True,
    ) -> list[PacketEnvelope] | str:
        info = self._ccsds_packet_info(
            prefix_len,
            require_valid_apid=require_valid_apid,
        )
        if info is None:
            return "invalid"
        packet_len = info[1]
        if packet_len < 0:
            return "need_more"
        frame_len = prefix_len + packet_len
        packet = bytes(self.buffer[prefix_len:frame_len])
        del self.buffer[:frame_len]
        self.stats.direct_packets += 1
        if not ax25_present:
            self.stats.ax25_stripped_frames_seen += 1
        return [
            PacketEnvelope(
                packet=packet,
                source=source or ("ax25_direct" if ax25_present else "direct_ccsds"),
                received_monotonic=time.monotonic(),
                ax25_present=ax25_present,
                segmented=False,
            )
        ]

    def _parse_segment_candidate(
        self,
        *,
        ax25_present: bool,
        prefix_len: int = 0,
    ) -> list[PacketEnvelope] | str:
        if len(self.buffer) < prefix_len + SEGMENT_HEADER_BYTES:
            return "need_more"
        header = self._segment_header_at(prefix_len)
        if header is None:
            return "invalid"
        payload_len = self._segment_payload_len(header)
        if payload_len is None:
            return "invalid"
        frame_len = prefix_len + SEGMENT_HEADER_BYTES + payload_len
        if len(self.buffer) < frame_len:
            return "need_more"

        payload_start = prefix_len + SEGMENT_HEADER_BYTES
        payload = bytes(self.buffer[payload_start:frame_len])
        del self.buffer[:frame_len]
        self.stats.segment_frames += 1
        if not ax25_present:
            self.stats.ax25_stripped_frames_seen += 1
        return self._add_segment(header, payload, ax25_present=ax25_present)

    def _ccsds_packet_info(
        self,
        offset: int,
        *,
        require_valid_apid: bool = True,
    ) -> tuple[int, int] | None:
        if offset + CCSDS_PRIMARY_HEADER_BYTES > len(self.buffer):
            return None
        header = self.buffer[offset : offset + CCSDS_PRIMARY_HEADER_BYTES]
        first_word = int.from_bytes(header[0:2], "big")
        version = (first_word >> 13) & 0x07
        packet_type = (first_word >> 12) & 0x01
        secondary_header_flag = (first_word >> 11) & 0x01
        apid = first_word & 0x07FF
        if version != 0 or packet_type != 0 or secondary_header_flag != 1:
            return None
        if (
            require_valid_apid
            and self.valid_apids is not None
            and apid not in self.valid_apids
        ):
            return None
        packet_len = CCSDS_PRIMARY_HEADER_BYTES + int.from_bytes(header[4:6], "big") + 1
        if packet_len < CCSDS_PRIMARY_HEADER_BYTES + 1:
            return None
        if packet_len > self.framing.max_packet_bytes:
            return None
        if offset + packet_len > len(self.buffer):
            return apid, -packet_len
        return apid, packet_len

    def _segment_header_at(self, offset: int) -> SegmentHeader | None:
        if offset + SEGMENT_HEADER_BYTES > len(self.buffer):
            return None
        raw = self.buffer[offset : offset + SEGMENT_HEADER_BYTES]
        header = SegmentHeader(
            packet_id=int.from_bytes(raw[0:2], "big"),
            sequence_count=int.from_bytes(raw[2:4], "big"),
            packet_length=int.from_bytes(raw[4:6], "big"),
            payload_id=int.from_bytes(raw[6:8], "big"),
            payload_sequence_count=int.from_bytes(raw[8:10], "big"),
            segment_count=raw[10],
            segment_flags=raw[11],
        )
        if header.packet_length <= self.framing.segmented_packet_threshold_bytes:
            return None
        if header.packet_length > self.framing.max_packet_bytes:
            return None
        if header.segment_flags not in self._valid_segment_flags():
            return None
        if header.segment_count > self._max_segment_index(header.packet_length):
            return None
        return header

    def _valid_segment_flags(self) -> set[int]:
        return {
            self.framing.segment_flag_start,
            self.framing.segment_flag_middle,
            self.framing.segment_flag_end,
        }

    def _max_segment_index(self, packet_length: int) -> int:
        max_payload = max(1, self.framing.max_segment_payload_bytes)
        return max(0, (packet_length - 1) // max_payload)

    def _segment_payload_len(self, header: SegmentHeader) -> int | None:
        max_payload = max(1, self.framing.max_segment_payload_bytes)
        offset = header.segment_count * max_payload
        remaining = header.packet_length - offset
        if remaining <= 0:
            return None
        if header.segment_flags == self.framing.segment_flag_end:
            return min(max_payload, remaining)
        return min(max_payload, remaining)

    def _add_segment(
        self,
        header: SegmentHeader,
        payload: bytes,
        *,
        ax25_present: bool,
    ) -> list[PacketEnvelope]:
        now = time.monotonic()
        key = header.key
        assembly = self._segments.get(key)
        if assembly is None or header.segment_flags == self.framing.segment_flag_start:
            assembly = SegmentAssembly(
                total_length=header.packet_length,
                created_monotonic=now,
                updated_monotonic=now,
            )
            self._segments[key] = assembly

        assembly.updated_monotonic = now
        assembly.chunks[header.segment_count] = payload
        if header.segment_count not in assembly.arrival_order:
            assembly.arrival_order.append(header.segment_count)
        if header.segment_flags == self.framing.segment_flag_start:
            assembly.saw_start = True
        if header.segment_flags == self.framing.segment_flag_end:
            assembly.saw_end = True

        packet = self._assembled_packet_if_complete(assembly)
        if packet is None:
            return []
        self._segments.pop(key, None)
        self.stats.segmented_packets_completed += 1
        return [
            PacketEnvelope(
                packet=packet,
                source="ax25_segmented" if ax25_present else "segmented",
                received_monotonic=now,
                ax25_present=ax25_present,
                segmented=True,
                segment_key=key,
            )
        ]

    def _assembled_packet_if_complete(self, assembly: SegmentAssembly) -> bytes | None:
        if not assembly.saw_end:
            return None
        expected_indices = range(self._max_segment_index(assembly.total_length) + 1)
        if any(index not in assembly.chunks for index in expected_indices):
            return None
        packet = b"".join(assembly.chunks[index] for index in expected_indices)
        packet = packet[: assembly.total_length]
        if len(packet) != assembly.total_length:
            return None
        if not self._looks_like_ccsds_packet(packet):
            self.stats.parse_errors += 1
            return None
        return packet

    def _looks_like_ccsds_packet(self, packet: bytes) -> bool:
        if len(packet) < CCSDS_PRIMARY_HEADER_BYTES:
            return False
        first_word = int.from_bytes(packet[0:2], "big")
        version = (first_word >> 13) & 0x07
        packet_type = (first_word >> 12) & 0x01
        secondary_header_flag = (first_word >> 11) & 0x01
        apid = first_word & 0x07FF
        packet_len = CCSDS_PRIMARY_HEADER_BYTES + int.from_bytes(packet[4:6], "big") + 1
        if version != 0 or packet_type != 0 or secondary_header_flag != 1:
            return False
        if self.valid_apids is not None and apid not in self.valid_apids:
            return False
        return packet_len == len(packet)

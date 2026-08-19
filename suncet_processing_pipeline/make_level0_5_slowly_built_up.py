"""
Build a new Level 0.5 ingest path one small, checkable stage at a time.

Current first milestones:
1. Read configured raw binary files and concatenate them to ``merged.bin``.
2. Convert that merged stream to ``merged_fixed.bin``:
   - for X-band GSE input:
   - remove the X-band ASM / TM-primary / padding wrapper from each frame,
   - remove a 4-byte transfer-frame trailer when it is filler or validates,
   - remove out-of-phase X-band frame seams that look like RF lapse/resync artifacts.
   - undo the FPGA/VCDU 32-bit endian swap on the remaining frame payload stream.
   - for hardline realtime CCSDS input:
     unwrap validated direct APID 72 playback chains; otherwise pass through unchanged.
   - for UHF/Hydra CCSDS input:
     recursively merge nested ``ccsds_*`` files; ordinary direct CCSDS captures pass
     through unchanged, while APID 73 segmented playback captures are de-duplicated,
     reassembled into APID 72 packets, and stripped to their inner playback payload;
     validated direct APID 72 chains are also unwrapped.
3. Recover structurally plausible CCSDS packets from ``merged_fixed.bin`` and write
   them to ``packets_valid.bin``. Packet checksum validation can be re-enabled once
   the flight software checksum contract is nailed down.
4. Assemble uncompressed CSIE images from the full ``merged_fixed.bin`` payload stream.
   This does not depend on ``packets_valid.bin`` because image rows can dominate the
   downlink and may not be inside APID 68 playback wrappers.

This deliberately does not produce Level 0.5 science products yet. The point of this
script is to make each intermediate binary inspectable before the rest of the pipeline
is layered back in.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import struct
import sys
import types
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from suncet_processing_pipeline.config_parser import Config


SYNC_MARKER = b"\x1a\xcf\xfc\x1d"
DEFAULT_XBAND_PREFIX = "xband_gse"
DEFAULT_HARDLINE_CCSDS_PREFIX = "ccsds"
DEFAULT_UHF_SUBDIR_BASENAME = "hydra_uhf_rundirs"
DEFAULT_MERGED_BASENAME = "merged.bin"
DEFAULT_FIXED_BASENAME = "merged_fixed.bin"
DEFAULT_PACKETS_VALID_BASENAME = "packets_valid.bin"
DEFAULT_DECODED_DIR_BASENAME = "decoded_packets"
DEFAULT_PACKET_MANIFEST_BASENAME = "packets_manifest.csv"
DEFAULT_DECODE_SUMMARY_BASENAME = "decoded_packet_summary.csv"
DEFAULT_CSIE_IMAGE_DIR_BASENAME = "csie_images"
DEFAULT_CSIE_INVENTORY_BASENAME = "csie_image_inventory.csv"
DEFAULT_SOURCE_PRODUCTS_DIR_BASENAME = "source_products"
COMBINED_PACKETS_VALID_BASENAME = "combined_packets_valid.bin"
COMBINED_DECODED_DIR_BASENAME = "combined_decoded_packets"
COMBINED_PACKET_MANIFEST_BASENAME = "combined_packet_manifest.csv"
COMBINED_DECODE_SUMMARY_BASENAME = "combined_decode_summary.csv"
COMBINED_SOURCE_SUMMARY_BASENAME = "combined_source_summary.csv"

# The attached X-band frame definition and the current GSE files agree on this:
#   4-byte ASM + 6-byte TM primary header + 2-byte padding + 2044-byte data field.
TRANSFER_FRAME_SIZE = 2056
TRANSFER_FRAME_SYNC_LEN = 4
TRANSFER_FRAME_PRIMARY_LEN = 6
TRANSFER_FRAME_PADDING_LEN = 2
TRANSFER_FRAME_DATA_START = (
    TRANSFER_FRAME_SYNC_LEN + TRANSFER_FRAME_PRIMARY_LEN + TRANSFER_FRAME_PADDING_LEN
)
TRANSFER_FRAME_DATA_LEN = TRANSFER_FRAME_SIZE - TRANSFER_FRAME_DATA_START
TRANSFER_FRAME_TRAILER_LEN = 4
PLAYBACK_PRIMARY_HEADER_LEN = 6
PLAYBACK_METADATA_LEN = 10
UHF_SEGMENTED_APID = 73
UHF_PLAYBACK_APID = 72
UHF_SEGMENT_HEADER_LEN = 6
UHF_MAX_SEGMENT_PAYLOAD_LEN = 244
UHF_MAX_SEGMENT_PACKET_LEN = (
    PLAYBACK_PRIMARY_HEADER_LEN
    + UHF_SEGMENT_HEADER_LEN
    + UHF_MAX_SEGMENT_PAYLOAD_LEN
)
UHF_SEGMENT_FLAG_MIDDLE = 0
UHF_SEGMENT_FLAG_START = 1
UHF_SEGMENT_FLAG_END = 2
MIN_DIRECT_PLAYBACK_CHAIN_PACKETS = 2

DEFAULT_SPACECRAFT_ID = 66
INPUT_MODE_AUTO = "auto"
INPUT_MODE_XBAND = "xband"
INPUT_MODE_CCSDS = "ccsds"
INPUT_MODE_HARDLINE = "hardline"
INPUT_MODE_UHF = "uhf"
INPUT_MODE_COMBINED = "combined"
CSIE_DATA_APID = 536
CSIE_META_APID = 538
CSIE_SECONDARY_HEADER_LEN = 6
CSIE_ROW_CHECKSUM_LEN = 4
CSIE_MAX_SENSOR_ROWS = 2000
CSIE_MAX_SENSOR_COLS = 1504
JPEG_LS_EOI = b"\xff\xd9"
# FIXME(FSW/CTDB): APID 35 is present in current hardline data and is named
# APID_TLM_DSPS_DATA_PKT in generated CTDB constants/state maps, but it is missing
# from suncet_v2-0-1 packet_definitions/ct_pkt.csv and ct_tlm.csv. Treat it as
# dsps_data until the CTDB packet definitions carry the real row.
TEMP_DSPS_DATA_APID = 35
TEMP_DSPS_DATA_PACKET_NAME = "dsps_data"
TEMP_DSPS_DATA_PACKET_BYTES = 124
SYNTHETIC_XBAND_APID_NAMES = {
    68: "xband_playback",
}
FILLER_BYTES = frozenset((0x00, 0x55, 0xFF))
FRAME_TRAILER_FILLERS = {
    b"\x55" * TRANSFER_FRAME_TRAILER_LEN: "filler_55",
}
MAX_REPAIR_EXTRA_BYTES = 64


@dataclass
class MergeStats:
    input_file_count: int
    input_bytes: int
    output_path: Path


@dataclass
class TransferFrameStripStats:
    mode: str = ""
    boundary_records_seen: int = 0
    boundary_records_with_sync: int = 0
    boundary_records_without_sync: int = 0
    expected_boundary_frame_headers: int = 0
    unexpected_boundary_frame_headers: int = 0
    xband_wrappers_removed: int = 0
    xband_header_bytes_removed: int = 0
    frame_footer_fletcher32_be: int = 0
    frame_footer_fletcher32_le: int = 0
    frame_footer_filler_00: int = 0
    frame_footer_filler_55: int = 0
    frame_footer_filler_ff: int = 0
    frame_footer_unknown_kept: int = 0
    frame_footer_bytes_removed: int = 0
    idle_frames_dropped: int = 0
    idle_frame_bytes_dropped: int = 0
    frame_payload_bytes_kept: int = 0
    internal_xband_like_sequences_seen: int = 0
    marker_only_sequences_seen: int = 0
    internal_xband_wrappers_removed: int = 0
    internal_xband_wrapper_bytes_removed: int = 0
    internal_xband_0x55_trailers_removed: int = 0
    internal_xband_trailer_bytes_removed: int = 0
    internal_xband_wrappers_without_0x55_trailer: int = 0
    internal_sync_markers_removed: int = 0
    internal_sync_bytes_removed: int = 0
    frame_footer_checksum_failures: int = 0
    vcdu_word32_words_reversed: int = 0
    vcdu_word32_bytes_reversed: int = 0
    passthrough_bytes: int = 0


@dataclass
class UhfPlaybackReassemblyStats:
    segment_packets_seen: int = 0
    segment_packet_bytes_seen: int = 0
    segment_groups_seen: int = 0
    unique_segments_seen: int = 0
    duplicate_segment_packets: int = 0
    conflicting_segment_indices: int = 0
    complete_playback_packets: int = 0
    incomplete_playback_packets: int = 0
    orphan_segment_groups: int = 0
    segment_wrapper_bytes_removed: int = 0
    playback_header_bytes_removed: int = 0
    playback_payload_bytes_emitted: int = 0
    non_wrapper_bytes_seen: int = 0
    non_wrapper_bytes_preserved: int = 0
    non_wrapper_bytes_dropped: int = 0
    direct_packets_preserved: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class DirectPlaybackUnwrapStats:
    candidates_found: int = 0
    validated_chains: int = 0
    wrappers_stripped: int = 0
    wrapper_bytes_removed: int = 0
    payload_bytes_emitted: int = 0
    first_wrapper_offset: int | None = None
    last_wrapper_offset: int | None = None


@dataclass(frozen=True)
class UhfSegmentCandidate:
    offset: int
    packet_len: int
    payload_apid: int
    payload_sequence_count: int
    segment_index: int
    segment_flags: int
    payload: bytes = field(repr=False)


@dataclass(frozen=True)
class DirectPlaybackWrapperCandidate:
    offset: int
    packet_len: int
    sequence_count: int


@dataclass
class ApidPacketStats:
    candidate_packets: int = 0
    candidate_bytes: int = 0
    checksum_valid_packets: int = 0
    checksum_valid_bytes: int = 0
    checksum_bypassed_packets: int = 0
    checksum_bypassed_bytes: int = 0
    checksum_failed_packets: int = 0
    checksum_failed_bytes: int = 0
    incomplete_packets: int = 0
    repaired_packets: int = 0
    repaired_repeated_words_removed: int = 0
    min_valid_packet_bytes: int | None = None
    max_valid_packet_bytes: int = 0
    algorithms: Counter = field(default_factory=Counter)


@dataclass
class PacketizeStats:
    packets: int = 0
    packet_bytes: int = 0
    candidate_packets: int = 0
    candidate_bytes: int = 0
    checksum_valid_packets: int = 0
    checksum_bypassed_packets: int = 0
    checksum_bypassed_bytes: int = 0
    fletcher32_valid_packets: int = 0
    csie_additive_valid_packets: int = 0
    repaired_packets: int = 0
    repaired_repeated_words_removed: int = 0
    checksum_failed_candidates: int = 0
    checksum_failed_candidate_bytes: int = 0
    dropped_filler_bytes: int = 0
    dropped_non_filler_bytes: int = 0
    incomplete_packet_candidates: int = 0
    resync_gap_count: int = 0
    max_resync_gap_bytes: int = 0
    playback_wrappers_seen: int = 0
    playback_inner_candidates: int = 0
    malformed_playback_wrappers: int = 0
    apids: Counter = field(default_factory=Counter)
    algorithms: Counter = field(default_factory=Counter)
    per_apid: dict[int, ApidPacketStats] = field(default_factory=dict)
    records: list["PacketRecord"] = field(default_factory=list, repr=False)


@dataclass
class PacketRecord:
    packet_index: int
    source_offset: int
    apid: int
    packet_len: int
    source: str
    acceptance_mode: str
    checksum_validated: bool
    original_primary_header_endian: str
    primary_header_normalized: bool
    payload_16bit_words_swapped: bool
    packet: bytes = field(repr=False)
    source_id: str = ""
    source_mode: str = ""
    source_root: str = ""
    input_file_relative_path: str = ""
    source_packet_index: int | None = None
    source_acceptance_mode: str = ""
    source_provenance_quality: str = ""
    source_output_dir: str = ""
    packet_hash: str = ""
    duplicate_group_id: str = ""
    duplicate_group_size: int = 1
    is_duplicate_packet: bool = False
    combined_time_source: str = ""
    combined_time_coarse: int | None = None
    combined_time_fine: int | None = None


@dataclass
class SourceSpec:
    source_id: str
    input_mode: str
    search_root: Path
    input_paths: list[Path]
    output_dir: Path
    prefix: str


@dataclass
class SourceProduct:
    spec: SourceSpec
    packets_valid_path: Path
    output_dir: Path
    records: list[PacketRecord]
    reused: bool
    merged_path: Path | None = None
    fixed_path: Path | None = None
    packet_bytes: int = 0
    raw_bytes: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class DecoderBundle:
    bus_pkts: object | None
    dsps_decoders: object | None
    csie_pkts: object | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class DecodeStats:
    manifest_path: Path
    summary_path: Path
    decoded_dir: Path
    total_packets: int = 0
    decoded_packets: int = 0
    decode_failed_packets: int = 0
    no_decoder_packets: int = 0
    generated_decoder_packets: int = 0
    generic_ctdb_packets: int = 0
    primary_headers_normalized: int = 0
    payload_16bit_words_swapped: int = 0
    unexpected_little_endian_headers: int = 0
    per_apid: Counter = field(default_factory=Counter)
    decoded_per_apid: Counter = field(default_factory=Counter)
    failed_per_apid: Counter = field(default_factory=Counter)
    no_decoder_per_apid: Counter = field(default_factory=Counter)
    generic_per_apid: Counter = field(default_factory=Counter)
    generated_per_apid: Counter = field(default_factory=Counter)
    output_paths: list[Path] = field(default_factory=list)


@dataclass
class CsieImageStats:
    output_dir: Path
    inventory_path: Path
    source: str = ""
    meta_packets: int = 0
    data_packets: int = 0
    image_ids_seen: int = 0
    images_complete: int = 0
    images_partial: int = 0
    compressed_images_written: int = 0
    compressed_images_decoded: int = 0
    compressed_decode_failures: int = 0
    images_written: int = 0
    fits_written: int = 0
    jp2_written: int = 0
    png_written: int = 0
    jpegls_written: int = 0
    checksum_valid_rows: int = 0
    checksum_failed_rows: int = 0
    checksum_missing_rows: int = 0
    skipped_rows: int = 0
    duplicate_rows: int = 0
    warnings: list[str] = field(default_factory=list)
    output_paths: list[Path] = field(default_factory=list)


@dataclass
class FixStats:
    transfer_frame_strip: TransferFrameStripStats
    packetize: PacketizeStats | None
    fixed_bytes: int
    uhf_playback: UhfPlaybackReassemblyStats | None = None
    direct_playback: DirectPlaybackUnwrapStats | None = None


def resolve_config_data_folder(config: Config) -> Path:
    """Resolve ``paths.data_to_process_path`` using the same rule as ``make_level0_5.py``."""
    data_path = config.data_to_process_path
    if data_path.startswith("/") or data_path.startswith("~"):
        return Path(data_path).expanduser().resolve()

    data_root = os.getenv("suncet_data")
    if not data_root:
        raise RuntimeError(
            "Config data_to_process_path is relative, but the suncet_data environment "
            "variable is not set."
        )
    return (Path(data_root).expanduser() / data_path).resolve()


def discover_prefixed_binary_files(folder: Path, prefix: str) -> list[Path]:
    """
    Return top-level files in ``folder`` whose basename starts with ``prefix``.

    The first test folder also has parsed outputs in subfolders; this intentionally does
    not recurse because the requested first stage is about the raw files in the configured
    folder.
    """
    ignored = {
        DEFAULT_MERGED_BASENAME,
        DEFAULT_FIXED_BASENAME,
        DEFAULT_PACKETS_VALID_BASENAME,
    }
    paths = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.name.startswith(prefix) and path.name not in ignored
    ]
    return sorted(paths, key=lambda p: p.name)


def resolve_uhf_search_root(folder: Path) -> Path:
    """Return the recursive UHF search root for a data folder or UHF child folder."""
    uhf_child = folder / DEFAULT_UHF_SUBDIR_BASENAME
    if uhf_child.is_dir():
        return uhf_child
    return folder


def _uhf_timestamp_sort_parts(path: Path) -> tuple[int, int, int, int, int]:
    """Return sortable ccsds_yyyy_doy_hh_mm_ss parts when present."""
    match = re.match(
        r"^ccsds_(\d{4})_(\d{3})_(\d{2})_(\d{2})_(\d{2})$",
        path.name,
    )
    if match is None:
        return (9999, 999, 99, 99, 99)
    year, doy, hour, minute, second = (int(value) for value in match.groups())
    return (year, doy, hour, minute, second)


def discover_uhf_binary_files(folder: Path, prefix: str) -> list[Path]:
    """Return recursively discovered Hydra UHF files whose basename starts with prefix."""
    search_root = resolve_uhf_search_root(folder)
    paths = [
        path
        for path in search_root.rglob("*")
        if path.is_file() and path.name.startswith(prefix)
    ]
    return sorted(
        paths,
        key=lambda path: (
            _uhf_timestamp_sort_parts(path),
            str(path.relative_to(search_root)),
        ),
    )


def resolve_input_files_and_mode(
    folder: Path,
    *,
    prefix: str | None,
    input_mode: str,
) -> tuple[str, str, list[Path]]:
    """Resolve input mode, prefix, and files for staged ingest."""
    normalized_mode = INPUT_MODE_CCSDS if input_mode == INPUT_MODE_HARDLINE else input_mode
    if normalized_mode not in {
        INPUT_MODE_AUTO,
        INPUT_MODE_XBAND,
        INPUT_MODE_CCSDS,
        INPUT_MODE_UHF,
    }:
        raise ValueError(f"unsupported input mode: {input_mode}")

    if normalized_mode == INPUT_MODE_AUTO and prefix is None:
        xband_paths = discover_prefixed_binary_files(folder, DEFAULT_XBAND_PREFIX)
        ccsds_paths = discover_prefixed_binary_files(folder, DEFAULT_HARDLINE_CCSDS_PREFIX)
        if xband_paths and ccsds_paths:
            raise RuntimeError(
                "Auto input mode found both X-band and hardline CCSDS files. "
                "Pass --input-mode xband or --input-mode ccsds explicitly."
            )
        if xband_paths:
            return INPUT_MODE_XBAND, DEFAULT_XBAND_PREFIX, xband_paths
        if ccsds_paths:
            return INPUT_MODE_CCSDS, DEFAULT_HARDLINE_CCSDS_PREFIX, ccsds_paths
        raise FileNotFoundError(
            f"No top-level files starting with {DEFAULT_XBAND_PREFIX!r} or "
            f"{DEFAULT_HARDLINE_CCSDS_PREFIX!r} in {folder}"
        )

    if normalized_mode == INPUT_MODE_AUTO:
        assert prefix is not None
        inferred_mode = (
            INPUT_MODE_CCSDS
            if prefix.startswith(DEFAULT_HARDLINE_CCSDS_PREFIX)
            else INPUT_MODE_XBAND
        )
        paths = discover_prefixed_binary_files(folder, prefix)
        return inferred_mode, prefix, paths

    effective_prefix = prefix
    if effective_prefix is None:
        effective_prefix = (
            DEFAULT_HARDLINE_CCSDS_PREFIX
            if normalized_mode in {INPUT_MODE_CCSDS, INPUT_MODE_UHF}
            else DEFAULT_XBAND_PREFIX
        )
    if normalized_mode == INPUT_MODE_UHF:
        paths = discover_uhf_binary_files(folder, effective_prefix)
        return normalized_mode, effective_prefix, paths
    paths = discover_prefixed_binary_files(folder, effective_prefix)
    return normalized_mode, effective_prefix, paths


def write_merged_binary(input_paths: list[Path], output_path: Path) -> MergeStats:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_path.open("wb") as fout:
        for path in input_paths:
            with path.open("rb") as fin:
                while True:
                    chunk = fin.read(1024 * 1024)
                    if not chunk:
                        break
                    fout.write(chunk)
                    total += len(chunk)
    return MergeStats(
        input_file_count=len(input_paths),
        input_bytes=total,
        output_path=output_path,
    )


def uhf_segment_candidate_at(data: bytes, offset: int) -> UhfSegmentCandidate | None:
    """Parse one direct APID 73 UHF segment wrapper at ``offset`` when present."""
    minimum_len = PLAYBACK_PRIMARY_HEADER_LEN + UHF_SEGMENT_HEADER_LEN + 1
    if offset < 0 or offset + minimum_len > len(data):
        return None

    first_word = int.from_bytes(data[offset : offset + 2], "big")
    if first_word != UHF_SEGMENTED_APID:
        return None

    packet_len = int.from_bytes(data[offset + 4 : offset + 6], "big") + 7
    if (
        packet_len < minimum_len
        or packet_len > UHF_MAX_SEGMENT_PACKET_LEN
        or offset + packet_len > len(data)
    ):
        return None

    segment_start = offset + PLAYBACK_PRIMARY_HEADER_LEN
    payload_apid = int.from_bytes(data[segment_start : segment_start + 2], "big")
    if payload_apid != UHF_PLAYBACK_APID:
        return None
    payload_sequence_count = int.from_bytes(
        data[segment_start + 2 : segment_start + 4], "big"
    )
    segment_index = data[segment_start + 4]
    segment_flags = data[segment_start + 5]
    if segment_flags not in {
        UHF_SEGMENT_FLAG_MIDDLE,
        UHF_SEGMENT_FLAG_START,
        UHF_SEGMENT_FLAG_END,
    }:
        return None

    payload_start = segment_start + UHF_SEGMENT_HEADER_LEN
    payload = data[payload_start : offset + packet_len]
    if not payload or len(payload) > UHF_MAX_SEGMENT_PAYLOAD_LEN:
        return None
    return UhfSegmentCandidate(
        offset=offset,
        packet_len=packet_len,
        payload_apid=payload_apid,
        payload_sequence_count=payload_sequence_count,
        segment_index=segment_index,
        segment_flags=segment_flags,
        payload=payload,
    )


def direct_playback_wrapper_candidate_at(
    data: bytes,
    offset: int,
) -> DirectPlaybackWrapperCandidate | None:
    """Parse one complete, unsegmented APID 72 playback packet at ``offset``."""
    wrapper_header_len = PLAYBACK_PRIMARY_HEADER_LEN + PLAYBACK_METADATA_LEN
    if offset < 0 or offset + wrapper_header_len > len(data):
        return None

    first_word = int.from_bytes(data[offset : offset + 2], "big")
    if first_word != UHF_PLAYBACK_APID:
        return None

    sequence_word = int.from_bytes(data[offset + 2 : offset + 4], "big")
    if sequence_word >> 14 != 0b11:
        return None

    packet_len = int.from_bytes(data[offset + 4 : offset + 6], "big") + 7
    if packet_len < wrapper_header_len or offset + packet_len > len(data):
        return None
    return DirectPlaybackWrapperCandidate(
        offset=offset,
        packet_len=packet_len,
        sequence_count=sequence_word & 0x3FFF,
    )


def _discover_direct_playback_wrapper_candidates(
    data: bytes,
) -> list[DirectPlaybackWrapperCandidate]:
    candidates: list[DirectPlaybackWrapperCandidate] = []
    marker = UHF_PLAYBACK_APID.to_bytes(2, "big")
    search_at = 0
    while True:
        offset = data.find(marker, search_at)
        if offset < 0:
            break
        candidate = direct_playback_wrapper_candidate_at(data, offset)
        if candidate is not None:
            candidates.append(candidate)
        search_at = offset + 1
    return candidates


def unwrap_direct_playback_stream(
    data: bytes,
) -> tuple[bytes, DirectPlaybackUnwrapStats]:
    """
    Strip primary headers and playback metadata from validated APID 72 chains.

    Playback payload is a continuous inner CCSDS byte stream, so packet boundaries
    inside it need not line up with the outer APID 72 boundaries. Requiring adjacent
    wrappers with incrementing sequence counts avoids treating an APID-like byte
    pattern inside science data as a wrapper.
    """
    stats = DirectPlaybackUnwrapStats()
    candidates = _discover_direct_playback_wrapper_candidates(data)
    stats.candidates_found = len(candidates)
    if not candidates:
        return data, stats

    candidate_by_offset = {candidate.offset: candidate for candidate in candidates}
    successor_offsets: dict[int, int] = {}
    offsets_with_predecessors: set[int] = set()
    for candidate in candidates:
        next_offset = candidate.offset + candidate.packet_len
        successor = candidate_by_offset.get(next_offset)
        if (
            successor is not None
            and successor.sequence_count
            == ((candidate.sequence_count + 1) & 0x3FFF)
        ):
            successor_offsets[candidate.offset] = next_offset
            offsets_with_predecessors.add(next_offset)

    validated: list[DirectPlaybackWrapperCandidate] = []
    for candidate in candidates:
        if candidate.offset in offsets_with_predecessors:
            continue
        chain = [candidate]
        while chain[-1].offset in successor_offsets:
            chain.append(candidate_by_offset[successor_offsets[chain[-1].offset]])
        if len(chain) >= MIN_DIRECT_PLAYBACK_CHAIN_PACKETS:
            stats.validated_chains += 1
            validated.extend(chain)

    if not validated:
        return data, stats

    validated.sort(key=lambda candidate: candidate.offset)
    wrapper_header_len = PLAYBACK_PRIMARY_HEADER_LEN + PLAYBACK_METADATA_LEN
    fixed = bytearray()
    cursor = 0
    for candidate in validated:
        fixed.extend(data[cursor : candidate.offset])
        payload_start = candidate.offset + wrapper_header_len
        packet_end = candidate.offset + candidate.packet_len
        fixed.extend(data[payload_start:packet_end])
        cursor = packet_end
    fixed.extend(data[cursor:])

    stats.wrappers_stripped = len(validated)
    stats.wrapper_bytes_removed = wrapper_header_len * len(validated)
    stats.payload_bytes_emitted = sum(
        candidate.packet_len - wrapper_header_len for candidate in validated
    )
    stats.first_wrapper_offset = validated[0].offset
    stats.last_wrapper_offset = validated[-1].offset
    return bytes(fixed), stats


def _discover_uhf_segment_candidates(data: bytes) -> list[UhfSegmentCandidate]:
    candidates: list[UhfSegmentCandidate] = []
    offset = 0
    while offset + PLAYBACK_PRIMARY_HEADER_LEN + UHF_SEGMENT_HEADER_LEN < len(data):
        candidate = uhf_segment_candidate_at(data, offset)
        if candidate is None:
            offset += 1
            continue
        candidates.append(candidate)
        offset += candidate.packet_len
    return candidates


def _select_uhf_segment_copy(
    copies: list[UhfSegmentCandidate],
) -> tuple[UhfSegmentCandidate, bool]:
    """Select the majority-identical copy, preferring earliest arrival on a tie."""
    counts = Counter((copy.payload, copy.segment_flags) for copy in copies)
    best_count = max(counts.values())
    for copy in copies:
        if counts[(copy.payload, copy.segment_flags)] == best_count:
            return copy, len(counts) > 1
    raise RuntimeError("UHF segment copy selection received no candidates")


def unwrap_uhf_playback_stream(
    data: bytes,
    valid_apids: set[int] | None = None,
    expected_packet_bytes: dict[int, int] | None = None,
) -> tuple[bytes, UhfPlaybackReassemblyStats]:
    """
    Replace APID 73 segment wrappers with reconstructed APID 72 playback payloads.

    APID 73 packets may be repeated by the radio/Hydra path. Copies are grouped by
    the advertised inner APID/sequence and segment index, then majority-selected.
    A complete APID 72 packet is emitted once, without its six-byte primary header
    or ten-byte playback metadata. Incomplete groups are dropped with warnings.
    """
    stats = UhfPlaybackReassemblyStats()
    candidates = _discover_uhf_segment_candidates(data)
    if not candidates:
        stats.non_wrapper_bytes_preserved = len(data)
        return data, stats

    stats.segment_packets_seen = len(candidates)
    stats.segment_packet_bytes_seen = sum(candidate.packet_len for candidate in candidates)
    groups: dict[tuple[int, int], list[UhfSegmentCandidate]] = {}
    for candidate in candidates:
        key = (candidate.payload_apid, candidate.payload_sequence_count)
        groups.setdefault(key, []).append(candidate)
    stats.segment_groups_seen = len(groups)

    replacements: dict[int, bytes] = {}
    for key, group in sorted(
        groups.items(),
        key=lambda item: min(candidate.offset for candidate in item[1]),
    ):
        copies_by_index: dict[int, list[UhfSegmentCandidate]] = {}
        for candidate in group:
            copies_by_index.setdefault(candidate.segment_index, []).append(candidate)

        stats.unique_segments_seen += len(copies_by_index)
        stats.duplicate_segment_packets += sum(
            max(0, len(copies) - 1) for copies in copies_by_index.values()
        )
        selected_by_index: dict[int, UhfSegmentCandidate] = {}
        for segment_index, copies in sorted(copies_by_index.items()):
            selected, conflicting = _select_uhf_segment_copy(copies)
            selected_by_index[segment_index] = selected
            if conflicting:
                stats.conflicting_segment_indices += 1

        first_segment = selected_by_index.get(0)
        if first_segment is None:
            stats.incomplete_playback_packets += 1
            stats.orphan_segment_groups += 1
            stats.warnings.append(
                f"UHF APID 73 group payload_apid={key[0]} sequence={key[1]} "
                "has no segment 0; incomplete fragments were dropped"
            )
            continue
        if first_segment.segment_flags != UHF_SEGMENT_FLAG_START:
            stats.incomplete_playback_packets += 1
            stats.warnings.append(
                f"UHF APID 73 group payload_apid={key[0]} sequence={key[1]} "
                f"segment 0 has flag {first_segment.segment_flags}, not start flag "
                f"{UHF_SEGMENT_FLAG_START}; fragments were dropped"
            )
            continue
        if len(first_segment.payload) < PLAYBACK_PRIMARY_HEADER_LEN:
            stats.incomplete_playback_packets += 1
            stats.warnings.append(
                f"UHF APID 73 group payload_apid={key[0]} sequence={key[1]} "
                "has a short segment 0; fragments were dropped"
            )
            continue

        inner_first_word = int.from_bytes(first_segment.payload[0:2], "big")
        inner_version = (inner_first_word >> 13) & 0x07
        inner_apid = inner_first_word & 0x07FF
        expected_len = int.from_bytes(first_segment.payload[4:6], "big") + 7
        if (
            inner_version != 0
            or inner_apid != key[0]
            or expected_len < PLAYBACK_PRIMARY_HEADER_LEN + PLAYBACK_METADATA_LEN
            or expected_len > UHF_MAX_SEGMENT_PAYLOAD_LEN * 256
        ):
            stats.incomplete_playback_packets += 1
            stats.warnings.append(
                f"UHF APID 73 group payload_apid={key[0]} sequence={key[1]} "
                "does not begin with a matching, bounded CCSDS packet; fragments were dropped"
            )
            continue

        final_index = (expected_len - 1) // UHF_MAX_SEGMENT_PAYLOAD_LEN
        required_indices = set(range(final_index + 1))
        missing_indices = sorted(required_indices - set(selected_by_index))
        bad_layout = False
        if missing_indices:
            bad_layout = True
        for segment_index in sorted(required_indices & set(selected_by_index)):
            segment = selected_by_index[segment_index]
            expected_payload_len = (
                expected_len - UHF_MAX_SEGMENT_PAYLOAD_LEN * final_index
                if segment_index == final_index
                else UHF_MAX_SEGMENT_PAYLOAD_LEN
            )
            expected_flag = (
                UHF_SEGMENT_FLAG_START
                if segment_index == 0
                else UHF_SEGMENT_FLAG_END
                if segment_index == final_index
                else UHF_SEGMENT_FLAG_MIDDLE
            )
            if (
                len(segment.payload) != expected_payload_len
                or segment.segment_flags != expected_flag
            ):
                bad_layout = True

        if bad_layout:
            stats.incomplete_playback_packets += 1
            detail = (
                f"missing segment indices {missing_indices}"
                if missing_indices
                else "segment lengths/flags do not match the advertised packet length"
            )
            stats.warnings.append(
                f"UHF APID 73 group payload_apid={key[0]} sequence={key[1]} "
                f"is incomplete ({detail}); fragments were dropped"
            )
            continue

        assembled = b"".join(
            selected_by_index[index].payload for index in range(final_index + 1)
        )
        assembled = assembled[:expected_len]
        playback_payload = assembled[
            PLAYBACK_PRIMARY_HEADER_LEN + PLAYBACK_METADATA_LEN :
        ]
        replacements[min(candidate.offset for candidate in group)] = playback_payload
        stats.complete_playback_packets += 1
        stats.playback_header_bytes_removed += (
            PLAYBACK_PRIMARY_HEADER_LEN + PLAYBACK_METADATA_LEN
        )
        stats.playback_payload_bytes_emitted += len(playback_payload)

    if stats.conflicting_segment_indices:
        stats.warnings.append(
            f"{stats.conflicting_segment_indices} UHF segment index/indices had "
            "non-identical copies; the majority copy was selected"
        )

    candidate_by_offset = {candidate.offset: candidate for candidate in candidates}
    non_wrapper = bytearray()
    offset = 0
    while offset < len(data):
        candidate = candidate_by_offset.get(offset)
        if candidate is not None:
            stats.segment_wrapper_bytes_removed += candidate.packet_len
            offset += candidate.packet_len
            continue
        non_wrapper.append(data[offset])
        offset += 1

    stats.non_wrapper_bytes_seen = len(non_wrapper)
    if valid_apids is None:
        direct_packets = bytes(non_wrapper)
        stats.non_wrapper_bytes_preserved = len(direct_packets)
    else:
        direct_packets, direct_stats = packetize_checksum_valid_ccsds(
            bytes(non_wrapper),
            valid_apids,
            expected_packet_bytes=expected_packet_bytes,
            bypass_packet_checksums=True,
            extract_playback_wrappers=False,
        )
        stats.direct_packets_preserved = direct_stats.packets
        stats.non_wrapper_bytes_preserved = len(direct_packets)
        stats.non_wrapper_bytes_dropped = len(non_wrapper) - len(direct_packets)

    playback_payloads = b"".join(
        replacements[offset] for offset in sorted(replacements)
    )
    return direct_packets + playback_payloads, stats


def _ct_pkt_candidates(ctdb_path: str) -> tuple[Path, ...]:
    root = Path(ctdb_path).expanduser()
    return (
        root / "packet_definitions" / "ct_pkt.csv",
        root / "ct_pkt.csv",
    )


def read_valid_apids_from_config(config: Config) -> set[int]:
    """Read APIDs from bus and CSIE CTDB CSVs without importing generated decoders."""
    return set(read_apid_names_from_config(config))


def read_apid_names_from_config(config: Config) -> dict[int, str]:
    """Read APID -> packet name from bus and CSIE CTDB CSVs."""
    apids: dict[int, str] = dict(SYNTHETIC_XBAND_APID_NAMES)
    for ctdb_path in (config.bus_ctdb_path, config.csie_ctdb_path):
        csv_path = next((p for p in _ct_pkt_candidates(ctdb_path) if p.is_file()), None)
        if csv_path is None:
            raise FileNotFoundError(f"Could not find ct_pkt.csv under {ctdb_path}")
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    apid = int(row["APID"])
                except (KeyError, TypeError, ValueError):
                    continue
                apids.setdefault(apid, row.get("Name", "").strip())
    if not apids:
        raise RuntimeError("No APIDs were loaded from the configured CTDBs.")
    # FIXME(FSW/CTDB): Remove this once APID 35 is restored to ct_pkt.csv.
    apids.setdefault(TEMP_DSPS_DATA_APID, TEMP_DSPS_DATA_PACKET_NAME)
    return apids


def _ct_tlm_candidates(ctdb_path: str) -> tuple[Path, ...]:
    root = Path(ctdb_path).expanduser()
    return (
        root / "packet_definitions" / "ct_tlm.csv",
        root / "ct_tlm.csv",
    )


def _datatype_bits(data_type: str) -> int | None:
    match = re.fullmatch(r"[A-Z]+(\d+)", data_type.strip())
    if match is None:
        return None
    return int(match.group(1))


def read_expected_packet_bytes_from_config(config: Config) -> dict[int, int]:
    """
    Derive fixed total packet byte counts from CTDB telemetry definitions.

    The CTDB rows include the CCSDS primary header fields, secondary header fields,
    payload fields, and checksum/trailer fields, so the summed size is the full packet
    size expected on the wire. Synthetic variable-length APIDs such as playback APID 68
    are intentionally omitted.
    """
    packet_bits: Counter = Counter()
    for ctdb_path in (config.bus_ctdb_path, config.csie_ctdb_path):
        csv_path = next((p for p in _ct_tlm_candidates(ctdb_path) if p.is_file()), None)
        if csv_path is None:
            continue
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    apid = int(row["APID"])
                except (KeyError, TypeError, ValueError):
                    continue
                bits = _datatype_bits(row.get("DataType", ""))
                if bits is None:
                    continue
                packet_bits[apid] += bits

    expected: dict[int, int] = {}
    for apid, bits in packet_bits.items():
        if bits % 8 != 0:
            continue
        expected[apid] = bits // 8
    # FIXME(FSW/CTDB): Remove this once APID 35 has ct_tlm rows. Current packets
    # carry CCSDS length 117, so the total wire packet size is 6 + 117 + 1 = 124.
    expected.setdefault(TEMP_DSPS_DATA_APID, TEMP_DSPS_DATA_PACKET_BYTES)
    return expected


def parse_transfer_frame_header(frame_start: bytes) -> dict[str, int | bytes]:
    """
    Parse the first 12 bytes of a GSE X-band transfer frame.

    ``frame_start`` must include the 4-byte ASM, the 6-byte TM primary header, and
    the 2-byte padding word shown in the X-band definition.
    """
    if len(frame_start) < TRANSFER_FRAME_DATA_START:
        raise ValueError("Need at least 12 bytes to parse a transfer-frame start")
    primary = frame_start[TRANSFER_FRAME_SYNC_LEN : TRANSFER_FRAME_SYNC_LEN + 6]
    padding = frame_start[
        TRANSFER_FRAME_SYNC_LEN + 6 : TRANSFER_FRAME_DATA_START
    ]
    first_two = int.from_bytes(primary[0:2], "big")
    dfs_word = int.from_bytes(primary[4:6], "big")
    return {
        "transfer_frame_version": (first_two >> 14) & 0x03,
        "spacecraft_id": (first_two >> 4) & 0x03FF,
        "virtual_channel_id": (first_two >> 1) & 0x07,
        "ocf_flag": first_two & 0x01,
        "master_channel_frame_count": primary[2],
        "virtual_channel_frame_count": primary[3],
        "secondary_header_flag": (dfs_word >> 15) & 0x01,
        "sync_flag": (dfs_word >> 14) & 0x01,
        "packet_order_flag": (dfs_word >> 13) & 0x01,
        "segment_length_id": (dfs_word >> 11) & 0x03,
        "first_header_pointer": dfs_word & 0x07FF,
        "padding": padding,
    }


def looks_like_xband_wrapper(
    data: bytes,
    offset: int,
    *,
    spacecraft_id: int = DEFAULT_SPACECRAFT_ID,
    require_zero_padding: bool = True,
) -> bool:
    """
    Return True when ``data[offset:]`` starts with an X-band ASM + TM primary + padding.

    This intentionally does not require a 2056-byte record boundary. The current test
    data contains the same X-band wrapper bytes embedded inside payloads, and leaving
    the 8 bytes after the ASM behind creates false CCSDS packet starts.
    """
    if offset + TRANSFER_FRAME_DATA_START > len(data):
        return False
    if data[offset : offset + TRANSFER_FRAME_SYNC_LEN] != SYNC_MARKER:
        return False
    header = parse_transfer_frame_header(data[offset : offset + TRANSFER_FRAME_DATA_START])
    if header["transfer_frame_version"] != 0:
        return False
    if header["spacecraft_id"] != spacecraft_id:
        return False
    if require_zero_padding and header["padding"] != b"\x00\x00":
        return False
    return True


def validate_transfer_frame_checksum_footer(data_field: bytes) -> str | None:
    """
    Validate a future 4-byte transfer-frame Fletcher32 footer when present.

    Current files may have no frame checksum at all, so this returns None unless the
    last word matches the Fletcher32 calculated over the preceding data-field bytes.
    Both byte orders are checked while the flight-software convention is still settling.

    When the temporary ``0x55555555`` trailer is replaced by a mandatory checksum, wire
    checksum failures into ``TransferFrameStripStats.frame_footer_checksum_failures`` so
    ``print_fix_summary`` can warn that frame-level corruption was seen before packet
    checksums are applied.
    """
    if len(data_field) < TRANSFER_FRAME_TRAILER_LEN:
        return None
    trailer = data_field[-TRANSFER_FRAME_TRAILER_LEN:]
    body = data_field[:-TRANSFER_FRAME_TRAILER_LEN]
    calculated = fletcher32(body)
    if calculated == int.from_bytes(trailer, "big"):
        return "fletcher32_be"
    if calculated == int.from_bytes(trailer, "little"):
        return "fletcher32_le"
    return None


def classify_transfer_frame_trailer(data_field: bytes) -> str | None:
    """
    Return the 4-byte transfer-frame trailer kind if it should be stripped.

    ``None`` means the final 4 bytes look like real payload and should be preserved.
    """
    checksum_kind = validate_transfer_frame_checksum_footer(data_field)
    if checksum_kind is not None:
        return checksum_kind
    if len(data_field) < TRANSFER_FRAME_TRAILER_LEN:
        return None
    return FRAME_TRAILER_FILLERS.get(data_field[-TRANSFER_FRAME_TRAILER_LEN:])


def _count_trailer(stats: TransferFrameStripStats, trailer_kind: str | None) -> None:
    if trailer_kind == "fletcher32_be":
        stats.frame_footer_fletcher32_be += 1
    elif trailer_kind == "fletcher32_le":
        stats.frame_footer_fletcher32_le += 1
    elif trailer_kind == "filler_00":
        stats.frame_footer_filler_00 += 1
    elif trailer_kind == "filler_55":
        stats.frame_footer_filler_55 += 1
    elif trailer_kind == "filler_ff":
        stats.frame_footer_filler_ff += 1
    else:
        stats.frame_footer_unknown_kept += 1


def _is_idle_payload(payload: bytes) -> bool:
    """Return True for frames whose remaining payload is only the X-band idle pattern."""
    return bool(payload) and all(byte == 0x55 for byte in payload)


def _strip_out_of_phase_xband_artifacts(
    payload_stream: bytes,
    stats: TransferFrameStripStats,
    *,
    spacecraft_id: int = DEFAULT_SPACECRAFT_ID,
) -> bytes:
    """
    Remove out-of-phase X-band frame seams after the boundary-aligned strip.

    These are interpreted as RF-lapse/resync artifacts: a temporary 0x55555555 frame
    trailer followed immediately by another X-band ASM + primary header + padding.
    """
    out = bytearray()
    i = 0
    while True:
        marker_at = payload_stream.find(SYNC_MARKER, i)
        if marker_at < 0:
            out.extend(payload_stream[i:])
            break

        out.extend(payload_stream[i:marker_at])
        if looks_like_xband_wrapper(
            payload_stream,
            marker_at,
            spacecraft_id=spacecraft_id,
        ):
            stats.internal_xband_wrappers_removed += 1
            stats.internal_xband_wrapper_bytes_removed += TRANSFER_FRAME_DATA_START
            if out.endswith(b"\x55" * TRANSFER_FRAME_TRAILER_LEN):
                del out[-TRANSFER_FRAME_TRAILER_LEN:]
                stats.internal_xband_0x55_trailers_removed += 1
                stats.internal_xband_trailer_bytes_removed += TRANSFER_FRAME_TRAILER_LEN
            else:
                stats.internal_xband_wrappers_without_0x55_trailer += 1
            i = marker_at + TRANSFER_FRAME_DATA_START
        else:
            stats.internal_sync_markers_removed += 1
            stats.internal_sync_bytes_removed += len(SYNC_MARKER)
            i = marker_at + len(SYNC_MARKER)

    return bytes(out)


def count_internal_sync_like_sequences(
    payload_stream: bytes,
    stats: TransferFrameStripStats,
    *,
    spacecraft_id: int = DEFAULT_SPACECRAFT_ID,
) -> None:
    """Count sync-marker-looking payload sequences without modifying payload bytes."""
    i = 0
    while True:
        marker_at = payload_stream.find(SYNC_MARKER, i)
        if marker_at < 0:
            break
        if looks_like_xband_wrapper(
            payload_stream,
            marker_at,
            spacecraft_id=spacecraft_id,
        ):
            stats.internal_xband_like_sequences_seen += 1
        else:
            stats.marker_only_sequences_seen += 1
        i = marker_at + 1


def is_boundary_aligned_xband_stream(data: bytes) -> bool:
    """Return True when every 2056-byte record starts with the X-band ASM."""
    if not data or len(data) % TRANSFER_FRAME_SIZE != 0:
        return False
    return all(
        data.startswith(SYNC_MARKER, offset)
        for offset in range(0, len(data), TRANSFER_FRAME_SIZE)
    )


def strip_xband_frame_records(
    data: bytes,
    *,
    spacecraft_id: int = DEFAULT_SPACECRAFT_ID,
    drop_idle_frames: bool = True,
    strip_out_of_phase_xband_artifacts: bool = True,
) -> tuple[bytes, TransferFrameStripStats]:
    """
    Strip boundary-aligned X-band transfer-frame wrappers and optional 4-byte trailers.

    This is intentionally frame-aware but not first-header-pointer-aware: each data
    field is copied in order after removing only the outer frame wrapper and any trailer
    that is clearly filler or a valid frame checksum.
    """
    stats = TransferFrameStripStats(mode="frame_records")
    out = bytearray()
    for offset in range(0, len(data), TRANSFER_FRAME_SIZE):
        frame = data[offset : offset + TRANSFER_FRAME_SIZE]
        stats.boundary_records_seen += 1
        if len(frame) != TRANSFER_FRAME_SIZE or not frame.startswith(SYNC_MARKER):
            stats.boundary_records_without_sync += 1
            out.extend(frame)
            stats.frame_payload_bytes_kept += len(frame)
            continue

        stats.boundary_records_with_sync += 1
        header = parse_transfer_frame_header(frame[:TRANSFER_FRAME_DATA_START])
        if (
            header["transfer_frame_version"] == 0
            and header["spacecraft_id"] == spacecraft_id
            and header["padding"] == b"\x00\x00"
        ):
            stats.expected_boundary_frame_headers += 1
        else:
            stats.unexpected_boundary_frame_headers += 1

        stats.xband_wrappers_removed += 1
        stats.xband_header_bytes_removed += TRANSFER_FRAME_DATA_START
        payload = frame[TRANSFER_FRAME_DATA_START:]
        trailer_kind = classify_transfer_frame_trailer(payload)
        _count_trailer(stats, trailer_kind)
        if trailer_kind is not None:
            payload = payload[:-TRANSFER_FRAME_TRAILER_LEN]
            stats.frame_footer_bytes_removed += TRANSFER_FRAME_TRAILER_LEN

        if drop_idle_frames and _is_idle_payload(payload):
            stats.idle_frames_dropped += 1
            stats.idle_frame_bytes_dropped += len(payload)
            continue

        out.extend(payload)
        stats.frame_payload_bytes_kept += len(payload)

    fixed = bytes(out)
    count_internal_sync_like_sequences(
        fixed,
        stats,
        spacecraft_id=spacecraft_id,
    )
    if strip_out_of_phase_xband_artifacts:
        fixed = _strip_out_of_phase_xband_artifacts(
            fixed,
            stats,
            spacecraft_id=spacecraft_id,
        )
    stats.vcdu_word32_words_reversed = len(fixed) // 4
    stats.vcdu_word32_bytes_reversed = stats.vcdu_word32_words_reversed * 4
    fixed = reverse_32bit_words(fixed)
    stats.passthrough_bytes = len(fixed)
    return fixed, stats


def strip_xband_wrappers_and_asms(
    data: bytes,
    *,
    spacecraft_id: int = DEFAULT_SPACECRAFT_ID,
) -> tuple[bytes, TransferFrameStripStats]:
    """
    Remove X-band wrappers and standalone ASMs from the merged byte stream.

    If an ASM is followed by the expected X-band TM primary header and the 2-byte padding
    word, remove all 12 bytes. If an ASM is followed by something else (for example an
    actual CCSDS space-packet primary header), remove only the 4-byte ASM.
    """
    out = bytearray()
    stats = TransferFrameStripStats(mode="global_scan")
    i = 0
    n = len(data)
    while i < n:
        if data.startswith(SYNC_MARKER, i):
            if looks_like_xband_wrapper(data, i, spacecraft_id=spacecraft_id):
                stats.xband_wrappers_removed += 1
                stats.xband_header_bytes_removed += TRANSFER_FRAME_DATA_START
                if out.endswith(b"\x55" * TRANSFER_FRAME_TRAILER_LEN):
                    del out[-TRANSFER_FRAME_TRAILER_LEN:]
                    stats.internal_xband_0x55_trailers_removed += 1
                    stats.internal_xband_trailer_bytes_removed += TRANSFER_FRAME_TRAILER_LEN
                i += TRANSFER_FRAME_DATA_START
            else:
                stats.internal_sync_markers_removed += 1
                stats.internal_sync_bytes_removed += len(SYNC_MARKER)
                i += len(SYNC_MARKER)
            continue

        out.append(data[i])
        stats.passthrough_bytes += 1
        i += 1

    fixed = bytes(out)
    stats.vcdu_word32_words_reversed = len(fixed) // 4
    stats.vcdu_word32_bytes_reversed = stats.vcdu_word32_words_reversed * 4
    fixed = reverse_32bit_words(fixed)
    stats.passthrough_bytes = len(fixed)
    return fixed, stats


def strip_xband_payload_stream(
    data: bytes,
    *,
    spacecraft_id: int = DEFAULT_SPACECRAFT_ID,
    strip_out_of_phase_xband_artifacts: bool = True,
) -> tuple[bytes, TransferFrameStripStats]:
    """
    Strip X-band framing without parsing CCSDS packets.

    Boundary-aligned 2056-byte records use the record-aware path so frame trailers can be
    stripped. Non-aligned data falls back to the older ASM scan used for early test files.
    """
    if is_boundary_aligned_xband_stream(data):
        return strip_xband_frame_records(
            data,
            spacecraft_id=spacecraft_id,
            strip_out_of_phase_xband_artifacts=strip_out_of_phase_xband_artifacts,
        )
    return strip_xband_wrappers_and_asms(data, spacecraft_id=spacecraft_id)


def ccsds_packet_at(
    data: bytes,
    offset: int,
    valid_apids: set[int],
    expected_packet_bytes: dict[int, int] | None = None,
) -> tuple[int, int] | None:
    """Return ``(apid, packet_len)`` if a complete big-endian CCSDS packet starts."""
    if offset + 6 > len(data):
        return None
    header = data[offset : offset + 6]
    first_word = int.from_bytes(header[0:2], "big")
    version = (first_word >> 13) & 0x07
    if version != 0:
        return None
    packet_type = (first_word >> 12) & 0x01
    secondary_header_flag = (first_word >> 11) & 0x01
    if packet_type != 0 or secondary_header_flag != 1:
        return None
    apid = first_word & 0x07FF
    if apid not in valid_apids:
        return None
    packet_len = 6 + int.from_bytes(header[4:6], "big") + 1
    if packet_len < 7:
        return None
    if expected_packet_bytes is not None:
        expected = expected_packet_bytes.get(apid)
        if expected is None or packet_len not in (expected, expected + 1):
            return None
    if offset + packet_len > len(data):
        return apid, -packet_len
    return apid, packet_len


def fletcher32(data: bytes) -> int:
    """Fletcher-32 variant used by the existing Level 0.5 code for non-CSIE_DATA packets."""
    if len(data) % 2:
        data += b"\x00"
    sum1 = 0xFFFF
    sum2 = 0xFFFF
    for i in range(0, len(data), 2):
        word = data[i] | (data[i + 1] << 8)
        sum1 = (sum1 + word) % 0xFFFF
        sum2 = (sum2 + sum1) % 0xFFFF
    return (sum2 << 16) | sum1


def fletcher32_words_be(data: bytes) -> int:
    """Fletcher-32 with big-endian 16-bit words, kept as a diagnostic fallback."""
    if len(data) % 2:
        data += b"\x00"
    sum1 = 0xFFFF
    sum2 = 0xFFFF
    for i in range(0, len(data), 2):
        word = (data[i] << 8) | data[i + 1]
        sum1 = (sum1 + word) % 0xFFFF
        sum2 = (sum2 + sum1) % 0xFFFF
    return (sum2 << 16) | sum1


def csie_data_additive_checksum(row_payload: bytes) -> int:
    """APID 536 checksum: big-endian additive U32 sum over row payload words only."""
    if len(row_payload) % 4:
        row_payload += b"\x00" * (4 - (len(row_payload) % 4))
    total = 0
    for i in range(0, len(row_payload), 4):
        total = (total + int.from_bytes(row_payload[i : i + 4], "big")) & 0xFFFFFFFF
    return total


def validate_packet_checksum(packet: bytes, apid: int) -> str | None:
    """
    Return checksum algorithm name when ``packet`` validates, else None.

    APID 536 (CSIE_DATA) carries a U32 additive checksum over row bytes only. Other
    packet APIDs currently use the repo's Fletcher-32 validator over packet bytes up to
    but excluding the final 4 checksum bytes.
    """
    if len(packet) < 10:
        return None
    stored_be = int.from_bytes(packet[-4:], "big")
    stored_le = int.from_bytes(packet[-4:], "little")
    if apid == CSIE_DATA_APID:
        # Primary (6) + CSIE secondary (6) + row payload + additive checksum (4).
        if len(packet) < 6 + 6 + 4:
            return None
        row_payload = packet[12:-4]
        calculated = csie_data_additive_checksum(row_payload)
        if calculated == stored_be:
            return "csie_additive_be"
        if calculated == stored_le:
            return "csie_additive_le"
        return None
    calculated_le_words = fletcher32(packet[:-4])
    if calculated_le_words == stored_be:
        return "fletcher32_le_words_stored_be"
    if calculated_le_words == stored_le:
        return "fletcher32_le_words_stored_le"

    calculated_be_words = fletcher32_words_be(packet[:-4])
    if calculated_be_words == stored_be:
        return "fletcher32_be_words_stored_be"
    if calculated_be_words == stored_le:
        return "fletcher32_be_words_stored_le"
    return None


def _repaired_packet_candidate(
    data: bytes,
    offset: int,
    target_len: int,
    *,
    max_extra_bytes: int = MAX_REPAIR_EXTRA_BYTES,
) -> tuple[bytes, int, int] | None:
    """
    Build a target-length packet candidate while removing repeated 4-byte words.

    The repair is local and checksum-gated by the caller: it is used only when the raw
    candidate fails checksum validation, and the repaired bytes are emitted only if they
    validate. This avoids deleting legitimate repeated zero words globally.
    """
    out = bytearray()
    i = offset
    limit = min(len(data), offset + target_len + max_extra_bytes)
    removed = 0
    while len(out) < target_len and i < limit:
        if i + 8 <= len(data) and data[i : i + 4] == data[i + 4 : i + 8]:
            out.extend(data[i : i + 4])
            i += 8
            removed += 1
        else:
            out.append(data[i])
            i += 1
    if len(out) != target_len or removed == 0:
        return None
    return bytes(out), i - offset, removed


def _apid_packet_stats(stats: PacketizeStats, apid: int) -> ApidPacketStats:
    return stats.per_apid.setdefault(apid, ApidPacketStats())


def _record_candidate_packet(
    stats: PacketizeStats,
    apid: int,
    packet_len: int,
) -> ApidPacketStats:
    apid_stats = _apid_packet_stats(stats, apid)
    stats.candidate_packets += 1
    stats.candidate_bytes += packet_len
    apid_stats.candidate_packets += 1
    apid_stats.candidate_bytes += packet_len
    return apid_stats


def _record_valid_packet(
    stats: PacketizeStats,
    apid: int,
    packet_len: int,
    algorithm: str,
    *,
    repaired_words: int = 0,
) -> None:
    apid_stats = _apid_packet_stats(stats, apid)
    stats.packets += 1
    stats.checksum_valid_packets += 1
    stats.packet_bytes += packet_len
    stats.apids[apid] += 1
    stats.algorithms[algorithm] += 1
    apid_stats.checksum_valid_packets += 1
    apid_stats.checksum_valid_bytes += packet_len
    apid_stats.algorithms[algorithm] += 1
    if algorithm.startswith("checksum_bypassed"):
        stats.checksum_bypassed_packets += 1
        stats.checksum_bypassed_bytes += packet_len
        apid_stats.checksum_bypassed_packets += 1
        apid_stats.checksum_bypassed_bytes += packet_len
    if apid_stats.min_valid_packet_bytes is None:
        apid_stats.min_valid_packet_bytes = packet_len
    else:
        apid_stats.min_valid_packet_bytes = min(
            apid_stats.min_valid_packet_bytes,
            packet_len,
        )
    apid_stats.max_valid_packet_bytes = max(apid_stats.max_valid_packet_bytes, packet_len)
    if algorithm.startswith("fletcher32"):
        stats.fletcher32_valid_packets += 1
    elif algorithm.startswith("csie_additive"):
        stats.csie_additive_valid_packets += 1
    if repaired_words:
        stats.repaired_packets += 1
        stats.repaired_repeated_words_removed += repaired_words
        apid_stats.repaired_packets += 1
        apid_stats.repaired_repeated_words_removed += repaired_words


def _record_failed_packet(
    stats: PacketizeStats,
    apid: int,
    packet_len: int,
) -> None:
    apid_stats = _apid_packet_stats(stats, apid)
    stats.checksum_failed_candidates += 1
    stats.checksum_failed_candidate_bytes += packet_len
    apid_stats.checksum_failed_packets += 1
    apid_stats.checksum_failed_bytes += packet_len


def _record_incomplete_packet(stats: PacketizeStats, apid: int) -> None:
    apid_stats = _apid_packet_stats(stats, apid)
    stats.incomplete_packet_candidates += 1
    apid_stats.incomplete_packets += 1


def reverse_32bit_words(data: bytes) -> bytes:
    """Reverse byte order within each complete 32-bit VCDU word."""
    out = bytearray()
    limit = len(data) - (len(data) % 4)
    for i in range(0, limit, 4):
        out.extend(data[i : i + 4][::-1])
    out.extend(data[limit:])
    return bytes(out)


def playback_wrapper_packet_at(data: bytes, offset: int) -> tuple[int, int] | None:
    """Return ``(apid, packet_len)`` if a complete APID 68 playback wrapper starts."""
    if offset + PLAYBACK_PRIMARY_HEADER_LEN + PLAYBACK_METADATA_LEN > len(data):
        return None
    first_word = int.from_bytes(data[offset : offset + 2], "big")
    version = (first_word >> 13) & 0x07
    packet_type = (first_word >> 12) & 0x01
    apid = first_word & 0x07FF
    if version != 0 or packet_type != 0 or apid != 68:
        return None
    packet_len = PLAYBACK_PRIMARY_HEADER_LEN + int.from_bytes(
        data[offset + 4 : offset + 6],
        "big",
    ) + 1
    if packet_len < PLAYBACK_PRIMARY_HEADER_LEN + PLAYBACK_METADATA_LEN + 7:
        return None
    if offset + packet_len > len(data):
        return apid, -packet_len
    return apid, packet_len


def extract_playback_inner_packet(
    data: bytes,
    wrapper_at: int,
    wrapper_end: int,
    valid_apids: set[int],
    expected_packet_bytes: dict[int, int] | None,
) -> tuple[int, bytes, int] | None:
    """
    Extract the single inner CCSDS packet carried by an APID 68 playback wrapper.

    After the frame-level FPGA 32-bit endian swap is undone, the playback wrapper is:
      6-byte APID 68 primary header
      10-byte playback metadata
      1 normal big-endian CCSDS packet
    """
    inner_at = wrapper_at + PLAYBACK_PRIMARY_HEADER_LEN + PLAYBACK_METADATA_LEN
    packet_info = ccsds_packet_at(
        data,
        inner_at,
        valid_apids,
        expected_packet_bytes=expected_packet_bytes,
    )
    if packet_info is None:
        return None

    apid, packet_len = packet_info
    if packet_len < 0:
        return None
    if inner_at + packet_len != wrapper_end:
        return None
    return apid, data[inner_at:wrapper_end], inner_at


def _add_packet_record(
    stats: PacketizeStats,
    valid_records: list[tuple[int, bytes]],
    *,
    offset: int,
    apid: int,
    packet: bytes,
    source: str,
    acceptance_mode: str,
    checksum_validated: bool,
    original_primary_header_endian: str,
    primary_header_normalized: bool,
    payload_16bit_words_swapped: bool,
) -> None:
    valid_records.append((offset, packet))
    stats.records.append(
        PacketRecord(
            packet_index=-1,
            source_offset=offset,
            apid=apid,
            packet_len=len(packet),
            source=source,
            acceptance_mode=acceptance_mode,
            checksum_validated=checksum_validated,
            original_primary_header_endian=original_primary_header_endian,
            primary_header_normalized=primary_header_normalized,
            payload_16bit_words_swapped=payload_16bit_words_swapped,
            packet=packet,
        )
    )


def record_playback_inner_packet(
    data: bytes,
    wrapper_at: int,
    wrapper_len: int,
    valid_apids: set[int],
    expected_packet_bytes: dict[int, int] | None,
    stats: PacketizeStats,
    valid_records: list[tuple[int, bytes]],
    *,
    bypass_packet_checksums: bool = True,
) -> bool:
    """Recover and record the inner packet from one structurally valid playback wrapper."""
    wrapper_end = wrapper_at + wrapper_len
    reconstructed = extract_playback_inner_packet(
        data,
        wrapper_at,
        wrapper_end,
        valid_apids,
        expected_packet_bytes,
    )
    if reconstructed is None:
        stats.malformed_playback_wrappers += 1
        return False

    inner_apid, inner_packet, inner_at = reconstructed
    inner_len = len(inner_packet)

    stats.playback_inner_candidates += 1
    _record_candidate_packet(stats, inner_apid, inner_len)
    source = "playback_inner_apid68_vcdu_word32"
    if bypass_packet_checksums:
        algorithm = "checksum_bypassed_playback_inner_apid68_vcdu_word32"
        _record_valid_packet(
            stats,
            inner_apid,
            inner_len,
            algorithm,
        )
        _add_packet_record(
            stats,
            valid_records,
            offset=inner_at,
            apid=inner_apid,
            packet=inner_packet,
            source=source,
            acceptance_mode=algorithm,
            checksum_validated=False,
            original_primary_header_endian="big",
            primary_header_normalized=False,
            payload_16bit_words_swapped=False,
        )
        return True

    algorithm = validate_packet_checksum(inner_packet, inner_apid)
    if algorithm is None:
        _record_failed_packet(stats, inner_apid, inner_len)
        return False

    _record_valid_packet(stats, inner_apid, inner_len, algorithm)
    _add_packet_record(
        stats,
        valid_records,
        offset=inner_at,
        apid=inner_apid,
        packet=inner_packet,
        source=source,
        acceptance_mode=algorithm,
        checksum_validated=True,
        original_primary_header_endian="big",
        primary_header_normalized=False,
        payload_16bit_words_swapped=False,
    )
    return True


def packetize_checksum_valid_ccsds(
    data: bytes,
    valid_apids: set[int],
    expected_packet_bytes: dict[int, int] | None = None,
    *,
    bypass_packet_checksums: bool = True,
    extract_playback_wrappers: bool = True,
) -> tuple[bytes, PacketizeStats]:
    """
    Emit structurally plausible CCSDS packets, dropping filler and non-valid candidates.

    By default this bypasses packet checksum validation because the current flight
    software checksum span/algorithm is still being clarified. With
    ``bypass_packet_checksums=False``, APID 536 uses the additive row checksum and
    non-536 packets use Fletcher-32. In checksum-required mode, the packetizer tries a
    local repeated-word repair for APID 536 candidates and accepts repaired bytes only if
    they validate.
    """
    valid_records: list[tuple[int, bytes]] = []
    stats = PacketizeStats()
    i = 0
    n = len(data)
    while i + 6 <= n:
        if extract_playback_wrappers:
            playback_info = playback_wrapper_packet_at(data, i)
            if playback_info is not None:
                _playback_apid, playback_len = playback_info
                if playback_len > 0:
                    stats.playback_wrappers_seen += 1
                    record_playback_inner_packet(
                        data,
                        i,
                        playback_len,
                        valid_apids,
                        expected_packet_bytes,
                        stats,
                        valid_records,
                        bypass_packet_checksums=bypass_packet_checksums,
                    )
                    i += playback_len
                    continue

                stats.malformed_playback_wrappers += 1
                _record_incomplete_packet(stats, _playback_apid)

        packet_info = ccsds_packet_at(
            data,
            i,
            valid_apids,
            expected_packet_bytes=expected_packet_bytes,
        )
        if packet_info is not None:
            apid, packet_len = packet_info
            if packet_len > 0:
                _record_candidate_packet(stats, apid, packet_len)
                packet = data[i : i + packet_len]
                if bypass_packet_checksums:
                    algorithm = "checksum_bypassed_structural"
                    _record_valid_packet(
                        stats,
                        apid,
                        len(packet),
                        algorithm,
                    )
                    _add_packet_record(
                        stats,
                        valid_records,
                        offset=i,
                        apid=apid,
                        packet=packet,
                        source="direct_big_endian_structural",
                        acceptance_mode=algorithm,
                        checksum_validated=False,
                        original_primary_header_endian="big",
                        primary_header_normalized=False,
                        payload_16bit_words_swapped=False,
                    )
                    i += packet_len
                    continue

                algorithm = validate_packet_checksum(packet, apid)
                consumed = packet_len
                repaired_words = 0
                if algorithm is None and apid == CSIE_DATA_APID:
                    repaired = _repaired_packet_candidate(data, i, packet_len)
                    if repaired is not None:
                        repaired_packet, repaired_consumed, repaired_words = repaired
                        repaired_algorithm = validate_packet_checksum(
                            repaired_packet, apid
                        )
                        if repaired_algorithm is not None:
                            packet = repaired_packet
                            algorithm = repaired_algorithm
                            consumed = repaired_consumed

                if algorithm is not None:
                    _record_valid_packet(
                        stats,
                        apid,
                        len(packet),
                        algorithm,
                        repaired_words=repaired_words,
                    )
                    _add_packet_record(
                        stats,
                        valid_records,
                        offset=i,
                        apid=apid,
                        packet=packet,
                        source="direct_big_endian_structural",
                        acceptance_mode=algorithm,
                        checksum_validated=True,
                        original_primary_header_endian="big",
                        primary_header_normalized=False,
                        payload_16bit_words_swapped=False,
                    )
                    i += consumed
                    continue

                _record_failed_packet(stats, apid, packet_len)
                # Do not jump by packet_len on a checksum failure: this may be a false
                # packet start inside payload/filler, so slide one byte and keep searching.
            else:
                _record_incomplete_packet(stats, apid)

        if data[i] in FILLER_BYTES:
            stats.dropped_filler_bytes += 1
        else:
            stats.dropped_non_filler_bytes += 1
        i += 1

    if i < n:
        tail = data[i:]
        stats.dropped_filler_bytes += sum(1 for b in tail if b in FILLER_BYTES)
        stats.dropped_non_filler_bytes += sum(1 for b in tail if b not in FILLER_BYTES)

    valid_records.sort(key=lambda item: item[0])
    stats.records.sort(key=lambda item: item.source_offset)
    for packet_index, record in enumerate(stats.records):
        record.packet_index = packet_index

    last_valid_end = 0
    for offset, packet in valid_records:
        if offset > last_valid_end:
            stats.resync_gap_count += 1
            stats.max_resync_gap_bytes = max(
                stats.max_resync_gap_bytes,
                offset - last_valid_end,
            )
        last_valid_end = max(last_valid_end, offset + len(packet))

    return b"".join(packet for _offset, packet in valid_records), stats


def build_fixed_binary(
    merged_data: bytes,
    valid_apids: set[int] | None = None,
    expected_packet_bytes: dict[int, int] | None = None,
    *,
    input_mode: str = INPUT_MODE_XBAND,
    spacecraft_id: int = DEFAULT_SPACECRAFT_ID,
    packetize_checksum_valid: bool = False,
    strip_out_of_phase_xband_artifacts: bool = True,
) -> tuple[bytes, FixStats]:
    """Run the first-stage cleaning pipeline and return ``merged_fixed`` bytes."""
    if input_mode == INPUT_MODE_UHF:
        uhf_fixed, uhf_stats = unwrap_uhf_playback_stream(
            merged_data,
            valid_apids=valid_apids,
            expected_packet_bytes=expected_packet_bytes,
        )
        fixed, direct_playback_stats = unwrap_direct_playback_stream(uhf_fixed)
        if uhf_stats.segment_packets_seen:
            tf_stats = TransferFrameStripStats(
                mode="uhf_apid73_playback_reassembly",
                passthrough_bytes=uhf_stats.non_wrapper_bytes_preserved,
            )
        elif direct_playback_stats.wrappers_stripped:
            tf_stats = TransferFrameStripStats(
                mode="uhf_apid72_playback_unwrap",
                passthrough_bytes=len(fixed),
            )
        else:
            tf_stats = TransferFrameStripStats(
                mode="uhf_ccsds_passthrough",
                passthrough_bytes=len(merged_data),
            )
        return fixed, FixStats(
            transfer_frame_strip=tf_stats,
            packetize=None,
            fixed_bytes=len(fixed),
            uhf_playback=uhf_stats,
            direct_playback=direct_playback_stats,
        )

    if input_mode == INPUT_MODE_CCSDS:
        fixed, direct_playback_stats = unwrap_direct_playback_stream(merged_data)
        strip_mode = (
            "hardline_apid72_playback_unwrap"
            if direct_playback_stats.wrappers_stripped
            else "hardline_ccsds_passthrough"
        )
        tf_stats = TransferFrameStripStats(
            mode=strip_mode,
            passthrough_bytes=len(fixed),
        )
        return fixed, FixStats(
            transfer_frame_strip=tf_stats,
            packetize=None,
            fixed_bytes=len(fixed),
            direct_playback=direct_playback_stats,
        )

    payload_stream, tf_stats = strip_xband_payload_stream(
        merged_data,
        spacecraft_id=spacecraft_id,
        strip_out_of_phase_xband_artifacts=strip_out_of_phase_xband_artifacts,
    )
    payload_stream, direct_playback_stats = unwrap_direct_playback_stream(payload_stream)
    if packetize_checksum_valid:
        if valid_apids is None:
            raise ValueError("valid_apids is required when packetization is enabled")
        fixed, packet_stats = packetize_checksum_valid_ccsds(
            payload_stream,
            valid_apids,
            bypass_packet_checksums=False,
        )
    else:
        fixed = payload_stream
        packet_stats = None
    return fixed, FixStats(
        transfer_frame_strip=tf_stats,
        packetize=packet_stats,
        fixed_bytes=len(fixed),
        direct_playback=direct_playback_stats,
    )


def print_input_file_summary(
    paths: list[Path],
    *,
    input_mode: str,
    relative_to: Path | None = None,
) -> None:
    print("Input files:")
    for path in paths:
        size = path.stat().st_size
        frame_note = ""
        if input_mode == INPUT_MODE_XBAND and size % TRANSFER_FRAME_SIZE == 0:
            frame_note = f" ({size // TRANSFER_FRAME_SIZE} x {TRANSFER_FRAME_SIZE}-byte frames)"
        display_name = path.name
        if relative_to is not None:
            try:
                display_name = str(path.relative_to(relative_to))
            except ValueError:
                display_name = str(path)
        print(f"  {display_name}: {size:,} bytes{frame_note}")


def print_fix_summary(stats: FixStats) -> None:
    tf = stats.transfer_frame_strip
    pkt = stats.packetize
    uhf = stats.uhf_playback
    direct = stats.direct_playback
    print("\nFix summary:")
    print(f"  strip mode:                            {tf.mode}")
    if uhf is not None and uhf.segment_packets_seen:
        print(f"  APID 73 segment packets seen:          {uhf.segment_packets_seen:,}")
        print(f"  APID 73 segment packet bytes seen:     {uhf.segment_packet_bytes_seen:,}")
        print(f"  UHF playback groups seen:              {uhf.segment_groups_seen:,}")
        print(f"  unique UHF segments seen:              {uhf.unique_segments_seen:,}")
        print(f"  duplicate segment packets removed:     {uhf.duplicate_segment_packets:,}")
        print(f"  conflicting segment indices:           {uhf.conflicting_segment_indices:,}")
        print(f"  complete APID 72 packets rebuilt:      {uhf.complete_playback_packets:,}")
        print(f"  incomplete APID 72 packets dropped:    {uhf.incomplete_playback_packets:,}")
        print(f"  orphan segment groups dropped:         {uhf.orphan_segment_groups:,}")
        print(f"  APID 73 wrapper bytes removed:         {uhf.segment_wrapper_bytes_removed:,}")
        print(f"  APID 72 header/metadata bytes removed: {uhf.playback_header_bytes_removed:,}")
        print(f"  playback payload bytes emitted:        {uhf.playback_payload_bytes_emitted:,}")
        print(f"  non-wrapper bytes examined:            {uhf.non_wrapper_bytes_seen:,}")
        print(f"  direct CCSDS packets preserved:        {uhf.direct_packets_preserved:,}")
        print(f"  direct CCSDS packet bytes preserved:   {uhf.non_wrapper_bytes_preserved:,}")
        print(f"  non-packet side-channel bytes dropped: {uhf.non_wrapper_bytes_dropped:,}")
        print(f"  merged_fixed.bin bytes:                {stats.fixed_bytes:,}")
        if uhf.duplicate_segment_packets:
            print(
                "WARNING: UHF segment retransmissions were de-duplicated by inner "
                "APID/sequence and segment index."
            )
        for warning in uhf.warnings:
            print(f"WARNING: {warning}")
        return
    if direct is not None and direct.wrappers_stripped:
        print(f"  APID 72 wrapper candidates found:      {direct.candidates_found:,}")
        print(f"  validated APID 72 wrapper chains:      {direct.validated_chains:,}")
        print(f"  APID 72 wrappers stripped:             {direct.wrappers_stripped:,}")
        print(f"  APID 72 header/metadata bytes removed: {direct.wrapper_bytes_removed:,}")
        print(f"  playback payload bytes emitted:        {direct.payload_bytes_emitted:,}")
        print(f"  first APID 72 wrapper offset:          {direct.first_wrapper_offset:,}")
        print(f"  last APID 72 wrapper offset:           {direct.last_wrapper_offset:,}")
        if tf.mode in {
            "hardline_apid72_playback_unwrap",
            "uhf_apid72_playback_unwrap",
        }:
            print(f"  merged_fixed.bin bytes:                {stats.fixed_bytes:,}")
            return
    if tf.mode in {"hardline_ccsds_passthrough", "uhf_ccsds_passthrough"}:
        label = (
            "UHF/Hydra CCSDS input"
            if tf.mode == "uhf_ccsds_passthrough"
            else "hardline realtime CCSDS input"
        )
        print(f"  {label}:         no ASM or transfer-frame wrapper stripped")
        print(f"  bytes passed through unchanged:        {tf.passthrough_bytes:,}")
        print(f"  merged_fixed.bin bytes:                {stats.fixed_bytes:,}")
        return
    if tf.boundary_records_seen:
        print(f"  boundary frame records seen:           {tf.boundary_records_seen:,}")
        print(f"  boundary records with ASM:             {tf.boundary_records_with_sync:,}")
        print(f"  expected frame headers:                {tf.expected_boundary_frame_headers:,}")
        print(f"  unexpected frame headers:              {tf.unexpected_boundary_frame_headers:,}")
    print(f"  X-band 12-byte wrappers removed:       {tf.xband_wrappers_removed:,}")
    print(f"  X-band header bytes removed:           {tf.xband_header_bytes_removed:,}")
    if tf.boundary_records_seen:
        print(f"  frame Fletcher32 BE trailers stripped: {tf.frame_footer_fletcher32_be:,}")
        print(f"  frame Fletcher32 LE trailers stripped: {tf.frame_footer_fletcher32_le:,}")
        print(f"  frame 0x55555555 trailers stripped:    {tf.frame_footer_filler_55:,}")
        print(f"  frame trailers kept as payload:        {tf.frame_footer_unknown_kept:,}")
        print(f"  frame trailer bytes removed:           {tf.frame_footer_bytes_removed:,}")
        print(f"  idle frames dropped:                   {tf.idle_frames_dropped:,}")
        print(f"  idle frame bytes dropped:              {tf.idle_frame_bytes_dropped:,}")
        print(f"  frame payload bytes kept pre-ASM scan: {tf.frame_payload_bytes_kept:,}")
    print(f"  out-of-phase X-band seqs seen:         {tf.internal_xband_like_sequences_seen:,}")
    print(f"  out-of-phase X-band wrappers removed:  {tf.internal_xband_wrappers_removed:,}")
    print(f"  out-of-phase X-band header bytes rmvd: {tf.internal_xband_wrapper_bytes_removed:,}")
    print(f"  out-of-phase 0x55555555 trailers rmvd: {tf.internal_xband_0x55_trailers_removed:,}")
    print(f"  out-of-phase trailer bytes removed:    {tf.internal_xband_trailer_bytes_removed:,}")
    print(f"  out-of-phase wrappers lacking trailer: {tf.internal_xband_wrappers_without_0x55_trailer:,}")
    print(f"  marker-only seqs seen:                 {tf.marker_only_sequences_seen:,}")
    print(f"  marker-only ASMs removed:              {tf.internal_sync_markers_removed:,}")
    print(f"  marker-only ASM bytes removed:         {tf.internal_sync_bytes_removed:,}")
    print(f"  VCDU 32-bit words byte-reversed:       {tf.vcdu_word32_words_reversed:,}")
    print(f"  VCDU payload bytes byte-reversed:      {tf.vcdu_word32_bytes_reversed:,}")
    print(f"  bytes kept after frame/ASM removal:    {tf.passthrough_bytes:,}")
    if tf.internal_xband_like_sequences_seen and tf.internal_xband_wrappers_removed:
        print(
            "WARNING: found and stripped "
            f"{tf.internal_xband_like_sequences_seen:,} out-of-phase X-band frame "
            "header sequence(s) inside the nominal payload stream. This is likely "
            "from a momentary RF communication lapse/resync; downstream packet "
            "checksums should discard any corrupted inner packets."
        )
    elif tf.internal_xband_like_sequences_seen:
        print(
            "WARNING: found "
            f"{tf.internal_xband_like_sequences_seen:,} out-of-phase X-band frame "
            "header sequence(s) inside the nominal payload stream, but preserved "
            "them because diagnostic preservation mode is enabled."
        )
    if tf.frame_footer_checksum_failures:
        print(
            "WARNING: "
            f"{tf.frame_footer_checksum_failures:,} X-band frame checksum "
            "verification(s) failed. This is likely from a momentary RF "
            "communication lapse/resync; downstream packet checksums should "
            "discard any corrupted inner packets."
        )
    if pkt is None:
        print(f"  merged_fixed.bin bytes:                {stats.fixed_bytes:,}")
        return
    checksum_verified = pkt.checksum_valid_packets - pkt.checksum_bypassed_packets
    print(f"  accepted CCSDS packets emitted:        {pkt.checksum_valid_packets:,}")
    print(f"    checksum-verified packets:           {checksum_verified:,}")
    print(f"    checksum-bypassed packets:           {pkt.checksum_bypassed_packets:,}")
    print(f"    Fletcher-32 valid packets:           {pkt.fletcher32_valid_packets:,}")
    print(f"    CSIE additive valid packets:         {pkt.csie_additive_valid_packets:,}")
    print(f"  CCSDS packet bytes emitted:            {pkt.packet_bytes:,}")
    print(f"  repaired packets emitted:              {pkt.repaired_packets:,}")
    print(f"  repeated 4-byte words removed locally: {pkt.repaired_repeated_words_removed:,}")
    print(f"  checksum-failed candidates skipped:    {pkt.checksum_failed_candidates:,}")
    print(f"  dropped filler bytes:                  {pkt.dropped_filler_bytes:,}")
    print(f"  dropped non-filler bytes:              {pkt.dropped_non_filler_bytes:,}")
    print(f"  incomplete packet candidates:          {pkt.incomplete_packet_candidates:,}")
    print(f"  merged_fixed.bin bytes:                {stats.fixed_bytes:,}")
    if pkt.apids:
        top = ", ".join(f"{apid}:{count}" for apid, count in pkt.apids.most_common(16))
        print(f"  top APIDs:                             {top}")


def _avg(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def print_packet_summary(stats: PacketizeStats, apid_names: dict[int, str]) -> None:
    checksum_verified = stats.checksum_valid_packets - stats.checksum_bypassed_packets
    print("\nPacket recovery summary:")
    print(f"  CTDB APID candidates found:            {stats.candidate_packets:,}")
    print(f"  accepted packets emitted:              {stats.checksum_valid_packets:,}")
    print(f"  accepted packet bytes:                 {stats.packet_bytes:,}")
    print(f"  checksum-verified packets emitted:     {checksum_verified:,}")
    print(f"  checksum-bypassed packets emitted:     {stats.checksum_bypassed_packets:,}")
    print(f"  checksum-bypassed packet bytes:        {stats.checksum_bypassed_bytes:,}")
    print(f"  checksum-failed candidates:            {stats.checksum_failed_candidates:,}")
    print(f"  incomplete candidates:                 {stats.incomplete_packet_candidates:,}")
    print(f"  repaired packets emitted:              {stats.repaired_packets:,}")
    print(f"  repeated 4-byte words removed locally: {stats.repaired_repeated_words_removed:,}")
    print(f"  APID 68 playback wrappers seen:        {stats.playback_wrappers_seen:,}")
    print(f"  playback inner packet candidates:      {stats.playback_inner_candidates:,}")
    normalized_count = sum(1 for record in stats.records if record.primary_header_normalized)
    payload_swapped_count = sum(
        1 for record in stats.records if record.payload_16bit_words_swapped
    )
    print(f"  primary headers normalized to BE:      {normalized_count:,}")
    print(f"  payloads 16-bit word byte-swapped:     {payload_swapped_count:,}")
    print(f"  malformed playback wrappers skipped:   {stats.malformed_playback_wrappers:,}")
    print(f"  dropped filler bytes while scanning:   {stats.dropped_filler_bytes:,}")
    print(f"  dropped non-filler bytes while scan:   {stats.dropped_non_filler_bytes:,}")
    print(f"  resync gaps before valid packets:      {stats.resync_gap_count:,}")
    print(f"  largest resync gap:                    {stats.max_resync_gap_bytes:,} bytes")
    if stats.algorithms:
        algorithms = ", ".join(
            f"{name}:{count}" for name, count in stats.algorithms.most_common()
        )
        print(f"  acceptance modes:                      {algorithms}")
    if stats.checksum_bypassed_packets:
        print(
            "WARNING: packet checksum validation is bypassed for this run; "
            "packets_valid.bin contains structurally recovered packets, not "
            "checksum-confirmed packets."
        )
    if normalized_count:
        print(
            "WARNING: found packet headers requiring local normalization; "
            f"normalized {normalized_count:,} header(s) and 16-bit word byte-swapped "
            f"{payload_swapped_count:,} payload(s) for decoder compatibility."
        )

    if not stats.per_apid:
        print("  APID summary:                          no CTDB APID candidates found")
        return

    print("\nAPID summary:")
    print(
        "  APID  Name                           Cand   Accept   Bypass     Fail  Incomp "
        "  AvgB(acc)  MinB  MaxB  Modes"
    )
    rows = sorted(
        stats.per_apid.items(),
        key=lambda item: (
            -item[1].checksum_valid_packets,
            -item[1].checksum_failed_packets,
            item[0],
        ),
    )
    for apid, apid_stats in rows:
        name = apid_names.get(apid, "")
        if len(name) > 28:
            name = name[:25] + "..."
        avg_valid = _avg(
            apid_stats.checksum_valid_bytes,
            apid_stats.checksum_valid_packets,
        )
        min_valid = (
            str(apid_stats.min_valid_packet_bytes)
            if apid_stats.min_valid_packet_bytes is not None
            else "-"
        )
        max_valid = str(apid_stats.max_valid_packet_bytes) if apid_stats.max_valid_packet_bytes else "-"
        algorithms = (
            ",".join(f"{name}:{count}" for name, count in apid_stats.algorithms.items())
            if apid_stats.algorithms
            else "-"
        )
        print(
            f"  {apid:4d}  {name:<28.28} "
            f"{apid_stats.candidate_packets:6d} "
            f"{apid_stats.checksum_valid_packets:8d} "
            f"{apid_stats.checksum_bypassed_packets:8d} "
            f"{apid_stats.checksum_failed_packets:8d} "
            f"{apid_stats.incomplete_packets:7d} "
            f"{avg_valid:11.1f} "
            f"{min_valid:>5} "
            f"{max_valid:>5}  "
            f"{algorithms}"
        )


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _restore_modules(saved: dict[str, object], missing_sentinel: object) -> None:
    for name, module in saved.items():
        if module is missing_sentinel:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def import_bus_decoder_bundle(config: Config) -> DecoderBundle:
    """Import CTDB-generated bus and DSPS packet decoders without editing CTDB files."""
    decoder_path = Path(config.packet_definitions_path).expanduser()
    dsps_dir = decoder_path / "dsps_decoders"
    warnings: list[str] = []
    if not decoder_path.is_dir():
        raise FileNotFoundError(f"Bus decoder folder does not exist: {decoder_path}")

    missing_sentinel = object()
    saved_modules = {
        name: sys.modules.get(name, missing_sentinel)
        for name in ("gen_pkts", "gen_eus", "gen_states")
    }
    saved_path = list(sys.path)
    try:
        # Load DSPS codegen first with its own gen_eus/gen_states bound in module globals.
        dsps_decoders = None
        if dsps_dir.is_dir():
            sys.path.insert(0, str(dsps_dir))
            dsps_modules = []
            for module_name in ("gen_eus", "gen_states", "gen_pkts"):
                path = dsps_dir / f"{module_name}.py"
                if path.is_file():
                    with contextlib.redirect_stdout(io.StringIO()):
                        dsps_modules.append(_load_module_from_path(module_name, path))
            dsps_decoders = types.ModuleType("dsps_decoders")
            for module in dsps_modules:
                for attr, value in module.__dict__.items():
                    if not attr.startswith("_"):
                        setattr(dsps_decoders, attr, value)
        else:
            warnings.append(f"DSPS decoder folder not found: {dsps_dir}")

        # Then load bus codegen with bus gen_eus/gen_states bound in module globals.
        _restore_modules(saved_modules, missing_sentinel)
        sys.path[:] = saved_path
        sys.path.insert(0, str(decoder_path))
        for module_name in ("gen_eus", "gen_states"):
            path = decoder_path / f"{module_name}.py"
            if path.is_file():
                with contextlib.redirect_stdout(io.StringIO()):
                    _load_module_from_path(module_name, path)
        bus_gen_pkts_path = decoder_path / "gen_pkts.py"
        if not bus_gen_pkts_path.is_file():
            raise FileNotFoundError(f"Bus gen_pkts.py not found: {bus_gen_pkts_path}")
        with contextlib.redirect_stdout(io.StringIO()):
            bus_pkts = _load_module_from_path("gen_pkts", bus_gen_pkts_path)

    finally:
        _restore_modules(saved_modules, missing_sentinel)
        sys.path[:] = saved_path

    csie_pkts = None
    csie_gen_pkts_path = Path(config.csie_ctdb_path).expanduser() / "decoders" / "gen_pkts.py"
    if csie_gen_pkts_path.is_file():
        saved_path = list(sys.path)
        saved_csie = sys.modules.get("csie_pkts", missing_sentinel)
        try:
            sys.path.insert(0, str(csie_gen_pkts_path.parent))
            with contextlib.redirect_stdout(io.StringIO()):
                csie_pkts = _load_module_from_path("csie_pkts", csie_gen_pkts_path)
        finally:
            if saved_csie is missing_sentinel:
                sys.modules.pop("csie_pkts", None)
            else:
                sys.modules["csie_pkts"] = saved_csie
            sys.path[:] = saved_path
    else:
        warnings.append(
            "CSIE generated decoder not found; CSIE packets will use raw CTDB CSV "
            f"field decoding when possible: {csie_gen_pkts_path}"
        )

    return DecoderBundle(
        bus_pkts=bus_pkts,
        dsps_decoders=dsps_decoders,
        csie_pkts=csie_pkts,
        warnings=warnings,
    )


def read_ctdb_field_definitions_from_config(config: Config) -> dict[int, list[dict[str, str]]]:
    """Read ordered CTDB field rows, used as a raw CSV decoder fallback."""
    fields_by_apid: dict[int, list[dict[str, str]]] = {}
    for ctdb_path in (config.bus_ctdb_path, config.csie_ctdb_path):
        csv_path = next((p for p in _ct_tlm_candidates(ctdb_path) if p.is_file()), None)
        if csv_path is None:
            continue
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    apid = int(row["APID"])
                except (KeyError, TypeError, ValueError):
                    continue
                fields_by_apid.setdefault(apid, []).append(row)
    return fields_by_apid


def _read_bits_big_endian(data: bytes, bit_offset: int, bit_len: int) -> int:
    value = 0
    for rel_bit in range(bit_len):
        absolute_bit = bit_offset + rel_bit
        byte_index = absolute_bit // 8
        if byte_index >= len(data):
            raise ValueError("field extends beyond packet")
        bit_index = 7 - (absolute_bit % 8)
        value = (value << 1) | ((data[byte_index] >> bit_index) & 0x01)
    return value


def _decode_ctdb_field(data: bytes, bit_offset: int, data_type: str):
    bits = _datatype_bits(data_type)
    if bits is None:
        return None, 0
    kind = data_type.strip()[:1].upper()
    if kind == "F" and bit_offset % 8 == 0 and bits in (32, 64):
        start = bit_offset // 8
        end = start + bits // 8
        if end > len(data):
            raise ValueError("float field extends beyond packet")
        fmt = ">f" if bits == 32 else ">d"
        return struct.unpack(fmt, data[start:end])[0], bits

    value = _read_bits_big_endian(data, bit_offset, bits)
    if kind == "I" and bits > 0 and value & (1 << (bits - 1)):
        value -= 1 << bits
    if kind == "C" and bit_offset % 8 == 0 and bits % 8 == 0:
        start = bit_offset // 8
        end = start + bits // 8
        raw = data[start:end]
        try:
            return raw.decode("ascii").rstrip("\x00"), bits
        except UnicodeDecodeError:
            return raw.hex(), bits
    return value, bits


def generic_ctdb_decode_packet(
    packet: bytes,
    field_rows: list[dict[str, str]],
) -> dict[str, object]:
    decoded: dict[str, object] = {}
    bit_offset = 0
    used_names: Counter = Counter()
    for row in field_rows:
        data_type = row.get("DataType", "")
        value, bits = _decode_ctdb_field(packet, bit_offset, data_type)
        if bits == 0:
            continue
        item_name = row.get("ItemName", "").strip()
        if item_name:
            used_names[item_name] += 1
            out_name = item_name
            if used_names[item_name] > 1:
                out_name = f"{item_name}_{used_names[item_name]}"
            decoded[out_name] = value
        bit_offset += bits
    return decoded


def _select_generated_decoder(
    packet_name: str,
    bundle: DecoderBundle,
) -> tuple[object | None, str]:
    class_name = packet_name.upper()
    if "dsps" in packet_name and bundle.dsps_decoders is not None:
        packet_class = getattr(bundle.dsps_decoders, class_name, None)
        if packet_class is not None:
            return packet_class, "generated_dsps"
        # FIXME(FSW/CTDB): APID 35 is temporarily named dsps_data here because
        # that is the generated constant/state-map name, but v2.0.1 codegen
        # currently exposes the packet class as DSPS_PASS.
        if packet_name == TEMP_DSPS_DATA_PACKET_NAME:
            packet_class = getattr(bundle.dsps_decoders, "DSPS_PASS", None)
            if packet_class is not None:
                return packet_class, "generated_dsps_temp_dsps_data_alias"
    if packet_name.startswith("csie") and bundle.csie_pkts is not None:
        packet_class = getattr(bundle.csie_pkts, class_name, None)
        if packet_class is not None:
            return packet_class, "generated_csie"
    if bundle.bus_pkts is not None:
        packet_class = getattr(bundle.bus_pkts, class_name, None)
        if packet_class is not None:
            return packet_class, "generated_bus"
    return None, ""


def _packet_header_metadata(packet: bytes) -> dict[str, int]:
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


def _csv_safe_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.hex()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple, dict)):
        try:
            return json.dumps(value, default=str)
        except TypeError:
            return str(value)
    return str(value)


def _decoder_object_fields(packet_object: object) -> dict[str, object]:
    fields: dict[str, object] = {}
    for attr_name in dir(packet_object):
        if attr_name.startswith("_"):
            continue
        try:
            value = getattr(packet_object, attr_name)
        except Exception:
            continue
        if callable(value):
            continue
        fields[attr_name] = _csv_safe_value(value)
    return fields


def _safe_csv_stem(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("_") or "unknown"


def _write_csv_rows(path: Path, rows: list[dict[str, object]], preferred_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = list(preferred_fields)
    seen = set(field_names)
    for row in rows:
        for key in row:
            if key not in seen:
                field_names.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _clean_generated_decode_csvs(decoded_dir: Path) -> None:
    """Remove prior generated decode CSVs so absent APIDs cannot leave stale files behind."""
    for pattern in (
        "decoded_apid_*.csv",
        DEFAULT_PACKET_MANIFEST_BASENAME,
        DEFAULT_DECODE_SUMMARY_BASENAME,
    ):
        for path in decoded_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def decode_packet_records_to_csv(
    packet_records: list[PacketRecord],
    config: Config,
    apid_names: dict[int, str],
    decoded_dir: Path,
    *,
    manifest_name: str = DEFAULT_PACKET_MANIFEST_BASENAME,
    summary_name: str = DEFAULT_DECODE_SUMMARY_BASENAME,
    manifest_path_override: Path | None = None,
    summary_path_override: Path | None = None,
) -> DecodeStats:
    decoded_dir.mkdir(parents=True, exist_ok=True)
    _clean_generated_decode_csvs(decoded_dir)
    manifest_path = manifest_path_override or decoded_dir / manifest_name
    summary_path = summary_path_override or decoded_dir / summary_name
    stats = DecodeStats(
        manifest_path=manifest_path,
        summary_path=summary_path,
        decoded_dir=decoded_dir,
    )
    bundle = import_bus_decoder_bundle(config)
    field_definitions = read_ctdb_field_definitions_from_config(config)

    manifest_rows: list[dict[str, object]] = []
    rows_by_apid: dict[int, list[dict[str, object]]] = {}
    decoder_messages_by_apid: Counter = Counter()

    for record in packet_records:
        stats.total_packets += 1
        stats.per_apid[record.apid] += 1
        if record.primary_header_normalized:
            stats.primary_headers_normalized += 1
        if record.payload_16bit_words_swapped:
            stats.payload_16bit_words_swapped += 1
        if record.original_primary_header_endian != "big":
            stats.unexpected_little_endian_headers += 1

        packet_name = apid_names.get(record.apid, f"apid_{record.apid}")
        header_meta = _packet_header_metadata(record.packet)
        base_row: dict[str, object] = {
            "packet_index": record.packet_index,
            "source_offset": record.source_offset,
            "source_id": record.source_id,
            "source_mode": record.source_mode,
            "source_root": record.source_root,
            "input_file_relative_path": record.input_file_relative_path,
            "source_packet_index": record.source_packet_index,
            "source_acceptance_mode": record.source_acceptance_mode,
            "source_provenance_quality": record.source_provenance_quality,
            "source_output_dir": record.source_output_dir,
            "packet_hash": record.packet_hash,
            "duplicate_group_id": record.duplicate_group_id,
            "duplicate_group_size": record.duplicate_group_size,
            "is_duplicate_packet": record.is_duplicate_packet,
            "combined_time_source": record.combined_time_source,
            "combined_time_coarse": record.combined_time_coarse,
            "combined_time_fine": record.combined_time_fine,
            "source": record.source,
            "acceptance_mode": record.acceptance_mode,
            "checksum_validated": record.checksum_validated,
            "original_primary_header_endian": record.original_primary_header_endian,
            "primary_header_normalized": record.primary_header_normalized,
            "payload_16bit_words_swapped": record.payload_16bit_words_swapped,
            "packet_len": record.packet_len,
            "packet_name": packet_name,
            **header_meta,
        }
        row = dict(base_row)
        decoder_kind = ""
        decode_status = "decoded"
        decode_error = ""
        decoder_stdout = ""

        packet_class, decoder_kind = _select_generated_decoder(packet_name, bundle)
        if packet_class is not None:
            try:
                decoder_output = io.StringIO()
                with contextlib.redirect_stdout(decoder_output):
                    packet_object = packet_class(
                        record.packet[6:],
                        bytearray(record.packet[:6]),
                        "packets_valid.bin",
                    )
                decoder_stdout = decoder_output.getvalue().strip().replace("\n", " | ")
                row.update(_decoder_object_fields(packet_object))
                stats.generated_decoder_packets += 1
                stats.generated_per_apid[record.apid] += 1
            except Exception as exc:
                decode_status = "decode_failed"
                decode_error = f"{type(exc).__name__}: {exc}"
        else:
            field_rows = field_definitions.get(record.apid)
            if field_rows:
                try:
                    row.update(generic_ctdb_decode_packet(record.packet, field_rows))
                    decoder_kind = "generic_ctdb_csv"
                    stats.generic_ctdb_packets += 1
                    stats.generic_per_apid[record.apid] += 1
                except Exception as exc:
                    decode_status = "decode_failed"
                    decode_error = f"{type(exc).__name__}: {exc}"
                    decoder_kind = "generic_ctdb_csv"
            else:
                decode_status = "no_decoder"
                decode_error = "no generated decoder or CTDB field rows"
                decoder_kind = "none"

        if decoder_stdout:
            decoder_messages_by_apid[record.apid] += 1
            row["decoder_messages"] = decoder_stdout[:1000]

        row["decoder_kind"] = decoder_kind
        row["decode_status"] = decode_status
        row["decode_error"] = decode_error

        if decode_status == "decoded":
            stats.decoded_packets += 1
            stats.decoded_per_apid[record.apid] += 1
        elif decode_status == "decode_failed":
            stats.decode_failed_packets += 1
            stats.failed_per_apid[record.apid] += 1
        else:
            stats.no_decoder_packets += 1
            stats.no_decoder_per_apid[record.apid] += 1

        rows_by_apid.setdefault(record.apid, []).append(row)
        csv_name = f"decoded_apid_{record.apid:04d}_{_safe_csv_stem(packet_name)}.csv"
        manifest_row = dict(base_row)
        manifest_row.update(
            {
                "decoder_kind": decoder_kind,
                "decode_status": decode_status,
                "decode_error": decode_error,
                "decoded_csv": csv_name,
            }
        )
        manifest_rows.append(manifest_row)

    manifest_fields = [
        "packet_index",
        "source_offset",
        "source_id",
        "source_mode",
        "input_file_relative_path",
        "source_packet_index",
        "source_acceptance_mode",
        "source_provenance_quality",
        "packet_hash",
        "duplicate_group_id",
        "duplicate_group_size",
        "is_duplicate_packet",
        "combined_time_source",
        "combined_time_coarse",
        "combined_time_fine",
        "apid",
        "packet_name",
        "packet_len",
        "sequence_count",
        "source",
        "acceptance_mode",
        "checksum_validated",
        "original_primary_header_endian",
        "primary_header_normalized",
        "payload_16bit_words_swapped",
        "decoder_kind",
        "decode_status",
        "decode_error",
        "decoded_csv",
    ]
    _write_csv_rows(manifest_path, manifest_rows, manifest_fields)
    stats.output_paths.append(manifest_path)

    per_packet_fields = [
        "packet_index",
        "source_offset",
        "source_id",
        "source_mode",
        "input_file_relative_path",
        "source_packet_index",
        "source_acceptance_mode",
        "source_provenance_quality",
        "packet_hash",
        "duplicate_group_id",
        "duplicate_group_size",
        "is_duplicate_packet",
        "combined_time_source",
        "combined_time_coarse",
        "combined_time_fine",
        "apid",
        "packet_name",
        "packet_len",
        "sequence_count",
        "source",
        "acceptance_mode",
        "checksum_validated",
        "original_primary_header_endian",
        "primary_header_normalized",
        "payload_16bit_words_swapped",
        "decoder_kind",
        "decode_status",
        "decode_error",
    ]
    for apid, rows in sorted(rows_by_apid.items()):
        packet_name = apid_names.get(apid, f"apid_{apid}")
        path = decoded_dir / f"decoded_apid_{apid:04d}_{_safe_csv_stem(packet_name)}.csv"
        _write_csv_rows(path, rows, per_packet_fields)
        stats.output_paths.append(path)

    summary_rows: list[dict[str, object]] = []
    for apid in sorted(stats.per_apid):
        packet_count = stats.per_apid[apid]
        packet_name = apid_names.get(apid, f"apid_{apid}")
        rows = rows_by_apid.get(apid, [])
        avg_packet_bytes = _avg(sum(int(row["packet_len"]) for row in rows), packet_count)
        summary_rows.append(
            {
                "apid": apid,
                "packet_name": packet_name,
                "packets": packet_count,
                "decoded": stats.decoded_per_apid[apid],
                "decode_failed": stats.failed_per_apid[apid],
                "no_decoder": stats.no_decoder_per_apid[apid],
                "generated_decoder": stats.generated_per_apid[apid],
                "generic_ctdb_csv": stats.generic_per_apid[apid],
                "primary_headers_normalized": sum(
                    1 for row in rows if row["primary_header_normalized"]
                ),
                "payload_16bit_words_swapped": sum(
                    1 for row in rows if row["payload_16bit_words_swapped"]
                ),
                "decoder_messages": decoder_messages_by_apid[apid],
                "avg_packet_bytes": f"{avg_packet_bytes:.1f}",
                "min_packet_bytes": min((int(row["packet_len"]) for row in rows), default=0),
                "max_packet_bytes": max((int(row["packet_len"]) for row in rows), default=0),
            }
        )
    _write_csv_rows(
        summary_path,
        summary_rows,
        [
            "apid",
            "packet_name",
            "packets",
            "decoded",
            "decode_failed",
            "no_decoder",
            "generated_decoder",
            "generic_ctdb_csv",
            "primary_headers_normalized",
            "payload_16bit_words_swapped",
            "decoder_messages",
            "avg_packet_bytes",
            "min_packet_bytes",
            "max_packet_bytes",
        ],
    )
    stats.output_paths.append(summary_path)

    for warning in bundle.warnings:
        print(f"WARNING: {warning}")
    return stats


def _to_int_or_none(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_int(meta: dict[str, object], *names: str) -> int | None:
    for name in names:
        value = _to_int_or_none(meta.get(name))
        if value is not None:
            return value
    return None


def parse_csie_icm_proc_config(config_value: int) -> tuple[bool, int | None, int, int]:
    """
    Decode CSIE image-processing mode bits.

    Only the encoding-mode bits are used for control flow here. The threshold and
    position fields are retained for diagnostics while the FSW/CTDB contract settles.
    """
    encoding_mode = config_value & 0x03
    pixel_threshold = (config_value >> 4) & 0x03
    position_selection = (config_value >> 6) & 0x03
    if encoding_mode == 0:
        return False, 16, pixel_threshold, position_selection
    if encoding_mode == 1:
        return False, 8, pixel_threshold, position_selection
    if encoding_mode == 2:
        return False, 12, pixel_threshold, position_selection
    if encoding_mode == 3:
        return True, None, pixel_threshold, position_selection
    return False, None, pixel_threshold, position_selection


def decode_csie_meta_packet(
    packet: bytes,
    field_definitions: dict[int, list[dict[str, str]]],
) -> dict[str, object]:
    rows = field_definitions.get(CSIE_META_APID)
    if not rows:
        raise RuntimeError("No CTDB field rows available for APID 538 csie_meta")
    return generic_ctdb_decode_packet(packet, rows)


def csie_meta_dimensions(meta: dict[str, object]) -> tuple[int, int, list[str]]:
    warnings: list[str] = []
    proc_config = _first_int(meta, "csie_meta_fpm_proc_config")
    row_bin_raw = _first_int(meta, "fpm_proc_cfg_row_bin_meta")
    col_bin_raw = _first_int(meta, "fpm_proc_cfg_col_bin_meta")
    roi_rows = _first_int(meta, "csie_meta_fpm_row_per_frame")
    roi_cols = _first_int(meta, "csie_meta_fpm_pix_per_row")
    if roi_rows is None or roi_cols is None:
        raise ValueError(
            "metadata is missing one of csie_meta_fpm_row_per_frame or "
            "csie_meta_fpm_pix_per_row"
        )

    if proc_config is not None:
        row_bin = (proc_config & 0xFF) + 1
        col_bin = ((proc_config >> 8) & 0xFF) + 1
    elif row_bin_raw is not None and col_bin_raw is not None:
        row_bin = row_bin_raw + 1
        col_bin = col_bin_raw + 1
    else:
        row_bin = 1
        col_bin = 1
        warnings.append(
            "metadata is missing FPM row/column binning fields; assuming unbinned rows/columns"
        )

    if row_bin <= 0 or col_bin <= 0:
        raise ValueError(f"invalid row/column binning: row_bin={row_bin}, col_bin={col_bin}")
    if roi_rows % row_bin:
        warnings.append(
            f"roi_rows {roi_rows} is not evenly divisible by row_bin {row_bin}; using floor division"
        )
    if roi_cols % col_bin:
        warnings.append(
            f"roi_cols {roi_cols} is not evenly divisible by col_bin {col_bin}; using floor division"
        )
    rows = roi_rows // row_bin
    cols = roi_cols // col_bin
    if rows <= 0 or cols <= 0:
        raise ValueError(f"invalid derived image dimensions: rows={rows}, cols={cols}")
    return rows, cols, warnings


def csie_meta_processing_mode(meta: dict[str, object]) -> tuple[bool, int | None, list[str]]:
    warnings: list[str] = []
    combined_config = _first_int(meta, "csie_meta_icm_proc_config")
    if combined_config is not None:
        compression_enabled, bit_depth, _threshold, _position = parse_csie_icm_proc_config(
            combined_config
        )
        return compression_enabled, bit_depth, warnings

    encoding = _first_int(meta, "icm_proc_cfg_encoding_meta")
    if encoding is None:
        warnings.append(
            "metadata is missing ICM encoding fields; assuming uncompressed 16-bit rows"
        )
        return False, 16, warnings
    if encoding == 0:
        return False, 16, warnings
    if encoding == 1:
        return False, 8, warnings
    if encoding == 2:
        return False, 12, warnings
    if encoding == 3:
        return True, None, warnings
    warnings.append(f"unrecognized ICM encoding value {encoding}; treating as unknown bit depth")
    return False, None, warnings


def parse_csie_data_row_packet(record: PacketRecord) -> tuple[dict[str, object], object | None]:
    """
    Parse one APID 536 row packet.

    Returns a row summary and a native-endian uint16 row vector. The row is returned even
    when the additive checksum fails so the inventory can show what would be assembled;
    the warning is carried in the summary for later policy changes.
    """
    import numpy as np

    packet = record.packet
    header = _packet_header_metadata(packet)
    data_field_len = header["ccsds_length_field"] + 1
    total_len = 6 + data_field_len
    row: dict[str, object] = {
        "packet_index": record.packet_index,
        "source_offset": record.source_offset,
        "sequence_count": header["sequence_count"],
        "packet_len": len(packet),
        "ccsds_total_length_from_header": total_len,
    }
    if len(packet) < min(total_len, 6 + CSIE_SECONDARY_HEADER_LEN):
        row["status"] = "skipped_short_packet"
        return row, None

    image_id = int.from_bytes(packet[6:10], "big")
    secondary_aux = int.from_bytes(packet[10:12], "big")
    payload_len = data_field_len - CSIE_SECONDARY_HEADER_LEN - CSIE_ROW_CHECKSUM_LEN
    row.update(
        {
            "image_id": image_id,
            "secondary_aux": secondary_aux,
            "row_index": header["sequence_count"],
            "row_payload_bytes": payload_len,
        }
    )
    if payload_len <= 0 or payload_len % 2:
        row["status"] = "skipped_bad_row_length"
        return row, None

    row_start = 6 + CSIE_SECONDARY_HEADER_LEN
    checksum_start = row_start + payload_len
    checksum_end = checksum_start + CSIE_ROW_CHECKSUM_LEN
    if len(packet) < checksum_end:
        row_payload = packet[row_start:min(len(packet), checksum_start)]
        stored_checksum = None
        row["checksum_status"] = "missing"
    else:
        row_payload = packet[row_start:checksum_start]
        stored_checksum = packet[checksum_start:checksum_end]
        calculated = csie_data_additive_checksum(row_payload)
        stored_be = int.from_bytes(stored_checksum, "big")
        stored_le = int.from_bytes(stored_checksum, "little")
        if calculated == stored_be:
            row["checksum_status"] = "valid_be"
        elif calculated == stored_le:
            row["checksum_status"] = "valid_le"
        else:
            row["checksum_status"] = "failed"
        row["stored_checksum_hex"] = stored_checksum.hex()
        row["calculated_checksum_hex"] = f"{calculated:08x}"

    if len(row_payload) != payload_len:
        row["status"] = "skipped_incomplete_row_payload"
        return row, None
    pixels = np.frombuffer(row_payload, dtype=">u2").astype(np.uint16, copy=True)
    row["row_pixels"] = int(pixels.size)
    row["status"] = "parsed"
    return row, pixels


def _csie_preview_rgb_uint8(image_array):
    from matplotlib import cm
    from matplotlib.colors import Normalize
    import numpy as np

    preview = np.rot90(np.asanyarray(image_array), k=1)
    a = np.asarray(preview, dtype=np.float64)
    vmin = float(np.nanmin(a))
    vmax = float(np.nanmax(a))
    if vmin == vmax:
        norm = np.zeros_like(a, dtype=np.float64)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)(a)
    rgb = cm.inferno(norm)[..., :3]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _decode_csie_jpegls_uint16(codestream: bytes):
    """Decode one JPEG-LS codestream as a detached two-dimensional uint16 array."""
    import imagecodecs
    import numpy as np

    decoded = np.asarray(imagecodecs.jpegls_decode(codestream), dtype=np.uint16).copy()
    if decoded.ndim != 2:
        raise ValueError(
            f"expected a two-dimensional CSIE JPEG-LS image, got shape {decoded.shape}"
        )
    return decoded


def assemble_csie_uncompressed_image(
    rows: dict[int, object],
    selected_quality: dict[int, object],
    *,
    expected_rows: int,
    expected_cols: int,
):
    """Build a zero-filled image, excluding rows with failed additive checksums."""
    import numpy as np

    image = np.zeros((expected_rows, expected_cols), dtype=np.uint16)
    for row_index, row_pixels in rows.items():
        row_number = int(row_index)
        if row_number < 1 or row_number > expected_rows:
            continue
        if selected_quality.get(row_index) == "failed":
            continue
        row_array = np.asarray(row_pixels, dtype=np.uint16)
        if row_array.size != expected_cols:
            continue
        image[row_number - 1, :] = row_array
    return image


def _write_csie_fits(
    path: Path,
    image,
    *,
    image_id: int,
    meta: dict[str, object],
    inventory_row: dict[str, object],
) -> None:
    from astropy.io import fits
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(np.asanyarray(image, dtype=np.uint16))
    header = hdu.header
    header["IMAGEID"] = int(image_id)
    header["ROWS"] = int(image.shape[0])
    header["COLS"] = int(image.shape[1])
    header["ROWPKTS"] = int(inventory_row.get("unique_rows", 0) or 0)
    header["MISROWS"] = int(inventory_row.get("missing_rows", 0) or 0)
    header["DUPROWS"] = int(inventory_row.get("duplicate_rows", 0) or 0)
    header["CHKFAIL"] = int(inventory_row.get("checksum_failed_rows", 0) or 0)
    header["CHKMISS"] = int(inventory_row.get("checksum_missing_rows", 0) or 0)
    header["SELVALID"] = int(inventory_row.get("selected_checksum_valid_rows", 0) or 0)
    header["SELFAIL"] = int(inventory_row.get("selected_checksum_failed_rows", 0) or 0)
    header["SELMISS"] = int(inventory_row.get("selected_checksum_missing_rows", 0) or 0)
    header["PARTIAL"] = bool(inventory_row.get("partial_uncompressed_image", False))
    header["ZEROFILL"] = int(inventory_row.get("zero_filled_total_rows", 0) or 0)
    header["MISZERO"] = int(inventory_row.get("zero_filled_missing_rows", 0) or 0)
    header["CHKZERO"] = int(
        inventory_row.get("zero_filled_checksum_failed_rows", 0) or 0
    )
    header["OUTRANGE"] = int(inventory_row.get("out_of_range_rows", 0) or 0)

    used_keys = set(header)
    for attr_name, value in sorted(meta.items()):
        if attr_name.startswith("_") or callable(value):
            continue
        if not isinstance(value, (str, int, float, bool)):
            continue
        name = attr_name
        if name.startswith("csie_"):
            name = name[5:]
        if name.startswith("meta_"):
            name = name[5:]
        base_key = re.sub(r"[^A-Za-z0-9]", "", name).upper()[:8] or "CSIE"
        fits_key = base_key
        if fits_key in used_keys:
            for suffix in range(1, 10):
                candidate = f"{base_key[:7]}{suffix}"
                if candidate not in used_keys:
                    fits_key = candidate
                    break
            else:
                continue
        try:
            header[fits_key] = value
        except Exception:
            continue
        used_keys.add(fits_key)

    hdu.writeto(path, overwrite=True)


def _write_csie_jpeg2000(path: Path, image) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _csie_preview_rgb_uint8(image)
    Image.fromarray(rgb, mode="RGB").save(path, format="JPEG2000")


def _write_csie_png(path: Path, image) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _csie_preview_rgb_uint8(image)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG")


def _write_csie_meta_json(path: Path, image_id: int, meta: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"image_id": image_id, **{k: _csv_safe_value(v) for k, v in meta.items()}}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def _trim_jpegls_at_eoi(data: bytes) -> tuple[bytes, bool, int]:
    """Trim bytes after the first JPEG/JPEG-LS EOI marker."""
    eoi_at = data.find(JPEG_LS_EOI)
    if eoi_at < 0:
        return data, False, 0
    end = eoi_at + len(JPEG_LS_EOI)
    return data[:end], True, len(data) - end


def format_product_suffix(suffix: str | None) -> str:
    """Return a filename suffix with a leading hyphen, or empty string."""
    if suffix is None:
        return ""
    suffix = suffix.strip()
    if not suffix:
        return ""
    if suffix.startswith(("-", "_")):
        return suffix
    return f"-{suffix}"


def _make_csie_stream_record(
    *,
    packet_index: int,
    source_offset: int,
    apid: int,
    packet: bytes,
    source: str,
) -> PacketRecord:
    return PacketRecord(
        packet_index=packet_index,
        source_offset=source_offset,
        apid=apid,
        packet_len=len(packet),
        source=source,
        acceptance_mode="csie_full_stream_structural",
        checksum_validated=False,
        original_primary_header_endian="big",
        primary_header_normalized=False,
        payload_16bit_words_swapped=False,
        packet=packet,
    )


def _scan_csie_meta_records_in_stream(
    stream: bytes,
    *,
    source: str,
    expected_meta_len: int | None,
) -> list[PacketRecord]:
    records: list[PacketRecord] = []
    offset = 0
    while True:
        offset = stream.find(b"\x0a\x1a", offset)
        if offset < 0:
            break
        if offset + 6 <= len(stream):
            packet_len = int.from_bytes(stream[offset + 4 : offset + 6], "big") + 7
            packet = stream[offset : offset + packet_len]
            if offset + packet_len <= len(stream):
                header = _packet_header_metadata(packet)
                if (
                    header["ccsds_version"] == 0
                    and header["ccsds_type"] == 0
                    and header["ccsds_secondary_header_flag"] == 1
                    and header["apid"] == CSIE_META_APID
                    and (expected_meta_len is None or packet_len == expected_meta_len)
                ):
                    records.append(
                        _make_csie_stream_record(
                            packet_index=len(records),
                            source_offset=offset,
                            apid=CSIE_META_APID,
                            packet=packet,
                            source=source,
                        )
                    )
        offset += 1
    return records


def _metadata_expectations_from_records(
    records: list[PacketRecord],
    field_definitions: dict[int, list[dict[str, str]]],
) -> tuple[set[int], dict[int, int], list[str]]:
    image_ids: set[int] = set()
    expected_cols_by_image: dict[int, int] = {}
    warnings: list[str] = []
    for record in records:
        try:
            meta = decode_csie_meta_packet(record.packet, field_definitions)
        except Exception as exc:
            warnings.append(
                f"APID 538 source_offset={record.source_offset} could not be decoded "
                f"during CSIE stream scan: {type(exc).__name__}: {exc}"
            )
            continue
        image_id = _to_int_or_none(meta.get("csie_meta_img_id"))
        if image_id is None:
            warnings.append(
                f"APID 538 source_offset={record.source_offset} decoded without "
                "csie_meta_img_id during CSIE stream scan"
            )
            continue
        image_ids.add(image_id)
        try:
            _rows, cols, dimension_warnings = csie_meta_dimensions(meta)
        except Exception as exc:
            warnings.append(
                f"APID 538 image_id={image_id} metadata dimensions could not be decoded "
                f"during CSIE stream scan: {type(exc).__name__}: {exc}"
            )
            continue
        expected_cols_by_image[image_id] = cols
        for warning in dimension_warnings:
            warnings.append(f"APID 538 image_id={image_id}: {warning}")
    return image_ids, expected_cols_by_image, warnings


def _filter_plausible_csie_meta_records(
    records: list[PacketRecord],
    field_definitions: dict[int, list[dict[str, str]]],
) -> tuple[list[PacketRecord], list[str]]:
    """Reject APID-shaped payload coincidences with impossible detector dimensions."""
    accepted: list[PacketRecord] = []
    warnings: list[str] = []
    rejected_dimensions = 0
    rejected_decode = 0
    for record in records:
        try:
            meta = decode_csie_meta_packet(record.packet, field_definitions)
        except Exception:
            rejected_decode += 1
            continue
        roi_rows = _first_int(meta, "csie_meta_fpm_row_per_frame")
        roi_cols = _first_int(meta, "csie_meta_fpm_pix_per_row")
        if (
            roi_rows is None
            or roi_cols is None
            or not 1 <= roi_rows <= CSIE_MAX_SENSOR_ROWS
            or not 1 <= roi_cols <= CSIE_MAX_SENSOR_COLS
        ):
            rejected_dimensions += 1
            continue
        accepted.append(record)

    if rejected_dimensions:
        warnings.append(
            f"Discarded {rejected_dimensions} APID 538-shaped stream candidate(s) "
            "whose decoded detector dimensions were outside the physical "
            f"{CSIE_MAX_SENSOR_ROWS}x{CSIE_MAX_SENSOR_COLS} sensor bounds."
        )
    if rejected_decode:
        warnings.append(
            f"Discarded {rejected_decode} APID 538-shaped stream candidate(s) that "
            "could not be decoded by the configured CSIE CTDB."
        )
    return accepted, warnings


def _scan_csie_data_records_in_stream(
    stream: bytes,
    *,
    source: str,
    known_image_ids: set[int],
    expected_cols_by_image: dict[int, int],
) -> tuple[list[PacketRecord], list[str]]:
    records: list[PacketRecord] = []
    warnings: list[str] = []
    nonstandard_primary_flags = Counter()
    if not known_image_ids:
        return records, warnings

    for first_word in (0x0A18, 0x1218):
        pattern = first_word.to_bytes(2, "big")
        offset = 0
        while True:
            offset = stream.find(pattern, offset)
            if offset < 0:
                break
            if offset + 12 <= len(stream):
                packet_len = int.from_bytes(stream[offset + 4 : offset + 6], "big") + 7
                packet_end = offset + packet_len
                image_id = int.from_bytes(stream[offset + 6 : offset + 10], "big")
                row_payload_len = packet_len - 6 - CSIE_SECONDARY_HEADER_LEN - CSIE_ROW_CHECKSUM_LEN
                expected_cols = expected_cols_by_image.get(image_id)
                header = _packet_header_metadata(stream[offset : offset + 6])
                if (
                    packet_end <= len(stream)
                    and header["ccsds_version"] == 0
                    and header["apid"] == CSIE_DATA_APID
                    and packet_len >= 6 + CSIE_SECONDARY_HEADER_LEN + CSIE_ROW_CHECKSUM_LEN
                    and image_id in known_image_ids
                    and row_payload_len > 0
                    and row_payload_len % 2 == 0
                    and (expected_cols is None or row_payload_len == expected_cols * 2)
                ):
                    if (
                        header["ccsds_type"] != 0
                        or header["ccsds_secondary_header_flag"] != 1
                    ):
                        nonstandard_primary_flags[
                            (
                                header["ccsds_type"],
                                header["ccsds_secondary_header_flag"],
                                first_word,
                            )
                        ] += 1
                    records.append(
                        _make_csie_stream_record(
                            packet_index=len(records),
                            source_offset=offset,
                            apid=CSIE_DATA_APID,
                            packet=stream[offset:packet_end],
                            source=source,
                        )
                    )
            offset += 1

    if nonstandard_primary_flags:
        detail = ", ".join(
            f"type={packet_type} sec_hdr={sec_hdr} first_word=0x{word:04x}: {count}"
            for (packet_type, sec_hdr, word), count in sorted(nonstandard_primary_flags.items())
        )
        warnings.append(
            "APID 536 CSIE data rows were accepted with non-standard primary header "
            f"flags while scanning the full stream ({detail})."
        )
    records.sort(key=lambda item: item.source_offset)
    for packet_index, record in enumerate(records):
        record.packet_index = packet_index
    return records, warnings


def scan_csie_records_from_fixed_stream(
    fixed_payload_stream: bytes,
    field_definitions: dict[int, list[dict[str, str]]],
    expected_packet_bytes: dict[int, int],
) -> tuple[list[PacketRecord], str, list[str]]:
    """
    Recover CSIE metadata/row packets from the full fixed payload stream.

    HK/ADCS packets currently decode from the VCDU 32-bit-reversed ``merged_fixed``
    stream. CSIE image rows in this test file appear in the opposite byte-order view,
    so scan both views and choose the one with the most CSIE structure.
    """
    expected_meta_len = expected_packet_bytes.get(CSIE_META_APID)
    scan_results: list[tuple[int, list[PacketRecord], str, list[str]]] = []
    for source, stream in (
        ("merged_fixed_full_stream", fixed_payload_stream),
        ("merged_fixed_pre_vcdu_word32_full_stream", reverse_32bit_words(fixed_payload_stream)),
    ):
        warnings: list[str] = []
        meta_records = _scan_csie_meta_records_in_stream(
            stream,
            source=source,
            expected_meta_len=expected_meta_len,
        )
        meta_records, meta_filter_warnings = _filter_plausible_csie_meta_records(
            meta_records,
            field_definitions,
        )
        warnings.extend(meta_filter_warnings)
        known_image_ids, expected_cols_by_image, expectation_warnings = (
            _metadata_expectations_from_records(meta_records, field_definitions)
        )
        warnings.extend(expectation_warnings)
        data_records, data_warnings = _scan_csie_data_records_in_stream(
            stream,
            source=source,
            known_image_ids=known_image_ids,
            expected_cols_by_image=expected_cols_by_image,
        )
        warnings.extend(data_warnings)
        records = sorted(meta_records + data_records, key=lambda item: item.source_offset)
        for packet_index, record in enumerate(records):
            record.packet_index = packet_index
        score = len(meta_records) * 10 + len(data_records)
        scan_results.append((score, records, source, warnings))

    _score, records, source, warnings = max(scan_results, key=lambda item: item[0])
    if source == "merged_fixed_pre_vcdu_word32_full_stream" and records:
        warnings.insert(
            0,
            "CSIE image extraction used reverse_32bit_words(merged_fixed.bin) for the "
            "full-stream scan. HK/ADCS decoding still uses merged_fixed.bin directly; "
            "this CSIE byte-order exception is a temporary FSW/FPGA contract item to revisit.",
        )
    if records:
        warnings.append(
            "CSIE image extraction scans merged_fixed.bin directly, not packets_valid.bin."
        )
        warnings.append(
            "APID 536 secondary header bytes are interpreted as CSIE image_id/filler "
            "for image assembly, even if the current CTDB labels those fields as SHCOARSE/SHFINE."
        )
    return records, source, warnings


def _csie_row_quality_rank(checksum_status: object) -> int:
    if checksum_status in ("valid_be", "valid_le"):
        return 3
    if checksum_status == "missing":
        return 2
    if checksum_status == "failed":
        return 1
    return 0


def write_csie_image_products(
    packet_records: list[PacketRecord],
    config: Config,
    output_dir: Path,
    *,
    fixed_payload_stream: bytes | None = None,
    output_suffix: str = "",
    write_jpeg2000: bool = True,
    write_png: bool = True,
    write_meta_json: bool | None = None,
    inventory_name: str = DEFAULT_CSIE_INVENTORY_BASENAME,
) -> CsieImageStats:
    """
    Inventory and assemble CSIE images from the full fixed stream.

    Recovered JPEG-LS streams are preserved as ``.jls`` and decoded to native uint16
    FITS plus rotated inferno PNG/JP2 previews. Uncompressed rows use the same product
    writers. Every assumption or skipped condition is recorded in the inventory.
    """
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / inventory_name
    stats = CsieImageStats(output_dir=output_dir, inventory_path=inventory_path)
    field_definitions = read_ctdb_field_definitions_from_config(config)
    expected_packet_bytes = read_expected_packet_bytes_from_config(config)
    if write_meta_json is None:
        write_meta_json = bool(getattr(config, "also_save_csie_meta_json", False))

    packet_csie_records = [
        record for record in packet_records if record.apid in (CSIE_DATA_APID, CSIE_META_APID)
    ]
    stream_csie_records: list[PacketRecord] = []
    stream_source = ""
    if fixed_payload_stream is not None:
        stream_csie_records, stream_source, stream_warnings = scan_csie_records_from_fixed_stream(
            fixed_payload_stream,
            field_definitions,
            expected_packet_bytes,
        )
        stats.warnings.extend(stream_warnings)

    csie_records = packet_csie_records
    stats.source = "packets_valid_records"
    if len(stream_csie_records) > len(packet_csie_records):
        csie_records = stream_csie_records
        stats.source = stream_source
    elif packet_csie_records and stream_csie_records:
        stats.warnings.append(
            "CSIE packets were found in both packets_valid records and the full fixed stream; "
            "using packets_valid records because they are at least as complete."
        )

    metadata_by_image: dict[int, dict[str, object]] = {}
    metadata_counts: Counter = Counter()
    rows_by_image: dict[int, dict[int, object]] = {}
    row_quality_by_image: dict[int, dict[int, object]] = {}
    row_summaries_by_image: dict[int, list[dict[str, object]]] = {}
    row_lengths_by_image: dict[int, Counter] = {}
    compressed_chunks_by_image: dict[int, list[dict[str, object]]] = {}

    for record in csie_records:
        if record.apid == CSIE_META_APID:
            stats.meta_packets += 1
            try:
                meta = decode_csie_meta_packet(record.packet, field_definitions)
            except Exception as exc:
                stats.warnings.append(
                    f"APID 538 packet_index={record.packet_index} could not be decoded: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            image_id = _to_int_or_none(meta.get("csie_meta_img_id"))
            if image_id is None:
                stats.warnings.append(
                    f"APID 538 packet_index={record.packet_index} decoded without csie_meta_img_id"
                )
                continue
            metadata_counts[image_id] += 1
            metadata_by_image[image_id] = meta
            continue

        if record.apid != CSIE_DATA_APID:
            continue

        stats.data_packets += 1
        row_summary, pixels = parse_csie_data_row_packet(record)
        image_id = _to_int_or_none(row_summary.get("image_id"))
        if image_id is None:
            stats.skipped_rows += 1
            stats.warnings.append(
                f"APID 536 packet_index={record.packet_index} skipped before image_id parse: "
                f"{row_summary.get('status')}"
            )
            continue
        row_summaries_by_image.setdefault(image_id, []).append(row_summary)
        if row_summary.get("checksum_status") in ("valid_be", "valid_le"):
            stats.checksum_valid_rows += 1
        elif row_summary.get("checksum_status") == "failed":
            stats.checksum_failed_rows += 1
        elif row_summary.get("checksum_status") == "missing":
            stats.checksum_missing_rows += 1
        payload_len = _to_int_or_none(row_summary.get("row_payload_bytes"))
        if payload_len is not None and row_summary.get("status") == "parsed":
            payload_start = 6 + CSIE_SECONDARY_HEADER_LEN
            payload_end = payload_start + payload_len
            compressed_chunks_by_image.setdefault(image_id, []).append(
                {
                    "source_offset": record.source_offset,
                    "sequence_count": row_summary.get("row_index"),
                    "checksum_status": row_summary.get("checksum_status"),
                    "payload": record.packet[payload_start:payload_end],
                }
            )
        if pixels is None:
            stats.skipped_rows += 1
            continue
        row_index = _to_int_or_none(row_summary.get("row_index"))
        if row_index is None:
            stats.skipped_rows += 1
            row_summary["status"] = "skipped_bad_row_index"
            continue
        image_rows = rows_by_image.setdefault(image_id, {})
        image_row_quality = row_quality_by_image.setdefault(image_id, {})
        if row_index in image_rows:
            stats.duplicate_rows += 1
            old_rank = _csie_row_quality_rank(image_row_quality.get(row_index))
            new_rank = _csie_row_quality_rank(row_summary.get("checksum_status"))
            if new_rank < old_rank:
                continue
        image_rows[row_index] = pixels
        image_row_quality[row_index] = row_summary.get("checksum_status")
        row_lengths_by_image.setdefault(image_id, Counter())[int(len(pixels))] += 1

    all_image_ids = sorted(set(metadata_by_image) | set(rows_by_image) | set(row_summaries_by_image))
    stats.image_ids_seen = len(all_image_ids)
    inventory_rows: list[dict[str, object]] = []

    for image_id in all_image_ids:
        meta = metadata_by_image.get(image_id)
        rows = rows_by_image.get(image_id, {})
        row_summaries = row_summaries_by_image.get(image_id, [])
        row_lengths = row_lengths_by_image.get(image_id, Counter())
        selected_quality = row_quality_by_image.get(image_id, {})
        compressed_chunks = compressed_chunks_by_image.get(image_id, [])
        warnings: list[str] = []
        if metadata_counts[image_id] > 1:
            warnings.append(f"duplicate metadata packets seen: {metadata_counts[image_id]}; last one used")

        expected_rows: int | None = None
        expected_cols: int | None = None
        compression_enabled = False
        bit_depth: int | None = None
        if meta is None:
            warnings.append("metadata_missing; image dimensions unknown; products not written")
        else:
            compression_enabled, bit_depth, processing_warnings = csie_meta_processing_mode(meta)
            warnings.extend(processing_warnings)
            try:
                expected_rows, expected_cols, dimension_warnings = csie_meta_dimensions(meta)
                warnings.extend(dimension_warnings)
            except Exception as exc:
                warnings.append(f"metadata_dimension_decode_failed: {type(exc).__name__}: {exc}")

        unique_rows = len(rows)
        valid_row_indices = [
            row_index
            for row_index in rows
            if compression_enabled
            or expected_rows is None
            or 1 <= int(row_index) <= expected_rows
        ]
        out_of_range_rows = unique_rows - len(valid_row_indices)
        if out_of_range_rows:
            warnings.append(f"{out_of_range_rows} row index/indices outside metadata range")

        row_length_min = min(row_lengths) if row_lengths else None
        row_length_max = max(row_lengths) if row_lengths else None
        row_length_mode = row_lengths.most_common(1)[0][0] if row_lengths else None
        bad_length_rows = 0
        if expected_cols is not None and not compression_enabled:
            bad_length_rows = sum(
                count for row_len, count in row_lengths.items() if int(row_len) != expected_cols
            )
            if bad_length_rows:
                warnings.append(
                    f"{bad_length_rows} row(s) do not match metadata columns={expected_cols}"
                )

        parsed_row_indices = {int(i) for i in rows}
        missing_rows = None
        if expected_rows is not None and not compression_enabled:
            missing_rows = len(set(range(1, expected_rows + 1)) - parsed_row_indices)
            if missing_rows:
                warnings.append(f"{missing_rows} expected row(s) missing")

        checksum_failed = sum(
            1 for row in row_summaries if row.get("checksum_status") == "failed"
        )
        checksum_missing = sum(
            1 for row in row_summaries if row.get("checksum_status") == "missing"
        )
        checksum_valid = sum(
            1 for row in row_summaries if row.get("checksum_status") in ("valid_be", "valid_le")
        )
        selected_checksum_valid = sum(
            1 for status in selected_quality.values() if status in ("valid_be", "valid_le")
        )
        selected_checksum_failed = sum(
            1 for status in selected_quality.values() if status == "failed"
        )
        selected_checksum_missing = sum(
            1 for status in selected_quality.values() if status == "missing"
        )
        skipped_rows = sum(
            1 for row in row_summaries if row.get("status") != "parsed"
        )
        duplicate_rows = max(0, len(row_summaries) - unique_rows - skipped_rows)
        if checksum_failed:
            warnings.append(
                f"{checksum_failed} CSIE row additive checksum failure(s); row data still inventoried"
            )
        if checksum_missing:
            warnings.append(f"{checksum_missing} CSIE row checksum(s) missing")
        if duplicate_rows:
            warnings.append(
                f"{duplicate_rows} duplicate row packet(s); best checksum-ranked packet used"
            )
        if selected_checksum_failed:
            if compression_enabled:
                warnings.append(
                    f"{selected_checksum_failed} selected compressed chunk(s) still have "
                    "additive checksum failures"
                )
            else:
                warnings.append(
                    f"{selected_checksum_failed} selected row(s) have additive checksum "
                    "failures and will be zero-filled in uncompressed products"
                )
        if selected_checksum_missing:
            warnings.append(
                f"{selected_checksum_missing} selected row(s) still have missing additive checksums"
            )
        if compression_enabled:
            warnings.append(
                "for JPEG-LS, APID 536 sequence counts are treated as compressed chunk "
                "indices for de-duplication, not image row numbers"
            )

        can_write_uncompressed = (
            meta is not None
            and not compression_enabled
            and expected_rows is not None
            and expected_cols is not None
            and bad_length_rows == 0
            and unique_rows > 0
        )
        complete = (
            can_write_uncompressed
            and missing_rows == 0
            and out_of_range_rows == 0
            and selected_checksum_failed == 0
        )
        partial_uncompressed = can_write_uncompressed and not complete
        zero_filled_missing_rows = missing_rows if partial_uncompressed and missing_rows else 0
        zero_filled_checksum_failed_rows = (
            selected_checksum_failed if can_write_uncompressed else 0
        )
        zero_filled_total_rows = (
            (zero_filled_missing_rows or 0) + zero_filled_checksum_failed_rows
        )
        if partial_uncompressed:
            stats.images_partial += 1
            if zero_filled_missing_rows:
                warnings.append(
                    f"partial uncompressed image written with {zero_filled_missing_rows} "
                    "missing row(s) left as zeros"
                )
            if out_of_range_rows:
                warnings.append(
                    f"partial uncompressed image written while ignoring {out_of_range_rows} "
                    "out-of-range row index/indices"
                )
            if zero_filled_checksum_failed_rows:
                warnings.append(
                    f"partial uncompressed image written with "
                    f"{zero_filled_checksum_failed_rows} checksum-failed row(s) left as zeros"
                )
        fits_path = ""
        jp2_path = ""
        png_path = ""
        jpegls_path = ""
        meta_json_path = ""
        jpegls_chunks = ""
        jpegls_selected_chunks = ""
        jpegls_bytes = ""
        jpegls_eoi_found = ""
        jpegls_bytes_trimmed_after_eoi = ""
        jpegls_decoded: bool | str = ""
        jpegls_decode_error = ""
        suffix = format_product_suffix(output_suffix)

        if compression_enabled and meta is not None and compressed_chunks:
            selected_chunks_by_seq: dict[int, dict[str, object]] = {}
            for chunk in compressed_chunks:
                sequence_count = _to_int_or_none(chunk.get("sequence_count"))
                if sequence_count is None:
                    continue
                old_chunk = selected_chunks_by_seq.get(sequence_count)
                if old_chunk is not None:
                    old_rank = _csie_row_quality_rank(old_chunk.get("checksum_status"))
                    new_rank = _csie_row_quality_rank(chunk.get("checksum_status"))
                    if new_rank < old_rank:
                        continue
                selected_chunks_by_seq[sequence_count] = chunk

            selected_chunks = [
                selected_chunks_by_seq[sequence_count]
                for sequence_count in sorted(selected_chunks_by_seq)
            ]
            codestream = b"".join(
                bytes(chunk.get("payload", b"")) for chunk in selected_chunks
            )
            codestream, eoi_found, bytes_trimmed = _trim_jpegls_at_eoi(codestream)
            jls_file = output_dir / f"image_{image_id}{suffix}.jls"
            jls_file.write_bytes(codestream)
            jpegls_path = str(jls_file)
            jpegls_chunks = len(compressed_chunks)
            jpegls_selected_chunks = len(selected_chunks)
            jpegls_bytes = len(codestream)
            jpegls_eoi_found = eoi_found
            jpegls_bytes_trimmed_after_eoi = bytes_trimmed
            stats.jpegls_written += 1
            stats.compressed_images_written += 1
            stats.images_written += 1
            stats.output_paths.append(jls_file)
            if eoi_found and bytes_trimmed:
                warnings.append(
                    f"trimmed {bytes_trimmed} byte(s) after first JPEG-LS EOI marker"
                )
            elif not eoi_found:
                warnings.append(
                    "JPEG-LS EOI marker 0xffd9 was not found; wrote recovered partial stream"
                )

            try:
                image = _decode_csie_jpegls_uint16(codestream)
            except Exception as exc:
                jpegls_decoded = False
                jpegls_decode_error = f"{type(exc).__name__}: {exc}"
                stats.compressed_decode_failures += 1
                warnings.append(f"JPEG-LS decode failed: {jpegls_decode_error}")
            else:
                jpegls_decoded = True
                stats.compressed_images_decoded += 1
                expected_shape = (
                    (expected_rows, expected_cols)
                    if expected_rows is not None and expected_cols is not None
                    else None
                )
                if expected_shape is not None and image.shape != expected_shape:
                    warnings.append(
                        f"JPEG-LS decoded shape {image.shape} does not match metadata "
                        f"shape {expected_shape}; codestream dimensions were preserved"
                    )

                fits_file = output_dir / f"image_{image_id}{suffix}.fits"
                jp2_file = output_dir / f"image_{image_id}{suffix}.jp2"
                png_file = output_dir / f"image_{image_id}{suffix}.png"
                _write_csie_fits(
                    fits_file,
                    image,
                    image_id=image_id,
                    meta=meta,
                    inventory_row={
                        "unique_rows": len(selected_chunks),
                        "missing_rows": 0,
                        "duplicate_rows": duplicate_rows,
                        "checksum_failed_rows": checksum_failed,
                        "checksum_missing_rows": checksum_missing,
                        "selected_checksum_valid_rows": selected_checksum_valid,
                        "selected_checksum_failed_rows": selected_checksum_failed,
                        "selected_checksum_missing_rows": selected_checksum_missing,
                        "partial_uncompressed_image": False,
                        "zero_filled_missing_rows": 0,
                        "zero_filled_checksum_failed_rows": 0,
                        "zero_filled_total_rows": 0,
                        "out_of_range_rows": 0,
                    },
                )
                fits_path = str(fits_file)
                stats.fits_written += 1
                stats.output_paths.append(fits_file)
                if write_png:
                    _write_csie_png(png_file, image)
                    png_path = str(png_file)
                    stats.png_written += 1
                    stats.output_paths.append(png_file)
                if write_jpeg2000:
                    _write_csie_jpeg2000(jp2_file, image)
                    jp2_path = str(jp2_file)
                    stats.jp2_written += 1
                    stats.output_paths.append(jp2_file)
            if write_meta_json:
                meta_json_file = output_dir / f"image_{image_id}{suffix}_meta.json"
                _write_csie_meta_json(meta_json_file, image_id, meta)
                meta_json_path = str(meta_json_file)
                stats.output_paths.append(meta_json_file)

        if can_write_uncompressed:
            if complete:
                stats.images_complete += 1
            image = assemble_csie_uncompressed_image(
                rows,
                selected_quality,
                expected_rows=expected_rows,
                expected_cols=expected_cols,
            )
            fits_file = output_dir / f"image_{image_id}{suffix}.fits"
            jp2_file = output_dir / f"image_{image_id}{suffix}.jp2"
            png_file = output_dir / f"image_{image_id}{suffix}.png"
            _write_csie_fits(
                fits_file,
                image,
                image_id=image_id,
                meta=meta,
                inventory_row={
                    "unique_rows": unique_rows,
                    "missing_rows": missing_rows or 0,
                    "duplicate_rows": duplicate_rows,
                    "checksum_failed_rows": checksum_failed,
                    "checksum_missing_rows": checksum_missing,
                    "selected_checksum_valid_rows": selected_checksum_valid,
                    "selected_checksum_failed_rows": selected_checksum_failed,
                    "selected_checksum_missing_rows": selected_checksum_missing,
                    "partial_uncompressed_image": partial_uncompressed,
                    "zero_filled_missing_rows": zero_filled_missing_rows or 0,
                    "zero_filled_checksum_failed_rows": zero_filled_checksum_failed_rows,
                    "zero_filled_total_rows": zero_filled_total_rows,
                    "out_of_range_rows": out_of_range_rows,
                },
            )
            fits_path = str(fits_file)
            stats.fits_written += 1
            stats.output_paths.append(fits_file)
            if write_meta_json:
                meta_json_file = output_dir / f"image_{image_id}{suffix}_meta.json"
                _write_csie_meta_json(meta_json_file, image_id, meta)
                meta_json_path = str(meta_json_file)
                stats.output_paths.append(meta_json_file)
            if write_png:
                _write_csie_png(png_file, image)
                png_path = str(png_file)
                stats.png_written += 1
                stats.output_paths.append(png_file)
            if write_jpeg2000:
                _write_csie_jpeg2000(jp2_file, image)
                jp2_path = str(jp2_file)
                stats.jp2_written += 1
                stats.output_paths.append(jp2_file)
            stats.images_written += 1

        inventory_row = {
            "image_id": image_id,
            "source": stats.source,
            "meta_packets": metadata_counts[image_id],
            "data_packets": len(row_summaries),
            "parsed_rows": unique_rows,
            "unique_rows": unique_rows,
            "duplicate_rows": duplicate_rows,
            "skipped_rows": skipped_rows,
            "expected_rows": expected_rows if expected_rows is not None else "",
            "expected_cols": expected_cols if expected_cols is not None else "",
            "missing_rows": missing_rows if missing_rows is not None else "",
            "row_length_min": row_length_min if row_length_min is not None else "",
            "row_length_mode": row_length_mode if row_length_mode is not None else "",
            "row_length_max": row_length_max if row_length_max is not None else "",
            "bad_length_rows": bad_length_rows,
            "out_of_range_rows": out_of_range_rows,
            "compression_enabled": compression_enabled,
            "bit_depth": bit_depth if bit_depth is not None else "",
            "checksum_valid_rows": checksum_valid,
            "checksum_failed_rows": checksum_failed,
            "checksum_missing_rows": checksum_missing,
            "selected_checksum_valid_rows": selected_checksum_valid,
            "selected_checksum_failed_rows": selected_checksum_failed,
            "selected_checksum_missing_rows": selected_checksum_missing,
            "complete_uncompressed_image": complete,
            "partial_uncompressed_image": partial_uncompressed,
            "zero_filled_missing_rows": zero_filled_missing_rows or 0,
            "zero_filled_checksum_failed_rows": zero_filled_checksum_failed_rows,
            "zero_filled_total_rows": zero_filled_total_rows,
            "fits_path": fits_path,
            "jpeg2000_path": jp2_path,
            "png_path": png_path,
            "jpegls_path": jpegls_path,
            "jpegls_chunks": jpegls_chunks,
            "jpegls_selected_chunks": jpegls_selected_chunks,
            "jpegls_bytes": jpegls_bytes,
            "jpegls_eoi_found": jpegls_eoi_found,
            "jpegls_bytes_trimmed_after_eoi": jpegls_bytes_trimmed_after_eoi,
            "jpegls_decoded": jpegls_decoded,
            "jpegls_decode_error": jpegls_decode_error,
            "meta_json_path": meta_json_path,
            "warnings": " | ".join(warnings),
        }
        inventory_rows.append(inventory_row)

    _write_csv_rows(
        inventory_path,
        inventory_rows,
        [
            "image_id",
            "source",
            "meta_packets",
            "data_packets",
            "parsed_rows",
            "unique_rows",
            "duplicate_rows",
            "skipped_rows",
            "expected_rows",
            "expected_cols",
            "missing_rows",
            "row_length_min",
            "row_length_mode",
            "row_length_max",
            "bad_length_rows",
            "out_of_range_rows",
            "compression_enabled",
            "bit_depth",
            "checksum_valid_rows",
            "checksum_failed_rows",
            "checksum_missing_rows",
            "selected_checksum_valid_rows",
            "selected_checksum_failed_rows",
            "selected_checksum_missing_rows",
            "complete_uncompressed_image",
            "partial_uncompressed_image",
            "zero_filled_missing_rows",
            "zero_filled_checksum_failed_rows",
            "zero_filled_total_rows",
            "fits_path",
            "jpeg2000_path",
            "png_path",
            "jpegls_path",
            "jpegls_chunks",
            "jpegls_selected_chunks",
            "jpegls_bytes",
            "jpegls_eoi_found",
            "jpegls_bytes_trimmed_after_eoi",
            "jpegls_decoded",
            "jpegls_decode_error",
            "meta_json_path",
            "warnings",
        ],
    )
    stats.output_paths.append(inventory_path)
    if not stats.meta_packets and not stats.data_packets:
        stats.warnings.append(
            "No active APID 536 csie_data or APID 538 csie_meta packets were recovered; "
            "no CSIE image products written."
        )
    return stats


def print_csie_image_summary(stats: CsieImageStats) -> None:
    print("\nCSIE image assembly summary:")
    print(f"  source:                               {stats.source or 'none'}")
    print(f"  APID 538 metadata packets:             {stats.meta_packets:,}")
    print(f"  APID 536 data row packets:             {stats.data_packets:,}")
    print(f"  image IDs seen:                        {stats.image_ids_seen:,}")
    print(f"  complete uncompressed images:          {stats.images_complete:,}")
    print(f"  partial uncompressed images:           {stats.images_partial:,}")
    print(f"  compressed JPEG-LS streams written:    {stats.compressed_images_written:,}")
    print(f"  compressed JPEG-LS images decoded:     {stats.compressed_images_decoded:,}")
    print(f"  compressed JPEG-LS decode failures:    {stats.compressed_decode_failures:,}")
    print(f"  images written:                        {stats.images_written:,}")
    print(f"    FITS files written:                  {stats.fits_written:,}")
    print(f"    JPEG2000 files written:              {stats.jp2_written:,}")
    print(f"    PNG files written:                   {stats.png_written:,}")
    print(f"    JPEG-LS files written:               {stats.jpegls_written:,}")
    print(f"  CSIE row checksum-valid rows:          {stats.checksum_valid_rows:,}")
    print(f"  CSIE row checksum-failed rows:         {stats.checksum_failed_rows:,}")
    print(f"  CSIE row checksums missing:            {stats.checksum_missing_rows:,}")
    print(f"  skipped CSIE row packets:              {stats.skipped_rows:,}")
    print(f"  duplicate CSIE row packets:            {stats.duplicate_rows:,}")
    print(f"  inventory CSV:                         {stats.inventory_path}")
    print(f"  image output folder:                   {stats.output_dir}")
    for warning in stats.warnings:
        print(f"WARNING: {warning}")


def print_decode_summary(stats: DecodeStats) -> None:
    print("\nDecode/export summary:")
    print(f"  packets offered to decoders:           {stats.total_packets:,}")
    print(f"  decoded packets:                       {stats.decoded_packets:,}")
    print(f"    generated CTDB decoder packets:      {stats.generated_decoder_packets:,}")
    print(f"    generic CTDB CSV decoded packets:    {stats.generic_ctdb_packets:,}")
    print(f"  decode failures:                       {stats.decode_failed_packets:,}")
    print(f"  packets without decoder/field rows:    {stats.no_decoder_packets:,}")
    print(f"  primary headers normalized to BE:      {stats.primary_headers_normalized:,}")
    print(f"  payloads 16-bit word byte-swapped:     {stats.payload_16bit_words_swapped:,}")
    print(f"  manifest CSV:                          {stats.manifest_path}")
    print(f"  decode summary CSV:                    {stats.summary_path}")
    print(f"  per-APID CSV folder:                   {stats.decoded_dir}")
    if stats.unexpected_little_endian_headers:
        print(
            "WARNING: decoded "
            f"{stats.unexpected_little_endian_headers:,} packet(s) recovered from "
            "unexpected APID 68 playback wrappers after reconstructing shuffled "
            "CCSDS headers and normalizing payload byte order."
        )


def _relative_path_text(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _total_input_bytes(paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in paths)


def _products_are_current(input_paths: list[Path], required_outputs: list[Path]) -> bool:
    if not input_paths or not required_outputs:
        return False
    if any(not path.is_file() for path in required_outputs):
        return False
    newest_input = max(path.stat().st_mtime for path in input_paths)
    oldest_output = min(path.stat().st_mtime for path in required_outputs)
    return oldest_output >= newest_input


def discover_combined_source_specs(folder: Path) -> list[SourceSpec]:
    """Discover top-level X-band/hardline files and explicit nested UHF captures."""
    source_products_root = folder / DEFAULT_SOURCE_PRODUCTS_DIR_BASENAME
    specs: list[SourceSpec] = []

    xband_paths = discover_prefixed_binary_files(folder, DEFAULT_XBAND_PREFIX)
    if xband_paths:
        specs.append(
            SourceSpec(
                source_id="xband",
                input_mode=INPUT_MODE_XBAND,
                search_root=folder,
                input_paths=xband_paths,
                output_dir=source_products_root / "xband",
                prefix=DEFAULT_XBAND_PREFIX,
            )
        )

    hardline_paths = discover_prefixed_binary_files(folder, DEFAULT_HARDLINE_CCSDS_PREFIX)
    if hardline_paths:
        specs.append(
            SourceSpec(
                source_id="hardline",
                input_mode=INPUT_MODE_CCSDS,
                search_root=folder,
                input_paths=hardline_paths,
                output_dir=source_products_root / "hardline",
                prefix=DEFAULT_HARDLINE_CCSDS_PREFIX,
            )
        )

    uhf_search_root = resolve_uhf_search_root(folder)
    if uhf_search_root.name == DEFAULT_UHF_SUBDIR_BASENAME or (
        folder / DEFAULT_UHF_SUBDIR_BASENAME
    ).is_dir():
        uhf_paths = discover_uhf_binary_files(folder, DEFAULT_HARDLINE_CCSDS_PREFIX)
        if uhf_paths:
            specs.append(
                SourceSpec(
                    source_id="uhf",
                    input_mode=INPUT_MODE_UHF,
                    search_root=uhf_search_root,
                    input_paths=uhf_paths,
                    output_dir=source_products_root / "uhf",
                    prefix=DEFAULT_HARDLINE_CCSDS_PREFIX,
                )
            )

    if not specs:
        raise FileNotFoundError(
            "Combined input mode found no top-level X-band/hardline files and no "
            f"nested {DEFAULT_UHF_SUBDIR_BASENAME!r} CCSDS files under {folder}"
        )
    return specs


def _source_file_for_record_offset(
    spec: SourceSpec,
    source_offset: int,
    *,
    packet_offsets_are_passthrough: bool,
) -> str:
    if len(spec.input_paths) == 1:
        return _relative_path_text(spec.input_paths[0], spec.search_root)
    if not packet_offsets_are_passthrough:
        return ""

    running = 0
    for path in spec.input_paths:
        size = path.stat().st_size
        if running <= source_offset < running + size:
            return _relative_path_text(path, spec.search_root)
        running += size
    return ""


def annotate_records_with_source_metadata(
    records: list[PacketRecord],
    spec: SourceSpec,
    *,
    provenance_quality: str,
    packet_offsets_are_passthrough: bool,
) -> None:
    for record in records:
        record.source_packet_index = record.packet_index
        record.source_id = spec.source_id
        record.source_mode = spec.input_mode
        record.source_root = str(spec.search_root)
        record.input_file_relative_path = _source_file_for_record_offset(
            spec,
            record.source_offset,
            packet_offsets_are_passthrough=packet_offsets_are_passthrough,
        )
        record.source_acceptance_mode = record.acceptance_mode
        record.source_provenance_quality = provenance_quality
        record.source_output_dir = str(spec.output_dir)


def load_reusable_packet_records(
    packets_valid_path: Path,
    valid_apids: set[int],
    expected_packet_bytes: dict[int, int],
) -> tuple[bytes, list[PacketRecord], list[str]]:
    data = packets_valid_path.read_bytes()
    repacketized, stats = packetize_checksum_valid_ccsds(
        data,
        valid_apids,
        expected_packet_bytes=expected_packet_bytes,
        bypass_packet_checksums=True,
        extract_playback_wrappers=False,
    )
    warnings: list[str] = []
    if len(repacketized) != len(data):
        warnings.append(
            "Reused packets_valid.bin did not reparse byte-for-byte; combined mode "
            f"kept {len(repacketized):,} of {len(data):,} bytes."
        )
    warnings.append(
        "Reused packets_valid.bin for this source; source_offset values are offsets "
        "within that compact packet stream, not the original merged_fixed.bin."
    )
    return repacketized, stats.records, warnings


def process_source_product(
    spec: SourceSpec,
    args: argparse.Namespace,
    *,
    config: Config,
    apid_names: dict[int, str],
    valid_apids: set[int],
    expected_packet_bytes: dict[int, int],
) -> SourceProduct:
    output_dir = spec.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / DEFAULT_MERGED_BASENAME
    fixed_path = output_dir / DEFAULT_FIXED_BASENAME
    packets_valid_path = output_dir / DEFAULT_PACKETS_VALID_BASENAME
    csie_inventory_path = (
        output_dir
        / args.csie_dir_name
        / DEFAULT_CSIE_INVENTORY_BASENAME
    )
    required_outputs = [packets_valid_path]
    if not args.skip_csie_images:
        required_outputs.append(csie_inventory_path)

    print(f"\n=== Source {spec.source_id} ({spec.input_mode}) ===")
    print(f"Search root: {spec.search_root}")
    print(f"Product folder: {output_dir}")
    print_input_file_summary(
        spec.input_paths,
        input_mode=spec.input_mode,
        relative_to=spec.search_root,
    )
    if spec.input_mode == INPUT_MODE_UHF:
        print(
            "UHF/Hydra assumption: APID 73 segmented playback is de-duplicated and "
            "reassembled, and validated direct APID 72 playback chains are unwrapped "
            "before packet recovery."
        )

    if (
        not args.force_source_reprocess
        and _products_are_current(spec.input_paths, required_outputs)
    ):
        packets_valid_data, records, warnings = load_reusable_packet_records(
            packets_valid_path,
            valid_apids,
            expected_packet_bytes,
        )
        annotate_records_with_source_metadata(
            records,
            spec,
            provenance_quality="reused_packets_valid_only",
            packet_offsets_are_passthrough=False,
        )
        print(
            f"Reused current source products from {output_dir}: "
            f"{len(records):,} packet(s), {len(packets_valid_data):,} packet byte(s)"
        )
        for warning in warnings:
            print(f"WARNING: {warning}")
        return SourceProduct(
            spec=spec,
            packets_valid_path=packets_valid_path,
            output_dir=output_dir,
            records=records,
            reused=True,
            merged_path=merged_path if merged_path.is_file() else None,
            fixed_path=fixed_path if fixed_path.is_file() else None,
            packet_bytes=len(packets_valid_data),
            raw_bytes=_total_input_bytes(spec.input_paths),
            warnings=warnings,
        )

    merge_stats = write_merged_binary(spec.input_paths, merged_path)
    print(
        f"Wrote {merge_stats.output_path} from {merge_stats.input_file_count} file(s): "
        f"{merge_stats.input_bytes:,} bytes"
    )

    merged_data = merged_path.read_bytes()
    fixed_data, fix_stats = build_fixed_binary(
        merged_data,
        valid_apids=valid_apids,
        expected_packet_bytes=expected_packet_bytes,
        input_mode=spec.input_mode,
        spacecraft_id=args.spacecraft_id,
        strip_out_of_phase_xband_artifacts=not args.preserve_rf_lapse_artifacts,
    )
    fixed_path.write_bytes(fixed_data)
    print(f"Wrote {fixed_path}: {len(fixed_data):,} bytes")
    print_fix_summary(fix_stats)

    packets_valid_data, packet_stats = packetize_checksum_valid_ccsds(
        fixed_data,
        valid_apids,
        expected_packet_bytes=expected_packet_bytes,
        bypass_packet_checksums=not args.require_packet_checksums,
        extract_playback_wrappers=spec.input_mode == INPUT_MODE_XBAND,
    )
    packets_valid_path.write_bytes(packets_valid_data)
    print(f"Wrote {packets_valid_path}: {len(packets_valid_data):,} bytes")
    print_packet_summary(packet_stats, apid_names)

    annotate_records_with_source_metadata(
        packet_stats.records,
        spec,
        provenance_quality="source_offset_from_merged_fixed_stream",
        packet_offsets_are_passthrough=(
            (
                spec.input_mode == INPUT_MODE_CCSDS
                and not (
                    fix_stats.direct_playback
                    and fix_stats.direct_playback.wrappers_stripped
                )
            )
            or (
                spec.input_mode == INPUT_MODE_UHF
                and not (
                    fix_stats.uhf_playback
                    and fix_stats.uhf_playback.segment_packets_seen
                )
                and not (
                    fix_stats.direct_playback
                    and fix_stats.direct_playback.wrappers_stripped
                )
            )
        ),
    )

    if not args.skip_csie_images:
        csie_stats = write_csie_image_products(
            packet_stats.records,
            config,
            output_dir / args.csie_dir_name,
            fixed_payload_stream=fixed_data,
            output_suffix=args.product_suffix,
            write_jpeg2000=not args.skip_csie_jpeg2000,
            write_png=not args.skip_csie_png,
        )
        print_csie_image_summary(csie_stats)

    return SourceProduct(
        spec=spec,
        packets_valid_path=packets_valid_path,
        output_dir=output_dir,
        records=packet_stats.records,
        reused=False,
        merged_path=merged_path,
        fixed_path=fixed_path,
        packet_bytes=len(packets_valid_data),
        raw_bytes=merge_stats.input_bytes,
    )


def _record_sort_time(record: PacketRecord) -> tuple[int, int] | None:
    if record.apid == CSIE_DATA_APID or len(record.packet) < 12:
        return None
    first_word = int.from_bytes(record.packet[0:2], "big")
    packet_version = (first_word >> 13) & 0x07
    packet_type = (first_word >> 12) & 0x01
    secondary_header_flag = (first_word >> 11) & 0x01
    if packet_version != 0 or packet_type != 0 or secondary_header_flag != 1:
        return None
    coarse = int.from_bytes(record.packet[6:10], "big")
    fine = int.from_bytes(record.packet[10:12], "big")
    record.combined_time_source = "ccsds_secondary_header"
    record.combined_time_coarse = coarse
    record.combined_time_fine = fine
    return coarse, fine


def prepare_combined_records(source_products: list[SourceProduct]) -> list[PacketRecord]:
    records: list[PacketRecord] = []
    for product in source_products:
        records.extend(product.records)

    packet_hash_counts: Counter = Counter()
    for record in records:
        record.packet_hash = hashlib.blake2b(record.packet, digest_size=16).hexdigest()
        packet_hash_counts[record.packet_hash] += 1

    duplicate_hashes = sorted(
        packet_hash for packet_hash, count in packet_hash_counts.items() if count > 1
    )
    duplicate_group_ids = {
        packet_hash: f"dup_{index:06d}"
        for index, packet_hash in enumerate(duplicate_hashes, start=1)
    }
    for record in records:
        group_size = packet_hash_counts[record.packet_hash]
        record.duplicate_group_size = group_size
        record.is_duplicate_packet = group_size > 1
        record.duplicate_group_id = duplicate_group_ids.get(record.packet_hash, "")
        if _record_sort_time(record) is None:
            record.combined_time_source = "source_order_fallback"
            record.combined_time_coarse = None
            record.combined_time_fine = None

    def sort_key(record: PacketRecord) -> tuple[object, ...]:
        source_packet_index = (
            record.source_packet_index
            if record.source_packet_index is not None
            else record.packet_index
        )
        if record.combined_time_source == "ccsds_secondary_header":
            return (
                0,
                record.combined_time_coarse or 0,
                record.combined_time_fine or 0,
                record.source_id,
                record.input_file_relative_path,
                source_packet_index,
                record.source_offset,
            )
        return (
            1,
            record.source_id,
            record.input_file_relative_path,
            source_packet_index,
            record.source_offset,
        )

    records.sort(key=sort_key)
    for packet_index, record in enumerate(records):
        record.packet_index = packet_index
    return records


def packetize_stats_from_records(records: list[PacketRecord]) -> PacketizeStats:
    stats = PacketizeStats(records=records)
    for record in records:
        algorithm = record.acceptance_mode or "combined_record"
        _record_candidate_packet(stats, record.apid, record.packet_len)
        _record_valid_packet(stats, record.apid, record.packet_len, algorithm)
    return stats


def write_combined_source_summary(
    path: Path,
    source_products: list[SourceProduct],
) -> None:
    rows: list[dict[str, object]] = []
    for product in source_products:
        apids = Counter(record.apid for record in product.records)
        rows.append(
            {
                "source_id": product.spec.source_id,
                "source_mode": product.spec.input_mode,
                "source_root": str(product.spec.search_root),
                "source_output_dir": str(product.output_dir),
                "input_files": len(product.spec.input_paths),
                "input_bytes": product.raw_bytes,
                "packets": len(product.records),
                "packet_bytes": product.packet_bytes,
                "reused": product.reused,
                "packets_valid_path": str(product.packets_valid_path),
                "top_apids": ", ".join(
                    f"{apid}:{count}" for apid, count in apids.most_common(12)
                ),
                "warnings": " | ".join(product.warnings),
            }
        )
    _write_csv_rows(
        path,
        rows,
        [
            "source_id",
            "source_mode",
            "source_root",
            "source_output_dir",
            "input_files",
            "input_bytes",
            "packets",
            "packet_bytes",
            "reused",
            "packets_valid_path",
            "top_apids",
            "warnings",
        ],
    )


def run_combined_pipeline(
    args: argparse.Namespace,
    *,
    config: Config,
    folder: Path,
) -> None:
    if args.prefix is not None:
        print(
            "WARNING: --prefix is ignored in combined mode; using default source "
            "discovery for top-level X-band/hardline and nested UHF files."
        )

    specs = discover_combined_source_specs(folder)
    print(f"Data folder: {folder}")
    print(f"Input mode:  {INPUT_MODE_COMBINED}")
    print("Combined sources:")
    for spec in specs:
        print(
            f"  {spec.source_id}: {spec.input_mode}, "
            f"{len(spec.input_paths):,} file(s), search root {spec.search_root}"
        )

    apid_names = read_apid_names_from_config(config)
    valid_apids = set(apid_names)
    print(f"Loaded {len(valid_apids)} valid APIDs from configured CTDBs.")
    if TEMP_DSPS_DATA_APID in apid_names:
        print(
            "WARNING: FIXME temporary CTDB workaround active: treating APID "
            f"{TEMP_DSPS_DATA_APID} as {TEMP_DSPS_DATA_PACKET_NAME!r} with "
            f"{TEMP_DSPS_DATA_PACKET_BYTES} total packet bytes until ct_pkt.csv "
            "and ct_tlm.csv include DSPS data."
        )
    expected_packet_bytes = read_expected_packet_bytes_from_config(config)
    print(
        f"Loaded fixed packet byte sizes for {len(expected_packet_bytes)} APIDs "
        "from configured CTDB telemetry definitions."
    )

    source_products = [
        process_source_product(
            spec,
            args,
            config=config,
            apid_names=apid_names,
            valid_apids=valid_apids,
            expected_packet_bytes=expected_packet_bytes,
        )
        for spec in specs
    ]

    combined_records = prepare_combined_records(source_products)
    combined_data = b"".join(record.packet for record in combined_records)
    combined_packets_path = folder / COMBINED_PACKETS_VALID_BASENAME
    combined_packets_path.write_bytes(combined_data)
    print(
        f"\nWrote {combined_packets_path}: {len(combined_data):,} bytes from "
        f"{len(combined_records):,} merged packet record(s)"
    )

    combined_stats = packetize_stats_from_records(combined_records)
    print_packet_summary(combined_stats, apid_names)

    source_summary_path = folder / COMBINED_SOURCE_SUMMARY_BASENAME
    write_combined_source_summary(source_summary_path, source_products)
    print(f"Wrote combined source summary: {source_summary_path}")

    if not args.skip_decode_csv:
        decode_stats = decode_packet_records_to_csv(
            combined_records,
            config,
            apid_names,
            folder / COMBINED_DECODED_DIR_BASENAME,
            manifest_path_override=folder / COMBINED_PACKET_MANIFEST_BASENAME,
            summary_path_override=folder / COMBINED_DECODE_SUMMARY_BASENAME,
        )
        print_decode_summary(decode_stats)

    if args.skip_csie_images:
        print("CSIE image assembly skipped for all source products.")
    else:
        print(
            "CSIE image products remain source-specific under "
            f"{folder / DEFAULT_SOURCE_PRODUCTS_DIR_BASENAME}/*/"
            f"{args.csie_dir_name}."
        )


def main(argv: list[str] | None = None) -> None:
    default_config = Path(__file__).resolve().parent / "config_files" / "config_default.ini"
    parser = argparse.ArgumentParser(
        description=(
            "First staged Level 0.5 ingest builder: merge raw files, clean wrappers, "
            "packetize, decode, and assemble CSIE products."
        )
    )
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Path to processing config INI.",
    )
    parser.add_argument(
        "--folder",
        default=None,
        help="Override config data folder. Defaults to paths.data_to_process_path.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help=(
            "Input binary basename prefix. Default is mode-dependent: "
            f"{DEFAULT_XBAND_PREFIX!r} for X-band, {DEFAULT_HARDLINE_CCSDS_PREFIX!r} "
            "for hardline CCSDS and UHF. In auto mode, only top-level X-band and "
            "hardline defaults are tried."
        ),
    )
    parser.add_argument(
        "--input-mode",
        choices=(
            INPUT_MODE_AUTO,
            INPUT_MODE_XBAND,
            INPUT_MODE_CCSDS,
            INPUT_MODE_HARDLINE,
            INPUT_MODE_UHF,
            INPUT_MODE_COMBINED,
        ),
        default=INPUT_MODE_AUTO,
        help=(
            "Input wrapper format. 'xband' strips X-band transfer frames; 'ccsds' "
            "or 'hardline' treats merged input as a direct CCSDS packet stream; "
            "'uhf' recursively reads Hydra UHF ccsds_* files, preserving direct "
            "CCSDS packets and reassembling APID 73 segmented APID 72 playback; "
            "'combined' processes top-level X-band/hardline plus nested UHF sources "
            "separately, then merges recovered CCSDS packet records; "
            "'auto' detects top-level default prefixes only."
        ),
    )
    parser.add_argument(
        "--force-source-reprocess",
        action="store_true",
        help=(
            "Combined mode only: rebuild per-source products even when existing "
            "source_products outputs are newer than the raw inputs."
        ),
    )
    parser.add_argument(
        "--spacecraft-id",
        type=int,
        default=DEFAULT_SPACECRAFT_ID,
        help=f"Expected outer transfer-frame spacecraft ID. Default: {DEFAULT_SPACECRAFT_ID}.",
    )
    parser.add_argument(
        "--merged-name",
        default=DEFAULT_MERGED_BASENAME,
        help=f"Merged raw output basename. Default: {DEFAULT_MERGED_BASENAME}.",
    )
    parser.add_argument(
        "--fixed-name",
        default=DEFAULT_FIXED_BASENAME,
        help=f"Fixed payload-stream output basename. Default: {DEFAULT_FIXED_BASENAME}.",
    )
    parser.add_argument(
        "--packets-valid-name",
        default=DEFAULT_PACKETS_VALID_BASENAME,
        help=(
            "Accepted CCSDS packet output basename. "
            f"Default: {DEFAULT_PACKETS_VALID_BASENAME}."
        ),
    )
    parser.add_argument(
        "--require-packet-checksums",
        action="store_true",
        help=(
            "Require inner packet checksum validation before writing packets_valid.bin. "
            "Default is to bypass packet checksums while the FSW checksum contract is "
            "being clarified."
        ),
    )
    parser.add_argument(
        "--decoded-dir-name",
        default=DEFAULT_DECODED_DIR_BASENAME,
        help=(
            "Folder basename for decoded packet CSV exports. "
            f"Default: {DEFAULT_DECODED_DIR_BASENAME}."
        ),
    )
    parser.add_argument(
        "--skip-decode-csv",
        action="store_true",
        help="Skip CTDB decoder CSV export after writing packets_valid.bin.",
    )
    parser.add_argument(
        "--csie-dir-name",
        default=DEFAULT_CSIE_IMAGE_DIR_BASENAME,
        help=(
            "Folder basename for staged CSIE image inventory/products. "
            f"Default: {DEFAULT_CSIE_IMAGE_DIR_BASENAME}."
        ),
    )
    parser.add_argument(
        "--skip-csie-images",
        action="store_true",
        help="Skip CSIE image inventory and JLS/FITS/PNG/JPEG2000 product assembly.",
    )
    parser.add_argument(
        "--skip-csie-jpeg2000",
        action="store_true",
        help="Write CSIE FITS products but skip JPEG2000 preview products.",
    )
    parser.add_argument(
        "--skip-csie-png",
        action="store_true",
        help="Write CSIE FITS products but skip inferno PNG preview products.",
    )
    parser.add_argument(
        "--product-suffix",
        default="",
        help=(
            "Optional suffix for staged CSIE image filenames. Default is empty; config "
            "structure.output_suffix is intentionally not used here."
        ),
    )
    parser.add_argument(
        "--preserve-rf-lapse-artifacts",
        action="store_true",
        help=(
            "Diagnostic mode: preserve out-of-phase X-band-looking seams instead of "
            "stripping them as RF-lapse/resync artifacts."
        ),
    )
    args = parser.parse_args(argv)

    config = Config(os.path.abspath(os.path.expanduser(args.config)))
    folder = (
        Path(args.folder).expanduser().resolve()
        if args.folder is not None
        else resolve_config_data_folder(config)
    )
    if not folder.is_dir():
        raise FileNotFoundError(f"Data folder does not exist: {folder}")

    if args.input_mode == INPUT_MODE_COMBINED:
        run_combined_pipeline(args, config=config, folder=folder)
        return

    input_mode, input_prefix, input_paths = resolve_input_files_and_mode(
        folder,
        prefix=args.prefix,
        input_mode=args.input_mode,
    )
    if not input_paths:
        if input_mode == INPUT_MODE_UHF:
            uhf_search_root = resolve_uhf_search_root(folder)
            raise FileNotFoundError(
                f"No recursive UHF files starting with {input_prefix!r} under "
                f"{uhf_search_root}. Pass --folder as either the parent folder "
                f"containing {DEFAULT_UHF_SUBDIR_BASENAME!r} or the UHF folder itself."
            )
        raise FileNotFoundError(
            f"No top-level files starting with {input_prefix!r} in {folder}"
        )

    print(f"Data folder: {folder}")
    print(f"Input mode:  {input_mode}")
    print(f"Input prefix: {input_prefix!r}")
    uhf_search_root = resolve_uhf_search_root(folder) if input_mode == INPUT_MODE_UHF else None
    if uhf_search_root is not None:
        print(f"UHF search root: {uhf_search_root}")
    print_input_file_summary(
        input_paths,
        input_mode=input_mode,
        relative_to=uhf_search_root,
    )
    if input_mode == INPUT_MODE_XBAND:
        print(
            "\nFrame assumption: 2056-byte records = 4-byte ASM + 6-byte TM primary "
            "+ 2-byte padding + 2044-byte data field."
        )
    elif input_mode == INPUT_MODE_UHF:
        print(
            "\nUHF/Hydra assumption: ordinary captures are direct CCSDS streams and "
            "pass through unchanged. APID 73 segmented playback is reassembled, and "
            "validated direct APID 72 playback chains are unwrapped before packet "
            "recovery."
        )
    else:
        print(
            "\nHardline assumption: input files contain CCSDS packets without ASM or "
            "transfer-frame wrappers. Validated direct APID 72 playback chains are "
            "unwrapped before inner packet recovery."
        )

    merged_path = folder / args.merged_name
    fixed_path = folder / args.fixed_name
    packets_valid_path = folder / args.packets_valid_name

    merge_stats = write_merged_binary(input_paths, merged_path)
    print(
        f"\nWrote {merge_stats.output_path} from {merge_stats.input_file_count} file(s): "
        f"{merge_stats.input_bytes:,} bytes"
    )

    apid_names = read_apid_names_from_config(config)
    valid_apids = set(apid_names)
    print(f"Loaded {len(valid_apids)} valid APIDs from configured CTDBs.")
    if TEMP_DSPS_DATA_APID in apid_names:
        print(
            "WARNING: FIXME temporary CTDB workaround active: treating APID "
            f"{TEMP_DSPS_DATA_APID} as {TEMP_DSPS_DATA_PACKET_NAME!r} with "
            f"{TEMP_DSPS_DATA_PACKET_BYTES} total packet bytes until ct_pkt.csv "
            "and ct_tlm.csv include DSPS data."
        )
    expected_packet_bytes = read_expected_packet_bytes_from_config(config)
    print(
        f"Loaded fixed packet byte sizes for {len(expected_packet_bytes)} APIDs "
        "from configured CTDB telemetry definitions."
    )
    merged_data = merged_path.read_bytes()
    fixed_data, fix_stats = build_fixed_binary(
        merged_data,
        valid_apids=valid_apids,
        expected_packet_bytes=expected_packet_bytes,
        input_mode=input_mode,
        spacecraft_id=args.spacecraft_id,
        strip_out_of_phase_xband_artifacts=not args.preserve_rf_lapse_artifacts,
    )
    fixed_path.write_bytes(fixed_data)
    print(f"Wrote {fixed_path}: {len(fixed_data):,} bytes")
    print_fix_summary(fix_stats)

    packets_valid_data, packet_stats = packetize_checksum_valid_ccsds(
        fixed_data,
        valid_apids,
        expected_packet_bytes=expected_packet_bytes,
        bypass_packet_checksums=not args.require_packet_checksums,
        extract_playback_wrappers=input_mode == INPUT_MODE_XBAND,
    )
    packets_valid_path.write_bytes(packets_valid_data)
    print(f"\nWrote {packets_valid_path}: {len(packets_valid_data):,} bytes")
    print_packet_summary(packet_stats, apid_names)
    single_source_spec = SourceSpec(
        source_id=input_mode,
        input_mode=input_mode,
        search_root=uhf_search_root or folder,
        input_paths=input_paths,
        output_dir=folder,
        prefix=input_prefix,
    )
    annotate_records_with_source_metadata(
        packet_stats.records,
        single_source_spec,
        provenance_quality="source_offset_from_merged_fixed_stream",
        packet_offsets_are_passthrough=(
            (
                input_mode == INPUT_MODE_CCSDS
                and not (
                    fix_stats.direct_playback
                    and fix_stats.direct_playback.wrappers_stripped
                )
            )
            or (
                input_mode == INPUT_MODE_UHF
                and not (
                    fix_stats.uhf_playback
                    and fix_stats.uhf_playback.segment_packets_seen
                )
                and not (
                    fix_stats.direct_playback
                    and fix_stats.direct_playback.wrappers_stripped
                )
            )
        ),
    )
    for record in packet_stats.records:
        record.packet_hash = hashlib.blake2b(record.packet, digest_size=16).hexdigest()
        if _record_sort_time(record) is None:
            record.combined_time_source = "single_source_order"

    if not args.skip_decode_csv:
        decoded_dir = folder / args.decoded_dir_name
        decode_stats = decode_packet_records_to_csv(
            packet_stats.records,
            config,
            apid_names,
            decoded_dir,
        )
        print_decode_summary(decode_stats)

    if not args.skip_csie_images:
        csie_stats = write_csie_image_products(
            packet_stats.records,
            config,
            folder / args.csie_dir_name,
            fixed_payload_stream=fixed_data,
            output_suffix=args.product_suffix,
            write_jpeg2000=not args.skip_csie_jpeg2000,
            write_png=not args.skip_csie_png,
        )
        print_csie_image_summary(csie_stats)


if __name__ == "__main__":
    main()

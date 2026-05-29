"""Columnar Level 0.5 HDF5 merge helpers (h5py / numpy only — no pandas / make_level0_5)."""

from __future__ import annotations

import os
from datetime import datetime

import h5py
import numpy as np


def level05_h5_filename(config, data_type: str) -> str:
    version = config.version_pipeline
    suf = (config.output_suffix or "").strip()
    suffix = f"-{suf}" if suf else ""
    return f"suncet_{data_type}_mission_length_v{version}{suffix}.h5"


def _decode_packet_type(val):
    if isinstance(val, bytes):
        return val.decode("utf-8").rstrip("\x00")
    return str(val)


def read_columnar_level05_h5(path):
    """Load rows from HDF5 written by ``save_packet_data_to_hdf5`` (root datasets, not legacy group)."""
    rows = []
    with h5py.File(path, "r") as f:
        if "timestamp_seconds_since_boot" not in f or "packet_type" not in f:
            return rows
        ts = f["timestamp_seconds_since_boot"][:]
        pt = f["packet_type"][:]
        n = int(ts.shape[0])
        field_ds = [
            k
            for k in f.keys()
            if k not in ("timestamp_seconds_since_boot", "packet_type")
        ]
        for i in range(n):
            row = {
                "timestamp_seconds_since_boot": float(ts[i]),
                "packet_type": _decode_packet_type(pt[i]),
            }
            for k in field_ds:
                v = f[k][i]
                fv = float(v)
                if np.isnan(fv):
                    continue
                row[k] = fv
            rows.append(row)
    return rows


def deduplicate_points(points):
    """Same rule as ``Level0_5.deduplicate_telemetry_points`` (last wins)."""
    dedup = {}
    for p in points:
        key = (p["timestamp_seconds_since_boot"], p["packet_type"])
        dedup[key] = p
    return list(dedup.values())


def write_columnar_level05_h5(path, unique_packets, data_type, version, attrs_extra):
    """Write merged file in the same columnar layout as ``save_packet_data_to_hdf5``."""
    if not unique_packets:
        return
    unique_packets.sort(key=lambda x: x["timestamp_seconds_since_boot"])
    field_names = set()
    for packet in unique_packets:
        for key in packet:
            if key not in ("timestamp_seconds_since_boot", "packet_type"):
                field_names.add(key)
    field_names = sorted(field_names)
    num_packets = len(unique_packets)
    timestamps = np.array(
        [p["timestamp_seconds_since_boot"] for p in unique_packets], dtype="f8"
    )
    packet_types = np.array(
        [p["packet_type"].encode("utf-8") for p in unique_packets], dtype="S50"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "timestamp_seconds_since_boot",
            data=timestamps,
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "packet_type", data=packet_types, compression="gzip", compression_opts=4
        )
        for field_name in field_names:
            field_data = np.full(num_packets, np.nan, dtype="f8")
            for i, packet in enumerate(unique_packets):
                if field_name in packet:
                    try:
                        field_data[i] = float(packet[field_name])
                    except (ValueError, TypeError):
                        field_data[i] = np.nan
            f.create_dataset(
                field_name, data=field_data, compression="gzip", compression_opts=4
            )
        f.attrs["version"] = version
        f.attrs[f"total_{data_type}_packets"] = len(unique_packets)
        f.attrs["time_range_start"] = float(timestamps[0])
        f.attrs["time_range_end"] = float(timestamps[-1])
        f.attrs["processing_timestamp"] = datetime.now().isoformat()
        f.attrs["field_names"] = field_names
        f.attrs["packet_types"] = list(set(p["packet_type"] for p in unique_packets))
        for k, v in attrs_extra.items():
            f.attrs[k] = v


def _probe_readable_level05_h5(path: str) -> None:
    """Raise if any dataset cannot be read (e.g. truncated gzip on one column)."""
    with h5py.File(path, "r") as f:
        for k in f.keys():
            _ = f[k][:]


def _field_names_union(source_paths: list[str]) -> list[str]:
    names = set()
    for path in source_paths:
        with h5py.File(path, "r") as f:
            for k in f.keys():
                if k not in ("timestamp_seconds_since_boot", "packet_type"):
                    names.add(k)
    return sorted(names)


def _merge_columnar_sources_vectorized(source_paths: list[str]) -> tuple:
    """Concatenate sources and deduplicate on (timestamp, packet_type); last row wins."""
    field_names = _field_names_union(source_paths)
    ts_parts: list[np.ndarray] = []
    pt_parts: list[np.ndarray] = []
    field_parts: dict[str, list[np.ndarray]] = {k: [] for k in field_names}

    for path in source_paths:
        with h5py.File(path, "r") as f:
            ts = np.asarray(f["timestamp_seconds_since_boot"][:], dtype=np.float64)
            pt = f["packet_type"][:]
            n = int(ts.shape[0])
            ts_parts.append(ts)
            pt_parts.append(pt)
            for k in field_names:
                if k in f:
                    field_parts[k].append(np.asarray(f[k][:], dtype=np.float64))
                else:
                    field_parts[k].append(np.full(n, np.nan, dtype=np.float64))

    ts = np.concatenate(ts_parts)
    pt = np.concatenate(pt_parts)
    fields = {k: np.concatenate(field_parts[k]) for k in field_names}

    n = int(ts.shape[0])
    keep = np.zeros(n, dtype=bool)
    seen = set()
    for i in range(n - 1, -1, -1):
        key = (float(ts[i]), _decode_packet_type(pt[i]))
        if key not in seen:
            seen.add(key)
            keep[i] = True
    keep_idx = np.flatnonzero(keep)
    keep_idx.sort()

    ts_out = ts[keep_idx]
    pt_out = pt[keep_idx]
    fields_out = {k: fields[k][keep_idx] for k in field_names}
    removed = n - int(keep_idx.shape[0])
    return ts_out, pt_out, fields_out, n, removed


def write_columnar_level05_h5_vectorized(
    path: str,
    ts: np.ndarray,
    pt_raw: np.ndarray,
    fields: dict[str, np.ndarray],
    data_type: str,
    version: str,
    attrs_extra: dict,
) -> None:
    """Write columnar Level 0.5 HDF5 from merged arrays (matches dict-based layout)."""
    num_packets = int(ts.shape[0])
    if num_packets == 0:
        return
    timestamps = np.asarray(ts, dtype=np.float64)
    packet_types = np.array(
        [_decode_packet_type(pt_raw[i]).encode("utf-8") for i in range(num_packets)],
        dtype="S50",
    )
    field_names = sorted(fields.keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "timestamp_seconds_since_boot",
            data=timestamps,
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "packet_type", data=packet_types, compression="gzip", compression_opts=4
        )
        for field_name in field_names:
            d = np.asarray(fields[field_name], dtype=np.float64)
            f.create_dataset(field_name, data=d, compression="gzip", compression_opts=4)
        f.attrs["version"] = version
        f.attrs[f"total_{data_type}_packets"] = num_packets
        f.attrs["time_range_start"] = float(timestamps[0])
        f.attrs["time_range_end"] = float(timestamps[-1])
        f.attrs["processing_timestamp"] = datetime.now().isoformat()
        f.attrs["field_names"] = field_names
        unique_pt_bytes = np.unique(pt_raw)
        f.attrs["packet_types"] = [
            _decode_packet_type(x) for x in unique_pt_bytes
        ]
        for k, v in attrs_extra.items():
            f.attrs[k] = v


def merge_per_folder_h5s(test_data, succeeded_names, config, merge_root):
    """Merge ``telemetry`` and ``dsps`` HDF5 from each named folder into ``merge_root``."""
    merge_level05 = os.path.join(merge_root, "level0_5")
    os.makedirs(merge_level05, exist_ok=True)
    version = config.version_pipeline

    for data_type in ("telemetry", "dsps"):
        fname = level05_h5_filename(config, data_type)
        sources = []
        for name in sorted(succeeded_names):
            p = os.path.join(test_data, name, "level0_5", fname)
            if os.path.isfile(p):
                sources.append((name, p))
        if not sources:
            print(f"[merge skip] no {data_type} files found under folders ({fname})")
            continue
        usable = []
        for folder_name, path in sources:
            try:
                _probe_readable_level05_h5(path)
            except OSError as e:
                print(
                    f"  [merge warn] unreadable HDF5 skipped ({folder_name}) {path}: {e}"
                )
                continue
            usable.append((folder_name, path))
        sources = usable
        if not sources:
            print(
                f"[merge skip] no readable {data_type} HDF5 remaining after probing ({fname})"
            )
            continue
        source_paths = [path for _, path in sources]
        for folder_name, path in sources:
            with h5py.File(path, "r") as f:
                nrow = f["timestamp_seconds_since_boot"].shape[0]
            print(f"  [merge {data_type}] {folder_name}: {nrow} rows from {path}")
        ts_out, pt_out, fields_out, n_in, removed = _merge_columnar_sources_vectorized(
            source_paths
        )
        out_path = os.path.join(merge_level05, fname)
        attrs_extra = {
            "merge_source_folders": ",".join(n for n, _ in sources),
            "num_source_h5_files": len(sources),
            "num_input_files_this_run": len(sources),
            "num_rows_before_dedup": n_in,
            "num_duplicate_rows_removed": removed,
        }
        write_columnar_level05_h5_vectorized(
            out_path, ts_out, pt_out, fields_out, data_type, version, attrs_extra
        )
        print(
            f"[merge ok] {data_type}: {len(ts_out)} rows -> {out_path} "
            f"(from {len(sources)} files, removed {removed} duplicates)"
        )

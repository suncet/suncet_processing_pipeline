#!/usr/bin/env python3
"""
Like ``read_em_bla_ingest_and_display_csie_em_image.py``, but first **scans the raw bytes** for
**consecutive identical 4-byte words**; each time two words match back-to-back, the duplicate
word is **removed** (4 bytes dropped), then the usual pipeline runs (``0xFF`` strip, sync split,
APID-536 …).

Ingests a raw **EM / CSIE** byte capture with CCSDS ``1A CF FC 1D`` frame sync markers between
space packets, strip initial ``0xFF`` filler, collect **APID 536 (csie_data)** row payloads, and
write **inferno** PNGs and optional reference diffs.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np

from suncet_processing_pipeline.data_paths import data_path


def _default_reference_fits():
    return data_path("test_data", "reference_test_pattern_96.fits")


def _default_input():
    return data_path(
        "test_data",
        "2026-05-08_em_xband_fixed_bluefin_firmware_downlink_test",
        "parsed_output_raw_csie_12p5mbps_4sps",
    )


SYNC_PATTERN = b"\x1a\xcf\xfc\x1d"
SYNC_LEN = 4
PRIMARY_LEN = 6
SECONDARY_LEN = 6
CHECKSUM_LEN = 4
CSIE_DATA_APID = 536


def strip_leading_0xff(data: bytes) -> bytes:
    i = 0
    n = len(data)
    while i < n and data[i] == 0xFF:
        i += 1
    return data[i:]


def remove_consecutive_repeated_4byte_words(data: bytes) -> bytes:
    """
    Sequentially scan ``data``; whenever ``data[i:i+4] == data[i+4:i+8]**, emit the first word
    only and advance by **8** bytes (drop the second copy). Otherwise emit one byte and advance
    by 1. This removes *consecutive* duplicate 32-bit words at any offset.
    """
    b = bytearray()
    i = 0
    n = len(data)
    while i < n:
        if i + 8 <= n and data[i : i + 4] == data[i + 4 : i + 8]:
            b.extend(data[i : i + 4])
            i += 8
        else:
            b.append(data[i])
            i += 1
    return bytes(b)


def split_by_sync(data: bytes) -> list[bytes]:
    """
    Return raw bytes for each inter-sync **frame** (region after one sync, before the next).

    A frame is typically one space packet, sometimes truncated a few bytes short of the
    full CCSDS length (e.g. missing checksum) in EM captures.
    """
    if not data:
        return []
    matches = [m.start() for m in re.finditer(re.escape(SYNC_PATTERN), data)]
    if not matches:
        return [data]
    out: list[bytes] = []
    for a, b in zip(matches, matches[1:] + [len(data)]):
        out.append(data[a + SYNC_LEN : b])
    return out


def parse_space_packet_header(packet: bytes) -> tuple[int, int, int] | None:
    """(apid, ccsds_sequence, data_field_length) or None if too short / invalid version."""
    if len(packet) < PRIMARY_LEN:
        return None
    if ((packet[0] >> 5) & 0x7) != 0:
        return None
    apid = int.from_bytes(packet[0:2], "big") & 0x7FF
    seq = int.from_bytes(packet[2:4], "big") & 0x3FFF
    dlen = int.from_bytes(packet[4:6], "big") + 1
    return apid, seq, dlen


def extract_csie_data_row_bytes(
    packet: bytes,
) -> tuple[int, int, bytes, int, bytes | None] | None:
    """
    From one CCSDS space packet, return
    (``image_id``, 1-based row index, **row** big-endian uint16 **wire** ``bytes``,
    ``n_cols``, **wire_checksum** or ``None``) for APID-536 (``csie_data``) rows, or None.

    The data field is: 6-byte secondary + ``n_cols``×2 row pixels + 4-byte **additive**
    checksum (not Fletcher-32). Pixels for assembly are taken **without** the checksum bytes;
    we still **do not** reject rows or drop packets based on checksum (no validation here).
    ``wire_checksum`` is returned for future **additive** checks if needed.

    The CSIE **secondary** is six bytes: U32 ``image_id`` (big endian), U16 (ICD).
    Row index is the CCSDS **sequence** field (1-based), matching ``make_level0_5``.
    """
    h = parse_space_packet_header(packet)
    if h is None or h[0] != CSIE_DATA_APID:
        return None
    _apid, seq, dlen = h
    if len(packet) < PRIMARY_LEN + SECONDARY_LEN:
        return None
    image_id = int.from_bytes(packet[PRIMARY_LEN : PRIMARY_LEN + 4], "big")
    p0 = PRIMARY_LEN + SECONDARY_LEN
    expected_data_len = dlen - SECONDARY_LEN - CHECKSUM_LEN
    if expected_data_len < 2 or (expected_data_len & 1):
        return None
    need_total = PRIMARY_LEN + dlen
    wire_checksum: bytes | None
    if len(packet) >= need_total:
        if len(packet) < p0 + expected_data_len + CHECKSUM_LEN:
            return None
        rowb = packet[p0 : p0 + expected_data_len]
        wire_checksum = bytes(
            packet[p0 + expected_data_len : p0 + expected_data_len + CHECKSUM_LEN]
        )
    else:
        raw = packet[p0:]
        n = len(raw)
        if n >= expected_data_len + CHECKSUM_LEN:
            rowb = raw[:expected_data_len]
            wire_checksum = bytes(
                raw[expected_data_len : expected_data_len + CHECKSUM_LEN]
            )
        elif n >= expected_data_len:
            rowb = raw[:expected_data_len]
            wire_checksum = None
        else:
            rowb = raw
            wire_checksum = None
        if not rowb or (len(rowb) & 1):
            return None
    if not rowb or (len(rowb) & 1):
        return None
    ncols = len(rowb) // 2
    return image_id, seq, rowb, ncols, wire_checksum


def row_bytes_to_u16be(row: bytes) -> np.ndarray:
    """Device send order is big-endian uint16; match ``make_level0_5`` (byteswap to native u16)."""
    return np.frombuffer(row, dtype=np.uint16).byteswap().copy()


def assemble_images(
    row_map: dict[int, dict[int, np.ndarray]],
) -> dict[int, np.ndarray]:
    """``row_map[image_id][1-based row seq] = 1D uint16 row`` → full images."""
    out: dict[int, np.ndarray] = {}
    for image_id, rows in row_map.items():
        if not rows:
            continue
        max_row = max(rows)
        n_rows = int(max_row)
        col_candidates = {int(r.size) for r in rows.values() if r.size}
        n_cols = max(col_candidates) if col_candidates else 0
        if n_rows < 1 or n_cols < 1:
            continue
        im = np.zeros((n_rows, n_cols), dtype=np.uint16)
        for r1, rvec in rows.items():
            r0 = int(r1) - 1
            if 0 <= r0 < n_rows:
                c = min(n_cols, int(rvec.size))
                im[r0, :c] = rvec[:c].reshape(1, c)
        out[image_id] = im
    return out


def preview_for_png(image: np.ndarray) -> np.ndarray:
    """Array in assembly order for ``imsave`` / ``imshow``; ``origin=upper`` in callers."""
    return np.asanyarray(image, dtype=np.uint16)


def _configure_matplotlib_for_show(show: bool) -> None:
    """``show``: interactive window backend; else headless ``Agg`` (must run before any pyplot)."""
    import matplotlib

    if not show:
        matplotlib.use("Agg", force=True)
    elif sys.platform == "darwin":
        matplotlib.use("macosx", force=True)
    else:
        matplotlib.use("TkAgg", force=True)


# Inferno clim for per-pixel difference (absolute) in float space
INFERNO_DIFF_VMAX: float = 2.0


def write_inferno_png(path: str, array_u16: np.ndarray, *, show: bool = False) -> None:
    import matplotlib.pyplot as plt

    preview = preview_for_png(array_u16)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    plt.imsave(path, preview, cmap="inferno", origin="upper")
    if show:
        fig, ax = plt.subplots(figsize=(8, 6))
        try:
            fig.canvas.manager.set_window_title(  # type: ignore[union-attr]
                f"CSIE: {os.path.basename(path)}"
            )
        except (AttributeError, TypeError):
            pass
        ax.imshow(preview, cmap="inferno", origin="upper")
        ax.set_axis_off()
    print(f"Wrote {path}  (shape {array_u16.shape[0]}×{array_u16.shape[1]})")


def show_interactive_top_row_zoom(
    array_u16: np.ndarray, *, label: str = "CSIE top row (y=0) · inferno vmin=0 vmax=2"
) -> None:
    """
    Only when ``--show``: wide, short figure with **one** row of the image, inferno, ``vmax=2``.
    """
    if array_u16.shape[0] < 1 or array_u16.shape[1] < 1:
        return
    import matplotlib.pyplot as plt

    row0 = array_u16[0:1, :].astype(np.float64)
    fig, ax = plt.subplots(figsize=(16, 2.2), dpi=100)
    fig.patch.set_facecolor("black")
    im = ax.imshow(
        row0, cmap="inferno", origin="upper", aspect="auto", vmin=0.0, vmax=INFERNO_DIFF_VMAX
    )
    ax.set_title(label, color="white", fontsize=10)
    ax.set_axis_off()
    try:
        fig.canvas.manager.set_window_title(  # type: ignore[union-attr]
            "CSIE: top row zoom (inferno 0..2)"
        )
    except (AttributeError, TypeError):
        pass
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.ax.tick_params(labelsize=7, colors="0.85")


def load_reference_fits_array(path: str) -> np.ndarray:
    from astropy.io import fits

    p = os.path.expanduser(path)
    data = fits.getdata(p)
    if data is None:
        raise SystemExit(f"No data array in FITS: {p}")
    a = np.asanyarray(data)
    if a.itemsize != 2 or a.dtype.kind not in "iu":
        raise SystemExit(
            f"Reference FITS must be 16-bit integer for CSIE diff; got {a.dtype!r} in {p}"
        )
    # Match parsed CSIE (uint16): re-interpret bit patterns as unsigned (FITS is often int16)
    return np.asanyarray(a, dtype=np.uint16)


def overlap_parsed_ref_diff(
    parsed: np.ndarray, ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    """
    Top-left ``min`` overlap in each axis: ``(parsed_crop, ref_crop, diff, note|None)``.

    ``diff = parsed_crop − ref_crop`` in float64 (handles e.g. a missing row in the capture).
    """
    h = min(int(parsed.shape[0]), int(ref.shape[0]))
    w = min(int(parsed.shape[1]), int(ref.shape[1]))
    if h < 1 or w < 1:
        raise SystemExit(
            f"No overlap: parsed {parsed.shape} vs reference FITS {ref.shape}"
        )
    p_sub = parsed[:h, :w]
    r_sub = ref[:h, :w]
    diff = p_sub.astype(np.float64) - r_sub.astype(np.float64)
    if tuple(parsed.shape) == tuple(ref.shape):
        return p_sub, r_sub, diff, None
    return p_sub, r_sub, diff, (
        f"note: top-left {h}×{w} overlap (parsed {parsed.shape[0]}×{parsed.shape[1]}"
        f" vs FITS {ref.shape[0]}×{ref.shape[1]})"
    )


def print_and_format_diff_stats(
    diff: np.ndarray, overlap_note: str | None = None
) -> str:
    """Log statistics to stdout; return the same text for the figure annotation."""
    d = diff.ravel()
    n = d.size
    lines = [
        "Difference: parsed - reference (FITS)",
    ]
    if overlap_note:
        lines.append(overlap_note)
    lines.extend(
        [
            f"shape: {diff.shape[0]} x {diff.shape[1]}  (n={n})",
            f"min:   {float(np.min(d)):.6g}",
            f"max:   {float(np.max(d)):.6g}",
            f"mean:  {float(np.mean(d)):.6g}",
            f"std:   {float(np.std(d)):.6g}",
            f"rmse:  {float(np.sqrt(np.mean(d**2))):.6g}",
            f"max|d|: {float(np.max(np.abs(d))):.6g}",
            f"count (|d|>0): {int(np.count_nonzero(d))} / {n}",
        ]
    )
    print("\n=== Reference FITS difference ===")
    for line in lines:
        print(line)
    return "\n".join(lines)


def write_diff_inferno_png(
    out_path: str, diff: np.ndarray, annotation: str, *, show: bool = False
) -> None:
    import matplotlib.pyplot as plt

    vis = np.asanyarray(diff, dtype=np.float64)
    ad = np.abs(vis)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5), dpi=120)
    dmax = float(np.max(ad))
    if dmax == 0.0:
        ax.imshow(vis, cmap="inferno", origin="upper", vmin=0.0, vmax=1.0)
    else:
        ax.imshow(
            ad,
            cmap="inferno",
            origin="upper",
            vmin=0.0,
            vmax=INFERNO_DIFF_VMAX,
        )
    ax.set_axis_off()
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.65),
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.patch.set_facecolor("black")
    fig.savefig(out_path, bbox_inches="tight", facecolor="black", pad_inches=0.1)
    if not show:
        plt.close(fig)
    print(f"Wrote difference PNG {out_path}  (shape {diff.shape[0]}×{diff.shape[1]})")


def write_parsed_ref_diff_panel_png(
    out_path: str,
    p_sub: np.ndarray,
    r_sub: np.ndarray,
    diff: np.ndarray,
    annotation: str,
    *,
    reference_title: str = "Reference (FITS)",
    diff_title: str = "Difference (parsed − ref)",
    show: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    p_vis = np.asanyarray(p_sub, dtype=np.float64)
    r_vis = np.asanyarray(r_sub, dtype=np.float64)
    d_vis = np.asanyarray(diff, dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=120)
    fig.patch.set_facecolor("black")
    fig.subplots_adjust(bottom=0.2, wspace=0.08, top=0.9)
    for ax, im, title, is_diff in (
        (axes[0], p_vis, "Parsed (CSIE)", False),
        (axes[1], r_vis, reference_title, False),
        (axes[2], d_vis, diff_title, True),
    ):
        if is_diff:
            ad = np.abs(np.asanyarray(im, dtype=np.float64))
            dmax = float(np.max(ad))
            if dmax == 0.0:
                ax.imshow(ad, cmap="inferno", origin="upper", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(
                    ad,
                    cmap="inferno",
                    origin="upper",
                    vmin=0.0,
                    vmax=INFERNO_DIFF_VMAX,
                )
        else:
            ax.imshow(
                im,
                cmap="inferno",
                origin="upper",
                vmin=0.0,
                vmax=65535.0,
            )
        ax.set_title(title, color="white", fontsize=11)
        ax.set_axis_off()
    fig.text(
        0.5,
        0.01,
        annotation,
        transform=fig.transFigure,
        ha="center",
        va="bottom",
        fontsize=6,
        color="0.85",
        family="monospace",
        linespacing=1.15,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="black", pad_inches=0.2)
    if not show:
        plt.close(fig)
    print(f"Wrote parsed/ref/diff panel PNG {out_path}  (shape {diff.shape[0]}×{diff.shape[1]})")


def emit_reference_diff(
    main_png_path: str, arr: np.ndarray, ref: np.ndarray, *, show: bool = False
) -> None:
    p_sub, r_sub, diff, overlap_note = overlap_parsed_ref_diff(arr, ref)
    ann = print_and_format_diff_stats(diff, overlap_note)
    base, ext = os.path.splitext(main_png_path)
    diff_path = f"{base}_diff{ext}"
    write_diff_inferno_png(diff_path, diff, ann, show=show)
    triple_path = f"{base}_parsed_ref_diff{ext}"
    write_parsed_ref_diff_panel_png(triple_path, p_sub, r_sub, diff, ann, show=show)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Remove consecutive duplicate 4-byte words, then same CSIE parse as read_em"
    )
    p.add_argument(
        "input",
        nargs="?",
        default=_default_input(),
        help="Raw binary capture (default: EM test file path)",
    )
    p.add_argument(
        "-o",
        "--output",
        help="Output PNG path (default: <input_basename>_csie.png next to the input file)",
    )
    p.add_argument(
        "--no-reference",
        action="store_true",
        help="Do not load reference FITS, difference stats, _diff.png, or parsed/ref panel PNGs",
    )
    p.add_argument(
        "--reference-fits",
        default=_default_reference_fits(),
        metavar="PATH",
        help="FITS test pattern to subtract (default: $suncet_data/test_data/reference_test_pattern_96.fits)",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Open Matplotlib window(s) for the CSIE, diff, and triptych figures (keeps run alive until you close all)",
    )
    p.add_argument(
        "--no-dedup",
        action="store_true",
        help="Skip consecutive duplicate 4-byte word removal (same bytes as input after 0xFF strip)",
    )
    args = p.parse_args()
    _configure_matplotlib_for_show(args.show)

    in_path = os.path.expanduser(args.input)
    with open(in_path, "rb") as f:
        data = f.read()
    n0 = len(data)
    if args.no_dedup:
        print("Skipping consecutive 4-byte dedup (--no-dedup).\n")
    else:
        data = remove_consecutive_repeated_4byte_words(data)
        n1 = len(data)
        if n1 != n0:
            print(
                f"Consecutive 4-byte dedup: {n0} -> {n1} bytes ({n0 - n1} byte(s) removed)\n"
            )
    data = strip_leading_0xff(data)
    frames = split_by_sync(data)

    row_by_image: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    n_seen = 0
    n_skip = 0
    for fr in frames:
        ext = extract_csie_data_row_bytes(fr)
        if ext is None:
            n_skip += 1
            continue
        image_id, seq, rowb, _n_cols, _row_checksum = ext
        n_seen += 1
        rvec = row_bytes_to_u16be(rowb)
        row_by_image[image_id][seq] = rvec.ravel()

    if not any(row_by_image.values()):
        raise SystemExit(
            "No APID-536 (csie_data) rows were recovered. Check sync framing / file contents."
        )

    images = assemble_images(row_by_image)
    if not images:
        raise SystemExit("Could not assemble any images from the parsed rows.")

    in_base, _ = os.path.splitext(in_path)
    n_img = len(images)
    ref: np.ndarray | None = None
    if not args.no_reference:
        ref = load_reference_fits_array(args.reference_fits)
    if n_img == 1:
        image_id, arr = next(iter(images.items()))
        if args.output:
            op = os.path.expanduser(args.output)
            if os.path.isdir(op):
                op = os.path.join(op, f"csie_id{image_id}.png")
        else:
            op = f"{in_base}_csie.png"
        write_inferno_png(op, arr, show=args.show)
        if args.show:
            show_interactive_top_row_zoom(arr, label="CSIE top row (y=0) · inferno [0, 2]")
        if ref is not None:
            emit_reference_diff(op, arr, ref, show=args.show)
    else:
        for image_id, arr in images.items():
            if args.output:
                op = os.path.expanduser(args.output)
                if os.path.isdir(op):
                    pth = os.path.join(op, f"csie_id{image_id}.png")
                else:
                    stem, ext = os.path.splitext(op)
                    if ext.lower() == ".png":
                        pth = f"{stem}_id{image_id}{ext}"
                    else:
                        pth = f"{in_base}_csie_id{image_id}.png"
            else:
                pth = f"{in_base}_csie_id{image_id}.png"
            write_inferno_png(pth, arr, show=args.show)
            if args.show:
                show_interactive_top_row_zoom(
                    arr, label=f"CSIE id={image_id} top row (y=0) · inferno [0, 2]"
                )
            if ref is not None:
                emit_reference_diff(pth, arr, ref, show=args.show)

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()

    if n_skip:
        print(
            f"Note: {n_skip} inter-sync frame(s) did not parse as complete APID-536 "
            f"(filler, metadata APID-538, or other)."
        )
    print(f"Decoded {n_seen} csie_data row packet(s) into {len(images)} image(s).")


if __name__ == "__main__":
    main()

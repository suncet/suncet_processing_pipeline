import json
import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from suncet_processing_pipeline.make_level0_5 import Level0_5
from suncet_processing_pipeline.config_parser import Config

# Full-resolution CSIE image dimensions
FULL_ROWS = 2000
FULL_COLS = 1504

# Binning factor: 1 = no binning, 2/4/... = image has been spatially binned
BINNING_FACTOR = 1

# Multiplot: only show panels for csie_data image_id in this inclusive range
DISPLAY_IMAGE_ID_MIN = 0
DISPLAY_IMAGE_ID_MAX = 3000


def _load_config():
    config_path = os.path.join(
        os.path.dirname(__file__),
        "suncet_processing_pipeline",
        "config_files",
        "config_default.ini",
    )
    return Config(config_path)


def _resolve_data_folder(config):
    data_path = config.data_to_process_path

    if data_path.startswith("/") or data_path.startswith("~"):
        folder = os.path.expanduser(data_path)
    else:
        folder = os.path.join(os.getenv("suncet_data", ""), data_path)

    return folder


def _build_file_list(folder):
    if not os.path.isdir(folder):
        return []
    files = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.startswith("hardline_playback") or name.startswith("pbk_") and os.path.isfile(os.path.join(folder, name))
    ]
    # files.sort()
    return files

def _build_level0_5(file_paths, config):
    return Level0_5(
        file_paths,
        config.packet_definitions_path,
        config.bus_ctdb_path,
        config.csie_ctdb_path,
        output_base_folder=_resolve_data_folder(config),
        processing_config=config,
    )


def _detect_source(filename, level0_5):
    from_hydra_realtime = filename.startswith("ccsds_")
    from_xband_gse = filename.startswith("suncet_")
    from_xband_flight = (
        level0_5.XBAND_FLIGHT_FILENAME_RE.match(filename) is not None
    )
    from_uhf_playback = filename.startswith("pbk_")
    from_hardline_playback = filename.startswith("hardline_playback_")

    if from_hydra_realtime:
        return "hydra_realtime"
    if from_xband_gse:
        return "xband_gse"
    if from_xband_flight:
        return "xband_flight"
    if from_uhf_playback:
        return "uhf_playback"
    if from_hardline_playback:
        return "hardline_playback"
    return "csie"


# CCSDS primary header constants for packet sync
_PRIMARY_HDR_LEN = 6


def _is_valid_ccsds_header(data, offset, valid_apids=None):
    """
    Return True if the 6 bytes at data[offset:offset+6] form a valid CCSDS primary header:
    - Version number (first 3 bits of first byte) must be 000.
    - If valid_apids is provided (set/frozenset of int), APID (11 bits) must be in that set.
    - Length L in bytes 4-5 (big-endian); total packet size = L + 7 (must fit in buffer).
    """
    if offset + _PRIMARY_HDR_LEN > len(data):
        return False, 0
    h = data[offset : offset + _PRIMARY_HDR_LEN]
    version = (h[0] >> 5) & 0x07
    if version != 0:
        return False, 0
    apid = int.from_bytes(h[0:2], "big") & 0x7FF
    if valid_apids is not None and apid not in valid_apids:
        return False, 0
    L = int.from_bytes(h[4:6], "big")
    packet_len = L + 7
    if packet_len < _PRIMARY_HDR_LEN or offset + packet_len > len(data):
        return False, 0
    return True, packet_len


def strip_uhf_segmentation_header(packet):
    """
    For APID 73 (UHF segmented) packets, strip off:
    - The first 6 bytes: original CCSDS primary header
    - The next 6 bytes: UHF segmentation header
        PAY_APID   (2 bytes)
        PAY_SEQ_CNT(2 bytes)
        SEG_COUNT  (1 byte)
        SEG_FLAGS  (1 byte)

    Returns:
        (pay_apid, seg_flags, payload_bytes)
    """
    if len(packet) < _PRIMARY_HDR_LEN + 6:
        return None, None, b""

    # PAY_APID is stored in the first 2 bytes after the primary header
    pay_apid_offset = _PRIMARY_HDR_LEN
    pay_apid = int.from_bytes(packet[pay_apid_offset : pay_apid_offset + 2], "big")

    # SEG_FLAGS is the last byte of the 6-byte segmentation header
    seg_flags_offset = _PRIMARY_HDR_LEN + 5
    seg_flags = packet[seg_flags_offset]

    payload_start = _PRIMARY_HDR_LEN + 6
    payload = packet[payload_start:]
    return pay_apid, seg_flags, payload


def extract_ccsds_packets_unaligned(concatenated_bytes, valid_apids=None):
    """
    Extract CCSDS Space Packets from a raw byte array that may not be aligned to
    a packet boundary. Sliding window: at each position, if the 6 bytes form a
    valid primary header (and optional valid_apids), extract that packet and
    advance by packet length; otherwise advance by 1 byte. No sync-word or
    deep-sync assumption.

    valid_apids: optional set/frozenset of int APIDs; if provided, only headers
        with APID in this set are considered valid.

    Returns:
        list[bytes]: One element per extracted packet (full packet including 6-byte header).
    """
    data = concatenated_bytes
    packets = []
    offset = 0
    n = len(data)

    while offset + _PRIMARY_HDR_LEN <= n:
        ok, packet_len = _is_valid_ccsds_header(data, offset, valid_apids=valid_apids)
        if ok:
            packet = bytes(data[offset : offset + packet_len])
            packets.append(packet)
            offset += packet_len
        else:
            offset += 1
    return packets


def _csie_meta_to_plain_dict(meta_packet):
    """Export CSIE_META decoder instance to JSON-serializable fields."""
    out = {}
    for name in sorted(dir(meta_packet)):
        if name.startswith("_"):
            continue
        val = getattr(meta_packet, name)
        if callable(val):
            continue
        if isinstance(val, (bool, str)) or val is None:
            out[name] = val
        elif isinstance(val, (int, float)):
            out[name] = val
        elif isinstance(val, np.integer):
            out[name] = int(val)
        elif isinstance(val, np.floating):
            out[name] = float(val)
        elif isinstance(val, np.ndarray):
            out[name] = val.tolist()
        elif isinstance(val, (bytes, bytearray)):
            out[name] = val.hex()
        else:
            out[name] = repr(val)
    return out


def main():
    config = _load_config()
    folder = _resolve_data_folder(config)

    print(f"Using data folder: {folder}")
    file_paths = _build_file_list(folder)
    if not file_paths:
        print("No input files found in folder.")
        return

    level0_5 = _build_level0_5(file_paths, config)

    playback_chunks = []
    total_packets = 0
    playback_packets = 0
    segmented_uhf_buffer = b""

    for path in file_paths:
        filename = os.path.basename(path)
        source = _detect_source(filename, level0_5)

        packets = level0_5.extract_packets(path, source=source)
        if not packets:
            continue

        total_packets += len(packets)

        for packet in packets:
            apid, length, sequence_number, header = level0_5.parse_header(packet)

            # UHF segmented packets (APID 73): strip segmentation header and
            # reassemble full inner CCSDS packet payload across segments.
            if apid == 73:
                pay_apid, seg_flags, payload = strip_uhf_segmentation_header(packet)
                if seg_flags is None:
                    continue

                # SEG_FLAGS: 1 = Start, 0 = Middle, 2 = End of packet
                if seg_flags == 1:
                    # Start of a new segmented packet
                    segmented_uhf_buffer = payload
                elif seg_flags == 0:
                    # Middle of packet: append payload
                    segmented_uhf_buffer += payload
                elif seg_flags == 2:
                    # End of packet: append final payload and process assembled
                    segmented_uhf_buffer += payload
                    reconstructed = segmented_uhf_buffer
                    segmented_uhf_buffer = b""

                    # Now process the reconstructed inner CCSDS packet as usual
                    inner_apid, _, _, _ = level0_5.parse_header(reconstructed)
                    if inner_apid in (68, 72):
                        playback_packets += 1
                        playback_chunks.append(
                            reconstructed[
                                level0_5.PRIMARY_HDR_LEN
                                + level0_5.PLAYBACK_HEADER_LEN :
                            ]
                        )
                continue

            # Regular playback packets (APID 68 or 72)
            if apid == 68 or apid == 72:
                playback_packets += 1
                playback_chunks.append(
                    packet[level0_5.PRIMARY_HDR_LEN + level0_5.PLAYBACK_HEADER_LEN :]
                )

    if not playback_chunks:
        print("No APID 68 or 72 or 73playback packets found.")
        return

    concatenated_bytes = b"".join(playback_chunks)

    # Extract inner CCSDS packets and run Level0_5 csie_data extraction (same as make_level0_5)
    valid_apids = set(level0_5.apid_df["APID"].astype(int))
    apid_538_rows = level0_5.apid_df[level0_5.apid_df["APID"] == 538]
    if len(apid_538_rows):
        name_538 = apid_538_rows["Name"].values[0]
        print(
            f"CTDB: APID 538 -> Name={name_538!r} "
            f"(decoded with CSIE_META when Name is 'csie_meta', same as make_level0_5.process_packet)"
        )
    else:
        print(
            "Warning: APID 538 not in merged ct_pkt CSV — inner metadata packets "
            "will not be recognized unless you add the CSIE packet table path / version."
        )

    inner_packets = extract_ccsds_packets_unaligned(
        concatenated_bytes, valid_apids=valid_apids
    )
    inner_apid_counts = Counter(
        int.from_bytes(p[0:2], "big") & 0x7FF for p in inner_packets
    )
    print(
        "Inner playback stream: "
        + ", ".join(f"APID {k}: {v}" for k, v in sorted(inner_apid_counts.items()))
    )

    metadata_dict = {}
    data_dict = {}
    telemetry_dict = {}
    dsps_dict = {}

    for pkt in inner_packets:
        apid = int.from_bytes(pkt[0:2], "big") & 0x7FF
        names = level0_5.apid_df[level0_5.apid_df["APID"] == apid]["Name"].values
        filename = names[0] if len(names) else str(apid)
        level0_5.process_packet(
            pkt,
            metadata_dict,
            data_dict,
            telemetry_dict,
            dsps_dict,
            filename=filename,
        )

    if metadata_dict:
        print(f"Decoded CSIE metadata (csie_meta / e.g. APID 538): {len(metadata_dict)} image_id(s)")
        for img_id, meta in sorted(metadata_dict.items()):
            fields = _csie_meta_to_plain_dict(meta)
            brief = {
                k: fields[k]
                for k in (
                    "csie_meta_img_id",
                    "csie_meta_icm_proc_config",
                    "csie_meta_fpm_proc_config",
                    "csie_meta_fpm_row_per_frame",
                    "csie_meta_fpm_pix_per_row",
                )
                if k in fields
            }
            print(f"  image_id={img_id}: {brief}")
    else:
        print(
            "No CSIE metadata packets decoded in this playback stream "
            "(metadata_dict empty — expect csie_meta inner packets if APID 538 is present)."
        )

    rows = FULL_ROWS // BINNING_FACTOR
    cols = FULL_COLS // BINNING_FACTOR
    expected_size = rows * cols

    def _prepare_csie_image(raw):
        # Returns (2d_array, n_valid): source pixel count in raster order (excludes pad).
        if isinstance(raw, bytes):
            arr = level0_5.decompress_jpegls_image(raw)
        else:
            arr = np.asarray(raw, dtype=np.uint16)
        n_raw = arr.size
        if n_raw != expected_size:
            print(
                f"Warning: csie_data image size {n_raw} != {rows}*{cols}; reshaping to fit."
            )
            if n_raw < expected_size:
                padded = np.zeros(expected_size, dtype=np.uint16)
                padded[:n_raw] = arr.ravel()
                return padded.reshape((rows, cols)), n_raw
            return arr.ravel()[:expected_size].reshape((rows, cols)), expected_size
        return arr.reshape((rows, cols)), expected_size

    def _pixels_for_stats(arr, n_valid):
        flat = arr.ravel()
        n = int(min(max(n_valid, 0), flat.size))
        return flat[:n]

    # Prefer image(s) assembled from csie_data packets
    image_arrays_by_id = {}
    image_valid_pixel_count_by_id = {}
    if data_dict:
        for image_id in sorted(data_dict.keys()):
            img, n_valid = _prepare_csie_image(data_dict[image_id])
            image_arrays_by_id[image_id] = img
            image_valid_pixel_count_by_id[image_id] = n_valid
            print(
                f"Using csie_data image (image_id={image_id}) from {len(inner_packets)} inner packets."
            )
    else:
        # Fallback: treat concatenated payload as raw uint16 image
        concatenated = np.frombuffer(concatenated_bytes, dtype=np.uint16)
        n_cat = concatenated.size
        if n_cat != expected_size:
            print(
                f"Warning: Concatenated stream is {n_cat} elements, does not match {rows} x {cols}. Truncating or padding."
            )
            if n_cat < expected_size:
                padded = np.zeros(expected_size, dtype=np.uint16)
                padded[:n_cat] = concatenated
                image_arr = padded.reshape((rows, cols))
                n_valid = n_cat
            else:
                image_arr = concatenated[:expected_size].reshape((rows, cols))
                n_valid = expected_size
        else:
            image_arr = concatenated.reshape((rows, cols))
            n_valid = expected_size
        print("No csie_data packets found; using raw playback payload as image.")
        image_arrays_by_id[None] = image_arr
        image_valid_pixel_count_by_id[None] = n_valid

    print(f"Total packets processed: {total_packets}")
    print(f"Playback (APID 68 or 72) packets: {playback_packets}")
    print(f"Concatenated playback stream length: {len(concatenated_bytes)} bytes")

    image_arrays_for_plot = {
        k: v
        for k, v in image_arrays_by_id.items()
        if k is not None
        and DISPLAY_IMAGE_ID_MIN <= int(k) <= DISPLAY_IMAGE_ID_MAX
    }

    if not image_arrays_for_plot:
        print(
            f"No images with image_id in [{DISPLAY_IMAGE_ID_MIN}, {DISPLAY_IMAGE_ID_MAX}] "
            "(multiplot skipped)."
        )
    else:
        n_panes = len(image_arrays_for_plot)
        ncols = int(np.ceil(np.sqrt(n_panes)))
        nrows = int(np.ceil(n_panes / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
        )
        ax_flat = axes.ravel()

        global_vmax = float(max(np.max(arr) for arr in image_arrays_for_plot.values()))
        global_vmin = float(min(np.min(arr) for arr in image_arrays_for_plot.values()))
        # Minimum separation between vmin and vmax (uint16-friendly)
        _clim_eps = (
            1.0
            if global_vmax - global_vmin >= 1.0
            else max(np.finfo(float).eps, (global_vmax - global_vmin) * 0.5)
        )

        debug_plot = {
            "fig": fig,
            "axes": [],
            "images": [],
            "colorbars": [],
            "image_ids": [],
        }

        for i, (image_id, arr) in enumerate(image_arrays_for_plot.items()):
            ax = ax_flat[i]
            n_valid = image_valid_pixel_count_by_id.get(image_id, arr.size)
            stats_px = _pixels_for_stats(arr, n_valid)
            mean_val = float(np.mean(stats_px)) if stats_px.size else float("nan")
            median_val = (
                float(np.median(stats_px)) if stats_px.size else float("nan")
            )
            ax.set_title(
                f"image_id={image_id}, μ={mean_val:.4g}, median={median_val:.4g}"
            )
            im = ax.imshow(
                arr,
                cmap="inferno",
                origin="upper",
                vmin=global_vmin,
                vmax=global_vmax,
            )
            cb = fig.colorbar(im, ax=ax, label="DN")
            debug_plot["axes"].append(ax)
            debug_plot["images"].append(im)
            debug_plot["colorbars"].append(cb)
            debug_plot["image_ids"].append(image_id)

        for j in range(n_panes, len(ax_flat)):
            ax_flat[j].set_visible(False)

        def _apply_vmin_vmax():
            vmin = float(vmin_slider.val)
            vmax = float(vmax_slider.val)
            if vmin >= vmax:
                vmin = vmax - _clim_eps
                if vmin < global_vmin:
                    vmin = global_vmin
                    vmax = min(global_vmax, vmin + _clim_eps)
                vmin_slider.eventson = False
                vmin_slider.set_val(vmin)
                vmin_slider.eventson = True
                vmax_slider.eventson = False
                vmax_slider.set_val(vmax)
                vmax_slider.eventson = True
            for im, cb in zip(debug_plot["images"], debug_plot["colorbars"]):
                im.set_clim(vmin, vmax)
                cb.update_normal(im)
            fig.canvas.draw_idle()

        def _on_vmin_vmax_change(_val):
            _apply_vmin_vmax()

        if global_vmax > global_vmin:
            ax_vmin = fig.add_axes((0.12, 0.02, 0.76, 0.028))
            ax_vmax = fig.add_axes((0.12, 0.052, 0.76, 0.028))
            vmin_slider = Slider(
                ax_vmin,
                "vmin",
                global_vmin,
                global_vmax,
                valinit=global_vmin,
                valstep=None,
            )
            vmax_slider = Slider(
                ax_vmax,
                "vmax",
                global_vmin,
                global_vmax,
                valinit=global_vmax,
                valstep=None,
            )
            vmin_slider.on_changed(_on_vmin_vmax_change)
            vmax_slider.on_changed(_on_vmin_vmax_change)
            fig.tight_layout(rect=(0, 0.11, 1, 1))
        else:
            fig.tight_layout()
        # Breakpoint: `debug_plot` — fig, axes, images, colorbars, image_ids (same order).
        plt.show()

    for image_id, arr in image_arrays_by_id.items():
        suffix = "raw" if image_id is None else str(image_id)
        png_out_path = os.path.join(folder, f"apid68_playback_stream_{suffix}.png")
        plt.imsave(png_out_path, arr, cmap="inferno", origin="upper")
        print(f"Saved rendered image as PNG to: {png_out_path}")

        out_path = os.path.join(folder, f"apid68_playback_stream_{suffix}.npy")
        np.save(out_path, arr)
        print(f"Saved image array to: {out_path}")

    if metadata_dict:
        meta_all = os.path.join(folder, "apid68_playback_csie_metadata_all.json")
        with open(meta_all, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    str(k): _csie_meta_to_plain_dict(v)
                    for k, v in sorted(metadata_dict.items())
                },
                fh,
                indent=2,
            )
        print(f"Saved all decoded CSIE metadata to: {meta_all}")


if __name__ == "__main__":
    main()


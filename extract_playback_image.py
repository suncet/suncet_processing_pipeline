import os
import configparser
import numpy as np
import matplotlib.pyplot as plt

from suncet_processing_pipeline.make_level0_5 import Level0_5, _version_to_path_format


def _load_config():
    config = configparser.ConfigParser()
    config_path = os.path.join(
        os.path.dirname(__file__),
        "suncet_processing_pipeline",
        "config_files",
        "config_default.ini",
    )
    config.read(config_path)
    return config


def _resolve_data_folder(config):
    data_path = config.get("paths", "data_to_process_path")

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
        if name.startswith("hardline_playback") and os.path.isfile(os.path.join(folder, name))
    ]
    files.sort()
    return files

def _build_level0_5(file_paths, config):
    version = config["structure"]["version"]
    version_path = _version_to_path_format(version)
    ctdb_base = os.path.expanduser(
        "~/Library/CloudStorage/Box-Box/SunCET Private/suncet_ctdb"
    )
    packet_definitions_path = os.path.join(
        ctdb_base, f"suncet_{version_path}", f"suncet_{version_path}_decoders"
    )
    return Level0_5(file_paths, packet_definitions_path)


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

    for path in file_paths:
        filename = os.path.basename(path)
        source = _detect_source(filename, level0_5)

        packets = level0_5.extract_packets(path, source=source)
        if not packets:
            continue

        total_packets += len(packets)

        for packet in packets:
            apid, length, sequence_number, header = level0_5.parse_header(packet)
            if apid == 68 or apid == 72:
                playback_packets += 1
                playback_chunks.append(packet[level0_5.PRIMARY_HDR_LEN :])

    if not playback_chunks:
        print("No APID 68 or 72 playback packets found.")
        return

    concatenated = np.frombuffer(b"".join(playback_chunks), dtype=np.uint16)

    # Reform the concatenated array into a 2000 x 1504 image
    if concatenated.size != 2000 * 1504:
        print(f"Warning: Concatenated stream is {concatenated.size} bytes, does not match 2000 x 1504 image size ({2000 * 1504} bytes). The output will be truncated or padded.")
        # If too short, pad with zeros; if too long, truncate
        if concatenated.size < 2000 * 1504:
            padded = np.zeros(2000 * 1504, dtype=np.uint16)
            padded[:concatenated.size] = concatenated
            image_arr = padded.reshape((2000, 1504))
        else:
            image_arr = concatenated[:2000 * 1504].reshape((2000, 1504))
    else:
        image_arr = concatenated.reshape((2000, 1504))


    print(f"Total packets processed: {total_packets}")    
    print(f"Playback (APID 68 or 72) packets: {playback_packets}")
    print(f"Concatenated playback stream length: {concatenated.size} bytes")

    plt.figure(figsize=(10, 8))
    plt.imshow(image_arr, cmap='inferno', origin='upper')
    plt.title("Playback Image (APID 68/72)")
    plt.colorbar(label='Pixel Value')
    plt.tight_layout()
    plt.show()

    png_out_path = os.path.join(folder, "apid68_playback_stream.png")
    plt.imsave(png_out_path, image_arr, cmap='inferno', origin='upper')
    print(f"Saved rendered image as PNG to: {png_out_path}")
    

    out_path = os.path.join(folder, "apid68_playback_stream.npy")
    #np.save(out_path, concatenated)
    print(f"Saved concatenated playback stream to: {out_path}")


if __name__ == "__main__":
    main()


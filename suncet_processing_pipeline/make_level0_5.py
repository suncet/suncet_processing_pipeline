"""
This is the code to make the Level 0.5 data product. 
We use 0_5 in the filename to prevent issues with some operating systems not being able to handle a period in a filename
"""
import gen_pkts  # Import the gen_pkts module from the same directory
from gen_pkts import CSIE_META

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import re
import psutil
import os
import time  # Added for time measurement
from datetime import datetime

#######################################################################

# Plot images
PLOT = False

# Default testMode value if metadata is not attached to image
DEFAULT_TESTMODE = 96

# Path to image binary(s)
ABSOLUTE_FILE_PATH = [
    "/Users/masonjp2/Downloads/2025-02-13 sample cubixss data/CSIE_GSE_raw_219.bin"
]

#######################################################################

# CSIE HW/SW configuration values. Do not edit unless you know why
CSIE_ROWS = 2000
CSIE_COLS = 1504
PRIMARY_HDR_LEN = 6
SECONDARY_HDR_LEN = 6
SYNC_MARKER = [b'\x1A', b'\xCF', b'\xFC', b'\x1D']
CSIE_META_APID = [26]
CSIE_DATA_APID = [24]

# Globals to allow external access and multiple test modes
image_arrays = []
figlist = []


def has_sufficient_memory(file_path):
    """Check if there is sufficient memory to load the file into memory."""
    available_memory = psutil.virtual_memory().available
    file_size = os.path.getsize(file_path)
    return file_size <= available_memory


def parse_header(packet):
    """Extract APID, length, and sequence number from a packet."""
    # Standard CCSDS primary header assumed
    header_data = bytearray(packet[:6])

    apid = int.from_bytes(header_data[0:2], 'big') & 0x7FF
    seqnum = int.from_bytes(header_data[2:4], 'big') & 0x3FFF
    length = int.from_bytes(header_data[4:6], 'big')

    length += 1
        
    return apid, length, seqnum, header_data


def parse_secondary_header(packet):
    """Extract image ID and packet number from the secondary header."""
    secondary_header = packet[PRIMARY_HDR_LEN:PRIMARY_HDR_LEN + SECONDARY_HDR_LEN]
    image_id = int.from_bytes(secondary_header[0:4], 'big')
    packet_num = int.from_bytes(secondary_header[4:6], 'big')
    return image_id, packet_num


def parse_csie_meta(packet, header):
    """Extract metadata packet into a class using autogen constructor."""
    return CSIE_META(packet[6:], header, "HSSI BIN") # Note: CSIE_META expects the header to not be included as part of the packet so this [6:] gets rid of it
    

def get_image_meta(meta_packet):
    row_bin = (meta_packet.csie_meta_fpm_proc_config & 0xFF) + 1
    col_bin = ((meta_packet.csie_meta_fpm_proc_config & 0xFF00) >> 8) + 1
    roi_rows = meta_packet.csie_meta_fpm_row_per_frame
    roi_cols = meta_packet.csie_meta_fpm_pix_per_row
    row_offset = meta_packet.csie_meta_fpm_row_ptr_offset
    col_offset = (meta_packet.csie_meta_fpm_col_chan_offset * 376 +
                  meta_packet.csie_meta_fpm_col_pix_offset)
    rows = roi_rows / row_bin
    cols = roi_cols / col_bin
    return int(rows), int(cols), int(roi_rows), int(roi_cols), int(row_offset), int(col_offset)


def parse_csie_data(packet):
    """Extract raw data from an extracted packet, first packet contains reduced metadata."""
    checksum_size = 4
    image_id, packet_num = parse_secondary_header(packet)

    data_array = []
    # Calculate the total header size (primary + secondary)
    total_header_size = PRIMARY_HDR_LEN + SECONDARY_HDR_LEN
    
    if len(packet) >= (total_header_size + checksum_size):
        # Extract data starting after both headers and ending before checksum
        data_array_raw = packet[total_header_size:len(packet) - checksum_size]
        
        # Convert bytes to 16-bit integers (2 bytes per value)
        data_array = [int.from_bytes(data_array_raw[i:i + 2], 'big')
                      for i in range(0, len(data_array_raw), 2)]
        
        # Checksum is not used, but can be extracted if needed
        # checksum = packet[len(packet) - checksum_size:]
    else:
        print(f'Read Length Error length: {len(packet)}, packet too short')

    return packet_num, data_array, image_id


def process_csie_data(data_list, rows, cols):
    """Reconcile data into a rectangular np array."""
    flat_array = []
    for row_data in data_list:
        flat_array.extend(row_data)
    img_array = np.array(flat_array).reshape(rows, cols)
    return img_array


def plot_csie_array(data_array, meta, image_num, vmin=4300, vmax=5000, cmap='gray'):
    """Plot data with enhanced contrast to highlight test pattern.
    
    Args:
        data_array: The image data to plot
        meta: Metadata for the image
        image_num: Image number or identifier
        vmin: Minimum value for color scaling (default: 4300)
        vmax: Maximum value for color scaling (default: 5000)
        cmap: Colormap to use (default: 'gray')
    """
    mean = np.nanmean(data_array)
    median = np.nanmedian(data_array)
    min_val = np.nanmin(data_array)
    max_val = np.nanmax(data_array)
    
    # Create a single figure with enhanced contrast
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot with enhanced scaling to highlight test pattern
    im = ax.imshow(data_array, interpolation='none', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'Image {image_num} (Enhanced Contrast: vmin={vmin}, vmax={vmax})')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pixel Value')
    
    # Add text with image stats
    text = fig.text(0.5, 0.01,
                   f'Min: {min_val} | Max: {max_val} | Mean: {mean:.1f} | Median: {median:.1f}',
                   ha='center', va='bottom')
    
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return fig


def get_binned_ref(reference, binned_rows, start_rows, binned_cols, start_cols, row_offset, col_offset):
    """Bin reference according to metadata."""
    ref = reference[row_offset: row_offset + start_rows, col_offset: col_offset + start_cols]
    return np.bitwise_and(ref.reshape(int(binned_rows), int(start_rows / binned_rows),
                                      int(binned_cols), int(start_cols / binned_cols)).sum(3).sum(1), 0xFFFF)


def process_file_in_chunks(file_path, chunk_size=1024 * 1024):
    """Process the file in chunks if there's not enough memory to load it all at once."""
    sync_pattern = b''.join(SYNC_MARKER)
    sync_indices = []
    data = b''

    with open(file_path, 'rb') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break

            data += chunk
            # Find all occurrences of the sync pattern in the current data
            for match in re.finditer(re.escape(sync_pattern), data):
                sync_indices.append(match.start())

            # Keep only the last few bytes of the buffer in case the pattern is split
            data = data[-(len(sync_pattern) - 1):]

    return sync_indices


def find_all_SYNC_MARKERs(file_path):
    """Find all occurrences of the sync pattern (1A CF FC 1D) in the file using regex."""
    if has_sufficient_memory(file_path):
        with open(file_path, 'rb') as file:
            data = file.read()  # Read the entire file into memory
        
        sync_pattern = b''.join(SYNC_MARKER)
        sync_indices = []
        for match in re.finditer(re.escape(sync_pattern), data):
            sync_indices.append(match.start())
            
        return sync_indices, data
    else:
        print("Not enough memory to load the file into memory. Processing in chunks.")
        sync_indices = process_file_in_chunks(file_path)
        return sync_indices, None  # Return None for data since it's processed in chunks


def extract_packets(sync_indices, data):
    """Extract packets from data using sync indices."""
    packets = []
    for i in range(len(sync_indices) - 1):
        start = sync_indices[i] + len(SYNC_MARKER)
        end = sync_indices[i + 1]
        packet = data[start:end]
        packets.append(packet)
    return packets


def process_packet(packet, metadata_dict, data_dict):
    """Process a single packet.
    
    Args:
        packet: The packet to process
        metadata_dict: Dictionary mapping image_id to metadata packets
        data_dict: Dictionary mapping image_id to lists of data packets
    """
    apid, length, seqnum, header = parse_header(packet)
    if apid in CSIE_META_APID:
        # Process metadata packet
        meta_packet = parse_csie_meta(packet, header)
        image_id_meta = meta_packet.csie_meta_img_id
        metadata_dict[image_id_meta] = meta_packet
        
    elif apid in CSIE_DATA_APID:
        # Process data packet
        packet_num, data, image_id_data = parse_csie_data(packet)
        
        if image_id_data in metadata_dict:
            meta_packet = metadata_dict[image_id_data]
            rows, cols, roi_rows, roi_cols, row_offset, col_offset = get_image_meta(meta_packet)
            # Truncate data to expected length based on binning
            data = data[:cols]  # cols will be 752 for 2x2 binned data
        
        # Store the data packet
        if image_id_data not in data_dict:
            data_dict[image_id_data] = []
        data_dict[image_id_data].append(data)


def save_image_as_fits(image_array, meta_packet, image_id, base_path=None):
    """Save an image array as a FITS file with metadata.
    
    Args:
        image_array: The image data array
        meta_packet: The metadata packet containing timestamp info
        image_id: The image ID
        base_path: Base path for saving files. If None, uses ABSOLUTE_FILE_PATH directory
    """
    if base_path is None:
        base_path = os.path.dirname(ABSOLUTE_FILE_PATH[0])
    
    # Create processed_images directory if it doesn't exist
    output_dir = os.path.join(base_path, "processed_images")
    os.makedirs(output_dir, exist_ok=True)
    
    # Format the filename using just the image ID
    filename = f"image_{image_id}.fits"
    filepath = os.path.join(output_dir, filename)
    
    # Create primary HDU with the image data
    hdu = fits.PrimaryHDU(image_array)
    
    # Add metadata to header
    header = hdu.header
    
    # Add all metadata fields to the header
    for attr_name in dir(meta_packet):
        # Skip private attributes and methods
        if not attr_name.startswith('_'):
            try:
                value = getattr(meta_packet, attr_name)
                # Remove 'csie_' prefix if present and convert to FITS-compliant format
                if attr_name.startswith('csie_'):
                    attr_name = attr_name[5:]  # Remove 'csie_' prefix
                fits_key = attr_name.upper()[:8]
                header[fits_key] = value
            except (AttributeError, TypeError):
                # Skip if we can't get the value or it's not a simple type
                continue
    
    # Add image dimensions
    header['ROWS'] = image_array.shape[0]
    header['COLS'] = image_array.shape[1]
    
    # Save the FITS file
    hdu.writeto(filepath, overwrite=True)
    print(f"Saved FITS file: {filepath}")


def main():
    start_time = time.time()
    
    metadata_dict = {}  # Dictionary to store metadata packets
    data_dict = {}      # Dictionary to store data packets by image_id
    processed_images = 0
    
    for path in ABSOLUTE_FILE_PATH:
        print(f"Processing file: {path}")
        sync_indices, data = find_all_SYNC_MARKERs(path)
        if not sync_indices:
            print(f"No sync words found in {path}")
            continue
            
        packets = extract_packets(sync_indices, data)
        print(f"Extracted {len(packets)} packets")
        
        for i, packet in enumerate(packets):
            process_packet(packet, metadata_dict, data_dict)

    elapsed_time = time.time() - start_time
    
    print("\nPacket parsing complete. Summary:")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Metadata packets found: {len(metadata_dict)}")
    print(f"Data for {len(data_dict)} different image IDs collected")
    
    # Process any remaining data that hasn't been processed yet
    for image_id, data_packets in data_dict.items():
        if image_id in metadata_dict:
            meta_packet = metadata_dict[image_id]
            rows, cols, roi_rows, roi_cols, row_offset, col_offset = get_image_meta(meta_packet)
            
            # Check if we have enough rows to form a complete image
            if len(data_packets) == rows:
                data_array = process_csie_data(data_packets, rows, cols)
                print(f"Processed complete image for image_id: {image_id}")
                
                # Store the processed image in image_arrays for later use
                image_arrays.append(data_array)
                processed_images += 1
                
                # Save as FITS file
                save_image_as_fits(data_array, meta_packet, image_id)
                
                # Optionally plot the image with enhanced contrast
                if PLOT:
                    figlist.append(plot_csie_array(data_array, meta_packet, image_id))
            else:
                print(f"Incomplete image for image_id: {image_id}, only {len(data_packets)}/{rows} rows received")
    
    elapsed_time = time.time() - start_time
    print(f"Total images processed: {processed_images}")
    print(f"Total images in image_arrays: {len(image_arrays)}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    
    if PLOT and figlist:
        print("Displaying plots...")
        plt.show()
        
        # Save the last image if any were processed
        if image_arrays:
            # Save grayscale image with enhanced contrast
            plt.figure(figsize=(10, 8))
            plt.imshow(image_arrays[-1], interpolation='none', cmap='gray', vmin=4300, vmax=5000)
            plt.colorbar(label='Pixel Value')
            plt.title(f'Last Processed Image (Enhanced Contrast: vmin=4300, vmax=5000)')
            plt.savefig('enhanced_image_gray.png', dpi=300, bbox_inches='tight')
            print("Saved enhanced grayscale image to 'enhanced_image_gray.png'")
            
            # Save inverted grayscale (white features on black background)
            plt.figure(figsize=(10, 8))
            plt.imshow(image_arrays[-1], interpolation='none', cmap='gray_r', vmin=4300, vmax=5000)
            plt.colorbar(label='Pixel Value')
            plt.title(f'Last Processed Image (Inverted Grayscale)')
            plt.savefig('enhanced_image_gray_inverted.png', dpi=300, bbox_inches='tight')
            print("Saved inverted grayscale image to 'enhanced_image_gray_inverted.png'")
            

if __name__ == "__main__":
    main()

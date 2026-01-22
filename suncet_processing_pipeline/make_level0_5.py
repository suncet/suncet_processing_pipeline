"""
This is the code to make the Level 0.5 data product. 
We use 0_5 in the filename to prevent issues with some operating systems not being able to handle a period in a filename
"""

import numpy as np
from astropy.io import fits
import re
import psutil
import os
import sys
import time 
from datetime import datetime
import pandas as pd 
import h5py 
from pillow_jpls import Image
from io import BytesIO
from tqdm import tqdm
import importlib.util

class Level0_5:
    # Hardware and software configuration values. Do not edit unless you know why
    CSIE_ROWS = 2000
    CSIE_COLS = 1504
    PRIMARY_HDR_LEN = 6
    SECONDARY_HDR_LEN = 6
    CHECKSUM_LEN = 4
    SYNC_MARKER = [b'\x1A', b'\xCF', b'\xFC', b'\x1D']
    SYNC_MARKER_LEN = len(SYNC_MARKER)
    VCDU_HEADER_LEN = 8
    RETRANSMIT_HEADER_LEN = 16
    
    def __init__(self, file_paths, packet_definitions_path):
        """
        Initialize the Level0_5 processor.
        
        Args:
            file_paths (list): List of paths to the binary files to process
            packet_definitions_path (str): Path to folder containing gen_pkts.py and dsps_decoders.py.
                Must be provided. Path will be expanded (e.g., ~ will be expanded to home directory).
        
        Raises:
            ValueError: If packet_definitions_path is None or empty, or if the path doesn't exist.
        """
        if packet_definitions_path is None or not packet_definitions_path:
            raise ValueError("packet_definitions_path is required and cannot be None or empty")
        
        # Expand user path (e.g., ~ to home directory)
        packet_definitions_path = os.path.expanduser(packet_definitions_path)
        
        # Import gen_pkts and dsps_decoders from the specified path
        self.gen_pkts, self.dsps_decoders = self._import_packet_definitions(packet_definitions_path)
        
        self.ABSOLUTE_FILE_PATH = file_paths
        
        # Get APIDs from the packet definitions
        self.apid_df = self.read_ct_pkt_csv()
        
        # Initialize storage for processed data
        self.image_arrays = []
        
        # Initialize counters for statistics
        self.bad_checksum_counter = 0
        self.vcdu_headers = []
        self.retransmit_headers = []
    
    def _import_packet_definitions(self, packet_definitions_path):
        """
        Import gen_pkts and dsps_decoders from a specified path.
        
        Args:
            packet_definitions_path (str): Path to folder containing gen_pkts.py and dsps_decoders.py
            
        Returns:
            tuple: (gen_pkts module, dsps_decoders module)
        """
        # Check if path exists
        if not os.path.isdir(packet_definitions_path):
            raise ValueError(f"Packet definitions path does not exist: {packet_definitions_path}")
        
        # Check if required files exist
        gen_pkts_path = os.path.join(packet_definitions_path, 'gen_pkts.py')
        dsps_decoders_path = os.path.join(packet_definitions_path, 'dsps_decoders.py')
        
        if not os.path.isfile(gen_pkts_path):
            raise ValueError(f"gen_pkts.py not found at: {gen_pkts_path}")
        if not os.path.isfile(dsps_decoders_path):
            raise ValueError(f"dsps_decoders.py not found at: {dsps_decoders_path}")
        
        # Add the path to sys.path temporarily for imports
        original_path = sys.path.copy()
        try:
            if packet_definitions_path not in sys.path:
                sys.path.insert(0, packet_definitions_path)
            
            # Import using importlib to avoid module name conflicts
            spec_gen_pkts = importlib.util.spec_from_file_location("gen_pkts", gen_pkts_path)
            gen_pkts = importlib.util.module_from_spec(spec_gen_pkts)
            spec_gen_pkts.loader.exec_module(gen_pkts)
            
            spec_dsps_decoders = importlib.util.spec_from_file_location("dsps_decoders", dsps_decoders_path)
            dsps_decoders = importlib.util.module_from_spec(spec_dsps_decoders)
            spec_dsps_decoders.loader.exec_module(dsps_decoders)
            
            return gen_pkts, dsps_decoders
            
        finally:
            # Restore original sys.path
            sys.path[:] = original_path
    
    def fletcher32(self, data: bytes) -> int:
        if len(data) % 2:
            data += b'\x00'
        words = [data[i] | (data[i+1] << 8) for i in range(0, len(data), 2)]
        sum1 = 0xffff
        sum2 = 0xffff
        for word in words:
            sum1 = (sum1 + word) % 0xffff
            sum2 = (sum2 + sum1) % 0xffff
        return (sum2 << 16) | sum1

    def validate_checksum(self, packet):
        """
        Validate the checksum of a packet using Fletcher-32.
        
        All packets (including DSPS) now use Fletcher-32 checksums (4 bytes).
        
        Args:
            packet (bytes): The complete packet including header and data
            
        Returns:
            bool: True if checksum is valid, False otherwise
        """
        if len(packet) < self.PRIMARY_HDR_LEN + self.CHECKSUM_LEN:
            return False
        
        # Extract the data portion (everything except the 4-byte checksum)
        data_portion = packet[:-self.CHECKSUM_LEN]
        
        # Extract the stored checksum (last 4 bytes)
        stored_checksum = packet[-self.CHECKSUM_LEN:]
        
        # Calculate the expected checksum using Fletcher-32 algorithm and convert to big-endian format
        calculated_checksum = self.fletcher32(data_portion)
        calculated_checksum = calculated_checksum.to_bytes(4, 'big')
        
        return stored_checksum == calculated_checksum

    def get_checksum_stats(self):
        """
        Get checksum validation statistics.
        
        Returns:
            dict: Dictionary containing checksum validation statistics
        """
        return {
            'bad_checksum_count': self.bad_checksum_counter,
            'checksum_length_bytes': self.CHECKSUM_LEN,
            'primary_header_length_bytes': self.PRIMARY_HDR_LEN
        }

    def process(self):
        """Main processing function."""
        start_time = time.time()
        
        metadata_dict = {}  # Dictionary to store image metadata packets
        data_dict = {}      # Dictionary to store data packets by image_id
        telemetry_dict = {} # Dictionary to store other telemetry packets
        dsps_dict = {}      # Dictionary to store DSPS packets
        processed_images = 0
        total_packets_processed = 0
        
        for path in self.ABSOLUTE_FILE_PATH:
            print(f"Processing file: {path}")
            
            filename = os.path.basename(path)
            from_hydra = filename.startswith('ccsds_')
            from_xband_gse = filename.startswith('suncet_')
            
            source = "hydra" if from_hydra else "xband_gse" if from_xband_gse else "csie"
            packets = self.extract_packets(path, source=source)
            if not packets:
                print(f"No packets found in {path}")
                continue
                
            print(f"Extracted {len(packets)} packets")
            total_packets_processed += len(packets)
            
            for packet in tqdm(packets, desc=f"Processing packets in {filename}"):
                self.process_packet(packet, metadata_dict, data_dict, telemetry_dict, dsps_dict, filename)

        elapsed_time = time.time() - start_time
        
        print("\nPacket parsing complete. Summary:")
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print(f"Total packets processed: {total_packets_processed}")
        print(f"Telemetry packets found: {len(telemetry_dict)}")
        print(f"CSIE Metadata packets found: {len(metadata_dict)}")
        print(f"Data for {len(data_dict)} different image IDs collected")
        print(f"DSPS packets found: {len(dsps_dict)}")
        print(f"Packets with invalid checksums: {self.bad_checksum_counter}")
        
        # Process and save images to FITS files
        if data_dict:
            self.process_images(data_dict, metadata_dict)
            
        # Save telemetry data to HDF5 file
        if telemetry_dict:
            self.save_packet_data_to_hdf5(telemetry_dict, 'telemetry')
        
        # Save DSPS data to HDF5 file
        if dsps_dict:
            self.save_packet_data_to_hdf5(dsps_dict, 'dsps')
        
        elapsed_time = time.time() - start_time
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print(f"Total packets with invalid checksums: {self.bad_checksum_counter}")

    def process_packet(self, packet, metadata_dict, data_dict, telemetry_dict, dsps_dict, filename=None):
        """Process a single packet."""
        # Validate checksum before processing
        if not self.validate_checksum(packet):
            self.bad_checksum_counter += 1
            return
        
        apid, length, sequence_number, header = self.parse_header(packet)
        
        # Get packet name from APID
        packet_name = self.apid_df[self.apid_df['APID'] == apid]['Name'].values
        if len(packet_name) == 0:
            return
            
        packet_name = packet_name[0]
        
        # Handle playback packets by stripping extra header and recursively processing
        if packet_name == 'playback':
            remaining_data = self.strip_playback_header(packet)
            return self.process_packet(remaining_data, metadata_dict, data_dict, telemetry_dict, dsps_dict, filename)
        
        # Skip unwanted packet types
        if any(packet_name.startswith(prefix) for prefix in ['des_', 'fp_test', 'uhf_pass', 'log_', 'mem_', 'version']):
            return
            
        # Use gen_pkts to interpret the packet based on its type
        if packet_name == 'csie_meta':
            meta_packet = self.gen_pkts.CSIE_META(packet[self.PRIMARY_HDR_LEN:], header) # CSIE_META expects "packet" to not include the header
            image_id_meta = meta_packet.csie_meta_img_id
            metadata_dict[image_id_meta] = meta_packet
            
        elif packet_name == 'csie_data':
            # Parse the secondary header to get image_id
            image_id_data, packet_num, secondary_header = self.parse_secondary_header(packet)
            
            # Calculate the total header size (primary + secondary)
            total_header_size = self.PRIMARY_HDR_LEN + self.SECONDARY_HDR_LEN

            # Calculate which bytes are the data portion
            data_length = length - self.PRIMARY_HDR_LEN - self.CHECKSUM_LEN  # secondary header length isn't counted for this purpose
            
            # Extract just the data portion of the packet (after both headers)
            data = packet[total_header_size:total_header_size + data_length] 

            # Check if we have metadata for this image to determine compression
            compression_enabled = False
            if image_id_data in metadata_dict:
                meta_packet = metadata_dict[image_id_data]
                compression_enabled, _, _, _ = self.parse_csie_icm_proc_config(meta_packet.csie_meta_icm_proc_config)

            # If this is the first row of a new image (sequence_number == 1)
            if sequence_number == 1:
                if compression_enabled:
                    # For compressed data, store as bytes (will be decompressed later)
                    data_dict[image_id_data] = data
                else:
                    # For uncompressed data, convert to little-endian uint16
                    data_dict[image_id_data] = np.frombuffer(data, dtype=np.uint16).byteswap()
            else:
                if image_id_data in data_dict:
                    if compression_enabled:
                        # For compressed data, concatenate bytes
                        data_dict[image_id_data] = data_dict[image_id_data] + data
                    else:
                        # For uncompressed data, concatenate as little-endian uint16
                        new_data = np.frombuffer(data, dtype=np.uint16).byteswap()
                        data_dict[image_id_data] = np.concatenate([data_dict[image_id_data], new_data])
            
        elif 'dsps' in packet_name:            
            class_name = packet_name.upper()
            try:
                packet_class = getattr(self.dsps_decoders, class_name)
                packet_instance = packet_class(packet[self.PRIMARY_HDR_LEN:], header, filename)
                dsps_dict[f"{packet_name}_{sequence_number}"] = packet_instance
            except (AttributeError, TypeError) as e:
                print(f"Warning: Could not process packet type {packet_name}: {str(e)}")
        
        else:
            # For all other packet types, use the packet name to get the corresponding class
            # Convert packet name to uppercase to match class names in gen_pkts
            class_name = packet_name.upper()
            try:
                packet_class = getattr(self.gen_pkts, class_name)
                packet_instance = packet_class(packet[self.PRIMARY_HDR_LEN:], header, filename)
                # Store the interpreted packet in telemetry_dict using a unique key
                telemetry_dict[f"{packet_name}_{sequence_number}"] = packet_instance
            except (AttributeError, TypeError) as e:
                print(f"Warning: Could not process packet type {packet_name}: {str(e)}")
    
    def extract_packets(self, file_path, source="csie"):
        """Extract packets from data using the appropriate strategy for the source."""
        if source == "hydra":
            return self.extract_ccsds_packets(file_path)
        if source == "xband_gse":
            return self.extract_xband_packets(file_path)

        # For direct CSIE files, use sync markers
        sync_indices, data = self.find_all_sync_markers(file_path)
        if not sync_indices:
            return []

        packets = []
        for i in range(len(sync_indices) - 1):
            start = sync_indices[i] + self.SYNC_MARKER_LEN
            end = sync_indices[i + 1]
            packet = data[start:end]
            packets.append(packet)

        # Handle the final packet (from last sync marker to end of file)
        if sync_indices:
            start = sync_indices[-1] + self.SYNC_MARKER_LEN
            end = len(data)
            if start < end:
                packet = data[start:end]
                packets.append(packet)

        return packets

    def extract_xband_packets(self, file_path):
        """Extract CCSDS packets from X-band GSE VCDU formatted files."""
        sync_indices, data = self.find_all_sync_markers(file_path)
        if not sync_indices or data is None:
            if data is None:
                print("Warning: X-band file too large to load into memory for VCDU stripping.")
            return []

        ccsds_packets = []
        frame_boundaries = sync_indices + [len(data)]

        for current_index, next_index in zip(sync_indices, frame_boundaries[1:]):
            frame_start = current_index + self.SYNC_MARKER_LEN
            frame_end = next_index
            frame = data[frame_start:frame_end]
            minimum_frame_len = self.VCDU_HEADER_LEN + self.SYNC_MARKER_LEN + self.RETRANSMIT_HEADER_LEN
            if len(frame) < minimum_frame_len:
                continue

            vcdu_header = frame[:self.VCDU_HEADER_LEN]
            interpreted_header = self.parse_vcdu_header(vcdu_header)
            interpreted_header["file_path"] = file_path
            self.vcdu_headers.append(interpreted_header)

            frame_payload = frame[self.VCDU_HEADER_LEN:]
            swapped_payload = self.swap_bytes_in_words(frame_payload, word_size=4)

            retransmit_segments = self.segment_data_by_sync(swapped_payload)
            if not retransmit_segments:
                print("Warning: No retransmit segments found within VCDU payload.")
                continue

            for segment in retransmit_segments:
                if len(segment) < self.RETRANSMIT_HEADER_LEN:
                    continue

                retrans_header_bytes = segment[:self.RETRANSMIT_HEADER_LEN]
                retrans_header = self.parse_retransmit_header(retrans_header_bytes)
                retrans_header["file_path"] = file_path
                retrans_header["vcdu_master_frame_count"] = interpreted_header.get("master_channel_frame_count")
                retrans_header["vcdu_virtual_channel_id"] = interpreted_header.get("virtual_channel_id")
                self.retransmit_headers.append(retrans_header)

                ccsds_payload = segment[self.RETRANSMIT_HEADER_LEN:]
                if not ccsds_payload:
                    continue

                ccsds_packets.extend(self.extract_ccsds_packets(ccsds_payload))

        return ccsds_packets

    @staticmethod
    def swap_bytes_in_words(data, word_size=4):
        """Swap each word to big-endian order (default 4-byte words)."""
        swapped = bytearray()
        for i in range(0, len(data), word_size):
            chunk = data[i:i + word_size]
            swapped.extend(chunk[::-1])
        return bytes(swapped)

    def find_sync_markers_in_bytes(self, data):
        """Find sync marker indices within a bytes-like object."""
        sync_pattern = b''.join(self.SYNC_MARKER)
        return [match.start() for match in re.finditer(re.escape(sync_pattern), data)]

    def segment_data_by_sync(self, data):
        """Split bytes on sync markers, returning the payload segments between markers."""
        indices = self.find_sync_markers_in_bytes(data)
        if not indices:
            return []

        segments = []
        boundaries = indices + [len(data)]
        for start_idx, end_idx in zip(indices, boundaries[1:]):
            payload_start = start_idx + self.SYNC_MARKER_LEN
            if payload_start >= end_idx:
                continue
            segments.append(data[payload_start:end_idx])
        return segments

    def parse_vcdu_header(self, header_bytes):
        """Parse an 8-byte VCDU header and return a dictionary of fields."""
        if len(header_bytes) != self.VCDU_HEADER_LEN:
            raise ValueError("VCDU header must be 8 bytes long")

        primary = header_bytes[:6]
        data_field_status = primary[4:6]
        padding = header_bytes[6:8]

        # Primary header
        first_two = int.from_bytes(primary[0:2], "big")
        transfer_frame_version = (first_two >> 14) & 0x03
        spacecraft_id = (first_two >> 4) & 0x03FF
        virtual_channel_id = first_two & 0x0F
        ocf_flag = (primary[2] >> 7) & 0x01
        master_channel_frame_count = primary[2]
        virtual_channel_frame_count = primary[3]

        # Data field status
        dfs_word = int.from_bytes(data_field_status, "big")
        secondary_header_flag = (dfs_word >> 15) & 0x01
        sync_flag = (dfs_word >> 14) & 0x01
        packet_order_flag = (dfs_word >> 13) & 0x01
        segment_length_id = (dfs_word >> 11) & 0x03
        first_header_pointer = dfs_word & 0x07FF

        return {
            "transfer_frame_version": transfer_frame_version,
            "spacecraft_id": spacecraft_id,
            "virtual_channel_id": virtual_channel_id,
            "ocf_flag": ocf_flag,
            "master_channel_frame_count": master_channel_frame_count,
            "virtual_channel_frame_count": virtual_channel_frame_count,
            "secondary_header_flag": secondary_header_flag,
            "sync_flag": sync_flag,
            "packet_order_flag": packet_order_flag,
            "segment_length_id": segment_length_id,
            "first_header_pointer": first_header_pointer,
            "padding": padding,
        }

    def parse_retransmit_header(self, header_bytes):
        """Parse a 16-byte retransmit header and return a dictionary of fields."""
        if len(header_bytes) != self.RETRANSMIT_HEADER_LEN:
            raise ValueError("Retransmit header must be 16 bytes long")

        packet_id = int.from_bytes(header_bytes[0:2], "little")
        seq_count = int.from_bytes(header_bytes[2:4], "little")
        pkt_length = int.from_bytes(header_bytes[4:6], "little")
        partition_id = int.from_bytes(header_bytes[6:8], "little")
        page = int.from_bytes(header_bytes[8:12], "little")
        record_num = header_bytes[12]
        record_total = header_bytes[13]
        pad1 = header_bytes[14]
        pad2 = header_bytes[15]

        return {
            "packet_id": packet_id,
            "sequence_count": seq_count,
            "packet_length": pkt_length,
            "partition_id": partition_id,
            "page": page,
            "record_number": record_num,
            "record_total": record_total,
            "pad1": pad1,
            "pad2": pad2,
        }

    def extract_ccsds_packets(self, data_source):
        """Extract packets from CCSDS data using header length."""
        if isinstance(data_source, (bytes, bytearray, memoryview)):
            return self._extract_ccsds_packets_from_bytes(data_source)

        with open(data_source, 'rb') as file_obj:
            return self._extract_ccsds_packets_from_fileobj(file_obj)

    def _extract_ccsds_packets_from_bytes(self, data_bytes):
        packets = []
        view = memoryview(data_bytes)
        offset = 0
        total_length = len(view)

        while offset + self.PRIMARY_HDR_LEN <= total_length:
            # Skip filler (assumed 0x00) between packets
            while offset < total_length and view[offset] == 0x00:
                offset += 1
            if offset + self.PRIMARY_HDR_LEN > total_length:
                break

            header = bytes(view[offset:offset + self.PRIMARY_HDR_LEN])
            if all(b == 0x00 for b in header):
                offset += self.PRIMARY_HDR_LEN
                continue

            length = int.from_bytes(header[4:6], 'big') + 1
            if length <= 0:
                offset += self.PRIMARY_HDR_LEN
                continue

            packet_end = offset + self.PRIMARY_HDR_LEN + length
            if packet_end > total_length:
                break

            packet = bytes(view[offset:packet_end])
            packets.append(packet)
            offset = packet_end

        return packets

    def _extract_ccsds_packets_from_fileobj(self, file_obj):
        packets = []
        while True:
            header = file_obj.read(self.PRIMARY_HDR_LEN)
            if len(header) < self.PRIMARY_HDR_LEN:
                break

            if all(b == 0x00 for b in header):
                continue

            length = int.from_bytes(header[4:6], 'big') + 1
            packet_data = file_obj.read(length)
            if len(packet_data) < length:
                break

            packets.append(header + packet_data)

        return packets

    def find_all_sync_markers(self, file_path):
        """Find all occurrences of the sync pattern (1A CF FC 1D) in the file using regex."""
        if self.has_sufficient_memory(file_path):
            with open(file_path, 'rb') as file:
                data = file.read()  # Read the entire file into memory
            
            sync_pattern = b''.join(self.SYNC_MARKER)
            sync_indices = []
            for match in re.finditer(re.escape(sync_pattern), data):
                sync_indices.append(match.start())
                
            return sync_indices, data
        else:
            print("Not enough memory to load the file into memory. Processing in chunks.")
            sync_indices = self.process_file_in_chunks(file_path)
            return sync_indices, None  # Return None for data since it's processed in chunks

    def process_file_in_chunks(self, file_path, chunk_size=1024 * 1024):
        """Process the file in chunks if there's not enough memory to load it all at once."""
        sync_pattern = b''.join(self.SYNC_MARKER)
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

    def has_sufficient_memory(self, file_path):
        """Check if there is sufficient memory to load the file into memory."""
        available_memory = psutil.virtual_memory().available
        file_size = os.path.getsize(file_path)
        return file_size <= available_memory
    
    def parse_header(self, packet):
        """Extract APID, length, and sequence number from a packet."""
        # Standard CCSDS primary header assumed
        header_data = bytearray(packet[:6])

        apid = int.from_bytes(header_data[0:2], 'big') & 0x7FF
        sequence_number = int.from_bytes(header_data[2:4], 'big') & 0x3FFF
        length = int.from_bytes(header_data[4:6], 'big')

        length += 1
            
        return apid, length, sequence_number, header_data

    def parse_secondary_header(self, packet):
        """Extract image ID and packet number from the secondary header."""
        secondary_header = packet[self.PRIMARY_HDR_LEN:self.PRIMARY_HDR_LEN + self.SECONDARY_HDR_LEN]
        image_id = int.from_bytes(secondary_header[0:4], 'big')
        packet_num = int.from_bytes(secondary_header[4:6], 'big')
        return image_id, packet_num, secondary_header

    def strip_playback_header(self, packet):
        """Strip the playback header and return the remaining data."""
        return packet[6:]
    
    def parse_csie_meta(self, packet, header):
        """Extract metadata packet into a class using autogen constructor."""
        return self.gen_pkts.CSIE_META(packet[6:], header, "HSSI BIN") # Note: CSIE_META expects the header to not be included as part of the packet so this [6:] gets rid of it

    def get_image_meta(self, meta_packet):
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

    
    def save_image_as_fits(self, image_array, meta_packet, image_id, base_path=None):
        """Save an image array as a FITS file with metadata."""
        if base_path is None:
            base_path = os.path.dirname(self.ABSOLUTE_FILE_PATH[0])
        
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

    def process_images(self, data_dict, metadata_dict):
        """
        Process and save all complete images to FITS files.
        
        Args:
            data_dict (dict): Dictionary containing data packets by image_id
            metadata_dict (dict): Dictionary containing metadata packets by image_id
        """
        processed_images = 0
        
        # Images are bigger than packets, so we can only process an image once we've received all the packets for that image
        for image_id, data_packets in data_dict.items():
            if image_id in metadata_dict:
                meta_packet = metadata_dict[image_id]
                rows, cols, roi_rows, roi_cols, row_offset, col_offset = self.get_image_meta(meta_packet)
                
                # Get the processing configuration
                compression_enabled, _, _, _ = self.parse_csie_icm_proc_config(meta_packet.csie_meta_icm_proc_config)

                if compression_enabled:
                    # data_packets is already bytes for compressed data
                    decompressed_image = self.decompress_jpegls_image(data_packets)
                    print(f"Decompressed JPEG-LS image for image_id: {image_id}, shape: {decompressed_image.shape}")
                    self.image_arrays.append(decompressed_image)
                    processed_images += 1
                    self.save_image_as_fits(decompressed_image, meta_packet, image_id)
                    continue
                
                # Check if we have enough packets to form a complete image
                if len(data_packets) == rows * cols:
                    image = data_packets.reshape(rows, cols)
                    print(f"Completed processing for image_id: {image_id}, shape: {image.shape}")
                    
                    # Store the processed image in image_arrays for later use
                    self.image_arrays.append(image)
                    processed_images += 1
                    
                    self.save_image_as_fits(image, meta_packet, image_id)
                else:
                    print(f"Incomplete image for image_id: {image_id}, only {len(data_packets)}/{rows * cols} pixels received")
        
        print(f"Total images processed: {processed_images}")
        print(f"Total images in image_arrays: {len(self.image_arrays)}")

    def get_packet_timestamp(self, packet):
        """
        Extract timestamp from a packet by looking for attributes that start with 'ccsdsSecHeader2_sec'.
        
        Args:
            packet: The packet object to extract timestamp from
            
        Returns:
            float: The timestamp value, or None if not found
        """
        for attr_name in dir(packet):
            # Skip private attributes and methods
            if attr_name.startswith('_'):
                continue
                
            # Check if the attribute name starts with 'ccsdsSecHeader2_sec'
            if attr_name.startswith('ccsdsSecHeader2_sec'):
                try:
                    timestamp = getattr(packet, attr_name)
                    # Ensure it's a numeric value
                    if isinstance(timestamp, (int, float, np.number)):
                        return float(timestamp)
                except (AttributeError, TypeError, ValueError):
                    continue
        
        return None

    def get_version(self):
        """Read version from the version file in the repo root."""
        # Get the repo root directory (one level up from this file)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(current_dir)
        version_file = os.path.join(repo_root, 'version')
        
        try:
            with open(version_file, 'r') as f:
                version = f.read().strip()
            return version
        except FileNotFoundError:
            print(f"Warning: Version file not found at {version_file}, using 'unknown'")
            return 'unknown'
    
    def deduplicate_telemetry_points(self, points):
        """
        Remove duplicates based on timestamp and packet_type, keeping the *newly generated* data in case of duplicates.
        We do this by iterating over points in order: existing first, then new.
        To keep the new data, we build a dict mapping dedup_key to the point, so later (new) points overwrite earlier (existing) ones.
        """
        dedup_dict = {}
        for point in points:
            dedup_key = (point['timestamp_seconds_since_boot'], point['packet_type'])
            dedup_dict[dedup_key] = point  # This will keep the last occurrence, i.e., the new data if duplicate
        return list(dedup_dict.values())

    def save_packet_data_to_hdf5(self, packet_dict, data_type, base_path=None, filename=None):
        """
        Generic method to save packet data to HDF5 file.
        
        Args:
            packet_dict (dict): Dictionary containing packet data
            data_type (str): Type of data being saved (e.g., 'telemetry', 'dsps')
            base_path (str, optional): Base path for output. If None, uses the directory of the first input file.
            filename (str, optional): Custom filename. If None, uses default format with version.
        """
        if not packet_dict:
            print(f"No {data_type} packets to save")
            return
            
        if base_path is None:
            base_path = os.path.dirname(self.ABSOLUTE_FILE_PATH[0])
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(base_path, data_type)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with version if not provided
        if filename is None:
            version = self.get_version()
            filename = f'suncet_{data_type}_mission_length_v{version}.h5'
        
        filepath = os.path.join(output_dir, filename)
        
        # Read existing data if file exists
        existing_packets = []
        if os.path.exists(filepath):
            print(f"Reading existing {data_type} data from: {filepath}")
            try:
                with h5py.File(filepath, 'r') as f:
                    if data_type in f:
                        existing_data = f[data_type][:]
                        # Convert structured array back to list of dictionaries
                        for row in existing_data:
                            packet_dict_row = {}
                            for field_name in row.dtype.names:
                                value = row[field_name]
                                # Convert bytes back to string for packet_type
                                if field_name == 'packet_type':
                                    value = value.decode('utf-8')
                                packet_dict_row[field_name] = value
                            existing_packets.append(packet_dict_row)
                        print(f"Loaded {len(existing_packets)} existing {data_type} packets")
            except Exception as e:
                print(f"Warning: Could not read existing file {filepath}: {e}")
                existing_packets = []
        
        # Create a list to store all packets (existing + new)
        all_packets = existing_packets.copy()
        
        # Process each new packet
        new_packets = []
        for packet_key, packet in packet_dict.items():
            # Get the timestamp from the packet using the helper function
            timestamp = self.get_packet_timestamp(packet)
            if timestamp is None:
                print(f"Warning: {data_type} packet {packet_key} has no timestamp, skipping")
                continue
            
            # Create a dictionary for this packet
            packet_type_name = packet.__class__.__name__
            packet_dict_row = {'packet_type': packet_type_name, 'timestamp_seconds_since_boot': timestamp}
            
            # Add all attributes of the packet
            for attr_name in dir(packet):
                # Skip private attributes, methods, and timestamp attributes (we're using timestamp as index)
                if (not attr_name.startswith('_') and 
                    not attr_name.startswith('ccsdsSecHeader2_sec')):
                    try:
                        value = getattr(packet, attr_name)
                        # Convert value to a simple type if possible
                        if isinstance(value, (np.ndarray, list)):
                            # For arrays/lists, take the first value if it's a single value
                            if len(value) == 1:
                                value = value[0]
                            else:
                                # Skip arrays/lists with multiple values
                                continue
                        packet_dict_row[attr_name] = value
                    except (AttributeError, TypeError):
                        continue
            
            new_packets.append(packet_dict_row)
        
        # Add new packets to all packets
        all_packets.extend(new_packets)
        
        if not all_packets:
            print(f"No valid {data_type} packets to save")
            return
        
        # Remove duplicates based on timestamp and packet_type, keeping the newly generated data
        unique_packets = self.deduplicate_telemetry_points(all_packets)
        print(f"Removed {len(all_packets) - len(unique_packets)} duplicate {data_type} packets (kept new data if duplicate)")
        
        # Sort packets by timestamp to ensure chronological order
        unique_packets.sort(key=lambda x: x['timestamp_seconds_since_boot'])
        
        # Extract timestamps and data for HDF5 storage
        timestamps = [packet['timestamp_seconds_since_boot'] for packet in unique_packets]
        
        # Get all unique field names (excluding timestamp and packet_type)
        field_names = set()
        for packet in unique_packets:
            for key in packet.keys():
                if key not in ['timestamp_seconds_since_boot', 'packet_type']:
                    field_names.add(key)
        
        # Create structured array for HDF5
        dtype_list = [
            ('timestamp_seconds_since_boot', 'f8'),
            ('packet_type', 'S50')  # String type for packet type names
        ]
        
        # Add all other fields as float64 (we'll handle type conversion)
        for field_name in sorted(field_names):
            dtype_list.append((field_name, 'f8'))
        
        # Create structured array
        data_array = np.zeros(len(unique_packets), dtype=dtype_list)
        
        # Fill the structured array
        for i, packet in enumerate(unique_packets):
            data_array[i]['timestamp_seconds_since_boot'] = packet['timestamp_seconds_since_boot']
            data_array[i]['packet_type'] = packet['packet_type'].encode('utf-8')
            
            # Fill other fields
            for field_name in field_names:
                if field_name in packet:
                    try:
                        data_array[i][field_name] = float(packet[field_name])
                    except (ValueError, TypeError):
                        data_array[i][field_name] = np.nan
                else:
                    data_array[i][field_name] = np.nan
        
        # Save to HDF5
        with h5py.File(filepath, 'w') as f:
            # Create a dataset for the packet data
            f.create_dataset(data_type, data=data_array)
            
            # Store field names as attributes
            f[data_type].attrs['field_names'] = list(field_names)
            
            # Store metadata about the processing
            f.attrs['version'] = self.get_version()
            f.attrs[f'total_{data_type}_packets'] = len(unique_packets)
            f.attrs['time_range_start'] = timestamps[0]
            f.attrs['time_range_end'] = timestamps[-1]
            f.attrs['processing_timestamp'] = datetime.now().isoformat()
            f.attrs['num_input_files_this_run'] = len(self.ABSOLUTE_FILE_PATH)
        
        print(f"Saved {len(unique_packets)} total {data_type} packets to HDF5 file: {filepath}")
        print(f"Added {len(new_packets)} new packets from {len(self.ABSOLUTE_FILE_PATH)} input files")
        print(f"Time range: {timestamps[0]:.2f} to {timestamps[-1]:.2f} seconds since boot")
        print(f"Available {data_type} types: {', '.join(set(packet['packet_type'] for packet in unique_packets))}")
    
    @staticmethod
    def read_ct_pkt_csv():
        """
        Read the ct_pkt.csv file and return a pandas DataFrame containing only the Name and APID columns.
        
        Returns:
            pandas.DataFrame: DataFrame containing Name and APID columns from ct_pkt.csv
        """
        # Construct the path to ct_pkt.csv relative to the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, 'packet_definitions', 'ct_pkt.csv')
        
        # Read the CSV file and select only Name and APID columns
        df = pd.read_csv(csv_path, usecols=['Name', 'APID'])
        
        return df

    def parse_csie_icm_proc_config(self, config_value):
        """
        Parse CSIE processing configuration value according to bitfield definition.
        
        Args:
            config_value (int): The csie_meta_icm_proc_config value from the packet
            
        Returns:
            dict: Dictionary containing parsed configuration values:
                - bit_depth: Bit depth based on encoding mode (None if JPEG-LS)
                - compression_enabled: Boolean, True if JPEG-LS encoding (mode 3)
                - pixel_threshold: Pixel threshold selection (0-3)
                - position_selection: Position selection (0-3)
        """
        # Parse according to the correct bitfield definition:
        # Bits 0-1: Encoding mode
        # Bits 4-5: Pixel threshold selection  
        # Bits 6-7: Position selection
        
        encoding_mode = config_value & 0x03  # Bits 0-1
        pixel_threshold = (config_value >> 4) & 0x03  # Bits 4-5
        position_selection = (config_value >> 6) & 0x03  # Bits 6-7
        
        # Determine bit depth and compression based on encoding mode
        if encoding_mode == 0:
            bit_depth = 16
            compression_enabled = False
        elif encoding_mode == 1:
            bit_depth = 8
            compression_enabled = False
        elif encoding_mode == 2:
            bit_depth = 12
            compression_enabled = False
        elif encoding_mode == 3:
            bit_depth = None  # JPEG-LS doesn't have a fixed bit depth
            compression_enabled = True
        else:
            bit_depth = None
            compression_enabled = False
        
        return compression_enabled, bit_depth, pixel_threshold, position_selection

    def decompress_jpegls_image(self, compressed_data):
        """
        Decompress JPEG-LS compressed image data using pillow_jpls.Image.
        Args:
            compressed_data (bytes): The compressed image data as bytes
        Returns:
            np.ndarray: The decompressed image as a numpy array
        """
        with BytesIO(compressed_data) as bio:
            img = Image.open(bio)
            arr = np.array(img)
        return arr

def main():
    # Example usage
    #file_paths = [
    #    f"{os.getenv('suncet_data')}/test_data/2025_206_10_18_14_bus_dsps_data/ccsds_2025_206_10_25_30",
    #     os.path.expanduser("~/Downloads/2025-02-13 sample cubixss data/CSIE_GSE_raw_219_truncated.bin"),
    #     os.path.expanduser("~/Dropbox/suncet_dropbox/9000 Processing/data/test_data/2025-06-13_bluefin_tvac/ccsds_2025_164_11_46_23")
    #]
    
    import glob
    folder = f"{os.getenv('suncet_data')}/test_data/2026-01-16_7170-002_fm_flatsat_cpt/2026_016_09_23_06_FM_FLATSAT_CPT"
    file_paths = glob.glob(os.path.join(folder, "ccsds_*"))
    
    # Filter to only include files (not directories)
    file_paths = [path for path in file_paths if os.path.isfile(path)]
    
    # Sort the file paths for consistent ordering
    file_paths.sort()
    
    # Path to packet definitions
    packet_definitions_path = '~/Library/CloudStorage/Box-Box/SunCET Private/suncet_ctdb/suncet_bus_v1-0-0/suncet_v1-0-0_decoders'
    
    processor = Level0_5(file_paths, packet_definitions_path)
    
    processor.process()

if __name__ == "__main__":
    main()

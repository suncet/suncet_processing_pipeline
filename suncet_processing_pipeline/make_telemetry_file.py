# Intent of this code is to create a single hdf5 file for all the SunCET telemetry points spanning the entire mission
# First thing to build is a tool that can ingest updated CTDB files and update our public documentation of what's in the hdf5 file
# The second piece will be ingesting new downlinked binary files and storing that data into the hdf5 file at the right location (sorted by time). 
# Note that no changes to the CTDB will occur after spacecraft delivery because we don't really have the capability to update flight software (we kinda do but it's very very slow.)

import os
import pandas as pd
import glob
import h5py
from pathlib import Path
import packet_definitions.gen_pkts as gen_pkts
import configparser


class CTDBDocumenter:
    def __init__(self):
        pass

    def process_ctdb(self, ctdb_path_filename=None, output_telemetry_definition_path_filename=None):
        """Process CTDB file and create a cleaned telemetry definition file.
        
        Args:
            ctdb_path_filename (str, optional): Path to the CTDB file. If None, uses default path.
            output_telemetry_definition_path_filename (str, optional): Path to save the processed telemetry definition file. 
                If None, saves as 'telemetry_definition.csv' in current directory.
        
        Returns:
            pandas.DataFrame: The processed telemetry data
        """
        # Set default paths if not provided
        if ctdb_path_filename is None:
            ctdb_path_filename = os.path.join(os.getcwd(), 'ct_tlm.csv')
        
        if output_telemetry_definition_path_filename is None:
            output_telemetry_definition_path_filename = os.path.join(os.getenv('suncet_data'), 'metadata', 'telemetry_definition.csv')
        
        # Read the CTDB file
        df = pd.read_csv(ctdb_path_filename)
        
        # Rename ItemName column to Variable Name
        df = df.rename(columns={'ItemName': 'Variable Name'})
        
        # Drop specified columns
        columns_to_drop = ['ExternalElement', 'Packet', 'ContainerType', 'RepeatMethod', 'APID', 'LongDescription', 'Source', 'FSWDataType', 'DisplayForm', 'Equation', 'Limits', 'SystemName']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        # Drop rows with specific Variable Names
        df = df[~df['Variable Name'].isin(['VERSION', 'TYPE', 'SEC_HDR_FLAG', 'PKT_APID', 'SEQ_FLGS', 'SEQ_CTR', 'PKT_LEN'])]
        
        # Drop rows where Variable Name starts with specific prefixes
        df = df[~df['Variable Name'].str.startswith(('ccsds', 'des_', 'REUSABLE_', 'mem_dump', 'fp_test', 'xband_adc', 'uhf_pass_', 'pl_data', 'log_', 'mem_', 'ver_'))]

        # Drop rows where Variable Name contains specific strings
        df = df[~df['Variable Name'].str.contains('dbg|debug|cmd_bytes|checksum|task_state|wp_state|xband_virt_|version|chksm|opcode|unused', case=False, na=False)]
        df = df.reset_index(drop=True)
        
        # Add units column based on Variable Name patterns
        df['Units'] = df['Variable Name'].apply(self._determine_units)
        
        # Save to new CSV file
        df.to_csv(output_telemetry_definition_path_filename, index=False)
        
        return df

    def _determine_units(self, variable_name):
        """Determine the units for a telemetry variable based on its name.
        
        Args:
            variable_name (str): The name of the telemetry variable
            
        Returns:
            str: The determined unit for the variable, or None if no unit can be determined
        """
        variable_name = variable_name.lower()
        
        if 'temp' in variable_name:
            return 'Celsius'
        if 'cur' in variable_name or '_iin' in variable_name or '_iout' in variable_name or variable_name.endswith('_i'):
            return 'Amperes'
        if 'vcell' in variable_name or 'volt' in variable_name or 'vin' in variable_name or 'v_out' in variable_name or variable_name.endswith('_v'):
            return 'Volts'
        if 'power' in variable_name or 'pwr' in variable_name:
            return 'Watts'
        if 'time' in variable_name or 'sec' in variable_name:
            return 'Seconds'
        if 'angle' in variable_name or 'deg' in variable_name:
            return 'Degrees'
        if 'count' in variable_name or 'cnt' in variable_name:
            return 'Count'
        
        return None


class TelemetryProcessor:
    def __init__(self, version=None):
        """Initialize the TelemetryProcessor.
        
        Args:
            version (str, optional): Version string for the output hdf5 file. If None, reads from config file.
        """
        if version is None:
            # Read version from config file
            config = configparser.ConfigParser()
            config_path = os.path.join(os.path.dirname(__file__), 'config_files', 'config_default.ini')
            config.read(config_path)
            self.version = config['structure']['version']
        else:
            self.version = version
            
        self.hdf5_filename = f'suncet_telemetry_mission_length_v{self.version}.hdf5'
        
    def process_files(self, path=None, file_list=None):
        """Process telemetry files and store data in hdf5 file.
        
        Args:
            path (str, optional): Path to directory containing ccsds_ binary files
            file_list (list, optional): List of specific ccsds_ binary files to process
            
        Returns:
            None
        """
        files_to_process = []
        
        if path is not None:
            # Find all ccsds_ files in the given path
            files_to_process.extend(glob.glob(os.path.join(path, 'ccsds_*')))
            
        if file_list is not None:
            # Add specific files from the list
            files_to_process.extend([f for f in file_list if os.path.basename(f).startswith('ccsds_')])
            
        if not files_to_process:
            raise ValueError("No files to process. Please provide either a path or a list of files.")
            
        # Process each file
        all_data = []
        for file_path in files_to_process:
            with open(file_path, 'rb') as f:
                binary_data = f.read()
                
            # Use gen_pkts to interpret the binary data
            packet = gen_pkts.interpret_packet(binary_data)
            
            if packet is not None:
                # Extract time and data from packet
                packet_time = packet.get_time()  # Assuming this method exists in gen_pkts
                packet_data = packet.get_data()  # Assuming this method exists in gen_pkts
                
                all_data.append({
                    'time': packet_time,
                    'data': packet_data,
                    'packet_type': packet.get_type()  # Assuming this method exists in gen_pkts
                })
        
        # Sort all data by time
        all_data.sort(key=lambda x: x['time'])
        
        # Store in hdf5 file
        with h5py.File(self.hdf5_filename, 'a') as f:
            for data in all_data:
                # Create group for packet type if it doesn't exist
                if data['packet_type'] not in f:
                    f.create_group(data['packet_type'])
                
                # Store data in appropriate group
                group = f[data['packet_type']]
                group.create_dataset(str(data['time']), data=data['data'])


if __name__ == "__main__":
    # Create processor instance and process CTDB file
    documenter = CTDBDocumenter()
    df = documenter.process_ctdb(ctdb_path_filename=os.path.expanduser('~/Downloads/ctdbSunCET_Bus_v0-1-0/ctdb/ct_tlm.csv'))
    print("Telemetry definition file has been created successfully.")
    print(f"Number of telemetry points processed: {len(df)}")

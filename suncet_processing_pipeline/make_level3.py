"""
This is the code to make the Level 3 data product. 
"""
import argparse
from pathlib import Path
from pprint import pprint

import h5netcdf
import numpy as np
from termcolor import cprint

import configparser
import metadata_mgr 


class Level3:
    """Class for applying the Level2 -> Level3 processing stage."""
    def __init__(self, run_name, config):
        """
        Args
          run_name: string, name of run we are processing                  
        config, config_parser.Config,  SunCET Data Processing Pipeline 
           configration object
        """
        self.run_name = run_name
        self.run_dir = Path('processing_runs') / run_name
        self.config = config

        if not self.run_dir.exists():
            raise RuntimeError(f'Could not find directory {self.run_dir}')

    def run(self):
        """Main method to process the level2 -> level3 stage."""
        # Load metadata
        metadata = metadata_mgr.MetadataManager(self.run_dir)

        # Start NetCDF File
        nc_output_path = self.run_dir / 'level3' / 'suncet_level3.nc'
        nc = Level3NetCDFWriter(nc_output_path, metadata)

        # Write some blank values
        nc.write_variable('carring_lat', np.zeros(100))
        nc.write_variable('carring_long', np.ones(100))
        nc.close()


class Level3NetCDFWriter:
    """Class for writing Level3 NetCDF Output."""
    def __init__(self, output_path, metadata):
        self._output_path = output_path
        self._metadata = metadata
        self._nc_file = h5netcdf.File(self._output_path, 'w')
    
    def write_variable(self, internal_name, variable_value):
        """Write a variable and its associated metadata to the file. 

        This function is passed the internal name of the variable, and uses
        the metadata manager to look up the NetCDF4 name and associated 
        attrbutes.

        Args
          internal_name: Internal name of variable (within code)
          variable_value: Value for the variable in the file
        """
        variable_name = self._metadata.get_netcdf4_variable_name(internal_name)

        # Wrote variable data
        print(f'Writing internal variable ', end='')
        cprint(internal_name, 'yellow', end='')
        print(f' NetCDF variable ', end='')
        cprint(variable_name, 'yellow')

        # TODO: this is broken
        self._nc_file.dimensions[variable_name + '_dim'] = variable_value.shape
        
        nc_variable = self._nc_file.create_variable(
            name=variable_name,
            dimensions=(variable_name + '_dim',),
            dtype=variable_value.dtype
        )
        
        nc_variable[:] = variable_value

        # Write variable attributes
        attrs = self._metadata.get_netcdf4_attrs(internal_name)

        print('attributes:')
        pprint(attrs)

        for key, value in attrs.items():
            nc_variable.attrs[key] = value
        
        print()
        
    def close(self):
        """Close the NetCDF file, commiting all changes."""
        self._nc_file.close()

    
def final_shdr_compositing_fix(level2_data, config):
    """Fix any lingaring SHDR Compositing Issues.

   Args
     level2_data : dict, str -> array
        Level 2 data, mapping internal variable names to their values 
         (generally numpy arrays)
      config : config_parser.Config
         SunCET Data Processing Pipeline configration object

    Returns  
      level2_data_fixed : dict, str -> array
         Copy of level2 data with the fix applied.
    """
    raise NotImplementedError()


def _get_parser():
    """Get command line ArgumentParser object with options defined.
        
    Returns
       object which can be used to parse command line objects
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--run-name', type=str, required=True,
                        help='String name of the run')
    parser.add_argument('-v', '--verbose', help='Print more debugging output')
    
    return parser

 
def main():
    """Main method when running this script directly."""
    args = _get_parser().parse_args()

    # Load config
    config_filename = Path('processing_runs') / args.run_name / 'config.ini'
    config = configparser.ConfigParser()
    config.read(config_filename)

    # Call run() method on Level3 class
    level3 = Level3(args.run_name, config)
    level3.run()
    

if __name__ == '__main__':
    main()

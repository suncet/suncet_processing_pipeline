"""
This is the code to make the Level 3 data product. 
"""
import argparse
from pathlib import Path
from pprint import pprint

import netCDF4
import numpy as np
from termcolor import cprint

import config_parser
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

        # placeholder, TODO check real image size    
        image_shape = (16, 16)  
        image = np.random.rand(*image_shape)
        image_height = np.arange(image_shape[0])
        image_width = np.arange(image_shape[1])

        # Write some blank values
        nc.write_dimension('image_height', image_height)
        nc.write_dimension('image_width', image_width)
        nc.write_variable('image', image)
        nc.close()


class Level3NetCDFWriter:
    """Class for writing Level3 NetCDF Output."""
    def __init__(self, output_path, metadata):
        self._output_path = output_path
        self._metadata = metadata
        self._nc_file = netCDF4.Dataset(
            self._output_path, 'w', format="NETCDF4"
        )
        
    def write_dimension(self, internal_name, dim_value):
        """Write a dimension and its associated metadata to the file


        This function is passed the internal name of the dimension, and uses
        the metadata manager to look up the NetCDF4 name and associated 
        attrbutes.

        Args
          internal_name: Internal name of dimension (within code)
          var_value: Value for the dimension in the file
        """
        # Create dimension in file
        dim_name = self._metadata.get_netcdf4_variable_name(internal_name)

        self._nc_file.createDimension(dim_name, dim_value.size)

        # Write variable for dimension data (will be created automatically
        # if we don't)
        nc_dim_data = self._nc_file.createVariable(
            dim_name,
            dim_value.dtype,
            (dim_name,)
        )
        
        nc_dim_data[:] = dim_value

        # Write attributes
        attrs = self._metadata.get_netcdf4_attrs(internal_name)
        for key, value in attrs.items():
            setattr(nc_dim_data, key, value)

    def write_variable(self, internal_name, var_value):
        """Write a variable and its associated metadata to the file. 

        This function is passed the internal name of the variable, and uses
        the metadata manager to look up the NetCDF4 name and associated 
        attrbutes.

        Args
          internal_name: Internal name of variable (within code)
          var_value: Value for the variable in the file
        """
        var_name = self._metadata.get_netcdf4_variable_name(internal_name)
        dim_names = self._metadata.get_netcdf4_dimension_names(internal_name)

        # Wrote variable data
        print(f'Writing internal variable ', end='')
        cprint(internal_name, 'yellow', end='')
        print(f' NetCDF variable ', end='')
        cprint(var_name, 'yellow')
        
        # Add dimensions for this vairable
        print('Dimensions ', end='')
        cprint(dim_names, 'yellow')

        # Write variable to file
        nc_variable = self._nc_file.createVariable(
            var_name,
            var_value.dtype,
            dim_names,
        )
        
        nc_variable[:] = var_value

        # Write variable attributes
        attrs = self._metadata.get_netcdf4_attrs(internal_name)

        print('attributes:')
        cprint(attrs, 'yellow')
        for key, value in attrs.items():
            setattr(nc_variable, key, value)

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
    config = config_parser.Config(config_filename)

    # Call run() method on Level3 class
    level3 = Level3(args.run_name, config)
    level3.run()
    

if __name__ == '__main__':
    main()

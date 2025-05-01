"""
This is the code to make the Level 3 data product. 
"""
import argparse
from pathlib import Path
from pprint import pprint

from astropy.io import fits
import numpy as np
from termcolor import cprint

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from suncet_processing_pipeline import (
    config_parser, metadata_managers
)


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
        # Load metadata and add values
        metadata = metadata_managers.FitsMetadataManager(self.run_dir)
        metadata.load_from_dict({
            "project_name": "SunCET",
            "data_title": "SunCET Level 1 Image",
        })
        
        # Write fits file
        fits_path = self.run_dir / 'level3' / 'output.fits'

        if fits_path.exists():
            fits_path.unlink()

        fits_file = fits.open(fits_path, "append")
        
        image_data = np.zeros((1024, 1024), dtype=np.uint16)
        hdu = fits.ImageHDU(image_data)
        fits_file.append(hdu)

        metadata.generate_fits_header(fits_file)

        cprint(f'Wrote to {fits_path}', 'green')
        print()
        print(repr(fits_file[0].header))

        fits_file.close()

        print()
        
        # Test loading metadata from FITS
        cprint('Starting new FITS file', 'green')

        metadata_new = metadata_managers.FitsMetadataManager(self.run_dir)
        metadata_new.load_from_fits(fits_path)

        print('Loaded metadata from previous FITS:')
        print(metadata_new._metadata_values)

        fits_path_new = self.run_dir / 'level3' / 'output_new.fits'

        if fits_path_new.exists():
            fits_path_new.unlink()

        fits_file_new = fits.open(fits_path_new, "append")
        hdu = fits.ImageHDU(image_data)
        fits_file_new.append(hdu)
            
        metadata.generate_fits_header(fits_file_new)
        print()
        print(repr(fits_file_new[0].header))

        fits_file_new.close()
        

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

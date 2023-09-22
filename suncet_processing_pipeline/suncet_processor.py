"""
This is the main wrapper for most/all(?) of the other processor related python files. Those also inherit the common functionality of Processor.
"""
import os
from glob import glob
import pandas as pd
import astropy.units as u
import sunpy.map
from astropy.io import fits
from suncet_processing_pipeline import config_parser

class Processor:
    def __init__(self, config_filename=os.getcwd() + '/suncet_processing_pipeline/config_files/config_default.ini'):
        self.config_filename = config_filename
        self.config = self.__read_config(config_filename)
        self.metadata = self.__load_metadata_definition()

    def __read_config(self, config_filename):   
        return config_parser.Config(config_filename)
    

    def __load_metadata_definition(self):
        return pd.read_csv(os.getenv('suncet_data') + '/metadata/' + self.config.base_metadata_filename)
    
    
    def save_metadata(self, filename=None):
        if filename is None: 
            filename = self.config.base_metadata_filename
            base, extension = os.path.splitext(filename)
            filename = f"{base}{'_no_new_filename_specified'}{extension}"

        path = os.getenv('suncet_data') + '/metadata/'
        self.metadata.to_csv(path + filename, index=False)


if __name__ == "__main__":
    processor = Processor()
    processor.run(version='1.0.0')

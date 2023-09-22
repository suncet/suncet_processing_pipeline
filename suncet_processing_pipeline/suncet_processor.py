"""
This is the main wrapper for most/all(?) of the other processor related python files
"""
import os
from glob import glob
import pandas as pd
import astropy.units as u
import sunpy.map
from astropy.io import fits
from suncet_processing_pipeline import config_parser
from suncet_processing_pipeline.make_level0_5 import Level0_5
from suncet_processing_pipeline.make_level1 import Level1


class Processor:
    def __init__(self, config_filename=os.getcwd() + '/suncet_processing_pipeline/config_files/config_default.ini'):
        self.config_filename = config_filename
        self.config = self.__read_config(config_filename)
        self.metadata = self.__load_metadata_definition()
        pass

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


    def run(self, version='1.0.0'):
        data_path = os.getenv('suncet_data') + 'v' + version + '/'

        #level0_5 = Level0_5.make()
        level1 = Level1(config=self.config)
        level1.make(version=version, path=data_path)
        self.save_metadata()
        pass


if __name__ == "__main__":
    processor = Processor()
    processor.run(version='1.0.0')

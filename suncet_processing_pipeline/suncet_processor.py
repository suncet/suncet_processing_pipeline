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
    

    def __load_metadata_definition(self, single_point_values_path_filename=None):
        metadata = pd.read_csv(os.getenv('suncet_data') + '/metadata/' + self.config.base_metadata_filename)
        if single_point_values_path_filename is not None: 
            df = pd.read_csv(single_point_values_path_filename)
            metadata = pd.concat([metadata, df], ignore_index=True)  #TODO: Figure out if there can ever be redundancy here -- if so, figure out how to deal with it
        return metadata


    def run(self, version='1.0.0'):
        data_path = os.getenv('suncet_data') + 'v' + version + '/'

        #level0_5 = Level0_5.make()
        level1 = Level1(config=self.config)
        level1.make(version=version, path=data_path)
        pass


if __name__ == "__main__":
    processor = Processor()
    processor.run(version='1.0.0')

"""
This is the main wrapper for most/all(?) of the other processor related python files
"""
import os
from glob import glob
import astropy.units as u
import sunpy.map
from suncet_processing_pipeline import config_parser, make_level0_5, make_level1, make_level2, make_level3, make_level4

class Processor:
    def __init__(self, config_filename=os.getcwd() + '/suncet_processing_pipeline/config_files/config_default.ini'):
        self.config_filename = config_filename
        self.config = self.__read_config(config_filename)

    def __read_config(self, config_filename):   
        return config_parser.Config(config_filename)


    def run(self):
        pass
    

if __name__ == "__main__":
    processor = Processor()
    processor.run()

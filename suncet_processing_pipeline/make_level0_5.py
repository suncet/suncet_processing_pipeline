"""
This is the code to make the Level 0.5 data product. 
We use 0_5 in the filename to prevent issues with some operating systems not being able to handle a period in a filename
"""
import os
from glob import glob
import astropy.units as u
import sunpy.map
from suncet_processing_pipeline import config_parser
from suncet_processing_pipeline.packet_definitions import beacon_definition


class Level0_5:
    def __init__(self, config_filename=os.getcwd() + '/suncet_processing_pipeline/config_files/config_default.ini'):
        self.config_filename = config_filename
        self.config = self.__read_config(config_filename)
        self.level0 = self.__load_level0()

    def __read_config(self, config_filename):   
        return config_parser.Config(config_filename)
    

    def __load_level0(filename):
        pass


    def run(self):
        filename = os.getenv('suncet_data') + '/ctim_data/data_8646760_2023-12-03T23-32-22'
        with open(filename, 'rb') as file:
            binary_data = file.read()

        parsed_beacon = beacon_definition.packet.parse(binary_data)

        print(parsed_beacon)
        pass


if __name__ == "__main__":
    level0_5 = Level0_5()
    level0_5.run()


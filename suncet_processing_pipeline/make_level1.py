"""
This is the code to make the Level 1 data product. 
"""
import os
from glob import glob
from pathlib import Path
import astropy.units as u
import sunpy.map
from suncet_processing_pipeline import config_parser

class Level1:
    def __init__(self, config):
        self.config = config

    
    def make(self, version='1.0.0', path=None, filenames=None): 
        if path is not None: 
            filenames = glob(path + '/*.fits')
        if filenames is None: 
            raise ValueError('Need to provide either path to files or filenames that you want to process.')
        level0_5 = self.__load_level0_5(filenames)
        

        pass


    def __load_level0_5(self, filenames):
        # map_list = []
        # for file in filenames:
        #     map_list.append(sunpy.map.Map(file))

        # return sunpy.map.MapSequence(map_list)
    
        # Hack to get synthetic  image
        filename = os.getenv('suncet_data') + '/synthetic/level0_raw/fits/config_default.fits'
        map = sunpy.map.Map(filename)
        return sunpy.map.MapSequence(map)


    def run(self):
        pass
    

if __name__ == "__main__":
    level1 = Level1()
    level1.run()


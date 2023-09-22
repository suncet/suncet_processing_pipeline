"""
This is the code to make the Level 1 data product. 
"""
import os
from glob import glob
from pathlib import Path
import astropy.units as u
import sunpy.map
from suncet_processing_pipeline import config_parser
from suncet_processing_pipeline.suncet_processor import Processor

class Level1(Processor):
    def __init__(self):
        super().__init__()
        #self.metadata = self.__load_metadata_from_level0_5()

    
    def __load_metadata_from_level0_5(self):
         pass


    def make(self, version='1.0.0', path=None, filenames=None): 
        if path is not None: 
            filenames = glob(path + '/*.fits')
        if filenames is None: 
            raise ValueError('Need to provide either path to files or filenames that you want to process.')
        filenames = os.getenv('suncet_data') + '/synthetic/level0_raw/fits/config_default_OBS_2023-02-14T17:00:00.000.fits' # Hack to get synthetic  image
        level0_5 = self.__load_level0_5(filenames)
        
        meta_filename = self.__make_metadata_filename(filenames, version)
        self.save_metadata(filename=meta_filename)

        pass


    def __load_level0_5(self, filenames):
        # map_list = []
        # for file in filenames:
        #     map_list.append(sunpy.map.Map(file))

        # return sunpy.map.MapSequence(map_list)
        map = sunpy.map.Map(filenames)
        return map


    def __make_metadata_filename(self, filename, version):
        filename_with_extension = os.path.basename(filename)
        base, extension = os.path.splitext(filename_with_extension)
        return f"{base}{'_metadata_v'}{version}{'.csv'}"


    def run(self):
        pass
    

if __name__ == "__main__":
    level1 = Level1()
    level1.run()


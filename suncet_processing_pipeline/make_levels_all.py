"""
This script will run all of the processing from the bottom up
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


class MakeAllLevels:
    def __init__(self):
        pass

    def run(self, version='1.0.0'):
        data_path = os.getenv('suncet_data') + 'v' + version + '/'

        #level0_5 = Level0_5.make()
        level1 = Level1()
        level1.make(version=version, path=data_path)
        pass


if __name__ == "__main__":
    maker = MakeAllLevels()
    maker.run(version='1.0.0')

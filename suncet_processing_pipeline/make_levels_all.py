"""
This script will run all of the processing from the bottom up
"""
from glob import glob
import pandas as pd
import astropy.units as u
import sunpy.map
from astropy.io import fits
from suncet_processing_pipeline import config_parser
from suncet_processing_pipeline.make_level1 import Level1
from suncet_processing_pipeline.data_paths import data_path


class MakeAllLevels:
    def __init__(self):
        pass

    def run(self, version='1.0.0'):
        version_data_path = data_path('v' + version)

        level1 = Level1()
        level1.make(version=version, path=str(version_data_path))
        pass


if __name__ == "__main__":
    maker = MakeAllLevels()
    maker.run(version='1.0.0')

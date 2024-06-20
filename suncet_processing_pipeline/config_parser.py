import os
import configparser
import json
import astropy.units as u


class Config:

    def __init__(self, filename):
        config = configparser.ConfigParser()
        print('Reading configuration file: {}'.format(filename))
        if os.path.isfile(filename): 
            config.read(filename)
        else: 
            raise Exception('Config file does not exist or path is incorrect.')

        # behavior
        self.make_level0_5 = config['behavior'].getboolean('make_level0_5')
        self.make_level1 = config['behavior'].getboolean('make_level1')
        self.make_level2 = config['behavior'].getboolean('make_level2')
        self.make_level3 = config['behavior'].getboolean('make_level3')
        self.make_level4 = config['behavior'].getboolean('make_level4')

        # limits
        self.example_limit = json.loads(config.get('limits', 'example_limit')) * u.Angstrom

        # structure
        self.version = config['structure']['version']
        self.base_metadata_filename = config['structure']['base_metadata_filename']

        # calibration
        self.calibration_path = config['calibration']['calibration_path']
        self.dark_filename = config['calibration']['dark_filename']
        self.flat_filename = config['calibration']['flat_filename']
        self.badpix_filename = config['calibration']['badpix_filename']
        self.cosmic_ray_removal = config['calibration']['cosmic_ray_removal']


if __name__ == "__main__":
    filename = os.getcwd() + '/suncet_processing_pipeline/config_files/config_default.ini'
    config = Config(filename)
    print(config.example_behavior)  # Just an example to see something return when running as a script

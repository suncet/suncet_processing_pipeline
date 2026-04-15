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
        self.ignore_realtime = config['behavior'].getboolean('ignore_realtime', fallback=False)
        self.save_png = config['behavior'].getboolean('save_png', fallback=False)
        self.save_jpeg2000 = config['behavior'].getboolean('save_jpeg2000', fallback=False)

        # limits
        self.example_limit = json.loads(config.get('limits', 'example_limit')) * u.Angstrom

        # paths (optional - for configs without [paths] section)
        try:
            self.data_to_process_path = config.get('paths', 'data_to_process_path')
        except (configparser.NoSectionError, configparser.NoOptionError):
            self.data_to_process_path = ''

        # structure (needed for CTDB paths)
        self.version = config['structure']['version']
        self.csie_version = config['structure']['csie_version']
        self.base_metadata_filename = config['structure']['base_metadata_filename']
        self.output_suffix = config.get('structure', 'output_suffix', fallback='').strip()

        try:
            ctdb_base = os.path.expanduser(config.get('paths', 'ctdb_base'))
        except (configparser.NoSectionError, configparser.NoOptionError):
            ctdb_base = os.path.expanduser('~/Library/CloudStorage/Box-Box/SunCET Private')

        def _version_to_path_format(v):
            return 'v' + v.replace('.', '-')

        version_path = _version_to_path_format(self.version)
        csie_version_path = _version_to_path_format(self.csie_version)
        self.bus_ctdb_path = os.path.join(ctdb_base, 'suncet_ctdb', f'suncet_{version_path}')
        self.csie_ctdb_path = os.path.join(ctdb_base, 'suncet_ctdb', f'suncet_csie_ctdb_{csie_version_path}')
        self.packet_definitions_path = os.path.join(
            self.bus_ctdb_path, f'suncet_{version_path}_decoders'
        )

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

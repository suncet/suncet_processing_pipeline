"""
This is the main wrapper for most/all(?) of the other processor related python files. Those also inherit the common functionality of Processor.
"""
import os
from glob import glob
import pandas as pd
import astropy.units as u
import sunpy.map
from astropy.io import fits
from suncet_processing_pipeline import config_parser
from suncet_processing_pipeline.make_level0_5 import Level0_5
from suncet_processing_pipeline.make_level0_5 import discover_level0_5_input_files
from suncet_processing_pipeline.make_level1 import Level1

class Processor:
    def __init__(self, config_filename=None):
        if config_filename is None: 
            raise ValueError('It is important that you specify a path/filename to the config file you want to run with. That is your main method of interacting with the procesing.')
        self.config = self.__read_config(config_filename)
        self.metadata = self.__load_metadata_definition()


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


    def run(self):
        if self.config.make_level0_5:
            data_path = self.config.data_to_process_path
            # Resolve path: absolute (~ or /) use as-is; otherwise relative to suncet_data
            if data_path.startswith('/') or data_path.startswith('~'):
                folder = os.path.expanduser(data_path)
            else:
                folder = os.path.join(os.getenv('suncet_data', ''), data_path)

            file_paths = discover_level0_5_input_files(
                folder, ignore_realtime=getattr(self.config, "ignore_realtime", False)
            )

            processor = Level0_5(
                file_paths,
                self.config.packet_definitions_path,
                self.config.bus_ctdb_path,
                self.config.csie_ctdb_path,
                output_base_folder=folder,
                processing_config=self.config,
                save_png=getattr(self.config, "save_png", False),
                save_jpeg2000=getattr(self.config, "save_jpeg2000", False),
            )
            processor.process()

        if self.config.make_level1:
            level1 = Level1(self.config)
            level1.make(level0_5_to_process=os.getenv('suncet_data') + 'level0_5/')
        pass


if __name__ == "__main__":
    processor = Processor(config_filename=os.getcwd() + '/suncet_processing_pipeline/config_files/config_default.ini')
    processor.run()

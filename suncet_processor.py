"""
This is the main wrapper for most/all(?) of the other processor related python files. Those also inherit the common functionality of Processor.
"""
import argparse
import os
from glob import glob
from pathlib import Path
import pandas as pd
import astropy.units as u
import sunpy.map
from astropy.io import fits
from suncet_processing_pipeline import config_parser
from suncet_processing_pipeline.make_level0_5 import Level0_5
from suncet_processing_pipeline.make_level0_5 import discover_level0_5_input_files
from suncet_processing_pipeline.make_level1 import Level1
from suncet_processing_pipeline.data_paths import data_path, resolve_data_path
from suncet_processing_pipeline.run_provenance import (
    ProcessingRunProvenance,
    resolved_config_snapshot,
)

class Processor:
    def __init__(self, config_filename=None):
        if config_filename is None: 
            raise ValueError('It is important that you specify a path/filename to the config file you want to run with. That is your main method of interacting with the procesing.')
        self.config_filename = os.path.abspath(os.path.expanduser(config_filename))
        self.config = self.__read_config(self.config_filename)
        self.metadata = self.__load_metadata_definition()


    def __read_config(self, config_filename):   
        return config_parser.Config(config_filename)
    

    def __load_metadata_definition(self):
        return pd.read_csv(data_path('metadata', self.config.base_metadata_filename))
    
    
    def save_metadata(self, filename=None):
        if filename is None: 
            filename = self.config.base_metadata_filename
            base, extension = os.path.splitext(filename)
            filename = f"{base}{'_no_new_filename_specified'}{extension}"

        path = data_path('metadata', filename)
        self.metadata.to_csv(path, index=False)


    def run(self):
        configured_data_path = self.config.data_to_process_path
        folder = str(resolve_data_path(configured_data_path))

        file_paths = []
        if self.config.make_level0_5:
            file_paths = discover_level0_5_input_files(
                folder, ignore_realtime=getattr(self.config, "ignore_realtime", False)
            )

        provenance = ProcessingRunProvenance(
            data_root=folder,
            run_kind="suncet_processor",
            config_path=self.config_filename,
            resolved_config=resolved_config_snapshot(self.config, Path(folder)),
            arguments={"config": self.config_filename},
            repository_hint=Path(__file__).resolve().parent,
        )
        with provenance:
            provenance.record_inputs(file_paths)
            self._run_pipeline(folder, file_paths)

    def _run_pipeline(self, folder, file_paths):
        if self.config.make_level0_5:
            processor = Level0_5(
                file_paths,
                self.config.packet_definitions_path,
                self.config.bus_ctdb_path,
                self.config.csie_ctdb_path,
                output_base_folder=folder,
                processing_config=self.config,
                save_png=getattr(self.config, "save_png", False),
                save_jpeg2000=getattr(self.config, "save_jpeg2000", False),
                also_save_csie_meta_json=getattr(
                    self.config, "also_save_csie_meta_json", False
                ),
            )
            processor.process()

        if self.config.make_level1:
            level1 = Level1(self.config)
            level1.make(level0_5_to_process=str(data_path('level0_5')))


def main(argv=None):
    default_config = (
        Path(__file__).resolve().parent
        / 'suncet_processing_pipeline'
        / 'config_files'
        / 'config_default.ini'
    )
    parser = argparse.ArgumentParser(description='Run the configured SunCET pipeline.')
    parser.add_argument(
        '-c',
        '--config',
        type=Path,
        default=default_config,
        help='Processing configuration file (default: checked-in config_default.ini)',
    )
    args = parser.parse_args(argv)
    processor = Processor(config_filename=args.config)
    processor.run()


if __name__ == "__main__":
    main()

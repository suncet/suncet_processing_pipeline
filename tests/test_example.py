import os
import astropy.units as u
from suncet_processing_pipeline import config_parser
import setup_minimum_required_folders_files

def test_example():
    if os.getenv('suncet_data') == None:
        os.environ['suncet_data'] = './'
        setup_minimum_required_folders_files.run()
    
    config_filename = os.getcwd() + '/suncet_processing_pipeline/config_files/config_default.ini'
    config = config_parser.Config(config_filename)

    assert 2 + 2 == 4


if __name__ == "__main__":
    test_example()

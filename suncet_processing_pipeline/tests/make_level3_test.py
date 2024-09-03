import os
from .. import config_parser, make_level3


def test_Level3_object_instantiates():
    default_config = os.path.join(
        os.path.dirname(__file__), '..', 'config_files',
        'config_default.ini'
    )
    config = config_parser.Config(default_config)
    make_level3.Level3(config)
    

import os

from .. import config_parser


def test_read_default_config():
    default_config = os.path.join(
        os.path.dirname(__file__), '..', 'config_files',
        'config_default.ini'
    )
    config = config_parser.Config(default_config)

    assert hasattr(config, 'make_level0_5')
    assert hasattr(config, 'make_level1')
    assert hasattr(config, 'make_level2')
    assert hasattr(config, 'make_level3')
    assert hasattr(config, 'make_level4')

import os

from .. import config_parser


def test_read_default_config(tmp_path, monkeypatch):
    data_root = tmp_path / 'data'
    ctdb_root = tmp_path / 'ctdb'
    data_root.mkdir()
    ctdb_root.mkdir()
    monkeypatch.setenv('suncet_data', str(data_root))
    monkeypatch.setenv('suncet_ctdb', str(ctdb_root))
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
    assert hasattr(config, 'ignore_realtime')
    assert config.ignore_realtime is False
    assert hasattr(config, 'save_png')
    assert config.save_png is True
    assert hasattr(config, 'save_jpeg2000')
    assert config.save_jpeg2000 is False
    assert hasattr(config, 'also_save_csie_meta_json')
    assert config.also_save_csie_meta_json is True
    assert config.data_root == str(data_root)
    assert config.bus_ctdb_path.startswith(str(ctdb_root))
    assert config.csie_ctdb_path.startswith(str(ctdb_root))
    assert config.calibration_path == str(data_root / 'calibration')

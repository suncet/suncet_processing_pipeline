import os
from .. import config_parser, make_level3


def test_Level3_object_instantiates(tmp_path, monkeypatch):
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
    run_dir = data_root / 'processing_runs' / 'test-run'
    run_dir.mkdir(parents=True)

    level3 = make_level3.Level3('test-run', config)

    assert level3.run_dir == run_dir
        

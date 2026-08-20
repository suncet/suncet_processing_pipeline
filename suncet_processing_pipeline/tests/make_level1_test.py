import os
from .. import config_parser, make_level1


def test_Level1_object_instantiates(tmp_path, monkeypatch):
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
    make_level1.Level1(config)


def test_Level1_uses_requested_portable_input_directory(tmp_path, monkeypatch):
    data_root = tmp_path / 'data'
    ctdb_root = tmp_path / 'ctdb'
    input_root = data_root / 'level0_5'
    input_root.mkdir(parents=True)
    ctdb_root.mkdir()
    expected = input_root / 'image.fits'
    expected.touch()
    monkeypatch.setenv('suncet_data', str(data_root))
    monkeypatch.setenv('suncet_ctdb', str(ctdb_root))
    default_config = os.path.join(
        os.path.dirname(__file__), '..', 'config_files', 'config_default.ini'
    )
    level1 = make_level1.Level1(config_parser.Config(default_config))
    observed = {}
    monkeypatch.setattr(
        level1,
        '_Level1__load_level0_5',
        lambda filenames: observed.setdefault('filenames', filenames),
    )
    monkeypatch.setattr(
        level1,
        '_Level1__make_metadata_filename',
        lambda filename, version: 'unused.csv',
    )

    level1.make(input_root)

    assert observed['filenames'] == [str(expected)]

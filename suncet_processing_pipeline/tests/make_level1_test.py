import os
import numpy as np
import pytest
from .. import config_parser, make_level1


def test_effective_exposure_uses_stack_counts_filtering_and_right_shift():
    exposures = make_level1.effective_exposure_times_from_metadata(
        {
            "INTTIMEI": 0.035,
            "NSTACKI": 9,
            "STKNORMI": 8,
            "PIXFILTI": True,
            "INTTIMEO": 15.0,
            "NSTACKO": 4,
            "STKNORMO": 4,
            "PIXFILTO": True,
            "TELAPSE": 69.0,
        }
    )
    assert exposures == {"inner": 0.035, "outer": 11.25}


def test_2d_composite_exposure_requires_and_uses_inner_mask():
    image = np.zeros((2, 3))
    exposures = {"inner": 1.0, "outer": 4.0}
    with pytest.raises(ValueError, match="requires an inner_mask"):
        make_level1.create_exposure_time_mask(image, exposures)
    mask = make_level1.create_exposure_time_mask(
        image,
        exposures,
        inner_mask=np.array([[False, True, False], [True, True, False]]),
    )
    np.testing.assert_array_equal(mask, [[4, 1, 4], [1, 1, 4]])


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

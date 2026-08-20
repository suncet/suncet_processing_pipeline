"""Integration seams for the top-level configured pipeline wrapper."""

from pathlib import Path
from types import SimpleNamespace

import suncet_processor


def test_processor_delegates_level0_5_to_canonical_runner(monkeypatch, tmp_path):
    config_path = tmp_path / "config.ini"
    data_folder = tmp_path / "data"
    calls = []

    def fake_run(argv, *, _prepared):
        calls.append((argv, _prepared))

    monkeypatch.setattr(suncet_processor, "run_level0_5", fake_run)

    processor = object.__new__(suncet_processor.Processor)
    processor.config_filename = str(config_path)
    processor.config = SimpleNamespace(make_level0_5=True, make_level1=False)
    processor._run_pipeline(str(data_folder))

    assert calls == [
        (
            [
                "--config",
                str(config_path),
                "--folder",
                str(data_folder),
            ],
            (processor.config, Path(data_folder)),
        )
    ]

import local_system_setup


def test_bootstrap_can_initialize_tree_without_mutating_environment(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    ctdb_root = tmp_path / "ctdb"
    ctdb_root.mkdir()

    observed = {}
    def fake_initializer(args):
        observed["args"] = args
        return data_root

    monkeypatch.setattr(
        local_system_setup.setup_minimum_required_folders_files,
        "run",
        fake_initializer,
    )

    result = local_system_setup.bootstrap(
        [
            "--data-root", str(data_root),
            "--ctdb-root", str(ctdb_root),
            "--skip-environment",
            "--skip-metadata-download",
        ]
    )

    assert result == data_root
    assert observed["args"] == ["--skip-metadata-download"]

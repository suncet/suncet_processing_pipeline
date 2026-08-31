from suncet_processing_pipeline.level4.common.provenance import (
    collect_run_provenance,
    configuration_sha256,
)


def test_configuration_hash_is_order_independent() -> None:
    assert configuration_sha256({"a": 1, "b": [2, 3]}) == configuration_sha256(
        {"b": [2, 3], "a": 1}
    )


def test_provenance_records_configuration(tmp_path) -> None:
    provenance = collect_run_provenance(
        repository=tmp_path, configuration={"radial_step_px": 1.0}
    )

    assert provenance["configuration"]["radial_step_px"] == 1.0
    assert len(provenance["configuration_sha256"]) == 64
    assert provenance["git_commit"] is None

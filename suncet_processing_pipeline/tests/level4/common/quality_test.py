from suncet_processing_pipeline.level4.common.quality import (
    QualityFlag,
    decode_quality_flags,
    encode_quality_flags,
)


def test_quality_flags_round_trip() -> None:
    mask = encode_quality_flags(
        ["ASSUMED_CADENCE", QualityFlag.SYNTHETIC_BYPASS]
    )

    assert mask == QualityFlag.ASSUMED_CADENCE | QualityFlag.SYNTHETIC_BYPASS
    assert decode_quality_flags(mask) == (
        "ASSUMED_CADENCE",
        "SYNTHETIC_BYPASS",
    )


def test_zero_quality_mask_has_no_names() -> None:
    assert decode_quality_flags(0) == ()

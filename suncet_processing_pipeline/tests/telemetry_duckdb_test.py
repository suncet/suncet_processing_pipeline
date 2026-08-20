import pandas as pd

from ..make_telemetry_file import TelemetryProcessor
from ..readers import TelemetryReader


def test_duckdb_ingest_is_per_apid_and_idempotent(tmp_path):
    decoded = tmp_path / "decoded"
    decoded.mkdir()
    pd.DataFrame(
        {"time": [1.0, 3.0], "temperature": [10.0, 12.0]}
    ).to_csv(decoded / "decoded_apid_0001_beacon.csv", index=False)
    pd.DataFrame(
        {"time": [2.0, 4.0], "angle": [20.0, 40.0]}
    ).to_csv(decoded / "decoded_apid_0016_adcs.csv", index=False)
    database = tmp_path / "mission.duckdb"
    processor = TelemetryProcessor("test", database)

    first = processor.process_files(path=decoded)
    second = processor.process_files(path=decoded)

    assert first == {"files_ingested": 2, "files_skipped": 0, "rows_ingested": 4}
    assert second == {"files_ingested": 0, "files_skipped": 2, "rows_ingested": 0}
    with TelemetryReader(database) as reader:
        assert list(reader.get_packet_types()) == ["beacon", "adcs"]
        assert list(reader.query("beacon", order_by="time")["temperature"]) == [10, 12]
        joined = reader.asof_join(
            "beacon",
            "adcs",
            primary_time="time",
            secondary_time="time",
            secondary_fields=["angle"],
        )
        assert pd.isna(joined.loc[0, "adcs_angle"])
        assert joined.loc[1, "adcs_angle"] == 20

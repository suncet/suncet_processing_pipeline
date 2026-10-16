import duckdb
import pandas as pd
import pytest

from ..make_telemetry_file import TelemetryProcessor
from ..readers import TelemetryReader


def test_default_duckdb_is_versioned_at_top_of_test_data(tmp_path, monkeypatch):
    monkeypatch.setenv("suncet_data", str(tmp_path))

    processor = TelemetryProcessor("2.0.1")

    assert processor.database_path == (
        tmp_path / "test_data" / "suncet_telemetry_test_phase_v2.0.1.duckdb"
    )


def test_duckdb_rejects_ctdb_change_without_pipeline_version_bump(tmp_path):
    decoded = tmp_path / "decoded"
    decoded.mkdir()
    pd.DataFrame({"time": [1.0]}).to_csv(
        decoded / "decoded_apid_0001_beacon.csv", index=False
    )
    database = tmp_path / "test-phase.duckdb"
    TelemetryProcessor(
        "2.0.1",
        database,
        {"bus_ctdb_version": "2.0.5"},
    ).process_files(path=decoded)

    with pytest.raises(ValueError, match="Increment version_pipeline"):
        TelemetryProcessor(
            "2.0.1",
            database,
            {"bus_ctdb_version": "2.0.6"},
        ).process_files(path=decoded)


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

    assert first == {
        "files_ingested": 2,
        "files_replaced": 0,
        "files_skipped": 0,
        "rows_ingested": 4,
        "rows_removed": 0,
    }
    assert second == {
        "files_ingested": 0,
        "files_replaced": 0,
        "files_skipped": 2,
        "rows_ingested": 0,
        "rows_removed": 0,
    }
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


def test_duckdb_widens_numeric_column_for_later_enum_values(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    filename = "decoded_apid_0030_eps_hk.csv"
    pd.DataFrame({"eps_deploy_select": [1, 2]}).to_csv(
        first / filename, index=False
    )
    pd.DataFrame({"eps_deploy_select": ["SA_MINUS_Y", "OFF"]}).to_csv(
        second / filename, index=False
    )
    database = tmp_path / "test-phase.duckdb"
    processor = TelemetryProcessor("test", database)

    processor.process_files(path=first)
    processor.process_files(path=second)

    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute(
            "SELECT eps_deploy_select FROM apid_0030 ORDER BY eps_deploy_select"
        ).fetchall() == [("1",), ("2",), ("OFF",), ("SA_MINUS_Y",)]


def test_duckdb_replaces_changed_csv_at_same_path(tmp_path):
    decoded = tmp_path / "decoded"
    decoded.mkdir()
    source = decoded / "decoded_apid_0001_beacon.csv"
    database = tmp_path / "test-phase.duckdb"
    processor = TelemetryProcessor("test", database)

    pd.DataFrame({"time": [1.0, 2.0], "temperature": [10.0, 11.0]}).to_csv(
        source, index=False
    )
    processor.process_files(path=decoded)

    pd.DataFrame({"time": [3.0], "temperature": [12.0]}).to_csv(
        source, index=False
    )
    replacement = processor.process_files(path=decoded)

    assert replacement == {
        "files_ingested": 1,
        "files_replaced": 1,
        "files_skipped": 0,
        "rows_ingested": 1,
        "rows_removed": 2,
    }
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute(
            "SELECT time, temperature FROM apid_0001"
        ).fetchall() == [(3.0, 12.0)]
        assert connection.execute(
            "SELECT count(*) FROM _ingestions WHERE source_path = ?",
            [str(source.resolve())],
        ).fetchone() == (1,)


def test_direct_rows_replace_prior_csv_ingestion_and_are_idempotent(tmp_path):
    decoded = tmp_path / "decoded"
    decoded.mkdir()
    source = decoded / "decoded_apid_0001_beacon.csv"
    source.write_text("time,temperature\n1,10\n2,11\n")
    database = tmp_path / "test-phase.duckdb"
    processor = TelemetryProcessor("test", database)
    processor.process_files(file_list=[source])

    direct_rows = {
        1: [
            {"time": 3.0, "temperature": 12.0},
            {"time": 4.0, "temperature": 13.0},
        ]
    }
    first = processor.process_decoded_rows(
        direct_rows,
        {1: "beacon"},
        {1: source},
    )
    second = processor.process_decoded_rows(
        direct_rows,
        {1: "beacon"},
        {1: source},
    )

    assert first == {
        "files_ingested": 1,
        "files_replaced": 1,
        "files_skipped": 0,
        "rows_ingested": 2,
        "rows_removed": 2,
    }
    assert second == {
        "files_ingested": 0,
        "files_replaced": 0,
        "files_skipped": 1,
        "rows_ingested": 0,
        "rows_removed": 0,
    }
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute(
            "SELECT time, temperature FROM apid_0001 ORDER BY time"
        ).fetchall() == [(3.0, 12.0), (4.0, 13.0)]
        assert connection.execute(
            "SELECT source_path, rows_ingested FROM _ingestions"
        ).fetchall() == [(str(source.resolve()), 2)]


def test_direct_rows_replace_csv_while_widening_existing_schema(tmp_path):
    source = tmp_path / "decoded_apid_0001_beacon.csv"
    source.write_text("state\n1\n")
    database = tmp_path / "test-phase.duckdb"
    processor = TelemetryProcessor("test", database)
    processor.process_files(file_list=[source])

    replacement = processor.process_decoded_rows(
        {1: [{"state": "ON"}]},
        {1: "beacon"},
        {1: source},
    )

    assert replacement == {
        "files_ingested": 1,
        "files_replaced": 1,
        "files_skipped": 0,
        "rows_ingested": 1,
        "rows_removed": 1,
    }
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute("SELECT state FROM apid_0001").fetchall() == [
            ("ON",)
        ]
        assert connection.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'apid_0001' AND column_name = 'state'"
        ).fetchone() == ("VARCHAR",)

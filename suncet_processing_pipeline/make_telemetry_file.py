"""Build the mission-length, per-APID DuckDB telemetry store.

Level 0.5 already writes one decoded CSV stream per APID. This module ingests
those products transactionally, preserving each APID's natural sampling cadence
instead of forcing unrelated packets onto one shared timeline.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
from pathlib import Path
import re

import duckdb
import pandas as pd

from .config_parser import Config
from .data_paths import data_path


DECODED_CSV_RE = re.compile(r"decoded_apid_(?P<apid>\d+)_(?P<name>.+)\.csv$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CTDBDocumenter:
    """Create a public, human-focused subset of a private CTDB definition."""

    def process_ctdb(
        self,
        ctdb_path_filename=None,
        output_telemetry_definition_path_filename=None,
    ):
        if ctdb_path_filename is None:
            config = Config(
                Path(__file__).resolve().parent / "config_files" / "config_default.ini"
            )
            ctdb_path_filename = (
                Path(config.bus_ctdb_path) / "packet_definitions" / "ct_tlm.csv"
            )
        if output_telemetry_definition_path_filename is None:
            output_telemetry_definition_path_filename = data_path(
                "metadata", "telemetry_definition.csv"
            )

        frame = pd.read_csv(ctdb_path_filename).rename(
            columns={"ItemName": "Variable Name"}
        )
        drop = {
            "ExternalElement", "Packet", "ContainerType", "RepeatMethod", "APID",
            "LongDescription", "Source", "FSWDataType", "DisplayForm", "Equation",
            "Limits", "SystemName",
        }
        frame = frame.drop(columns=[name for name in drop if name in frame.columns])
        excluded = {
            "VERSION", "TYPE", "SEC_HDR_FLAG", "PKT_APID", "SEQ_FLGS", "SEQ_CTR",
            "PKT_LEN",
        }
        frame = frame[~frame["Variable Name"].isin(excluded)]
        frame = frame[
            ~frame["Variable Name"].str.startswith(
                (
                    "ccsds", "des_", "REUSABLE_", "mem_dump", "fp_test",
                    "xband_adc", "uhf_pass_", "pl_data", "log_", "mem_", "ver_",
                ),
                na=False,
            )
        ]
        frame = frame[
            ~frame["Variable Name"].str.contains(
                "dbg|debug|cmd_bytes|checksum|task_state|wp_state|xband_virt_|"
                "version|chksm|opcode|unused",
                case=False,
                na=False,
            )
        ].reset_index(drop=True)
        frame["Units"] = frame["Variable Name"].map(self._determine_units)
        output = Path(output_telemetry_definition_path_filename)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        return frame

    @staticmethod
    def _determine_units(variable_name):
        name = str(variable_name).lower()
        if "temp" in name:
            return "Celsius"
        if "cur" in name or "_iin" in name or "_iout" in name or name.endswith("_i"):
            return "Amperes"
        if "vcell" in name or "volt" in name or "vin" in name or "v_out" in name or name.endswith("_v"):
            return "Volts"
        if "power" in name or "pwr" in name:
            return "Watts"
        if "time" in name or "sec" in name:
            return "Seconds"
        if "angle" in name or "deg" in name:
            return "Degrees"
        if "count" in name or "cnt" in name:
            return "Count"
        return None


class TelemetryProcessor:
    """Ingest decoded per-APID CSVs into one mission-length DuckDB database."""

    def __init__(self, version=None, database_path=None):
        if version is None:
            config = configparser.ConfigParser()
            config.read(
                Path(__file__).resolve().parent / "config_files" / "config_default.ini"
            )
            version = config["structure"]["version_pipeline"]
        self.version = version
        self.database_path = Path(database_path) if database_path else data_path(
            "telemetry", f"suncet_telemetry_mission_length_v{version}.duckdb"
        )

    @staticmethod
    def _discover(path=None, file_list=None) -> list[Path]:
        candidates: list[Path] = []
        if path is not None:
            candidates.extend(Path(path).rglob("decoded_apid_*.csv"))
        if file_list:
            candidates.extend(Path(item) for item in file_list)
        unique = sorted({item.resolve() for item in candidates if item.is_file()})
        invalid = [str(item) for item in unique if DECODED_CSV_RE.match(item.name) is None]
        if invalid:
            raise ValueError("Unrecognized decoded telemetry filename(s): " + ", ".join(invalid))
        if not unique:
            raise ValueError("No decoded_apid_*.csv telemetry files were found")
        return unique

    @staticmethod
    def _initialize(connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS _apid_catalog (
                apid INTEGER PRIMARY KEY,
                packet_name VARCHAR NOT NULL,
                table_name VARCHAR NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS _ingestions (
                ingestion_id UUID PRIMARY KEY,
                source_path VARCHAR NOT NULL,
                source_sha256 VARCHAR NOT NULL UNIQUE,
                apid INTEGER NOT NULL,
                table_name VARCHAR NOT NULL,
                rows_ingested BIGINT NOT NULL,
                ingested_utc TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
            );
            """
        )

    def process_files(self, path=None, file_list=None) -> dict[str, int]:
        files = self._discover(path, file_list)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        ingested = 0
        skipped = 0
        rows_ingested = 0
        with duckdb.connect(str(self.database_path)) as connection:
            self._initialize(connection)
            for source in files:
                match = DECODED_CSV_RE.match(source.name)
                assert match is not None
                apid = int(match.group("apid"))
                packet_name = match.group("name")
                table_name = f"apid_{apid:04d}"
                checksum = _sha256(source)
                already_loaded = connection.execute(
                    "SELECT 1 FROM _ingestions WHERE source_sha256 = ?", [checksum]
                ).fetchone()
                if already_loaded:
                    skipped += 1
                    continue
                connection.execute("BEGIN")
                try:
                    connection.execute(
                        "INSERT INTO _apid_catalog VALUES (?, ?, ?) "
                        "ON CONFLICT (apid) DO UPDATE SET "
                        "packet_name = excluded.packet_name, table_name = excluded.table_name",
                        [apid, packet_name, table_name],
                    )
                    source_sql = str(source).replace("'", "''")
                    connection.execute(
                        f'CREATE TABLE IF NOT EXISTS "{table_name}" AS '
                        f"SELECT *, uuid() AS _ingestion_id, '{checksum}'::VARCHAR AS _source_sha256 "
                        f"FROM read_csv_auto('{source_sql}', header=true, all_varchar=false) LIMIT 0"
                    )
                    existing_columns = {
                        row[0]
                        for row in connection.execute(
                            f'DESCRIBE "{table_name}"'
                        ).fetchall()
                    }
                    source_columns = connection.execute(
                        f"DESCRIBE SELECT * FROM read_csv_auto("
                        f"'{source_sql}', header=true, all_varchar=false)"
                    ).fetchall()
                    for column_name, column_type, *_rest in source_columns:
                        if column_name in existing_columns:
                            continue
                        safe_column = str(column_name).replace('"', '""')
                        connection.execute(
                            f'ALTER TABLE "{table_name}" ADD COLUMN '
                            f'"{safe_column}" {column_type}'
                        )
                    ingestion_id = connection.execute("SELECT uuid()").fetchone()[0]
                    connection.execute(
                        f'INSERT INTO "{table_name}" BY NAME '
                        f"SELECT *, ?::UUID AS _ingestion_id, ?::VARCHAR AS _source_sha256 "
                        f"FROM read_csv_auto('{source_sql}', header=true, all_varchar=false)",
                        [ingestion_id, checksum],
                    )
                    count = connection.execute(
                        f'SELECT count(*) FROM "{table_name}" WHERE _ingestion_id = ?',
                        [ingestion_id],
                    ).fetchone()[0]
                    connection.execute(
                        "INSERT INTO _ingestions "
                        "(ingestion_id, source_path, source_sha256, apid, table_name, rows_ingested) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        [ingestion_id, str(source), checksum, apid, table_name, count],
                    )
                    connection.execute("COMMIT")
                except Exception:
                    connection.execute("ROLLBACK")
                    raise
                ingested += 1
                rows_ingested += count
        return {"files_ingested": ingested, "files_skipped": skipped, "rows_ingested": rows_ingested}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--version")
    args = parser.parse_args(argv)
    processor = TelemetryProcessor(args.version, args.database)
    files: list[Path] = []
    directories: list[Path] = []
    for path in args.paths:
        (directories if path.is_dir() else files).append(path)
    summary = {"files_ingested": 0, "files_skipped": 0, "rows_ingested": 0}
    for directory in directories or [None]:
        result = processor.process_files(path=directory, file_list=files)
        files = []
        for key in summary:
            summary[key] += result[key]
    print(f"DuckDB telemetry store: {processor.database_path}")
    print(summary)


if __name__ == "__main__":
    main()

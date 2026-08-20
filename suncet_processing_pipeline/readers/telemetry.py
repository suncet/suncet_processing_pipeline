"""High-level reader for the per-APID DuckDB telemetry store."""

from __future__ import annotations

from pathlib import Path

import duckdb


class TelemetryReader:
    """Query mission telemetry without assuming equal APID sample times."""

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        if not self.filepath.is_file():
            raise FileNotFoundError(f"Telemetry file not found: {self.filepath}")
        self.connection = duckdb.connect(str(self.filepath), read_only=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def get_catalog(self):
        return self.connection.execute(
            "SELECT apid, packet_name, table_name FROM _apid_catalog ORDER BY apid"
        ).fetchdf()

    def get_packet_types(self):
        return self.get_catalog()["packet_name"].to_numpy()

    def _table(self, packet_type) -> tuple[int, str, str]:
        if isinstance(packet_type, int) or str(packet_type).isdigit():
            row = self.connection.execute(
                "SELECT apid, packet_name, table_name FROM _apid_catalog WHERE apid = ?",
                [int(packet_type)],
            ).fetchone()
        else:
            row = self.connection.execute(
                "SELECT apid, packet_name, table_name FROM _apid_catalog "
                "WHERE lower(packet_name) = lower(?)",
                [str(packet_type)],
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown telemetry packet type/APID: {packet_type!r}")
        return row

    def get_field_names(self, packet_type=None):
        if packet_type is None:
            catalogs = self.get_catalog()
            fields = set()
            for packet_name in catalogs["packet_name"]:
                fields.update(self.get_field_names(packet_name))
            return sorted(fields)
        _apid, _name, table = self._table(packet_type)
        rows = self.connection.execute(f'DESCRIBE "{table}"').fetchall()
        return [
            row[0] for row in rows if not row[0].startswith("_")
        ]

    def query(
        self,
        packet_type,
        *,
        fields=None,
        time_field=None,
        start=None,
        stop=None,
        order_by=None,
    ):
        """Return one APID as a pandas DataFrame with optional time bounds."""
        _apid, _name, table = self._table(packet_type)
        available = self.get_field_names(packet_type)
        selected = available if fields is None else list(fields)
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise KeyError(f"Unknown field(s) for {packet_type}: {', '.join(unknown)}")
        columns = ", ".join(f'"{field}"' for field in selected)
        sql = f'SELECT {columns} FROM "{table}"'
        predicates = []
        parameters = []
        if start is not None or stop is not None:
            if not time_field or time_field not in available:
                raise KeyError("a valid time_field is required for bounded queries")
            if start is not None:
                predicates.append(f'"{time_field}" >= ?')
                parameters.append(start)
            if stop is not None:
                predicates.append(f'"{time_field}" <= ?')
                parameters.append(stop)
        if predicates:
            sql += " WHERE " + " AND ".join(predicates)
        ordering = order_by or time_field
        if ordering is not None:
            if ordering not in available:
                raise KeyError(f"Unknown order field: {ordering}")
            sql += f' ORDER BY "{ordering}"'
        return self.connection.execute(sql, parameters).fetchdf()

    def filter_by_packet_type(self, packet_type, case_sensitive=False):
        del case_sensitive
        return self.query(packet_type)

    def get_field(self, field_name, packet_type=None):
        if packet_type is None:
            matches = [
                packet
                for packet in self.get_packet_types()
                if field_name in self.get_field_names(packet)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Field {field_name!r} occurs in {len(matches)} packet types; "
                    "specify packet_type"
                )
            packet_type = matches[0]
        return self.query(packet_type, fields=[field_name])[field_name].to_numpy()

    def find_fields_containing(self, substring, packet_type=None, case_sensitive=False):
        fields = self.get_field_names(packet_type)
        if case_sensitive:
            return [field for field in fields if substring in field]
        needle = substring.lower()
        return [field for field in fields if needle in field.lower()]

    def get_timestamps(self, packet_type, time_field):
        return self.query(
            packet_type, fields=[time_field], order_by=time_field
        )[time_field].to_numpy()

    def asof_join(
        self,
        primary_packet,
        secondary_packet,
        *,
        primary_time,
        secondary_time,
        secondary_fields,
        tolerance=None,
    ):
        """Attach the latest secondary sample at or before each primary sample."""
        _a, _n, primary = self._table(primary_packet)
        _a, _n, secondary = self._table(secondary_packet)
        primary_fields = self.get_field_names(primary_packet)
        secondary_available = self.get_field_names(secondary_packet)
        if primary_time not in primary_fields or secondary_time not in secondary_available:
            raise KeyError("ASOF time field is not present in its APID table")
        missing = sorted(set(secondary_fields) - set(secondary_available))
        if missing:
            raise KeyError(f"Unknown secondary field(s): {', '.join(missing)}")
        selected = ", ".join(
            f's."{field}" AS "{secondary_packet}_{field}"'
            for field in secondary_fields
        )
        tolerance_clause = ""
        parameters = []
        if tolerance is not None:
            tolerance_clause = f' WHERE p."{primary_time}" - s."{secondary_time}" <= ?'
            parameters.append(tolerance)
        sql = (
            f'SELECT p.*, {selected} FROM "{primary}" p ASOF LEFT JOIN "{secondary}" s '
            f'ON p."{primary_time}" >= s."{secondary_time}"{tolerance_clause} '
            f'ORDER BY p."{primary_time}"'
        )
        return self.connection.execute(sql, parameters).fetchdf()

    def get_summary(self):
        catalog = self.get_catalog()
        counts = []
        for row in catalog.itertuples(index=False):
            count = self.connection.execute(
                f'SELECT count(*) FROM "{row.table_name}"'
            ).fetchone()[0]
            counts.append((int(row.apid), row.packet_name, count))
        return {"file": str(self.filepath), "packet_tables": counts}

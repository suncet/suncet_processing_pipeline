"""Throwaway interactive EPS bus-voltage plot.

Default input is the current TVAC decoded beacon CSV. Run from this repo with:

    conda run -n suncet python quick_interactive_eps_bus_voltage.py

If a window does not open on your machine, try adding ``--backend TkAgg`` or
``--backend MacOSX``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


DEFAULT_BEACON_CSV = Path(
    "/Users/masonjp2/Dropbox/suncet_dropbox/9000 Processing/data/test_data/"
    "2026-05-29_thermal_vacuum_tvac/decoded_packets/decoded_apid_0001_beacon.csv"
)
DEFAULT_MIN_UTC = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
J2000_UTC_EPOCH = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
EPS_BUS_V_FIELD = "beac_ana_eps_bus_v"

# Leap seconds inserted after the J2000 UTC epoch.
LEAP_SECOND_EFFECTIVE_UTC = [
    datetime(2006, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2009, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2012, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2015, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2017, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
]


def parse_utc_datetime(value: str) -> datetime | None:
    if value.strip().lower() == "none":
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def leap_seconds_after_j2000(seconds: float) -> int:
    leap_count = 0
    while True:
        corrected_time = J2000_UTC_EPOCH + timedelta(seconds=float(seconds) + leap_count)
        new_count = sum(
            1 for leap_time in LEAP_SECOND_EFFECTIVE_UTC if corrected_time >= leap_time
        )
        if new_count == leap_count:
            return leap_count
        leap_count = new_count


def j2000_seconds_to_datetime(
    seconds: float,
    *,
    add_leap_seconds: bool,
) -> datetime | None:
    if pd.isna(seconds):
        return None
    seconds_float = float(seconds)
    if add_leap_seconds:
        seconds_float += leap_seconds_after_j2000(seconds_float)
    return J2000_UTC_EPOCH + timedelta(seconds=seconds_float)


def find_time_column(df: pd.DataFrame, requested_time_column: str | None) -> str:
    if requested_time_column:
        if requested_time_column not in df.columns:
            raise KeyError(f"Requested time column not found: {requested_time_column}")
        return requested_time_column
    candidates = [column for column in df.columns if column.startswith("ccsdsSecHeader2_sec")]
    if not candidates:
        raise KeyError("No ccsdsSecHeader2_sec* time column found.")
    beacon_candidates = [column for column in candidates if "beacon" in column.lower()]
    return beacon_candidates[0] if beacon_candidates else candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open an interactive EPS bus-voltage plot from decoded beacon CSV."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_BEACON_CSV)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--min-utc", default=DEFAULT_MIN_UTC.isoformat().replace("+00:00", "Z"))
    parser.add_argument("--no-leap-second-correction", action="store_true")
    parser.add_argument("--no-3sigma", action="store_true")
    parser.add_argument(
        "--backend",
        default=None,
        help="Optional matplotlib GUI backend, e.g. TkAgg, QtAgg, or MacOSX.",
    )
    args = parser.parse_args()

    if args.backend:
        import matplotlib

        matplotlib.use(args.backend)

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.csv.expanduser(), low_memory=False)
    time_column = find_time_column(df, args.time_column)
    if EPS_BUS_V_FIELD not in df.columns:
        raise KeyError(f"Field not found in CSV: {EPS_BUS_V_FIELD}")

    add_leap_seconds = not args.no_leap_second_correction
    df["plot_time_utc"] = df[time_column].map(
        lambda seconds: j2000_seconds_to_datetime(
            seconds,
            add_leap_seconds=add_leap_seconds,
        )
    )
    df[EPS_BUS_V_FIELD] = pd.to_numeric(df[EPS_BUS_V_FIELD], errors="coerce")
    df = df.dropna(subset=["plot_time_utc", EPS_BUS_V_FIELD])
    df = df.sort_values("plot_time_utc").reset_index(drop=True)

    min_utc = parse_utc_datetime(args.min_utc)
    if min_utc is not None:
        df = df.loc[df["plot_time_utc"] >= min_utc].reset_index(drop=True)

    kept = df.copy()
    clipped = pd.DataFrame(columns=df.columns)
    if not args.no_3sigma and len(kept) > 1:
        values = kept[EPS_BUS_V_FIELD]
        mean = values.mean()
        std = values.std()
        if pd.notna(std) and std > 0:
            mask = (values >= mean - 3.0 * std) & (values <= mean + 3.0 * std)
            clipped = kept.loc[~mask]
            kept = kept.loc[mask].reset_index(drop=True)

    print(f"CSV: {args.csv}")
    print(f"time column: {time_column}")
    print(f"plotted points: {len(kept):,}")
    print(f"3sigma clipped points: {len(clipped):,}")
    if len(kept):
        print(
            "time range: "
            f"{kept['plot_time_utc'].iloc[0].isoformat().replace('+00:00', 'Z')} to "
            f"{kept['plot_time_utc'].iloc[-1].isoformat().replace('+00:00', 'Z')}"
        )

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.plot(
        kept["plot_time_utc"],
        kept[EPS_BUS_V_FIELD],
        marker=".",
        markersize=2.5,
        linewidth=0.8,
        color="#1f77b4",
        label=EPS_BUS_V_FIELD,
    )
    if len(clipped):
        ax.scatter(
            clipped["plot_time_utc"],
            clipped[EPS_BUS_V_FIELD],
            s=10,
            color="#b91c1c",
            alpha=0.55,
            label="3sigma clipped",
        )
    ax.set_title(EPS_BUS_V_FIELD)
    ax.set_xlabel("UTC time")
    ax.set_ylabel("V")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%dT%H:%M:%SZ"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
    fig.autofmt_xdate()
    plt.show()


if __name__ == "__main__":
    main()

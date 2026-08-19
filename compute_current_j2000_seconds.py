"""Print current seconds since J2000 midnight, including leap seconds."""

from __future__ import annotations

from datetime import datetime, timezone


J2000_MIDNIGHT_UTC = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# Leap seconds inserted after the J2000 UTC epoch.
LEAP_SECOND_EFFECTIVE_UTC = [
    datetime(2006, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2009, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2012, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2015, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2017, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
]


def main() -> None:
    now_utc = datetime.now(timezone.utc)
    elapsed_utc_seconds = (now_utc - J2000_MIDNIGHT_UTC).total_seconds()
    leap_seconds = sum(1 for leap in LEAP_SECOND_EFFECTIVE_UTC if now_utc >= leap)
    j2000_seconds_with_leaps = elapsed_utc_seconds + leap_seconds

    print(f"now_utc: {now_utc.isoformat().replace('+00:00', 'Z')}")
    print(f"j2000_epoch_utc: {J2000_MIDNIGHT_UTC.isoformat().replace('+00:00', 'Z')}")
    print(f"post_j2000_leap_seconds_included: {leap_seconds}")
    print(f"seconds_since_j2000_midnight_including_leaps: {j2000_seconds_with_leaps:.6f}")
    print(f"integer_seconds_since_j2000_midnight_including_leaps: {int(j2000_seconds_with_leaps)}")


if __name__ == "__main__":
    main()

"""Thread-safe rolling telemetry state for the realtime display."""

from __future__ import annotations

import fnmatch
import math
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from suncet_processing_pipeline.spacecraft_time import (
    combine_spacecraft_time_seconds,
)

from .color_limits import ColorLimitEvaluator
from .packet_decoder import DecodedPacket


DEFAULT_J2000_UTC_EPOCH = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
BEACON_SYSTEM_STATUS_FIELDS = (
    ("CDH", "beac_mode_system_mode", "mode"),
    ("ADCS", "beac_eps_pwr_state_adcs", "power"),
    ("UHF", "beac_eps_pwr_state_uhf", "power"),
    ("X-Band", "beac_eps_pwr_state_xband", "power"),
    ("CSIE", "beac_eps_pwr_state_csie", "power"),
    ("DSPS", "beac_eps_pwr_state_dsps", "power"),
)
LEAP_SECOND_EFFECTIVE_UTC = (
    datetime(2006, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2009, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2012, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2015, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2017, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
)


@dataclass
class TelemetryStats:
    packets_seen: int = 0
    packets_decoded: int = 0
    packets_without_decoder: int = 0
    decode_failures: int = 0
    checksum_failures: int = 0
    time_rejections: int = 0
    selected_points: int = 0
    value_filter_rejections: int = 0
    last_update_wall_time: float | None = None
    packet_counts_by_name: Counter = field(default_factory=Counter)


class TelemetrySelector:
    def __init__(
        self,
        *,
        packet_name_patterns: list[str] | None = None,
        field_patterns: list[str] | None = None,
        excluded_field_patterns: list[str] | None = None,
    ) -> None:
        self.packet_name_patterns = [p.lower() for p in (packet_name_patterns or [])]
        self.field_patterns = [
            p.lower()
            for p in (
                field_patterns
                or [
                    "*temp*",
                    "*temperature*",
                    "csie_det0_therm",
                    "csie_det1_therm",
                    "*bat*v*",
                    "*batt*v*",
                    "*battery*v*",
                    "*bus*v*",
                    "*bus*volt*",
                    "*volt*",
                    "*_v",
                    "*curr*",
                    "*current*",
                    "*_i",
                ]
            )
        ]
        self.excluded_field_patterns = [
            p.lower() for p in (excluded_field_patterns or [])
        ]

    def packet_matches(self, packet_name: str) -> bool:
        if not self.packet_name_patterns:
            return True
        name = packet_name.lower()
        return any(fnmatch.fnmatch(name, pattern) for pattern in self.packet_name_patterns)

    def field_matches(self, field_name: str) -> bool:
        name = field_name.lower()
        if any(fnmatch.fnmatch(name, pattern) for pattern in self.excluded_field_patterns):
            return False
        return any(fnmatch.fnmatch(name, pattern) for pattern in self.field_patterns)


class TelemetryStore:
    def __init__(
        self,
        *,
        selector: TelemetrySelector,
        color_limits: ColorLimitEvaluator | None = None,
        history_points: int = 300,
        stale_after_seconds: float = 10.0,
        j2000_epoch_utc: datetime = DEFAULT_J2000_UTC_EPOCH,
        add_post_j2000_leap_seconds: bool = True,
        min_valid_j2000_seconds: float = 0.0,
        max_plot_time_jump_seconds: float = 300.0,
        max_plot_time_wall_rate: float = 5.0,
        value_filter_enabled: bool = True,
        value_filter_sigma_threshold: float = 3.0,
        value_filter_min_samples: int = 12,
        value_filter_window_points: int = 50,
        value_filter_std_floor: float = 0.05,
        value_filter_relative_std_floor: float = 0.01,
    ) -> None:
        self.selector = selector
        self.color_limits = color_limits or ColorLimitEvaluator()
        self.history_points = history_points
        self.stale_after_seconds = stale_after_seconds
        self.j2000_epoch_utc = _ensure_utc(j2000_epoch_utc)
        self.add_post_j2000_leap_seconds = add_post_j2000_leap_seconds
        self.min_valid_j2000_seconds = min_valid_j2000_seconds
        self.max_plot_time_jump_seconds = max_plot_time_jump_seconds
        self.max_plot_time_wall_rate = max_plot_time_wall_rate
        self.value_filter_enabled = value_filter_enabled
        self.value_filter_sigma_threshold = value_filter_sigma_threshold
        self.value_filter_min_samples = value_filter_min_samples
        self.value_filter_window_points = value_filter_window_points
        self.value_filter_std_floor = value_filter_std_floor
        self.value_filter_relative_std_floor = value_filter_relative_std_floor
        self.stats = TelemetryStats()
        self.latest: dict[str, dict[str, Any]] = {}
        self.history: dict[str, deque[dict[str, Any]]] = {}
        self._accepted_packet_times: dict[str, tuple[float, float, str]] = {}
        self.last_packet: dict[str, Any] | None = None
        self.onboard_utc: str | None = None
        self.beacon_power_states: list[dict[str, Any]] = []
        self.beacon_power_received_time: float | None = None
        self.version = 0
        self._condition = threading.Condition()

    def add_packet(self, packet: DecodedPacket) -> None:
        now = time.time()
        with self._condition:
            self.stats.packets_seen += 1
            self.stats.packet_counts_by_name[packet.packet_name] += 1
            if packet.decode_status == "decoded":
                self.stats.packets_decoded += 1
            elif packet.decode_status == "no_decoder":
                self.stats.packets_without_decoder += 1
            elif packet.decode_status == "decode_failed":
                self.stats.decode_failures += 1
            if packet.checksum_status == "failed":
                self.stats.checksum_failures += 1

            packet_time, packet_time_source = self._packet_time(packet, now)
            time_rejection_reason = self._packet_time_rejection_reason(
                packet.packet_name,
                packet_time,
                packet_time_source,
                now,
            )
            if time_rejection_reason is None:
                self._accepted_packet_times[packet.packet_name] = (
                    packet_time,
                    now,
                    packet_time_source,
                )
            else:
                self.stats.time_rejections += 1
            packet_onboard_utc = (
                self._packet_onboard_utc(packet)
                if time_rejection_reason is None
                else None
            )
            if packet_onboard_utc is not None:
                self.onboard_utc = packet_onboard_utc
            beacon_power_states = self._packet_beacon_system_states(packet, now)
            if beacon_power_states:
                self.beacon_power_states = beacon_power_states
                self.beacon_power_received_time = now
            selected = 0
            filtered = 0
            filter_reasons: list[str] = []
            if (
                time_rejection_reason is None
                and packet.decode_status == "decoded"
                and self.selector.packet_matches(packet.packet_name)
            ):
                for field_name, value in packet.fields.items():
                    numeric = _as_float(value)
                    if numeric is None:
                        continue
                    if not self.selector.field_matches(field_name):
                        continue
                    value_rejection_reason = self._value_filter_rejection_reason(
                        field_name, numeric
                    )
                    if value_rejection_reason is not None:
                        filtered += 1
                        if len(filter_reasons) < 5:
                            filter_reasons.append(value_rejection_reason)
                        continue
                    selected += 1
                    point = {
                        "field": field_name,
                        "value": numeric,
                        "packet_time": packet_time,
                        "packet_utc": packet_onboard_utc,
                        "received_time": now,
                        "packet_name": packet.packet_name,
                        "apid": packet.apid,
                        "limit_status": self.color_limits.evaluate(field_name, numeric),
                    }
                    self.latest[field_name] = point
                    self.history.setdefault(
                        field_name, deque(maxlen=self.history_points)
                    ).append(point)

            self.stats.selected_points += selected
            self.stats.value_filter_rejections += filtered
            self.stats.last_update_wall_time = now
            self.last_packet = {
                "apid": packet.apid,
                "packet_name": packet.packet_name,
                "packet_time": packet_time,
                "onboard_utc": packet_onboard_utc,
                "decode_status": packet.decode_status,
                "decode_error": packet.decode_error,
                "decoder_kind": packet.decoder_kind,
                "checksum_status": packet.checksum_status,
                "source": packet.source,
                "selected_points": selected,
                "filtered_points": filtered,
                "value_filter_rejection_reasons": filter_reasons,
                "time_source": packet_time_source,
                "time_status": (
                    "accepted" if time_rejection_reason is None else "rejected"
                ),
                "time_rejection_reason": time_rejection_reason or "",
                "received_time": now,
            }
            self.version += 1
            self._condition.notify_all()

    def clear_history(self) -> None:
        with self._condition:
            self.latest.clear()
            self.history.clear()
            self._accepted_packet_times.clear()
            self.version += 1
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return self._snapshot_locked()

    def wait_for_snapshot(
        self, version: int | None, timeout: float
    ) -> tuple[int, dict[str, Any]]:
        with self._condition:
            if version is not None and version == self.version:
                self._condition.wait(timeout=timeout)
            return self.version, self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        now = time.time()
        fields = []
        for field_name, point in sorted(self.latest.items()):
            age = now - float(point["received_time"])
            fields.append(
                {
                    **point,
                    "age_seconds": age,
                    "stale": age > self.stale_after_seconds,
                    "history": list(self.history.get(field_name, ())),
                }
            )
        return {
            "version": self.version,
            "now": now,
            "onboard_utc": self.onboard_utc,
            "beacon_power_states": {
                "updated_time": self.beacon_power_received_time,
                "age_seconds": (
                    now - self.beacon_power_received_time
                    if self.beacon_power_received_time is not None
                    else None
                ),
                "states": self.beacon_power_states,
            },
            "stats": {
                "packets_seen": self.stats.packets_seen,
                "packets_decoded": self.stats.packets_decoded,
                "packets_without_decoder": self.stats.packets_without_decoder,
                "decode_failures": self.stats.decode_failures,
                "checksum_failures": self.stats.checksum_failures,
                "time_rejections": self.stats.time_rejections,
                "selected_points": self.stats.selected_points,
                "value_filter_rejections": self.stats.value_filter_rejections,
                "last_update_wall_time": self.stats.last_update_wall_time,
                "packet_counts_by_name": dict(self.stats.packet_counts_by_name),
            },
            "last_packet": self.last_packet,
            "fields": fields,
        }

    def _packet_time(self, packet: DecodedPacket, fallback: float) -> tuple[float, str]:
        secondary = self._secondary_header_seconds(packet)
        if secondary is not None:
            return secondary, "ccsds_secondary_header"
        if any(
            name.startswith("ccsdsSecHeader2_sec") for name in packet.fields
        ):
            return float("nan"), "ccsds_secondary_header"

        explicit = self._explicit_packet_time(packet)
        if explicit is not None:
            return explicit
        return fallback, "received_wall_time"

    @staticmethod
    def _explicit_packet_time(packet: DecodedPacket) -> tuple[float, str] | None:
        preferred = (
            "beac_time_mission_elapsed_time",
            "beac_time_since_boot",
            "sw_time_since_boot",
            "timestamp_seconds_since_boot",
        )
        for name in preferred:
            value = _as_float(packet.fields.get(name))
            if value is not None:
                return value, name
        return None

    def _packet_time_rejection_reason(
        self,
        packet_name: str,
        packet_time: float,
        packet_time_source: str,
        now: float,
    ) -> str | None:
        if not math.isfinite(packet_time):
            return "packet time is not finite"
        if packet_time < 0:
            return "packet time is negative"
        min_j2000 = self.min_valid_j2000_seconds
        if (
            packet_time_source == "ccsds_secondary_header"
            and min_j2000 > 0
            and packet_time < min_j2000
        ):
            return (
                f"secondary-header J2000 time {packet_time:.1f}s is below "
                f"minimum {min_j2000:.1f}s"
            )
        max_jump = self.max_plot_time_jump_seconds
        if max_jump <= 0:
            return None
        previous = self._accepted_packet_times.get(packet_name)
        if previous is None:
            return None
        previous_time, previous_wall_time, previous_source = previous
        if previous_source != packet_time_source:
            return None
        wall_elapsed = max(0.0, now - previous_wall_time)
        rate = max(0.0, self.max_plot_time_wall_rate)
        allowed_jump = max_jump + wall_elapsed * rate
        time_delta = packet_time - previous_time
        if abs(time_delta) <= allowed_jump:
            return None
        return (
            f"packet time jumped {time_delta:.1f}s from previous "
            f"{packet_name} packet; limit {allowed_jump:.1f}s"
        )

    def _value_filter_rejection_reason(
        self,
        field_name: str,
        value: float,
    ) -> str | None:
        if not self.value_filter_enabled:
            return None
        threshold = self.value_filter_sigma_threshold
        if threshold <= 0:
            return None
        history = self.history.get(field_name)
        if not history:
            return None
        window_points = max(1, self.value_filter_window_points)
        values = [float(point["value"]) for point in list(history)[-window_points:]]
        if len(values) < max(1, self.value_filter_min_samples):
            return None
        mean = sum(values) / len(values)
        variance = sum((sample - mean) ** 2 for sample in values) / len(values)
        std = math.sqrt(max(0.0, variance))
        std_floor = max(
            0.0,
            self.value_filter_std_floor,
            abs(mean) * max(0.0, self.value_filter_relative_std_floor),
        )
        effective_std = max(std, std_floor)
        if effective_std <= 0:
            return None
        sigma = abs(value - mean) / effective_std
        if sigma <= threshold:
            return None
        return (
            f"{field_name} value {value:.3g} is {sigma:.1f} sigma from "
            f"recent mean {mean:.3g}"
        )

    @staticmethod
    def _secondary_header_seconds(packet: DecodedPacket) -> float | None:
        for name, value in packet.fields.items():
            if name.startswith("ccsdsSecHeader2_sec"):
                coarse = _as_float(value)
                if coarse is None:
                    return None
                suffix = name[len("ccsdsSecHeader2_sec") :]
                fine = _as_float(
                    packet.fields.get(f"ccsdsSecHeader2_sub{suffix}", 0)
                )
                if fine is None:
                    return None
                try:
                    return combine_spacecraft_time_seconds(coarse, fine)
                except ValueError:
                    return None
        return None

    def _onboard_utc(self, packet: DecodedPacket) -> str | None:
        seconds = self._secondary_header_seconds(packet)
        if seconds is None:
            return None
        if self.add_post_j2000_leap_seconds:
            seconds += _leap_seconds_after_j2000(seconds, self.j2000_epoch_utc)
        utc = self.j2000_epoch_utc + timedelta(seconds=seconds)
        timespec = "milliseconds" if utc.microsecond else "seconds"
        return utc.isoformat(timespec=timespec).replace("+00:00", "Z")

    def _packet_onboard_utc(self, packet: DecodedPacket) -> str | None:
        return self._onboard_utc(packet)

    @staticmethod
    def _packet_beacon_system_states(
        packet: DecodedPacket, received_time: float
    ) -> list[dict[str, Any]]:
        if packet.decode_status != "decoded" or packet.packet_name.lower() != "beacon":
            return []
        states = []
        found_any = False
        for label, field_name, kind in BEACON_SYSTEM_STATUS_FIELDS:
            value = packet.fields.get(field_name)
            if value is not None:
                found_any = True
            state, display_value = _system_status(kind, value)
            states.append(
                {
                    "label": label,
                    "field": field_name,
                    "kind": kind,
                    "value": value,
                    "state": state,
                    "display_value": display_value,
                    "received_time": received_time,
                }
            )
        return states if found_any else []


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _system_status(kind: str, value: Any) -> tuple[str, str]:
    if kind == "mode":
        return _mode_state(value)
    state = _power_state(value)
    return state, state.upper() if state in ("on", "off") else "?"


def _mode_state(value: Any) -> tuple[str, str]:
    if value is None:
        return "unknown", "?"
    if isinstance(value, str):
        mode = value.strip().upper()
        if mode in ("PHOENIX", "SAFE", "SCIENCE", "DOWNLINK"):
            return _mode_status_class(mode), _mode_display(mode)
        numeric = _as_float(value)
        if numeric is None:
            return "unknown", mode or "?"
        numeric_value = numeric
    else:
        numeric_value = _as_float(value)
    mode_by_value = {
        0: "PHOENIX",
        1: "SAFE",
        2: "SCIENCE",
        3: "DOWNLINK",
    }
    mode = (
        mode_by_value.get(int(numeric_value))
        if numeric_value is not None and numeric_value.is_integer()
        else None
    )
    if mode is None:
        return "unknown", str(value)
    return _mode_status_class(mode), _mode_display(mode)


def _mode_status_class(mode: str) -> str:
    if mode == "PHOENIX":
        return "red"
    if mode == "SAFE":
        return "yellow"
    if mode in ("SCIENCE", "DOWNLINK"):
        return "green"
    return "unknown"


def _mode_display(mode: str) -> str:
    return f"{mode} MODE"


def _power_state(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "on":
            return "on"
        if normalized == "off":
            return "off"
        numeric = _as_float(value)
        if numeric is None:
            return "unknown"
        value = numeric
    if isinstance(value, bool):
        return "on" if value else "off"
    if value == 1:
        return "on"
    if value == 0:
        return "off"
    return "unknown"


def _leap_seconds_after_j2000(seconds: float, epoch_utc: datetime) -> int:
    epoch_utc = _ensure_utc(epoch_utc)
    leap_count = 0
    while True:
        corrected_time = epoch_utc + timedelta(seconds=float(seconds) + leap_count)
        new_count = sum(
            1 for leap_time in LEAP_SECOND_EFFECTIVE_UTC if corrected_time >= leap_time
        )
        if new_count == leap_count:
            return leap_count
        leap_count = new_count

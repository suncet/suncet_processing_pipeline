"""Configuration loading for the realtime telemetry display."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.ini"


def _bool(parser: configparser.ConfigParser, section: str, option: str, fallback: bool) -> bool:
    return parser.getboolean(section, option, fallback=fallback)


def _float(
    parser: configparser.ConfigParser, section: str, option: str, fallback: float
) -> float:
    return parser.getfloat(section, option, fallback=fallback)


def _int(parser: configparser.ConfigParser, section: str, option: str, fallback: int) -> int:
    return parser.getint(section, option, fallback=fallback)


def _int_auto(
    parser: configparser.ConfigParser, section: str, option: str, fallback: int
) -> int:
    value = parser.get(section, option, fallback=str(fallback)).strip()
    return int(value, 0)


def _bytes_auto(
    parser: configparser.ConfigParser, section: str, option: str, fallback: bytes
) -> bytes:
    value = parser.get(section, option, fallback=fallback.hex()).strip()
    if value.lower().startswith("0x"):
        value = value[2:]
    value = value.replace(" ", "").replace("_", "").replace("-", "")
    if len(value) % 2:
        raise ValueError(f"{section}.{option} must contain an even number of hex digits")
    return bytes.fromhex(value)


def _list(parser: configparser.ConfigParser, section: str, option: str) -> list[str]:
    value = parser.get(section, option, fallback="")
    return [line.strip() for line in value.splitlines() if line.strip()]


def _int_list_auto(
    parser: configparser.ConfigParser, section: str, option: str
) -> list[int]:
    values = _list(parser, section, option)
    return [int(value, 0) for value in values]


def _datetime_utc(
    parser: configparser.ConfigParser, section: str, option: str, fallback: str
) -> datetime:
    value = parser.get(section, option, fallback=fallback).strip()
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class TcpConfig:
    host: str
    port: int
    reconnect: bool
    reconnect_seconds: float
    idle_reconnect_seconds: float
    read_bytes: int


@dataclass(frozen=True)
class PipelineConfig:
    processing_config: Path


@dataclass(frozen=True)
class FramingConfig:
    allow_ccsds_asm: bool
    ccsds_asm: bytes
    allow_frame_prefix: bool
    frame_prefix: bytes
    frame_prefix_bytes: int
    allow_ax25_header: bool
    allow_missing_ax25_header: bool
    ax25_destination: str
    ax25_source: str
    ax25_ctrl: int
    ax25_pid: int
    segmentation_enabled: bool
    segmented_packet_threshold_bytes: int
    max_segment_payload_bytes: int
    segment_flag_start: int
    segment_flag_middle: int
    segment_flag_end: int
    max_packet_bytes: int
    max_buffer_bytes: int


@dataclass(frozen=True)
class DecodeConfig:
    require_packet_checksums: bool
    drop_failed_checksums: bool
    ignored_apids: list[int]
    ignored_packet_name_patterns: list[str]


@dataclass(frozen=True)
class LimitsConfig:
    color_limits_xml: Path | None


@dataclass(frozen=True)
class TimeConfig:
    j2000_epoch_utc: datetime
    add_post_j2000_leap_seconds: bool
    min_valid_j2000_seconds: float
    max_plot_time_jump_seconds: float
    max_plot_time_wall_rate: float


@dataclass(frozen=True)
class ValueFilterConfig:
    enabled: bool
    sigma_threshold: float
    min_samples: int
    window_points: int
    std_floor: float
    relative_std_floor: float


@dataclass(frozen=True)
class DisplayConfig:
    host: str
    port: int
    history_points: int
    stale_after_seconds: float
    sse_heartbeat_seconds: float


@dataclass(frozen=True)
class TelemetryConfig:
    packet_name_patterns: list[str]
    field_patterns: list[str]
    excluded_field_patterns: list[str]


@dataclass(frozen=True)
class RealtimeDisplayConfig:
    path: Path
    tcp: TcpConfig
    pipeline: PipelineConfig
    framing: FramingConfig
    decode: DecodeConfig
    limits: LimitsConfig
    time: TimeConfig
    value_filter: ValueFilterConfig
    display: DisplayConfig
    telemetry: TelemetryConfig


def load_config(path: str | Path | None = None) -> RealtimeDisplayConfig:
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    config_path = config_path.resolve()
    parser = configparser.ConfigParser()
    read = parser.read(config_path)
    if not read:
        raise FileNotFoundError(f"Realtime display config not found: {config_path}")

    processing_config = Path(
        parser.get(
            "pipeline",
            "processing_config",
            fallback="../config_files/config_default.ini",
        )
    ).expanduser()
    if not processing_config.is_absolute():
        processing_config = (config_path.parent / processing_config).resolve()

    limits_path_text = parser.get(
        "limits", "color_limits_xml", fallback="color_limits_tlm.xml"
    ).strip()
    limits_path: Path | None
    if limits_path_text.lower() in ("", "none", "false"):
        limits_path = None
    else:
        limits_path = Path(limits_path_text).expanduser()
        if not limits_path.is_absolute():
            limits_path = (config_path.parent / limits_path).resolve()

    return RealtimeDisplayConfig(
        path=config_path,
        tcp=TcpConfig(
            host=parser.get("tcp", "host", fallback="127.0.0.1"),
            port=_int(parser, "tcp", "port", 5000),
            reconnect=_bool(parser, "tcp", "reconnect", True),
            reconnect_seconds=_float(parser, "tcp", "reconnect_seconds", 2.0),
            idle_reconnect_seconds=_float(
                parser, "tcp", "idle_reconnect_seconds", 15.0
            ),
            read_bytes=_int(parser, "tcp", "read_bytes", 4096),
        ),
        pipeline=PipelineConfig(processing_config=processing_config),
        framing=FramingConfig(
            allow_ccsds_asm=_bool(parser, "framing", "allow_ccsds_asm", True),
            ccsds_asm=_bytes_auto(
                parser, "framing", "ccsds_asm", b"\x1a\xcf\xfc\x1d"
            ),
            allow_frame_prefix=_bool(parser, "framing", "allow_frame_prefix", False),
            frame_prefix=_bytes_auto(
                parser, "framing", "frame_prefix", b"\x1b\xad\xca\xfe"
            ),
            frame_prefix_bytes=_int(parser, "framing", "frame_prefix_bytes", 16),
            allow_ax25_header=_bool(parser, "framing", "allow_ax25_header", True),
            allow_missing_ax25_header=_bool(
                parser, "framing", "allow_missing_ax25_header", True
            ),
            ax25_destination=parser.get("framing", "ax25_destination", fallback="SUN1"),
            ax25_source=parser.get("framing", "ax25_source", fallback="LASP"),
            ax25_ctrl=_int_auto(parser, "framing", "ax25_ctrl", 0x03),
            ax25_pid=_int_auto(parser, "framing", "ax25_pid", 0xF0),
            segmentation_enabled=_bool(parser, "framing", "segmentation_enabled", False),
            segmented_packet_threshold_bytes=_int(
                parser, "framing", "segmented_packet_threshold_bytes", 244
            ),
            max_segment_payload_bytes=_int(
                parser, "framing", "max_segment_payload_bytes", 244
            ),
            segment_flag_start=_int(parser, "framing", "segment_flag_start", 1),
            segment_flag_middle=_int(parser, "framing", "segment_flag_middle", 0),
            segment_flag_end=_int(parser, "framing", "segment_flag_end", 2),
            max_packet_bytes=_int(parser, "framing", "max_packet_bytes", 65535),
            max_buffer_bytes=_int(parser, "framing", "max_buffer_bytes", 1048576),
        ),
        decode=DecodeConfig(
            require_packet_checksums=_bool(
                parser, "decode", "require_packet_checksums", False
            ),
            drop_failed_checksums=_bool(parser, "decode", "drop_failed_checksums", False),
            ignored_apids=_int_list_auto(parser, "decode", "ignored_apids"),
            ignored_packet_name_patterns=_list(
                parser, "decode", "ignored_packet_name_patterns"
            ),
        ),
        limits=LimitsConfig(color_limits_xml=limits_path),
        time=TimeConfig(
            j2000_epoch_utc=_datetime_utc(
                parser, "time", "j2000_epoch_utc", "2000-01-01T00:00:00Z"
            ),
            add_post_j2000_leap_seconds=_bool(
                parser, "time", "add_post_j2000_leap_seconds", True
            ),
            min_valid_j2000_seconds=_float(
                parser, "time", "min_valid_j2000_seconds", 0.0
            ),
            max_plot_time_jump_seconds=_float(
                parser, "time", "max_plot_time_jump_seconds", 300.0
            ),
            max_plot_time_wall_rate=_float(
                parser, "time", "max_plot_time_wall_rate", 5.0
            ),
        ),
        value_filter=ValueFilterConfig(
            enabled=_bool(parser, "value_filter", "enabled", True),
            sigma_threshold=_float(parser, "value_filter", "sigma_threshold", 3.0),
            min_samples=_int(parser, "value_filter", "min_samples", 12),
            window_points=_int(parser, "value_filter", "window_points", 50),
            std_floor=_float(parser, "value_filter", "std_floor", 0.05),
            relative_std_floor=_float(
                parser, "value_filter", "relative_std_floor", 0.01
            ),
        ),
        display=DisplayConfig(
            host=parser.get("display", "host", fallback="127.0.0.1"),
            port=_int(parser, "display", "port", 8050),
            history_points=_int(parser, "display", "history_points", 300),
            stale_after_seconds=_float(parser, "display", "stale_after_seconds", 10.0),
            sse_heartbeat_seconds=_float(
                parser, "display", "sse_heartbeat_seconds", 15.0
            ),
        ),
        telemetry=TelemetryConfig(
            packet_name_patterns=_list(parser, "telemetry", "packet_name_patterns"),
            field_patterns=_list(parser, "telemetry", "field_patterns"),
            excluded_field_patterns=_list(
                parser, "telemetry", "excluded_field_patterns"
            ),
        ),
    )

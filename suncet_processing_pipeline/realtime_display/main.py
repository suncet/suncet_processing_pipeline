"""Command-line entry point for the SunCET realtime telemetry display."""

from __future__ import annotations

import argparse
import fnmatch
import math
import signal
import socket
import sys
import time
import traceback
from pathlib import Path

from .color_limits import ColorLimitEvaluator
from .config import DEFAULT_CONFIG_PATH, DecodeConfig, load_config
from .dashboard_server import DashboardServer
from .packet_decoder import DecodedPacket, RealtimePacketDecoder
from .telemetry_state import TelemetrySelector, TelemetryStore
from .uhf_realtime_parser import UHFRealtimeParser


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)

    selector = TelemetrySelector(
        packet_name_patterns=config.telemetry.packet_name_patterns,
        field_patterns=config.telemetry.field_patterns,
        excluded_field_patterns=config.telemetry.excluded_field_patterns,
    )
    store = TelemetryStore(
        selector=selector,
        color_limits=ColorLimitEvaluator.from_xml(config.limits.color_limits_xml),
        history_points=config.display.history_points,
        stale_after_seconds=config.display.stale_after_seconds,
        j2000_epoch_utc=config.time.j2000_epoch_utc,
        add_post_j2000_leap_seconds=config.time.add_post_j2000_leap_seconds,
        min_valid_j2000_seconds=config.time.min_valid_j2000_seconds,
        max_plot_time_jump_seconds=config.time.max_plot_time_jump_seconds,
        max_plot_time_wall_rate=config.time.max_plot_time_wall_rate,
        value_filter_enabled=config.value_filter.enabled,
        value_filter_sigma_threshold=config.value_filter.sigma_threshold,
        value_filter_min_samples=config.value_filter.min_samples,
        value_filter_window_points=config.value_filter.window_points,
        value_filter_std_floor=config.value_filter.std_floor,
        value_filter_relative_std_floor=config.value_filter.relative_std_floor,
    )
    server = DashboardServer(
        store=store,
        host=config.display.host,
        port=config.display.port,
        heartbeat_seconds=config.display.sse_heartbeat_seconds,
    )
    server.start()
    print(f"Dashboard: {server.url}")

    stop = _StopFlag()

    def _signal_handler(_signum, _frame):
        stop.stop = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        if args.demo:
            _run_demo(store, stop)
            return 0

        decoder = RealtimePacketDecoder(
            config.pipeline.processing_config,
            require_packet_checksums=config.decode.require_packet_checksums,
            drop_failed_checksums=config.decode.drop_failed_checksums,
        )
        for warning in decoder.metadata.warnings:
            print(f"Decoder warning: {warning}", file=sys.stderr)

        parser = UHFRealtimeParser(
            config.framing,
            valid_apids=decoder.metadata.valid_apids,
        )

        if args.replay:
            _run_replay(
                Path(args.replay).expanduser(),
                parser,
                decoder,
                store,
                chunk_bytes=config.tcp.read_bytes,
                delay_seconds=args.replay_delay,
                decode_config=config.decode,
                stop=stop,
            )
        else:
            _run_tcp_client(config, parser, decoder, store, stop)
    finally:
        server.stop()
    return 0


class _StopFlag:
    stop = False


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SunCET realtime UHF telemetry display."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Realtime display config.ini path.",
    )
    parser.add_argument(
        "--replay",
        help="Read bytes from a captured file instead of connecting to TCP.",
    )
    parser.add_argument(
        "--replay-delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between replay chunks.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate synthetic dashboard values without a TCP connection.",
    )
    return parser.parse_args(argv)


def _run_tcp_client(
    config,
    parser: UHFRealtimeParser,
    decoder: RealtimePacketDecoder,
    store: TelemetryStore,
    stop: _StopFlag,
) -> None:
    host = config.tcp.host
    port = config.tcp.port
    while not stop.stop:
        try:
            print(f"Connecting to UHF TCP stream at {host}:{port}...")
            with socket.create_connection((host, port), timeout=10) as sock:
                sock.settimeout(1.0)
                print("Connected.")
                last_receive_monotonic = time.monotonic()
                while not stop.stop:
                    try:
                        chunk = sock.recv(config.tcp.read_bytes)
                    except socket.timeout:
                        idle_seconds = time.monotonic() - last_receive_monotonic
                        idle_limit = config.tcp.idle_reconnect_seconds
                        if idle_limit > 0 and idle_seconds >= idle_limit:
                            print(
                                "TCP stream idle for "
                                f"{idle_seconds:.1f}s; reconnecting..."
                            )
                            break
                        continue
                    if not chunk:
                        print("TCP stream closed by peer.")
                        break
                    last_receive_monotonic = time.monotonic()
                    _process_chunk(chunk, parser, decoder, store, config.decode)
        except OSError as exc:
            print(f"TCP connection error: {exc}", file=sys.stderr)
        if not config.tcp.reconnect:
            break
        time.sleep(config.tcp.reconnect_seconds)


def _run_replay(
    path: Path,
    parser: UHFRealtimeParser,
    decoder: RealtimePacketDecoder,
    store: TelemetryStore,
    *,
    chunk_bytes: int,
    delay_seconds: float,
    decode_config: DecodeConfig,
    stop: _StopFlag,
) -> None:
    print(f"Replaying {path}")
    with path.open("rb") as handle:
        while not stop.stop:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            _process_chunk(chunk, parser, decoder, store, decode_config)
            if delay_seconds > 0:
                time.sleep(delay_seconds)
    for envelope in parser.flush():
        decoded = decoder.decode(envelope.packet, source=envelope.source)
        if decoded is not None and not _packet_ignored(decoded, decode_config):
            store.add_packet(decoded)
    print("Replay complete. Press Ctrl-C to stop the dashboard.")
    while not stop.stop:
        time.sleep(0.2)


def _process_chunk(
    chunk: bytes,
    parser: UHFRealtimeParser,
    decoder: RealtimePacketDecoder,
    store: TelemetryStore,
    decode_config: DecodeConfig,
) -> None:
    try:
        envelopes = parser.feed(chunk)
    except Exception as exc:
        _log_processing_error("parser", exc)
        _clear_parser_buffer(parser)
        return

    for envelope in envelopes:
        try:
            decoded = decoder.decode(envelope.packet, source=envelope.source)
            if decoded is not None and not _packet_ignored(decoded, decode_config):
                store.add_packet(decoded)
        except Exception as exc:
            _log_processing_error("decoder/store", exc)


def _packet_ignored(packet: DecodedPacket, decode_config: DecodeConfig) -> bool:
    if packet.apid in set(decode_config.ignored_apids):
        return True
    packet_name = packet.packet_name.lower()
    return any(
        fnmatch.fnmatch(packet_name, pattern.lower())
        for pattern in decode_config.ignored_packet_name_patterns
    )


def _log_processing_error(stage: str, exc: Exception) -> None:
    print(
        f"Realtime {stage} error: {type(exc).__name__}: {exc}",
        file=sys.stderr,
    )
    traceback.print_exc()


def _clear_parser_buffer(parser: UHFRealtimeParser) -> None:
    dropped = len(parser.buffer)
    if dropped == 0:
        return
    parser.buffer.clear()
    parser.stats.bytes_dropped += dropped
    print(
        f"Cleared {dropped} buffered parser bytes after processing error.",
        file=sys.stderr,
    )


def _run_demo(store: TelemetryStore, stop: _StopFlag) -> None:
    print("Running synthetic dashboard demo. Press Ctrl-C to stop.")
    fields = {
        "beac_ana_bat1_v": 14.8,
        "beac_ana_bat2_v": 14.7,
        "beac_ana_sa_8_cell_str_v": 18.2,
        "beac_ana_sa_9_cell_str_v": 18.5,
        "beac_ana_eps_bus_v": 12.1,
        "beac_ana_xact_v": 5.2,
        "beac_ana_uhf_v": 5.0,
        "beac_ana_xband_v": 5.1,
        "beac_batt1_charge_current": 0.55,
        "beac_batt2_charge_current": 0.48,
        "beac_ana_sa_8_cell_str_i": 0.65,
        "beac_ana_sa_9_cell_str_i": 0.72,
        "beac_ana_eps_bus_i": 0.62,
        "beac_ana_xact_i": 0.55,
        "beac_ana_uhf_i": 0.44,
        "beac_ana_xband_i": 0.62,
        "beac_ana_cdh_temp": 22.0,
        "beac_batt1_temp": 18.0,
        "beac_batt2_temp": 18.4,
        "beac_uhf_temp": 26.0,
        "beac_dsps_sensor_board_temp": 24.0,
    }
    tick = 0
    while not stop.stop:
        now = time.time()
        values = {
            name: base + 0.35 * math.sin(tick / 12.0 + idx)
            for idx, (name, base) in enumerate(fields.items())
        }
        values.update(
            {
                "beac_mode_system_mode": "SCIENCE",
                "beac_eps_pwr_state_adcs": 1,
                "beac_eps_pwr_state_uhf": 1,
                "beac_eps_pwr_state_xband": 0,
                "beac_eps_pwr_state_csie": 1,
                "beac_eps_pwr_state_dsps": 1,
            }
        )
        values["beac_time_since_boot"] = tick
        values["ccsdsSecHeader2_sec_beacon"] = 833314000 + tick
        values["ccsdsSecHeader2_sub_beacon"] = 0
        store.add_packet(
            DecodedPacket(
                apid=1,
                packet_name="BEACON",
                fields=values,
                header={"apid": 1},
                decode_status="decoded",
                decoder_kind="synthetic",
                source="demo",
            )
        )
        tick += 1
        sleep_until = now + 1.0
        while not stop.stop and time.time() < sleep_until:
            time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())

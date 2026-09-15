#!/usr/bin/env python3
"""Measure one external command with Jetson ``tegrastats``.

The timed interval begins immediately before spawning the child and ends after
the child exits.  It therefore includes child-process startup and every read,
calculation, and write performed by that command.  Harness startup, telemetry
priming, the optional settling interval, and result-JSON writing are excluded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    # Permit direct execution from outside the repository root on the Jetson.
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from power_telemetry import TegrastatsSampler, summarize_interval


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--command-cwd", type=Path, default=Path.cwd())
    parser.add_argument(
        "--pre-run-idle-seconds",
        type=_nonnegative_float,
        default=0.0,
        help="settling interval before the command; excluded from command energy",
    )
    parser.add_argument(
        "--telemetry", choices=("auto", "required", "off"), default="required"
    )
    parser.add_argument("--tegrastats-path")
    parser.add_argument(
        "--telemetry-interval-ms", type=_positive_integer, default=100
    )
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _atomic_json(path: Path, payload: dict, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # link(2) publishes the complete file atomically and, unlike
            # replace(2), cannot overwrite a destination created in a race.
            os.link(temporary, path)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def _quality_warnings(summary: dict, interval_ms: int) -> list[str]:
    warnings: list[str] = []
    if not summary["fully_power_covered"]:
        warnings.append(
            "The command was not bracketed by complete power samples; gross "
            "energy is unavailable."
        )
    if summary["power_sample_count_inside_interval"] < 3:
        warnings.append(
            "Fewer than three telemetry samples fell inside the command interval."
        )
    if summary["duration_seconds"] < 100 * interval_ms / 1000:
        warnings.append(
            "The command is shorter than 100 telemetry intervals; repeat or batch "
            "the workload before interpreting energy or sampled peak power."
        )
    return warnings


def _shell_exit_code(returncode: int) -> int:
    if returncode < 0:
        return min(255, 128 - returncode)
    return min(255, returncode)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        separator_index = raw_argv.index("--")
    except ValueError:
        print("ERROR: an explicit -- must precede the measured command", file=sys.stderr)
        return 2
    arguments = _parser().parse_args(raw_argv[:separator_index])
    command = raw_argv[separator_index + 1 :]
    if not command:
        print("ERROR: provide a command after --", file=sys.stderr)
        return 2

    output_path = arguments.output_json.expanduser().resolve()
    if output_path.exists() and not arguments.overwrite:
        print(f"ERROR: Refusing to overwrite existing result: {output_path}", file=sys.stderr)
        return 2
    command_cwd = arguments.command_cwd.expanduser().resolve()
    if not command_cwd.is_dir():
        print(f"ERROR: command working directory does not exist: {command_cwd}", file=sys.stderr)
        return 2

    sampler = TegrastatsSampler(
        interval_ms=arguments.telemetry_interval_ms,
        executable=arguments.tegrastats_path,
        mode=arguments.telemetry,
    )
    origin_ns = time.monotonic_ns()
    report = {
        "schema": "suncet.external_command_power",
        "schema_version": 1,
        "status": "running",
        "command": command,
        "command_cwd": str(command_cwd),
        "pre_run_idle_seconds": arguments.pre_run_idle_seconds,
        "telemetry_interval_ms": arguments.telemetry_interval_ms,
    }
    returncode: int | None = None
    start_ns: int | None = None
    end_ns: int | None = None
    try:
        sampler.start()
        if sampler.status == "collecting":
            primed = sampler.wait_for_power_sample(timeout_seconds=3.0)
            if arguments.telemetry == "required" and not primed:
                raise RuntimeError("Required tegrastats power telemetry did not become ready")
        if arguments.pre_run_idle_seconds:
            time.sleep(arguments.pre_run_idle_seconds)

        start_utc = _utc_now()
        start_ns = time.monotonic_ns()
        completed = subprocess.run(
            command,
            cwd=command_cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        end_ns = time.monotonic_ns()
        end_utc = _utc_now()
        returncode = completed.returncode
        report.update(
            {
                "command_started_utc": start_utc,
                "command_finished_utc": end_utc,
                "command_returncode": returncode,
                "command_stdout": completed.stdout,
                "command_stderr": completed.stderr,
            }
        )
        report["command_status"] = "passed" if returncode == 0 else "failed"
        report["status"] = "passed" if returncode == 0 else "command_failed"

        post_sample = sampler.wait_for_power_sample(
            after_monotonic_ns=end_ns,
            timeout_seconds=max(2.0, 4 * arguments.telemetry_interval_ms / 1000),
        )
        if arguments.telemetry == "required" and not post_sample:
            report["status"] = "measurement_failed"
            report["measurement_error"] = (
                "Required post-command power sample was not observed"
            )
    except Exception as exc:
        if start_ns is not None and end_ns is None:
            end_ns = time.monotonic_ns()
        report["status"] = "measurement_failed"
        report["measurement_error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc().splitlines()
    finally:
        sampler.stop()

    if start_ns is not None and end_ns is not None:
        power = summarize_interval(sampler.samples, start_ns, end_ns)
        power["quality_warnings"] = _quality_warnings(
            power, arguments.telemetry_interval_ms
        )
        report["command_monotonic_bounds_ns"] = {"start": start_ns, "end": end_ns}
        report["command_power"] = power
        if arguments.telemetry == "required" and not power["fully_power_covered"]:
            report["status"] = "measurement_failed"
            report["measurement_error"] = (
                "Required power telemetry did not fully cover the command interval"
            )

    report["telemetry"] = sampler.report(origin_monotonic_ns=origin_ns)
    report["finished_utc"] = _utc_now()
    report["scope"] = {
        "included": (
            "child spawn/exec, interpreter/import startup, and all reads, "
            "computations, and writes completed before child exit"
        ),
        "excluded": (
            "harness startup, telemetry priming, optional settling interval, "
            "result JSON writing, system boot, and system shutdown"
        ),
        "power": (
            "gross on-module three-rail estimate; no idle subtraction and no "
            "carrier-board or upstream conversion losses"
        ),
    }
    try:
        _atomic_json(output_path, report, arguments.overwrite)
    except Exception as exc:
        print(f"ERROR: Could not write measurement result: {exc}", file=sys.stderr)
        return 1
    print(f"{report['status']}: wrote {output_path}", file=sys.stderr)
    # A nonzero child status remains the wrapper's status even if telemetry
    # also failed, so automation never loses the measured command's failure.
    if returncode is not None and returncode != 0:
        return _shell_exit_code(returncode)
    if report["status"] == "measurement_failed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

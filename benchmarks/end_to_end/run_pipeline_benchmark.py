#!/usr/bin/env python3
"""Run a staged pipeline under one continuous Jetson power measurement.

The supervisor deliberately does not copy, transform, or splice science data.
All source substitutions must be prepared before it starts.  A JSON plan gives
each measured phase an argv vector, working directory, path assertions, and
optional workload-unit counts.  Each exact phase boundary includes child
execution, closing its stdout/stderr logs, and a blocking filesystem flush so
large writes cannot be silently charged to the following phase.  One
uninterrupted ``tegrastats`` trace is reduced both per phase and across the
complete run.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping
import uuid


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _SCRIPT_DIR.parents[1]
_LEVEL2_BENCHMARK_DIR = _REPOSITORY_ROOT / "benchmarks" / "level2"
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
if str(_LEVEL2_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_LEVEL2_BENCHMARK_DIR))

from power_telemetry import (  # noqa: E402
    TegrastatsSampler,
    run_read_only_diagnostic,
    summarize_interval,
)


_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PLACEHOLDERS = ("${RUN_DIR}", "${PLAN_DIR}", "${REPO_ROOT}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


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
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        action="append",
        dest="selected_phases",
        metavar="NAME",
        help=(
            "run only this named phase; repeat to select several phases (their "
            "inputs must already be staged; see --phase-order)"
        ),
    )
    parser.add_argument(
        "--phase-order",
        choices=("plan", "requested"),
        default="plan",
        help=(
            "execution order for repeated --phase selections: 'plan' retains "
            "the JSON plan order (default), while 'requested' uses the exact "
            "command-line order"
        ),
    )
    parser.add_argument(
        "--pre-run-idle-seconds",
        type=_nonnegative_float,
        default=0.0,
        help="settling interval before total_start; excluded from measured energy",
    )
    parser.add_argument(
        "--telemetry", choices=("auto", "required", "off"), default="required"
    )
    parser.add_argument("--tegrastats-path")
    parser.add_argument(
        "--telemetry-interval-ms", type=_positive_integer, default=100
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="replace supervisor reports/logs; never removes science products",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the expanded plan without running commands",
    )
    return parser


def _expand(value: str, *, run_dir: Path, plan_dir: Path) -> str:
    replacements = {
        "${RUN_DIR}": str(run_dir),
        "${PLAN_DIR}": str(plan_dir),
        "${REPO_ROOT}": str(_REPOSITORY_ROOT),
    }
    expanded = value
    for token, replacement in replacements.items():
        expanded = expanded.replace(token, replacement)
    unresolved = [token for token in _PLACEHOLDERS if token in expanded]
    if unresolved:
        raise ValueError(f"Unresolved path placeholder(s): {unresolved}")
    return expanded


def _path_spec(
    raw: str | Mapping[str, Any],
    *,
    run_dir: Path,
    plan_dir: Path,
) -> dict[str, Any]:
    if isinstance(raw, str):
        path_value, kind, name = raw, "any", None
    elif isinstance(raw, Mapping):
        allowed = {"name", "path", "kind"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown path-spec keys: {sorted(unknown)}")
        path_value = raw.get("path")
        kind = raw.get("kind", "any")
        name = raw.get("name")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("Every path spec needs a nonempty string 'path'")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("Path-spec 'name' must be a nonempty string")
    else:
        raise ValueError("Path specs must be strings or objects")
    if kind not in {"any", "file", "directory"}:
        raise ValueError("Path-spec kind must be 'any', 'file', or 'directory'")
    expanded = Path(_expand(path_value, run_dir=run_dir, plan_dir=plan_dir))
    if not expanded.is_absolute():
        expanded = plan_dir / expanded
    return {"name": name, "path": str(expanded.resolve()), "kind": kind}


def _positive_units(raw: Any, context: str) -> dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be an object")
    result: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Invalid workload-unit name in {context}: {name!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{context}.{name} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{context}.{name} must be finite and positive")
        result[name] = number
    return result


def _load_plan(plan_path: Path, run_dir: Path) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve()
    try:
        raw = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON plan: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Plan root must be a JSON object")
    if raw.get("schema") != "suncet.end_to_end_plan":
        raise ValueError("Plan schema must be 'suncet.end_to_end_plan'")
    if raw.get("schema_version") != 1:
        raise ValueError("Only plan schema_version 1 is supported")
    phases_raw = raw.get("phases")
    if not isinstance(phases_raw, list) or not phases_raw:
        raise ValueError("Plan must contain a nonempty 'phases' array")

    plan_dir = plan_path.parent
    pre_staged = [
        _path_spec(item, run_dir=run_dir, plan_dir=plan_dir)
        for item in raw.get("pre_staged_paths", [])
    ]
    phases: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(phases_raw):
        if not isinstance(item, dict):
            raise ValueError(f"phases[{index}] must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Invalid phase name: {name!r}")
        if name in names:
            raise ValueError(f"Duplicate phase name: {name}")
        names.add(name)
        command = item.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(part, str) or not part for part in command)
        ):
            raise ValueError(f"Phase {name!r} command must be a nonempty argv array")
        expanded_command = [
            _expand(part, run_dir=run_dir, plan_dir=plan_dir) for part in command
        ]
        cwd_raw = item.get("cwd", "${REPO_ROOT}")
        if not isinstance(cwd_raw, str) or not cwd_raw:
            raise ValueError(f"Phase {name!r} cwd must be a nonempty string")
        cwd = Path(_expand(cwd_raw, run_dir=run_dir, plan_dir=plan_dir))
        if not cwd.is_absolute():
            cwd = plan_dir / cwd
        env_raw = item.get("env", {})
        if not isinstance(env_raw, dict) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            for key, value in env_raw.items()
        ):
            raise ValueError(f"Phase {name!r} env must map strings to strings")
        env = {
            key: _expand(value, run_dir=run_dir, plan_dir=plan_dir)
            for key, value in env_raw.items()
        }
        required = [
            _path_spec(spec, run_dir=run_dir, plan_dir=plan_dir)
            for spec in item.get("required_paths", [])
        ]
        expected = [
            _path_spec(spec, run_dir=run_dir, plan_dir=plan_dir)
            for spec in item.get("expected_outputs", [])
        ]
        phases.append(
            {
                "name": name,
                "description": item.get("description"),
                "command": expanded_command,
                "cwd": str(cwd.resolve()),
                "env": env,
                "required_paths": required,
                "expected_outputs": expected,
                "workload_units": _positive_units(
                    item.get("workload_units"), f"phases[{index}].workload_units"
                ),
            }
        )
    derived_raw = raw.get("derived_metrics", [])
    if not isinstance(derived_raw, list):
        raise ValueError("derived_metrics must be an array")
    derived_metrics: list[dict[str, Any]] = []
    derived_names: set[str] = set()
    for index, item in enumerate(derived_raw):
        if not isinstance(item, dict):
            raise ValueError(f"derived_metrics[{index}] must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Invalid derived metric name: {name!r}")
        if name in derived_names:
            raise ValueError(f"Duplicate derived metric name: {name}")
        derived_names.add(name)
        if item.get("type") != "paired_delta_projection":
            raise ValueError(
                f"Derived metric {name!r} type must be 'paired_delta_projection'"
            )
        baseline = item.get("baseline_phase")
        full = item.get("full_phase")
        if baseline not in names or full not in names:
            raise ValueError(
                f"Derived metric {name!r} references an unknown baseline/full phase"
            )
        if baseline == full:
            raise ValueError(
                f"Derived metric {name!r} baseline and full phases must differ"
            )
        unit = item.get("unit")
        if not isinstance(unit, str) or not _NAME_PATTERN.fullmatch(unit):
            raise ValueError(f"Derived metric {name!r} has an invalid unit")
        counts = _positive_units(
            {
                "observed": item.get("observed_units"),
                "target": item.get("target_units"),
            },
            f"derived_metrics[{index}]",
        )
        target_frame_count = item.get("target_frame_count")
        if target_frame_count is not None:
            target_frame_count = _positive_units(
                {"frames": target_frame_count},
                f"derived_metrics[{index}].target_frame_count",
            )["frames"]
        additional = item.get("additional_phases", [])
        if (
            not isinstance(additional, list)
            or any(not isinstance(value, str) for value in additional)
            or len(set(additional)) != len(additional)
        ):
            raise ValueError(
                f"Derived metric {name!r} additional_phases must be unique names"
            )
        unknown = set(additional) - names
        if unknown:
            raise ValueError(
                f"Derived metric {name!r} has unknown additional phases: "
                f"{sorted(unknown)}"
            )
        overlap = set(additional) & {baseline, full}
        if overlap:
            raise ValueError(
                f"Derived metric {name!r} cannot also add its paired phases: "
                f"{sorted(overlap)}"
            )
        derived_metrics.append(
            {
                "name": name,
                "type": "paired_delta_projection",
                "description": item.get("description"),
                "baseline_phase": baseline,
                "full_phase": full,
                "unit": unit,
                "observed_units": counts["observed"],
                "target_units": counts["target"],
                "target_frame_count": target_frame_count,
                "additional_phases": additional,
            }
        )
    return {
        "schema": raw["schema"],
        "schema_version": raw["schema_version"],
        "label": raw.get("label"),
        "notes": raw.get("notes"),
        "plan_path": str(plan_path),
        "pre_staged_paths": pre_staged,
        "derived_metrics": derived_metrics,
        "phases": phases,
    }


def _path_state(spec: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(spec["path"]))
    exists = path.exists()
    kind = str(spec["kind"])
    kind_ok = exists and (
        kind == "any"
        or (kind == "file" and path.is_file())
        or (kind == "directory" and path.is_dir())
    )
    record: dict[str, Any] = {**spec, "exists": exists, "kind_matches": kind_ok}
    if exists:
        try:
            stat = path.stat()
            record.update(
                {
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        except OSError as exc:
            record["stat_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _assert_paths(specs: Iterable[Mapping[str, Any]], context: str) -> list[dict]:
    states = [_path_state(spec) for spec in specs]
    invalid = [state for state in states if not state["kind_matches"]]
    if invalid:
        detail = ", ".join(f"{item['path']} ({item['kind']})" for item in invalid)
        raise FileNotFoundError(f"{context} path assertion failed: {detail}")
    return states


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _read_optional_file(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_record(root: Path) -> dict[str, Any]:
    commands = {
        "commit": ["rev-parse", "HEAD"],
        "branch": ["branch", "--show-current"],
        "status_porcelain": ["status", "--porcelain=v1", "--untracked-files=all"],
    }
    output: dict[str, Any] = {"root": str(root)}
    try:
        for key, arguments in commands.items():
            completed = subprocess.run(
                ["git", "-C", str(root), *arguments],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or "git query failed")
            value: Any = completed.stdout.strip()
            if key == "status_porcelain":
                value = value.splitlines() if value else []
            output[key] = value or None
        output["dirty"] = bool(output["status_porcelain"])
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--no-ext-diff", "--binary", "HEAD", "--"],
            capture_output=True,
            timeout=30,
            check=False,
        )
        if diff.returncode == 0:
            output["tracked_diff_sha256"] = hashlib.sha256(diff.stdout).hexdigest()
        untracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
            capture_output=True,
            timeout=30,
            check=False,
        )
        if untracked.returncode != 0:
            raise RuntimeError(
                untracked.stderr.decode("utf-8", errors="replace").strip()
                or "git untracked-file query failed"
            )
        untracked_records: list[dict[str, Any]] = []
        for raw_path in untracked.stdout.split(b"\0"):
            if not raw_path:
                continue
            relative = os.fsdecode(raw_path)
            candidate = root / relative
            record: dict[str, Any] = {"path": relative}
            try:
                if not candidate.is_file():
                    raise ValueError("untracked path is not a regular file")
                record.update(
                    {
                        "size_bytes": candidate.stat().st_size,
                        "sha256": _sha256_file(candidate),
                        "status": "hashed",
                    }
                )
            except (OSError, ValueError) as exc:
                record.update(
                    {
                        "status": "hash_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            untracked_records.append(record)
        output["untracked_paths"] = untracked_records
        output["status"] = "available"
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        output.update(
            {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
        )
    return output


def _system_record() -> dict[str, Any]:
    return {
        "captured_utc": _utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "environment_prefix": sys.prefix,
        "l4t_release": _read_optional_file("/etc/nv_tegra_release"),
        "os_release": _read_optional_file("/etc/os-release"),
        "packages": {
            name: _distribution_version(name)
            for name in ("astropy", "cupy-cuda13x", "numpy", "scipy", "sunpy")
        },
        "policy": {
            "nvpmodel": run_read_only_diagnostic(
                "nvpmodel",
                ["-q"],
                fallback_paths=("/usr/sbin/nvpmodel", "/usr/bin/nvpmodel"),
            ),
            "jetson_clocks": run_read_only_diagnostic(
                "jetson_clocks",
                ["--show"],
                fallback_paths=("/usr/bin/jetson_clocks", "/usr/sbin/jetson_clocks"),
            ),
            "nvidia_smi": run_read_only_diagnostic(
                "nvidia-smi", [], fallback_paths=("/usr/bin/nvidia-smi",)
            ),
        },
        "selected_environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_CACHE_DISABLE",
                "CUDA_VISIBLE_DEVICES",
                "CUPY_CACHE_IN_MEMORY",
                "CUPY_DISABLE_JITIFY_CACHE",
                "OMP_NUM_THREADS",
                "suncet_data",
                "suncet_ctdb",
            )
        },
    }


def _atomic_text(path: Path, content: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite result: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Any, overwrite: bool) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_text(path, encoded, overwrite)


def _csv_text(fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _quality_warnings(summary: Mapping[str, Any], interval_ms: int) -> list[str]:
    warnings: list[str] = []
    if not summary["fully_power_covered"]:
        warnings.append(
            "The interval was not bracketed by complete power samples; gross "
            "energy is unavailable."
        )
    if summary["power_sample_count_inside_interval"] < 3:
        warnings.append("Fewer than three telemetry samples fell inside the interval.")
    if summary["duration_seconds"] < 100 * interval_ms / 1000:
        warnings.append(
            "The interval is shorter than 100 telemetry intervals; batch or "
            "repeat before interpreting energy or sampled peak power."
        )
    return warnings


def _normalizations(
    power: Mapping[str, Any],
    workload_units: Mapping[str, float],
) -> dict[str, Any]:
    per_unit: dict[str, Any] = {}
    energy = power.get("gross_energy_joules")
    duration = power.get("duration_seconds")
    for unit, count in workload_units.items():
        per_unit[unit] = {
            "observed_count": count,
            "gross_energy_joules": energy / count if energy is not None else None,
            "duration_seconds": duration / count if duration is not None else None,
        }
    return {"per_unit": per_unit}


def _summarize_event(
    samples: list[dict[str, Any]],
    start_ns: int,
    end_ns: int,
    interval_ms: int,
) -> dict[str, Any]:
    summary = summarize_interval(samples, start_ns, end_ns)
    summary["quality_warnings"] = _quality_warnings(summary, interval_ms)
    return summary


def _phase_log_paths(run_dir: Path, name: str) -> tuple[Path, Path]:
    log_dir = run_dir / "logs"
    return log_dir / f"{name}.stdout.log", log_dir / f"{name}.stderr.log"


def _filesystem_flush_barrier() -> dict[str, Any]:
    """Block until pending filesystem writes have completed.

    ``os.sync`` maps to Linux ``sync(2)`` on the Jetson.  Linux waits for the
    pending I/O before returning, so taking the phase end timestamp after this
    call keeps delayed FITS/CSV/database writes inside the phase that created
    them.  It is intentionally a system-wide barrier: the benchmark host must
    remain otherwise quiescent so unrelated writes do not inflate a stage.

    The operation is returned as structured provenance instead of raising so a
    failed child, an interrupted run, or a flush error can still produce a
    complete measurement report and stop telemetry cleanly.
    """

    started_utc = _utc_now()
    start_ns = time.monotonic_ns()
    status = "completed"
    error: str | None = None
    try:
        sync_function = getattr(os, "sync", None)
        if not callable(sync_function):
            raise RuntimeError("os.sync is unavailable on this platform")
        sync_function()
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        error = f"{type(exc).__name__}: {exc}"
    end_ns = time.monotonic_ns()
    record: dict[str, Any] = {
        "method": "os.sync",
        "scope": "all mounted filesystems",
        "status": status,
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "monotonic_bounds_ns": {"start": start_ns, "end": end_ns},
        "duration_seconds": (end_ns - start_ns) / 1e9,
        "included_in_phase_boundary": True,
    }
    if error is not None:
        record["error"] = error
    return record


def _execute_phase(
    phase: Mapping[str, Any], run_dir: Path, *, overwrite_results: bool
) -> dict[str, Any]:
    name = str(phase["name"])
    stdout_path, stderr_path = _phase_log_paths(run_dir, name)
    for path in (stdout_path, stderr_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite_results:
            raise FileExistsError(f"Refusing to overwrite phase log: {path}")

    required_state = _assert_paths(
        phase["required_paths"], f"phase {name!r} required"
    )
    cwd = Path(str(phase["cwd"]))
    if not cwd.is_dir():
        raise FileNotFoundError(f"Phase {name!r} cwd does not exist: {cwd}")
    environment = os.environ.copy()
    environment.update(phase["env"])
    mode = "wb" if overwrite_results else "xb"
    result: dict[str, Any] = {
        "name": name,
        "description": phase.get("description"),
        "command": phase["command"],
        "cwd": str(cwd),
        "environment_overrides": phase["env"],
        "workload_units": phase["workload_units"],
        "required_paths": required_state,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "status": "running",
    }
    process: subprocess.Popen[bytes] | None = None
    returncode: int | None = None
    error: str | None = None
    status = "running"
    started_utc = _utc_now()
    start_ns = time.monotonic_ns()
    try:
        # The flush barrier below must execute only after both streams leave this
        # context. Closing them flushes Python's buffers; os.sync then waits for
        # those log writes and every child-produced science write to reach disk.
        with (
            stdout_path.open(mode) as stdout_stream,
            stderr_path.open(mode) as stderr_stream,
        ):
            try:
                process = subprocess.Popen(
                    phase["command"],
                    cwd=cwd,
                    env=environment,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                )
                returncode = process.wait()
                status = "passed" if returncode == 0 else "failed"
            except BaseException as exc:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                returncode = process.returncode if process is not None else None
                status = (
                    "interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "spawn_failed"
                )
                error = f"{type(exc).__name__}: {exc}"
    except BaseException as exc:
        # A log open/flush/close failure is itself a phase failure, but still run
        # the filesystem barrier and return a structured record to the caller.
        status = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "log_io_failed"
        )
        error = f"{type(exc).__name__}: {exc}"

    logs_closed_ns = time.monotonic_ns()
    filesystem_flush = _filesystem_flush_barrier()
    end_ns = int(filesystem_flush["monotonic_bounds_ns"]["end"])
    finished_utc = _utc_now()
    if filesystem_flush["status"] != "completed":
        flush_error = str(filesystem_flush.get("error") or "unknown flush failure")
        if status == "passed":
            status = (
                "interrupted"
                if filesystem_flush["status"] == "interrupted"
                else "filesystem_flush_failed"
            )
            error = f"Filesystem flush barrier failed: {flush_error}"
        else:
            error = (
                f"{error}; filesystem flush barrier also failed: {flush_error}"
                if error
                else f"Filesystem flush barrier failed: {flush_error}"
            )
    result.update(
        {
            "status": status,
            "returncode": returncode,
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "monotonic_bounds_ns": {"start": start_ns, "end": end_ns},
            "duration_seconds": (end_ns - start_ns) / 1e9,
            "child_and_log_close_duration_seconds": (logs_closed_ns - start_ns)
            / 1e9,
            "filesystem_flush": filesystem_flush,
        }
    )
    if error is not None:
        result["error"] = error
    if status == "passed":
        try:
            result["expected_outputs"] = _assert_paths(
                phase["expected_outputs"], f"phase {name!r} expected-output"
            )
        except FileNotFoundError as exc:
            result["expected_outputs"] = [
                _path_state(spec) for spec in phase["expected_outputs"]
            ]
            result["status"] = "output_validation_failed"
            result["error"] = str(exc)
    else:
        result["expected_outputs"] = [
            _path_state(spec) for spec in phase["expected_outputs"]
        ]
    return result


def _selected_plan(
    plan: dict[str, Any],
    selected: list[str] | None,
    *,
    phase_order: str = "plan",
) -> dict[str, Any]:
    if not selected:
        return plan
    if phase_order not in {"plan", "requested"}:
        raise ValueError("phase_order must be 'plan' or 'requested'")
    requested = set(selected)
    available = {phase["name"] for phase in plan["phases"]}
    unknown = requested - available
    if unknown:
        raise ValueError(f"Unknown selected phase(s): {sorted(unknown)}")
    if len(requested) != len(selected):
        raise ValueError("Each --phase may be specified only once")
    if phase_order == "requested":
        phase_by_name = {phase["name"]: phase for phase in plan["phases"]}
        phases = [phase_by_name[name] for name in selected]
    else:
        phases = [phase for phase in plan["phases"] if phase["name"] in requested]
    return {**plan, "phases": phases}


def _event_row(name: str, kind: str, record: Mapping[str, Any]) -> dict[str, Any]:
    power = record.get("power", {})
    bounds = record.get("monotonic_bounds_ns", {})
    return {
        "name": name,
        "kind": kind,
        "status": record.get("status"),
        "returncode": record.get("returncode"),
        "start_monotonic_ns": bounds.get("start"),
        "end_monotonic_ns": bounds.get("end"),
        "duration_seconds": power.get("duration_seconds", record.get("duration_seconds")),
        "gross_energy_joules": power.get("gross_energy_joules"),
        "observed_energy_joules": power.get("observed_energy_joules"),
        "observed_average_power_watts": power.get("observed_average_power_watts"),
        "sampled_peak_power_watts": power.get("sampled_peak_power_watts"),
        "peak_temperature_c": power.get("peak_temperature_c"),
        "power_coverage_fraction": power.get("power_coverage_fraction"),
        "workload_units_json": json.dumps(record.get("workload_units", {}), sort_keys=True),
        "normalizations_json": json.dumps(record.get("normalizations", {}), sort_keys=True),
    }


def _telemetry_rows(telemetry: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for sample in telemetry.get("samples", []):
        rails = sample.get("rails_mw", {})
        temperatures = sample.get("temperatures_c", {})
        yield {
            "monotonic_ns": sample.get("monotonic_ns"),
            "offset_seconds": sample.get("offset_seconds"),
            "timestamp_utc": sample.get("timestamp_utc"),
            "covered_onboard_power_mw": sample.get("covered_onboard_power_mw"),
            "vdd_gpu_soc_current_mw": (rails.get("VDD_GPU_SOC") or {}).get("current_mw"),
            "vdd_cpu_cv_current_mw": (rails.get("VDD_CPU_CV") or {}).get("current_mw"),
            "vin_sys_5v0_current_mw": (rails.get("VIN_SYS_5V0") or {}).get("current_mw"),
            "tj_c": temperatures.get("tj"),
            "raw_line": sample.get("raw_line"),
        }


def _write_reports(run_dir: Path, report: dict[str, Any], overwrite: bool) -> None:
    events = [
        _event_row("pipeline_total", "total", report["pipeline_total"]),
        *[_event_row(p["name"], "phase", p) for p in report["phases"]],
        *[_event_row(g["name"], "interphase_gap", g) for g in report["interphase_gaps"]],
    ]
    event_fields = [
        "name",
        "kind",
        "status",
        "returncode",
        "start_monotonic_ns",
        "end_monotonic_ns",
        "duration_seconds",
        "gross_energy_joules",
        "observed_energy_joules",
        "observed_average_power_watts",
        "sampled_peak_power_watts",
        "peak_temperature_c",
        "power_coverage_fraction",
        "workload_units_json",
        "normalizations_json",
    ]
    telemetry_fields = [
        "monotonic_ns",
        "offset_seconds",
        "timestamp_utc",
        "covered_onboard_power_mw",
        "vdd_gpu_soc_current_mw",
        "vdd_cpu_cv_current_mw",
        "vin_sys_5v0_current_mw",
        "tj_c",
        "raw_line",
    ]
    _atomic_text(
        run_dir / "phase_summary.csv",
        _csv_text(event_fields, events),
        overwrite,
    )
    _atomic_text(
        run_dir / "telemetry_samples.csv",
        _csv_text(telemetry_fields, _telemetry_rows(report["telemetry"])),
        overwrite,
    )
    _atomic_json(run_dir / "measurement.json", report, overwrite)


def _paired_delta_metrics(
    phases: list[Mapping[str, Any]], specs: list[Mapping[str, Any]]
) -> dict[str, Any]:
    """Project only the variable delta between paired exact-file measurements.

    This avoids multiplying fixed Level 0.5 parsing/telemetry cost by an image
    count ratio.  The full and baseline phases remain independently visible in
    the report; a projection is emitted only when every required gross-energy
    measurement is available.
    """

    phase_by_name = {str(phase["name"]): phase for phase in phases}
    result: dict[str, Any] = {}
    for spec in specs:
        baseline = phase_by_name.get(str(spec["baseline_phase"]))
        full = phase_by_name.get(str(spec["full_phase"]))
        additional = [
            phase_by_name.get(str(name)) for name in spec["additional_phases"]
        ]
        warnings: list[str] = [
            "The projection assumes variable image-processing cost is linear in "
            f"{spec['unit']}; repeat the pair in reversed order to bound cache and "
            "thermal-order bias."
        ]
        if baseline is None or full is None or any(p is None for p in additional):
            result[str(spec["name"])] = {
                **spec,
                "status": "unavailable",
                "reason": "One or more referenced phases were not executed",
                "quality_warnings": warnings,
            }
            continue

        assert baseline is not None and full is not None
        baseline_energy = baseline["power"].get("gross_energy_joules")
        full_energy = full["power"].get("gross_energy_joules")
        baseline_duration = baseline["power"].get("duration_seconds")
        full_duration = full["power"].get("duration_seconds")
        additional_energy = [
            phase["power"].get("gross_energy_joules")
            for phase in additional
            if phase is not None
        ]
        additional_duration = [
            phase["power"].get("duration_seconds")
            for phase in additional
            if phase is not None
        ]
        pair_energy_available = baseline_energy is not None and full_energy is not None
        pair_duration_available = (
            baseline_duration is not None and full_duration is not None
        )
        scale = float(spec["target_units"]) / float(spec["observed_units"])
        energy_delta = (
            full_energy - baseline_energy if pair_energy_available else None
        )
        duration_delta = (
            full_duration - baseline_duration if pair_duration_available else None
        )
        if energy_delta is not None and energy_delta < 0:
            warnings.append(
                "The measured full-minus-baseline energy delta is negative; the "
                "projection is mathematically reported but is not physically "
                "credible without additional paired trials."
            )
        if duration_delta is not None and duration_delta < 0:
            warnings.append(
                "The measured full-minus-baseline duration delta is negative; "
                "repeat paired trials before using the duration projection."
            )
        projected_phase_energy = (
            baseline_energy + scale * energy_delta
            if energy_delta is not None
            else None
        )
        projected_phase_duration = (
            baseline_duration + scale * duration_delta
            if duration_delta is not None
            else None
        )
        projected_total_energy = (
            projected_phase_energy + sum(additional_energy)
            if projected_phase_energy is not None
            and all(value is not None for value in additional_energy)
            else None
        )
        projected_total_duration = (
            projected_phase_duration + sum(additional_duration)
            if projected_phase_duration is not None
            and all(value is not None for value in additional_duration)
            else None
        )
        exact_hybrid_energy = (
            full_energy + sum(additional_energy)
            if full_energy is not None
            and all(value is not None for value in additional_energy)
            else None
        )
        exact_hybrid_duration = (
            full_duration + sum(additional_duration)
            if full_duration is not None
            and all(value is not None for value in additional_duration)
            else None
        )
        target_frame_count = spec.get("target_frame_count")

        def _per_target_frame(value: float | None) -> float | None:
            if value is None or target_frame_count is None:
                return None
            return value / float(target_frame_count)

        result[str(spec["name"])] = {
            **spec,
            "status": (
                "available"
                if pair_energy_available
                and all(value is not None for value in additional_energy)
                else "partially_available"
                if pair_energy_available
                else "energy_unavailable"
            ),
            "formula": (
                "projected = baseline + "
                "(target_units / observed_units) * (full - baseline)"
            ),
            "scale_factor": scale,
            "exact_file_measurements": {
                "baseline": {
                    "phase": baseline["name"],
                    "gross_energy_joules": baseline_energy,
                    "duration_seconds": baseline_duration,
                },
                "full": {
                    "phase": full["name"],
                    "gross_energy_joules": full_energy,
                    "duration_seconds": full_duration,
                },
            },
            "variable_delta": {
                "gross_energy_joules": energy_delta,
                "duration_seconds": duration_delta,
            },
            "projected_variable_phase": {
                "gross_energy_joules": projected_phase_energy,
                "duration_seconds": projected_phase_duration,
                "gross_energy_joules_per_target_frame": _per_target_frame(
                    projected_phase_energy
                ),
                "duration_seconds_per_target_frame": _per_target_frame(
                    projected_phase_duration
                ),
            },
            "additional_phase_contributions": [
                {
                    "phase": phase["name"],
                    "gross_energy_joules": phase["power"].get(
                        "gross_energy_joules"
                    ),
                    "duration_seconds": phase["power"].get("duration_seconds"),
                }
                for phase in additional
                if phase is not None
            ],
            "exact_hybrid_phase_sum": {
                "gross_energy_joules": exact_hybrid_energy,
                "duration_seconds": exact_hybrid_duration,
                "average_power_watts": (
                    exact_hybrid_energy / exact_hybrid_duration
                    if exact_hybrid_energy is not None
                    and exact_hybrid_duration is not None
                    and exact_hybrid_duration > 0
                    else None
                ),
                "gross_energy_joules_per_target_frame": _per_target_frame(
                    exact_hybrid_energy
                ),
                "duration_seconds_per_target_frame": _per_target_frame(
                    exact_hybrid_duration
                ),
                "scope": (
                    "Direct sum of the exact full-file phase and declared "
                    "additional science phases; excludes diagnostic baseline and "
                    "all supervisor gaps."
                ),
            },
            "projected_total": {
                "gross_energy_joules": projected_total_energy,
                "duration_seconds": projected_total_duration,
                "average_power_watts": (
                    projected_total_energy / projected_total_duration
                    if projected_total_energy is not None
                    and projected_total_duration is not None
                    and projected_total_duration > 0
                    else None
                ),
                "gross_energy_joules_per_target_frame": _per_target_frame(
                    projected_total_energy
                ),
                "duration_seconds_per_target_frame": _per_target_frame(
                    projected_total_duration
                ),
            },
            "quality_warnings": warnings,
            "scope": (
                "Phase-only model; excludes paired diagnostic duplication, "
                "interphase gaps, substitution/staging, supervisor work, network "
                "transfer, boot, and shutdown."
            ),
        }
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    run_dir = arguments.run_dir.expanduser().resolve()
    try:
        plan = _selected_plan(
            _load_plan(arguments.plan, run_dir),
            arguments.selected_phases,
            phase_order=arguments.phase_order,
        )
        pre_staged_state = _assert_paths(
            plan["pre_staged_paths"], "pre-staged input"
        )
        for phase in plan["phases"]:
            cwd = Path(phase["cwd"])
            if not cwd.is_dir():
                raise FileNotFoundError(
                    f"Phase {phase['name']!r} cwd does not exist: {cwd}"
                )
        if arguments.dry_run:
            print(json.dumps({**plan, "pre_staged_paths": pre_staged_state}, indent=2))
            return 0
        run_dir.mkdir(parents=True, exist_ok=True)
        report_paths = (
            run_dir / "measurement.json",
            run_dir / "phase_summary.csv",
            run_dir / "telemetry_samples.csv",
        )
        if not arguments.overwrite_results:
            existing = [path for path in report_paths if path.exists()]
            existing.extend(
                path
                for phase in plan["phases"]
                for path in _phase_log_paths(run_dir, phase["name"])
                if path.exists()
            )
            if existing:
                raise FileExistsError(
                    "Refusing to overwrite existing supervisor artifacts: "
                    + ", ".join(map(str, existing))
                )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    sampler = TegrastatsSampler(
        interval_ms=arguments.telemetry_interval_ms,
        executable=arguments.tegrastats_path,
        mode=arguments.telemetry,
    )
    origin_ns = time.monotonic_ns()
    report: dict[str, Any] = {
        "schema": "suncet.end_to_end_power_measurement",
        "schema_version": 1,
        "status": "running",
        "created_utc": _utc_now(),
        "run_directory": str(run_dir),
        "plan": plan,
        "pre_staged_path_state": pre_staged_state,
        "telemetry_interval_ms": arguments.telemetry_interval_ms,
        "pre_run_idle_seconds": arguments.pre_run_idle_seconds,
        "system": _system_record(),
        "git": _git_record(_REPOSITORY_ROOT),
        "phases": [],
        "interphase_gaps": [],
    }
    total_start_ns: int | None = None
    total_end_ns: int | None = None
    total_started_utc: str | None = None
    total_finished_utc: str | None = None
    wrapper_failure: str | None = None
    try:
        sampler.start()
        if sampler.status == "collecting":
            primed = sampler.wait_for_power_sample(timeout_seconds=3.0)
            if arguments.telemetry == "required" and not primed:
                raise RuntimeError("Required tegrastats power telemetry did not become ready")
        if arguments.pre_run_idle_seconds:
            time.sleep(arguments.pre_run_idle_seconds)

        total_started_utc = _utc_now()
        total_start_ns = time.monotonic_ns()
        previous_end: int | None = None
        for phase in plan["phases"]:
            phase_result = _execute_phase(
                phase, run_dir, overwrite_results=arguments.overwrite_results
            )
            phase_start = phase_result["monotonic_bounds_ns"]["start"]
            if previous_end is not None:
                report["interphase_gaps"].append(
                    {
                        "name": f"after_{report['phases'][-1]['name']}",
                        "status": "completed",
                        "monotonic_bounds_ns": {
                            "start": previous_end,
                            "end": phase_start,
                        },
                        "duration_seconds": (phase_start - previous_end) / 1e9,
                    }
                )
            report["phases"].append(phase_result)
            previous_end = phase_result["monotonic_bounds_ns"]["end"]
            print(
                f"{phase_result['status']}: {phase_result['name']} "
                f"({phase_result['duration_seconds']:.3f} s)",
                file=sys.stderr,
                flush=True,
            )
            if phase_result["status"] != "passed":
                break
        total_end_ns = time.monotonic_ns()
        total_finished_utc = _utc_now()
        if report["phases"]:
            first_start = report["phases"][0]["monotonic_bounds_ns"]["start"]
            last_end = report["phases"][-1]["monotonic_bounds_ns"]["end"]
            report["interphase_gaps"].insert(
                0,
                {
                    "name": "before_first_phase",
                    "status": "completed",
                    "monotonic_bounds_ns": {
                        "start": total_start_ns,
                        "end": first_start,
                    },
                    "duration_seconds": (first_start - total_start_ns) / 1e9,
                },
            )
            report["interphase_gaps"].append(
                {
                    "name": "after_last_phase",
                    "status": "completed",
                    "monotonic_bounds_ns": {
                        "start": last_end,
                        "end": total_end_ns,
                    },
                    "duration_seconds": (total_end_ns - last_end) / 1e9,
                }
            )
        else:
            report["interphase_gaps"].append(
                {
                    "name": "pipeline_without_started_phase",
                    "status": "completed",
                    "monotonic_bounds_ns": {
                        "start": total_start_ns,
                        "end": total_end_ns,
                    },
                    "duration_seconds": (total_end_ns - total_start_ns) / 1e9,
                }
            )
        post_sample = sampler.wait_for_power_sample(
            after_monotonic_ns=total_end_ns,
            timeout_seconds=max(2.0, 4 * arguments.telemetry_interval_ms / 1000),
        )
        if arguments.telemetry == "required" and not post_sample:
            wrapper_failure = "Required post-pipeline power sample was not observed"
    except BaseException as exc:
        if total_start_ns is not None and total_end_ns is None:
            total_end_ns = time.monotonic_ns()
            total_finished_utc = _utc_now()
        wrapper_failure = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc().splitlines()
    finally:
        sampler.stop()

    samples = sampler.samples
    for phase in report["phases"]:
        bounds = phase["monotonic_bounds_ns"]
        phase["power"] = _summarize_event(
            samples,
            bounds["start"],
            bounds["end"],
            arguments.telemetry_interval_ms,
        )
        phase["normalizations"] = _normalizations(
            phase["power"], phase["workload_units"]
        )
    for gap in report["interphase_gaps"]:
        bounds = gap["monotonic_bounds_ns"]
        gap["power"] = _summarize_event(
            samples,
            bounds["start"],
            bounds["end"],
            arguments.telemetry_interval_ms,
        )

    if total_start_ns is not None and total_end_ns is not None:
        total_power = _summarize_event(
            samples,
            total_start_ns,
            total_end_ns,
            arguments.telemetry_interval_ms,
        )
        report["pipeline_total"] = {
            "status": "completed",
            "started_utc": total_started_utc,
            "finished_utc": total_finished_utc,
            "monotonic_bounds_ns": {
                "start": total_start_ns,
                "end": total_end_ns,
            },
            "power": total_power,
        }
    else:
        now_ns = time.monotonic_ns()
        report["pipeline_total"] = {
            "status": "not_started",
            "monotonic_bounds_ns": {"start": now_ns, "end": now_ns},
            "power": _summarize_event(
                samples, now_ns, now_ns, arguments.telemetry_interval_ms
            ),
        }

    phase_energy = [
        phase["power"].get("gross_energy_joules") for phase in report["phases"]
    ]
    gap_energy = [
        gap["power"].get("gross_energy_joules")
        for gap in report["interphase_gaps"]
    ]
    total_energy = report["pipeline_total"]["power"].get("gross_energy_joules")
    all_reconciled = (
        total_energy is not None
        and all(value is not None for value in phase_energy)
        and all(value is not None for value in gap_energy)
    )
    report["energy_reconciliation"] = {
        "phase_sum_joules": sum(phase_energy) if all(v is not None for v in phase_energy) else None,
        "interphase_gap_sum_joules": (
            sum(gap_energy) if all(v is not None for v in gap_energy) else None
        ),
        "total_minus_phase_sum_joules": (
            total_energy - sum(phase_energy)
            if total_energy is not None and all(v is not None for v in phase_energy)
            else None
        ),
        "numerical_reconciliation_error_joules": (
            total_energy - sum(phase_energy) - sum(gap_energy)
            if all_reconciled
            else None
        ),
        "interpretation": (
            "The total-minus-phase residual is measured supervisor/interphase "
            "overhead. It is excluded from phase-only workload projections."
        ),
    }
    report["derived_metrics"] = _paired_delta_metrics(
        report["phases"], plan["derived_metrics"]
    )
    report["telemetry"] = sampler.report(origin_monotonic_ns=origin_ns)
    report["finished_utc"] = _utc_now()
    phase_failed = any(phase["status"] != "passed" for phase in report["phases"])
    incomplete = len(report["phases"]) != len(plan["phases"])
    if wrapper_failure:
        report["status"] = "measurement_failed"
        report["measurement_error"] = wrapper_failure
    elif arguments.telemetry == "required" and not report["pipeline_total"][
        "power"
    ]["fully_power_covered"]:
        report["status"] = "measurement_failed"
        report["measurement_error"] = (
            "Required power telemetry did not fully cover the pipeline interval"
        )
    elif phase_failed or incomplete:
        report["status"] = "phase_failed"
    else:
        report["status"] = "passed"
    report["scope"] = {
        "included": (
            "Each phase includes child spawn/exec, imports, reads, computations, "
            "writes, child/log-handle closure, and a blocking os.sync filesystem "
            "flush. The directly measured total also includes interphase supervisor "
            "gaps and every diagnostic phase, including a paired baseline when "
            "configured."
        ),
        "excluded": (
            "Plan validation, data staging/substitution, system-state queries, "
            "telemetry priming, optional settling, report generation, network "
            "transfer, boot, and shutdown."
        ),
        "power": (
            "Gross on-module three-rail estimate; no idle subtraction and no "
            "carrier-board or upstream conversion losses."
        ),
        "projection": (
            "Only explicitly declared paired-delta models are projected. They "
            "preserve the fixed baseline and scale full-minus-baseline variable "
            "work; they are not directly observed end-to-end energy."
        ),
    }
    try:
        _write_reports(run_dir, report, arguments.overwrite_results)
    except Exception as exc:
        print(f"ERROR: Could not write benchmark reports: {exc}", file=sys.stderr)
        return 1
    print(f"{report['status']}: wrote {run_dir / 'measurement.json'}", file=sys.stderr)
    if report["status"] == "passed":
        return 0
    for phase in report["phases"]:
        returncode = phase.get("returncode")
        if phase["status"] == "failed" and isinstance(returncode, int):
            return min(255, 128 - returncode if returncode < 0 else returncode)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

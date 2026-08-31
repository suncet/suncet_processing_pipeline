"""Benchmark the frozen known-window CME tracker on a Jetson or workstation.

This harness measures the existing full-window CPU reference.  It deliberately
does not describe the result as streaming: the temporal background, path
optimization, and final kinematic fit use completed-window context.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Sequence
import uuid

import numpy as np
import psutil

from .data_paths import get_data_root
from .level4.cme_tracking.config import read_configuration
from .level4.cme_tracking.input import load_sequence_from_manifest
from .level4.cme_tracking.pipeline import (
    CMETrackingRun,
    run_known_window,
    run_known_window_from_images,
    write_known_window_products,
)
from .level4.common.provenance import collect_run_provenance
from .level4.jetson_metrics import (
    COVERED_ONBOARD_RAILS,
    covered_onboard_power_mw,
    parse_tegrastats_line,
    select_samples_in_window,
    summarize_tegrastats_samples,
    summarize_tegrastats_window,
)


_SCOPES = ("compute", "product_end_to_end")
_DEFAULT_CADENCES_SECONDS = (15.0, 10.0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _parse_cadences(value: str) -> tuple[float, ...]:
    parsed = tuple(_positive_float(item.strip()) for item in value.split(","))
    if not parsed:
        raise argparse.ArgumentTypeError("must contain at least one cadence")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Root used to resolve the manifest's portable data paths.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Immutable benchmark parent directory; defaults to "
            "$suncet_data/benchmarks/cme_tracking."
        ),
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        help="Temporary product location, preferably on the benchmarked NVMe.",
    )
    parser.add_argument(
        "--scope",
        choices=(*_SCOPES, "both"),
        default="compute",
    )
    parser.add_argument("--warmups", type=_positive_int, default=1)
    parser.add_argument("--repetitions", type=_positive_int, default=5)
    parser.add_argument(
        "--minimum-measurement-seconds",
        type=_positive_float,
        default=30.0,
        help="Repeat the event inside each measured window to reach this duration.",
    )
    parser.add_argument(
        "--idle-baseline-seconds",
        type=_nonnegative_float,
        default=30.0,
    )
    parser.add_argument(
        "--tegrastats-interval-ms",
        type=_positive_int,
        default=200,
    )
    parser.add_argument(
        "--deadline-cadences",
        type=_parse_cadences,
        default=_DEFAULT_CADENCES_SECONDS,
        metavar="SECONDS[,SECONDS...]",
        help=(
            "Arrival cadences used only for throughput sizing; these do not "
            "change the manifest's scientific time axis."
        ),
    )
    parser.add_argument(
        "--allow-inconsistent-synthetic-geometry",
        action="store_true",
    )
    parser.add_argument(
        "--expected-science-reference-npz",
        type=Path,
        help="Frozen array artifact from the visually accepted reference run.",
    )
    parser.add_argument(
        "--expected-science-reference-json",
        type=Path,
        help="Signature/state metadata accompanying the frozen NPZ artifact.",
    )
    parser.add_argument(
        "--require-exact-reference",
        action="store_true",
        help="Abort before measured repetitions unless the frozen reference is exact.",
    )
    parser.add_argument(
        "--require-reference-equivalence",
        action="store_true",
        help=(
            "Abort unless discrete outputs are exact and floating outputs meet "
            "the recorded cross-platform CPU tolerance policy."
        ),
    )
    return parser


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, values: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(values), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _safe_command(arguments: Sequence[str], *, timeout: float = 15.0) -> dict[str, Any]:
    try:
        result = subprocess.run(
            list(arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except OSError as exc:
        return {
            "available": False,
            "arguments": list(arguments),
            "error": f"{type(exc).__name__}: {exc}",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "available": True,
            "timed_out": True,
            "arguments": list(arguments),
            "stdout": _jsonable(exc.stdout),
            "stderr": _jsonable(exc.stderr),
        }
    return {
        "available": True,
        "arguments": list(arguments),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip("\0\n")
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _safe_probe(function: Callable[[], Any]) -> Any:
    try:
        return function()
    except (OSError, RuntimeError, psutil.Error):
        return None


def collect_system_snapshot(repository: Path, configuration: dict[str, Any]) -> dict[str, Any]:
    """Collect read-only execution state without changing clocks or power mode."""

    process = psutil.Process()
    return {
        "captured_at_utc": _utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "device_model": _read_text("/proc/device-tree/model"),
        "os_release": _read_text("/etc/os-release"),
        "nv_tegra_release": _read_text("/etc/nv_tegra_release"),
        "cpu_count_logical": _safe_probe(lambda: psutil.cpu_count(logical=True)),
        "cpu_count_physical": _safe_probe(lambda: psutil.cpu_count(logical=False)),
        "cpu_affinity": (
            _safe_probe(process.cpu_affinity)
            if hasattr(process, "cpu_affinity")
            else None
        ),
        "memory_total_bytes": _safe_probe(lambda: psutil.virtual_memory().total),
        "swap_total_bytes": _safe_probe(lambda: psutil.swap_memory().total),
        "covered_onboard_rails": list(COVERED_ONBOARD_RAILS),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "commands": {
            "nvpmodel": _safe_command(("nvpmodel", "-q", "--verbose")),
            "jetson_clocks": _safe_command(("jetson_clocks", "--show")),
            "fan": _safe_command(("nvfancontrol", "-q")),
            "nvcc": _safe_command(("nvcc", "--version")),
            "python": _safe_command((os.sys.executable, "--version")),
            "numpy_configuration": _safe_command(
                (os.sys.executable, "-c", "import numpy; numpy.show_config()")
            ),
            "lscpu": _safe_command(("lscpu", "--json")),
            "mamba_packages": _safe_command(
                ("mamba", "list", "--json"), timeout=30.0
            ),
            "lsblk": _safe_command(
                ("lsblk", "-o", "NAME,MODEL,SIZE,FSTYPE,MOUNTPOINTS")
            ),
        },
        "run_provenance": collect_run_provenance(
            repository=repository,
            configuration=configuration,
        ),
    }


class TegrastatsCollector:
    """Timestamp and retain every line emitted by one owned tegrastats process."""

    def __init__(self, interval_ms: int) -> None:
        self.interval_ms = interval_ms
        self.samples: list[dict[str, Any]] = []
        self.unparsed_lines: list[dict[str, Any]] = []
        self.error: str | None = None
        self.command: list[str] | None = None
        self.returncode: int | None = None
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return (
            self._process is not None
            and self.error is None
            and any(covered_onboard_power_mw(sample) is not None for sample in self.samples)
        )

    def start(self) -> None:
        executable = shutil.which("tegrastats")
        if executable is None:
            self.error = "tegrastats executable not found"
            return
        self.command = [executable, "--interval", str(self.interval_ms)]
        try:
            self._process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        for raw in self._process.stdout:
            line = raw.rstrip("\n")
            monotonic_ns = time.monotonic_ns()
            timestamp_utc = _utc_now()
            try:
                sample = parse_tegrastats_line(
                    line,
                    monotonic_ns=monotonic_ns,
                    timestamp_utc=timestamp_utc,
                )
            except (TypeError, ValueError) as exc:
                with self._lock:
                    self.unparsed_lines.append(
                        {
                            "monotonic_ns": monotonic_ns,
                            "timestamp_utc": timestamp_utc,
                            "raw_line": line,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            else:
                with self._lock:
                    self.samples.append(sample)

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.returncode = self._process.returncode
        if self.returncode not in (0, -15) and self.error is None:
            self.error = f"tegrastats exited with status {self.returncode}"
        if not self.samples and self.error is None:
            self.error = "tegrastats emitted no recognized telemetry samples"

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.samples)


class PeakRSSMonitor:
    """Read the process RSS high-water mark without periodic polling.

    Even a single-process ``/proc`` read at 20 Hz measurably perturbed this
    workload on Jetson.  The kernel-maintained ``ru_maxrss`` counter captures
    native NumPy/SciPy allocations and requires only one read at each boundary.
    It is a process-lifetime high-water mark, not a per-interval sampled peak.
    """

    def __init__(self, interval_seconds: float = 0.05) -> None:
        # Retained for API compatibility with the earlier sampling monitor.
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = 0
        self.initial_peak_rss_bytes = 0

    def start(self) -> None:
        self.initial_peak_rss_bytes = self._sample_once()
        self.peak_rss_bytes = self.initial_peak_rss_bytes

    @staticmethod
    def _sample_once() -> int:
        try:
            maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except (OSError, ValueError):
            return 0
        # Darwin reports bytes; Linux and the other supported Unix platforms
        # report KiB.  The benchmark records bytes everywhere else.
        return maximum_rss if sys.platform == "darwin" else maximum_rss * 1024

    def stop(self) -> None:
        self.peak_rss_bytes = max(self.peak_rss_bytes, self._sample_once())


def _science_arrays(run: CMETrackingRun) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "likelihood_score": run.likelihood.score,
        "front_state": run.front.state,
        "front_radius_px": run.front.radius_px,
        "front_score": run.front.score,
        "front_observed_mask": run.front.observed_mask,
        "summary_height_rsun": run.summary.height_rsun,
        "summary_height_sigma_rsun": run.summary.height_sigma_rsun,
        "summary_position_angle_deg": run.summary.position_angle_deg,
        "summary_angular_width_deg": run.summary.angular_width_deg,
        "summary_coverage_fraction": run.summary.coverage_fraction,
        "summary_confidence": run.summary.confidence,
        "field_of_view_limited_mask": run.field_of_view_limited_mask,
    }
    if run.kinematics is not None:
        for name in (
            "raw_height",
            "raw_height_sigma",
            "measurement_mask",
            "fit_domain_mask",
            "fitted_height",
            "speed",
            "acceleration",
            "fitted_height_sigma",
            "speed_sigma",
            "acceleration_sigma",
            "endpoint_mask",
            "unsupported_gap_mask",
            "valid_mask",
        ):
            arrays[f"kinematics_{name}"] = getattr(run.kinematics, name)
    return {name: np.asarray(values) for name, values in arrays.items()}


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    dtype = contiguous.dtype
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and os.sys.byteorder == "big"):
        contiguous = contiguous.astype(dtype.newbyteorder("<"), copy=False)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    # Feed the contiguous array buffer directly to OpenSSL.  ``tobytes``
    # duplicates the complete array and previously made post-run validation,
    # rather than the tracker, determine the measured RSS peak on Jetson.
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def science_signature(run: CMETrackingRun) -> dict[str, Any]:
    components = {
        name: _array_sha256(array) for name, array in _science_arrays(run).items()
    }
    state = {
        "event_detected": bool(run.front.event_detected),
        "kinematics_available": run.kinematics is not None,
        "kinematics_error": run.kinematics_error,
    }
    digest = hashlib.sha256()
    for name, value in sorted(components.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    digest.update(
        json.dumps(
            state,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return {
        "schema_version": 2,
        "overall_sha256": digest.hexdigest(),
        "component_sha256": components,
        **state,
    }


def compare_science_reference(
    run: CMETrackingRun,
    *,
    expected_npz_path: Path,
    expected_json_path: Path,
) -> dict[str, Any]:
    """Compare one candidate to an independently preserved reference artifact."""

    candidate_arrays = _science_arrays(run)
    candidate_signature = science_signature(run)
    with expected_json_path.open("r", encoding="utf-8") as stream:
        expected_signature = json.load(stream)
    expected_components = expected_signature.get("component_sha256", {})
    tolerance_policy = {
        "name": "suncet_cpu_cross_platform_v1",
        "default": {"absolute": 1e-12, "relative": 1e-12},
        "overrides": {
            # Different NumPy/SciPy builds can move the intermediate evidence
            # score by a few float32 ULPs while leaving the selected discrete
            # path and retained front exactly unchanged.
            "likelihood_score": {"absolute": 5e-7, "relative": 1e-7},
        },
    }
    comparisons: dict[str, Any] = {}
    with np.load(expected_npz_path, allow_pickle=False) as archive:
        expected_names = set(archive.files)
        candidate_names = set(candidate_arrays)
        for name in sorted(expected_names | candidate_names):
            if name not in expected_names:
                comparisons[name] = {"status": "unexpected_candidate_component"}
                continue
            if name not in candidate_names:
                comparisons[name] = {"status": "missing_candidate_component"}
                continue
            expected = np.asarray(archive[name])
            candidate = np.asarray(candidate_arrays[name])
            if candidate.shape != expected.shape:
                comparisons[name] = {
                    "status": "shape_mismatch",
                    "within_tolerance": False,
                    "expected_shape": list(expected.shape),
                    "candidate_shape": list(candidate.shape),
                }
                continue
            if candidate.dtype != expected.dtype:
                comparisons[name] = {
                    "status": "dtype_mismatch",
                    "within_tolerance": False,
                    "expected_dtype": str(expected.dtype),
                    "candidate_dtype": str(candidate.dtype),
                }
                continue

            exact = bool(
                np.array_equal(candidate, expected, equal_nan=True)
                if np.issubdtype(candidate.dtype, np.inexact)
                else np.array_equal(candidate, expected)
            )
            component: dict[str, Any] = {
                "status": "exact" if exact else "value_mismatch",
                "exact": exact,
                "expected_sha256": expected_components.get(name),
                "candidate_sha256": candidate_signature["component_sha256"].get(name),
            }
            if np.issubdtype(candidate.dtype, np.inexact):
                candidate_finite = np.isfinite(candidate)
                expected_finite = np.isfinite(expected)
                common = candidate_finite & expected_finite
                finite_state_mismatches = int(
                    np.count_nonzero(candidate_finite != expected_finite)
                )
                if np.any(common):
                    difference = np.asarray(candidate[common], dtype=np.float64) - np.asarray(
                        expected[common], dtype=np.float64
                    )
                    component.update(
                        {
                            "common_finite_count": int(np.count_nonzero(common)),
                            "finite_state_mismatch_count": finite_state_mismatches,
                            "maximum_absolute_difference": float(
                                np.max(np.abs(difference))
                            ),
                            "root_mean_square_difference": float(
                                np.sqrt(np.mean(difference**2))
                            ),
                        }
                    )
                else:
                    component.update(
                        {
                            "common_finite_count": 0,
                            "finite_state_mismatch_count": finite_state_mismatches,
                            "maximum_absolute_difference": None,
                            "root_mean_square_difference": None,
                        }
                    )
                tolerance = tolerance_policy["overrides"].get(
                    name,
                    tolerance_policy["default"],
                )
                component["tolerance"] = tolerance
                component["within_tolerance"] = bool(
                    finite_state_mismatches == 0
                    and np.allclose(
                        candidate,
                        expected,
                        rtol=tolerance["relative"],
                        atol=tolerance["absolute"],
                        equal_nan=True,
                    )
                )
            else:
                component["mismatch_count"] = int(
                    np.count_nonzero(candidate != expected)
                )
                component["within_tolerance"] = exact
            comparisons[name] = component

    state_fields = ("event_detected", "kinematics_available", "kinematics_error")
    state_comparison = {
        name: {
            "expected": expected_signature.get(name),
            "candidate": candidate_signature.get(name),
            "exact": expected_signature.get(name) == candidate_signature.get(name),
        }
        for name in state_fields
    }
    arrays_exact = all(
        value.get("exact") is True for value in comparisons.values()
    )
    arrays_equivalent = all(
        value.get("within_tolerance") is True for value in comparisons.values()
    )
    state_exact = all(value["exact"] for value in state_comparison.values())
    return {
        "schema_version": 1,
        "expected_npz_path": str(expected_npz_path),
        "expected_npz_sha256": _file_sha256(expected_npz_path),
        "expected_json_path": str(expected_json_path),
        "expected_json_sha256": _file_sha256(expected_json_path),
        "candidate_signature": candidate_signature,
        "arrays_exact": arrays_exact,
        "arrays_equivalent": arrays_equivalent,
        "state_exact": state_exact,
        "exact_match": arrays_exact and state_exact,
        "numerically_equivalent": arrays_equivalent and state_exact,
        "tolerance_policy": tolerance_policy,
        "components": comparisons,
        "state": state_comparison,
    }
def _deadline_metrics(
    duration_seconds_per_iteration: float,
    *,
    frame_count: int,
    cadences_seconds: Sequence[float],
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for cadence in cadences_seconds:
        acquisition_horizon = frame_count * cadence
        utilization = duration_seconds_per_iteration / acquisition_horizon
        metrics[f"{cadence:g}"] = {
            "replay_arrival_cadence_seconds": cadence,
            "acquisition_horizon_seconds": acquisition_horizon,
            "batch_utilization_fraction": utilization,
            "throughput_margin_ratio": (
                acquisition_horizon / duration_seconds_per_iteration
                if duration_seconds_per_iteration > 0
                else None
            ),
            "headroom_fraction": 1.0 - utilization,
            "bounded_backlog_by_average_rate": utilization < 1.0,
        }
    return metrics


@dataclass
class WorkloadResult:
    run: CMETrackingRun
    duration_seconds: float
    start_monotonic_ns: int
    end_monotonic_ns: int
    workload_intervals_monotonic_ns: tuple[tuple[int, int], ...]
    science_signatures: tuple[dict[str, Any], ...]
    initial_peak_rss_bytes: int
    peak_rss_bytes: int


def _measure_workload(
    workload: Callable[[int], CMETrackingRun],
    *,
    iterations: int,
) -> WorkloadResult:
    gc.collect()
    monitor = PeakRSSMonitor()
    monitor.start()
    run: CMETrackingRun | None = None
    intervals: list[tuple[int, int]] = []
    signatures: list[dict[str, Any]] = []
    try:
        for iteration in range(iterations):
            if run is not None:
                # Release the prior event before allocating the next result;
                # otherwise Python evaluates the RHS while both large result
                # graphs are resident and inflates peak memory.
                del run
                run = None
            start = time.monotonic_ns()
            run = workload(iteration)
            end = time.monotonic_ns()
            intervals.append((start, end))
            # Hash outside the timed science interval but verify every inner
            # iteration rather than only the final result.
            signatures.append(science_signature(run))
    finally:
        monitor.stop()
    assert run is not None
    assert intervals
    return WorkloadResult(
        run=run,
        duration_seconds=sum(end - start for start, end in intervals) / 1e9,
        start_monotonic_ns=intervals[0][0],
        end_monotonic_ns=intervals[-1][1],
        workload_intervals_monotonic_ns=tuple(intervals),
        science_signatures=tuple(signatures),
        initial_peak_rss_bytes=monitor.initial_peak_rss_bytes,
        peak_rss_bytes=monitor.peak_rss_bytes,
    )


def _summary_statistics(values: Sequence[float]) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {name: None for name in ("minimum", "median", "mean", "maximum", "iqr")}
    return {
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "maximum": float(np.max(finite)),
        "iqr": float(np.percentile(finite, 75) - np.percentile(finite, 25)),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_root(argument: Path | None, *, suffix: Sequence[str]) -> Path:
    if argument is not None:
        return argument.expanduser().resolve()
    return get_data_root().joinpath(*suffix)


def _write_telemetry(stage: Path, collector: TegrastatsCollector) -> None:
    raw_path = stage / "tegrastats.raw.log"
    with raw_path.open("w", encoding="utf-8") as stream:
        for sample in collector.samples:
            stream.write(
                f"{sample['monotonic_ns']}\t{sample['timestamp_utc']}\t"
                f"{sample['raw_line']}\n"
            )
        for item in collector.unparsed_lines:
            stream.write(
                f"{item['monotonic_ns']}\t{item['timestamp_utc']}\t"
                f"UNPARSED {item['error']}\t{item['raw_line']}\n"
            )
    with (stage / "tegrastats.samples.jsonl").open("w", encoding="utf-8") as stream:
        for sample in collector.samples:
            stream.write(json.dumps(_jsonable(sample), sort_keys=True, allow_nan=False))
            stream.write("\n")


def _enrich_power(
    row: dict[str, Any],
    samples: Sequence[dict[str, Any]],
    *,
    idle_power_mw: float | None,
) -> None:
    intervals = [tuple(values) for values in row["workload_intervals_monotonic_ns"]]
    interval_summaries = [
        summarize_tegrastats_window(
            samples,
            start_monotonic_ns=int(start),
            end_monotonic_ns=int(end),
        )
        for start, end in intervals
    ]
    selected_by_timestamp: dict[int, dict[str, Any]] = {}
    for start, end in intervals:
        for sample in select_samples_in_window(
            samples,
            start_monotonic_ns=int(start),
            end_monotonic_ns=int(end),
        ):
            selected_by_timestamp[int(sample["monotonic_ns"])] = sample
    summary = summarize_tegrastats_samples(list(selected_by_timestamp.values()))
    energy_values = [
        values["gross_energy_joules"]
        for values in interval_summaries
        if isinstance(values.get("gross_energy_joules"), (int, float))
    ]
    gross = float(sum(energy_values)) if energy_values else None
    covered_seconds = float(
        sum(values.get("power_coverage_seconds", 0.0) for values in interval_summaries)
    )
    duration = row["duration_seconds"]
    peak_values = [
        values["peak_covered_onboard_power_mw"]
        for values in interval_summaries
        if isinstance(values.get("peak_covered_onboard_power_mw"), (int, float))
    ]
    summary.update(
        {
            "workload_interval_count": len(intervals),
            "requested_duration_seconds": duration,
            "power_coverage_seconds": covered_seconds,
            "power_coverage_fraction": (
                covered_seconds / duration if duration > 0 else None
            ),
            "gross_energy_joules": gross,
            "mean_covered_onboard_power_mw": (
                gross / covered_seconds * 1000.0
                if gross is not None and covered_seconds > 0
                else None
            ),
            "peak_covered_onboard_power_mw": (
                max(peak_values) if peak_values else None
            ),
            "exact_boundary_interpolation": True,
        }
    )
    row["tegrastats"] = summary
    iterations = row["iterations"]
    row["gross_energy_joules_per_event"] = (
        gross / iterations if isinstance(gross, (int, float)) else None
    )
    row["gross_energy_joules_per_frame"] = (
        gross / (iterations * row["frame_count"])
        if isinstance(gross, (int, float))
        else None
    )
    incremental = (
        gross - idle_power_mw / 1000.0 * covered_seconds
        if isinstance(gross, (int, float)) and idle_power_mw is not None
        else None
    )
    row["incremental_energy_joules"] = incremental
    row["incremental_energy_joules_per_event"] = (
        incremental / iterations if incremental is not None else None
    )
    row["incremental_energy_joules_per_frame"] = (
        incremental / (iterations * row["frame_count"])
        if incremental is not None
        else None
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if (arguments.expected_science_reference_npz is None) != (
        arguments.expected_science_reference_json is None
    ):
        parser.error(
            "--expected-science-reference-npz and "
            "--expected-science-reference-json must be supplied together"
        )
    if (
        arguments.require_exact_reference
        or arguments.require_reference_equivalence
    ) and arguments.expected_science_reference_npz is None:
        parser.error(
            "reference enforcement requires a frozen NPZ/JSON reference pair"
        )
    repository = Path(__file__).resolve().parents[1]
    manifest_path = arguments.manifest.expanduser().resolve()
    config_path = arguments.config.expanduser().resolve()
    expected_npz_path = (
        arguments.expected_science_reference_npz.expanduser().resolve()
        if arguments.expected_science_reference_npz is not None
        else None
    )
    expected_json_path = (
        arguments.expected_science_reference_json.expanduser().resolve()
        if arguments.expected_science_reference_json is not None
        else None
    )
    data_root = _resolve_root(arguments.data_root, suffix=())
    output_root = _resolve_root(
        arguments.output_root,
        suffix=("benchmarks", "cme_tracking"),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    scratch_root = (
        arguments.scratch_root.expanduser().resolve()
        if arguments.scratch_root is not None
        else output_root / ".scratch"
    )
    scratch_root.mkdir(parents=True, exist_ok=True)

    configuration = read_configuration(config_path)
    configuration_values = configuration.to_dict()
    scopes = _SCOPES if arguments.scope == "both" else (arguments.scope,)
    sequence, _manifest = load_sequence_from_manifest(
        manifest_path,
        data_root=data_root,
        verify_hashes=True,
        allow_inconsistent_geometry=arguments.allow_inconsistent_synthetic_geometry,
    )
    resident_input_bytes = (
        sequence.frame_count
        * math.prod(sequence.geometry.image_shape_yx)
        * np.dtype(np.float32).itemsize
    )
    resident_images = (
        sequence.materialize(dtype=np.float32) if "compute" in scopes else None
    )

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + "_"
        + socket.gethostname().replace(" ", "-")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    final = output_root / run_id
    stage = output_root / f".{run_id}.incomplete"
    stage.mkdir()

    rows: list[dict[str, Any]] = []
    collector = TegrastatsCollector(arguments.tegrastats_interval_ms)
    idle_window: tuple[int, int] | None = None
    reference_signature: dict[str, Any] | None = None
    reference_comparison: dict[str, Any] | None = None

    try:
        _write_json(
            stage / "system_snapshot.json",
            collect_system_snapshot(repository, configuration_values),
        )
        collector.start()
        if collector.command is not None:
            # Establish a sample before the idle/workload start marker so the
            # first boundary can be interpolated instead of left uncovered.
            time.sleep(max(0.5, 2.0 * collector.interval_ms / 1000.0))
        if arguments.idle_baseline_seconds > 0:
            idle_start = time.monotonic_ns()
            time.sleep(arguments.idle_baseline_seconds)
            idle_end = time.monotonic_ns()
            idle_window = (idle_start, idle_end)

        for scope in scopes:
            def compute_workload(_iteration: int) -> CMETrackingRun:
                assert resident_images is not None
                return run_known_window_from_images(
                    sequence,
                    resident_images,
                    configuration,
                )

            temporary_parent: tempfile.TemporaryDirectory[str] | None = None
            if scope == "compute":
                workload = compute_workload
            else:
                temporary_parent = tempfile.TemporaryDirectory(
                    prefix=f"suncet-{run_id}-{scope}-",
                    dir=scratch_root,
                )
                product_parent = Path(temporary_parent.name)
                invocation = 0

                def product_workload(_iteration: int) -> CMETrackingRun:
                    nonlocal invocation
                    invocation += 1
                    loaded, _loaded_manifest = load_sequence_from_manifest(
                        manifest_path,
                        data_root=data_root,
                        verify_hashes=True,
                        allow_inconsistent_geometry=(
                            arguments.allow_inconsistent_synthetic_geometry
                        ),
                    )
                    result = run_known_window(loaded, configuration)
                    write_known_window_products(
                        result,
                        product_parent / f"invocation-{invocation:04d}",
                        "benchmark-event",
                        repository=repository,
                        include_diagnostic_movie=False,
                    )
                    return result

                workload = product_workload

            try:
                warmup_result = _measure_workload(workload, iterations=1)
                if reference_signature is None:
                    reference_signature = warmup_result.science_signatures[0]
                    np.savez_compressed(
                        stage / "science_reference.npz",
                        **_science_arrays(warmup_result.run),
                    )
                    _write_json(
                        stage / "science_reference.json",
                        reference_signature,
                    )
                    if expected_npz_path is not None:
                        assert expected_json_path is not None
                        reference_comparison = compare_science_reference(
                            warmup_result.run,
                            expected_npz_path=expected_npz_path,
                            expected_json_path=expected_json_path,
                        )
                        _write_json(
                            stage / "expected_reference_comparison.json",
                            reference_comparison,
                        )
                        if (
                            arguments.require_exact_reference
                            and not reference_comparison["exact_match"]
                        ):
                            raise RuntimeError(
                                "Candidate does not exactly match the frozen "
                                "science reference; see "
                                "expected_reference_comparison.json"
                            )
                        if (
                            arguments.require_reference_equivalence
                            and not reference_comparison["numerically_equivalent"]
                        ):
                            raise RuntimeError(
                                "Candidate exceeds the frozen-reference CPU "
                                "equivalence tolerances; see "
                                "expected_reference_comparison.json"
                            )
                warmup_signature = warmup_result.science_signatures[0]
                if warmup_signature["overall_sha256"] != reference_signature["overall_sha256"]:
                    raise RuntimeError(f"{scope} warmup changed the frozen science result")
                warmup_duration_seconds = warmup_result.duration_seconds
                del warmup_result
                gc.collect()
                for _ in range(max(arguments.warmups - 1, 0)):
                    extra_warmup = _measure_workload(workload, iterations=1)
                    if any(
                        signature["overall_sha256"]
                        != reference_signature["overall_sha256"]
                        for signature in extra_warmup.science_signatures
                    ):
                        raise RuntimeError(
                            f"{scope} repeated warmup changed the frozen science result"
                        )
                    del extra_warmup
                    gc.collect()

                iterations = max(
                    1,
                    math.ceil(
                        arguments.minimum_measurement_seconds
                        / warmup_duration_seconds
                    ),
                )
                for repetition in range(arguments.repetitions):
                    measured = _measure_workload(workload, iterations=iterations)
                    if any(
                        signature["overall_sha256"]
                        != reference_signature["overall_sha256"]
                        for signature in measured.science_signatures
                    ):
                        raise RuntimeError(
                            f"{scope} repetition {repetition} changed the frozen "
                            "science result in at least one inner iteration"
                        )
                    duration_per_event = measured.duration_seconds / iterations
                    rows.append(
                        {
                            "scope": scope,
                            "repetition": repetition,
                            "iterations": iterations,
                            "start_monotonic_ns": measured.start_monotonic_ns,
                            "end_monotonic_ns": measured.end_monotonic_ns,
                            "workload_intervals_monotonic_ns": [
                                list(interval)
                                for interval in measured.workload_intervals_monotonic_ns
                            ],
                            "duration_seconds": measured.duration_seconds,
                            "duration_seconds_per_event": duration_per_event,
                            "frame_count": sequence.frame_count,
                            "frames_per_second": (
                                sequence.frame_count / duration_per_event
                            ),
                            "seconds_per_frame": (
                                duration_per_event / sequence.frame_count
                            ),
                            "peak_rss_bytes": measured.peak_rss_bytes,
                            "initial_peak_rss_bytes": (
                                measured.initial_peak_rss_bytes
                            ),
                            "verified_science_result_count": len(
                                measured.science_signatures
                            ),
                            "science_signature_sha256": reference_signature[
                                "overall_sha256"
                            ],
                            "deadline_analysis": _deadline_metrics(
                                duration_per_event,
                                frame_count=sequence.frame_count,
                                cadences_seconds=arguments.deadline_cadences,
                            ),
                        }
                    )
                    del measured
                    gc.collect()
            finally:
                if temporary_parent is not None:
                    temporary_parent.cleanup()
            if scope == "compute" and "product_end_to_end" in scopes:
                # Do not inflate the production-scope RSS with the resident
                # cube retained solely for the compute-only scope.
                resident_images = None
                gc.collect()

        # Retain one sample after the final stop marker so exact-boundary power
        # interpolation covers the last workload interval.
        time.sleep(max(0.5, 1.5 * collector.interval_ms / 1000.0))
        collector.stop()
        samples = collector.snapshot()
        if idle_window is not None:
            idle_summary = summarize_tegrastats_window(
                samples,
                start_monotonic_ns=idle_window[0],
                end_monotonic_ns=idle_window[1],
            )
        else:
            idle_summary = {}
        idle_power = idle_summary.get("mean_covered_onboard_power_mw")
        if not isinstance(idle_power, (int, float)):
            idle_power = None
        for row in rows:
            _enrich_power(row, samples, idle_power_mw=idle_power)

        assert reference_signature is not None
        _write_telemetry(stage, collector)
        with (stage / "repetitions.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(_jsonable(row), sort_keys=True, allow_nan=False))
                stream.write("\n")

        scope_summaries: dict[str, Any] = {}
        for scope in scopes:
            selected = [row for row in rows if row["scope"] == scope]
            scope_summaries[scope] = {
                "duration_seconds_per_event": _summary_statistics(
                    [row["duration_seconds_per_event"] for row in selected]
                ),
                "frames_per_second": _summary_statistics(
                    [row["frames_per_second"] for row in selected]
                ),
                "gross_energy_joules_per_event": _summary_statistics(
                    [
                        row["gross_energy_joules_per_event"]
                        for row in selected
                        if row["gross_energy_joules_per_event"] is not None
                    ]
                ),
                "incremental_energy_joules_per_event": _summary_statistics(
                    [
                        row["incremental_energy_joules_per_event"]
                        for row in selected
                        if row["incremental_energy_joules_per_event"] is not None
                    ]
                ),
                "peak_rss_bytes": _summary_statistics(
                    [float(row["peak_rss_bytes"]) for row in selected]
                ),
                "median_deadline_analysis": _deadline_metrics(
                    float(
                        np.median(
                            [row["duration_seconds_per_event"] for row in selected]
                        )
                    ),
                    frame_count=sequence.frame_count,
                    cadences_seconds=arguments.deadline_cadences,
                ),
            }

        benchmark = {
            "schema_version": 1,
            "run_id": run_id,
            "completed_at_utc": _utc_now(),
            "status": "complete",
            "algorithm_backend": "numpy_cpu",
            "scheduling_policy": "full_window",
            "causality": "noncausal_completed_window_reference",
            "manifest": str(manifest_path),
            "manifest_sha256": sequence.manifest_sha256,
            "configuration": str(config_path),
            "scientific_cadence_seconds": sequence.cadence_seconds,
            "replay_arrival_cadences_seconds": list(arguments.deadline_cadences),
            "frame_count": sequence.frame_count,
            "image_shape_yx": list(sequence.geometry.image_shape_yx),
            "resident_input_bytes": resident_input_bytes,
            "scopes": list(scopes),
            "warmups": arguments.warmups,
            "repetitions": arguments.repetitions,
            "minimum_measurement_seconds": arguments.minimum_measurement_seconds,
            "idle_baseline_seconds": arguments.idle_baseline_seconds,
            "idle_tegrastats": idle_summary,
            "memory_measurement": {
                "metric": "process_lifetime_peak_rss_bytes",
                "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
                "periodic_polling": False,
                "includes_native_thread_allocations": True,
                "includes_child_processes": False,
            },
            "tegrastats": {
                "available": collector.available,
                "error": collector.error,
                "command": collector.command,
                "returncode": collector.returncode,
                "interval_ms": collector.interval_ms,
                "parsed_sample_count": len(collector.samples),
                "unparsed_line_count": len(collector.unparsed_lines),
                "covered_onboard_rails": list(COVERED_ONBOARD_RAILS),
            },
            "science_reference_sha256": reference_signature["overall_sha256"],
            "expected_reference": reference_comparison,
            "science_validation_status": (
                "compared to the independently preserved, visually accepted "
                "engineering reference; no authoritative physical truth"
                if reference_comparison is not None
                else (
                    "within-run deterministic consistency only; no independent "
                    "reference supplied and no authoritative physical truth"
                )
            ),
            "scope_summaries": scope_summaries,
            "interpretation_limits": [
                "The 30 s manifest cadence is the scientific time axis; 15 s and 10 s are replay sizing cadences only.",
                "Average full-window throughput does not demonstrate a causal streaming implementation.",
                "Covered onboard rails are comparative and are not developer-kit DC input power.",
                "This run measures Level 4 alone, not combined Level 2/3/4 schedulability.",
                "product_end_to_end is a warm-cache userspace scope and does not prove durable fsync-complete NVMe publication.",
                "peak_rss_bytes is the Python process lifetime high-water mark; initial_peak_rss_bytes records the value before each measured interval.",
            ],
        }
        _write_json(stage / "benchmark.json", benchmark)

        artifacts = {}
        for path in sorted(stage.iterdir()):
            if path.is_file():
                artifacts[path.name] = {
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
        _write_json(
            stage / "COMPLETE.json",
            {
                "run_id": run_id,
                "completed_at_utc": _utc_now(),
                "artifacts": artifacts,
            },
        )
        stage.replace(final)
    except BaseException as exc:
        collector.stop()
        _write_json(
            stage / "FAILURE.json",
            {
                "failed_at_utc": _utc_now(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise

    print(f"Benchmark complete: {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

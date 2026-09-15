"""Benchmark the exact SunCET Level 2 deconvolution core on CPU or GPU.

This program measures one process-local calibration preparation, a first
application, warm-up applications, and a measured application batch.  The
CuPy path includes host-to-device input and device-to-host output transfers,
because that is the public Level 2 backend contract.  FITS product writing is
deliberately outside this core benchmark.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Iterator
import uuid


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    # The isolated Jetson environment intentionally need not install the
    # mutable checkout. Import the exact source tree containing this harness.
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from power_telemetry import (
    TegrastatsSampler,
    run_read_only_diagnostic,
    summarize_interval,
)


SCHEMA_VERSION = 1


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "cupy"), required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--diffraction-psf-file", type=Path, required=True)
    parser.add_argument("--scatter-psf-file", type=Path, required=True)
    parser.add_argument("--spectrum-file", type=Path, required=True)
    parser.add_argument("--spectral-response-file", type=Path, required=True)
    parser.add_argument(
        "--correction-factor", type=_finite_float, default=0.4
    )
    parser.add_argument(
        "--warmups",
        type=_nonnegative_integer,
        default=2,
        help="unreported prepared applications before the measured batch",
    )
    parser.add_argument(
        "--repetitions",
        type=_positive_integer,
        default=10,
        help=(
            "minimum measured prepared applications (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--minimum-measured-seconds",
        type=_nonnegative_float,
        default=10.0,
        help=(
            "continue the measured batch until this duration and the requested "
            "repetition count are both met (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--pre-run-idle-seconds",
        type=_nonnegative_float,
        default=0.0,
        help=(
            "optional recorded settling interval; it is never subtracted from "
            "gross workload energy (default: %(default)s)"
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--save-last-output",
        type=Path,
        help="optional .npy output for a separate CPU/GPU numerical comparison",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--telemetry",
        choices=("auto", "required", "off"),
        default="auto",
        help="whether usable tegrastats power telemetry is optional or required",
    )
    parser.add_argument(
        "--tegrastats-path",
        help="explicit tegrastats executable (normally discovered automatically)",
    )
    parser.add_argument(
        "--telemetry-interval-ms",
        type=_positive_integer,
        default=100,
    )
    parser.add_argument(
        "--git-root",
        type=Path,
        default=_REPOSITORY_ROOT,
    )
    return parser


def _atomic_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npy(path: Path, array: Any, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing array: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            # Imported lazily by the workload; array is already a NumPy host array.
            import numpy as np

            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    if not resolved.is_file():
        raise FileNotFoundError(f"Benchmark input is not a file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": _sha256_file(resolved),
    }


def _run_git(root: Path, arguments: list[str], *, binary: bool = False) -> Any:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=not binary,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        stderr = (
            completed.stderr.decode("utf-8", errors="replace")
            if binary
            else completed.stderr
        )
        raise RuntimeError(stderr.strip() or "git command failed")
    return completed.stdout


def _git_record(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    try:
        commit = _run_git(root, ["rev-parse", "HEAD"]).strip()
        branch = _run_git(root, ["branch", "--show-current"]).strip() or None
        status = _run_git(
            root,
            ["status", "--porcelain=v1", "--untracked-files=all"],
        ).splitlines()
        tracked_diff = _run_git(
            root,
            ["diff", "--no-ext-diff", "--binary", "HEAD", "--"],
            binary=True,
        )
        return {
            "status": "available",
            "root": str(root),
            "commit": commit,
            "branch": branch,
            "dirty": bool(status),
            "status_porcelain": status,
            "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        }
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        return {
            "status": "unavailable",
            "root": str(root),
            "reason": f"{type(exc).__name__}: {exc}",
        }


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


def _platform_record() -> dict[str, Any]:
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except OSError:
            pass
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "environment_prefix": sys.prefix,
        "cpu_affinity": affinity,
        "l4t_release": _read_optional_file("/etc/nv_tegra_release"),
        "os_release": _read_optional_file("/etc/os-release"),
        "packages": {
            name: _distribution_version(name)
            for name in (
                "astropy",
                "cupy-cuda13x",
                "numpy",
                "scipy",
                "sunpy",
                "suncet",
            )
        },
    }


def _system_policy_record() -> dict[str, Any]:
    return {
        "nvpmodel_query": run_read_only_diagnostic(
            "nvpmodel",
            ["-q"],
            fallback_paths=("/usr/sbin/nvpmodel", "/usr/bin/nvpmodel"),
        ),
        "jetson_clocks_query": run_read_only_diagnostic(
            "jetson_clocks",
            ["--show"],
            fallback_paths=("/usr/bin/jetson_clocks", "/usr/sbin/jetson_clocks"),
        ),
        "nvidia_smi_query": run_read_only_diagnostic(
            "nvidia-smi",
            [],
            fallback_paths=("/usr/bin/nvidia-smi",),
        ),
        "environment_clock_controls": {
            key: os.environ.get(key)
            for key in (
                "CUDA_CACHE_DISABLE",
                "CUDA_VISIBLE_DEVICES",
                "CUPY_CACHE_IN_MEMORY",
                "CUPY_DISABLE_JITIFY_CACHE",
                "OMP_NUM_THREADS",
            )
        },
    }


class PhaseRecorder:
    """Record named, non-overlapping or nested monotonic-clock phases."""

    def __init__(self) -> None:
        self.intervals: dict[str, tuple[int, int]] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if name in self.intervals:
            raise ValueError(f"Phase {name!r} was already recorded")
        started = time.monotonic_ns()
        try:
            yield
        finally:
            self.intervals[name] = (started, time.monotonic_ns())

    def timing_report(self) -> dict[str, Any]:
        return {
            name: {"duration_seconds": (end - start) / 1_000_000_000}
            for name, (start, end) in self.intervals.items()
        }


def _timing_summary(values: list[float]) -> dict[str, Any]:
    return {
        "seconds": values,
        "count": len(values),
        "total_seconds": sum(values),
        "minimum_seconds": min(values),
        "median_seconds": statistics.median(values),
        "maximum_seconds": max(values),
    }


def _timed_apply(prepared: Any, image: Any, diagnostics: io.StringIO) -> tuple[Any, float]:
    prepared.synchronize()
    started = time.monotonic_ns()
    with redirect_stdout(diagnostics):
        result = prepared.apply(image)
    prepared.synchronize()
    elapsed = (time.monotonic_ns() - started) / 1_000_000_000
    return result, elapsed


def _output_record(array: Any) -> dict[str, Any]:
    import numpy as np

    host = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    finite = np.isfinite(host)
    return {
        "shape": list(host.shape),
        "dtype": str(host.dtype),
        "all_finite": bool(np.all(finite)),
        "finite_pixel_count": int(np.count_nonzero(finite)),
        "minimum": float(np.min(host)) if bool(np.all(finite)) else None,
        "maximum": float(np.max(host)) if bool(np.all(finite)) else None,
        "sum": float(np.sum(host)) if bool(np.all(finite)) else None,
        "sha256_c_contiguous_bytes": hashlib.sha256(host.tobytes()).hexdigest(),
    }


def _load_image(path: Path) -> Any:
    import numpy as np
    from astropy.io import fits

    with fits.open(path, memmap=False, checksum=True) as hdul:
        if not hdul or hdul[0].data is None:
            raise ValueError(f"Input FITS has no primary image: {path}")
        image = np.asarray(hdul[0].data, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"Input image must be two-dimensional, got {image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError("Input image contains non-finite values")
    return image


def _power_quality(summary: dict[str, Any], interval_ms: int) -> list[str]:
    warnings: list[str] = []
    if not summary["fully_power_covered"]:
        warnings.append(
            "The phase was not bracketed by power samples; gross energy is null."
        )
    if summary["power_sample_count_inside_interval"] < 3:
        warnings.append(
            "Fewer than three power samples fell inside this phase; repeat a "
            "longer batch before interpreting energy or peak power."
        )
    if summary["duration_seconds"] < 100 * interval_ms / 1000:
        warnings.append(
            "The measured batch is shorter than 100 telemetry intervals "
            "(10 seconds at the default 100 ms cadence); power estimates may "
            "be strongly quantized or observer-sensitive."
        )
    return warnings


def _run_workload(
    arguments: argparse.Namespace,
    sampler: TegrastatsSampler,
    phases: PhaseRecorder,
) -> tuple[dict[str, Any], Any]:
    diagnostics = io.StringIO()
    warmup_durations: list[float] = []
    measured_durations: list[float] = []

    with phases.phase("workload_total"):
        with phases.phase("imports"):
            # Include scientific imports in the explicitly reported cold scope.
            from suncet_processing_pipeline import suncet_deconv

        with phases.phase("input_load"):
            image = _load_image(arguments.input_file)

        with phases.phase("calibration_prepare"):
            prepared = suncet_deconv.prepare_deconv(
                arguments.diffraction_psf_file,
                arguments.scatter_psf_file,
                arguments.spectral_response_file,
                arguments.spectrum_file,
                correction_factor=arguments.correction_factor,
                backend=arguments.backend,
            )
            prepared.synchronize()

        with phases.phase("cold_first_apply"):
            last_output, cold_first_apply_seconds = _timed_apply(
                prepared, image, diagnostics
            )

        if arguments.warmups:
            with phases.phase("warmup_apply_batch"):
                for _ in range(arguments.warmups):
                    last_output, elapsed = _timed_apply(
                        prepared, image, diagnostics
                    )
                    warmup_durations.append(elapsed)

        with phases.phase("measured_apply_batch"):
            batch_started_ns = time.monotonic_ns()
            while (
                len(measured_durations) < arguments.repetitions
                or (time.monotonic_ns() - batch_started_ns) / 1e9
                < arguments.minimum_measured_seconds
            ):
                last_output, elapsed = _timed_apply(prepared, image, diagnostics)
                measured_durations.append(elapsed)

    workload_start = phases.intervals["workload_total"][0]
    workload_end = phases.intervals["workload_total"][1]
    sampler.wait_for_power_sample(
        after_monotonic_ns=workload_end,
        timeout_seconds=max(2.0, 4 * arguments.telemetry_interval_ms / 1000),
    )
    workload = {
        "backend_requested": arguments.backend,
        "backend_actual": prepared.backend,
        "input_shape": list(image.shape),
        "input_dtype": str(image.dtype),
        "correction_factor": arguments.correction_factor,
        "warmups": arguments.warmups,
        "minimum_repetitions_requested": arguments.repetitions,
        "minimum_measured_seconds_requested": arguments.minimum_measured_seconds,
        "measured_repetitions_completed": len(measured_durations),
        "scope_contract": {
            "calibration_prepare": (
                "CPU FITS/GENX reads, spectral merge/rebin, backend transfer, "
                "and both prepared PSF FFTs"
            ),
            "cold_first_apply": (
                "first prepared host-array application after calibration preparation"
            ),
            "measured_apply_batch": (
                "prepared applications; CuPy includes input H2D, both FP64/"
                "complex128 inverse stages, and final D2H"
            ),
            "excluded": (
                "Level 1 calibration, FITS product construction/writing, file "
                "hashing, and process boot/shutdown"
            ),
        },
        "cold_first_apply_seconds": cold_first_apply_seconds,
        "warmup_apply_timing": (
            _timing_summary(warmup_durations) if warmup_durations else None
        ),
        "measured_apply_timing": _timing_summary(measured_durations),
        "diagnostic_stdout": diagnostics.getvalue().splitlines(),
        "workload_monotonic_bounds_ns": {
            "start": workload_start,
            "end": workload_end,
        },
    }
    return workload, last_output


def main(argv: list[str] | None = None) -> int:
    program_started_ns = time.monotonic_ns()
    arguments = _parser().parse_args(argv)
    output_path = arguments.output_json.expanduser().resolve()
    if output_path.exists() and not arguments.overwrite:
        print(f"ERROR: Refusing to overwrite existing result: {output_path}", file=sys.stderr)
        return 2

    for path in (
        arguments.input_file,
        arguments.diffraction_psf_file,
        arguments.scatter_psf_file,
        arguments.spectrum_file,
        arguments.spectral_response_file,
    ):
        if not path.expanduser().is_file():
            print(f"ERROR: Required input does not exist: {path}", file=sys.stderr)
            return 2

    measurement_origin_ns = time.monotonic_ns()
    sampler = TegrastatsSampler(
        interval_ms=arguments.telemetry_interval_ms,
        executable=arguments.tegrastats_path,
        mode=arguments.telemetry,
    )
    phases = PhaseRecorder()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "suncet_level2_deconvolution_core",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "backend_requested": arguments.backend,
    }
    last_output = None
    exit_code = 0
    try:
        sampler.start()
        if sampler.status == "collecting":
            primed = sampler.wait_for_power_sample(timeout_seconds=3.0)
            if arguments.telemetry == "required" and not primed:
                raise RuntimeError(
                    "Required tegrastats input-power telemetry did not become ready"
                )
        if arguments.pre_run_idle_seconds:
            with phases.phase("pre_run_idle"):
                time.sleep(arguments.pre_run_idle_seconds)
        workload, last_output = _run_workload(arguments, sampler, phases)
        report["workload"] = workload
        report["status"] = "passed"
    except Exception as exc:
        exit_code = 1
        report["status"] = "failed"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc().splitlines(),
        }
    finally:
        sampler.stop()

    samples = sampler.samples
    timing = phases.timing_report()
    power_by_phase: dict[str, Any] = {}
    for name, (start_ns, end_ns) in phases.intervals.items():
        summary = summarize_interval(samples, start_ns, end_ns)
        summary["quality_warnings"] = _power_quality(
            summary, arguments.telemetry_interval_ms
        )
        power_by_phase[name] = summary
    if "measured_apply_batch" in power_by_phase:
        measured_power = power_by_phase["measured_apply_batch"]
        batch_energy = measured_power["gross_energy_joules"]
        completed_repetitions = (
            report.get("workload", {}).get("measured_repetitions_completed")
        )
        measured_power["gross_energy_per_apply_joules"] = (
            batch_energy / completed_repetitions
            if batch_energy is not None and completed_repetitions
            else None
        )

    report.update(
        {
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "program_wall_seconds_before_result_write": (
                time.monotonic_ns() - program_started_ns
            ) / 1_000_000_000,
            "timing_by_phase": timing,
            "power_by_phase": power_by_phase,
            "telemetry": sampler.report(origin_monotonic_ns=measurement_origin_ns),
            "power_interpretation": {
                "idle_subtraction_performed": False,
                "pre_run_idle_seconds": arguments.pre_run_idle_seconds,
                "energy_definition": (
                    "gross sampled module-input energy integrated by the "
                    "trapezoidal rule over monotonic collector timestamps"
                ),
                "peak_definition": (
                    "maximum simultaneous sampled total, not a sum of "
                    "per-rail historical maxima"
                ),
                "temperature_definition": (
                    "peak current tj/TJ_MAX when present; all dynamic thermal "
                    "zone peaks are retained"
                ),
            },
            "system": _platform_record(),
            "system_policy": _system_policy_record(),
            "git": _git_record(arguments.git_root),
        }
    )

    # Hash after all measured work so reading the 1.3 GB calibration cannot
    # prewarm the filesystem cache used by the reported preparation phase.
    try:
        report["files"] = {
            "input": _file_record(arguments.input_file),
            "diffraction_psf": _file_record(arguments.diffraction_psf_file),
            "scatter_psf": _file_record(arguments.scatter_psf_file),
            "spectrum": _file_record(arguments.spectrum_file),
            "spectral_response": _file_record(arguments.spectral_response_file),
            "benchmark_program": _file_record(Path(__file__)),
            "telemetry_module": _file_record(
                Path(__file__).with_name("power_telemetry.py")
            ),
        }
    except Exception as exc:
        report["status"] = "failed"
        report["file_provenance_error"] = f"{type(exc).__name__}: {exc}"
        exit_code = 1

    if last_output is not None:
        report["last_output"] = _output_record(last_output)
        if arguments.save_last_output is not None:
            try:
                _atomic_npy(
                    arguments.save_last_output,
                    last_output,
                    arguments.overwrite,
                )
                report["last_output"]["saved_npy"] = str(
                    arguments.save_last_output.expanduser().resolve()
                )
            except Exception as exc:
                report["status"] = "failed"
                report["output_save_error"] = f"{type(exc).__name__}: {exc}"
                exit_code = 1

    try:
        _atomic_json(output_path, report, arguments.overwrite)
    except Exception as exc:
        print(f"ERROR: Could not write benchmark result: {exc}", file=sys.stderr)
        return 1
    print(
        f"{report['status']}: wrote {output_path}",
        file=sys.stderr,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

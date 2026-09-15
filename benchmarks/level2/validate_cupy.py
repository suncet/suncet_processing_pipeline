"""Run a synchronized, read-only FP64 CuPy FFT/IFFT smoke test."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import math
import os
import platform
import statistics
import sys
import time
from typing import Any


EXIT_IMPORT_FAILURE = 2
EXIT_RUNTIME_FAILURE = 3
EXIT_NUMERICAL_FAILURE = 4


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


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=_positive_integer,
        default=1500,
        help="FFT row count (default: %(default)s).",
    )
    parser.add_argument(
        "--columns",
        type=_positive_integer,
        default=2000,
        help="FFT column count (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        type=_nonnegative_integer,
        default=0,
        help="zero-based CUDA device index (default: %(default)s).",
    )
    parser.add_argument(
        "--warmups",
        type=_nonnegative_integer,
        default=2,
        help="untimed FFT/IFFT pairs before measurement (default: %(default)s).",
    )
    parser.add_argument(
        "--repetitions",
        type=_positive_integer,
        default=5,
        help="number of synchronized timed pairs (default: %(default)s).",
    )
    parser.add_argument(
        "--max-absolute-error",
        type=_nonnegative_float,
        default=1.0e-9,
        help="largest permitted FP64 round-trip error (default: %(default)g).",
    )
    parser.add_argument(
        "--max-relative-l2-error",
        type=_nonnegative_float,
        default=1.0e-12,
        help="largest permitted relative L2 error (default: %(default)g).",
    )
    return parser


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cuda_version(value: int) -> str:
    """Format CUDA's integer version convention without inventing a patch."""
    major = value // 1000
    minor = (value % 1000) // 10
    return f"{major}.{minor}"


def _property(properties: dict[Any, Any], name: str) -> Any:
    if name in properties:
        return properties[name]
    encoded = name.encode("ascii")
    return properties.get(encoded)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _timing_summary(durations: list[float]) -> dict[str, Any]:
    return {
        "seconds": durations,
        "minimum_seconds": min(durations),
        "median_seconds": statistics.median(durations),
        "maximum_seconds": max(durations),
    }


def _import_cupy() -> Any:
    # Keep this diagnostic free of persistent CuPy and driver compilation
    # caches. These settings affect only this child process.
    os.environ.setdefault("CUPY_CACHE_IN_MEMORY", "1")
    os.environ.setdefault("CUPY_DISABLE_JITIFY_CACHE", "1")
    os.environ.setdefault("CUDA_CACHE_DISABLE", "1")
    try:
        import cupy  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "CuPy could not be imported. Install the L4T-matched "
            "cuda-libraries-13-2 and cuda-cudart-dev-13-2 system packages, "
            "then install "
            "requirements-gpu-jetson.txt in the isolated GPU environment. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return cupy


def _run(arguments: argparse.Namespace, cp: Any) -> dict[str, Any]:
    if arguments.device >= cp.cuda.runtime.getDeviceCount():
        raise RuntimeError(
            f"CUDA device {arguments.device} does not exist; "
            f"device count is {cp.cuda.runtime.getDeviceCount()}"
        )

    with cp.cuda.Device(arguments.device):
        properties = cp.cuda.runtime.getDeviceProperties(arguments.device)
        free_before, total_memory = cp.cuda.runtime.memGetInfo()
        stream = cp.cuda.Stream.null

        # A deterministic, nontrivial signal avoids timing random-number
        # generation and exercises CuPy elementwise compilation before the
        # measured FFT region.
        source = cp.arange(
            arguments.rows * arguments.columns, dtype=cp.float64
        ).reshape(arguments.rows, arguments.columns)
        source = cp.sin(source * 0.000_123) + cp.cos(source * 0.000_071)

        def round_trip() -> Any:
            return cp.fft.ifft2(cp.fft.fft2(source)).real

        for _ in range(arguments.warmups):
            restored = round_trip()
            stream.synchronize()

        durations: list[float] = []
        for _ in range(arguments.repetitions):
            stream.synchronize()
            started = time.perf_counter()
            restored = round_trip()
            stream.synchronize()
            durations.append(time.perf_counter() - started)

        difference = restored - source
        finite = bool(cp.all(cp.isfinite(restored)).item())
        maximum_absolute_error = float(cp.max(cp.abs(difference)).item())
        squared_error = float(cp.sum(difference * difference).item())
        squared_reference = float(cp.sum(source * source).item())
        relative_l2_error = math.sqrt(squared_error / squared_reference)
        free_after, _ = cp.cuda.runtime.memGetInfo()

        config_stream = io.StringIO()
        with redirect_stdout(config_stream):
            cp.show_config()

        driver_version = int(cp.cuda.runtime.driverGetVersion())
        runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
        local_runtime_value = cp.cuda.get_local_runtime_version()
        local_runtime_version = (
            int(local_runtime_value) if local_runtime_value is not None else None
        )
        result = {
            "status": "passed",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host": {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "python_executable": sys.executable,
            },
            "packages": {
                "cupy-cuda13x": _distribution_version("cupy-cuda13x"),
                "cuda-pathfinder": _distribution_version("cuda-pathfinder"),
                "numpy": _distribution_version("numpy"),
            },
            "cuda": {
                "driver_version_integer": driver_version,
                "driver_version": _cuda_version(driver_version),
                "runtime_version_integer": runtime_version,
                "runtime_version": _cuda_version(runtime_version),
                "local_runtime_version_integer": local_runtime_version,
                "local_runtime_version": (
                    _cuda_version(local_runtime_version)
                    if local_runtime_version is not None
                    else None
                ),
                "device_count": int(cp.cuda.runtime.getDeviceCount()),
                "selected_device": arguments.device,
                "device_name": _text(_property(properties, "name")),
                "compute_capability": (
                    f"{_property(properties, 'major')}."
                    f"{_property(properties, 'minor')}"
                ),
                "total_global_memory_bytes": int(
                    _property(properties, "totalGlobalMem") or total_memory
                ),
                "free_memory_before_bytes": int(free_before),
                "free_memory_after_bytes": int(free_after),
                "cupy_show_config": config_stream.getvalue().strip(),
            },
            "workload": {
                "shape": [arguments.rows, arguments.columns],
                "dtype": "float64",
                "warmups": arguments.warmups,
                "repetitions": arguments.repetitions,
                "operation": "ifft2(fft2(input)).real",
                "timing": _timing_summary(durations),
            },
            "numerics": {
                "all_output_pixels_finite": finite,
                "maximum_absolute_error": maximum_absolute_error,
                "maximum_absolute_error_limit": arguments.max_absolute_error,
                "relative_l2_error": relative_l2_error,
                "relative_l2_error_limit": arguments.max_relative_l2_error,
            },
        }

        if (
            not finite
            or maximum_absolute_error > arguments.max_absolute_error
            or relative_l2_error > arguments.max_relative_l2_error
        ):
            result["status"] = "failed_numerical_gate"
        return result


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        cp = _import_cupy()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_IMPORT_FAILURE

    try:
        result = _run(arguments, cp)
    except Exception as exc:
        print(
            "ERROR: CuPy CUDA/FFT validation failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_RUNTIME_FAILURE

    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result["status"] != "passed":
        return EXIT_NUMERICAL_FAILURE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

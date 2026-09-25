"""Compare CME tracking on matched raw and PSF-deconvolved image sequences.

This is an engineering-sensitivity experiment, not an accuracy benchmark: the
synthetic scenarios currently have no authoritative CME-front truth contour.
The same validated sequence geometry, time axis, and tracker configuration are
used for both arms.  Candidate images may be loaded from Level 2 FITS products
or generated with prepared PSF kernels.  Both arms reach the tracker as float32,
so tracking-time changes measure image content rather than a dtype change.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from astropy.io import fits
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from suncet_processing_pipeline.benchmark_cme_tracking import (
    TegrastatsCollector,
    science_signature,
)
from suncet_processing_pipeline import suncet_deconv
from suncet_processing_pipeline.level4.cme_tracking.config import (
    read_configuration,
)
from suncet_processing_pipeline.level4.cme_tracking.input import (
    load_sequence_from_manifest,
)
from suncet_processing_pipeline.level4.cme_tracking.pipeline import (
    CMETrackingRun,
    run_known_window_from_images,
)
from suncet_processing_pipeline.level4.jetson_metrics import (
    summarize_tegrastats_window,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--deconvolved-directory", type=Path)
    candidate.add_argument(
        "--prepared-psf-npz",
        type=Path,
        help=(
            "Prepared spatial diffraction/scatter kernels; deconvolve the raw "
            "sequence in memory on the selected backend."
        ),
    )
    parser.add_argument(
        "--deconvolution-backend", choices=("numpy", "cupy"), default="cupy"
    )
    parser.add_argument(
        "--deconvolved-array-output",
        type=Path,
        help="Optional .npy persistence of the generated float32 candidate cube.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--warmups", type=_positive_int, default=1)
    parser.add_argument("--repetitions", type=_positive_int, default=3)
    parser.add_argument("--telemetry-interval-ms", type=_positive_int, default=100)
    parser.add_argument(
        "--telemetry",
        choices=("auto", "required", "off"),
        default="required",
    )
    parser.add_argument("--allow-inconsistent-synthetic-geometry", action="store_true")
    return parser


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, value: Any) -> None:
    encoded = json.dumps(
        _jsonable(value), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {
            "count": 0,
            "bias": None,
            "mean_absolute": None,
            "median_absolute": None,
            "rms": None,
            "percentile_90_absolute": None,
            "percentile_95_absolute": None,
            "maximum_absolute": None,
        }
    absolute = np.abs(finite)
    return {
        "count": int(finite.size),
        "bias": float(np.mean(finite)),
        "mean_absolute": float(np.mean(absolute)),
        "median_absolute": float(np.median(absolute)),
        "rms": float(np.sqrt(np.mean(np.square(finite)))),
        "percentile_90_absolute": float(np.percentile(absolute, 90.0)),
        "percentile_95_absolute": float(np.percentile(absolute, 95.0)),
        "maximum_absolute": float(np.max(absolute)),
    }


def _first_true(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if indices.size else None


def _circular_difference_deg(candidate: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    return (np.asarray(candidate) - np.asarray(baseline) + 180.0) % 360.0 - 180.0


def _median_speed_km_s(run: CMETrackingRun) -> float | None:
    if run.kinematics is None:
        return None
    valid = run.kinematics.valid_mask
    values = run.kinematics.speed[valid] * run.configuration.solar_radius_km
    return float(np.median(values)) if values.size else None


def _variant_summary(run: CMETrackingRun) -> dict[str, Any]:
    measured = np.isfinite(run.summary.height_rsun)
    uncensored = measured & ~run.field_of_view_limited_mask
    return {
        "event_detected": bool(run.front.event_detected),
        "measured_frame_count": int(np.count_nonzero(measured)),
        "accepted_front_sample_count": int(np.count_nonzero(run.front.observed_mask)),
        "first_fov_limited_frame_index": _first_true(run.field_of_view_limited_mask),
        "fov_limited_frame_count": int(np.count_nonzero(run.field_of_view_limited_mask)),
        "minimum_uncensored_height_rsun": (
            float(np.nanmin(run.summary.height_rsun[uncensored]))
            if np.any(uncensored)
            else None
        ),
        "maximum_uncensored_height_rsun": (
            float(np.nanmax(run.summary.height_rsun[uncensored]))
            if np.any(uncensored)
            else None
        ),
        "median_projected_speed_km_s": _median_speed_km_s(run),
        "kinematically_valid_frame_count": (
            int(np.count_nonzero(run.kinematics.valid_mask))
            if run.kinematics is not None
            else 0
        ),
        "headline_height_outlier_frame_indices": np.flatnonzero(
            run.headline_height_filter.outlier_mask
        ),
        "tracker_quality_flags": list(run.front.quality_flags),
    }


def _comparison(baseline: CMETrackingRun, candidate: CMETrackingRun) -> dict[str, Any]:
    baseline_observed = baseline.front.observed_mask
    candidate_observed = candidate.front.observed_mask
    intersection = baseline_observed & candidate_observed
    union = baseline_observed | candidate_observed
    solar_radius_px = baseline.sequence.geometry.solar_radius_px
    solar_radius_km = baseline.configuration.solar_radius_km

    front_delta_px = (
        candidate.front.radius_px[intersection]
        - baseline.front.radius_px[intersection]
    )
    front_delta_rsun = front_delta_px / solar_radius_px

    # Remove each frame's common-angle median radial shift to isolate a change
    # in front shape from a simple bulk inward/outward displacement.
    shape_residual_px = np.full_like(candidate.front.radius_px, np.nan, dtype=np.float64)
    per_frame_support_jaccard = np.full(baseline.sequence.frame_count, np.nan)
    per_frame_front_rmse_px = np.full(baseline.sequence.frame_count, np.nan)
    per_frame_shape_rmse_px = np.full(baseline.sequence.frame_count, np.nan)
    for frame_index in range(baseline.sequence.frame_count):
        common = intersection[frame_index]
        local_union = union[frame_index]
        if np.any(local_union):
            per_frame_support_jaccard[frame_index] = (
                np.count_nonzero(common) / np.count_nonzero(local_union)
            )
        if not np.any(common):
            continue
        delta = (
            candidate.front.radius_px[frame_index, common]
            - baseline.front.radius_px[frame_index, common]
        )
        residual = delta - np.median(delta)
        shape_residual_px[frame_index, common] = residual
        per_frame_front_rmse_px[frame_index] = np.sqrt(np.mean(np.square(delta)))
        per_frame_shape_rmse_px[frame_index] = np.sqrt(np.mean(np.square(residual)))

    baseline_height = baseline.summary.height_rsun
    candidate_height = candidate.summary.height_rsun
    common_height = np.isfinite(baseline_height) & np.isfinite(candidate_height)
    both_uncensored = (
        common_height
        & ~baseline.field_of_view_limited_mask
        & ~candidate.field_of_view_limited_mask
    )
    before_first_fov = common_height.copy()
    fov_indices = [
        value
        for value in (
            _first_true(baseline.field_of_view_limited_mask),
            _first_true(candidate.field_of_view_limited_mask),
        )
        if value is not None
    ]
    if fov_indices:
        before_first_fov[min(fov_indices) :] = False
    both_kinematic = np.zeros(common_height.shape, dtype=bool)
    if baseline.kinematics is not None and candidate.kinematics is not None:
        both_kinematic = (
            baseline.kinematics.valid_mask
            & candidate.kinematics.valid_mask
            & common_height
        )

    height_delta = candidate_height - baseline_height
    pa_common = np.isfinite(baseline.summary.position_angle_deg) & np.isfinite(
        candidate.summary.position_angle_deg
    )
    width_common = np.isfinite(baseline.summary.angular_width_deg) & np.isfinite(
        candidate.summary.angular_width_deg
    )
    report = {
        "accepted_support": {
            "intersection_count": int(np.count_nonzero(intersection)),
            "union_count": int(np.count_nonzero(union)),
            "jaccard_fraction": (
                float(np.count_nonzero(intersection) / np.count_nonzero(union))
                if np.any(union)
                else None
            ),
            "gained_candidate_sample_count": int(
                np.count_nonzero(candidate_observed & ~baseline_observed)
            ),
            "lost_candidate_sample_count": int(
                np.count_nonzero(baseline_observed & ~candidate_observed)
            ),
            "per_frame_jaccard": _finite_stats(per_frame_support_jaccard),
        },
        "common_front_radius_difference_candidate_minus_baseline": {
            "pixels": _finite_stats(front_delta_px),
            "solar_radii": _finite_stats(front_delta_rsun),
            "kilometers": _finite_stats(front_delta_rsun * solar_radius_km),
            "per_frame_rmse_pixels": _finite_stats(per_frame_front_rmse_px),
        },
        "front_shape_change_after_removing_per_frame_bulk_shift": {
            "pixels": _finite_stats(shape_residual_px),
            "solar_radii": _finite_stats(shape_residual_px / solar_radius_px),
            "kilometers": _finite_stats(
                shape_residual_px / solar_radius_px * solar_radius_km
            ),
            "per_frame_rmse_pixels": _finite_stats(per_frame_shape_rmse_px),
        },
        "headline_height_difference_candidate_minus_baseline_rsun": {
            "all_common_frames": _finite_stats(height_delta[common_height]),
            "both_uncensored_frames": _finite_stats(height_delta[both_uncensored]),
            "before_either_fov_contact": _finite_stats(height_delta[before_first_fov]),
            "both_kinematically_valid_frames": _finite_stats(
                height_delta[both_kinematic]
            ),
        },
        "headline_height_difference_candidate_minus_baseline_km": {
            "both_uncensored_frames": _finite_stats(
                height_delta[both_uncensored] * solar_radius_km
            )
        },
        "central_position_angle_difference_deg": _finite_stats(
            _circular_difference_deg(
                candidate.summary.position_angle_deg[pa_common],
                baseline.summary.position_angle_deg[pa_common],
            )
        ),
        "angular_width_difference_deg": _finite_stats(
            candidate.summary.angular_width_deg[width_common]
            - baseline.summary.angular_width_deg[width_common]
        ),
        "fov_contact_frame_shift_candidate_minus_baseline": (
            _first_true(candidate.field_of_view_limited_mask)
            - _first_true(baseline.field_of_view_limited_mask)
            if _first_true(candidate.field_of_view_limited_mask) is not None
            and _first_true(baseline.field_of_view_limited_mask) is not None
            else None
        ),
    }

    if baseline.kinematics is not None and candidate.kinematics is not None:
        common = baseline.kinematics.valid_mask & candidate.kinematics.valid_mask
        speed_delta = (
            candidate.kinematics.speed - baseline.kinematics.speed
        ) * solar_radius_km
        acceleration_delta = (
            candidate.kinematics.acceleration - baseline.kinematics.acceleration
        ) * solar_radius_km * 1000.0
        report["kinematics_difference_candidate_minus_baseline"] = {
            "speed_km_s": _finite_stats(speed_delta[common]),
            "acceleration_m_s2": _finite_stats(acceleration_delta[common]),
            "common_valid_frame_count": int(np.count_nonzero(common)),
        }
    return report


def _load_deconvolved_images(sequence, directory: Path) -> tuple[np.ndarray, dict[str, Any]]:
    directory = directory.expanduser().resolve()
    images = np.empty(
        (sequence.frame_count, *sequence.geometry.image_shape_yx), dtype=np.float32
    )
    records: list[dict[str, Any]] = []
    calibration_ids: set[str] = set()
    backends: set[str] = set()
    for index, frame in enumerate(sequence.frames):
        matches = sorted(directory.glob(f"{frame.path.stem}_level2_v*.fits"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one Level 2 match for {frame.path.name}, found {len(matches)}"
            )
        path = matches[0]
        with fits.open(path, memmap=False, checksum=True) as hdul:
            hdul.verify("exception")
            if hdul[0].verify_checksum() != 1 or hdul[0].verify_datasum() != 1:
                raise ValueError(f"FITS checksum failed: {path}")
            header = hdul[0].header
            if str(header.get("L0PARENT", "")) != frame.path.name:
                raise ValueError(f"Level 2 parent mismatch in {path}")
            if not bool(header.get("DECONV", False)):
                raise ValueError(f"Level 2 product does not declare DECONV=T: {path}")
            data = np.asarray(hdul[0].data, dtype=np.float32)
            history = header.get("HISTORY", [])
            history_text = "\n".join(
                [history] if isinstance(history, str) else list(history)
            )
            backend = (
                "cupy" if "backend: cupy" in history_text.lower() else "unknown"
            )
            backends.add(backend)
            calibration_ids.add(str(header.get("CALID", "")))
            records.append(
                {
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "calibration_id": header.get("CALID"),
                    "deconvolution_backend": backend,
                    "flux_ratio": header.get("DECONRAT"),
                }
            )
        if data.shape != sequence.geometry.image_shape_yx:
            raise ValueError(f"Unexpected Level 2 shape {data.shape} in {path}")
        images[index] = data
    return images, {
        "directory": str(directory),
        "file_count": len(records),
        "calibration_ids": sorted(calibration_ids),
        "deconvolution_backends": sorted(backends),
        "first_file": records[0],
        "last_file": records[-1],
    }


def _interval_summary(
    collector: TegrastatsCollector,
    start_ns: int,
    end_ns: int,
) -> dict[str, Any] | None:
    if collector.error is not None:
        return None
    time.sleep(max(0.25, 2 * collector.interval_ms / 1000.0))
    return summarize_tegrastats_window(
        collector.snapshot(),
        start_monotonic_ns=start_ns,
        end_monotonic_ns=end_ns,
    )


def _require_power_summary(summary: dict[str, Any] | None, label: str) -> None:
    if summary is None or summary.get("gross_energy_joules") is None:
        raise RuntimeError(f"Required Jetson power telemetry is missing for {label}")
    coverage = summary.get("power_coverage_fraction")
    if coverage is None or float(coverage) < 0.90:
        raise RuntimeError(
            f"Required Jetson power coverage for {label} is incomplete: {coverage!r}"
        )


def _deconvolve_with_prepared_kernels(
    images: np.ndarray,
    prepared_path: Path,
    *,
    backend: str,
    telemetry: str,
    telemetry_interval_ms: int,
    array_output: Path | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    prepared_path = prepared_path.expanduser().resolve()
    sampler = TegrastatsCollector(interval_ms=telemetry_interval_ms)
    if telemetry != "off":
        sampler.start()
        time.sleep(max(0.5, 3 * telemetry_interval_ms / 1000.0))
        if telemetry == "required" and sampler.error is not None:
            raise RuntimeError(sampler.error)

    try:
        preparation_start = time.monotonic_ns()
        with np.load(prepared_path, allow_pickle=False) as archive:
            diffraction_kernel = np.asarray(
                archive["diffraction_kernel"], dtype=np.float64
            )
            scatter_kernel = np.asarray(archive["scatter_kernel"], dtype=np.float64)
            metadata = json.loads(str(archive["metadata_json"].item()))
        expected_shape = tuple(images.shape[1:])
        if metadata.get("schema") != "suncet.prepared_psf_spatial_kernels":
            raise ValueError("Unrecognized prepared-PSF metadata schema")
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported prepared-PSF metadata schema version")
        if tuple(metadata.get("image_shape", ())) != expected_shape:
            raise ValueError(
                "Prepared-PSF metadata image shape does not match the input: "
                f"{metadata.get('image_shape')!r}, {expected_shape!r}"
            )
        if not metadata.get("source_commit") or not metadata.get("sources"):
            raise ValueError("Prepared-PSF metadata lacks source provenance")
        if diffraction_kernel.shape != expected_shape or scatter_kernel.shape != expected_shape:
            raise ValueError(
                "Prepared spatial kernels must match the image shape: "
                f"{diffraction_kernel.shape}, {scatter_kernel.shape}, {expected_shape}"
            )
        array_module = suncet_deconv._array_module_for_backend(backend)
        diffraction_fpsf = suncet_deconv._padded_psf_fft(
            diffraction_kernel, backend=backend
        )
        scatter_array = array_module.asarray(
            scatter_kernel,
            dtype=array_module.float64 if backend == "cupy" else None,
        )
        scatter_fpsf = array_module.fft.fft2(scatter_array)
        deconvolver = suncet_deconv.PreparedDeconvolver(
            diffraction_fpsf,
            scatter_fpsf,
            expected_shape,
            backend=backend,
        )
        deconvolver.synchronize()
        preparation_end = time.monotonic_ns()
        preparation_power = (
            _interval_summary(sampler, preparation_start, preparation_end)
            if telemetry != "off"
            else None
        )
        if telemetry == "required":
            _require_power_summary(preparation_power, "PSF kernel preparation")

        candidate = np.empty_like(images, dtype=np.float32)
        flux_ratios = np.empty(images.shape[0], dtype=np.float64)
        application_start = time.monotonic_ns()
        with redirect_stdout(io.StringIO()):
            for index, image in enumerate(images):
                result = deconvolver.apply(image)
                result = np.asarray(result, dtype=np.float64)
                if not np.all(np.isfinite(result)):
                    raise ValueError(
                        f"Deconvolution produced non-finite values in frame {index}"
                    )
                raw_sum = float(np.sum(image, dtype=np.float64))
                if raw_sum == 0.0:
                    raise ValueError(f"Raw image sum is zero in frame {index}")
                with np.errstate(over="ignore", invalid="ignore"):
                    candidate[index] = result
                if not np.all(np.isfinite(candidate[index])):
                    raise ValueError(
                        f"Float32 deconvolution output overflowed in frame {index}"
                    )
                flux_ratios[index] = float(
                    np.sum(result, dtype=np.float64) / raw_sum
                )
        deconvolver.synchronize()
        application_end = time.monotonic_ns()
        application_power = (
            _interval_summary(sampler, application_start, application_end)
            if telemetry != "off"
            else None
        )
        if telemetry == "required":
            _require_power_summary(application_power, "PSF deconvolution sequence")
    finally:
        if telemetry != "off":
            sampler.stop()

    if telemetry == "required" and sampler.error is not None:
        raise RuntimeError(sampler.error)

    persistence: dict[str, Any] | None = None
    if array_output is not None:
        array_output = array_output.expanduser().resolve()
        array_output.parent.mkdir(parents=True, exist_ok=True)
        if array_output.exists():
            raise FileExistsError(f"Refusing to replace {array_output}")
        temporary = array_output.with_name(f".{array_output.name}.tmp")
        write_started = time.perf_counter()
        try:
            with temporary.open("xb") as stream:
                np.save(stream, candidate, allow_pickle=False)
            temporary.replace(array_output)
        finally:
            temporary.unlink(missing_ok=True)
        persistence = {
            "path": str(array_output),
            "size_bytes": array_output.stat().st_size,
            "sha256": _sha256(array_output),
            "write_and_hash_seconds": time.perf_counter() - write_started,
        }

    return candidate, {
        "source": "prepared_spatial_kernels",
        "prepared_psf_path": str(prepared_path),
        "prepared_psf_sha256": _sha256(prepared_path),
        "prepared_psf_metadata": metadata,
        "backend": backend,
        "output_dtype": str(candidate.dtype),
        "frame_count": int(candidate.shape[0]),
        "kernel_preparation": {
            "duration_seconds": (preparation_end - preparation_start) / 1e9,
            "power": preparation_power,
        },
        "sequence_application": {
            "duration_seconds": (application_end - application_start) / 1e9,
            "seconds_per_frame": (
                (application_end - application_start) / 1e9 / candidate.shape[0]
            ),
            "power": application_power,
            "flux_ratio_minimum": float(np.min(flux_ratios)),
            "flux_ratio_median": float(np.median(flux_ratios)),
            "flux_ratio_maximum": float(np.max(flux_ratios)),
        },
        "persistence": persistence,
        "telemetry_error": sampler.error if telemetry != "off" else None,
    }


def _measure_tracking(
    sequence,
    baseline_images: np.ndarray,
    candidate_images: np.ndarray,
    configuration,
    *,
    warmups: int,
    repetitions: int,
    telemetry: str,
    telemetry_interval_ms: int,
) -> tuple[CMETrackingRun, CMETrackingRun, dict[str, Any], dict[str, Any]]:
    variants = {"undeconvolved": baseline_images, "deconvolved": candidate_images}
    sampler = TegrastatsCollector(interval_ms=telemetry_interval_ms)
    if telemetry != "off":
        sampler.start()
        time.sleep(max(0.5, 3 * telemetry_interval_ms / 1000.0))
        if telemetry == "required" and sampler.error is not None:
            raise RuntimeError(sampler.error)

    references: dict[str, CMETrackingRun] = {}
    signatures: dict[str, dict[str, Any]] = {}
    timings: dict[str, list[dict[str, Any]]] = {name: [] for name in variants}
    try:
        for _ in range(warmups):
            for name in ("undeconvolved", "deconvolved"):
                warm = run_known_window_from_images(
                    sequence, variants[name], configuration
                )
                del warm
                gc.collect()

        for repetition in range(repetitions):
            order = (
                ("undeconvolved", "deconvolved")
                if repetition % 2 == 0
                else ("deconvolved", "undeconvolved")
            )
            for name in order:
                gc.collect()
                time.sleep(max(0.25, 2 * telemetry_interval_ms / 1000.0))
                start_ns = time.monotonic_ns()
                run = run_known_window_from_images(
                    sequence, variants[name], configuration
                )
                end_ns = time.monotonic_ns()
                time.sleep(max(0.25, 2 * telemetry_interval_ms / 1000.0))
                signature = science_signature(run)
                if name not in references:
                    references[name] = run
                    signatures[name] = signature
                else:
                    if signature["overall_sha256"] != signatures[name]["overall_sha256"]:
                        raise RuntimeError(f"Nondeterministic tracker output for {name}")
                    del run
                power = (
                    summarize_tegrastats_window(
                        sampler.snapshot(),
                        start_monotonic_ns=start_ns,
                        end_monotonic_ns=end_ns,
                    )
                    if telemetry != "off"
                    else None
                )
                if telemetry == "required":
                    _require_power_summary(
                        power, f"{name} tracker repetition {repetition + 1}"
                    )
                timings[name].append(
                    {
                        "repetition": repetition + 1,
                        "order_in_repetition": order.index(name) + 1,
                        "duration_seconds": (end_ns - start_ns) / 1e9,
                        "power": power,
                    }
                )
    finally:
        if telemetry != "off":
            sampler.stop()

    if telemetry == "required" and sampler.error is not None:
        raise RuntimeError(sampler.error)

    summaries: dict[str, Any] = {}
    for name, rows in timings.items():
        durations = [float(row["duration_seconds"]) for row in rows]
        energies = [
            float(row["power"]["gross_energy_joules"])
            for row in rows
            if row["power"] is not None
            and row["power"].get("gross_energy_joules") is not None
        ]
        average_powers = [
            float(row["power"]["mean_covered_onboard_power_mw"]) / 1000.0
            for row in rows
            if row["power"] is not None
            and row["power"].get("mean_covered_onboard_power_mw") is not None
        ]
        peaks = [
            float(row["power"]["peak_covered_onboard_power_mw"]) / 1000.0
            for row in rows
            if row["power"] is not None
            and row["power"].get("peak_covered_onboard_power_mw") is not None
        ]
        summaries[name] = {
            "repetitions": rows,
            "science_signature": signatures[name],
            "median_duration_seconds": statistics.median(durations),
            "duration_range_seconds": [min(durations), max(durations)],
            "median_gross_energy_joules": (
                statistics.median(energies) if energies else None
            ),
            "median_average_covered_power_watts": (
                statistics.median(average_powers) if average_powers else None
            ),
            "maximum_sampled_peak_power_watts": max(peaks) if peaks else None,
        }
    return (
        references["undeconvolved"],
        references["deconvolved"],
        summaries,
        {
            "status": sampler.error or ("completed" if telemetry != "off" else "off"),
            "sample_count": len(sampler.snapshot()) if telemetry != "off" else 0,
        },
    )


def _front_xy(run: CMETrackingRun, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    observed = run.front.observed_mask[frame_index]
    angles = np.deg2rad(run.front.position_angle_deg[observed])
    radii = run.front.radius_px[frame_index, observed]
    center_y, center_x = run.sequence.geometry.center_yx
    north_y, north_x = run.sequence.geometry.north_vector_yx
    east_y, east_x = run.sequence.geometry.east_vector_yx
    y = center_y + radii * (np.cos(angles) * north_y + np.sin(angles) * east_y)
    x = center_x + radii * (np.cos(angles) * north_x + np.sin(angles) * east_x)
    return x, y


def _write_plots(
    output: Path,
    baseline: CMETrackingRun,
    candidate: CMETrackingRun,
    baseline_images: np.ndarray,
    candidate_images: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    elapsed_minutes = baseline.sequence.elapsed_seconds / 60.0
    baseline_height = baseline.summary.height_rsun
    candidate_height = candidate.summary.height_rsun
    common = np.isfinite(baseline_height) & np.isfinite(candidate_height)

    figure, axes = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": (3, 1)}
    )
    axes[0].plot(elapsed_minutes, baseline_height, ".-", ms=3, lw=0.8, label="undeconvolved")
    axes[0].plot(elapsed_minutes, candidate_height, ".-", ms=3, lw=0.8, label="PSF deconvolved")
    axes[0].set_ylabel("headline height (R$_\\odot$)")
    axes[0].set_title("CME headline height sensitivity to PSF deconvolution")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(
        elapsed_minutes[common],
        candidate_height[common] - baseline_height[common],
        ".",
        ms=3,
        color="0.2",
    )
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].set_ylabel("decon − raw (R$_\\odot$)")
    axes[1].set_xlabel("elapsed time (minutes)")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "height_comparison.png", dpi=180)
    plt.close(figure)

    baseline_map = np.where(
        baseline.front.observed_mask, baseline.front.radius_rsun, np.nan
    ).T
    candidate_map = np.where(
        candidate.front.observed_mask, candidate.front.radius_rsun, np.nan
    ).T
    difference = candidate_map - baseline_map
    extent = (
        elapsed_minutes[0],
        elapsed_minutes[-1],
        baseline.front.position_angle_deg[0],
        baseline.front.position_angle_deg[-1],
    )
    finite_maps = np.concatenate(
        (baseline_map[np.isfinite(baseline_map)], candidate_map[np.isfinite(candidate_map)])
    )
    limits = np.percentile(finite_maps, (2, 98)) if finite_maps.size else (0.0, 1.0)
    finite_difference = np.abs(difference[np.isfinite(difference)])
    difference_limit = float(np.percentile(finite_difference, 95)) if finite_difference.size else 1.0
    figure, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True, sharey=True)
    for axis, values, title in (
        (axes[0], baseline_map, "undeconvolved retained front"),
        (axes[1], candidate_map, "PSF-deconvolved retained front"),
    ):
        image = axis.imshow(
            values, origin="lower", aspect="auto", extent=extent,
            vmin=limits[0], vmax=limits[1], cmap="viridis"
        )
        axis.set_title(title)
        axis.set_ylabel("solar position angle (deg)")
        figure.colorbar(image, ax=axis, label="radius (R$_\\odot$)")
    image = axes[2].imshow(
        difference, origin="lower", aspect="auto", extent=extent,
        vmin=-difference_limit, vmax=difference_limit, cmap="coolwarm"
    )
    axes[2].set_title("common-sample radial difference: deconvolved − raw")
    axes[2].set_ylabel("solar position angle (deg)")
    axes[2].set_xlabel("elapsed time (minutes)")
    figure.colorbar(image, ax=axes[2], label="difference (R$_\\odot$)")
    figure.tight_layout()
    figure.savefig(output / "front_map_comparison.png", dpi=180)
    plt.close(figure)

    valid = np.flatnonzero(
        np.isfinite(baseline_height)
        & np.isfinite(candidate_height)
        & ~baseline.field_of_view_limited_mask
        & ~candidate.field_of_view_limited_mask
    )
    if valid.size:
        selected = np.unique(
            valid[np.round(np.linspace(0.25, 0.85, 3) * (valid.size - 1)).astype(int)]
        )
        figure, axes = plt.subplots(len(selected), 2, figsize=(13, 4.1 * len(selected)))
        axes = np.atleast_2d(axes)
        for row, frame_index in enumerate(selected):
            pair = np.concatenate(
                (baseline_images[frame_index].ravel(), candidate_images[frame_index].ravel())
            )
            finite = pair[np.isfinite(pair)]
            low, high = np.percentile(finite, (1.0, 99.8))
            width = max((high - low) / 100.0, np.finfo(float).eps)
            display_minimum = 0.0
            display_maximum = float(np.arcsinh(100.0))
            for column, (image_data, run, label, color) in enumerate(
                (
                    (baseline_images[frame_index], baseline, "undeconvolved", "#00d7df"),
                    (candidate_images[frame_index], candidate, "PSF deconvolved", "#ff4fa3"),
                )
            ):
                display = np.arcsinh((np.asarray(image_data, dtype=np.float64) - low) / width)
                axis = axes[row, column]
                axis.imshow(
                    display,
                    origin="lower",
                    cmap="gray",
                    vmin=display_minimum,
                    vmax=display_maximum,
                )
                x, y = _front_xy(run, int(frame_index))
                axis.scatter(x, y, s=8, color=color, edgecolors="none")
                axis.set_title(
                    f"{label} | frame {frame_index} | t={elapsed_minutes[frame_index]:.1f} min"
                )
                axis.set_xticks([])
                axis.set_yticks([])
        figure.suptitle(
            "Matched images and exact retained fronts "
            "(true shared per-frame 1–99.8% asinh mapping)"
        )
        figure.tight_layout()
        figure.savefig(output / "front_overlay_comparison.png", dpi=180)
        plt.close(figure)


def main() -> int:
    arguments = _parser().parse_args()
    manifest = arguments.manifest.expanduser().resolve()
    output = arguments.output_directory.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}")

    if (
        arguments.deconvolved_array_output is not None
        and arguments.prepared_psf_npz is None
    ):
        raise ValueError(
            "--deconvolved-array-output requires --prepared-psf-npz"
        )
    if (
        arguments.deconvolved_array_output is not None
        and arguments.deconvolved_array_output.expanduser().resolve().exists()
    ):
        raise FileExistsError(
            f"Refusing to replace {arguments.deconvolved_array_output}"
        )

    sequence, _ = load_sequence_from_manifest(
        manifest,
        data_root=arguments.data_root,
        verify_hashes=True,
        allow_inconsistent_geometry=arguments.allow_inconsistent_synthetic_geometry,
    )
    baseline_images = sequence.materialize(dtype=np.float32)
    if arguments.prepared_psf_npz is not None:
        candidate_images, candidate_record = _deconvolve_with_prepared_kernels(
            baseline_images,
            arguments.prepared_psf_npz,
            backend=arguments.deconvolution_backend,
            telemetry=arguments.telemetry,
            telemetry_interval_ms=arguments.telemetry_interval_ms,
            array_output=arguments.deconvolved_array_output,
        )
    else:
        candidate_images, candidate_record = _load_deconvolved_images(
            sequence, arguments.deconvolved_directory
        )
    configuration = read_configuration(arguments.config)

    baseline, candidate, timing, telemetry = _measure_tracking(
        sequence,
        baseline_images,
        candidate_images,
        configuration,
        warmups=arguments.warmups,
        repetitions=arguments.repetitions,
        telemetry=arguments.telemetry,
        telemetry_interval_ms=arguments.telemetry_interval_ms,
    )
    comparison = _comparison(baseline, candidate)
    output.mkdir(parents=True, exist_ok=False)
    raw_duration = timing["undeconvolved"]["median_duration_seconds"]
    decon_duration = timing["deconvolved"]["median_duration_seconds"]
    comparison["tracking_compute"] = {
        "duration_ratio_deconvolved_over_undeconvolved": decon_duration / raw_duration,
        "duration_change_percent": 100.0 * (decon_duration / raw_duration - 1.0),
        "gross_energy_ratio_deconvolved_over_undeconvolved": (
            timing["deconvolved"]["median_gross_energy_joules"]
            / timing["undeconvolved"]["median_gross_energy_joules"]
            if timing["deconvolved"]["median_gross_energy_joules"] is not None
            and timing["undeconvolved"]["median_gross_energy_joules"] is not None
            else None
        ),
    }
    report = {
        "schema": "suncet.cme_tracking.psf_deconvolution_ab",
        "schema_version": 1,
        "interpretation": (
            "Sensitivity/stability comparison only; the scenario has no authoritative "
            "CME-front truth contour."
        ),
        "manifest": manifest.name,
        "manifest_sha256": _sha256(manifest),
        "configuration": arguments.config.name,
        "configuration_sha256": _sha256(arguments.config),
        "frame_count": sequence.frame_count,
        "cadence_seconds": float(np.median(np.diff(sequence.elapsed_seconds))),
        "tracking_dtype_both_arms": str(baseline_images.dtype),
        "deconvolved_products": candidate_record,
        "undeconvolved": {
            **_variant_summary(baseline),
            "tracking_measurement": timing["undeconvolved"],
        },
        "deconvolved": {
            **_variant_summary(candidate),
            "tracking_measurement": timing["deconvolved"],
        },
        "comparison": comparison,
        "telemetry": telemetry,
    }
    _atomic_json(output / "science_comparison.json", report)
    np.savez_compressed(
        output / "comparison_arrays.npz",
        elapsed_seconds=sequence.elapsed_seconds,
        position_angle_deg=baseline.front.position_angle_deg,
        undeconvolved_front_radius_px=baseline.front.radius_px,
        deconvolved_front_radius_px=candidate.front.radius_px,
        undeconvolved_front_observed_mask=baseline.front.observed_mask,
        deconvolved_front_observed_mask=candidate.front.observed_mask,
        undeconvolved_height_rsun=baseline.summary.height_rsun,
        deconvolved_height_rsun=candidate.summary.height_rsun,
        undeconvolved_fov_mask=baseline.field_of_view_limited_mask,
        deconvolved_fov_mask=candidate.field_of_view_limited_mask,
    )
    _write_plots(output, baseline, candidate, baseline_images, candidate_images)
    completion = {
        "schema": "suncet.cme_tracking.psf_deconvolution_ab.complete",
        "schema_version": 1,
        "files": {
            path.name: _sha256(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "COMPLETE.json"
        },
    }
    _atomic_json(output / "COMPLETE.json", completion)
    print(json.dumps(_jsonable(comparison), indent=2, sort_keys=True))
    print(f"Comparison artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

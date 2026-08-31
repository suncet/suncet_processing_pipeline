"""Run and summarize a controlled raw-versus-temporal-median CME experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from suncet_processing_pipeline.level4.cme_tracking.config import read_configuration
from suncet_processing_pipeline.level4.cme_tracking.input import (
    load_sequence_from_manifest,
)
from suncet_processing_pipeline.level4.cme_tracking.pipeline import (
    CMETrackingRun,
    run_known_window_from_images,
)
from suncet_processing_pipeline.level4.cme_tracking.tracking import (
    _frame_candidates,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-config", type=Path, required=True)
    parser.add_argument("--filtered-config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--allow-inconsistent-synthetic-geometry", action="store_true")
    return parser


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
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


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_sha256(run: CMETrackingRun) -> str:
    encoded = json.dumps(
        run.configuration.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {
            "count": 0,
            "median": None,
            "median_absolute": None,
            "rms": None,
            "percentile_95_absolute": None,
            "maximum_absolute": None,
        }
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "median_absolute": float(np.median(np.abs(finite))),
        "rms": float(np.sqrt(np.mean(finite**2))),
        "percentile_95_absolute": float(np.percentile(np.abs(finite), 95.0)),
        "maximum_absolute": float(np.max(np.abs(finite))),
    }


def _first_true(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if indices.size else None


def _last_true(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[-1]) if indices.size else None


def _candidate_inventory(run: CMETrackingRun) -> tuple[np.ndarray, list[list[list]]]:
    frame_count, angle_count, _ = run.likelihood.score.shape
    counts = np.zeros(frame_count, dtype=np.int64)
    inventory: list[list[list]] = [
        [[] for _ in range(angle_count)] for _ in range(frame_count)
    ]
    for frame_index in range(frame_count):
        for angle_index in range(angle_count):
            candidates = _frame_candidates(
                run.likelihood.score[frame_index, angle_index],
                run.polar_grid,
                run.configuration.tracking,
            )
            inventory[frame_index][angle_index] = list(candidates)
            counts[frame_index] += len(candidates)
    return counts, inventory


def _isolated_candidate_count(run: CMETrackingRun, inventory: list[list[list]]) -> int:
    frame_count = len(inventory)
    angle_count = len(inventory[0])
    radius_px = run.polar_grid.radius_px
    config = run.configuration.tracking

    def has_match(frame_index: int, angle_index: int, radius: float, step: int) -> bool:
        neighbor_frame = frame_index + step
        if not 0 <= neighbor_frame < frame_count:
            return False
        for offset in (-1, 0, 1):
            neighbor_angle = (angle_index + offset) % angle_count
            for candidate in inventory[neighbor_frame][neighbor_angle]:
                neighbor_radius = float(radius_px[candidate.radius_index])
                outward_delta = (
                    neighbor_radius - radius if step > 0 else radius - neighbor_radius
                )
                if (
                    -config.inward_localization_tolerance_px
                    <= outward_delta
                    <= config.maximum_outward_step_px_per_frame
                ):
                    return True
        return False

    isolated = 0
    for frame_index, by_angle in enumerate(inventory):
        for angle_index, candidates in enumerate(by_angle):
            for candidate in candidates:
                radius = float(radius_px[candidate.radius_index])
                if not has_match(frame_index, angle_index, radius, -1) and not has_match(
                    frame_index, angle_index, radius, 1
                ):
                    isolated += 1
    return isolated


def _kinematic_summary(run: CMETrackingRun) -> dict[str, Any]:
    if run.kinematics is None:
        return {"available": False, "error": run.kinematics_error}
    valid = run.kinematics.valid_mask
    solar_radius_km = run.configuration.solar_radius_km
    speed = run.kinematics.speed[valid] * solar_radius_km
    acceleration = run.kinematics.acceleration[valid] * solar_radius_km * 1000.0
    return {
        "available": True,
        "valid_frame_count": int(np.count_nonzero(valid)),
        "first_valid_frame_index": _first_true(valid),
        "last_valid_frame_index": _last_true(valid),
        "median_speed_km_s": float(np.median(speed)) if speed.size else None,
        "maximum_speed_km_s": float(np.max(speed)) if speed.size else None,
        "median_acceleration_m_s2": (
            float(np.median(acceleration)) if acceleration.size else None
        ),
    }


def _variant_summary(
    run: CMETrackingRun,
    runtime_seconds: float,
    threshold_voxels_by_frame: np.ndarray,
    candidate_count_by_frame: np.ndarray,
    isolated_candidate_count: int,
) -> dict[str, Any]:
    measured = np.isfinite(run.summary.height_rsun)
    uncensored = measured & ~run.field_of_view_limited_mask
    observed_samples = int(np.count_nonzero(run.front.observed_mask))
    total_candidates = int(np.sum(candidate_count_by_frame))
    return {
        "algorithm_name": run.configuration.algorithm_name,
        "configuration_sha256": _config_sha256(run),
        "temporal_median_window_frames": (
            run.configuration.evidence.temporal_median_window_frames
        ),
        "runtime_seconds": runtime_seconds,
        "event_detected": bool(run.front.event_detected),
        "tracker_quality_flags": list(run.front.quality_flags),
        "first_measured_frame_index": _first_true(measured),
        "last_measured_frame_index": _last_true(measured),
        "measured_frame_count": int(np.count_nonzero(measured)),
        "observed_front_sample_count": observed_samples,
        "first_fov_limited_frame_index": _first_true(
            run.field_of_view_limited_mask
        ),
        "fov_limited_frame_count": int(
            np.count_nonzero(run.field_of_view_limited_mask)
        ),
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
        "threshold_voxel_count": int(np.sum(threshold_voxels_by_frame)),
        "candidate_count": total_candidates,
        "isolated_candidate_count": isolated_candidate_count,
        "isolated_candidate_fraction": (
            isolated_candidate_count / total_candidates if total_candidates else None
        ),
        "headline_height_outlier_frame_indices": np.flatnonzero(
            run.headline_height_filter.outlier_mask
        ),
        "kinematics": _kinematic_summary(run),
    }


def _comparison(raw: CMETrackingRun, filtered: CMETrackingRun) -> dict[str, Any]:
    raw_observed = raw.front.observed_mask
    filtered_observed = filtered.front.observed_mask
    intersection = raw_observed & filtered_observed
    union = raw_observed | filtered_observed
    radius_difference_px = (
        filtered.front.radius_px[intersection] - raw.front.radius_px[intersection]
    )
    radius_difference_rsun = radius_difference_px / raw.sequence.geometry.solar_radius_px

    raw_height = raw.summary.height_rsun
    filtered_height = filtered.summary.height_rsun
    common_headline = np.isfinite(raw_height) & np.isfinite(filtered_height)
    common_uncensored = (
        common_headline
        & ~raw.field_of_view_limited_mask
        & ~filtered.field_of_view_limited_mask
    )
    headline_difference = filtered_height - raw_height
    raw_fov_start = _first_true(raw.field_of_view_limited_mask)
    before_raw_fov = common_headline.copy()
    if raw_fov_start is not None:
        before_raw_fov[raw_fov_start:] = False
    both_kinematically_valid = np.zeros(common_headline.shape, dtype=bool)
    if raw.kinematics is not None and filtered.kinematics is not None:
        both_kinematically_valid = (
            common_headline
            & raw.kinematics.valid_mask
            & filtered.kinematics.valid_mask
        )

    common_front_core = intersection & both_kinematically_valid[:, np.newaxis]
    core_radius_difference_px = (
        filtered.front.radius_px[common_front_core]
        - raw.front.radius_px[common_front_core]
    )

    return {
        "observed_mask": {
            "intersection_count": int(np.count_nonzero(intersection)),
            "union_count": int(np.count_nonzero(union)),
            "jaccard_fraction": (
                float(np.count_nonzero(intersection) / np.count_nonzero(union))
                if np.any(union)
                else None
            ),
            "gained_filtered_sample_count": int(
                np.count_nonzero(filtered_observed & ~raw_observed)
            ),
            "lost_filtered_sample_count": int(
                np.count_nonzero(raw_observed & ~filtered_observed)
            ),
        },
        "common_front_radius_difference_px_filtered_minus_raw": _finite_stats(
            radius_difference_px
        ),
        "common_front_radius_difference_rsun_filtered_minus_raw": _finite_stats(
            radius_difference_rsun
        ),
        "common_front_radius_difference_during_both_valid_kinematics_px": (
            _finite_stats(core_radius_difference_px)
        ),
        "common_front_radius_difference_during_both_valid_kinematics_rsun": (
            _finite_stats(
                core_radius_difference_px
                / raw.sequence.geometry.solar_radius_px
            )
        ),
        "headline_height_difference_rsun_filtered_minus_raw": {
            "all_common_frames": _finite_stats(
                headline_difference[common_headline]
            ),
            "both_uncensored_frames": _finite_stats(
                headline_difference[common_uncensored]
            ),
            "before_raw_fov_contact": _finite_stats(
                headline_difference[before_raw_fov]
            ),
            "both_kinematically_valid_frames": _finite_stats(
                headline_difference[both_kinematically_valid]
            ),
        },
    }


def _write_plots(
    output: Path,
    raw: CMETrackingRun,
    filtered: CMETrackingRun,
    raw_threshold: np.ndarray,
    filtered_threshold: np.ndarray,
    raw_candidates: np.ndarray,
    filtered_candidates: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    elapsed_minutes = raw.sequence.elapsed_seconds / 60.0
    raw_height = raw.summary.height_rsun
    filtered_height = filtered.summary.height_rsun
    common = np.isfinite(raw_height) & np.isfinite(filtered_height)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )
    axes[0].plot(elapsed_minutes, raw_height, ".-", ms=3, lw=0.8, label="raw")
    axes[0].plot(
        elapsed_minutes,
        filtered_height,
        ".-",
        ms=3,
        lw=0.8,
        label="temporal median (3 frames)",
    )
    for run, color, label in (
        (raw, "C0", "raw FOV contact"),
        (filtered, "C1", "median-3 FOV contact"),
    ):
        first = _first_true(run.field_of_view_limited_mask)
        if first is not None:
            axes[0].axvline(
                elapsed_minutes[first], color=color, ls="--", alpha=0.65, label=label
            )
    axes[0].set_ylabel("headline height (R$_\\odot$)")
    axes[0].set_title("SunCET CME track: raw versus centered temporal median")
    axes[0].legend(ncol=2, fontsize=9)
    axes[0].grid(alpha=0.2)
    axes[1].plot(
        elapsed_minutes[common],
        filtered_height[common] - raw_height[common],
        ".",
        ms=3,
        color="0.2",
    )
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].set_ylabel("median-3 − raw (R$_\\odot$)")
    axes[1].set_xlabel("elapsed time (minutes)")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "height_comparison.png", dpi=180)
    plt.close(figure)

    observed_raw = np.count_nonzero(raw.front.observed_mask, axis=1)
    observed_filtered = np.count_nonzero(filtered.front.observed_mask, axis=1)
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for axis, raw_values, filtered_values, label in (
        (axes[0], raw_threshold, filtered_threshold, "score voxels ≥ threshold"),
        (axes[1], raw_candidates, filtered_candidates, "radial peak candidates"),
        (axes[2], observed_raw, observed_filtered, "retained front angles"),
    ):
        axis.plot(elapsed_minutes, raw_values, lw=0.9, label="raw")
        axis.plot(elapsed_minutes, filtered_values, lw=0.9, label="median-3")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0].set_title("Evidence suppression and retained support")
    axes[0].legend()
    axes[-1].set_xlabel("elapsed time (minutes)")
    figure.tight_layout()
    figure.savefig(output / "evidence_support_comparison.png", dpi=180)
    plt.close(figure)

    raw_map = np.where(raw.front.observed_mask, raw.front.radius_rsun, np.nan).T
    filtered_map = np.where(
        filtered.front.observed_mask, filtered.front.radius_rsun, np.nan
    ).T
    difference = filtered_map - raw_map
    extent = (
        elapsed_minutes[0],
        elapsed_minutes[-1],
        raw.front.position_angle_deg[0],
        raw.front.position_angle_deg[-1],
    )
    figure, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True, sharey=True)
    common_limits = np.nanpercentile(
        np.concatenate((raw_map[np.isfinite(raw_map)], filtered_map[np.isfinite(filtered_map)])),
        (2, 98),
    )
    for axis, values, title in (
        (axes[0], raw_map, "raw retained front"),
        (axes[1], filtered_map, "temporal-median-3 retained front"),
    ):
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=common_limits[0],
            vmax=common_limits[1],
            cmap="viridis",
        )
        axis.set_title(title)
        axis.set_ylabel("solar position angle (deg)")
        figure.colorbar(image, ax=axis, label="radius (R$_\\odot$)")
    difference_limit = np.nanpercentile(np.abs(difference), 95.0)
    difference_image = axes[2].imshow(
        difference,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-difference_limit,
        vmax=difference_limit,
        cmap="coolwarm",
    )
    axes[2].set_title("common-sample radius difference: median-3 − raw")
    axes[2].set_ylabel("solar position angle (deg)")
    axes[2].set_xlabel("elapsed time (minutes)")
    figure.colorbar(difference_image, ax=axes[2], label="difference (R$_\\odot$)")
    figure.tight_layout()
    figure.savefig(output / "front_map_comparison.png", dpi=180)
    plt.close(figure)


def main() -> int:
    arguments = _parser().parse_args()
    manifest = arguments.manifest.expanduser().resolve()
    output = arguments.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    sequence, _ = load_sequence_from_manifest(
        manifest,
        data_root=arguments.data_root,
        verify_hashes=True,
        allow_inconsistent_geometry=arguments.allow_inconsistent_synthetic_geometry,
    )
    images = sequence.materialize(dtype=np.float32)
    raw_config = read_configuration(arguments.raw_config)
    filtered_config = read_configuration(arguments.filtered_config)
    if raw_config.polar != filtered_config.polar:
        raise ValueError("A/B configurations must use the same polar grid.")
    if raw_config.evidence.temporal_median_window_frames != 1:
        raise ValueError("The raw A/B configuration must use temporal window 1.")
    if filtered_config.evidence.temporal_median_window_frames <= 1:
        raise ValueError("The filtered A/B configuration must enable a temporal median.")

    started = time.perf_counter()
    raw = run_known_window_from_images(sequence, images, raw_config)
    raw_runtime = time.perf_counter() - started
    started = time.perf_counter()
    filtered = run_known_window_from_images(sequence, images, filtered_config)
    filtered_runtime = time.perf_counter() - started

    threshold = raw_config.tracking.score_threshold
    raw_threshold = np.count_nonzero(raw.likelihood.score >= threshold, axis=(1, 2))
    filtered_threshold = np.count_nonzero(
        filtered.likelihood.score >= threshold, axis=(1, 2)
    )
    raw_candidates, raw_inventory = _candidate_inventory(raw)
    filtered_candidates, filtered_inventory = _candidate_inventory(filtered)
    raw_isolated = _isolated_candidate_count(raw, raw_inventory)
    filtered_isolated = _isolated_candidate_count(filtered, filtered_inventory)

    report = {
        "schema_version": 1,
        "experiment": "centered_temporal_median_ab",
        "manifest": manifest.name,
        "manifest_sha256": _sha256(manifest),
        "frame_count": sequence.frame_count,
        "cadence_seconds": float(np.median(np.diff(sequence.elapsed_seconds))),
        "endpoint_policy": (
            "Median-3 frames 0 and N-1 are unsupported; no padding or timestamp shift."
        ),
        "latency_frames": filtered_config.evidence.temporal_median_window_frames // 2,
        "raw": _variant_summary(
            raw,
            raw_runtime,
            raw_threshold,
            raw_candidates,
            raw_isolated,
        ),
        "temporal_median": _variant_summary(
            filtered,
            filtered_runtime,
            filtered_threshold,
            filtered_candidates,
            filtered_isolated,
        ),
        "comparison": _comparison(raw, filtered),
    }
    report["comparison"]["threshold_voxel_reduction_fraction"] = (
        1.0 - np.sum(filtered_threshold) / np.sum(raw_threshold)
        if np.sum(raw_threshold)
        else None
    )
    report["comparison"]["candidate_reduction_fraction"] = (
        1.0 - np.sum(filtered_candidates) / np.sum(raw_candidates)
        if np.sum(raw_candidates)
        else None
    )
    report["comparison"]["isolated_candidate_reduction_fraction"] = (
        1.0 - filtered_isolated / raw_isolated if raw_isolated else None
    )
    report["comparison"]["runtime_ratio_filtered_over_raw"] = (
        filtered_runtime / raw_runtime
    )
    raw_variant = report["raw"]
    filtered_variant = report["temporal_median"]
    onset_shift = (
        filtered_variant["first_measured_frame_index"]
        - raw_variant["first_measured_frame_index"]
    )
    end_shift = (
        filtered_variant["last_measured_frame_index"]
        - raw_variant["last_measured_frame_index"]
    )
    speed_ratio = (
        filtered_variant["kinematics"]["median_speed_km_s"]
        / raw_variant["kinematics"]["median_speed_km_s"]
    )
    real_event_gates = {
        "isolated_candidate_reduction_at_least_80_percent": (
            report["comparison"]["isolated_candidate_reduction_fraction"] >= 0.80
        ),
        "measured_frame_retention_at_least_85_percent": (
            filtered_variant["measured_frame_count"]
            / raw_variant["measured_frame_count"]
            >= 0.85
        ),
        "endpoint_shift_no_more_than_one_frame": (
            abs(onset_shift) <= 1 and abs(end_shift) <= 1
        ),
        "median_headline_shift_before_raw_fov_no_more_than_0_05_rsun": (
            report["comparison"]["headline_height_difference_rsun_filtered_minus_raw"]
            ["before_raw_fov_contact"]["median_absolute"]
            <= 0.05
        ),
        "median_speed_ratio_between_0_8_and_1_2": 0.8 <= speed_ratio <= 1.2,
        "fov_contact_not_earlier": (
            filtered_variant["first_fov_limited_frame_index"]
            >= raw_variant["first_fov_limited_frame_index"]
        ),
    }
    report["comparison"]["real_event_acceptance"] = {
        "all_gates_pass": all(real_event_gates.values()),
        "gates": real_event_gates,
        "measured_frame_retention_fraction": (
            filtered_variant["measured_frame_count"]
            / raw_variant["measured_frame_count"]
        ),
        "onset_shift_frames": onset_shift,
        "end_shift_frames": end_shift,
        "median_speed_ratio_filtered_over_raw": speed_ratio,
        "interpretation": (
            "Engineering acceptance on this no-truth event is not an accuracy "
            "validation or approval as a universal default."
        ),
    }

    _write_json(output / "science_comparison.json", report)
    _write_plots(
        output,
        raw,
        filtered,
        raw_threshold,
        filtered_threshold,
        raw_candidates,
        filtered_candidates,
    )
    print(json.dumps(_jsonable(report["comparison"]), indent=2, sort_keys=True))
    print(f"Comparison artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

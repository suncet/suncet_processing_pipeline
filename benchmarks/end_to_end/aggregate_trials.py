#!/usr/bin/env python3
"""Aggregate repeated end-to-end Jetson ``measurement.json`` trials.

The input files remain the authoritative per-trial records.  This utility
checks that their phase workloads and paired-projection definitions are
compatible, then writes a compact machine-readable and human-readable trial
summary.  It deliberately does not average sampled peaks across phases or add
the paired diagnostic baseline to an operational pipeline total.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from io import StringIO
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


_MEASUREMENT_SCHEMA = "suncet.end_to_end_power_measurement"
_PLAN_SCHEMA = "suncet.end_to_end_plan"
_SUMMARY_SCHEMA = "suncet.end_to_end_trial_summary"
_OUTPUT_NAMES = (
    "trial_summary.json",
    "trial_summary.csv",
    "trial_summary.md",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "measurements",
        nargs="+",
        type=Path,
        help="measurement.json files, or run directories containing one",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing trial_summary.json/.csv/.md files",
    )
    return parser


def _measurement_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "measurement.json"
    return resolved


def _finite_number(value: Any, context: str, *, allow_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    if not allow_negative and number < 0:
        raise ValueError(f"{context} must be nonnegative")
    return number


def _optional_number(value: Any, context: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, context)


def _numeric_mapping(raw: Any, context: str) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be an object")
    result: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{context} contains an invalid unit name")
        number = _finite_number(value, f"{context}.{name}")
        if number <= 0:
            raise ValueError(f"{context}.{name} must be positive")
        result[name] = number
    return result


def _read_measurement(path: Path) -> dict[str, Any]:
    resolved = _measurement_path(path)
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {resolved}: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"Measurement root must be an object: {resolved}")
    if report.get("schema") != _MEASUREMENT_SCHEMA or report.get("schema_version") != 1:
        raise ValueError(f"Unsupported measurement schema: {resolved}")
    if report.get("status") != "passed":
        raise ValueError(
            f"Only passed measurements can be aggregated; {resolved} has "
            f"status {report.get('status')!r}"
        )
    plan = report.get("plan")
    if not isinstance(plan, Mapping):
        raise ValueError(f"Measurement has no plan object: {resolved}")
    if plan.get("schema") != _PLAN_SCHEMA or plan.get("schema_version") != 1:
        raise ValueError(f"Unsupported embedded plan schema: {resolved}")
    report["_source_path"] = str(resolved)
    return report


def _normalize_for_comparison(value: Any, report: Mapping[str, Any]) -> Any:
    """Remove only trial-directory variation from an expanded plan value."""

    run_directory = report.get("run_directory")
    if isinstance(value, str):
        if isinstance(run_directory, str) and run_directory:
            return value.replace(run_directory, "${RUN_DIR}")
        return value
    if isinstance(value, list):
        return [_normalize_for_comparison(item, report) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_for_comparison(item, report)
            for key, item in value.items()
        }
    return value


def _phase_definitions(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    plan_phases = report["plan"].get("phases")
    measured_phases = report.get("phases")
    if not isinstance(plan_phases, list) or not isinstance(measured_phases, list):
        raise ValueError("Both plan.phases and phases must be arrays")

    definitions: dict[str, dict[str, Any]] = {}
    for index, phase in enumerate(plan_phases):
        if not isinstance(phase, Mapping):
            raise ValueError(f"plan.phases[{index}] must be an object")
        name = phase.get("name")
        if not isinstance(name, str) or not name or name in definitions:
            raise ValueError(f"Invalid or duplicate plan phase name: {name!r}")
        definitions[name] = {
            "name": name,
            "description": phase.get("description"),
            "workload_units": _numeric_mapping(
                phase.get("workload_units", {}),
                f"plan phase {name!r} workload_units",
            ),
            "command": _normalize_for_comparison(phase.get("command"), report),
            "cwd": _normalize_for_comparison(phase.get("cwd"), report),
            "environment_overrides": _normalize_for_comparison(
                phase.get("env", {}), report
            ),
            "required_paths": _normalize_for_comparison(
                phase.get("required_paths", []), report
            ),
            "expected_outputs": _normalize_for_comparison(
                phase.get("expected_outputs", []), report
            ),
        }

    measured_names: list[str] = []
    for index, phase in enumerate(measured_phases):
        if not isinstance(phase, Mapping):
            raise ValueError(f"phases[{index}] must be an object")
        name = phase.get("name")
        if not isinstance(name, str) or not name or name in measured_names:
            raise ValueError(f"Invalid or duplicate measured phase name: {name!r}")
        measured_names.append(name)
        if phase.get("status") != "passed":
            raise ValueError(f"Measured phase {name!r} did not pass")
        units = _numeric_mapping(
            phase.get("workload_units", {}),
            f"measured phase {name!r} workload_units",
        )
        if name not in definitions or units != definitions[name]["workload_units"]:
            raise ValueError(
                f"Measured phase {name!r} does not match its plan workload"
            )
    if set(measured_names) != set(definitions):
        raise ValueError("Measured phases do not exactly match the declared plan phases")
    return definitions


def _derived_definitions(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    plan_metrics = report["plan"].get("derived_metrics", [])
    measured_metrics = report.get("derived_metrics", {})
    if not isinstance(plan_metrics, list) or not isinstance(measured_metrics, Mapping):
        raise ValueError("Plan and measured derived metrics have invalid containers")
    definitions: dict[str, dict[str, Any]] = {}
    signature_fields = (
        "type",
        "baseline_phase",
        "full_phase",
        "unit",
        "observed_units",
        "target_units",
        "target_frame_count",
        "additional_phases",
    )
    for index, metric in enumerate(plan_metrics):
        if not isinstance(metric, Mapping):
            raise ValueError(f"plan.derived_metrics[{index}] must be an object")
        name = metric.get("name")
        if not isinstance(name, str) or not name or name in definitions:
            raise ValueError(f"Invalid or duplicate derived metric name: {name!r}")
        definitions[name] = {
            "name": name,
            "description": metric.get("description"),
            **{field: metric.get(field) for field in signature_fields},
        }
    if set(measured_metrics) != set(definitions):
        raise ValueError(
            "Measured derived metrics do not exactly match the declared definitions"
        )
    return definitions


def _compatibility_signature(report: Mapping[str, Any]) -> dict[str, Any]:
    phase_definitions = _phase_definitions(report)
    derived_definitions = _derived_definitions(report)
    interval = report.get("telemetry_interval_ms")
    if interval is not None:
        interval = _finite_number(interval, "telemetry_interval_ms")
    return {
        "phase_definitions": phase_definitions,
        "derived_definitions": derived_definitions,
        "telemetry_interval_ms": interval,
        "plan_context": {
            "notes": report["plan"].get("notes"),
            "pre_staged_paths": _normalize_for_comparison(
                report["plan"].get("pre_staged_paths", []), report
            ),
        },
    }


def _validate_compatible(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reference = _compatibility_signature(reports[0])
    for index, report in enumerate(reports[1:], start=2):
        candidate = _compatibility_signature(report)
        if candidate["phase_definitions"] != reference["phase_definitions"]:
            raise ValueError(f"Measurement {index} has incompatible phase definitions")
        if candidate["derived_definitions"] != reference["derived_definitions"]:
            raise ValueError(
                f"Measurement {index} has incompatible paired-projection definitions"
            )
        if candidate["telemetry_interval_ms"] != reference["telemetry_interval_ms"]:
            raise ValueError(
                f"Measurement {index} uses a different telemetry sampling interval"
            )
        if candidate["plan_context"] != reference["plan_context"]:
            raise ValueError(
                f"Measurement {index} has incompatible plan notes or pre-staged inputs"
            )
    return reference


def _stats(values: Iterable[float | None]) -> dict[str, Any]:
    available = [float(value) for value in values if value is not None]
    if not available:
        return {"count": 0, "median": None, "minimum": None, "maximum": None, "range": None}
    minimum = min(available)
    maximum = max(available)
    return {
        "count": len(available),
        "median": statistics.median(available),
        "minimum": minimum,
        "maximum": maximum,
        "range": maximum - minimum,
    }


def _phase_by_name(report: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for phase in report["phases"]:
        if phase["name"] == name:
            return phase
    raise ValueError(f"Measurement is missing phase {name!r}")


def _phase_metric(
    phase: Mapping[str, Any], key: str, *, context: str
) -> float | None:
    power = phase.get("power")
    if not isinstance(power, Mapping):
        raise ValueError(f"{context}.power must be an object")
    return _optional_number(power.get(key), f"{context}.power.{key}")


def _divide(value: float | None, divisor: float | None) -> float | None:
    if value is None or divisor is None:
        return None
    return value / divisor


def _unique_text(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, str) and value and value not in result:
            result.append(value)
    return result


def _aggregate_phases(
    reports: Sequence[Mapping[str, Any]], definitions: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    order = [phase["name"] for phase in reports[0]["plan"]["phases"]]
    result: list[dict[str, Any]] = []
    for name in order:
        definition = definitions[name]
        phases = [_phase_by_name(report, name) for report in reports]
        energy = [
            _phase_metric(phase, "gross_energy_joules", context=f"phase {name!r}")
            for phase in phases
        ]
        duration = [
            _phase_metric(phase, "duration_seconds", context=f"phase {name!r}")
            for phase in phases
        ]
        peak_power = [
            _phase_metric(phase, "sampled_peak_power_watts", context=f"phase {name!r}")
            for phase in phases
        ]
        peak_temperature = [
            _phase_metric(phase, "peak_temperature_c", context=f"phase {name!r}")
            for phase in phases
        ]
        units = definition["workload_units"]
        frame_count = units.get("frames")
        megapixels = (
            units["image_pixels"] / 1_000_000.0
            if "image_pixels" in units
            else None
        )
        result.append(
            {
                "name": name,
                "description": definition.get("description"),
                "trial_count": len(reports),
                "workload_units": units,
                "gross_energy_joules": _stats(energy),
                "duration_seconds": _stats(duration),
                "sampled_peak_power_watts": _stats(peak_power),
                "peak_temperature_c": _stats(peak_temperature),
                "maximum_observed_temperature_c": (
                    max(value for value in peak_temperature if value is not None)
                    if any(value is not None for value in peak_temperature)
                    else None
                ),
                "gross_energy_joules_per_frame": _stats(
                    [_divide(value, frame_count) for value in energy]
                ),
                "gross_energy_joules_per_megapixel": _stats(
                    [_divide(value, megapixels) for value in energy]
                ),
            }
        )
    return result


def _derived_number(
    report: Mapping[str, Any], metric_name: str, section: str, field: str
) -> float | None:
    metric = report["derived_metrics"].get(metric_name)
    if not isinstance(metric, Mapping):
        raise ValueError(f"Derived metric {metric_name!r} must be an object")
    if metric.get("status") != "available":
        raise ValueError(
            f"Derived metric {metric_name!r} is not available in "
            f"{report.get('_source_path')}"
        )
    payload = metric.get(section)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Derived metric {metric_name!r}.{section} is missing")
    return _optional_number(payload.get(field), f"{metric_name}.{section}.{field}")


def _aggregate_derived(
    reports: Sequence[Mapping[str, Any]], definitions: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fields = (
        "gross_energy_joules",
        "duration_seconds",
        "average_power_watts",
        "gross_energy_joules_per_target_frame",
        "duration_seconds_per_target_frame",
    )
    for name, definition in definitions.items():
        sections: dict[str, Any] = {}
        for section in ("exact_hybrid_phase_sum", "projected_total"):
            sections[section] = {
                field: _stats(
                    [
                        _derived_number(report, name, section, field)
                        for report in reports
                    ]
                )
                for field in fields
            }
            target_units = definition.get("target_units")
            if definition.get("unit") == "image_pixels" and target_units is not None:
                divisor = _finite_number(
                    target_units, f"derived metric {name!r}.target_units"
                ) / 1_000_000.0
                energies = [
                    _derived_number(report, name, section, "gross_energy_joules")
                    for report in reports
                ]
                sections[section]["gross_energy_joules_per_target_megapixel"] = _stats(
                    [_divide(value, divisor) for value in energies]
                )
        measured = [report["derived_metrics"][name] for report in reports]
        result[name] = {
            "definition": definition,
            **sections,
            "quality_warnings": _unique_text(
                warning
                for metric in measured
                for warning in metric.get("quality_warnings", [])
            ),
            "scope": _unique_text(metric.get("scope") for metric in measured),
        }
    return result


def _caveats(
    reports: Sequence[Mapping[str, Any]],
    phases: Sequence[Mapping[str, Any]],
    derived: Mapping[str, Any],
) -> dict[str, Any]:
    plan_notes = _unique_text(report["plan"].get("notes") for report in reports)
    level3 = next((phase for phase in phases if phase["name"] == "level3"), None)
    description = level3.get("description") if level3 else None
    is_passthrough = bool(
        isinstance(description, str)
        and ("pass-through" in description.lower() or "passthrough" in description.lower())
    )
    return {
        "plan_notes": plan_notes,
        "level3": {
            "present": level3 is not None,
            "benchmark_only_passthrough": is_passthrough,
            "description": description,
            "interpretation": (
                "The Level 3 measurement covers the present pass-through, validation, "
                "and file-I/O work only; it is not an estimate of future Level 3 "
                "rotation, geometry, or special dark-correction algorithms."
                if is_passthrough
                else None
            ),
        },
        "paired_projection_models": [
            {
                "name": name,
                "description": payload["definition"].get("description"),
                "quality_warnings": payload["quality_warnings"],
                "scope": payload["scope"],
            }
            for name, payload in derived.items()
        ],
    }


def aggregate_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate and aggregate already-loaded measurement reports."""

    if not reports:
        raise ValueError("At least one measurement is required")
    compatibility = _validate_compatible(reports)
    phases = _aggregate_phases(reports, compatibility["phase_definitions"])
    derived = _aggregate_derived(reports, compatibility["derived_definitions"])
    return {
        "schema": _SUMMARY_SCHEMA,
        "schema_version": 1,
        "created_utc": _utc_now(),
        "trial_count": len(reports),
        "input_measurements": [report.get("_source_path") for report in reports],
        "plan_labels": _unique_text(report["plan"].get("label") for report in reports),
        "telemetry_interval_ms": compatibility["telemetry_interval_ms"],
        "compatibility": {
            "status": "compatible",
            "basis": (
                "Exact phase-name set, normalized phase commands/environments/path "
                "assertions, per-phase descriptions/workload units, pre-staged input "
                "declarations, plan notes, paired-projection definitions, and "
                "telemetry sampling interval"
            ),
        },
        "phases": phases,
        "paired_projections": derived,
        "caveats": _caveats(reports, phases, derived),
    }


def _format_number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _format_stats(stats: Mapping[str, Any], digits: int = 3) -> str:
    if stats.get("median") is None:
        return "—"
    return (
        f"{_format_number(stats['median'], digits)} "
        f"[{_format_number(stats['minimum'], digits)}–"
        f"{_format_number(stats['maximum'], digits)}]"
    )


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# SunCET end-to-end Jetson benchmark trial summary",
        "",
        f"Trials: **{summary['trial_count']}**. Values are median [minimum–maximum].",
        "",
        "## Per-phase measurements",
        "",
        "| Phase | Gross energy (J) | Duration (s) | Sampled peak power (W) | Peak temperature (°C) | J/frame | J/MP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in summary["phases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    phase["name"],
                    _format_stats(phase["gross_energy_joules"]),
                    _format_stats(phase["duration_seconds"]),
                    _format_stats(phase["sampled_peak_power_watts"]),
                    _format_stats(phase["peak_temperature_c"], 1),
                    _format_stats(phase["gross_energy_joules_per_frame"]),
                    _format_stats(phase["gross_energy_joules_per_megapixel"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "`MP` means one million processed image pixels. A dash means the phase "
            "did not declare the corresponding workload unit.",
            "",
            "## Hybrid totals and paired projection",
            "",
            "| Model | Total | Gross energy (J) | Duration (s) | Average power (W) | J/target frame | J/target MP |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, metric in summary["paired_projections"].items():
        for section, label in (
            ("exact_hybrid_phase_sum", "Exact hybrid phase sum"),
            ("projected_total", "Projected target workload"),
        ):
            values = metric[section]
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        label,
                        _format_stats(values["gross_energy_joules"]),
                        _format_stats(values["duration_seconds"]),
                        _format_stats(values["average_power_watts"]),
                        _format_stats(values["gross_energy_joules_per_target_frame"]),
                        _format_stats(
                            values.get("gross_energy_joules_per_target_megapixel", {})
                        ),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Scope and caveats", ""])
    level3 = summary["caveats"]["level3"]
    if level3["description"]:
        lines.append(f"- **Level 3:** {level3['description']}")
    if level3["interpretation"]:
        lines.append(f"  {level3['interpretation']}")
    for note in summary["caveats"]["plan_notes"]:
        lines.append(f"- **Plan note:** {note}")
    for model in summary["caveats"]["paired_projection_models"]:
        if model["description"]:
            lines.append(f"- **Projection `{model['name']}`:** {model['description']}")
        for warning in model["quality_warnings"]:
            lines.append(f"  - {warning}")
        for scope in model["scope"]:
            lines.append(f"  - Scope: {scope}")
    lines.append("")
    return "\n".join(lines)


def _csv_rows(summary: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for phase in summary["phases"]:
        yield {
            "kind": "phase",
            "name": phase["name"],
            "description": phase.get("description"),
            "trial_count": phase["trial_count"],
            "workload_units_json": json.dumps(phase["workload_units"], sort_keys=True),
            **_wide_stats("gross_energy_joules", phase["gross_energy_joules"]),
            **_wide_stats("duration_seconds", phase["duration_seconds"]),
            **_wide_stats("sampled_peak_power_watts", phase["sampled_peak_power_watts"]),
            **_wide_stats("peak_temperature_c", phase["peak_temperature_c"]),
            **_wide_stats(
                "gross_energy_joules_per_frame",
                phase["gross_energy_joules_per_frame"],
            ),
            **_wide_stats(
                "gross_energy_joules_per_megapixel",
                phase["gross_energy_joules_per_megapixel"],
            ),
        }
    for name, metric in summary["paired_projections"].items():
        for section in ("exact_hybrid_phase_sum", "projected_total"):
            values = metric[section]
            yield {
                "kind": section,
                "name": name,
                "description": metric["definition"].get("description"),
                "trial_count": summary["trial_count"],
                "workload_units_json": json.dumps(
                    {
                        "target_frame_count": metric["definition"].get("target_frame_count"),
                        metric["definition"].get("unit"): metric["definition"].get("target_units"),
                    },
                    sort_keys=True,
                ),
                **_wide_stats("gross_energy_joules", values["gross_energy_joules"]),
                **_wide_stats("duration_seconds", values["duration_seconds"]),
                **_wide_stats("average_power_watts", values["average_power_watts"]),
                **_wide_stats(
                    "gross_energy_joules_per_target_frame",
                    values["gross_energy_joules_per_target_frame"],
                ),
                **_wide_stats(
                    "gross_energy_joules_per_target_megapixel",
                    values.get("gross_energy_joules_per_target_megapixel", _stats([])),
                ),
            }


def _wide_stats(prefix: str, stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_{suffix}": stats.get(key)
        for suffix, key in (
            ("count", "count"),
            ("median", "median"),
            ("minimum", "minimum"),
            ("maximum", "maximum"),
            ("range", "range"),
        )
    }


def _csv_content(summary: Mapping[str, Any]) -> str:
    rows = list(_csv_rows(summary))
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _write_outputs(output_directory: Path, summary: Mapping[str, Any], overwrite: bool) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = [output_directory / name for name in _OUTPUT_NAMES]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing summary output(s): "
            + ", ".join(str(path) for path in existing)
        )
    payloads = (
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        _csv_content(summary),
        _markdown(summary),
    )
    for path, payload in zip(paths, payloads):
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        paths = [_measurement_path(path) for path in arguments.measurements]
        if len(paths) != len(set(paths)):
            raise ValueError("Each measurement may be specified only once")
        reports = [_read_measurement(path) for path in paths]
        summary = aggregate_reports(reports)
        _write_outputs(
            arguments.output_directory.expanduser().resolve(),
            summary,
            arguments.overwrite,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"Aggregated {summary['trial_count']} trial(s) into "
        f"{arguments.output_directory.expanduser().resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

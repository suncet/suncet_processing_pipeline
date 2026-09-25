"""Focused tests for repeated end-to-end trial aggregation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import aggregate_trials  # noqa: E402


def _report(trial: int, *, reverse_phases: bool = False) -> dict:
    phase_definitions = [
        {
            "name": "level1",
            "description": "Create science products",
            "workload_units": {"frames": 2, "image_pixels": 2_000_000},
        },
        {
            "name": "level3",
            "description": "Benchmark-only Level 3 pass-through; corrections unavailable",
            "workload_units": {"frames": 2, "image_pixels": 2_000_000},
        },
    ]
    phases = [
        {
            "name": "level1",
            "status": "passed",
            "workload_units": {"frames": 2, "image_pixels": 2_000_000},
            "power": {
                "gross_energy_joules": 6.0 + 4.0 * trial,
                "duration_seconds": 1.0 + trial,
                "sampled_peak_power_watts": 3.0 + 2.0 * trial,
                "peak_temperature_c": 38.0 + 2.0 * trial,
            },
        },
        {
            "name": "level3",
            "status": "passed",
            "workload_units": {"frames": 2, "image_pixels": 2_000_000},
            "power": {
                "gross_energy_joules": 2.0 + 2.0 * trial,
                "duration_seconds": 0.5 + 0.5 * trial,
                "sampled_peak_power_watts": 2.0 + trial,
                "peak_temperature_c": 37.0 + trial,
            },
        },
    ]
    if reverse_phases:
        phase_definitions.reverse()
        phases.reverse()
    definition = {
        "name": "hybrid",
        "type": "paired_delta_projection",
        "description": "Scale marginal image work only",
        "baseline_phase": "level1",
        "full_phase": "level3",
        "unit": "image_pixels",
        "observed_units": 1_000_000,
        "target_units": 2_000_000,
        "target_frame_count": 2,
        "additional_phases": [],
    }
    exact_energy = 18.0 + 4.0 * trial
    projected_energy = 28.0 + 4.0 * trial
    exact_duration = 3.0 + trial
    projected_duration = 4.0 + trial
    return {
        "schema": "suncet.end_to_end_power_measurement",
        "schema_version": 1,
        "status": "passed",
        "telemetry_interval_ms": 100,
        "plan": {
            "schema": "suncet.end_to_end_plan",
            "schema_version": 1,
            "label": "fixture",
            "notes": (
                "Synthetic arrays are binned. Level 3 is a benchmark-only "
                "pass-through because corrections do not apply."
            ),
            "phases": phase_definitions,
            "derived_metrics": [definition],
        },
        "phases": phases,
        "derived_metrics": {
            "hybrid": {
                **definition,
                "status": "available",
                "exact_hybrid_phase_sum": {
                    "gross_energy_joules": exact_energy,
                    "duration_seconds": exact_duration,
                    "average_power_watts": exact_energy / exact_duration,
                    "gross_energy_joules_per_target_frame": exact_energy / 2,
                    "duration_seconds_per_target_frame": exact_duration / 2,
                },
                "projected_total": {
                    "gross_energy_joules": projected_energy,
                    "duration_seconds": projected_duration,
                    "average_power_watts": projected_energy / projected_duration,
                    "gross_energy_joules_per_target_frame": projected_energy / 2,
                    "duration_seconds_per_target_frame": projected_duration / 2,
                },
                "quality_warnings": ["Linear image-pixel scaling assumption"],
                "scope": "Phase-only model; boot and shutdown excluded.",
            }
        },
    }


class AggregateTrialsTests(unittest.TestCase):
    def test_aggregates_phase_ranges_and_separate_hybrid_totals(self):
        first = _report(1)
        second = _report(2, reverse_phases=True)
        first["_source_path"] = "/trials/one/measurement.json"
        second["_source_path"] = "/trials/two/measurement.json"

        summary = aggregate_trials.aggregate_reports([first, second])

        self.assertEqual(summary["trial_count"], 2)
        level1 = next(phase for phase in summary["phases"] if phase["name"] == "level1")
        self.assertEqual(level1["gross_energy_joules"]["median"], 12.0)
        self.assertEqual(level1["gross_energy_joules"]["range"], 4.0)
        self.assertEqual(level1["gross_energy_joules_per_frame"]["median"], 6.0)
        self.assertEqual(
            level1["gross_energy_joules_per_megapixel"]["median"], 6.0
        )
        self.assertEqual(level1["maximum_observed_temperature_c"], 42.0)

        hybrid = summary["paired_projections"]["hybrid"]
        self.assertEqual(
            hybrid["exact_hybrid_phase_sum"]["gross_energy_joules"]["median"],
            24.0,
        )
        self.assertEqual(
            hybrid["projected_total"]["gross_energy_joules"]["median"], 34.0
        )
        self.assertTrue(
            summary["caveats"]["level3"]["benchmark_only_passthrough"]
        )

    def test_rejects_incompatible_phase_workload(self):
        first = _report(1)
        second = _report(2)
        second["plan"]["phases"][0]["workload_units"]["frames"] = 3
        second["phases"][0]["workload_units"]["frames"] = 3

        with self.assertRaisesRegex(ValueError, "incompatible phase definitions"):
            aggregate_trials.aggregate_reports([first, second])

    def test_cli_writes_json_csv_and_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run1 = root / "run1"
            run2 = root / "run2"
            run1.mkdir()
            run2.mkdir()
            (run1 / "measurement.json").write_text(
                json.dumps(_report(1)), encoding="utf-8"
            )
            (run2 / "measurement.json").write_text(
                json.dumps(_report(2)), encoding="utf-8"
            )
            output = root / "summary"

            returncode = aggregate_trials.main(
                [str(run1), str(run2 / "measurement.json"), "--output-directory", str(output)]
            )

            self.assertEqual(returncode, 0)
            summary = json.loads((output / "trial_summary.json").read_text())
            self.assertEqual(summary["schema"], "suncet.end_to_end_trial_summary")
            with (output / "trial_summary.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                [row["kind"] for row in rows],
                ["phase", "phase", "exact_hybrid_phase_sum", "projected_total"],
            )
            markdown = (output / "trial_summary.md").read_text()
            self.assertIn("Exact hybrid phase sum", markdown)
            self.assertIn("Projected target workload", markdown)
            self.assertIn("Benchmark-only Level 3 pass-through", markdown)


if __name__ == "__main__":
    unittest.main()

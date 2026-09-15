"""Tests for the Level 2 ownership layer around shared Jetson telemetry."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

from power_telemetry import (  # noqa: E402
    parse_tegrastats_line,
    summarize_interval,
)


LIVE_R39_LINE = (
    "09-15-2026 12:19:33 RAM 999/62841MB CPU [0%@729,off] "
    "GR3D_FREQ 0% cpu@43.093C/43.093C soc2@39.625C/39.687C "
    "soc0@40.031C/40.093C tj@42.906C/42.968C "
    "soc1@40.125C/40.156C VDD_GPU_SOC 2792mW/2792mW/2792mW "
    "VDD_CPU_CV 0mW/0mW/0mW VIN_SYS_5V0 3918mW/3952mW/4019mW"
)


def _sample(timestamp_ns: int, power_watts: float, temperature_c: float = 40.0):
    power_mw = power_watts * 1000.0
    return {
        "monotonic_ns": timestamp_ns,
        "timestamp_utc": "2026-09-15T00:00:00Z",
        "raw_line": "fixture",
        "temperatures_c": {"tj": temperature_c},
        "rails_mw": {
            "VDD_GPU_SOC": {"current_mw": power_mw, "average_mw": power_mw},
            "VDD_CPU_CV": {"current_mw": 0.0, "average_mw": 0.0},
            "VIN_SYS_5V0": {"current_mw": 0.0, "average_mw": 0.0},
        },
    }


class PowerTelemetryTests(unittest.TestCase):
    def test_shared_parser_uses_current_values_and_exact_total_rails(self):
        parsed = parse_tegrastats_line(
            LIVE_R39_LINE,
            monotonic_ns=1,
            timestamp_utc="2026-09-15T00:00:00Z",
        )

        self.assertAlmostEqual(parsed["covered_onboard_power_mw"], 6710.0)
        self.assertAlmostEqual(parsed["rails_mw"]["VIN_SYS_5V0"]["current_mw"], 3918.0)
        self.assertAlmostEqual(parsed["temperatures_c"]["cpu"], 43.093)
        self.assertAlmostEqual(parsed["temperatures_c"]["tj"], 42.906)

    def test_incomplete_rail_set_does_not_invent_input_power(self):
        parsed = parse_tegrastats_line(
            "VDD_GPU_SOC 1000mW/1000mW VIN_SYS_5V0 3000mW/3000mW",
            monotonic_ns=1,
            timestamp_utc="2026-09-15T00:00:00Z",
        )
        self.assertIsNone(parsed["covered_onboard_power_mw"])

    def test_vddq_is_not_double_counted_in_r39_total(self):
        parsed = parse_tegrastats_line(
            "VDD_GPU_SOC 1000mW/1000mW VDD_CPU_CV 2000mW/2000mW "
            "VIN_SYS_5V0 3000mW/3000mW VDDQ_VDD2_1V8AO 900mW/900mW",
            monotonic_ns=1,
            timestamp_utc="2026-09-15T00:00:00Z",
        )
        self.assertAlmostEqual(parsed["covered_onboard_power_mw"], 6000.0)
        self.assertAlmostEqual(parsed["rails_mw"]["VDDQ_VDD2_1V8AO"]["current_mw"], 900.0)

    def test_interval_integration_interpolates_exact_boundaries(self):
        samples = [
            _sample(0, 10.0, 40.0),
            _sample(1_000_000_000, 20.0, 42.0),
            _sample(2_000_000_000, 30.0, 41.0),
        ]
        result = summarize_interval(samples, 500_000_000, 1_500_000_000)

        self.assertTrue(result["fully_power_covered"])
        self.assertAlmostEqual(result["power_coverage_seconds"], 1.0)
        self.assertAlmostEqual(result["gross_energy_joules"], 20.0)
        self.assertAlmostEqual(result["observed_average_power_watts"], 20.0)
        self.assertAlmostEqual(result["sampled_peak_power_watts"], 25.0)
        self.assertAlmostEqual(result["peak_temperature_c"], 42.0)
        self.assertEqual(result["peak_temperature_sensor"], "tj")

    def test_partial_interval_never_labels_observed_energy_as_gross(self):
        result = summarize_interval(
            [_sample(1_000_000_000, 10.0), _sample(2_000_000_000, 10.0)],
            0,
            2_000_000_000,
        )

        self.assertFalse(result["fully_power_covered"])
        self.assertAlmostEqual(result["power_coverage_fraction"], 0.5)
        self.assertAlmostEqual(result["observed_energy_joules"], 10.0)
        self.assertIsNone(result["gross_energy_joules"])


if __name__ == "__main__":
    unittest.main()

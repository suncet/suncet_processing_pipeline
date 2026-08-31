import json

import pytest

from suncet_processing_pipeline.level4.jetson_metrics import (
    COVERED_ONBOARD_RAILS,
    covered_onboard_power_mw,
    integrate_gross_energy_joules,
    integrate_gross_energy_window_joules,
    parse_tegrastats_line,
    select_samples_in_window,
    summarize_tegrastats_samples,
    summarize_tegrastats_window,
)


R39_AGX_ORIN_LINE = (
    "RAM 5530/62841MB (lfb 13106x4MB) "
    "SWAP 16/31420MB (cached 8MB) "
    "CPU [3%@729,off,12%@1190,0%@729] "
    "EMC_FREQ 7%@2133 GR3D_FREQ 18%@[612,611] "
    "NVENC off NVDEC off VIC_FREQ 0%@115 "
    "cpu@42.312C soc2@40.5C tj@43C "
    "VDD_GPU_SOC 3123mW/3000mW "
    "VDD_CPU_CV 1456mW/1300mW "
    "VIN_SYS_5V0 2876mW/2800mW "
    "VDDQ_VDD2_1V8AO 343mW/340mW"
)


def _sample(
    monotonic_ns: int,
    *,
    gpu_mw: float,
    cpu_mw: float = 0.0,
    system_mw: float = 0.0,
    gr3d_percent: float = 10.0,
    ram_used_mb: float = 100.0,
    temperature_c: float = 40.0,
):
    return {
        "monotonic_ns": monotonic_ns,
        "rails_mw": {
            "VDD_GPU_SOC": {"current_mw": gpu_mw, "average_mw": gpu_mw},
            "VDD_CPU_CV": {"current_mw": cpu_mw, "average_mw": cpu_mw},
            "VIN_SYS_5V0": {"current_mw": system_mw, "average_mw": system_mw},
            "VDDQ_VDD2_1V8AO": {"current_mw": 99999.0, "average_mw": 99999.0},
        },
        "ram": {"used_mb": ram_used_mb, "total_mb": 1000.0},
        "gr3d": {
            "utilization_percent": gr3d_percent,
            "frequency_mhz": 612.0,
            "frequencies_mhz": [612.0],
        },
        "cpu": [
            {
                "core": 0,
                "online": True,
                "utilization_percent": 50.0,
                "frequency_mhz": 729.0,
                "raw": "50%@729",
            },
            {
                "core": 1,
                "online": False,
                "utilization_percent": None,
                "frequency_mhz": None,
                "raw": "off",
            },
        ],
        "temperatures_c": {"cpu": temperature_c},
    }


def test_parse_current_r39_agx_orin_line_and_preserve_raw_text():
    raw = R39_AGX_ORIN_LINE + "\n"
    sample = parse_tegrastats_line(
        raw,
        monotonic_ns=123456789,
        timestamp_utc="2026-08-26T15:00:00.123456Z",
    )

    assert sample["raw_line"] == raw
    assert sample["monotonic_ns"] == 123456789
    assert sample["timestamp_utc"] == "2026-08-26T15:00:00.123456Z"
    assert sample["ram"] == {
        "used_mb": 5530.0,
        "total_mb": 62841.0,
        "largest_free_block_count": 13106,
        "largest_free_block_mb": 4.0,
    }
    assert sample["swap"] == {
        "used_mb": 16.0,
        "total_mb": 31420.0,
        "cached_mb": 8.0,
    }
    assert sample["cpu"] == [
        {
            "core": 0,
            "online": True,
            "utilization_percent": 3.0,
            "frequency_mhz": 729.0,
            "raw": "3%@729",
        },
        {
            "core": 1,
            "online": False,
            "utilization_percent": None,
            "frequency_mhz": None,
            "raw": "off",
        },
        {
            "core": 2,
            "online": True,
            "utilization_percent": 12.0,
            "frequency_mhz": 1190.0,
            "raw": "12%@1190",
        },
        {
            "core": 3,
            "online": True,
            "utilization_percent": 0.0,
            "frequency_mhz": 729.0,
            "raw": "0%@729",
        },
    ]
    assert sample["emc"] == {
        "utilization_percent": 7.0,
        "frequency_mhz": 2133.0,
        "frequencies_mhz": [2133.0],
    }
    assert sample["gr3d"] == {
        "utilization_percent": 18.0,
        "frequency_mhz": None,
        "frequencies_mhz": [612.0, 611.0],
    }
    assert sample["temperatures_c"] == {
        "cpu": 42.312,
        "soc2": 40.5,
        "tj": 43.0,
    }
    assert sample["rails_mw"]["VDD_GPU_SOC"] == {
        "current_mw": 3123.0,
        "average_mw": 3000.0,
    }
    assert sample["covered_onboard_power_mw"] == 7455.0
    assert COVERED_ONBOARD_RAILS == (
        "VDD_GPU_SOC",
        "VDD_CPU_CV",
        "VIN_SYS_5V0",
    )
    json.dumps(sample, allow_nan=False)


def test_parse_missing_and_off_fields_without_inventing_values():
    sample = parse_tegrastats_line(
        "RAM 100/1000MB CPU [off,50%@1020] GR3D_FREQ 25%",
        monotonic_ns=10,
        timestamp_utc="2026-08-26T15:00:00Z",
    )

    assert sample["swap"] is None
    assert sample["emc"] is None
    assert sample["gr3d"] == {
        "utilization_percent": 25.0,
        "frequency_mhz": None,
        "frequencies_mhz": [],
    }
    assert sample["cpu"][0]["online"] is False
    assert sample["cpu"][0]["utilization_percent"] is None
    assert sample["temperatures_c"] == {}
    assert sample["rails_mw"] == {}
    assert sample["covered_onboard_power_mw"] is None
    assert covered_onboard_power_mw(sample) is None
    json.dumps(sample, allow_nan=False)


def test_parser_rejects_process_errors_as_non_telemetry():
    with pytest.raises(ValueError, match="recognized tegrastats telemetry"):
        parse_tegrastats_line(
            "tegrastats: permission denied",
            monotonic_ns=10,
            timestamp_utc="2026-08-26T15:00:00Z",
        )


def test_parse_frequency_only_accelerator_forms_documented_for_r39():
    sample = parse_tegrastats_line(
        "GR3D_FREQ @[1098,1098] EMC_FREQ @2133",
        monotonic_ns=11,
        timestamp_utc="2026-08-26T15:00:01Z",
    )

    assert sample["gr3d"] == {
        "utilization_percent": None,
        "frequency_mhz": None,
        "frequencies_mhz": [1098.0, 1098.0],
    }
    assert sample["emc"] == {
        "utilization_percent": None,
        "frequency_mhz": 2133.0,
        "frequencies_mhz": [2133.0],
    }


def test_covered_power_requires_all_three_current_rails_and_ignores_vddq():
    sample = _sample(0, gpu_mw=1000.0, cpu_mw=2000.0, system_mw=3000.0)
    assert covered_onboard_power_mw(sample) == 6000.0

    del sample["rails_mw"]["VDD_CPU_CV"]
    assert covered_onboard_power_mw(sample) is None


def test_select_window_is_inclusive_and_sorted():
    samples = [_sample(3, gpu_mw=3), _sample(1, gpu_mw=1), _sample(2, gpu_mw=2)]
    selected = select_samples_in_window(
        samples,
        start_monotonic_ns=2,
        end_monotonic_ns=3,
    )
    assert [sample["monotonic_ns"] for sample in selected] == [2, 3]


def test_trapezoidal_gross_energy_uses_monotonic_time_and_milliwatts():
    # 0--1 s: mean 1.5 W => 1.5 J; 1--3 s: mean 2.5 W => 5 J.
    samples = [
        _sample(0, gpu_mw=1000.0),
        _sample(1_000_000_000, gpu_mw=2000.0),
        _sample(3_000_000_000, gpu_mw=3000.0),
    ]
    assert integrate_gross_energy_joules(samples) == pytest.approx(6.5)

    window = select_samples_in_window(
        samples,
        start_monotonic_ns=1_000_000_000,
        end_monotonic_ns=3_000_000_000,
    )
    assert integrate_gross_energy_joules(window) == pytest.approx(5.0)


def test_energy_does_not_bridge_an_incomplete_power_sample():
    samples = [
        _sample(0, gpu_mw=1000.0),
        {"monotonic_ns": 1_000_000_000, "rails_mw": {}},
        _sample(2_000_000_000, gpu_mw=1000.0),
    ]
    assert integrate_gross_energy_joules(samples) is None


def test_exact_window_energy_uses_bracketing_samples_and_clipped_bounds():
    # Linear ramp from 1 to 3 W over 2 s. The exact 0.5--1.5 s interval has a
    # mean power of 2 W even though neither boundary is a sample timestamp.
    samples = [
        _sample(0, gpu_mw=1000.0),
        _sample(1_000_000_000, gpu_mw=2000.0),
        _sample(2_000_000_000, gpu_mw=3000.0),
    ]

    energy = integrate_gross_energy_window_joules(
        samples,
        start_monotonic_ns=500_000_000,
        end_monotonic_ns=1_500_000_000,
    )
    summary = summarize_tegrastats_window(
        samples,
        start_monotonic_ns=500_000_000,
        end_monotonic_ns=1_500_000_000,
    )

    assert energy == pytest.approx(2.0)
    assert summary["gross_energy_joules"] == pytest.approx(2.0)
    assert summary["power_coverage_seconds"] == 1.0
    assert summary["power_coverage_fraction"] == 1.0
    assert summary["mean_covered_onboard_power_mw"] == pytest.approx(2000.0)
    assert summary["peak_covered_onboard_power_mw"] == pytest.approx(2500.0)


def test_exact_window_reports_partial_coverage_without_negative_idle_bias():
    samples = [
        _sample(200_000_000, gpu_mw=1000.0),
        _sample(800_000_000, gpu_mw=1000.0),
    ]
    summary = summarize_tegrastats_window(
        samples,
        start_monotonic_ns=0,
        end_monotonic_ns=1_000_000_000,
    )

    assert summary["gross_energy_joules"] == pytest.approx(0.6)
    assert summary["power_coverage_seconds"] == pytest.approx(0.6)
    assert summary["power_coverage_fraction"] == pytest.approx(0.6)


def test_summary_reports_power_temperature_ram_and_utilization():
    samples = [
        _sample(
            0,
            gpu_mw=1000.0,
            gr3d_percent=10.0,
            ram_used_mb=100.0,
            temperature_c=40.0,
        ),
        _sample(
            2_000_000_000,
            gpu_mw=3000.0,
            gr3d_percent=30.0,
            ram_used_mb=250.0,
            temperature_c=45.0,
        ),
    ]
    summary = summarize_tegrastats_samples(samples)

    assert summary["sample_count"] == 2
    assert summary["duration_seconds"] == 2.0
    assert summary["power_coverage_seconds"] == 2.0
    assert summary["gross_energy_joules"] == pytest.approx(4.0)
    assert summary["mean_covered_onboard_power_mw"] == pytest.approx(2000.0)
    assert summary["peak_covered_onboard_power_mw"] == 3000.0
    assert summary["maximum_temperature_c"] == 45.0
    assert summary["maximum_temperatures_c"] == {"cpu": 45.0}
    assert summary["maximum_ram_used_mb"] == 250.0
    assert summary["maximum_ram_used_percent"] == 25.0
    assert summary["mean_gr3d_utilization_percent"] == 20.0
    assert summary["peak_gr3d_utilization_percent"] == 30.0
    # Each sample has one 50%-busy core and one off core: full-capacity mean 25%.
    assert summary["mean_cpu_utilization_percent"] == 25.0
    assert summary["peak_sample_mean_cpu_utilization_percent"] == 25.0
    assert summary["peak_cpu_core_utilization_percent"] == 50.0
    json.dumps(summary, allow_nan=False)


def test_empty_summary_and_single_power_sample_are_json_safe():
    empty = summarize_tegrastats_samples([])
    assert empty["sample_count"] == 0
    assert empty["gross_energy_joules"] is None
    assert empty["mean_covered_onboard_power_mw"] is None
    assert empty["maximum_temperature_c"] is None
    json.dumps(empty, allow_nan=False)

    single = summarize_tegrastats_samples([_sample(5, gpu_mw=1234.0)])
    assert single["gross_energy_joules"] is None
    assert single["mean_covered_onboard_power_mw"] == 1234.0


@pytest.mark.parametrize(
    "arguments",
    [
        {"raw_line": "", "monotonic_ns": -1, "timestamp_utc": "x"},
        {"raw_line": "", "monotonic_ns": True, "timestamp_utc": "x"},
        {"raw_line": "", "monotonic_ns": 1, "timestamp_utc": ""},
    ],
)
def test_parser_rejects_invalid_external_timestamps(arguments):
    with pytest.raises(ValueError):
        parse_tegrastats_line(**arguments)

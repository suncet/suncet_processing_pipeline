"""Parse and reduce NVIDIA Jetson ``tegrastats`` telemetry.

The parser intentionally accepts timestamps captured by the supervising
process.  ``tegrastats`` output does not provide the monotonic timestamp needed
to align power samples with an exact benchmark window, so callers should stamp
each line immediately after reading it from the subprocess pipe.

All public functions return ordinary Python dictionaries, lists, numbers,
strings, booleans, and ``None`` so their results can be written directly as
JSON.  Missing telemetry remains missing; the module never substitutes zero
for an absent rail, temperature, or accelerator field.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any


COVERED_ONBOARD_RAILS = (
    "VDD_GPU_SOC",
    "VDD_CPU_CV",
    "VIN_SYS_5V0",
)
"""AGX Orin rails used for the comparative onboard-power estimate.

``VDDQ_VDD2_1V8AO`` is deliberately absent because NVIDIA documents it as
already included in ``VIN_SYS_5V0``.
"""


_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)"
_RAM_RE = re.compile(
    rf"\bRAM\s+(?P<used>{_NUMBER})/(?P<total>{_NUMBER})MB"
    rf"(?:\s+\(lfb\s+(?P<blocks>\d+)x(?P<block_mb>{_NUMBER})MB\))?",
    re.IGNORECASE,
)
_SWAP_RE = re.compile(
    rf"\bSWAP\s+(?P<used>{_NUMBER})/(?P<total>{_NUMBER})MB"
    rf"(?:\s+\(cached\s+(?P<cached>{_NUMBER})MB\))?",
    re.IGNORECASE,
)
_CPU_RE = re.compile(r"\bCPU\s*\[(?P<cores>[^]]*)\]", re.IGNORECASE)
_CPU_ONLINE_RE = re.compile(
    rf"^(?P<util>{_NUMBER})%(?:@(?P<frequency>{_NUMBER}))?$",
    re.IGNORECASE,
)
_TEMPERATURE_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<name>[A-Za-z][A-Za-z0-9_]*)"
    rf"@(?P<value>-?{_NUMBER})C\b",
    re.IGNORECASE,
)
_RAIL_RE = re.compile(
    rf"\b(?P<name>[A-Za-z][A-Za-z0-9_]*)\s+"
    rf"(?P<current>{_NUMBER})\s*mW\s*/\s*"
    rf"(?P<average>{_NUMBER})\s*mW\b",
    re.IGNORECASE,
)


def _accelerator_pattern(name: str) -> re.Pattern[str]:
    return re.compile(
        rf"\b{re.escape(name)}\s+"
        rf"(?P<body>(?:{_NUMBER}%(?:@(?:{_NUMBER}|\[[^]]+\]))?"
        rf"|@(?:{_NUMBER}|\[[^]]+\])))",
        re.IGNORECASE,
    )


_EMC_RE = _accelerator_pattern("EMC_FREQ")
_GR3D_RE = _accelerator_pattern("GR3D_FREQ")


def _float(value: str | None) -> float | None:
    return float(value) if value is not None else None


def _memory_values(match: re.Match[str] | None, *, swap: bool) -> dict[str, Any] | None:
    if match is None:
        return None
    values: dict[str, Any] = {
        "used_mb": float(match.group("used")),
        "total_mb": float(match.group("total")),
    }
    if swap:
        values["cached_mb"] = _float(match.group("cached"))
    else:
        values["largest_free_block_count"] = (
            int(match.group("blocks")) if match.group("blocks") else None
        )
        values["largest_free_block_mb"] = _float(match.group("block_mb"))
    return values


def _cpu_values(line: str) -> list[dict[str, Any]]:
    match = _CPU_RE.search(line)
    if match is None:
        return []

    cores: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(match.group("cores").split(",")):
        entry = raw_entry.strip()
        if not entry:
            continue
        if entry.lower() == "off":
            cores.append(
                {
                    "core": index,
                    "online": False,
                    "utilization_percent": None,
                    "frequency_mhz": None,
                    "raw": entry,
                }
            )
            continue
        online = _CPU_ONLINE_RE.fullmatch(entry)
        cores.append(
            {
                "core": index,
                "online": True,
                "utilization_percent": (
                    float(online.group("util")) if online is not None else None
                ),
                "frequency_mhz": (
                    _float(online.group("frequency"))
                    if online is not None
                    else None
                ),
                "raw": entry,
            }
        )
    return cores


def _accelerator_values(
    match: re.Match[str] | None,
) -> dict[str, Any] | None:
    if match is None:
        return None
    body = match.group("body")
    utilization: float | None = None
    frequency_text: str | None = None
    if body.startswith("@"):
        frequency_text = body[1:]
    else:
        utilization_text, separator, frequency_text = body.partition("%@")
        if not separator:
            utilization_text = body.removesuffix("%")
            frequency_text = None
        utilization = float(utilization_text)

    scalar: float | None = None
    frequencies: list[float] = []
    if frequency_text is not None and frequency_text.startswith("["):
        vector_text = frequency_text.removeprefix("[").removesuffix("]")
        for item in vector_text.split(","):
            item = item.strip()
            if re.fullmatch(_NUMBER, item):
                frequencies.append(float(item))
    elif frequency_text is not None and re.fullmatch(_NUMBER, frequency_text):
        scalar = float(frequency_text)
        frequencies.append(scalar)
    return {
        "utilization_percent": utilization,
        "frequency_mhz": scalar,
        "frequencies_mhz": frequencies,
    }


def parse_tegrastats_line(
    raw_line: str,
    *,
    monotonic_ns: int,
    timestamp_utc: str,
) -> dict[str, Any]:
    """Parse one Jetson Linux r39-style ``tegrastats`` line.

    Unknown tokens are retained only in ``raw_line``.  Recognized fields that
    are absent are represented by ``None`` or an empty collection, allowing a
    single schema to cover different stock power modes and idle/off engines.
    """

    if not isinstance(raw_line, str):
        raise TypeError("raw_line must be a string.")
    if (
        isinstance(monotonic_ns, bool)
        or not isinstance(monotonic_ns, int)
        or monotonic_ns < 0
    ):
        raise ValueError("monotonic_ns must be a nonnegative integer.")
    if not isinstance(timestamp_utc, str) or not timestamp_utc.strip():
        raise ValueError("timestamp_utc must be a non-empty string.")

    line = raw_line.strip()
    temperatures = {
        match.group("name").lower(): float(match.group("value"))
        for match in _TEMPERATURE_RE.finditer(line)
    }
    rails = {
        match.group("name").upper(): {
            "current_mw": float(match.group("current")),
            "average_mw": float(match.group("average")),
        }
        for match in _RAIL_RE.finditer(line)
    }
    sample: dict[str, Any] = {
        "schema": "suncet.jetson_tegrastats_sample",
        "schema_version": 1,
        "monotonic_ns": monotonic_ns,
        "timestamp_utc": timestamp_utc,
        "raw_line": raw_line,
        "ram": _memory_values(_RAM_RE.search(line), swap=False),
        "swap": _memory_values(_SWAP_RE.search(line), swap=True),
        "cpu": _cpu_values(line),
        "emc": _accelerator_values(_EMC_RE.search(line)),
        "gr3d": _accelerator_values(_GR3D_RE.search(line)),
        "temperatures_c": temperatures,
        "rails_mw": rails,
    }
    if (
        sample["ram"] is None
        and sample["swap"] is None
        and not sample["cpu"]
        and sample["emc"] is None
        and sample["gr3d"] is None
        and not sample["temperatures_c"]
        and not sample["rails_mw"]
    ):
        raise ValueError("Line does not contain recognized tegrastats telemetry.")
    sample["covered_onboard_power_mw"] = covered_onboard_power_mw(sample)
    return sample


def _finite_nonnegative(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0 else None


def covered_onboard_power_mw(sample: Mapping[str, Any]) -> float | None:
    """Return the documented three-rail current-power sum for one sample.

    All three current readings are required.  Returning ``None`` for a partial
    sample prevents an absent rail from being silently treated as zero.
    """

    rails = sample.get("rails_mw")
    if not isinstance(rails, Mapping):
        return None
    readings: list[float] = []
    for name in COVERED_ONBOARD_RAILS:
        pair = rails.get(name)
        if not isinstance(pair, Mapping):
            return None
        current = _finite_nonnegative(pair.get("current_mw"))
        if current is None:
            return None
        readings.append(current)
    return float(sum(readings))


def _sample_monotonic_ns(sample: Mapping[str, Any]) -> int:
    value = sample.get("monotonic_ns")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Every sample needs a nonnegative integer monotonic_ns.")
    return value


def select_samples_in_window(
    samples: Sequence[Mapping[str, Any]],
    *,
    start_monotonic_ns: int,
    end_monotonic_ns: int,
) -> list[dict[str, Any]]:
    """Return samples inside the inclusive monotonic interval, time-sorted."""

    if (
        isinstance(start_monotonic_ns, bool)
        or isinstance(end_monotonic_ns, bool)
        or not isinstance(start_monotonic_ns, int)
        or not isinstance(end_monotonic_ns, int)
        or start_monotonic_ns < 0
        or end_monotonic_ns < 0
        or end_monotonic_ns < start_monotonic_ns
    ):
        raise ValueError("The monotonic window must be nonnegative and ordered.")
    selected = [
        dict(sample)
        for sample in samples
        if start_monotonic_ns
        <= _sample_monotonic_ns(sample)
        <= end_monotonic_ns
    ]
    selected.sort(key=_sample_monotonic_ns)
    return selected


def _ordered_samples(
    samples: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    ordered = list(samples)
    for sample in ordered:
        _sample_monotonic_ns(sample)
    ordered.sort(key=_sample_monotonic_ns)
    return ordered


def _energy_and_covered_duration(
    samples: Sequence[Mapping[str, Any]],
) -> tuple[float | None, float]:
    ordered = _ordered_samples(samples)
    energy_joules = 0.0
    covered_seconds = 0.0
    for first, second in zip(ordered, ordered[1:]):
        first_power = covered_onboard_power_mw(first)
        second_power = covered_onboard_power_mw(second)
        delta_ns = _sample_monotonic_ns(second) - _sample_monotonic_ns(first)
        if first_power is None or second_power is None or delta_ns <= 0:
            continue
        delta_seconds = delta_ns / 1_000_000_000.0
        energy_joules += (
            0.5 * (first_power + second_power) / 1000.0 * delta_seconds
        )
        covered_seconds += delta_seconds
    if covered_seconds == 0.0:
        return None, 0.0
    return float(energy_joules), float(covered_seconds)


def integrate_gross_energy_joules(
    samples: Sequence[Mapping[str, Any]],
) -> float | None:
    """Trapezoidally integrate covered current power over adjacent samples.

    An interval contributes only when both endpoint samples contain all three
    covered rails.  Duplicate/non-increasing timestamps and incomplete
    intervals are ignored.  Fewer than two usable samples returns ``None``.
    """

    energy, _duration = _energy_and_covered_duration(samples)
    return energy


def _window_energy_coverage_and_power_values(
    samples: Sequence[Mapping[str, Any]],
    *,
    start_monotonic_ns: int,
    end_monotonic_ns: int,
) -> tuple[float | None, float, list[float]]:
    # Reuse the public validator without requiring a sample to lie inside the
    # window. Exact-bound integration needs the adjacent samples outside it.
    select_samples_in_window(
        (),
        start_monotonic_ns=start_monotonic_ns,
        end_monotonic_ns=end_monotonic_ns,
    )
    ordered = _ordered_samples(samples)
    energy_joules = 0.0
    covered_seconds = 0.0
    power_values: list[float] = []
    for first, second in zip(ordered, ordered[1:]):
        first_ns = _sample_monotonic_ns(first)
        second_ns = _sample_monotonic_ns(second)
        overlap_start = max(first_ns, start_monotonic_ns)
        overlap_end = min(second_ns, end_monotonic_ns)
        if overlap_end <= overlap_start or second_ns <= first_ns:
            continue
        first_power = covered_onboard_power_mw(first)
        second_power = covered_onboard_power_mw(second)
        if first_power is None or second_power is None:
            continue
        interval_ns = second_ns - first_ns

        def interpolate(at_ns: int) -> float:
            fraction = (at_ns - first_ns) / interval_ns
            return first_power + fraction * (second_power - first_power)

        start_power = interpolate(overlap_start)
        end_power = interpolate(overlap_end)
        duration_seconds = (overlap_end - overlap_start) / 1_000_000_000.0
        energy_joules += (
            0.5 * (start_power + end_power) / 1000.0 * duration_seconds
        )
        covered_seconds += duration_seconds
        power_values.extend((start_power, end_power))
    if covered_seconds == 0:
        return None, 0.0, []
    return float(energy_joules), float(covered_seconds), power_values


def integrate_gross_energy_window_joules(
    samples: Sequence[Mapping[str, Any]],
    *,
    start_monotonic_ns: int,
    end_monotonic_ns: int,
) -> float | None:
    """Integrate power over exact workload bounds using bracketing samples.

    Each overlapping telemetry interval is clipped to the requested monotonic
    bounds and its power is linearly interpolated at those boundaries. Missing
    rails leave the affected interval uncovered rather than substituting zero.
    """

    energy, _coverage, _powers = _window_energy_coverage_and_power_values(
        samples,
        start_monotonic_ns=start_monotonic_ns,
        end_monotonic_ns=end_monotonic_ns,
    )
    return energy


def _finite_values(values: Sequence[object]) -> list[float]:
    output: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        converted = float(value)
        if math.isfinite(converted):
            output.append(converted)
    return output


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _maximum(values: Sequence[float]) -> float | None:
    return float(max(values)) if values else None


def summarize_tegrastats_samples(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize one already-windowed sequence of parsed samples."""

    ordered = _ordered_samples(samples)
    powers = _finite_values(
        [covered_onboard_power_mw(sample) for sample in ordered]
    )
    energy, covered_seconds = _energy_and_covered_duration(ordered)
    mean_power = (
        energy / covered_seconds * 1000.0
        if energy is not None and covered_seconds > 0
        else _mean(powers)
    )

    maximum_temperatures: dict[str, float] = {}
    for sample in ordered:
        temperatures = sample.get("temperatures_c")
        if not isinstance(temperatures, Mapping):
            continue
        for name, raw_value in temperatures.items():
            values = _finite_values([raw_value])
            if values:
                normalized = str(name).lower()
                maximum_temperatures[normalized] = max(
                    maximum_temperatures.get(normalized, -math.inf), values[0]
                )

    ram_used: list[float] = []
    ram_percent: list[float] = []
    gr3d_utilization: list[float] = []
    cpu_sample_means: list[float] = []
    cpu_core_utilization: list[float] = []
    for sample in ordered:
        ram = sample.get("ram")
        if isinstance(ram, Mapping):
            used = _finite_nonnegative(ram.get("used_mb"))
            total = _finite_nonnegative(ram.get("total_mb"))
            if used is not None:
                ram_used.append(used)
            if used is not None and total is not None and total > 0:
                ram_percent.append(100.0 * used / total)

        gr3d = sample.get("gr3d")
        if isinstance(gr3d, Mapping):
            gr3d_utilization.extend(
                _finite_values([gr3d.get("utilization_percent")])
            )

        cores = sample.get("cpu")
        if not isinstance(cores, Sequence) or isinstance(cores, (str, bytes)):
            continue
        sample_core_values: list[float] = []
        for core in cores:
            if not isinstance(core, Mapping):
                continue
            if core.get("online") is False:
                # An off core contributes zero to utilization of the full CPU
                # capacity while remaining explicitly off in the raw sample.
                sample_core_values.append(0.0)
                cpu_core_utilization.append(0.0)
                continue
            values = _finite_values([core.get("utilization_percent")])
            if values:
                sample_core_values.append(values[0])
                cpu_core_utilization.append(values[0])
        if sample_core_values:
            cpu_sample_means.append(
                float(sum(sample_core_values) / len(sample_core_values))
            )

    all_temperatures = list(maximum_temperatures.values())
    duration_seconds = (
        (_sample_monotonic_ns(ordered[-1]) - _sample_monotonic_ns(ordered[0]))
        / 1_000_000_000.0
        if len(ordered) >= 2
        else 0.0
    )
    return {
        "schema": "suncet.jetson_tegrastats_summary",
        "schema_version": 1,
        "sample_count": len(ordered),
        "power_sample_count": len(powers),
        "start_monotonic_ns": (
            _sample_monotonic_ns(ordered[0]) if ordered else None
        ),
        "end_monotonic_ns": (
            _sample_monotonic_ns(ordered[-1]) if ordered else None
        ),
        "duration_seconds": float(duration_seconds),
        "power_coverage_seconds": covered_seconds,
        "gross_energy_joules": energy,
        "mean_covered_onboard_power_mw": mean_power,
        "peak_covered_onboard_power_mw": _maximum(powers),
        "maximum_temperature_c": _maximum(all_temperatures),
        "maximum_temperatures_c": maximum_temperatures,
        "maximum_ram_used_mb": _maximum(ram_used),
        "maximum_ram_used_percent": _maximum(ram_percent),
        "mean_gr3d_utilization_percent": _mean(gr3d_utilization),
        "peak_gr3d_utilization_percent": _maximum(gr3d_utilization),
        "mean_cpu_utilization_percent": _mean(cpu_sample_means),
        "peak_sample_mean_cpu_utilization_percent": _maximum(cpu_sample_means),
        "peak_cpu_core_utilization_percent": _maximum(cpu_core_utilization),
    }


def summarize_tegrastats_window(
    samples: Sequence[Mapping[str, Any]],
    *,
    start_monotonic_ns: int,
    end_monotonic_ns: int,
) -> dict[str, Any]:
    """Summarize telemetry and integrate power over exact workload bounds."""

    selected = select_samples_in_window(
        samples,
        start_monotonic_ns=start_monotonic_ns,
        end_monotonic_ns=end_monotonic_ns,
    )
    summary = summarize_tegrastats_samples(selected)
    energy, covered_seconds, boundary_powers = (
        _window_energy_coverage_and_power_values(
            samples,
            start_monotonic_ns=start_monotonic_ns,
            end_monotonic_ns=end_monotonic_ns,
        )
    )
    requested_seconds = (end_monotonic_ns - start_monotonic_ns) / 1_000_000_000.0
    sample_powers = _finite_values(
        [covered_onboard_power_mw(sample) for sample in selected]
    )
    summary.update(
        {
            "requested_start_monotonic_ns": start_monotonic_ns,
            "requested_end_monotonic_ns": end_monotonic_ns,
            "requested_duration_seconds": requested_seconds,
            "power_coverage_seconds": covered_seconds,
            "power_coverage_fraction": (
                covered_seconds / requested_seconds
                if requested_seconds > 0
                else None
            ),
            "gross_energy_joules": energy,
            "mean_covered_onboard_power_mw": (
                energy / covered_seconds * 1000.0
                if energy is not None and covered_seconds > 0
                else None
            ),
            "peak_covered_onboard_power_mw": _maximum(
                [*sample_powers, *boundary_powers]
            ),
        }
    )
    return summary


__all__ = [
    "COVERED_ONBOARD_RAILS",
    "covered_onboard_power_mw",
    "integrate_gross_energy_joules",
    "integrate_gross_energy_window_joules",
    "parse_tegrastats_line",
    "select_samples_in_window",
    "summarize_tegrastats_samples",
    "summarize_tegrastats_window",
]

"""Level 2 ownership layer for the shared Jetson telemetry reducer.

Parsing and exact-boundary integration live in
``suncet_processing_pipeline.level4.jetson_metrics`` and are reused here so
Level 2 and Level 4 cannot silently assign different meanings to the same
Jetson rails.  This module owns only the ``tegrastats`` subprocess and applies
the stricter Level 2 reporting rule that gross energy is unavailable unless
the requested interval is completely covered.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any, Iterable

from suncet_processing_pipeline.level4.jetson_metrics import (
    COVERED_ONBOARD_RAILS,
    covered_onboard_power_mw,
    parse_tegrastats_line,
    summarize_tegrastats_window,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def summarize_interval(
    samples: Iterable[dict[str, Any]],
    start_monotonic_ns: int,
    end_monotonic_ns: int,
) -> dict[str, Any]:
    """Return a conservative SI-unit summary for one exact workload window."""
    source = list(samples)
    shared = summarize_tegrastats_window(
        source,
        start_monotonic_ns=start_monotonic_ns,
        end_monotonic_ns=end_monotonic_ns,
    )
    requested_seconds = (end_monotonic_ns - start_monotonic_ns) / 1e9
    covered_seconds = float(shared.get("power_coverage_seconds") or 0.0)
    coverage_fraction = (
        covered_seconds / requested_seconds if requested_seconds > 0 else 1.0
    )
    # Nanosecond rounding can create a tiny representational difference, but no
    # missing telemetry interval may be called fully covered.
    fully_covered = abs(covered_seconds - requested_seconds) <= 1e-9
    observed_energy = shared.get("gross_energy_joules")
    mean_power_mw = shared.get("mean_covered_onboard_power_mw")
    peak_power_mw = shared.get("peak_covered_onboard_power_mw")
    temperatures = dict(shared.get("maximum_temperatures_c") or {})
    principal_sensor = "tj" if "tj" in temperatures else None
    if principal_sensor is None and temperatures:
        principal_sensor = max(temperatures, key=temperatures.get)
    hottest_sensor = max(temperatures, key=temperatures.get) if temperatures else None
    selected_count = int(shared.get("sample_count") or 0)
    return {
        "duration_seconds": requested_seconds,
        "input_power_source": "derived:sum_of_instantaneous_r39_module_rails",
        "input_power_components": list(COVERED_ONBOARD_RAILS),
        "power_sample_count_inside_interval": selected_count,
        "power_coverage_seconds": covered_seconds,
        "power_coverage_fraction": coverage_fraction,
        "fully_power_covered": fully_covered,
        "observed_energy_joules": observed_energy,
        "gross_energy_joules": observed_energy if fully_covered else None,
        "observed_average_power_watts": (
            float(mean_power_mw) / 1000.0 if mean_power_mw is not None else None
        ),
        "sampled_peak_power_watts": (
            float(peak_power_mw) / 1000.0 if peak_power_mw is not None else None
        ),
        "peak_temperature_c": (
            temperatures.get(principal_sensor) if principal_sensor else None
        ),
        "peak_temperature_sensor": principal_sensor,
        "hottest_temperature_any_sensor_c": (
            temperatures.get(hottest_sensor) if hottest_sensor else None
        ),
        "hottest_temperature_any_sensor": hottest_sensor,
        "peak_temperatures_by_sensor_c": temperatures,
        "temperature_sample_count_inside_interval": selected_count,
        "shared_reducer_summary": shared,
    }


def _resolve_executable(explicit_path: str | None) -> str | None:
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        return str(candidate) if candidate.is_file() else None
    discovered = shutil.which("tegrastats")
    if discovered:
        return discovered
    for candidate in (Path("/usr/bin/tegrastats"), Path("/usr/sbin/tegrastats")):
        if candidate.is_file():
            return str(candidate)
    return None


class TegrastatsSampler:
    """Timestamp and retain lines from one owned, unprivileged process."""

    def __init__(
        self,
        *,
        interval_ms: int = 100,
        executable: str | None = None,
        mode: str = "auto",
    ) -> None:
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        if mode not in {"auto", "required", "off"}:
            raise ValueError("mode must be 'auto', 'required', or 'off'")
        self.interval_ms = int(interval_ms)
        self.mode = mode
        self.executable = None if mode == "off" else _resolve_executable(executable)
        self.status = "disabled" if mode == "off" else "not_started"
        self.unavailable_reason: str | None = None
        self._process: subprocess.Popen[str] | None = None
        self._samples: list[dict[str, Any]] = []
        self._unparsed_lines: list[dict[str, Any]] = []
        self._stderr_lines: list[str] = []
        self._condition = threading.Condition()
        self._threads: list[threading.Thread] = []

    @property
    def samples(self) -> list[dict[str, Any]]:
        with self._condition:
            return list(self._samples)

    def start(self) -> None:
        if self.mode == "off":
            return
        if self.executable is None:
            self.status = "unavailable"
            self.unavailable_reason = "tegrastats executable was not found"
            if self.mode == "required":
                raise RuntimeError(self.unavailable_reason)
            return
        try:
            self._process = subprocess.Popen(
                [self.executable, "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self.status = "unavailable"
            self.unavailable_reason = f"Could not start tegrastats: {exc}"
            if self.mode == "required":
                raise RuntimeError(self.unavailable_reason) from exc
            return
        self.status = "collecting"
        self._threads = [
            threading.Thread(target=self._read_stdout, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for raw in self._process.stdout:
            line = raw.rstrip("\r\n")
            monotonic_ns = time.monotonic_ns()
            timestamp_utc = _utc_now()
            try:
                sample = parse_tegrastats_line(
                    line,
                    monotonic_ns=monotonic_ns,
                    timestamp_utc=timestamp_utc,
                )
            except (TypeError, ValueError) as exc:
                with self._condition:
                    self._unparsed_lines.append(
                        {
                            "monotonic_ns": monotonic_ns,
                            "timestamp_utc": timestamp_utc,
                            "raw_line": line,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    self._condition.notify_all()
            else:
                with self._condition:
                    self._samples.append(sample)
                    self._condition.notify_all()

    def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for raw in self._process.stderr:
            with self._condition:
                self._stderr_lines.append(raw.rstrip("\r\n"))
                self._condition.notify_all()

    def wait_for_power_sample(
        self,
        *,
        after_monotonic_ns: int | None = None,
        timeout_seconds: float = 3.0,
    ) -> bool:
        if self.status != "collecting":
            return False
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while True:
                if any(
                    covered_onboard_power_mw(sample) is not None
                    and (
                        after_monotonic_ns is None
                        or int(sample["monotonic_ns"]) >= after_monotonic_ns
                    )
                    for sample in self._samples
                ):
                    return True
                if self._process is not None and self._process.poll() is not None:
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
        for thread in self._threads:
            thread.join(timeout=1.0)
        self.status = "completed" if self._samples else "completed_without_samples"
        if not any(
            covered_onboard_power_mw(sample) is not None
            for sample in self._samples
        ):
            self.unavailable_reason = (
                "tegrastats produced no complete R39 three-rail power sample"
            )

    def report(self, *, origin_monotonic_ns: int) -> dict[str, Any]:
        with self._condition:
            samples = list(self._samples)
            unparsed = list(self._unparsed_lines)
            stderr = list(self._stderr_lines)
        return {
            "status": self.status,
            "mode": self.mode,
            "executable": self.executable,
            "interval_ms": self.interval_ms,
            "unavailable_reason": self.unavailable_reason,
            "stderr_lines": stderr,
            "unparsed_lines": unparsed,
            "samples": [
                {
                    **sample,
                    "offset_seconds": (
                        int(sample["monotonic_ns"]) - origin_monotonic_ns
                    ) / 1e9,
                }
                for sample in samples
            ],
        }


def run_read_only_diagnostic(
    executable_name: str,
    arguments: list[str],
    *,
    fallback_paths: tuple[str, ...] = (),
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Run a bounded read-only platform query and retain its exact output."""
    executable = shutil.which(executable_name)
    if executable is None:
        executable = next(
            (path for path in fallback_paths if Path(path).is_file()), None
        )
    if executable is None:
        return {
            "status": "unavailable",
            "command": [executable_name, *arguments],
            "reason": "executable not found",
        }
    command = [executable, *arguments]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "failed",
            "command": command,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        "status": "completed" if completed.returncode == 0 else "failed",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.rstrip(),
        "stderr": completed.stderr.rstrip(),
    }

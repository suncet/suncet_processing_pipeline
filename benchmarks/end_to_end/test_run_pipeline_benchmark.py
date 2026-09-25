"""Focused tests for the staged end-to-end benchmark supervisor."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_pipeline_benchmark  # noqa: E402


def _write_plan(
    path: Path,
    source: Path,
    phases: list[dict],
    derived_metrics: list[dict] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "suncet.end_to_end_plan",
                "schema_version": 1,
                "label": "unit-test",
                "pre_staged_paths": [
                    {"name": "source", "path": str(source), "kind": "file"}
                ],
                "phases": phases,
                "derived_metrics": derived_metrics or [],
            }
        ),
        encoding="utf-8",
    )


class EndToEndSupervisorTests(unittest.TestCase):
    def setUp(self):
        # Unit tests exercise boundary/provenance behavior without flushing the
        # developer workstation's real filesystems.
        patcher = mock.patch.object(
            run_pipeline_benchmark.os,
            "sync",
            create=True,
        )
        self.sync_mock = patcher.start()
        self.addCleanup(patcher.stop)

    def test_paired_delta_projection_preserves_fixed_baseline(self):
        phases = [
            {
                "name": "level0_5_skip_images",
                "power": {"gross_energy_joules": 10.0, "duration_seconds": 2.0},
            },
            {
                "name": "level0_5_full",
                "power": {"gross_energy_joules": 14.0, "duration_seconds": 3.0},
            },
            {
                "name": "level1",
                "power": {"gross_energy_joules": 20.0, "duration_seconds": 4.0},
            },
        ]
        metrics = run_pipeline_benchmark._paired_delta_metrics(
            phases,
            [
                {
                    "name": "hour",
                    "type": "paired_delta_projection",
                    "description": None,
                    "baseline_phase": "level0_5_skip_images",
                    "full_phase": "level0_5_full",
                    "unit": "image_pixels",
                    "observed_units": 100.0,
                    "target_units": 250.0,
                    "target_frame_count": 5.0,
                    "additional_phases": ["level1"],
                }
            ],
        )

        metric = metrics["hour"]
        self.assertEqual(metric["variable_delta"]["gross_energy_joules"], 4.0)
        self.assertEqual(
            metric["projected_variable_phase"]["gross_energy_joules"], 20.0
        )
        self.assertEqual(metric["projected_total"]["gross_energy_joules"], 40.0)
        self.assertEqual(
            metric["projected_total"]["gross_energy_joules_per_target_frame"],
            8.0,
        )
        self.assertEqual(
            metric["exact_hybrid_phase_sum"]["gross_energy_joules"], 34.0
        )

    def test_git_record_hashes_untracked_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "Test"], check=True
            )
            tracked = root / "tracked.txt"
            tracked.write_text("tracked\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-q", "-m", "fixture"],
                check=True,
            )
            untracked = root / "new.txt"
            untracked.write_text("untracked\n", encoding="utf-8")

            record = run_pipeline_benchmark._git_record(root)

            self.assertEqual(record["status"], "available")
            self.assertEqual(record["untracked_paths"][0]["path"], "new.txt")
            self.assertEqual(record["untracked_paths"][0]["status"], "hashed")
            self.assertEqual(len(record["untracked_paths"][0]["sha256"]), 64)

    def test_continuous_fake_telemetry_covers_phases_total_and_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("input", encoding="utf-8")
            fake_tegrastats = root / "tegrastats"
            fake_tegrastats.write_text(
                "#!/usr/bin/env python3\n"
                "import time\n"
                "while True:\n"
                "    print('RAM 1/10MB tj@40.0C VDD_GPU_SOC 1000mW/1000mW '"
                "          'VDD_CPU_CV 2000mW/2000mW VIN_SYS_5V0 3000mW/3000mW',"
                "          flush=True)\n"
                "    time.sleep(0.01)\n",
                encoding="utf-8",
            )
            fake_tegrastats.chmod(0o755)
            plan = root / "plan.json"
            run_dir = root / "run"
            _write_plan(
                plan,
                source,
                [
                    {
                        "name": "level0_5",
                        "command": [
                            sys.executable,
                            "-c",
                            "import time; time.sleep(0.15)",
                        ],
                        "workload_units": {"frames": 4},
                    },
                    {
                        "name": "level1",
                        "command": [
                            sys.executable,
                            "-c",
                            "import time; time.sleep(0.15)",
                        ],
                        "workload_units": {"frames": 8},
                    },
                ],
            )

            with redirect_stderr(io.StringIO()):
                exit_code = run_pipeline_benchmark.main(
                    [
                        "--plan",
                        str(plan),
                        "--run-dir",
                        str(run_dir),
                        "--telemetry",
                        "required",
                        "--tegrastats-path",
                        str(fake_tegrastats),
                        "--telemetry-interval-ms",
                        "20",
                    ]
                )

            self.assertEqual(exit_code, 0)
            report = json.loads((run_dir / "measurement.json").read_text())
            self.assertTrue(report["pipeline_total"]["power"]["fully_power_covered"])
            self.assertAlmostEqual(
                report["pipeline_total"]["power"]["observed_average_power_watts"],
                6.0,
                places=6,
            )
            self.assertTrue(
                all(
                    phase["power"]["gross_energy_joules"] is not None
                    for phase in report["phases"]
                )
            )
            self.assertAlmostEqual(
                report["energy_reconciliation"][
                    "numerical_reconciliation_error_joules"
                ],
                0.0,
                places=8,
            )

    def test_two_stages_share_paths_without_supervisor_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("science\n", encoding="utf-8")
            plan = root / "plan.json"
            run_dir = root / "run"
            phase1_program = (
                "from pathlib import Path; "
                "Path(r'${RUN_DIR}/handoff.txt').write_text("
                "Path(r'" + str(source) + "').read_text())"
            )
            phase2_program = (
                "from pathlib import Path; "
                "p=Path(r'${RUN_DIR}/handoff.txt'); "
                "Path(r'${RUN_DIR}/final.txt').write_text(p.read_text().upper()); "
                "print('second stdout')"
            )
            _write_plan(
                plan,
                source,
                [
                    {
                        "name": "level0_5",
                        "command": [sys.executable, "-c", phase1_program],
                        "workload_units": {"frames": 4},
                        "expected_outputs": [
                            {"path": "${RUN_DIR}/handoff.txt", "kind": "file"}
                        ],
                    },
                    {
                        "name": "level1",
                        "command": [sys.executable, "-c", phase2_program],
                        "required_paths": [
                            {"path": "${RUN_DIR}/handoff.txt", "kind": "file"}
                        ],
                        "workload_units": {"frames": 8},
                        "expected_outputs": [
                            {"path": "${RUN_DIR}/final.txt", "kind": "file"}
                        ],
                    },
                ],
            )

            with redirect_stderr(io.StringIO()):
                exit_code = run_pipeline_benchmark.main(
                    [
                        "--plan",
                        str(plan),
                        "--run-dir",
                        str(run_dir),
                        "--telemetry",
                        "off",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual((run_dir / "final.txt").read_text(), "SCIENCE\n")
            report = json.loads((run_dir / "measurement.json").read_text())
            self.assertEqual(report["status"], "passed")
            self.assertEqual([p["name"] for p in report["phases"]], ["level0_5", "level1"])
            self.assertEqual(
                report["phases"][0]["normalizations"]["per_unit"]["frames"][
                    "observed_count"
                ],
                4.0,
            )
            self.assertEqual(
                [gap["name"] for gap in report["interphase_gaps"]],
                ["before_first_phase", "after_level0_5", "after_last_phase"],
            )
            self.assertIn("second stdout", (run_dir / "logs/level1.stdout.log").read_text())
            self.assertTrue((run_dir / "phase_summary.csv").is_file())
            self.assertTrue((run_dir / "telemetry_samples.csv").is_file())
            self.assertEqual(self.sync_mock.call_count, 2)
            self.assertTrue(
                all(
                    phase["filesystem_flush"]["status"] == "completed"
                    for phase in report["phases"]
                )
            )

    def test_phase_boundary_includes_flush_after_child_logs_are_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("input", encoding="utf-8")
            plan = root / "plan.json"
            run_dir = root / "run"
            _write_plan(
                plan,
                source,
                [
                    {
                        "name": "only",
                        "command": [
                            sys.executable,
                            "-c",
                            "print('child output is complete', flush=True)",
                        ],
                    }
                ],
            )

            observed_at_flush: list[str] = []

            def fake_sync():
                observed_at_flush.append(
                    (run_dir / "logs/only.stdout.log").read_text(encoding="utf-8")
                )
                time.sleep(0.01)

            self.sync_mock.side_effect = fake_sync
            with redirect_stderr(io.StringIO()):
                exit_code = run_pipeline_benchmark.main(
                    [
                        "--plan",
                        str(plan),
                        "--run-dir",
                        str(run_dir),
                        "--telemetry",
                        "off",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(observed_at_flush, ["child output is complete\n"])
            report = json.loads((run_dir / "measurement.json").read_text())
            phase = report["phases"][0]
            flush = phase["filesystem_flush"]
            self.assertEqual(flush["status"], "completed")
            self.assertTrue(flush["included_in_phase_boundary"])
            self.assertEqual(
                flush["monotonic_bounds_ns"]["end"],
                phase["monotonic_bounds_ns"]["end"],
            )
            self.assertGreaterEqual(
                flush["monotonic_bounds_ns"]["start"],
                phase["monotonic_bounds_ns"]["start"],
            )
            self.assertGreaterEqual(flush["duration_seconds"], 0.01)

    def test_flush_failure_is_reported_and_stops_the_pipeline_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("input", encoding="utf-8")
            plan = root / "plan.json"
            run_dir = root / "run"
            fake_tegrastats = root / "tegrastats"
            fake_tegrastats.write_text(
                "#!/usr/bin/env python3\n"
                "import time\n"
                "while True:\n"
                "    print('tj@40.0C VDD_GPU_SOC 1000mW/1000mW '"
                "          'VDD_CPU_CV 2000mW/2000mW VIN_SYS_5V0 3000mW/3000mW',"
                "          flush=True)\n"
                "    time.sleep(0.01)\n",
                encoding="utf-8",
            )
            fake_tegrastats.chmod(0o755)
            _write_plan(
                plan,
                source,
                [
                    {
                        "name": "only",
                        "command": [sys.executable, "-c", "print('completed')"],
                    },
                    {
                        "name": "must_not_run",
                        "command": [
                            sys.executable,
                            "-c",
                            "from pathlib import Path; Path(r'${RUN_DIR}/bad').touch()",
                        ],
                    },
                ],
            )
            self.sync_mock.side_effect = OSError("simulated writeback failure")

            with redirect_stderr(io.StringIO()):
                exit_code = run_pipeline_benchmark.main(
                    [
                        "--plan",
                        str(plan),
                        "--run-dir",
                        str(run_dir),
                        "--telemetry",
                        "required",
                        "--tegrastats-path",
                        str(fake_tegrastats),
                        "--telemetry-interval-ms",
                        "20",
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertFalse((run_dir / "bad").exists())
            report = json.loads((run_dir / "measurement.json").read_text())
            self.assertEqual(report["status"], "phase_failed")
            self.assertEqual(report["telemetry"]["status"], "completed")
            self.assertGreaterEqual(len(report["telemetry"]["samples"]), 2)
            self.assertEqual(len(report["phases"]), 1)
            phase = report["phases"][0]
            self.assertEqual(phase["status"], "filesystem_flush_failed")
            self.assertEqual(phase["returncode"], 0)
            self.assertEqual(phase["filesystem_flush"]["status"], "failed")
            self.assertIn("simulated writeback failure", phase["error"])

    def test_failed_phase_stops_following_phase_and_preserves_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("input", encoding="utf-8")
            plan = root / "plan.json"
            run_dir = root / "run"
            _write_plan(
                plan,
                source,
                [
                    {
                        "name": "failure",
                        "command": [sys.executable, "-c", "raise SystemExit(7)"],
                        "workload_units": {"frames": 4},
                    },
                    {
                        "name": "must_not_run",
                        "command": [
                            sys.executable,
                            "-c",
                            "from pathlib import Path; Path(r'${RUN_DIR}/bad').touch()",
                        ],
                        "workload_units": {"frames": 8},
                    },
                ],
            )

            with redirect_stderr(io.StringIO()):
                exit_code = run_pipeline_benchmark.main(
                    [
                        "--plan",
                        str(plan),
                        "--run-dir",
                        str(run_dir),
                        "--telemetry",
                        "off",
                    ]
                )

            self.assertEqual(exit_code, 7)
            self.assertFalse((run_dir / "bad").exists())
            report = json.loads((run_dir / "measurement.json").read_text())
            self.assertEqual(report["status"], "phase_failed")
            self.assertEqual(len(report["phases"]), 1)
            self.assertEqual(report["phases"][0]["returncode"], 7)
            self.assertEqual(
                report["phases"][0]["filesystem_flush"]["status"],
                "completed",
            )
            self.sync_mock.assert_called_once_with()

    def test_dry_run_expands_placeholders_without_creating_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("input", encoding="utf-8")
            plan = root / "plan.json"
            run_dir = root / "future-run"
            _write_plan(
                plan,
                source,
                [
                    {
                        "name": "only",
                        "command": [sys.executable, "-c", "print(r'${RUN_DIR}')"],
                        "workload_units": {"frames": 8},
                    }
                ],
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                exit_code = run_pipeline_benchmark.main(
                    [
                        "--plan",
                        str(plan),
                        "--run-dir",
                        str(run_dir),
                        "--dry-run",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertFalse(run_dir.exists())
            expanded = json.loads(stdout.getvalue())
            self.assertIn(str(run_dir), expanded["phases"][0]["command"][-1])

    def test_selected_phases_default_to_plan_order_but_can_use_requested_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("input", encoding="utf-8")
            plan = root / "plan.json"
            _write_plan(
                plan,
                source,
                [
                    {"name": "first", "command": [sys.executable, "-c", "pass"]},
                    {"name": "second", "command": [sys.executable, "-c", "pass"]},
                    {"name": "third", "command": [sys.executable, "-c", "pass"]},
                ],
            )

            def dry_run(extra: list[str]) -> list[str]:
                stdout = io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                    exit_code = run_pipeline_benchmark.main(
                        [
                            "--plan",
                            str(plan),
                            "--run-dir",
                            str(root / "future-run"),
                            "--phase",
                            "second",
                            "--phase",
                            "first",
                            "--dry-run",
                            *extra,
                        ]
                    )
                self.assertEqual(exit_code, 0)
                expanded = json.loads(stdout.getvalue())
                return [phase["name"] for phase in expanded["phases"]]

            self.assertEqual(dry_run([]), ["first", "second"])
            self.assertEqual(
                dry_run(["--phase-order", "requested"]),
                ["second", "first"],
            )


if __name__ == "__main__":
    unittest.main()

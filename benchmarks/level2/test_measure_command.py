"""Focused tests for the external-command Jetson power wrapper."""

from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import measure_command  # noqa: E402


class MeasureCommandTests(unittest.TestCase):
    def test_requires_explicit_command_separator(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            output = Path(directory) / "result.json"
            exit_code = measure_command.main(
                [
                    "--telemetry",
                    "off",
                    "--output-json",
                    str(output),
                    sys.executable,
                    "-c",
                    "print('must not run')",
                ]
            )

            self.assertEqual(exit_code, 2)
            self.assertFalse(output.exists())

    def test_captures_streams_and_preserves_nonzero_child_status(self):
        program = (
            "import sys; "
            "print('captured stdout'); "
            "print('captured stderr', file=sys.stderr); "
            "raise SystemExit(7)"
        )
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            output = Path(directory) / "result.json"
            exit_code = measure_command.main(
                [
                    "--telemetry",
                    "off",
                    "--output-json",
                    str(output),
                    "--",
                    sys.executable,
                    "-c",
                    program,
                ]
            )

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 7)
            self.assertEqual(report["status"], "command_failed")
            self.assertEqual(report["command_returncode"], 7)
            self.assertEqual(report["command_stdout"], "captured stdout\n")
            self.assertEqual(report["command_stderr"], "captured stderr\n")
            self.assertLess(
                report["command_monotonic_bounds_ns"]["start"],
                report["command_monotonic_bounds_ns"]["end"],
            )
            self.assertLessEqual(
                report["command_started_utc"], report["command_finished_utc"]
            )
            self.assertIsNone(report["command_power"]["gross_energy_joules"])
            self.assertTrue(report["command_power"]["quality_warnings"])
            self.assertIn("child spawn/exec", report["scope"]["included"])
            self.assertIn("system boot", report["scope"]["excluded"])

    def test_refuses_existing_result_before_running_child(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            root = Path(directory)
            output = root / "result.json"
            sentinel = root / "child-ran"
            output.write_text("original\n", encoding="utf-8")
            program = (
                "from pathlib import Path; "
                f"Path({str(sentinel)!r}).write_text('ran', encoding='utf-8')"
            )

            exit_code = measure_command.main(
                [
                    "--telemetry",
                    "off",
                    "--output-json",
                    str(output),
                    "--",
                    sys.executable,
                    "-c",
                    program,
                ]
            )

            self.assertEqual(exit_code, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "original\n")
            self.assertFalse(sentinel.exists())

    def test_successful_command_records_exact_argv_and_working_directory(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            root = Path(directory)
            output = root / "result.json"
            command = [sys.executable, "-c", "print('ok')", "--literal-child-arg"]
            exit_code = measure_command.main(
                [
                    "--telemetry",
                    "off",
                    "--command-cwd",
                    str(root),
                    "--output-json",
                    str(output),
                    "--",
                    *command,
                ]
            )

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["command"], command)
            self.assertEqual(report["command_cwd"], str(root.resolve()))
            self.assertEqual(report["command_stdout"], "ok\n")
            self.assertEqual(report["command_stderr"], "")


if __name__ == "__main__":
    unittest.main()

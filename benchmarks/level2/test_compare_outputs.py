"""Unit tests for the Level 2 CPU/GPU output comparator."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_outputs import compare_arrays  # noqa: E402


class CompareOutputsTests(unittest.TestCase):
    def test_small_fft_roundoff_passes(self):
        reference = np.arange(12, dtype=np.float64).reshape(3, 4)
        candidate = reference.copy()
        candidate[1, 2] += 1e-11

        result = compare_arrays(
            reference,
            candidate,
            max_absolute_error=1e-8,
            max_relative_l2_error=1e-12,
        )

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["within_tolerance"])
        self.assertFalse(result["bitwise_exact"])

    def test_large_error_fails(self):
        reference = np.ones((2, 2), dtype=np.float64)
        candidate = reference.copy()
        candidate[0, 0] = 2.0

        result = compare_arrays(
            reference,
            candidate,
            max_absolute_error=1e-8,
            max_relative_l2_error=1e-12,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "numerical_tolerance_exceeded")

    def test_shape_mismatch_fails_without_broadcasting(self):
        result = compare_arrays(
            np.ones((2, 2)),
            np.ones((2, 1)),
            max_absolute_error=1e-8,
            max_relative_l2_error=1e-12,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "shape_mismatch")

    def test_dtype_mismatch_fails_even_when_values_match(self):
        result = compare_arrays(
            np.ones((2, 2), dtype=np.float64),
            np.ones((2, 2), dtype=np.float32),
            max_absolute_error=1e-8,
            max_relative_l2_error=1e-12,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "dtype_mismatch")

    def test_nonzero_candidate_against_zero_reference_is_json_safe(self):
        result = compare_arrays(
            np.zeros((2, 2), dtype=np.float64),
            np.ones((2, 2), dtype=np.float64),
            max_absolute_error=2.0,
            max_relative_l2_error=1.0,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["relative_l2_error"])


if __name__ == "__main__":
    unittest.main()

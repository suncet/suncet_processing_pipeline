"""Compare saved CPU and GPU Level 2 benchmark arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np


def _finite_nonnegative(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-absolute-error", type=_finite_nonnegative, default=1e-8)
    parser.add_argument("--max-relative-l2-error", type=_finite_nonnegative, default=1e-12)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    max_absolute_error: float,
    max_relative_l2_error: float,
) -> dict[str, object]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    same_shape = reference.shape == candidate.shape
    same_dtype = reference.dtype == candidate.dtype
    result: dict[str, object] = {
        "same_shape": same_shape,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "same_dtype": same_dtype,
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "max_absolute_error_limit": max_absolute_error,
        "relative_l2_error_limit": max_relative_l2_error,
    }
    if not same_shape:
        result.update({"status": "failed", "reason": "shape_mismatch"})
        return result
    if reference.size == 0:
        result.update({"status": "failed", "reason": "empty_arrays"})
        return result

    reference_finite = np.isfinite(reference)
    candidate_finite = np.isfinite(candidate)
    finite_state_mismatch_count = int(
        np.count_nonzero(reference_finite != candidate_finite)
    )
    all_finite = bool(np.all(reference_finite) and np.all(candidate_finite))
    result.update(
        {
            "all_pixels_finite": all_finite,
            "finite_state_mismatch_count": finite_state_mismatch_count,
            "bitwise_exact": bool(np.array_equal(reference, candidate)),
        }
    )
    if not all_finite:
        result.update({"status": "failed", "reason": "nonfinite_pixels"})
        return result

    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    maximum_absolute_error = float(np.max(np.abs(difference)))
    root_mean_square_error = float(np.sqrt(np.mean(difference**2)))
    reference_norm = float(np.linalg.norm(np.asarray(reference, dtype=np.float64)))
    difference_norm = float(np.linalg.norm(difference))
    relative_l2_error = (
        difference_norm / reference_norm
        if reference_norm > 0
        else (0.0 if difference_norm == 0 else None)
    )
    within_tolerance = bool(
        same_dtype
        and maximum_absolute_error <= max_absolute_error
        and relative_l2_error is not None
        and relative_l2_error <= max_relative_l2_error
    )
    result.update(
        {
            "maximum_absolute_error": maximum_absolute_error,
            "root_mean_square_error": root_mean_square_error,
            "relative_l2_error": relative_l2_error,
            "within_tolerance": within_tolerance,
            "status": "passed" if within_tolerance else "failed",
            "reason": (
                None
                if within_tolerance
                else (
                    "dtype_mismatch"
                    if not same_dtype
                    else "numerical_tolerance_exceeded"
                )
            ),
        }
    )
    return result


def _write_json(path: Path, payload: dict[str, object], overwrite: bool) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    reference_path = arguments.reference.expanduser().resolve()
    candidate_path = arguments.candidate.expanduser().resolve()
    reference = np.load(reference_path, allow_pickle=False)
    candidate = np.load(candidate_path, allow_pickle=False)
    result = compare_arrays(
        reference,
        candidate,
        max_absolute_error=arguments.max_absolute_error,
        max_relative_l2_error=arguments.max_relative_l2_error,
    )
    result.update(
        {
            "schema_version": 1,
            "reference_path": str(reference_path),
            "reference_sha256": _sha256(reference_path),
            "candidate_path": str(candidate_path),
            "candidate_sha256": _sha256(candidate_path),
        }
    )
    _write_json(arguments.output_json, result, arguments.overwrite)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

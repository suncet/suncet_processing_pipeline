"""Canonical paths below the required ``suncet_data`` root.

The environment variable deliberately remains lower-case for compatibility with
the existing SunCET development environments. All managed data paths should be
constructed here instead of embedding a workstation path or relying on the
repository's current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path


StrPath = str | os.PathLike[str]


SUNCET_DATA_ENV = "suncet_data"
SUNCET_CTDB_ENV = "suncet_ctdb"


class SuncetDataPathError(RuntimeError):
    """Raised when the configured SunCET data root is missing or invalid."""


def _required_absolute_directory(variable: str, *, must_exist: bool) -> Path:
    raw_value = os.environ.get(variable, "").strip()
    if not raw_value:
        raise SuncetDataPathError(
            f"Required environment variable {variable!r} is not set."
        )

    unexpanded = Path(raw_value).expanduser()
    if not unexpanded.is_absolute():
        raise SuncetDataPathError(
            f"Environment variable {variable!r} must be an absolute path; "
            f"got {raw_value!r}."
        )

    root = unexpanded.resolve(strict=False)
    if must_exist and not root.is_dir():
        raise SuncetDataPathError(
            f"Directory named by {variable!r} does not exist: {root}"
        )
    return root


def get_data_root(*, must_exist: bool = True) -> Path:
    """Return the absolute path named by the required ``suncet_data`` variable."""
    return _required_absolute_directory(SUNCET_DATA_ENV, must_exist=must_exist)


def get_ctdb_root(*, must_exist: bool = True) -> Path:
    """Return the separate private CTDB root named by ``suncet_ctdb``."""
    return _required_absolute_directory(SUNCET_CTDB_ENV, must_exist=must_exist)


def resolve_ctdb_root(configured_path: StrPath | None = None) -> Path:
    """Resolve the CTDB base without placing it below ``suncet_data``.

    Portable configs should use ``${suncet_ctdb}``. An explicit absolute path is
    accepted for specialized configurations.
    """
    if configured_path is None:
        path = get_ctdb_root()
    else:
        raw_value = os.fspath(configured_path).strip()
        expanded = os.path.expandvars(os.path.expanduser(raw_value))
        if "$" in expanded:
            raise SuncetDataPathError(
                f"CTDB base contains an undefined environment variable: {raw_value!r}"
            )
        path = Path(expanded)
        if not path.is_absolute():
            raise SuncetDataPathError(
                f"CTDB base must be an absolute path or ${{{SUNCET_CTDB_ENV}}}; "
                f"got {raw_value!r}."
            )
        path = path.resolve(strict=False)
        if not path.is_dir():
            raise SuncetDataPathError(f"CTDB root does not exist: {path}")
    data_root = get_data_root()
    if (
        path == data_root
        or path.is_relative_to(data_root)
        or data_root.is_relative_to(path)
    ):
        raise SuncetDataPathError(
            "The private CTDB root and public suncet_data tree must not overlap: "
            f"ctdb={path}, data={data_root}"
        )
    return path


def data_path(*parts: StrPath, must_exist: bool = False) -> Path:
    """Return a managed path below ``suncet_data``.

    Absolute components and paths which escape the data root are rejected. This
    keeps the directory structure identical when only ``suncet_data`` changes.
    """
    root = get_data_root()
    candidate = root
    for part in parts:
        component = Path(part).expanduser()
        if component.is_absolute():
            raise SuncetDataPathError(
                f"Managed SunCET path components must be relative; got {part!r}."
            )
        candidate /= component

    candidate = candidate.resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise SuncetDataPathError(
            f"Managed path escapes the SunCET data root {root}: {candidate}"
        )
    if must_exist and not candidate.exists():
        raise SuncetDataPathError(f"Required SunCET data path does not exist: {candidate}")
    return candidate


def resolve_data_path(path: StrPath, *, must_exist: bool = False) -> Path:
    """Resolve a configured path, rooting relative values below ``suncet_data``.

    Explicit absolute paths remain supported for one-off command-line inputs,
    but ``suncet_data`` is still validated so every pipeline run has a declared
    managed data root.
    """
    root = get_data_root()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve(strict=False)
    if must_exist and not candidate.exists():
        raise SuncetDataPathError(f"Required data path does not exist: {candidate}")
    return candidate


def processing_runs_root() -> Path:
    """Return the canonical directory containing named processing runs."""
    return data_path("processing_runs")


def processing_run_path(run_name: str) -> Path:
    """Return one named processing-run directory below ``suncet_data``."""
    candidate = Path(run_name)
    if (
        not run_name.strip()
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or run_name in {".", ".."}
    ):
        raise SuncetDataPathError(
            f"Run name must be one non-empty path component; got {run_name!r}."
        )
    return data_path("processing_runs", run_name)

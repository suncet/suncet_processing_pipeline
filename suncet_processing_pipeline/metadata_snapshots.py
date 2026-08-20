"""Versioned metadata definitions and immutable per-run snapshots."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .data_paths import get_data_root


METADATA_VERSION = "1.0.2dev"
FITS_SOURCE_FILENAME = (
    f"suncet_metadata_definition_v{METADATA_VERSION}-FITS.csv"
)
NETCDF_ZARR_SOURCE_FILENAME = (
    f"suncet_metadata_definition_v{METADATA_VERSION}-NetCDF-Zarr.csv"
)
FITS_SNAPSHOT_FILENAME = "suncet_metadata_definition_fits.csv"
NETCDF_ZARR_SNAPSHOT_FILENAME = "suncet_metadata_definition_nczarr.csv"
VERSION_FILENAME = "suncet_metadata_definition_version.txt"
MANIFEST_FILENAME = "metadata_snapshot.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_metadata_for_run(
    run_dir: str | Path,
    *,
    data_root: str | Path | None = None,
) -> dict[str, object]:
    """Copy the current authoritative definitions into a new run directory.

    The copies have stable historical filenames for existing readers. A manifest
    records the source names and checksums so an accidental later edit is detected.
    """
    run_path = Path(run_dir)
    root = Path(data_root) if data_root is not None else get_data_root()
    metadata_dir = root / "metadata"
    sources = {
        FITS_SNAPSHOT_FILENAME: metadata_dir / FITS_SOURCE_FILENAME,
        NETCDF_ZARR_SNAPSHOT_FILENAME: metadata_dir / NETCDF_ZARR_SOURCE_FILENAME,
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Initialize $suncet_data metadata before creating a run; missing: "
            + ", ".join(missing)
        )

    copied: dict[str, dict[str, object]] = {}
    for snapshot_name, source in sources.items():
        destination = run_path / snapshot_name
        shutil.copy2(source, destination)
        copied[snapshot_name] = {
            "source": str(source.relative_to(root)),
            "sha256": sha256_file(destination),
            "size_bytes": destination.stat().st_size,
        }

    version_path = run_path / VERSION_FILENAME
    version_path.write_text(METADATA_VERSION + "\n", encoding="utf-8")
    copied[VERSION_FILENAME] = {
        "source": f"metadata/{VERSION_FILENAME}",
        "sha256": sha256_file(version_path),
        "size_bytes": version_path.stat().st_size,
    }

    manifest: dict[str, object] = {
        "schema_version": 1,
        "metadata_version": METADATA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": copied,
    }
    manifest_path = run_path / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def verify_run_metadata_snapshot(run_dir: str | Path) -> dict[str, object] | None:
    """Validate a run snapshot when a manifest is present.

    Historical runs without a manifest remain readable. New runs fail loudly if
    any snapshotted definition no longer matches its recorded checksum.
    """
    run_path = Path(run_dir)
    manifest_path = run_path / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, record in manifest.get("files", {}).items():
        path = run_path / filename
        if not path.is_file():
            raise FileNotFoundError(f"Metadata snapshot file is missing: {path}")
        expected = record.get("sha256")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Metadata snapshot checksum mismatch for {path}: "
                f"expected {expected}, got {actual}"
            )
    return manifest

"""Reviewed input-manifest contract for Level 4 CME image sequences.

Historical synthetic SunCET FITS files have useful image content but unreliable
timestamps and mixed provenance. This module keeps file selection, ordering,
cadence assumptions, and upstream-processing state explicit. In particular,
the numeric suffix of a filename is never treated as a time coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ManifestValidationError(ValueError):
    """Raised when a CME sequence manifest is unsafe or contradictory."""


class InputSourceKind(str, Enum):
    """Scientifically distinct Level 4 input paths."""

    SYNTHETIC_BYPASS = "synthetic_bypass"
    PRODUCTION_LEVEL3 = "production_level3"


class TimeAxisKind(str, Enum):
    """Supported sources of the sequence time coordinate."""

    FIXED_CADENCE = "fixed_cadence"
    FITS_HEADERS = "fits_headers"


class CadenceStatus(str, Enum):
    """Whether a fixed cadence is provisional or established."""

    ASSUMED = "assumed"
    VERIFIED = "verified"


class CorrectionState(str, Enum):
    """Whether an upstream processing operation has been applied."""

    APPLIED = "applied"
    NOT_APPLIED = "not_applied"
    UNKNOWN = "unknown"


class ManifestPathBase(str, Enum):
    """Root against which portable manifest paths are resolved."""

    MANIFEST_DIRECTORY = "manifest_directory"
    SUNCET_DATA = "suncet_data"


def _enum_value(enum_type: type[Enum], value: object, field_name: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(repr(member.value) for member in enum_type)
        raise ManifestValidationError(
            f"{field_name} must be one of {choices}; got {value!r}."
        ) from exc


def _review_time(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ManifestValidationError(
            "review.reviewed_at_utc must be a valid ISO 8601 timestamp."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ManifestValidationError(
            "review.reviewed_at_utc must include a UTC offset or trailing Z."
        )
    return parsed


def _validate_portable_path(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ManifestValidationError("Frame paths must be non-empty.")
    if "\\" in cleaned:
        raise ManifestValidationError(
            f"Manifest paths must use portable forward slashes: {value!r}."
        )
    posix = PurePosixPath(cleaned)
    windows = PureWindowsPath(cleaned)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ManifestValidationError(
            f"Manifest frame paths must be relative: {value!r}."
        )
    if any(part in {"", ".", ".."} for part in posix.parts):
        raise ManifestValidationError(
            f"Manifest frame path may not contain '.' or '..': {value!r}."
        )
    return posix.as_posix()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for one input file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ManifestReview:
    """Human review record for file ordering and declared assumptions."""

    status: str
    reviewed_by: str
    reviewed_at_utc: str

    def __post_init__(self) -> None:
        if self.status != "reviewed":
            raise ManifestValidationError(
                "A usable sequence manifest must have review.status='reviewed'."
            )
        if not self.reviewed_by.strip():
            raise ManifestValidationError("review.reviewed_by must be non-empty.")
        _review_time(self.reviewed_at_utc)

    def to_dict(self) -> dict[str, str]:
        return {
            "status": self.status,
            "reviewed_by": self.reviewed_by,
            "reviewed_at_utc": self.reviewed_at_utc,
        }

    @classmethod
    def from_dict(cls, values: object) -> "ManifestReview":
        if not isinstance(values, Mapping):
            raise ManifestValidationError("review must be a JSON object.")
        return cls(
            status=str(values.get("status", "")),
            reviewed_by=str(values.get("reviewed_by", "")),
            reviewed_at_utc=str(values.get("reviewed_at_utc", "")),
        )


@dataclass(frozen=True)
class ManifestTimeAxis:
    """Time-coordinate declaration independent of FITS header contents."""

    kind: TimeAxisKind | str
    cadence_seconds: float | None = None
    cadence_status: CadenceStatus | str | None = None
    absolute_time_valid: bool = False

    def __post_init__(self) -> None:
        kind = _enum_value(TimeAxisKind, self.kind, "time_axis.kind")
        object.__setattr__(self, "kind", kind)

        status: CadenceStatus | None = None
        if self.cadence_status is not None:
            status = _enum_value(
                CadenceStatus,
                self.cadence_status,
                "time_axis.cadence_status",
            )
            object.__setattr__(self, "cadence_status", status)

        if not isinstance(self.absolute_time_valid, bool):
            raise ManifestValidationError(
                "time_axis.absolute_time_valid must be true or false."
            )

        if kind == TimeAxisKind.FIXED_CADENCE:
            if self.cadence_seconds is None or status is None:
                raise ManifestValidationError(
                    "fixed_cadence requires cadence_seconds and cadence_status."
                )
            try:
                cadence = float(self.cadence_seconds)
            except (TypeError, ValueError) as exc:
                raise ManifestValidationError(
                    "time_axis.cadence_seconds must be numeric."
                ) from exc
            if not math.isfinite(cadence) or cadence <= 0:
                raise ManifestValidationError(
                    "time_axis.cadence_seconds must be finite and greater than zero."
                )
            object.__setattr__(self, "cadence_seconds", cadence)
        else:
            if self.cadence_seconds is not None or status is not None:
                raise ManifestValidationError(
                    "FITS-header timing may not also declare a fixed cadence."
                )
            if not self.absolute_time_valid:
                raise ManifestValidationError(
                    "FITS-header timing requires absolute_time_valid=true."
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "cadence_seconds": self.cadence_seconds,
            "cadence_status": (
                self.cadence_status.value
                if isinstance(self.cadence_status, CadenceStatus)
                else None
            ),
            "absolute_time_valid": self.absolute_time_valid,
        }

    @classmethod
    def from_dict(cls, values: object) -> "ManifestTimeAxis":
        if not isinstance(values, Mapping):
            raise ManifestValidationError("time_axis must be a JSON object.")
        return cls(
            kind=values.get("kind", ""),
            cadence_seconds=values.get("cadence_seconds"),
            cadence_status=values.get("cadence_status"),
            absolute_time_valid=values.get("absolute_time_valid", False),
        )


@dataclass(frozen=True)
class UpstreamProcessing:
    """State of the Level 2 and Level 3 operations bypassed by synthetic data."""

    level2_psf_deconvolution: CorrectionState | str
    level3_geometric_correction: CorrectionState | str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "level2_psf_deconvolution",
            _enum_value(
                CorrectionState,
                self.level2_psf_deconvolution,
                "upstream_processing.level2_psf_deconvolution",
            ),
        )
        object.__setattr__(
            self,
            "level3_geometric_correction",
            _enum_value(
                CorrectionState,
                self.level3_geometric_correction,
                "upstream_processing.level3_geometric_correction",
            ),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "level2_psf_deconvolution": self.level2_psf_deconvolution.value,
            "level3_geometric_correction": self.level3_geometric_correction.value,
        }

    @classmethod
    def from_dict(cls, values: object) -> "UpstreamProcessing":
        if not isinstance(values, Mapping):
            raise ManifestValidationError(
                "upstream_processing must be a JSON object."
            )
        return cls(
            level2_psf_deconvolution=values.get(
                "level2_psf_deconvolution", ""
            ),
            level3_geometric_correction=values.get(
                "level3_geometric_correction", ""
            ),
        )


@dataclass(frozen=True)
class ManifestFrame:
    """One explicitly ordered frame and its reproducibility metadata."""

    path: str
    frame_number: int
    sha256: str
    source_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _validate_portable_path(self.path))
        if isinstance(self.frame_number, bool) or not isinstance(
            self.frame_number, int
        ):
            raise ManifestValidationError("frame_number must be an integer.")
        if self.frame_number < 0:
            raise ManifestValidationError("frame_number may not be negative.")
        if self.source_index is not None and (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
        ):
            raise ManifestValidationError("source_index must be an integer or null.")
        digest = self.sha256.strip().lower()
        if not _SHA256_PATTERN.fullmatch(digest):
            raise ManifestValidationError(
                f"Frame {self.path!r} must declare a 64-character SHA-256 digest."
            )
        object.__setattr__(self, "sha256", digest)

    def to_dict(self) -> dict[str, object]:
        values: dict[str, object] = {
            "path": self.path,
            "frame_number": self.frame_number,
            "sha256": self.sha256,
        }
        if self.source_index is not None:
            values["source_index"] = self.source_index
        return values

    @classmethod
    def from_dict(cls, values: object) -> "ManifestFrame":
        if not isinstance(values, Mapping):
            raise ManifestValidationError("Each frame must be a JSON object.")
        frame_number = values.get("frame_number")
        if isinstance(frame_number, bool) or not isinstance(frame_number, int):
            raise ManifestValidationError("Each frame requires integer frame_number.")
        source_index = values.get("source_index")
        if source_index is not None and (
            isinstance(source_index, bool) or not isinstance(source_index, int)
        ):
            raise ManifestValidationError("source_index must be an integer or null.")
        return cls(
            path=str(values.get("path", "")),
            frame_number=frame_number,
            source_index=source_index,
            sha256=str(values.get("sha256", "")),
        )


@dataclass(frozen=True)
class SequenceManifest:
    """Portable, reviewed description of one ordered image sequence."""

    scenario_id: str
    source_kind: InputSourceKind | str
    review: ManifestReview
    time_axis: ManifestTimeAxis
    upstream_processing: UpstreamProcessing
    frames: tuple[ManifestFrame, ...]
    path_base: ManifestPathBase | str = ManifestPathBase.MANIFEST_DIRECTORY
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or (
            self.schema_version != MANIFEST_SCHEMA_VERSION
        ):
            raise ManifestValidationError(
                "Unsupported CME sequence manifest schema version "
                f"{self.schema_version!r}; expected {MANIFEST_SCHEMA_VERSION}."
            )
        if not self.scenario_id.strip():
            raise ManifestValidationError("scenario_id must be non-empty.")
        source_kind = _enum_value(
            InputSourceKind, self.source_kind, "source_kind"
        )
        path_base = _enum_value(ManifestPathBase, self.path_base, "path_base")
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "path_base", path_base)
        object.__setattr__(self, "frames", tuple(self.frames))

        if not isinstance(self.review, ManifestReview):
            raise ManifestValidationError("review must be a ManifestReview.")
        if not isinstance(self.time_axis, ManifestTimeAxis):
            raise ManifestValidationError("time_axis must be a ManifestTimeAxis.")
        if not isinstance(self.upstream_processing, UpstreamProcessing):
            raise ManifestValidationError(
                "upstream_processing must be an UpstreamProcessing record."
            )
        if len(self.frames) < 2:
            raise ManifestValidationError(
                "A tracking sequence manifest must contain at least two frames."
            )
        if any(not isinstance(frame, ManifestFrame) for frame in self.frames):
            raise ManifestValidationError("frames must contain ManifestFrame records.")

        paths = [frame.path for frame in self.frames]
        if len(paths) != len(set(paths)):
            raise ManifestValidationError("Manifest frame paths must be unique.")
        numbers = [frame.frame_number for frame in self.frames]
        if any(right <= left for left, right in zip(numbers, numbers[1:])):
            raise ManifestValidationError(
                "Manifest frame_number values must be strictly increasing in "
                "declared file order."
            )

        if source_kind == InputSourceKind.SYNTHETIC_BYPASS:
            if (
                self.time_axis.kind == TimeAxisKind.FIXED_CADENCE
                and self.time_axis.absolute_time_valid
            ):
                raise ManifestValidationError(
                    "Synthetic fixed-cadence input may not claim valid absolute UTC."
                )
        else:
            if self.time_axis.kind != TimeAxisKind.FITS_HEADERS:
                raise ManifestValidationError(
                    "production_level3 must use strictly increasing FITS timestamps."
                )
            if (
                self.upstream_processing.level2_psf_deconvolution
                != CorrectionState.APPLIED
                or self.upstream_processing.level3_geometric_correction
                != CorrectionState.APPLIED
            ):
                raise ManifestValidationError(
                    "production_level3 requires applied Level 2 PSF deconvolution "
                    "and Level 3 geometric correction."
                )

        try:
            json.dumps(dict(self.metadata))
        except (TypeError, ValueError) as exc:
            raise ManifestValidationError(
                "Manifest metadata must be JSON serializable."
            ) from exc

    @property
    def files(self) -> tuple[str, ...]:
        """Compatibility view of the ordered frame paths."""

        return tuple(frame.path for frame in self.frames)

    @property
    def input_stage(self) -> str:
        """Compatibility alias for the explicit source kind."""

        return self.source_kind.value

    @property
    def time_mode(self) -> str:
        return self.time_axis.kind.value

    @property
    def cadence_seconds(self) -> float | None:
        return self.time_axis.cadence_seconds

    @property
    def start_time_utc(self) -> None:
        """Synthetic manifests deliberately do not invent an absolute start."""

        return None

    def resolve_files(
        self,
        manifest_directory: str | Path,
        *,
        data_root: str | Path | None = None,
    ) -> tuple[Path, ...]:
        """Resolve safe relative paths without changing their declared order."""

        if self.path_base == ManifestPathBase.SUNCET_DATA:
            if data_root is None:
                from suncet_processing_pipeline.data_paths import get_data_root

                base = get_data_root()
            else:
                base = Path(data_root).expanduser().resolve()
        else:
            base = Path(manifest_directory).expanduser().resolve()

        resolved: list[Path] = []
        for frame in self.frames:
            candidate = (base / PurePosixPath(frame.path)).resolve()
            if not candidate.is_relative_to(base):
                raise ManifestValidationError(
                    f"Manifest frame escapes its declared path base: {frame.path!r}."
                )
            resolved.append(candidate)
        return tuple(resolved)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "source_kind": self.source_kind.value,
            "path_base": self.path_base.value,
            "review": self.review.to_dict(),
            "time_axis": self.time_axis.to_dict(),
            "upstream_processing": self.upstream_processing.to_dict(),
            "frames": [frame.to_dict() for frame in self.frames],
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "SequenceManifest":
        """Validate and construct a manifest from decoded JSON."""

        if not isinstance(values, Mapping):
            raise ManifestValidationError("Manifest root must be a JSON object.")
        frames = values.get("frames")
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
            if "files" in values:
                raise ManifestValidationError(
                    "Legacy files-only manifests are not safe for fixed-cadence "
                    "tracking. Replace files with reviewed frame records containing "
                    "frame_number and sha256."
                )
            raise ManifestValidationError("frames must be a JSON array.")

        source_kind = values.get("source_kind", values.get("input_stage", ""))
        if source_kind == "level3":
            source_kind = InputSourceKind.PRODUCTION_LEVEL3.value

        schema_version = values.get("schema_version", -1)
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ManifestValidationError("schema_version must be an integer.")

        metadata = values.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ManifestValidationError("metadata must be a JSON object.")

        return cls(
            schema_version=schema_version,
            scenario_id=str(values.get("scenario_id", "")),
            source_kind=source_kind,
            path_base=values.get(
                "path_base", ManifestPathBase.MANIFEST_DIRECTORY.value
            ),
            review=ManifestReview.from_dict(values.get("review")),
            time_axis=ManifestTimeAxis.from_dict(values.get("time_axis")),
            upstream_processing=UpstreamProcessing.from_dict(
                values.get("upstream_processing")
            ),
            frames=tuple(ManifestFrame.from_dict(item) for item in frames),
            notes=(
                str(values["notes"])
                if values.get("notes") is not None
                else None
            ),
            metadata=dict(metadata),
        )


def read_manifest(path: str | Path) -> SequenceManifest:
    """Read and validate a JSON sequence manifest."""

    manifest_path = Path(path).expanduser().resolve()
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            values = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(
            f"Manifest is not valid JSON: {manifest_path}: {exc}"
        ) from exc
    return SequenceManifest.from_dict(values)


def write_manifest(manifest: SequenceManifest, path: str | Path) -> Path:
    """Write a manifest atomically and return its resolved path."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary_path.replace(output_path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return output_path


def manifest_from_paths(
    paths: Sequence[str | Path],
    *,
    scenario_id: str,
    source_kind: InputSourceKind | str,
    review: ManifestReview,
    relative_to: str | Path,
    path_base: ManifestPathBase | str = ManifestPathBase.MANIFEST_DIRECTORY,
    cadence_seconds: float | None = None,
    cadence_status: CadenceStatus | str = CadenceStatus.ASSUMED,
    frame_numbers: Sequence[int] | None = None,
    source_indices: Sequence[int | None] | None = None,
    upstream_processing: UpstreamProcessing | None = None,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SequenceManifest:
    """Create a reviewed manifest and hash every declared input.

    ``frame_numbers`` are explicit time-grid positions. For example, if source
    suffixes normally advance by three but suffix 018 is absent, suffix 015 can
    be frame 5 and suffix 021 frame 7. Their elapsed-time separation is then
    two cadences; no filename arithmetic occurs in the loader.
    """

    resolved = tuple(Path(path).expanduser().resolve() for path in paths)
    if frame_numbers is None:
        numbers = tuple(range(len(resolved)))
    else:
        numbers = tuple(frame_numbers)
    if source_indices is None:
        indices: tuple[int | None, ...] = (None,) * len(resolved)
    else:
        indices = tuple(source_indices)
    if len(numbers) != len(resolved) or len(indices) != len(resolved):
        raise ManifestValidationError(
            "paths, frame_numbers, and source_indices must have equal lengths."
        )

    base = Path(relative_to).expanduser().resolve()
    frames: list[ManifestFrame] = []
    for path, frame_number, source_index in zip(
        resolved, numbers, indices, strict=True
    ):
        if not path.is_file():
            raise ManifestValidationError(f"Input file does not exist: {path}")
        try:
            relative = path.relative_to(base)
        except ValueError as exc:
            raise ManifestValidationError(
                f"Input file is outside the requested path base {base}: {path}"
            ) from exc
        frames.append(
            ManifestFrame(
                path=relative.as_posix(),
                frame_number=frame_number,
                source_index=source_index,
                sha256=sha256_file(path),
            )
        )

    normalized_source = _enum_value(InputSourceKind, source_kind, "source_kind")
    if normalized_source == InputSourceKind.SYNTHETIC_BYPASS:
        if cadence_seconds is None:
            time_axis = ManifestTimeAxis(
                kind=TimeAxisKind.FITS_HEADERS,
                absolute_time_valid=True,
            )
        else:
            time_axis = ManifestTimeAxis(
                kind=TimeAxisKind.FIXED_CADENCE,
                cadence_seconds=cadence_seconds,
                cadence_status=cadence_status,
                absolute_time_valid=False,
            )
        corrections = upstream_processing or UpstreamProcessing(
            level2_psf_deconvolution=CorrectionState.NOT_APPLIED,
            level3_geometric_correction=CorrectionState.NOT_APPLIED,
        )
    else:
        if cadence_seconds is not None:
            raise ManifestValidationError(
                "Production Level 3 manifests may not override FITS timestamps."
            )
        time_axis = ManifestTimeAxis(
            kind=TimeAxisKind.FITS_HEADERS,
            absolute_time_valid=True,
        )
        corrections = upstream_processing or UpstreamProcessing(
            level2_psf_deconvolution=CorrectionState.APPLIED,
            level3_geometric_correction=CorrectionState.APPLIED,
        )

    return SequenceManifest(
        scenario_id=scenario_id,
        source_kind=normalized_source,
        path_base=path_base,
        review=review,
        time_axis=time_axis,
        upstream_processing=corrections,
        frames=tuple(frames),
        notes=notes,
        metadata=metadata or {},
    )

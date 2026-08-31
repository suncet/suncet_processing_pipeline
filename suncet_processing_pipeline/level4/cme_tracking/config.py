"""Serializable configuration for the first known-window CME tracker."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .likelihood import EvidenceConfig
from .polar import PolarConfig
from .summary import FrontSummaryConfig
from .tracking import TrackingConfig


@dataclass(frozen=True)
class KinematicsConfig:
    """Spline and uncertainty settings kept separate from front selection."""

    smoothing_factor: float | None = None
    endpoint_samples: int = 2
    uncertainty_samples: int = 128
    random_seed: int = 20260826
    minimum_front_coverage: float = 0.5
    minimum_confidence: float = 0.2
    maximum_gap_factor: float = 3.0
    height_outlier_filter_enabled: bool = False
    height_outlier_filter_window_samples: int = 7
    height_outlier_filter_absolute_tolerance_rsun: float = 0.2
    height_outlier_filter_minimum_neighbors: int = 4

    def __post_init__(self) -> None:
        if self.smoothing_factor is not None and (
            not math.isfinite(self.smoothing_factor) or self.smoothing_factor < 0
        ):
            raise ValueError("smoothing_factor must be finite and nonnegative.")
        if self.endpoint_samples < 0:
            raise ValueError("endpoint_samples must be nonnegative.")
        if self.uncertainty_samples == 1 or self.uncertainty_samples < 0:
            raise ValueError("uncertainty_samples must be zero or at least two.")
        for name, value in (
            ("minimum_front_coverage", self.minimum_front_coverage),
            ("minimum_confidence", self.minimum_confidence),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1].")
        if not math.isfinite(self.maximum_gap_factor) or self.maximum_gap_factor < 1:
            raise ValueError("maximum_gap_factor must be finite and at least one.")
        if not isinstance(self.height_outlier_filter_enabled, bool):
            raise ValueError("height_outlier_filter_enabled must be boolean.")
        window = self.height_outlier_filter_window_samples
        if isinstance(window, bool) or not isinstance(window, int) or window < 3 or window % 2 != 1:
            raise ValueError(
                "height_outlier_filter_window_samples must be an odd integer "
                "of at least three."
            )
        tolerance = self.height_outlier_filter_absolute_tolerance_rsun
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError(
                "height_outlier_filter_absolute_tolerance_rsun must be finite "
                "and positive."
            )
        neighbors = self.height_outlier_filter_minimum_neighbors
        if (
            isinstance(neighbors, bool)
            or not isinstance(neighbors, int)
            or neighbors < 2
            or neighbors >= window
        ):
            raise ValueError(
                "height_outlier_filter_minimum_neighbors must be at least two "
                "and smaller than the filter window."
            )


@dataclass(frozen=True)
class CMETrackingConfig:
    """All science-affecting settings for a reproducible prototype run."""

    polar: PolarConfig = field(default_factory=PolarConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    summary: FrontSummaryConfig = field(default_factory=FrontSummaryConfig)
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)
    solar_radius_km: float = 695_700.0
    field_of_view_margin_px: float = 20.0
    field_of_view_top_fraction: float = 0.20
    field_of_view_contact_fraction: float = 0.25
    field_of_view_minimum_contact_angles: int = 2
    field_of_view_minimum_consecutive_frames: int = 2
    algorithm_name: str = "polar_coherent_fragments_v1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.solar_radius_km) or self.solar_radius_km <= 0:
            raise ValueError("solar_radius_km must be finite and positive.")
        if (
            not math.isfinite(self.field_of_view_margin_px)
            or self.field_of_view_margin_px < 0
        ):
            raise ValueError("field_of_view_margin_px must be finite and nonnegative.")
        for name, value in (
            ("field_of_view_top_fraction", self.field_of_view_top_fraction),
            ("field_of_view_contact_fraction", self.field_of_view_contact_fraction),
        ):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0, 1].")
        if (
            isinstance(self.field_of_view_minimum_contact_angles, bool)
            or not isinstance(self.field_of_view_minimum_contact_angles, int)
            or self.field_of_view_minimum_contact_angles < 1
        ):
            raise ValueError(
                "field_of_view_minimum_contact_angles must be a positive integer."
            )
        if (
            isinstance(self.field_of_view_minimum_consecutive_frames, bool)
            or not isinstance(self.field_of_view_minimum_consecutive_frames, int)
            or self.field_of_view_minimum_consecutive_frames < 1
        ):
            raise ValueError(
                "field_of_view_minimum_consecutive_frames must be a positive "
                "integer."
            )
        if not self.algorithm_name.strip():
            raise ValueError("algorithm_name must be non-empty.")
        if self.schema_version != 1:
            raise ValueError("Only CME tracking configuration schema 1 is supported.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible structure suitable for provenance hashing."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CMETrackingConfig":
        """Validate a decoded configuration without ignoring unknown keys."""

        allowed = {
            "polar",
            "evidence",
            "tracking",
            "summary",
            "kinematics",
            "solar_radius_km",
            "field_of_view_margin_px",
            "field_of_view_top_fraction",
            "field_of_view_contact_fraction",
            "field_of_view_minimum_contact_angles",
            "field_of_view_minimum_consecutive_frames",
            "algorithm_name",
            "schema_version",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(
                "Unknown CME tracking configuration fields: "
                + ", ".join(sorted(unknown))
            )

        def section(name: str, constructor):
            section_values = values.get(name, {})
            if not isinstance(section_values, Mapping):
                raise ValueError(f"Configuration section {name!r} must be an object.")
            try:
                return constructor(**section_values)
            except TypeError as exc:
                raise ValueError(
                    f"Invalid fields in configuration section {name!r}: {exc}"
                ) from exc

        scalars = {
            key: values[key]
            for key in (
                "solar_radius_km",
                "field_of_view_margin_px",
                "field_of_view_top_fraction",
                "field_of_view_contact_fraction",
                "field_of_view_minimum_contact_angles",
                "field_of_view_minimum_consecutive_frames",
                "algorithm_name",
                "schema_version",
            )
            if key in values
        }
        return cls(
            polar=section("polar", PolarConfig),
            evidence=section("evidence", EvidenceConfig),
            tracking=section("tracking", TrackingConfig),
            summary=section("summary", FrontSummaryConfig),
            kinematics=section("kinematics", KinematicsConfig),
            **scalars,
        )


def read_configuration(path: str | Path) -> CMETrackingConfig:
    """Read a complete JSON algorithm configuration."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, Mapping):
        raise ValueError("CME tracking configuration root must be an object.")
    return CMETrackingConfig.from_dict(values)


def write_configuration(
    configuration: CMETrackingConfig,
    path: str | Path,
) -> Path:
    """Write a stable JSON configuration atomically."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(configuration.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return output

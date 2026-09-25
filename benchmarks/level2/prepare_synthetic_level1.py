#!/usr/bin/env python3
"""Prepare a provisional synthetic Level 1 DN/s product from simulator truth.

This utility is intentionally separate from the flight-data Level 1 writer.
Historical simulator FITS files carry template exposure metadata, so the
normalization parameters are read from the simulator configuration that made
the sequence and recorded explicitly in a provenance sidecar.
"""

from __future__ import annotations

import argparse
import configparser
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from astropy.io import fits
import numpy as np

from suncet_processing_pipeline.make_level1 import (
    create_circular_inner_mask,
    normalize_to_dn_per_second,
)
from suncet_processing_pipeline import metadata_managers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _elapsed_seconds_from_header(header: fits.Header) -> float:
    """Return numeric TELAPSE, repairing legacy simulator ``N/A`` values."""
    observed = header.get("TELAPSE")
    try:
        elapsed = float(observed)
    except (TypeError, ValueError):
        elapsed = math.nan
    if math.isfinite(elapsed) and elapsed >= 0:
        return elapsed

    start_value = header.get("DATE-BEG", header.get("DATE-OBS"))
    end_value = header.get("DATE-END")
    if not start_value or not end_value:
        raise ValueError(
            "Legacy non-numeric TELAPSE requires DATE-BEG/DATE-OBS and DATE-END"
        )

    def parse(value: object) -> datetime:
        text = str(value).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    elapsed = (parse(end_value) - parse(start_value)).total_seconds()
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("DATE-END precedes DATE-BEG/DATE-OBS")
    return elapsed


def _read_simulator_truth(config_path: Path) -> dict[str, object]:
    parser = configparser.ConfigParser()
    if not parser.read(config_path):
        raise FileNotFoundError(f"Simulator config not found: {config_path}")

    filtering = parser.getboolean("behavior", "filter_out_particle_hits")
    short_integration = parser.getfloat("shdr", "exposure_time_short")
    long_integration = parser.getfloat("shdr", "exposure_time_long")
    configured_short_stack = parser.getint(
        "shdr", "num_short_exposures_to_stack"
    )
    configured_long_stack = parser.getint(
        "shdr", "num_long_exposures_to_stack"
    )
    # Match Config._apply_shdr_stack_config_rules() in the simulator.  When
    # filtering is disabled, the runtime stack contains one integration even
    # if the historical INI retains the nominal filtered-stack counts.
    short_stack = configured_short_stack if filtering else 1
    long_stack = configured_long_stack if filtering else 1
    shift_bits = parser.getint("shdr", "num_shift_bits_32_to_16")
    inner_radius_solar = parser.getfloat("shdr", "inner_fov_circle_radius")
    binning = json.loads(parser.get("detector", "num_pixels_to_bin"))
    if len(binning) != 2 or any(int(value) <= 0 for value in binning):
        raise ValueError("num_pixels_to_bin must contain two positive integers")
    if filtering and (short_stack < 2 or long_stack < 2):
        raise ValueError("sum-minus-maximum filtering needs at least two frames")
    normalization = 2**shift_bits
    short_contributors = short_stack - (1 if filtering else 0)
    long_contributors = long_stack - (1 if filtering else 0)

    return {
        "filter_out_particle_hits": filtering,
        "short_integration_seconds": short_integration,
        "long_integration_seconds": long_integration,
        "configured_short_stack_count": configured_short_stack,
        "configured_long_stack_count": configured_long_stack,
        "short_stack_count": short_stack,
        "long_stack_count": long_stack,
        "right_shift_bits": shift_bits,
        "right_shift_divisor": normalization,
        "inner_radius_solar": inner_radius_solar,
        "binning_columns": int(binning[0]),
        "binning_rows": int(binning[1]),
        "effective_exposure_inner_seconds": (
            short_contributors * short_integration / normalization
        ),
        "effective_exposure_outer_seconds": (
            long_contributors * long_integration / normalization
        ),
    }


def _update_statistics(header: fits.Header, data: np.ndarray) -> None:
    values = np.asarray(data, dtype=np.float64)
    nonzero = values[values != 0]
    header["DATAZER"] = (int(np.count_nonzero(values == 0)), "Zero-valued pixels")
    header["DATAAVG"] = (float(np.mean(values)), "Mean of Level 1 pixels")
    header["DATAMDN"] = (float(np.median(values)), "Median of Level 1 pixels")
    header["DATASIG"] = (
        float(np.std(nonzero)) if nonzero.size else 0.0,
        "Std. deviation of non-zero Level 1 pixels",
    )
    header["DATAMIN"] = (float(np.min(values)), "Minimum Level 1 value")
    header["DATAMAX"] = (float(np.max(values)), "Maximum Level 1 value")
    if nonzero.size:
        for percentile in (1, 10, 25, 50, 75, 90, 95, 98, 99):
            header[f"DATAP{percentile:02d}"] = (
                float(np.percentile(nonzero, percentile)),
                f"Non-zero Level 1 {percentile}th percentile",
            )


def prepare_level1(
    input_path: Path,
    output_path: Path,
    simulator_config: Path,
    metadata_definition: Path,
    *,
    simulator_commit: str,
    pipeline_version: str,
    generation_rsun_arcsec: float | None = None,
    geometry_source: str | None = None,
    saturation_dn: int = 65535,
    overwrite: bool = False,
    _truth: dict[str, object] | None = None,
    _simulator_config_sha256: str | None = None,
    _metadata_definition_sha256: str | None = None,
) -> tuple[Path, Path]:
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    simulator_config = Path(simulator_config).expanduser().resolve()
    metadata_definition = Path(metadata_definition).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input FITS not found: {input_path}")
    if not metadata_definition.is_file():
        raise FileNotFoundError(
            f"Metadata definition not found: {metadata_definition}"
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing output: {output_path}")

    truth = dict(_truth) if _truth is not None else _read_simulator_truth(
        simulator_config
    )
    simulator_config_sha256 = (
        _simulator_config_sha256 or _sha256(simulator_config)
    )
    metadata_definition_sha256 = (
        _metadata_definition_sha256 or _sha256(metadata_definition)
    )
    with fits.open(input_path, memmap=False, checksum=False) as hdul:
        if hdul[0].data is None:
            raise ValueError("Input primary HDU has no image data")
        source = np.asarray(hdul[0].data)
        header = hdul[0].header.copy()
    declared_source_bunit = header.get("BUNIT")

    if source.ndim != 2:
        raise ValueError(f"Expected a two-dimensional image, got {source.shape}")
    if not np.all(np.isfinite(source)):
        raise ValueError("Input contains non-finite values")
    level = str(header.get("LEVEL", "")).strip().upper().removeprefix("LEVEL")
    if level not in {"0.5", ".5"}:
        raise ValueError(f"Input must declare LEVEL=0.5, got {header.get('LEVEL')!r}")

    rows, columns = source.shape
    bin_columns = int(truth["binning_columns"])
    bin_rows = int(truth["binning_rows"])
    if int(header.get("NBIN1", bin_columns)) != bin_columns:
        raise ValueError("NBIN1 does not match the simulator configuration")
    if int(header.get("NBIN2", bin_rows)) != bin_rows:
        raise ValueError("NBIN2 does not match the simulator configuration")

    product_rsun_arcsec = header.get("RSUN_OBS", header.get("RSUN"))
    if product_rsun_arcsec is None:
        raise ValueError("Input header lacks RSUN_OBS/RSUN for simulator geometry")
    if generation_rsun_arcsec is None:
        generation_rsun_arcsec = float(product_rsun_arcsec)
    generation_rsun_arcsec = float(generation_rsun_arcsec)
    if not math.isfinite(generation_rsun_arcsec) or generation_rsun_arcsec <= 0:
        raise ValueError("generation_rsun_arcsec must be finite and positive")
    scale_column = abs(float(header["CDELT1"]))
    scale_row = abs(float(header["CDELT2"]))
    if not math.isclose(scale_column, scale_row, rel_tol=0, abs_tol=1e-12):
        raise ValueError("This circular simulator fixture requires square pixels")

    # The generation-era simulator formed the circle before binning and used
    # SunPy nearest-neighbor resampling with center=False.  Output pixel (x, y)
    # therefore sampled unbinned pixel (bin_x*x, bin_y*y).
    center_column = columns / 2.0 - 1.0 / (2.0 * bin_columns)
    center_row = rows / 2.0 - 1.0 / (2.0 * bin_rows)
    radius_pixels = (
        float(truth["inner_radius_solar"])
        * generation_rsun_arcsec
        / scale_column
    )
    inner_mask = create_circular_inner_mask(
        source.shape,
        center_column=center_column,
        center_row=center_row,
        radius_pixels=radius_pixels,
    )
    exposures = {
        "inner": float(truth["effective_exposure_inner_seconds"]),
        "outer": float(truth["effective_exposure_outer_seconds"]),
    }
    level1 = normalize_to_dn_per_second(
        source,
        exposures,
        inner_mask=inner_mask,
    )
    saturation_mask = source >= saturation_dn

    for key in ("CHECKSUM", "DATASUM"):
        header.remove(key, ignore_missing=True, remove_all=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header["TITLE"] = ("SunCET Provisional Synthetic Level 1 Image", "Product title")
    header["FILENAME"] = (output_path.name, "Filename of this product")
    header["LEVEL"] = (1.0, "Data processing level number")
    header["BUNIT"] = ("DN/s", "Primary-array units")
    header["PIPEVRSN"] = (
        f"v{str(pipeline_version).removeprefix('v')}",
        "Processing pipeline version",
    )
    header["PROCSTAT"] = ("PROVISIONAL", "Processing maturity")
    header["SYNTHET"] = (True, "Synthetic input product")
    header["L0PARENT"] = (input_path.name, "Immediate parent product filename")
    header["DATE"] = (_utc_now(), "File generation time in UTC")
    header["TELAPSE"] = (
        _elapsed_seconds_from_header(header),
        "Elapsed time from first integration start through final integration end",
    )
    header["EFFEXPI"] = (exposures["inner"], "Effective inner exposure in seconds")
    header["EFFEXPO"] = (exposures["outer"], "Effective outer exposure in seconds")
    header["INTTIMEI"] = (
        float(truth["short_integration_seconds"]),
        "Short integration time in seconds",
    )
    header["INTTIMEO"] = (
        float(truth["long_integration_seconds"]),
        "Long integration time in seconds",
    )
    header["NSTACKI"] = (int(truth["short_stack_count"]), "Short stack count")
    header["NSTACKO"] = (int(truth["long_stack_count"]), "Long stack count")
    header["STKNORMI"] = (
        int(truth["right_shift_divisor"]),
        "Short-stack right-shift divisor",
    )
    header["STKNORMO"] = (
        int(truth["right_shift_divisor"]),
        "Long-stack right-shift divisor",
    )
    filtering = bool(truth["filter_out_particle_hits"])
    header["PIXFILTI"] = (
        filtering,
        "Inner stack used sum-minus-maximum filtering",
    )
    header["PIXFILTO"] = (
        filtering,
        "Outer stack used sum-minus-maximum filtering",
    )
    header["SIMICMX"] = (center_column, "Synthetic ICM center column, zero based")
    header["SIMICMY"] = (center_row, "Synthetic ICM center row, zero based")
    header["SIMICMR"] = (radius_pixels, "Synthetic ICM radius in output pixels")
    header["SIMRSUN"] = (
        generation_rsun_arcsec,
        "Apparent solar radius used when simulator formed composite",
    )
    header["EXP_MASK"] = ("SIMULATOR_CONFIG_CIRCLE", "Exposure-mask provenance")
    header["SOLAR_R"] = (
        float(product_rsun_arcsec) / scale_column,
        "Apparent solar radius in pixels",
    )
    if "RSUN_OBS" not in header:
        header["RSUN_OBS"] = (
            float(product_rsun_arcsec),
            "Apparent solar radius in arcsec",
        )
    header["DATASAT"] = (
        int(np.count_nonzero(saturation_mask)),
        "Pixels clipped at the pre-normalization DN ceiling",
    )
    header["DSATVAL"] = (float(saturation_dn), "Pre-normalization DN ceiling")
    header["CALDARK"] = (
        "NOT_APPLIED_SYNTHETIC",
        "Dark calibration not applied to this provisional fixture",
    )
    header["CALFLAT"] = (
        "NOT_APPLIED_SYNTHETIC",
        "Flat calibration not applied to this provisional fixture",
    )
    if "TYPECODE" in header:
        header["TYPECODE"] = (str(header["TYPECODE"]), header.comments["TYPECODE"])
    for key in ("INT_LED", "EXT_LED"):
        if key not in header or isinstance(header[key], (bool, np.bool_)):
            continue
        normalized = str(header[key]).strip().upper()
        if normalized in {"0", "F", "FALSE", "OFF", "NO"}:
            value = False
        elif normalized in {"1", "T", "TRUE", "ON", "YES"}:
            value = True
        else:
            raise ValueError(f"Cannot normalize {key}={header[key]!r} to boolean")
        header[key] = (value, header.comments[key])
    _update_statistics(header, level1)
    header.add_history(
        "PROVISIONAL SYNTHETIC LEVEL 1: applied only pixelwise exposure normalization"
    )
    header.add_history(
        "No dark, flat-field, bad-pixel, or other Level 1 calibration was applied"
    )
    if filtering:
        header.add_history(
            "Effective exposures derived from simulator sum-minus-maximum stacks and right shift"
        )
    else:
        header.add_history(
            "Particle filtering disabled; simulator runtime forced both stack counts to one"
        )
        header.add_history(
            "Effective exposures derived from single integrations and right shift"
        )
    header.add_history(
        f"Simulator generation commit: {simulator_commit}; config: {simulator_config.name}"
    )
    if geometry_source:
        header.add_history(f"Synthetic composite geometry source: {geometry_source}")
    header.add_history(
        "Source legacy CHECKSUM/DATASUM placeholders were discarded and recomputed"
    )
    if np.any(saturation_mask):
        header.add_history(
            "Pre-normalization saturated pixels are unrecoverable and retained numerically"
        )

    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        fits.PrimaryHDU(data=level1, header=header).writeto(
            temporary,
            overwrite=False,
            checksum=True,
        )
        with fits.open(temporary, checksum=True) as hdul:
            hdul.verify("exception")
            if hdul[0].verify_checksum() != 1 or hdul[0].verify_datasum() != 1:
                raise ValueError("Generated Level 1 FITS checksums did not validate")
            metadata_managers.validate_fits_header(
                hdul[0].header,
                metadata_definition,
                1,
                float_output_statistics=("DATAMIN", "DATAMAX"),
            )
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to replace existing output: {output_path}")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)

    provenance_path = output_path.with_suffix(".provenance.json")
    provenance = {
        "schema": "suncet.provisional_synthetic_level1",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "input": {
            "path": str(input_path),
            "sha256": _sha256(input_path),
            "declared_bunit_ignored_as_legacy_template": declared_source_bunit,
        },
        "simulator": {
            "config_path": str(simulator_config),
            "config_sha256": simulator_config_sha256,
            "generation_commit": simulator_commit,
            **truth,
        },
        "metadata_definition": {
            "path": str(metadata_definition),
            "sha256": metadata_definition_sha256,
        },
        "geometry": {
            "center_column_zero_based": center_column,
            "center_row_zero_based": center_row,
            "radius_pixels": radius_pixels,
            "generation_rsun_arcsec": generation_rsun_arcsec,
            "product_header_rsun_arcsec": float(product_rsun_arcsec),
            "source": geometry_source,
            "boundary_included": True,
            "inner_pixel_count": int(np.count_nonzero(inner_mask)),
        },
        "quality": {
            "pre_normalization_saturation_dn": saturation_dn,
            "saturated_pixel_count": int(np.count_nonzero(saturation_mask)),
            "saturated_values_recoverable": False,
        },
        "output": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "shape": list(level1.shape),
            "dtype": str(level1.dtype),
            "bunit": "DN/s",
        },
    }
    encoded = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    if provenance_path.exists() and not overwrite:
        output_path.unlink(missing_ok=True)
        raise FileExistsError(
            f"Refusing to replace existing provenance: {provenance_path}"
        )
    provenance_path.write_text(encoded, encoding="utf-8")
    return output_path, provenance_path


def prepare_level1_directory(
    input_directory: Path,
    output_directory: Path,
    simulator_config: Path,
    metadata_definition: Path,
    *,
    simulator_commit: str,
    pipeline_version: str,
    pattern: str = "*.fits",
    generation_rsun_arcsec: float | None = None,
    geometry_source: str | None = None,
    saturation_dn: int = 65535,
    overwrite: bool = False,
) -> list[tuple[Path, Path]]:
    """Prepare a deterministic directory of provisional synthetic Level 1 files.

    Static configuration and definition inputs are parsed/hashed once so a
    many-frame benchmark measures image processing rather than avoidable
    repeated setup work.
    """
    input_directory = Path(input_directory).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()
    simulator_config = Path(simulator_config).expanduser().resolve()
    metadata_definition = Path(metadata_definition).expanduser().resolve()
    if not input_directory.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_directory}")
    inputs = sorted(path for path in input_directory.glob(pattern) if path.is_file())
    if not inputs:
        raise FileNotFoundError(
            f"No inputs matching {pattern!r} in {input_directory}"
        )

    truth = _read_simulator_truth(simulator_config)
    simulator_config_sha256 = _sha256(simulator_config)
    metadata_definition_sha256 = _sha256(metadata_definition)
    version = str(pipeline_version).removeprefix("v")
    output_directory.mkdir(parents=True, exist_ok=True)

    products: list[tuple[Path, Path]] = []
    for input_path in inputs:
        output_path = output_directory / (
            f"{input_path.stem}_level1_v{version}_provisional.fits"
        )
        products.append(
            prepare_level1(
                input_path,
                output_path,
                simulator_config,
                metadata_definition,
                simulator_commit=simulator_commit,
                pipeline_version=version,
                generation_rsun_arcsec=generation_rsun_arcsec,
                geometry_source=geometry_source,
                saturation_dn=saturation_dn,
                overwrite=overwrite,
                _truth=truth,
                _simulator_config_sha256=simulator_config_sha256,
                _metadata_definition_sha256=metadata_definition_sha256,
            )
        )
    return products


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path)
    inputs.add_argument("--input-directory", type=Path)
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--output", type=Path)
    outputs.add_argument("--output-directory", type=Path)
    parser.add_argument("--pattern", default="*.fits")
    parser.add_argument("--simulator-config", type=Path, required=True)
    parser.add_argument("--metadata-definition", type=Path, required=True)
    parser.add_argument("--simulator-commit", required=True)
    parser.add_argument("--pipeline-version", default="2.0.0")
    parser.add_argument("--generation-rsun-arcsec", type=float)
    parser.add_argument("--geometry-source")
    parser.add_argument("--saturation-dn", type=int, default=65535)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (args.input is None) != (args.output is None):
        raise SystemExit("--input must be paired with --output")
    if (args.input_directory is None) != (args.output_directory is None):
        raise SystemExit(
            "--input-directory must be paired with --output-directory"
        )
    if args.input_directory is not None:
        products = prepare_level1_directory(
            args.input_directory,
            args.output_directory,
            args.simulator_config,
            args.metadata_definition,
            simulator_commit=args.simulator_commit,
            pipeline_version=args.pipeline_version,
            pattern=args.pattern,
            generation_rsun_arcsec=args.generation_rsun_arcsec,
            geometry_source=args.geometry_source,
            saturation_dn=args.saturation_dn,
            overwrite=args.overwrite,
        )
        for output, provenance in products:
            print(output)
            print(provenance)
        return 0

    output, provenance = prepare_level1(
        args.input,
        args.output,
        args.simulator_config,
        args.metadata_definition,
        simulator_commit=args.simulator_commit,
        pipeline_version=args.pipeline_version,
        generation_rsun_arcsec=args.generation_rsun_arcsec,
        geometry_source=args.geometry_source,
        saturation_dn=args.saturation_dn,
        overwrite=args.overwrite,
    )
    print(output)
    print(provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create SunCET Level 2 PSF-deconvolved image products.

The production boundary is Level 1 -> Level 2. An explicit synthetic Level
0.5 bypass exists solely for development fixtures while the Level 1 writer and
mission calibration set are still under construction.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid

from astropy.io import fits
import numpy as np
from termcolor import cprint

from suncet_processing_pipeline import config_parser
from suncet_processing_pipeline import metadata_managers
from suncet_processing_pipeline import suncet_deconv
from suncet_processing_pipeline.data_paths import data_path
from suncet_processing_pipeline.run_provenance import (
    ProcessingRunProvenance,
    resolved_config_snapshot,
    sha256_file,
)


class Level2:
    """Apply PSF deconvolution at the Level 1 -> Level 2 boundary."""

    EXPECTED_IMAGE_SHAPE = (750, 1000)
    DEFAULT_FLUX_RATIO_TOLERANCE = 0.05

    INPUT_KIND_LEVEL1 = "level1"
    INPUT_KIND_SYNTHETIC_LEVEL0_5_BYPASS = "synthetic_level0_5_bypass"
    INPUT_KINDS = (
        INPUT_KIND_LEVEL1,
        INPUT_KIND_SYNTHETIC_LEVEL0_5_BYPASS,
    )

    def __init__(
        self,
        config=None,
        diffraction_psf_file=None,
        scatter_psf_file=None,
        spec_file=None,
        resp_file=None,
        correction_factor=0.4,
        input_kind=INPUT_KIND_LEVEL1,
        product_status="PROVISIONAL",
        metadata_definition_file=None,
        overwrite=False,
        flux_ratio_tolerance=DEFAULT_FLUX_RATIO_TOLERANCE,
    ):
        """Configure Level 2 processing.

        ``synthetic_level0_5_bypass`` is an interface-fixture mode. It accepts
        only Level 0.5 data already expressed as a DN/time rate and records the
        bypass prominently in the output header.
        """
        if config is None:
            default_config_path = (
                Path(__file__).parent / "config_files" / "config_default.ini"
            )
            self.config = config_parser.Config(str(default_config_path))
        else:
            self.config = config

        self.diffraction_psf_file = diffraction_psf_file
        self.scatter_psf_file = scatter_psf_file
        self.spec_file = spec_file
        self.resp_file = resp_file
        self.correction_factor = correction_factor
        if input_kind not in self.INPUT_KINDS:
            raise ValueError(
                f"input_kind must be one of {self.INPUT_KINDS}, got {input_kind!r}"
            )
        self.input_kind = input_kind
        self.product_status = str(product_status).strip().upper()
        if not self.product_status:
            raise ValueError("product_status cannot be empty")
        self.overwrite = bool(overwrite)
        self.flux_ratio_tolerance = float(flux_ratio_tolerance)
        if not math.isfinite(self.flux_ratio_tolerance):
            raise ValueError("flux_ratio_tolerance must be finite")
        if not 0 <= self.flux_ratio_tolerance < 1:
            raise ValueError("flux_ratio_tolerance must be in the range [0, 1)")
        self.correction_factor = float(self.correction_factor)
        if not math.isfinite(self.correction_factor):
            raise ValueError("correction_factor must be finite")
        if metadata_definition_file is None:
            metadata_filename = getattr(
                self.config,
                "base_metadata_filename",
                None,
            )
            if not metadata_filename:
                raise ValueError(
                    "A FITS metadata definition is required for Level 2 validation"
                )
            metadata_definition_file = data_path("metadata", metadata_filename)
        self.metadata_definition_file = Path(metadata_definition_file).expanduser()
        if not self.metadata_definition_file.is_file():
            raise FileNotFoundError(
                "FITS metadata definition file not found: "
                f"{self.metadata_definition_file}"
            )

    def run(self, input_path, output_path=None):
        """Process one FITS image or every FITS image directly in a directory."""
        input_path = Path(input_path)
        if input_path.is_dir():
            fits_files = sorted(input_path.glob("*.fits"))
        elif input_path.is_file():
            fits_files = [input_path]
        else:
            raise ValueError(f"Input path does not exist: {input_path}")

        if not fits_files:
            cprint(f"No FITS files found in {input_path}", "yellow")
            return []

        self._validate_calibration_inputs()
        if output_path is None:
            output_path = (
                data_path("synthetic", "level2")
                if "synthetic" in str(input_path)
                else data_path("level2")
            )
        else:
            output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        calibration_manifest, manifest_created = self._write_calibration_manifest(
            output_path
        )

        cprint(f"Processing {len(fits_files)} FITS file(s) from {input_path}", "green")
        cprint(f"Output directory: {output_path}", "green")
        print()

        outputs = []
        try:
            for fits_file in fits_files:
                outputs.append(
                    self._process_single_file(
                        fits_file,
                        output_path,
                        calibration_manifest.name,
                    )
                )
        except Exception:
            # Do not leave an unreferenced calibration manifest when a
            # single-product run fails before publishing any FITS product.
            if manifest_created and not outputs:
                calibration_manifest.unlink(missing_ok=True)
            raise
        return outputs

    @staticmethod
    def _calibration_file_record(path):
        path = Path(path).expanduser().resolve()
        return {
            "filename": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    def _write_calibration_manifest(self, output_path):
        """Write a stable, machine-readable calibration-set manifest."""
        payload = {
            "schema_version": 1,
            "product_stage": "level2_psf_deconvolution",
            "correction_factor": float(self.correction_factor),
            "components": {
                "diffraction_psf": self._calibration_file_record(
                    self.diffraction_psf_file
                ),
                "scatter_psf": self._calibration_file_record(
                    self.scatter_psf_file
                ),
                "spectrum": self._calibration_file_record(self.spec_file),
                "spectral_response": self._calibration_file_record(self.resp_file),
            },
        }
        encoded = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ) + "\n"
        identifier = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
        destination = (
            Path(output_path) / f"level2_psf_calibration_{identifier}.json"
        )
        if destination.exists() and destination.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(
                f"Calibration manifest identifier collision at {destination}"
            )
        if destination.exists():
            return destination, False

        temporary = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination, True

    def _validate_calibration_inputs(self):
        calibration_inputs = {
            "diffraction PSF": self.diffraction_psf_file,
            "scatter PSF": self.scatter_psf_file,
            "spectrum": self.spec_file,
            "spectral response": self.resp_file,
        }
        for label, value in calibration_inputs.items():
            if value is None:
                raise ValueError(f"Missing required {label} file")
            path = Path(value).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"{label.capitalize()} file not found: {path}")

    @staticmethod
    def _normalized_level(value):
        if value is None:
            return None
        return str(value).strip().upper().removeprefix("LEVEL").removeprefix("L")

    @staticmethod
    def _is_rate_unit(value):
        normalized = str(value or "").strip().lower().replace(" ", "")
        return normalized in {"dn/s", "dn/sec", "dn/second", "dn*s-1", "dn*s^-1"}

    def _load_input_data(self, input_file):
        """Load the declared upstream array without silently recalibrating it."""
        with fits.open(input_file, memmap=False, checksum=True) as hdul:
            hdul.verify("exception")
            if hdul[0].data is None:
                raise ValueError(f"Primary HDU contains no image data: {input_file}")
            for checksum_name in ("CHECKSUM", "DATASUM"):
                if checksum_name not in hdul[0].header:
                    raise ValueError(
                        f"Input FITS is missing required {checksum_name}: {input_file}"
                    )
            if (
                hdul[0].verify_checksum() != 1
                or hdul[0].verify_datasum() != 1
            ):
                raise ValueError(f"Input FITS checksums did not validate: {input_file}")
            data = np.asarray(hdul[0].data, dtype=np.float64)
            header = hdul[0].header
            level = self._normalized_level(header.get("LEVEL"))
            time_system = str(header.get("TIMESYS", "")).strip().upper()
            if time_system != "UTC":
                raise ValueError(
                    "SunCET products use UTC; input FITS must declare "
                    f"TIMESYS='UTC', got {header.get('TIMESYS')!r} in {input_file}"
                )

            if self.input_kind == self.INPUT_KIND_LEVEL1:
                if level not in {"1", "1.0"}:
                    raise ValueError(
                        "Level 2 production input must declare LEVEL=1; "
                        f"{input_file} declares {header.get('LEVEL')!r}. "
                        "Use synthetic_level0_5_bypass only for an explicitly "
                        "provisional synthetic interface fixture."
                    )
            else:
                if level not in {"0.5", ".5"}:
                    raise ValueError(
                        "The synthetic bypass requires an input declaring LEVEL=0.5; "
                        f"{input_file} declares {header.get('LEVEL')!r}"
                    )
                if not self._is_rate_unit(header.get("BUNIT")):
                    raise ValueError(
                        "The synthetic bypass accepts only data already normalized "
                        f"to a DN/time rate; BUNIT={header.get('BUNIT')!r}"
                    )

        if data.ndim != 2:
            raise ValueError(
                f"Level 2 requires a two-dimensional primary image, got {data.shape}"
            )
        if data.shape != self.EXPECTED_IMAGE_SHAPE:
            raise ValueError(
                "Level 2 deconvolution is calibrated for detector shape "
                f"{self.EXPECTED_IMAGE_SHAPE}, got {data.shape}"
            )
        if not np.all(np.isfinite(data)):
            raise ValueError(f"Input image contains non-finite values: {input_file}")
        return data

    def _process_single_file(
        self,
        input_file,
        output_dir,
        calibration_manifest_name,
    ):
        input_file = Path(input_file)
        print(f"Processing: {input_file.name}")

        output_file = (
            Path(output_dir)
            / f"{input_file.stem}_level2_v{self.config.version}.fits"
        )
        with self._reserve_output(output_file):
            l1_data = self._load_input_data(input_file)
            decon_data = suncet_deconv.apply_deconv(
                l1_data,
                self.diffraction_psf_file,
                self.scatter_psf_file,
                self.resp_file,
                self.spec_file,
                correction_factor=self.correction_factor,
            )
            flux_ratio = self._validate_deconvolution(l1_data, decon_data)
            self._save_fits(
                decon_data,
                input_file,
                output_file,
                calibration_manifest_name,
                flux_ratio,
            )
        cprint(f"  Saved to: {output_file.name}", "green")
        print()
        return output_file

    @contextmanager
    def _reserve_output(self, output_file):
        """Serialize writers targeting the same product pathname."""
        output_file = Path(output_file)
        lock_path = output_file.with_name(f".{output_file.name}.lock")
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise RuntimeError(
                "Another Level 2 writer is active, or a stale lock needs "
                f"operator review: {lock_path}"
            ) from exc

        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(f"pid={os.getpid()}\n")
                stream.flush()
                os.fsync(stream.fileno())
            if output_file.exists() and not self.overwrite:
                raise FileExistsError(
                    f"Refusing to replace existing Level 2 product: {output_file}. "
                    "Choose a new output directory/version or pass overwrite=True "
                    "explicitly."
                )
            yield
        finally:
            lock_path.unlink(missing_ok=True)

    def _validate_deconvolution(self, input_data, output_data):
        output_data = np.asarray(output_data)
        if output_data.shape != input_data.shape:
            raise ValueError(
                "Deconvolution changed the image shape from "
                f"{input_data.shape} to {output_data.shape}; inherited WCS would "
                "no longer describe the output"
            )
        if not np.all(np.isfinite(output_data)):
            raise ValueError("Deconvolved image contains non-finite values")

        input_flux = float(np.sum(input_data, dtype=np.float64))
        output_flux = float(np.sum(output_data, dtype=np.float64))
        if not math.isfinite(input_flux) or input_flux <= 0:
            raise ValueError(
                "Cannot validate deconvolution flux conservation because the "
                f"input sum is {input_flux!r}"
            )
        flux_ratio = output_flux / input_flux
        if not math.isfinite(flux_ratio):
            raise ValueError(
                f"Deconvolution produced a non-finite flux ratio: {flux_ratio!r}"
            )
        if abs(flux_ratio - 1.0) > self.flux_ratio_tolerance:
            raise ValueError(
                "Deconvolution flux-conservation gate failed: output/input "
                f"ratio {flux_ratio:.8g} is outside "
                f"1 +/- {self.flux_ratio_tolerance:.8g}"
            )
        return flux_ratio

    @staticmethod
    def _elapsed_seconds_from_header(header):
        try:
            start = datetime.fromisoformat(
                str(header["DATE-BEG"]).replace("Z", "+00:00")
            )
            end = datetime.fromisoformat(
                str(header["DATE-END"]).replace("Z", "+00:00")
            )
        except (KeyError, TypeError, ValueError):
            return None
        return (end - start).total_seconds()

    @staticmethod
    def _update_image_statistics(header, data):
        finite = np.isfinite(data)
        if not np.all(finite):
            raise ValueError("Deconvolved image contains non-finite values")
        values = np.asarray(data[finite], dtype=np.float64)
        nonzero_values = values[values != 0]
        if nonzero_values.size == 0:
            raise ValueError("Deconvolved image contains no non-zero pixels")
        header["NPIXBAD"] = (int(data.size - values.size), "Non-finite output pixels")
        header["DATAZER"] = (
            int(np.count_nonzero(values == 0)),
            "Zero-valued output pixels",
        )
        header["DATAAVG"] = (float(np.mean(values)), "Mean of output pixels")
        header["DATAMDN"] = (float(np.median(values)), "Median of output pixels")
        header["DATASIG"] = (
            float(np.std(nonzero_values)),
            "Std. deviation of non-zero output pixels",
        )
        header["DATAMIN"] = (float(np.min(values)), "Minimum output value")
        header["DATAMAX"] = (float(np.max(values)), "Maximum output value")
        for percentile in (1, 10, 25, 50, 75, 90, 95, 98, 99):
            header[f"DATAP{percentile:02d}"] = (
                float(np.percentile(nonzero_values, percentile)),
                f"Non-zero output {percentile}th percentile",
            )

    @staticmethod
    def _coerce_fits_boolean(value, key):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        normalized = str(value).strip().upper()
        if normalized in {"1", "T", "TRUE", "ON", "YES"}:
            return True
        if normalized in {"0", "F", "FALSE", "OFF", "NO"}:
            return False
        raise ValueError(f"Cannot normalize {key}={value!r} to a FITS boolean")

    @staticmethod
    def _normalize_inherited_metadata_types(header):
        """Normalize unambiguous legacy encodings to the metadata contract."""
        if "TYPECODE" in header:
            header["TYPECODE"] = (
                str(header["TYPECODE"]),
                header.comments["TYPECODE"],
            )

        for key in ("INT_LED", "EXT_LED"):
            if key not in header or isinstance(header[key], (bool, np.bool_)):
                continue
            value = Level2._coerce_fits_boolean(header[key], key)
            header[key] = (value, header.comments[key])

        if "DATASAT" in header:
            header["DATASAT"] = (
                int(header["DATASAT"]),
                "Saturated input pixels before deconvolution",
            )
        if "DSATVAL" in header:
            header["DSATVAL"] = (
                float(header["DSATVAL"]),
                "Input saturation value before deconvolution",
            )

        header["TIMESYS"] = ("UTC", "Principal time system: UTC")
        utc_time_comments = {
            "DATE-BEG": "Beginning observation time in UTC ISO 8601 format",
            "DATE-OBS": "Reference observation time in UTC ISO 8601 format",
            "DATE-END": "Ending observation time in UTC ISO 8601 format",
        }
        for key, comment in utc_time_comments.items():
            if key in header:
                header[key] = (header[key], comment)

    def _save_fits(
        self,
        data,
        input_file,
        output_file,
        calibration_manifest_name,
        flux_ratio,
    ):
        input_file = Path(input_file)
        output_file = Path(output_file)
        with fits.open(input_file) as hdul:
            header = hdul[0].header.copy()

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Deconvolved output must be two-dimensional, got {data.shape}")
        for key in ("CHECKSUM", "DATASUM"):
            header.remove(key, ignore_missing=True, remove_all=True)
        self._normalize_inherited_metadata_types(header)

        pipeline_version = str(self.config.version).removeprefix("v")
        header["TITLE"] = ("SunCET Level 2 Image", "A descriptive name of the data")
        header["FILENAME"] = (output_file.name, "Filename of this product")
        header["LEVEL"] = (2, "Data processing level number")
        header["PIPEVRSN"] = (f"v{pipeline_version}", "Processing pipeline version")
        header["DATE"] = (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "File generation time in UTC",
        )
        header["ORIGIN"] = ("JHU/APL", "Institution responsible for this product")
        header["PROCSTAT"] = (self.product_status, "Processing maturity")
        header["DECONV"] = (True, "PSF deconvolution applied")
        header["DECONVCF"] = (
            self.correction_factor,
            "Deconvolution correction factor",
        )
        header["DIFPSF"] = (
            Path(self.diffraction_psf_file).name,
            "Diffraction PSF input",
        )
        header["SCATPSF"] = (
            Path(self.scatter_psf_file).name,
            "Scatter PSF input",
        )
        header["SPECFIL"] = (Path(self.spec_file).name, "Spectrum input")
        header["RESPFIL"] = (
            Path(self.resp_file).name,
            "Spectral response input",
        )
        header["CALPSF"] = (
            calibration_manifest_name,
            "PSF calibration manifest filename",
        )
        calibration_identifier = Path(calibration_manifest_name).stem.removeprefix(
            "level2_psf_calibration_"
        )
        header["CALID"] = (calibration_identifier, "PSF calibration-set identity")
        synthetic_bypass = (
            self.input_kind == self.INPUT_KIND_SYNTHETIC_LEVEL0_5_BYPASS
        )
        upstream_synthetic = self._coerce_fits_boolean(
            header.get("SYNTHET", False),
            "SYNTHET",
        )
        header["SYNTHET"] = (
            synthetic_bypass or upstream_synthetic,
            "Synthetic input product",
        )
        header["L1BYPASS"] = (
            synthetic_bypass,
            "Level 1 stage explicitly bypassed",
        )
        header["L0PARENT"] = (input_file.name, "Immediate parent product filename")
        header["IMAGEW"] = (int(data.shape[1]), "Image width in pixels")
        header["IMAGEH"] = (int(data.shape[0]), "Image height in pixels")

        try:
            header["TELAPSE"] = (
                float(header["TELAPSE"]),
                "Elapsed observation time in seconds",
            )
        except (KeyError, TypeError, ValueError):
            elapsed = self._elapsed_seconds_from_header(header)
            if elapsed is not None:
                header["TELAPSE"] = (
                    elapsed,
                    "Elapsed observation time in seconds",
                )

        if synthetic_bypass:
            header["DOI"] = (
                "NOT_ASSIGNED",
                "No DOI assigned to synthetic development fixture",
            )
            header["OBSTYPE"] = (
                "SYNTHETIC DEVELOPMENT FIXTURE",
                "Synthetic observation designation",
            )
            header["FILE_RAW"] = (
                "NOT_APPLICABLE_SYNTHETIC",
                "No raw telemetry parent for synthetic fixture",
            )
            for key in ("CALDARK", "CALFLAT", "CALVI", "CALBPM", "CALSL"):
                if key in header:
                    header[key] = (
                        "NOT_APPLIED_SYNTHETIC_BYPASS",
                        "Level 1 calibration not applied",
                    )
            header["EXP_MASK"] = (
                "NOT_APPLIED_SYNTHETIC_BYPASS",
                "Level 1 exposure mask not applied",
            )

        header["DECONRAT"] = (float(flux_ratio), "Output/input summed-flux ratio")
        header["DECONTOL"] = (
            float(self.flux_ratio_tolerance),
            "Allowed absolute deviation of DECONRAT from unity",
        )

        self._update_image_statistics(header, data)
        header.add_history("Applied diffraction and scatter PSF deconvolution")
        if synthetic_bypass:
            header.add_history(
                "DEVELOPMENT FIXTURE: synthetic Level 0.5 rate data used directly; "
                "Level 1 calibration was not applied"
            )
        header.add_history(
            f"Diffraction PSF: {Path(self.diffraction_psf_file).name}"
        )
        header.add_history(f"Scatter PSF: {Path(self.scatter_psf_file).name}")
        header.add_history(f"Spectrum: {Path(self.spec_file).name}")
        header.add_history(f"Spectral response: {Path(self.resp_file).name}")
        header.add_history(f"Calibration manifest: {calibration_manifest_name}")

        temporary = output_file.with_name(
            f".{output_file.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            fits.PrimaryHDU(data=data, header=header).writeto(
                temporary,
                overwrite=True,
                checksum=True,
            )
            with fits.open(temporary, checksum=True) as hdul:
                hdul.verify("exception")
                if hdul[0].verify_checksum() != 1 or hdul[0].verify_datasum() != 1:
                    raise ValueError("Generated Level 2 FITS checksums did not validate")
                metadata_managers.validate_fits_header(
                    hdul[0].header,
                    self.metadata_definition_file,
                    2,
                    float_output_statistics=("DATAMIN", "DATAMAX"),
                )
            if output_file.exists() and not self.overwrite:
                raise FileExistsError(
                    f"Refusing to replace existing Level 2 product: {output_file}"
                )
            os.replace(temporary, output_file)
        finally:
            temporary.unlink(missing_ok=True)


def _get_parser():
    parser = argparse.ArgumentParser(
        description="Apply deconvolution to SunCET Level 1 data to produce Level 2"
    )
    parser.add_argument(
        "-i",
        "--input-path",
        required=True,
        help="Directory containing Level 1 FITS files, or one FITS file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing product with the same filename",
    )
    parser.add_argument(
        "--flux-ratio-tolerance",
        type=float,
        default=Level2.DEFAULT_FLUX_RATIO_TOLERANCE,
        help=(
            "Maximum allowed absolute deviation of summed output/input flux "
            "ratio from unity (default: 0.05)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-path",
        default=None,
        help="Output directory (default: $suncet_data/level2 or synthetic/level2)",
    )
    parser.add_argument("-c", "--config-file", default=None)
    parser.add_argument("--diffraction-psf-file", required=True)
    parser.add_argument("--scatter-psf-file", required=True)
    parser.add_argument("--spec-file", required=True)
    parser.add_argument("--resp-file", required=True)
    parser.add_argument("--correction-factor", type=float, default=0.4)
    parser.add_argument(
        "--input-kind",
        choices=Level2.INPUT_KINDS,
        default=Level2.INPUT_KIND_LEVEL1,
        help=(
            "Upstream product contract. The synthetic Level 0.5 bypass is "
            "development-only and must be selected explicitly."
        ),
    )
    parser.add_argument(
        "--product-status",
        default="PROVISIONAL",
        help="Value written to FITS PROCSTAT (default: PROVISIONAL)",
    )
    parser.add_argument(
        "--metadata-definition-file",
        default=None,
        help=(
            "Authoritative FITS metadata CSV (default: the configured filename "
            "below $suncet_data/metadata)"
        ),
    )
    return parser


def main(argv=None):
    args = _get_parser().parse_args(argv)
    config_path = (
        Path(args.config_file).expanduser().resolve()
        if args.config_file
        else Path(__file__).resolve().parent / "config_files" / "config_default.ini"
    )
    config = config_parser.Config(str(config_path))
    level2 = Level2(
        config=config,
        diffraction_psf_file=args.diffraction_psf_file,
        scatter_psf_file=args.scatter_psf_file,
        spec_file=args.spec_file,
        resp_file=args.resp_file,
        correction_factor=args.correction_factor,
        input_kind=args.input_kind,
        product_status=args.product_status,
        metadata_definition_file=args.metadata_definition_file,
        overwrite=args.overwrite,
        flux_ratio_tolerance=args.flux_ratio_tolerance,
    )

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path is not None
        else (
            data_path("synthetic", "level2")
            if "synthetic" in str(input_path)
            else data_path("level2")
        )
    )
    input_files = (
        sorted(input_path.glob("*.fits"))
        if input_path.is_dir()
        else [input_path]
    )
    input_files.extend(
        Path(path).expanduser().resolve()
        for path in (
            args.diffraction_psf_file,
            args.scatter_psf_file,
            args.spec_file,
            args.resp_file,
        )
    )
    input_files.append(level2.metadata_definition_file)
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    provenance = ProcessingRunProvenance(
        data_root=output_path,
        run_kind="make_level2",
        config_path=config_path,
        resolved_config=resolved_config_snapshot(config, output_path),
        arguments=vars(args),
        argv=[str(Path(__file__).resolve()), *argv_list],
        repository_hint=Path(__file__).resolve().parents[1],
        public=True,
    )
    with provenance:
        provenance.record_inputs(input_files)
        level2.run(input_path, output_path)


if __name__ == "__main__":
    main()

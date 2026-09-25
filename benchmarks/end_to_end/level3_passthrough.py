"""Create explicitly benchmark-only Level 3 pass-through FITS products.

This adapter exists only for end-to-end energy characterization while the
science Level 3 geometric and special dark-correction algorithms are not yet
defined.  It performs no numerical image operation: the Level 2 array and WCS
are preserved, while the output is marked prominently as a provisional Level
3 pass-through product.

Do not use this module as the production implementation of Level 3.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sys
import uuid

from astropy.io import fits
import numpy as np

# Permit direct execution from a source checkout as well as module execution
# from an installed package environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from suncet_processing_pipeline import metadata_managers
from suncet_processing_pipeline.run_provenance import ProcessingRunProvenance


_LEVEL2_VALUES = {"2", "2.0", "L2", "LEVEL2"}
_WCS_CARD_PATTERN = re.compile(
    r"^(?:"
    r"WCSAXES|WCSNAME|CTYPE\d+|CRPIX\d+|CRVAL\d+|CDELT\d+|CUNIT\d+|"
    r"CROTA\d+|CD\d+_\d+|PC\d+_\d+|PV\d+_\d+|PS\d+_\d+|"
    r"LONPOLE|LATPOLE|RADESYS|EQUINOX|"
    r"A_ORDER|B_ORDER|AP_ORDER|BP_ORDER|A_\d+_\d+|B_\d+_\d+|"
    r"AP_\d+_\d+|BP_\d+_\d+"
    r")$"
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _normalized_level(value: object) -> str:
    return str(value or "").strip().upper().replace(" ", "")


def _wcs_cards(header: fits.Header) -> dict[str, object]:
    """Return the image-WCS cards whose values must survive unchanged."""
    return {
        key: header[key]
        for key in header
        if key and _WCS_CARD_PATTERN.fullmatch(key)
    }


def _output_name(input_path: Path) -> str:
    """Derive an unambiguous Level 3 benchmark filename from a Level 2 name."""
    stem = input_path.stem
    match = re.fullmatch(r"(.+)_level2(?:_(.+))?", stem, flags=re.IGNORECASE)
    if match:
        prefix, suffix = match.groups()
        suffix_text = f"_{suffix}" if suffix else ""
        return f"{prefix}_level3_benchmark_passthrough{suffix_text}.fits"
    return f"{stem}_level3_benchmark_passthrough.fits"


class BenchmarkLevel3PassThrough:
    """Copy validated Level 2 FITS images into an honest Level 3 test boundary."""

    def __init__(
        self,
        *,
        metadata_definition_file: str | Path,
        overwrite: bool = False,
        product_status: str = "PROVISIONAL",
        generated_at=None,
    ) -> None:
        self.metadata_definition_file = Path(
            metadata_definition_file
        ).expanduser().resolve()
        if not self.metadata_definition_file.is_file():
            raise FileNotFoundError(
                "FITS metadata definition file not found: "
                f"{self.metadata_definition_file}"
            )
        self.overwrite = bool(overwrite)
        self.product_status = str(product_status).strip().upper()
        if not self.product_status:
            raise ValueError("product_status cannot be empty")
        self._generated_at = generated_at or _utc_now

    def run(
        self,
        input_directory: str | Path,
        output_directory: str | Path,
    ) -> list[Path]:
        """Process every direct ``*.fits`` child in deterministic name order."""
        input_directory = Path(input_directory).expanduser().resolve()
        output_directory = Path(output_directory).expanduser().resolve()
        if not input_directory.is_dir():
            raise NotADirectoryError(
                f"Level 2 input directory does not exist: {input_directory}"
            )
        if input_directory == output_directory:
            raise ValueError("Level 2 input and Level 3 output directories must differ")

        input_files = sorted(input_directory.glob("*.fits"), key=lambda path: path.name)
        if not input_files:
            raise ValueError(f"No FITS files found in {input_directory}")
        output_directory.mkdir(parents=True, exist_ok=True)

        return [
            self._process_one(input_file, output_directory)
            for input_file in input_files
        ]

    def _load_level2(self, input_file: Path) -> tuple[np.ndarray, fits.Header]:
        with fits.open(input_file, memmap=False, checksum=True) as hdul:
            hdul.verify("exception")
            if len(hdul) != 1:
                raise ValueError(
                    "Benchmark Level 3 accepts one-primary-HDU Level 2 products; "
                    f"{input_file} contains {len(hdul)} HDUs"
                )
            primary = hdul[0]
            if primary.data is None or primary.data.ndim != 2:
                raise ValueError(
                    f"Level 2 primary HDU must contain one 2-D image: {input_file}"
                )
            for key in ("CHECKSUM", "DATASUM"):
                if key not in primary.header:
                    raise ValueError(
                        f"Input FITS is missing required {key}: {input_file}"
                    )
            if primary.verify_checksum() != 1 or primary.verify_datasum() != 1:
                raise ValueError(f"Input FITS checksums did not validate: {input_file}")
            if _normalized_level(primary.header.get("LEVEL")) not in _LEVEL2_VALUES:
                raise ValueError(
                    "Benchmark Level 3 input must declare LEVEL=2; "
                    f"{input_file} declares {primary.header.get('LEVEL')!r}"
                )
            if primary.header.get("DECONV") is not True:
                raise ValueError(
                    "Benchmark Level 3 expects a PSF-deconvolved Level 2 input "
                    f"with DECONV=T: {input_file}"
                )
            if str(primary.header.get("TIMESYS", "")).strip().upper() != "UTC":
                raise ValueError(
                    f"Level 2 input must declare TIMESYS='UTC': {input_file}"
                )
            metadata_managers.validate_fits_header(
                primary.header,
                self.metadata_definition_file,
                2,
                float_output_statistics=("DATAMIN", "DATAMAX"),
            )
            return np.array(primary.data, copy=True), primary.header.copy()

    @contextmanager
    def _reserve_output(self, output_file: Path):
        lock_path = output_file.with_name(f".{output_file.name}.lock")
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise RuntimeError(
                "Another benchmark Level 3 writer is active, or a stale lock "
                f"requires review: {lock_path}"
            ) from exc
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(f"pid={os.getpid()}\n")
                stream.flush()
                os.fsync(stream.fileno())
            if output_file.exists() and not self.overwrite:
                raise FileExistsError(
                    "Refusing to replace existing benchmark Level 3 product: "
                    f"{output_file}"
                )
            yield
        finally:
            lock_path.unlink(missing_ok=True)

    def _process_one(self, input_file: Path, output_directory: Path) -> Path:
        output_file = output_directory / _output_name(input_file)
        with self._reserve_output(output_file):
            data, header = self._load_level2(input_file)
            source_wcs = _wcs_cards(header)
            for key in ("CHECKSUM", "DATASUM"):
                header.remove(key, ignore_missing=True, remove_all=True)

            header["TITLE"] = (
                "SunCET Level 3 Benchmark Pass-Through Image",
                "Product",
            )
            # These benchmark filenames can be long enough that adding a FITS
            # comment would force Astropy to truncate it.
            header["FILENAME"] = (output_file.name, "")
            header["LEVEL"] = (3, "Data processing level number")
            header["DATE"] = (
                str(self._generated_at()),
                "File generation time in UTC",
            )
            header["PROCSTAT"] = (self.product_status, "Processing maturity")
            header["L2PARENT"] = input_file.name
            header["BMRKONLY"] = (True, "Benchmark-only Level 3 adapter")
            header["L3PASS"] = (True, "Level 3 array passed through unchanged")
            header["L3GEOM"] = (False, "Level 3 geometric correction applied")
            header["L3DARK"] = (False, "Level 3 special dark correction applied")
            header["L3CORR"] = (
                "NONE",
                "Level 3 corrections applied by this adapter",
            )
            header.add_history(
                "BENCHMARK ONLY: promoted Level 2 to Level 3 without changing "
                "image values or WCS"
            )
            header.add_history(
                "No Level 3 fine rotation, solar-north alignment, recentering, "
                "resampling, or special dark correction was applied"
            )

            temporary = output_file.with_name(
                f".{output_file.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                fits.PrimaryHDU(data=data, header=header).writeto(
                    temporary,
                    overwrite=True,
                    checksum=True,
                )
                self._validate_output(
                    temporary,
                    expected_data=data,
                    expected_wcs=source_wcs,
                    expected_parent=input_file.name,
                )
                if output_file.exists() and not self.overwrite:
                    raise FileExistsError(
                        "Refusing to replace existing benchmark Level 3 product: "
                        f"{output_file}"
                    )
                os.replace(temporary, output_file)
            finally:
                temporary.unlink(missing_ok=True)
        return output_file

    def _validate_output(
        self,
        output_file: Path,
        *,
        expected_data: np.ndarray,
        expected_wcs: dict[str, object],
        expected_parent: str,
    ) -> None:
        with fits.open(output_file, memmap=False, checksum=True) as hdul:
            hdul.verify("exception")
            primary = hdul[0]
            if primary.verify_checksum() != 1 or primary.verify_datasum() != 1:
                raise ValueError("Generated Level 3 FITS checksums did not validate")
            if not np.array_equal(primary.data, expected_data, equal_nan=True):
                raise ValueError("Benchmark Level 3 pass-through changed image values")
            if primary.data.dtype != expected_data.dtype:
                raise ValueError(
                    "Benchmark Level 3 pass-through changed image dtype from "
                    f"{expected_data.dtype} to {primary.data.dtype}"
                )
            output_wcs = _wcs_cards(primary.header)
            if output_wcs != expected_wcs:
                raise ValueError("Benchmark Level 3 pass-through changed image WCS")
            if primary.header.get("L2PARENT") != expected_parent:
                raise ValueError("Generated Level 3 parent provenance is incorrect")
            if not (
                primary.header.get("BMRKONLY") is True
                and primary.header.get("L3PASS") is True
                and primary.header.get("L3GEOM") is False
                and primary.header.get("L3DARK") is False
                and primary.header.get("L3CORR") == "NONE"
            ):
                raise ValueError("Generated Level 3 benchmark status is ambiguous")
            metadata_managers.validate_fits_header(
                primary.header,
                self.metadata_definition_file,
                3,
                float_output_statistics=("DATAMIN", "DATAMAX"),
            )


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create benchmark-only Level 3 pass-through products from a "
            "directory of Level 2 FITS files"
        )
    )
    parser.add_argument("--input-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--metadata-definition-file", required=True)
    parser.add_argument(
        "--product-status",
        default="PROVISIONAL",
        help="FITS PROCSTAT value (default: PROVISIONAL)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing outputs with matching names",
    )
    return parser


def main(argv=None) -> int:
    args = _get_parser().parse_args(argv)
    input_directory = Path(args.input_directory).expanduser().resolve()
    output_directory = Path(args.output_directory).expanduser().resolve()
    processor = BenchmarkLevel3PassThrough(
        metadata_definition_file=args.metadata_definition_file,
        overwrite=args.overwrite,
        product_status=args.product_status,
    )
    input_files = sorted(input_directory.glob("*.fits"), key=lambda path: path.name)
    provenance = ProcessingRunProvenance(
        data_root=output_directory,
        run_kind="benchmark_level3_passthrough",
        arguments=vars(args),
        argv=[str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
        repository_hint=Path(__file__).resolve().parents[2],
        public=True,
    )
    with provenance:
        provenance.record_inputs([*input_files, processor.metadata_definition_file])
        outputs = processor.run(input_directory, output_directory)
    print(f"Created {len(outputs)} benchmark-only Level 3 pass-through products")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

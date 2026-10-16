"""
This is the code to make the Level 1 data product. 
"""
import os
import argparse
import json
from glob import glob
import copy
from pathlib import Path
from astropy.io import fits
import numpy as np
from pykdtree.kdtree import KDTree
from suncet_processing_pipeline.level1_orientation import (
    CoarseOrientationDecision,
    apply_coarse_solar_north_rotation,
    load_apid46_attitude_samples,
    plan_coarse_solar_north_rotation,
)
from suncet_processing_pipeline import config_parser
from suncet_processing_pipeline.spacecraft_time import combine_spacecraft_time_seconds


CSIE_GEOMETRY_FIELDS = (
    "csie_115_hk_icm_rad_sqr",
    "csie_115_hk_icm_cent_col",
    "csie_115_hk_icm_cent_row",
)


def _metadata_flag(value):
    """Interpret CTDB/FITS boolean states without treating ``"DIS"`` as true."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "ena", "enabled"}:
        return True
    if normalized in {"", "0", "false", "no", "dis", "disabled", "none"}:
        return False
    raise ValueError(f"unrecognized metadata boolean state: {value!r}")


def effective_exposure_times_from_metadata(metadata):
    """Return inner/outer radiometric exposures in seconds.

    ``TELAPSE`` is deliberately not used: it is the wall-clock span from the
    first integration start to the final integration end, including gaps. The
    values here describe the integrated signal after FSW stacking, optional
    particle rejection, and right-shift normalization.
    """
    def value(name, default=None):
        if hasattr(metadata, "get"):
            observed = metadata.get(name, default)
        else:
            observed = getattr(metadata, name, default)
        return default if observed is None else observed

    result = {}
    for component, suffix in (("inner", "I"), ("outer", "O")):
        direct = value(f"EFFEXP{suffix}")
        if direct is not None:
            result[component] = float(direct)
            continue
        integration = value(f"INTTIME{suffix}")
        count = value(f"NSTACK{suffix}")
        normalization = value(f"STKNORM{suffix}")
        if integration is None or count is None or normalization is None:
            continue
        particle_filtering = _metadata_flag(value(f"PIXFILT{suffix}", False))
        contributing = int(count) - (1 if particle_filtering else 0)
        if contributing < 0 or float(normalization) <= 0:
            raise ValueError(f"invalid {component} exposure metadata")
        result[component] = (
            contributing * float(integration) / float(normalization)
        )
    if not result:
        raise ValueError("no effective inner/outer exposure metadata is available")
    return result


def create_exposure_time_mask(image_data, exposures, *, inner_mask=None):
    """Create the effective-exposure mask used for DN/s normalization."""
    image = np.asanyarray(image_data)
    inner = exposures.get("inner")
    outer = exposures.get("outer")
    if image.ndim == 3:
        ordered = [value for value in (inner, outer) if value is not None]
        if len(ordered) != image.shape[0]:
            raise ValueError("effective exposures do not match image sections")
        return np.broadcast_to(
            np.asarray(ordered, dtype=float)[:, None, None], image.shape
        ).copy()
    if image.ndim != 2:
        raise ValueError("image data must be a 2D composite or 3D section stack")
    if inner is None:
        return np.full(image.shape, outer, dtype=float)
    if outer is None or inner == outer:
        return np.full(image.shape, inner, dtype=float)
    if inner_mask is None:
        raise ValueError(
            "a 2D composite with different inner/outer exposures requires an inner_mask"
        )
    mask = np.asanyarray(inner_mask, dtype=bool)
    if mask.shape != image.shape:
        raise ValueError("inner_mask shape does not match image data")
    return np.where(mask, float(inner), float(outer))


def create_circular_inner_mask(
    image_shape,
    *,
    center_column,
    center_row,
    radius_pixels,
):
    """Return a circular composite mask in zero-based array coordinates.

    The boundary is included in the inner region, matching the simulator and
    flight-configuration convention.  Geometry must be supplied explicitly;
    this helper deliberately does not substitute the solar WCS circle for
    missing instrument-composite geometry.
    """
    if len(image_shape) != 2:
        raise ValueError("image_shape must contain exactly two dimensions")
    rows, columns = (int(value) for value in image_shape)
    if rows <= 0 or columns <= 0:
        raise ValueError("image_shape dimensions must be positive")

    center_column = float(center_column)
    center_row = float(center_row)
    radius_pixels = float(radius_pixels)
    if not np.all(np.isfinite((center_column, center_row, radius_pixels))):
        raise ValueError("circle geometry must be finite")
    if radius_pixels < 0:
        raise ValueError("radius_pixels must be non-negative")

    row_grid, column_grid = np.ogrid[:rows, :columns]
    return (
        (column_grid - center_column) ** 2
        + (row_grid - center_row) ** 2
        <= radius_pixels**2
    )


def normalize_to_dn_per_second(image_data, exposures, *, inner_mask=None):
    """Normalize a Level 0.5 DN array by its pixelwise effective exposure."""
    image = np.asanyarray(image_data, dtype=np.float64)
    exposure_mask = create_exposure_time_mask(
        image,
        exposures,
        inner_mask=inner_mask,
    )
    if not np.all(np.isfinite(exposure_mask)):
        raise ValueError("effective exposure mask contains non-finite values")
    if np.any(exposure_mask <= 0):
        raise ValueError("effective exposure mask must be strictly positive")
    return image / exposure_mask


def _metadata_value(metadata, *names, default=None):
    casefolded = None
    if hasattr(metadata, "items"):
        casefolded = {str(key).casefold(): value for key, value in metadata.items()}
    for name in names:
        if hasattr(metadata, "get"):
            value = metadata.get(name)
        else:
            value = getattr(metadata, name, None)
        if value is None and casefolded is not None:
            value = casefolded.get(str(name).casefold())
        if value is not None:
            return value
    return default


def _finite_float(metadata, *names, default=None):
    value = _metadata_value(metadata, *names, default=default)
    if value is None or value == "":
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"metadata value {names[0]} must be finite")
    return result


def create_csie_composite_inner_mask(image_shape, metadata):
    """Build the inner-exposure mask from the APID 537 ICM image coordinates.

    The CSIE ICM operates after FPM binning: APID 537 therefore reports center
    and radius-squared in the processed image's pixel coordinates. For example,
    the 2x2-binned TVAC configuration reports (376, 500), half the unbinned
    (752, 1000) center. Applying APID 538 binning again would be incorrect.
    """
    if len(image_shape) != 2:
        raise ValueError("CSIE composite geometry requires a two-dimensional image")
    rows, columns = (int(value) for value in image_shape)
    radius_squared = _finite_float(
        metadata,
        "ICMRAD2",
        "icm_circle_radius_squared_pixels",
        "icm_circle_radius_squared_detector_pixels",
        "csie_115_hk_icm_rad_sqr",
    )
    center_column = _finite_float(
        metadata,
        "ICMCOL",
        "icm_circle_center_column_pixels",
        "icm_circle_center_column_detector_pixels",
        "csie_115_hk_icm_cent_col",
    )
    center_row = _finite_float(
        metadata,
        "ICMROW",
        "icm_circle_center_row_pixels",
        "icm_circle_center_row_detector_pixels",
        "csie_115_hk_icm_cent_row",
    )
    if radius_squared is None or center_column is None or center_row is None:
        raise ValueError(
            "composite image metadata is missing the APID 537 ICM circle "
            "radius-squared and/or center coordinates"
        )
    if radius_squared < 0:
        raise ValueError("ICM circle radius squared must be non-negative")

    row_grid, column_grid = np.ogrid[:rows, :columns]
    return (
        (column_grid - center_column) ** 2 + (row_grid - center_row) ** 2
        <= radius_squared
    )


def load_csie_composite_geometry(database_path, image_j2000):
    """Read the most recent APID 537 ICM geometry valid for an image."""
    import duckdb

    database = Path(database_path).expanduser()
    if not database.is_file():
        raise FileNotFoundError(database)
    timestamp = float(image_j2000)
    if not np.isfinite(timestamp):
        raise ValueError("image J2000 time must be finite")
    with duckdb.connect(str(database), read_only=True) as connection:
        exists = connection.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            ["apid_0537"],
        ).fetchone()[0]
        if not exists:
            raise LookupError("DuckDB does not contain APID 537 table apid_0537")
        columns = {
            row[0] for row in connection.execute('DESCRIBE "apid_0537"').fetchall()
        }
        required = {"SHCOARSE", "SHFINE", *CSIE_GEOMETRY_FIELDS}
        missing = sorted(required - columns)
        if missing:
            raise LookupError(
                "APID 537 table is missing required field(s): " + ", ".join(missing)
            )
        time_expression = 'CAST("SHCOARSE" AS DOUBLE) + CAST("SHFINE" AS DOUBLE) / 1000.0'
        field_sql = ", ".join(f'"{name}"' for name in CSIE_GEOMETRY_FIELDS)
        row = connection.execute(
            f"SELECT {time_expression} AS sample_time, {field_sql} "
            f'FROM "apid_0537" WHERE {time_expression} <= ? '
            f"ORDER BY {time_expression} DESC LIMIT 1",
            [timestamp],
        ).fetchone()
    if row is None:
        raise LookupError(
            "APID 537 has no ICM geometry sample at or before the image time"
        )
    sample_time, radius_squared, center_column, center_row = row
    if any(value is None for value in (radius_squared, center_column, center_row)):
        raise LookupError("selected APID 537 row has incomplete ICM circle geometry")
    return {
        "ICMRAD2": int(radius_squared),
        "ICMCOL": int(center_column),
        "ICMROW": int(center_row),
        "ICMHKJ2": float(sample_time),
        "ICMAGE": float(timestamp) - float(sample_time),
        "ICMSRC": "APID 537",
    }


def resolve_level1_telemetry_database(config, explicit_path=None):
    """Resolve the canonical all-APID test-phase database, never local analyses."""
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    data_root = Path(config.data_root).expanduser()
    expected = (
        data_root
        / "test_data"
        / f"suncet_telemetry_test_phase_v{config.version_pipeline}.duckdb"
    )
    return expected if expected.is_file() else None


def load_level0_5_image_metadata(source, header):
    """Combine FITS metadata with its optional long-name CSIE JSON companion."""
    source = Path(source)
    metadata = {}
    companion = source.with_name(f"{source.stem}_meta.json")
    if companion.is_file():
        with companion.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        metadata.update(
            {
                key: value
                for key, value in payload.items()
                if key != "metadata_packets"
            }
        )
    metadata.update(dict(header))
    return metadata


def image_j2000_from_metadata(metadata):
    direct = _metadata_value(
        metadata,
        "J2K-AVG",
        "capture_average_j2000",
        "J2K-BEG",
        "capture_start_j2000",
    )
    if direct is not None:
        return float(direct)
    coarse = _metadata_value(metadata, "SHCOARSE", "shcoarse")
    fine = _metadata_value(metadata, "SHFINE", "shfine", default=0)
    if coarse is None:
        return None
    return combine_spacecraft_time_seconds(coarse, fine or 0)


def csie_metadata_indicates_composite(metadata):
    """Return whether CSIE metadata explicitly identifies a two-exposure composite.

    The v1.1.9 CTDB names position-selection state 3
    ``FPM0_INSIDE_FPM1_OUTSIDE``.  Integer mode 3 or that explicit state name is
    accepted.  Missing, OFF, and the currently ambiguous one-sided state names
    are not treated as permission to apply two different exposure times.
    """
    combined = _metadata_value(metadata, "csie_meta_icm_proc_config")
    if combined is not None:
        try:
            return (int(combined) >> 6) & 0x03 == 3
        except (TypeError, ValueError):
            pass

    position = _metadata_value(
        metadata,
        "icm_proc_cfg_posi_select_meta",
        "ICMPROC1",
        "ICMPOS",
    )
    try:
        return int(position) == 3
    except (TypeError, ValueError):
        return str(position).strip().upper() == "FPM0_INSIDE_FPM1_OUTSIDE"


def normalize_level1_image_to_dn_per_second(
    image_data,
    metadata,
    *,
    inner_mask=None,
):
    """Normalize one Level 0.5 image using only metadata-authorized exposure logic.

    A two-dimensional image receives separate inner/outer exposure times only
    when the CSIE metadata explicitly declares the composite state.  Ambiguous
    non-composite metadata with unequal exposures fails closed.  A three-
    dimensional section stack remains explicitly separable and uses one
    exposure per section.
    """
    image = np.asanyarray(image_data)
    exposure_mask = create_level1_exposure_time_mask(
        image,
        metadata,
        inner_mask=inner_mask,
    )
    if not np.all(np.isfinite(exposure_mask)):
        raise ValueError("effective exposure mask contains non-finite values")
    if np.any(exposure_mask <= 0):
        raise ValueError("effective exposure mask must be strictly positive")
    return np.asanyarray(image, dtype=np.float64) / exposure_mask


def create_level1_exposure_time_mask(
    image_data,
    metadata,
    *,
    inner_mask=None,
):
    """Build the exposure mask only after applying the composite-mode gate."""
    image = np.asanyarray(image_data)
    exposures = effective_exposure_times_from_metadata(metadata)
    if image.ndim == 3:
        return create_exposure_time_mask(image, exposures)
    if image.ndim != 2:
        raise ValueError("image data must be a 2D image or 3D section stack")

    if csie_metadata_indicates_composite(metadata):
        if inner_mask is None:
            inner_mask = create_csie_composite_inner_mask(image.shape, metadata)
        return create_exposure_time_mask(
            image,
            exposures,
            inner_mask=inner_mask,
        )

    explicit_single = _metadata_value(
        metadata,
        "EFFEXP",
        "effective_exposure_time",
        "effective_exposure_time_single",
    )
    if explicit_single is not None:
        exposure = float(explicit_single)
    elif len(exposures) == 1:
        exposure = float(next(iter(exposures.values())))
    elif len(exposures) == 2 and exposures.get("inner") == exposures.get("outer"):
        exposure = float(exposures["inner"])
    else:
        raise ValueError(
            "non-composite metadata does not identify one unambiguous effective exposure"
        )
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("effective exposure must be finite and strictly positive")
    return np.full(image.shape, exposure, dtype=float)


class Level1:
    def __init__(self, config):
        self.config = config
        self.metadata = None
        self.cosmic_rays = np.array([])

    def make(
        self,
        level0_5_to_process=None,
        *,
        output_dir=None,
        telemetry_database=None,
        overwrite=True,
    ):
        if level0_5_to_process is None:
            raise ValueError(
                'Provide a Level 0.5 directory, FITS filename, or list of FITS filenames.'
            )
        if isinstance(level0_5_to_process, list):
            filenames = [str(Path(filename).expanduser()) for filename in level0_5_to_process]
        elif isinstance(level0_5_to_process, (str, os.PathLike)):
            input_path = Path(level0_5_to_process).expanduser()
            filenames = (
                sorted(glob(os.path.join(input_path, '*.fits')))
                if input_path.is_dir()
                else [str(input_path)]
            )
        else: 
            raise TypeError(
                'Provide a path-like object or list of FITS filenames.'
            )
        if not filenames:
            raise FileNotFoundError(
                f'No Level 0.5 FITS files found at {level0_5_to_process}'
            )
        print("make_level1 is going to process the following input files:")
        for file in filenames:
            print(f"- {file}")

        database = resolve_level1_telemetry_database(
            self.config, explicit_path=telemetry_database
        )
        if output_dir is None:
            first_parent = Path(filenames[0]).resolve().parent
            destination = (
                first_parent.parent / "level1"
                if first_parent.name == "csie_images"
                else first_parent / "level1"
            )
        else:
            destination = Path(output_dir).expanduser()
        destination.mkdir(parents=True, exist_ok=True)

        outputs = []
        for filename in filenames:
            source = Path(filename).expanduser()
            with fits.open(source, memmap=False) as hdul:
                if hdul[0].data is None:
                    raise ValueError(f"Level 0.5 FITS has no primary image: {source}")
                data = np.asanyarray(hdul[0].data)
                header = hdul[0].header.copy()
            if data.ndim != 2:
                raise ValueError(f"Level 0.5 image must be two-dimensional: {source}")

            metadata = load_level0_5_image_metadata(source, header)
            image_time = image_j2000_from_metadata(metadata)
            if csie_metadata_indicates_composite(metadata):
                try:
                    create_csie_composite_inner_mask(data.shape, metadata)
                except ValueError as geometry_error:
                    if database is None or image_time is None:
                        raise ValueError(
                            f"{source.name}: {geometry_error}; no canonical telemetry "
                            "database/time is available for APID 537 lookup"
                        ) from geometry_error
                    geometry = load_csie_composite_geometry(database, image_time)
                    metadata.update(geometry)
                    for key, value in geometry.items():
                        header[key] = value

            normalized = normalize_level1_image_to_dn_per_second(data, metadata)
            self.metadata = metadata
            try:
                if database is None:
                    raise FileNotFoundError("canonical telemetry DuckDB not found")
                rotated = self.__coarse_rotate(normalized, database)
                decision = self.coarse_orientation_decision
            except (FileNotFoundError, LookupError) as exc:
                decision = CoarseOrientationDecision(
                    status="attitude_unavailable",
                    quarter_turns_ccw=None,
                    valid_sample_count=0,
                    invalid_sample_count=0,
                    reason=str(exc),
                )
                self.coarse_orientation_decision = decision
                rotated = normalized.copy()

            header["LEVEL"] = (1.0, "SunCET processing level")
            header["BUNIT"] = ("DN / s", "Primary-array units")
            header["PIPEVERS"] = (str(self.config.version_pipeline), "Pipeline version")
            header["L1STATUS"] = ("PROVISIONAL", "First-light calibration remains pending")
            header["EXPNORM"] = (True, "DN normalized by effective exposure")
            if csie_metadata_indicates_composite(metadata):
                header["EXPMASK"] = ("ICM_CIRCLE", "Inner/outer exposure mask")
            for key, value in decision.fits_keywords().items():
                header[key] = value
            header.add_history("Level 1 converted stored DN to DN/s using CSIE metadata.")
            if decision.status == "stable":
                header.add_history("Applied lossless quarter-turn solar-north orientation.")
            else:
                header.add_history(
                    f"No detector rotation applied: {decision.status}: {decision.reason}"
                )
            target = destination / source.name
            fits.PrimaryHDU(
                np.asarray(rotated, dtype=np.float32), header=header
            ).writeto(target, overwrite=overwrite, checksum=True)
            outputs.append(target)
        return outputs

    def __load_level0_5(self, filenames):
        import sunpy.map

        # map_list = []
        # for file in filenames:
        #     map_list.append(sunpy.map.Map(file))

        # return sunpy.map.MapSequence(map_list)
        level05_map = sunpy.map.Map(filenames)
        return level05_map

    def __make_metadata_filename(self, filename, version):
        filename_with_extension = os.path.basename(filename)
        base, extension = os.path.splitext(filename_with_extension)
        return f"{base}{'_metadata_v'}{version}{'.csv'}"

    def run(self, level0_5_to_process, **kwargs):
        return self.make(level0_5_to_process, **kwargs)

    def apply_basic_corrections(self, filename, dark_filename, flat_filename, badpixel_filename):
        # Expects a map of level 0.5
        # Walks through a dark correction, flat field correction, bad pixel removal, and cosmic ray removal
        # Returns a sunpy map

        level05_map = self.__load_level0_5(filename)

        dark_corrected = self.__dark_correction(level05_map, os.path.join(self.config.calibration_path,
                                                                          self.config.dark_filename))
        flat_corrected = self.__flat_correction(dark_corrected, os.path.join(self.config.calibration_path,
                                                                             self.config.flat_filename))
        badpixel_corrected = self.__badpixel_removal(flat_corrected, os.path.join(self.config.calibration_path,
                                                                                  self.config.badpix_filename))
        cosmic_ray_corrected = self.__cosmic_ray_removal(badpixel_corrected)

        basic_map = cosmic_ray_corrected

        return basic_map

    def __dark_correction(self, input_map, dark_image_path):
        # Read the target image and dark frame and extract data
        with fits.open(dark_image_path) as dark_hdul:

            dark_data = dark_hdul.data[0].data

            # Subtract dark frame from the target image to correct for dark bias
            corrected_data = input_map.data - dark_data

            # Update the data in the target map with corrected data
            dark_corrected_map = input_map
            dark_corrected_map.data[:] = corrected_data[:]

        return dark_corrected_map

    def __flat_correction(self, input_map, flat_image_path):
        # Read the target image and flat frame and extract data
        with fits.open(flat_image_path) as flat_hdul:

            flat_data = flat_hdul.data[0].data

            # Divide flat frame from the target image to correct for flat bias
            corrected_data = input_map.data / flat_data

            # Update the data in the target map with corrected data
            flat_corrected_map = input_map
            flat_corrected_map.data[:] = corrected_data[:]

        return flat_corrected_map

    def __badpixel_removal(self, input_map, badpixel_path):
        # Read the target image and bad pixel frame and extract data
        with fits.open(badpixel_path) as bad_pixel_hdul:
            bad_pixel_mask = bad_pixel_hdul[0].data

            # Create a new array with bad pixels replaced with NaN
            corrected_data = np.where(bad_pixel_mask == 1, np.nan, input_map.data)

            # Update the data in the target map with corrected data
            badpixel_corrected_map = input_map
            badpixel_corrected_map[:] = corrected_data[:]

        return badpixel_corrected_map

    def __cosmic_ray_removal(self, input_map):
        """
        Interpolates over bad pixels in the input image using linear interpolation if the config file has
        cosmic_ray_removal set to True.

        Parameters:
            input_map (sunpy.map): Input 2D map (image) containing pixel values.

        Returns:
            sunpy.map: Interpolated sunpy map with bad pixels replaced by interpolated values.
        """
        if self.config.cosmic_ray_removal:
            from suncet_processing_pipeline import suncet_utility as utilities

            outlier_mask = utilities.detect_outliers(input_map.data, 20)
            row_indices, col_indices = np.where(outlier_mask)
            removed_values = input_map.data[row_indices, col_indices]

            # Create coordinate grid for good pixels
            y, x = np.indices(input_map.data.shape)
            good_pixel_coords = np.column_stack((y[~outlier_mask], x[~outlier_mask]))

            # Values of good pixels
            good_pixel_values = input_map.data[~outlier_mask]

            # Coordinates of bad pixels
            bad_pixel_coords = np.column_stack((y[outlier_mask], x[outlier_mask]))

            # Build the kdTree
            tree = KDTree(good_pixel_coords)

            # Query the nearest neighbors for the bad pixel coordinates
            distances, indices = tree.query(bad_pixel_coords, k=4)  # k=4 nearest neighbors

            # Retrieve the coordinates and values of the 4 nearest neighbors
            neighbor_coords = good_pixel_coords[indices]
            neighbor_values = good_pixel_values[indices]

            # Compute bilinear weights based on the inverse of the distances
            weights = 1 / (distances + 1e-12)  # Avoid division by zero
            weights /= np.sum(weights, axis=1, keepdims=True)

            # Compute the linear interpolated values as a weighted sums
            interpolated_values = np.sum(weights * neighbor_values, axis=1)

            # A slow cubic interpolation at the `bad_pixel_coords` points
            # interpolated_values = interpolate.griddata(good_pixel_coords, good_pixel_values, bad_pixel_coords, method='cubic')

            # Replace bad pixel values with interpolated values
            cosmic_ray_corrected_data = copy.deepcopy(input_map.data)
            cosmic_ray_corrected_data[bad_pixel_coords[:, 0], bad_pixel_coords[:, 1]] = interpolated_values

            corrected_values = cosmic_ray_corrected_data[row_indices, col_indices]
            index_1d_ravel = np.ravel_multi_index((row_indices, col_indices), input_map.data.shape)

            # Set cosmic ray output file (1d location, removed spike value, interpolated spike value)
            self.cosmic_rays = np.column_stack((index_1d_ravel, removed_values, corrected_values))
        else:
            cosmic_ray_corrected_data = input_map.data.copy()

        import sunpy.map

        return sunpy.map.Map(cosmic_ray_corrected_data, input_map.meta.copy())

    def __coarse_rotate(self, data, telemetry):
        """
        A rotation by a factor of 90 degrees to ensure solar north is in the top quadrant.
        Updates the rotation metadata

        Parameters:
            data (numpy.ndarray): Input 2D array (image) containing pixel values.
            telemetry (dict): a dictionary of single point values from SunCET

       Returns:
           numpy.ndarray: 2D array (image) containing pixel values rotated to keep solar north in the top quadrant
        """

        exposure_start = _metadata_value(
            self.metadata,
            "J2K-BEG",
            "capture_start_j2000",
        )
        exposure_end = _metadata_value(
            self.metadata,
            "J2K-END",
            "capture_end_j2000",
        )
        if exposure_start is None or exposure_end is None:
            decision = CoarseOrientationDecision(
                status="missing_exposure_time",
                quarter_turns_ccw=None,
                valid_sample_count=0,
                invalid_sample_count=0,
                reason="Level 0.5 metadata does not define the full exposure interval",
            )
        else:
            if isinstance(telemetry, (str, os.PathLike)):
                samples = load_apid46_attitude_samples(
                    telemetry,
                    float(exposure_start),
                    float(exposure_end),
                )
            else:
                samples = (
                    telemetry
                    if isinstance(telemetry, (list, tuple))
                    else [telemetry]
                )
            decision = plan_coarse_solar_north_rotation(
                samples,
                exposure_start_j2000=float(exposure_start),
                exposure_end_j2000=float(exposure_end),
            )
        self.coarse_orientation_decision = decision

        return apply_coarse_solar_north_rotation(data, decision)

    def __create_exposure_time_mask(self, image_data): #TODO check composite image dimensions expected
        """
        Create an exposure time mask matching the shape of image_data.

        Parameters:
        -----------
        image_data : np.ndarray
            Composite image data, shape can be (n_sections, height, width) or (height, width)

        Returns:
        --------
        np.ndarray
            Exposure time mask with the same shape as image_data.
        """
        inner_mask = getattr(self, "inner_fov_mask", None)
        return create_level1_exposure_time_mask(
            image_data,
            self.metadata,
            inner_mask=inner_mask,
        )

    def __convert_to_DN(self, image_data):
        """
        Convert image data to DN/s by dividing each pixel by its exposure time.

        Parameters:
        -----------
        image_data : np.ndarray
            Composite image data, expected to be 2D or 3D

        Returns:
        --------
        np.ndarray
            Image data converted to DN/s.
        """
        inner_mask = getattr(self, "inner_fov_mask", None)
        return normalize_level1_image_to_dn_per_second(
            image_data,
            self.metadata,
            inner_mask=inner_mask,
        )

def __compute_1AU_scaling_factor(self): #TODO decide where this belongs in the meta data
    """
    Compute a scaling factor to adjust image intensity to 1 AU distance.

    Returns:
    --------
    float
        Multiplicative scaling factor to adjust observed intensity to what it would be at 1 AU.
    """
    AU_METERS = 1.496e11  # 1 AU in meters
    dsun_obs = self.metadata.DSUN_OBS

    if dsun_obs is None or dsun_obs <= 0:
        raise ValueError("Invalid DSUN_OBS value: must be a positive number in meters.")

    # Inverse-square law: intensity ∝ 1 / distance^2 → scale by (AU / dsun_obs)^2
    scaling_factor = (AU_METERS / dsun_obs) ** 2
    return scaling_factor


def parameterize_noise(self, image_data, n_regions=9, region_size=64): #TODO test for desired outputs
    """
    Estimate image noise sigma in representative regions and store results in metadata.

    Parameters:
    -----------
    image_data : np.ndarray
        2D or 3D array representing image(s).
    n_regions : int
        Total number of regions to sample (must be a perfect square for grid).
    region_size : int
        Size of square region in pixels for each sample.

    Stores:
    -------
    self.metadata.noise_map : dict
        Dictionary with region locations and corresponding noise sigma estimates.
    """
    from sunkit_image.utils import noise_estimation

    if image_data.ndim == 3:
        # Average over sections to get a representative 2D image
        image = np.mean(image_data, axis=0)
    elif image_data.ndim == 2:
        image = image_data
    else:
        raise ValueError("Unsupported image_data shape: must be 2D or 3D.")

    height, width = image.shape
    noise_map = {}
    grid_dim = int(np.sqrt(n_regions))
    if grid_dim ** 2 != n_regions:
        raise ValueError("n_regions must be a perfect square (e.g., 9, 16, 25).")

    step_y = height // grid_dim
    step_x = width // grid_dim

    for i in range(grid_dim):
        for j in range(grid_dim):
            y0 = i * step_y + (step_y - region_size) // 2
            x0 = j * step_x + (step_x - region_size) // 2

            # Ensure boundaries are within the image
            y0 = max(0, min(y0, height - region_size))
            x0 = max(0, min(x0, width - region_size))

            region = image[y0:y0 + region_size, x0:x0 + region_size]
            sigma = noise_estimation(region)

            noise_map[(i, j)] = {
                "center": (y0 + region_size // 2, x0 + region_size // 2),
                "sigma": float(sigma)
            }

    self.metadata.noise_map = noise_map



def main(argv=None):
    parser = argparse.ArgumentParser(description="Create SunCET Level 1 FITS images")
    parser.add_argument("input", help="Level 0.5 FITS file or directory")
    parser.add_argument("--output-dir")
    parser.add_argument("--telemetry-database")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config_files" / "config_default.ini"),
    )
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args(argv)
    level1 = Level1(config_parser.Config(args.config))
    outputs = level1.run(
        args.input,
        output_dir=args.output_dir,
        telemetry_database=args.telemetry_database,
        overwrite=not args.no_overwrite,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
